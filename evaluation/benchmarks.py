"""Standalone functions for benchmarking editor performance across metrics."""
import logging
import random
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from itertools import chain
from typing import Any, Callable, Sequence, cast

from evaluation import data, metrics, models, precompute
from evaluation.evaluator import * 
from evaluation.utils import tokenizer_utils
from evaluation.utils.typing import Dataset, Device, StrSequence
from pastalib import pasta

import numpy as np
import scipy.stats
import torch
import torch.utils.data
from dataclasses_json import DataClassJsonMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_PREFIX = "The following is an excerpt from a Wikipedia article:\n\n"
DEFAULT_PROMPT_TEMPLATE = "{} is"
DEFAULT_MAX_LENGTH = 100
DEFAULT_MAX_LENGTH_ERROR_CORRECTION = 150
DEFAULT_TOP_K = 3
DEFAULT_N_TOP_TOKENS = DEFAULT_TOP_K
DEFAULT_BATCH_SIZE = 16


@dataclass(frozen=True)
class EfficacySample(DataClassJsonMixin):
    """Wrapper around a single efficacy sample."""

    id: str
    prompt: str
    target_score: float
    comparator_score: float


@dataclass(frozen=True)
class EfficacyBenchmarkResults(DataClassJsonMixin):
    """Wrapper around efficacy benchmark results."""

    samples: list[EfficacySample]
    metrics: metrics.EfficacyMetrics


@dataclass(frozen=True)
class CounterFactEvaluationResult(DataClassJsonMixin):
    """Result for single sample from `Editor.evaluate`."""

    sample: dict

    top_tokens: list[str] | None = None
    top_logps: list[float] | None = None
    generations: list[str] | None = None

    target_mediated_score: float | None = None
    target_unmediated_score: float | None = None


@dataclass(frozen=True)
class CounterFactEvaluateRun(DataClassJsonMixin):
    """Wrapper around a list of individual evaluation results."""

    results: list[CounterFactEvaluationResult]


@torch.inference_mode()
def counterfact_evaluate(
    mt, 
    dataset: Dataset,
    pasta_steerer: pasta.PASTA|None = None, 
    batch_size: int = 16,
    n_top: int = 3,
    max_length: int | None = None,
    max_new_tokens: int | None = None,
    desc: str|None = None,
    device: Device|None = None,
    return_mediated: bool = True,
    return_unmediated: bool = True,
    add_unmediated_fact: bool = True,
    add_few_shot: int | None = None, 
    few_shot_index: str | None = None,  
    add_marker: str | None = None, 
) -> CounterFactEvaluateRun:
    """
    Evaluation on CounterFact 

    Args:
        mt: The model to be evaluated. 
        dataset: Dataset to evaluate on.
        pasta_steerer: The PASTA steerer for attention steering. 
        batch_size: Model batch size.
        n_top: Number of top words/probs to return.
        max_length: Number of tokens to generate including prompt.
        max_new_tokens: Number of tokens to generate not including prompt.
        desc: The tqdm description.
        device: Send all data to this device. Defaults to None.
        return_mediated: Return mediated token probability.
        return_unmediated: Return unmediated token probability.
        add_unmediated_fact: Whether to present the model both new and old facts.  

    Returns:
        The evaluation results, one per entry in the dataset.

    """
    mt.eval_()
    mt.to_(device)
    include_target_probs = "target_mediated" in dataset.column_names
    if desc is None:
        desc = f"Evaluate CounterFact"
    if max_length is None and max_new_tokens is None:
        max_length = DEFAULT_MAX_LENGTH

    exclude_columns = []
    if not return_mediated:
        exclude_columns.append("target_mediated")
    if not return_unmediated:
        exclude_columns.append("target_unmediated")
    columns = data.column_names(dataset, exclude=exclude_columns)

    if add_few_shot is not None and add_few_shot > 0: 
        if few_shot_index is not None:
            few_shot_index = [int(idx) for idx in few_shot_index.split(",")]
            assert len(few_shot_index) == add_few_shot
        else:
            few_shot_index = np.random.randint(len(dataset), size=add_few_shot).tolist()
        fewshot_examples = precompute.prepare_counterfact_few_shot_examples(
            dataset.select(few_shot_index), add_unmediated_fact=add_unmediated_fact, 
        )
    else:
        fewshot_examples = None 

    results = []
    with dataset.formatted_as("torch", columns=columns):
        loader = torch.utils.data.DataLoader(
            cast(torch.utils.data.Dataset, dataset), batch_size=batch_size
        )
        for batch in tqdm(loader, desc=desc):
            if not precompute.has_editor_inputs(batch):
                batch.update(
                    precompute.editor_inputs_from_batch(
                        mt=mt, batch=batch, device=device, 
                        return_target_token_ids=return_mediated or return_unmediated,
                        target_token_first_space=True if "gpt" in mt.model.name_or_path else False,
                    )
                )

            prompts = batch["prompt"]
            prompts = batch["prompt"]
            contexts = batch["context"]
            attributes = batch["attribute"]
            targets_mediated = batch["target_mediated"]
            targets_unmediated = batch["target_unmediated"]
            current_batch_size = len(prompts)

            if add_unmediated_fact:
                # Modify the prompts and provide the model both new and old facts 
                new_prompts = []
                for prompt, context, target_mediated, target_unmediated in zip(
                    prompts, contexts, targets_mediated, targets_unmediated
                ):
                    unmediated_prefix = "Previously "
                    mediated_prefix = "Currently "
                    unmediated_fact = context.replace(target_mediated, target_unmediated)+". "
                    new_prompt = f"{unmediated_prefix}{unmediated_fact}{mediated_prefix}{prompt}"

                    new_prompts.append(new_prompt)
                prompts = new_prompts

            if add_marker is not None:
                prompts = [
                    prompt.replace(attr, add_marker+attr+add_marker) for prompt,attr in zip(prompts, attributes)
                ]
            if fewshot_examples is not None:
                prompts = [
                    fewshot_examples + prompt for prompt in prompts
                ]
            
            with models.set_padding_side(mt, padding_side="left"):
                inputs, prompt_offset_mapping = precompute.inputs_from_batch(mt, prompts, device=device)
            generate_kwargs = dict(
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                pad_token_id=mt.tokenizer.eos_token_id,
            )
            # Here, we emphasize the new facts (attributes) with PASTA
            if pasta_steerer is not None:
                with pasta_steerer.apply_steering(
                    model=mt.model, 
                    strings=prompts, 
                    substrings=attributes, 
                    model_input=inputs, 
                    offsets_mapping=prompt_offset_mapping
                ) as steered_model: 
                    outputs = steered_model.generate(**inputs, **generate_kwargs)
            else:
                outputs = mt.model.generate(**inputs, **generate_kwargs)
            
            batched_results: dict = {}
            first_token_logps = torch.log_softmax(outputs.scores[0], dim=-1)
            top_logps, top_token_ids = first_token_logps.topk(k=n_top, dim=-1)
            top_tokens = tokenizer_utils.batch_convert_ids_to_tokens(top_token_ids, mt.tokenizer)
            generations = mt.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

            batched_results[f"top_logps"] = top_logps.tolist()
            batched_results[f"top_tokens"] = top_tokens
            batched_results[f"generations"] = [[g] for g in generations]
            if include_target_probs:
                target_keys = []
                if return_mediated:
                    target_keys.append("mediated")
                if return_unmediated:
                    target_keys.append("unmediated")
                batch_indices = torch.arange(current_batch_size)
                for target_key in target_keys:
                    target_id = batch[f"target_{target_key}.token_id"]
                    target_probs = first_token_logps[batch_indices, target_id]
                    target_prob_key = f"target_{target_key}_score"
                    batched_results[target_prob_key] = target_probs.tolist()

            for bi in range(current_batch_size):
                result: dict = {k: vs[bi] for k, vs in batched_results.items()}
                results.append(result)

    # Finally, decorate results with original sample data.
    assert len(results) == len(dataset)
    for sample, result in zip(dataset, results):
        result.update(
            sample={
                key: sample[key]
                for key in data.ContextMediationSample.__required_keys__
            }
        )

    return CounterFactEvaluateRun([CounterFactEvaluationResult(**r) for r in results])



@torch.inference_mode()
def counterfact_efficacy(
    *,
    mt: models.ModelAndTokenizer,
    dataset: Dataset,
    pasta_steerer: pasta.PASTA|None = None, 
    desc: str | None = None,
    **kwargs: Any,
) -> EfficacyBenchmarkResults:
    """Run the CounterFact efficacy benchmark.

    Args:
        mt: The model to evaluate. 
        dataset: The dataset to evaluate on.
        pasta_steerer: The PASTA steerer to steer the model generaion.

    Returns:
        Evaluation results 
    
    """
    if desc is None:
        desc = "efficacy benchmark"

    evaluate_kwargs = dict(max_new_tokens=1, desc=desc, **kwargs)
    run = counterfact_evaluate(mt, dataset, pasta_steerer=pasta_steerer, **evaluate_kwargs)
    target_score_key = "target_mediated_score"
    comparator_score_key = "target_unmediated_score"

    samples = []
    for result in run.results:
        sid = result.sample["id"]
        prompt = result.sample["prompt"]

        target_score = getattr(result, target_score_key)
        assert target_score is not None

        comparator_score = getattr(result, comparator_score_key)
        assert comparator_score is not None

        logger.debug(f"ID={sid} SCORE_T={target_score} SCORE_COMP={comparator_score}")
        sample = EfficacySample(
            id=sid,
            prompt=prompt,
            target_score=target_score,
            comparator_score=comparator_score,
        )
        samples.append(sample)

    efficacy_metrics = metrics.efficacy(
        [[sample.target_score] for sample in samples],
        [[sample.comparator_score] for sample in samples],
        store_values=False,
    )
    return EfficacyBenchmarkResults(samples=samples, metrics=efficacy_metrics)


@dataclass(frozen=True)
class ParaphraseSample(DataClassJsonMixin):
    """Wrapper around a single paraphrase benchmark sample."""

    id: str
    prompts: list[EfficacySample]
    efficacy_score: float
    efficacy_magnitude: float


@dataclass(frozen=True)
class CounterFactParaphraseBenchmarkResults(DataClassJsonMixin):
    """Wrapper around paraphrase benchmark results."""

    samples: list[ParaphraseSample]
    metrics: metrics.EfficacyMetrics


@torch.inference_mode()
def counterfact_paraphrase(
    *,
    mt: models.ModelAndTokenizer | None = None,
    pasta_steerer: pasta.PASTA|None = None,
    dataset: Dataset,
    desc: str | None = None,
    **kwargs: Any,
) -> CounterFactParaphraseBenchmarkResults:
    """Run the CounterFact paraphrase benchmark.

    Since this benchmark relies on extra data, it can only be used with the CounterFact
    dataset. The `counterfact_generation` benchmark is like this as well.

    This function expects that each sample in the dataset supports an access like:

        prompts = sample["source"]["generation_prompts"]

    """
    if desc is None:
        desc = "paraphrase benchmark"
    dataset = _counterfact_select_and_flatten(
        dataset, "paraphrase_prompts", desc=f"{desc} [flatten dataset]"
    )
    efficacy_benchmark = counterfact_efficacy(
        mt=mt,
        pasta_steerer=pasta_steerer,
        dataset=dataset,
        desc=desc,
        **kwargs,
    )

    results_by_sample_id: dict = defaultdict(list)
    for result in efficacy_benchmark.samples:
        results_by_sample_id[result.id].append(result)
    results_by_sample_id = OrderedDict(results_by_sample_id)

    efficacy_metrics = metrics.efficacy(
        [
            [result.target_score for result in results]
            for results in results_by_sample_id.values()
        ],
        [
            [result.comparator_score for result in results]
            for results in results_by_sample_id.values()
        ],
    )

    # Reformat EfficacySample -> ParaphraseSample
    samples = []
    for (sid, results), efficacy_score, efficacy_magnitude in zip(
        results_by_sample_id.items(),
        cast(list, efficacy_metrics.score.values),
        cast(list, efficacy_metrics.magnitude.values),
    ):
        sample = ParaphraseSample(
            id=sid,
            prompts=results,
            efficacy_score=efficacy_score,
            efficacy_magnitude=efficacy_magnitude,
        )
        samples.append(sample)

    return CounterFactParaphraseBenchmarkResults(
        samples=samples,
        metrics=efficacy_metrics.without_values(),
    )


@dataclass(frozen=True)
class GenerationSample(DataClassJsonMixin):
    """Wrapper around a single sample from the generation benchmark."""

    id: str
    generations: list[str]
    references: list[str]
    fluency_score: float
    consistency_score: float


@dataclass(frozen=True)
class GenerationMetrics(DataClassJsonMixin):
    """Wrapper around all generation metrics."""

    fluency: metrics.Metric
    consistency: metrics.Metric


@dataclass(frozen=True)
class CounterFactGenerationBenchmarkResults(DataClassJsonMixin):
    """Wrapper around generation benchmark results."""

    samples: list[GenerationSample]
    metrics: GenerationMetrics


@torch.inference_mode()
def counterfact_generation(
    *,
    mt: models.ModelAndTokenizer,
    dataset: Dataset,
    pasta_steerer: pasta.PASTA|None = None,
    attribute_snippets: data.AttributeSnippets | None = None,
    tfidf_vectorizer: TfidfVectorizer | None = None,
    max_length: int | None = None,
    max_new_tokens: int | None = None,
    desc: str | None = None,
    **kwargs: Any,
) -> CounterFactGenerationBenchmarkResults:
    """Run the CounterFact generation benchmark.

    Free-form generates on several "generation prompts" per sample, and records
    the fluency of the generations (measured by weighted n-gram entropy) and
    consistency with other texts about entities with the same attribute.

    This benchmark *requires* the dataset to be CounterFact or something that looks
    like it, since it uses extra data that is specific to CounterFact.

    Specifically, it expects each sample can be accessed like:

        prompts = sample["source"]["generation_prompts"]

    """
    if attribute_snippets is None:
        attribute_snippets = data.load_attribute_snippets()
    if tfidf_vectorizer is None:
        tfidf_vectorizer = data.load_counterfact_tfidf_vectorizer()
    if max_new_tokens is None and max_length is None:
        max_length = DEFAULT_MAX_LENGTH
    if desc is None:
        desc = "generate benchmark"

    dataset = _counterfact_select_and_flatten(
        dataset, "generation_prompts", desc=f"{desc} [flatten dataset]"
    )

    evaluate_kwargs = dict(
        max_new_tokens=max_new_tokens,
        max_length=max_length,
        desc=f"{desc} [run model]",
        **kwargs,
    )
    run = counterfact_evaluate(mt, dataset, pasta_steerer=pasta_steerer, **evaluate_kwargs)
    generations_key = "generations"
    run_results_by_id = _group_results_by_id(run)

    samples = []
    for sid, results in tqdm(run_results_by_id.items(), desc=f"{desc} [tfidf]"):
        result = next(iter(results))
        cf_requested_rewrite = result.sample["source"]["requested_rewrite"]
        relation_id = cf_requested_rewrite["relation_id"]
        target_id = cf_requested_rewrite["target_new"]["id"]

        generations = [getattr(result, generations_key)[0] for result in results]
        references = [
            snippet["text"] for snippet in attribute_snippets[relation_id][target_id]
        ]

        consistency_score = metrics.tfidf_similarity(
            generations, references, tfidf_vectorizer
        )
        fluency_score = metrics.weighted_n_gram_entropy(generations)

        entity = result.sample["entity"]
        attribute = result.sample["attribute"]
        logger.debug(f"ID={sid} ENTITY={entity}, ATTR={attribute}")
        logger.debug(f"ID={sid} REFERENCES={references}")
        logger.debug(f"ID={sid} GENERATIONS={generations}")

        sample = GenerationSample(
            id=sid,
            generations=generations,
            references=references,
            fluency_score=fluency_score,
            consistency_score=consistency_score,
        )
        samples.append(sample)

    fluency = metrics.Metric.aggregate(
        [sample.fluency_score for sample in samples], store_values=False
    )
    consistency = metrics.Metric.aggregate(
        [sample.consistency_score for sample in samples], store_values=False
    )
    generation_metrics = GenerationMetrics(fluency=fluency, consistency=consistency)
    return CounterFactGenerationBenchmarkResults(
        samples=samples, metrics=generation_metrics
    )


def _counterfact_select_and_flatten(
    dataset: Dataset, column: str, desc: str | None = None
) -> Dataset:
    """Select the given column in counterfact, dedupe it, and flatten it."""
    column_names = data.column_names(dataset)

    def select_and_flatten_counterfact_row(row: dict) -> dict:
        prompts = list(set(row["source"][0][column]))
        result = {"prompt": prompts}
        for key in data.ContextMediationSample.__required_keys__:
            if key not in result:
                result[key] = [row[key][0]] * len(prompts)
        return result

    return dataset.map(
        select_and_flatten_counterfact_row,
        batched=True,
        batch_size=1,
        remove_columns=column_names,
        desc=desc,
    )


def _group_results_by_id(results: CounterFactEvaluateRun) -> OrderedDict:
    """Group results by sample ID."""
    grouped = defaultdict(list)
    for result in results.results:
        grouped[result.sample["id"]].append(result)
    return OrderedDict(grouped)


@dataclass(frozen=True)
class BiasBiosEvaluationSample:
    """Wrapper around error correction sample."""

    id: str
    prompt: str
    generation: str

    predictions: list[str]
    target: str

    logp_predictions: list[float]
    logp_target: float

    fluency: float
    consistency: float


@dataclass(frozen=True)
class BiasBiosEvaluationMetrics(DataClassJsonMixin):
    """Wrapper around aggregated error correction metrics."""

    top1_accuracy: float
    topk_accuracy: float
    k: int

    fluency: metrics.Metric
    consistency: metrics.Metric


@dataclass(frozen=True)
class BiasBiosEvaluationResults(DataClassJsonMixin):
    """Wrapper around error correction benchmark."""

    samples: list[BiasBiosEvaluationSample]
    metrics: BiasBiosEvaluationMetrics


@torch.inference_mode()
def biasbios_prediction_evaluation(
    *,
    mt: models.ModelAndTokenizer,
    dataset: Dataset,
    pasta_steerer: pasta.PASTA|None = None, 
    tfidf_vectorizer: TfidfVectorizer | None = None,
    references: dict | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    top_k: int = DEFAULT_TOP_K,
    max_length: int | None = None,
    max_new_tokens: int | None = None,
    device: Device | None = None,
    desc: str | None = None,
    add_few_shot: int | None = None, 
    few_shot_index: str | None = None,  
    add_marker: str | None = None, 
    **kwargs,
) -> BiasBiosEvaluationResults:
    """Run BiasBios prediction benchmark.

    This benchmark involves measuring accuracy on the BiasBios prediction task. 

    Args:
        mt: The model to evaluate. 
        dataset: The dataset to evaluate on.
        pasta_steerer: The PASTA steerer to steer the model generaion. 
        tfidf_vectorizer: For computing consistency score.
        references: Mapping from label to reference texts for that label. By default,
            full bios for each label will be used.
        batch_size: Batch size for model.
        top_k: Compute top-k labels predicted by model.
        prompt_key: Which column in dataset to use as prompt.
        max_length: Max sequence length (input+output).
        max_new_tokens: Max number of new tokens to generate. Cannot be used with
            `max_length`, see huggingface docs.
        device: Send model and data to this device.
        desc: TQDM description. 
        add_few_shot: Whether to apply few-shot prompting. 
        few_shot_index: Sample index for few shots. 
        add_marker: Whether to apply marked prompting. 

    Returns:
        Benchmark results.

    """
    if max_length is None and max_new_tokens is None:
        max_length = DEFAULT_MAX_LENGTH_ERROR_CORRECTION
    if desc is None:
        desc = "BiasBios Evaluation"

    if tfidf_vectorizer is None:
        tfidf_vectorizer = data.load_biosbias_tfidf_vectorizer()
    if references is None:
        references = defaultdict(list)
        for sample in dataset:
            references[sample["target_mediated"]].append(sample["source"]["bio"])

    labels = sorted({x["target_mediated"] for x in dataset})
    label_add_space = True if "gpt" in mt.model.name_or_path else False
    labels_token_idx = precompute.first_token_ids_from_batch(mt, labels, add_space=label_add_space)

    reference_tfidfs = {
        key: tfidf_vectorizer.transform(texts).mean(axis=0).A
        for key, texts in tqdm(references.items(), desc=f"{desc} [reference tfidfs]")
    }

    if add_few_shot is not None and add_few_shot > 0: 
        if few_shot_index is not None:
            few_shot_index = [int(idx) for idx in few_shot_index.split(",")]
            assert len(few_shot_index) == add_few_shot
        else:
            few_shot_index = np.random.randint(len(dataset), size=add_few_shot).tolist()
        fewshot_examples = prepare_biasbios_few_shot_examples(
            dataset.select(few_shot_index), text_key="context"
        )

    columns = data.column_names(dataset, exclude=["target_unmediated"])
    with dataset.formatted_as("torch", columns=columns):
        loader = torch.utils.data.DataLoader(
            cast(torch.utils.data.Dataset, dataset),
            batch_size=batch_size,
        )

        samples = []
        for batch in tqdm(loader, desc=desc):
            ids = batch["id"]
            prompts = batch["prompt"]
            targets = batch["target_mediated"]
            targets_idx = precompute.first_token_ids_from_batch(mt, targets, add_space=label_add_space)

            if add_few_shot is not None and add_few_shot > 0:
                batch['prompt'] = [fewshot_examples+prompt for prompt in batch['prompt']]
                prompts = [fewshot_examples+prompt for prompt in prompts]

            if add_marker is not None:
                batch['prompt'] = [
                    prompt.replace(attr, add_marker+attr+add_marker) for prompt,attr in zip(batch['prompt'], batch['attribute'])
                ]
                prompts = batch['prompt']

            with models.set_padding_side(mt, padding_side="left"):
                inputs, prompt_offset_mapping = precompute.inputs_from_batch(mt, prompts, device=device)

            generate_kwargs = dict(
                return_dict_in_generate=True,
                output_scores=True,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                pad_token_id=mt.tokenizer.eos_token_id,
            )
            if pasta_steerer is not None:
                # Apply PASTA steering
                # Here we emphasize the first sentence `attribute`
                with pasta_steerer.apply_steering(
                    model=mt.model, 
                    strings=prompts, 
                    substrings=batch['attribute'], 
                    model_input=inputs, 
                    offsets_mapping=prompt_offset_mapping
                ) as steered_model: 
                    outputs = steered_model.generate(**inputs, **generate_kwargs)
            else:
                outputs = mt.model.generate(**inputs, **generate_kwargs)

            generations = mt.tokenizer.batch_decode(
                outputs.sequences[:, inputs.input_ids.shape[1] :],
                skip_special_tokens=True,
            )
            distributions = torch.log_softmax(outputs.scores[0], dim=-1)

            for sid, prompt, distribution, generation, target, target_idx in zip(
                ids, prompts, distributions, generations, targets, targets_idx
            ):
                label_log_probs = distribution[labels_token_idx]

                logp_predictions, predictions_idx = label_log_probs.topk(
                    k=top_k, dim=-1
                )
                predictions = [labels[idx] for idx in predictions_idx]

                logp_target = distribution[target_idx]

                fluency_score = metrics.weighted_n_gram_entropy(generation)

                [generation_tfidf] = tfidf_vectorizer.transform([generation]).A
                reference_tfidf = reference_tfidfs[target]
                consistency_score = metrics.vector_similarity(
                    generation_tfidf.squeeze(), reference_tfidf.squeeze()
                )

                sample = BiasBiosEvaluationSample(
                    id=sid,
                    prompt=prompt,
                    generation=generation,
                    predictions=predictions,
                    logp_predictions=logp_predictions.tolist(),
                    target=target,
                    logp_target=logp_target.item(),
                    fluency=fluency_score,
                    consistency=consistency_score,
                )
                samples.append(sample)

    n_correct_top1 = sum(x.predictions[0] == x.target for x in samples)
    n_correct_topk = sum(x.target in x.predictions for x in samples)
    top1_accuracy = n_correct_top1 / len(samples)
    topk_accuracy = n_correct_topk / len(samples)

    fluency = metrics.Metric.aggregate([x.fluency for x in samples], store_values=False)
    consistency = metrics.Metric.aggregate(
        [x.consistency for x in samples], store_values=False
    )

    error_correction_metrics = BiasBiosEvaluationMetrics(
        top1_accuracy=top1_accuracy,
        topk_accuracy=topk_accuracy,
        k=top_k,
        fluency=fluency,
        consistency=consistency,
    )

    return BiasBiosEvaluationResults(
        samples=samples,
        metrics=error_correction_metrics,
    )


@torch.inference_mode()
def biasbios_instruction_evaluation(
    *,
    mt: models.ModelAndTokenizer,
    dataset: Dataset,
    task: str, 
    prompt_idx: int | list | None = None, 
    pasta_steerer: pasta.PASTA|None = None, 
    emphasized_text: list | None = None, 
    tfidf_vectorizer: TfidfVectorizer | None = None,
    references: dict | None = None,
    batch_size: int = 16,
    top_k: int = DEFAULT_TOP_K,
    max_length: int | None = None,
    max_new_tokens: int | None = None,
    device: Device | None = None,
    desc: str | None = None,
    add_few_shot: int | None = None, 
    few_shot_index: str | None = None,  
    **kwargs,
) -> BiosBiasInstructionEvaluationResults:
    """ Evaluate the instruction following tasks  

    This benchmark evaluation model ability of instruction following 

    Args:
        mt: The model to evaluate. 
        dataset: The dataset to evaluate on.
        task: The name of task (json|pronchange). 
        prompt_idx: The index of prompt template. 
        pasta_steerer: The PASTA steerer to steer the model generaion. 
        emphasized_text: The input spans to be highlighted. 
        tfidf_vectorizer: For computing consistency score.
        references: Mapping from label to reference texts for that label. By default,
            full bios for each label will be used.
        batch_size: Batch size for model.
        top_k: Compute top-k labels predicted by model.
        prompt_key: Which column in dataset to use as prompt.
        max_length: Max sequence length (input+output).
        max_new_tokens: Max number of new tokens to generate. Cannot be used with
            `max_length`, see huggingface docs.
        device: Send model and data to this device.
        desc: TQDM description. 
        add_few_shot: Whether to apply few-shot prompting. 
        few_shot_index: Sample index for few shots. 

    Returns:
        Benchmark results.
    """
    if max_length is None and max_new_tokens is None:
        max_length = DEFAULT_MAX_LENGTH_ERROR_CORRECTION
    if desc is None:
        desc = "Instruction Evaluation"

    if tfidf_vectorizer is None:
        tfidf_vectorizer = data.load_biosbias_tfidf_vectorizer()
    if references is None:
        references = defaultdict(list)
        for sample in dataset:
            references[sample["target_mediated"]].append(sample["source"]["bio"])

    # Load the InstructionEvaluator 
    evaluator = InstructionEvaluator(task, prompt_idx) 

    labels = sorted({x["target_mediated"] for x in dataset})
    label_add_space = True if "gpt" in mt.model.name_or_path else False
    labels_token_idx = precompute.first_token_ids_from_batch(mt, labels, add_space=label_add_space)

    reference_tfidfs = {
        key: tfidf_vectorizer.transform(texts).mean(axis=0).A
        for key, texts in tqdm(references.items(), desc=f"{desc} [reference tfidfs]")
    }

    # If applied, prepare the few-shot demonstration 
    if add_few_shot is not None and add_few_shot > 0: 
        if few_shot_index is not None:
            few_shot_index = [int(idx) for idx in few_shot_index.split(",")]
            assert len(few_shot_index) == add_few_shot
        else:
            few_shot_index = np.random.randint(len(dataset), size=add_few_shot).tolist()
        fewshot_examples = evaluator.prepare_fewshot_examples(dataset.select(few_shot_index))

    columns = data.column_names(dataset, exclude=["target_unmediated"])
    with dataset.formatted_as("torch", columns=columns):
        loader = torch.utils.data.DataLoader(
            cast(torch.utils.data.Dataset, dataset),
            batch_size=batch_size,
        )

        samples = []
        for batch in tqdm(loader, desc=desc):
            ids = batch["id"]
            contexts = batch["context"]
            attributes = batch["attribute"]
            entities = batch["entity"]
            targets = batch["target_mediated"]

            prompts, instructions, entities, (ids, attributes, targets) = evaluator.parapare_prompt_inputs(
                contexts, entities, (ids, attributes, targets)
            )
            targets_idx = precompute.first_token_ids_from_batch(mt, targets, add_space=label_add_space)
            if add_few_shot is not None and add_few_shot > 0:
                prompts = [fewshot_examples+prompt for prompt in prompts]

            with models.set_padding_side(mt, padding_side="left"):
                inputs, prompt_offset_mapping = precompute.inputs_from_batch(mt, prompts, device=device)

            generate_kwargs = dict(
                return_dict_in_generate=True,
                output_scores=True,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                pad_token_id=mt.tokenizer.eos_token_id,
            )
            if pasta_steerer is not None:
                # Emphasize the selected texts with PASTA 
                if emphasized_text == "instruct":
                    substrings = instructions
                elif emphasized_text == "attribute":
                    substrings = attributes
                else:
                    raise ValueError(f"Unimplemented substrings: {emphasized_text}")
                # Steer the attention scores of selected input spans 
                with pasta_steerer.apply_steering(
                    model=mt.model, 
                    strings=prompts, 
                    substrings=substrings, 
                    model_input=inputs, 
                    offsets_mapping=prompt_offset_mapping
                ) as steered_model: 
                    outputs = steered_model.generate(**inputs, **generate_kwargs)
            else:
                outputs = mt.model.generate(**inputs, **generate_kwargs)

            generations = mt.tokenizer.batch_decode(
                outputs.sequences[:, inputs.input_ids.shape[1] :],
                skip_special_tokens=True,
            )
            distributions = torch.log_softmax(outputs.scores[0], dim=-1)

            for idx, sid, prompt, distribution, generation, target, target_idx in zip(
               range(len(ids)), ids, prompts, distributions, generations, targets, targets_idx
            ):
                label_log_probs = distribution[labels_token_idx]

                logp_predictions, predictions_idx = label_log_probs.topk(
                    k=top_k, dim=-1
                )
                predictions = [labels[idx] for idx in predictions_idx]

                logp_target = distribution[target_idx]

                fluency_score = metrics.weighted_n_gram_entropy(generation)

                [generation_tfidf] = tfidf_vectorizer.transform([generation]).A
                reference_tfidf = reference_tfidfs[target]
                consistency_score = metrics.vector_similarity(
                    generation_tfidf.squeeze(), reference_tfidf.squeeze()
                )

                instruction_evaluation = evaluator.evaluate_sample(generation=generation, target=target)

                sample = InstructionEvaluationSample(
                    id=sid,
                    prompt=prompt,
                    generation=generation,
                    predictions=predictions,
                    logp_predictions=logp_predictions.tolist(),
                    target=target,
                    logp_target=logp_target.item(),
                    fluency=fluency_score,
                    consistency=consistency_score,
                    instruction_evaluation=instruction_evaluation, 
                    sample_attn_scores=None, 
                )
                samples.append(sample)

    n_correct_top1 = sum(x.predictions[0] == x.target for x in samples)
    n_correct_topk = sum(x.target in x.predictions for x in samples)
    top1_accuracy = n_correct_top1 / len(samples)
    topk_accuracy = n_correct_topk / len(samples)

    fluency = metrics.Metric.aggregate([x.fluency for x in samples], store_values=False)
    consistency = metrics.Metric.aggregate(
        [x.consistency for x in samples], store_values=False
    )

    instruction_evaluation = evaluator.aggregate_evaluation_results(samples)
    instruction_evaluation_metrics = InstructionEvaluationMetrics(
        top1_accuracy=top1_accuracy,
        topk_accuracy=topk_accuracy,
        k=top_k,
        fluency=fluency,
        consistency=consistency,
        instruction_evaluation=instruction_evaluation, 
    )

    return BiosBiasInstructionEvaluationResults(
        samples=samples,
        metrics=instruction_evaluation_metrics,
        attentions=None, 
    )
