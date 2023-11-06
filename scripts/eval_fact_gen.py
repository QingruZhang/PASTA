"""Evaluate editors on the Counterfact benchmark."""
import argparse
import json
import logging
import random
from functools import partial
from pathlib import Path
from typing import cast

from evaluation import benchmarks, data, models, precompute
from evaluation.utils import experiment_utils, logging_utils
from pastalib import pasta

import torch
import torch.utils.data
from tqdm.auto import tqdm

from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

BENCHMARKS = (
    "efficacy",
    "paraphrase",
    "generation",
)


def _prefix_context(sample: dict) -> dict:
    """Prepend context to all prompts used in the eval."""
    entity = sample["entity"]
    prompt = sample["prompt"]
    context = sample["context"]

    prompt_in_context = precompute.prompt_in_context_from_sample(
        entity, prompt, context
    )

    source = {**sample["source"]}
    for key in ("generation_prompts", "paraphrase_prompts"):
        source[key] = [
            precompute.prompt_in_context_from_sample(entity, other_prompt, context)
            for other_prompt in source[key]
        ]
    return {"source": source, "prompt": prompt_in_context}


def main(args: argparse.Namespace) -> None:
    """Run the CounterFact benchmark."""
    experiment = experiment_utils.setup_experiment(args)
    logging_utils.configure(args=args)
    data.disable_caching()
    if args.debug: 
        experiment_utils.set_ipdb_trace()

    # Initialize the model and tokenizer 
    device = args.device or "cuda" if torch.cuda.is_available() else "cpu"
    mt = models.load_model(args.model, device=device, fp16=args.fp16)
    # Set up the PASTA steerer 
    if args.apply_pasta:
        head_config = experiment_utils.read_head_config(args.pasta_head_config)
        pasta_steerer = pasta.PASTA(
            mt.model, 
            mt.tokenizer,
            head_config=head_config, 
            alpha=args.alpha, 
            scale_position=args.scale_position,
        )
    else:
        pasta_steerer = None

    # Set up the evaluation data 
    logger.info("loading several data sources")
    if args.example_subset is not None:
        split = f"train[{args.example_subset}]"
    else:
        split = "train[5000:10000]"
    dataset = data.load_dataset("counterfact", split=split)
    dataset = precompute.from_args(args, dataset)
    attribute_snippets = data.load_attribute_snippets()
    tfidf_vectorizer = data.load_counterfact_tfidf_vectorizer()

    dataset = precompute.editor_inputs_from_dataset(
            dataset=dataset,
            mt=mt,
            return_entity_hiddens=False,
            return_attribute_hiddens=False,
            return_token_ranges=False,
            return_target_token_ids=True,
            target_token_first_space=True if "gptj" in args.model else False, 
            desc="precompute target token ids",
    )
    dataset = dataset.map(_prefix_context, desc="prefix context")

    # Set up the benchmark arguments 
    benchmark_kwargs: dict = dict(dataset=dataset, device=device)
    benchmark_kwargs["mt"] = mt
    benchmark_kwargs["pasta_steerer"] = pasta_steerer 
    output_dir = "{pasta}".format(
        pasta = f"pasta-{args.alpha}-{args.scale_position}" if args.apply_pasta else "baseline",
    )
    if args.add_few_shot:
        output_dir = output_dir + "_few-%d-%s"%(args.add_few_shot, str(args.few_shot_index))
    if args.add_marker:
        output_dir = output_dir + "_add-marker-%s"%(args.add_marker)
    results_output_dir = (
        experiment.results_dir 
        / f"{args.model.split('/')[-1]}"
        / output_dir 
    )
    if args.pasta_head_config is not None:
        results_output_dir = results_output_dir / f"{args.pasta_head_config.split('/')[-1]}"
    results_output_dir.mkdir(exist_ok=True, parents=True)
    tb_writter = SummaryWriter(log_dir=str(results_output_dir.resolve()))

    results: (
        benchmarks.EfficacyBenchmarkResults
        | benchmarks.CounterFactParaphraseBenchmarkResults
        | benchmarks.CounterFactGenerationBenchmarkResults
        | benchmarks.EssenceBenchmarkResults
    )
    logger.info(f"eval counterfact")
    
    for benchmark_name in args.benchmarks:
        results_file = results_output_dir / f"{benchmark_name}.json"
        if results_file.exists() and not args.overwrite_output_dir:
            logger.info(
                f"found existing {benchmark_name} results "
                f"at {results_file}"
            )
            continue

        benchmark_kwargs["add_unmediated_fact"] = args.add_unmediated_fact 
        benchmark_kwargs["batch_size"] = args.batch_size
        benchmark_kwargs["max_length"] = args.max_length 
        benchmark_kwargs["add_few_shot"] = args.add_few_shot 
        benchmark_kwargs["few_shot_index"] = args.few_shot_index 
        benchmark_kwargs["add_marker"] = args.add_marker 

        if benchmark_name == "efficacy":
            results = benchmarks.counterfact_efficacy(**benchmark_kwargs)
        elif benchmark_name == "paraphrase":
            results = benchmarks.counterfact_paraphrase(**benchmark_kwargs)
        elif benchmark_name == "generation":
            results = benchmarks.counterfact_generation(
                attribute_snippets=attribute_snippets,
                tfidf_vectorizer=tfidf_vectorizer,
                **benchmark_kwargs,
            )
        else:
            raise ValueError(f"unknown benchmark: {benchmark_name}")

        logging.info(
            f"{benchmark_name} benchmark complete! results:\n%s",
            json.dumps(results.metrics.to_dict(), indent=1),
        )
        for key, value in results.metrics.to_dict().items():
            tb_writter.add_scalar(f"{benchmark_name}/{key}", value['mean'], 1)
        
        with results_file.open("w") as handle:
            json.dump(results.to_dict(), handle)

        metrics_file = results_file.parent / f"{benchmark_name}_metrics.json"
        with metrics_file.open("w") as handle:
            json.dump(results.metrics.to_dict(), handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate editors")
    parser.add_argument(
        "--benchmarks",
        "-b",
        nargs="+",
        choices=BENCHMARKS,
        default=BENCHMARKS,
        help="benchmarks to run, defaults depend on dataset",
    )
    models.add_model_args(parser)
    precompute.add_preprocessing_args(parser)
    experiment_utils.add_experiment_args(parser)
    logging_utils.add_logging_args(parser)
    args = parser.parse_args()
    main(args)
