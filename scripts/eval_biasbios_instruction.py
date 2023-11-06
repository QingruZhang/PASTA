"""Evaluate editor effects on generation for bias setting."""
import argparse
import json
import logging
from pathlib import Path

from evaluation import benchmarks, data, models, precompute
from evaluation.utils import experiment_utils, logging_utils
from pastalib import pasta

import torch
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    """Run the evaluation for instruction following tasks."""
    experiment = experiment_utils.setup_experiment(args)
    logging_utils.configure(args=args)
    data.disable_caching()
    if args.debug: 
        experiment_utils.set_ipdb_trace()

    # Set up the model and tokenzier 
    device = args.device or "cuda" if torch.cuda.is_available() else "cpu"
    mt = models.load_model(args.model, device=device, fp16=args.fp16)

    # Initialize the PASTA steerer 
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
    if args.example_subset is not None:
        split = f"train[{args.example_subset}]"
    else:
        split = "train[5000:10000]"
    dataset = data.load_dataset("biosbias", file=args.dataset_file, split=split)
    dataset = precompute.from_args(args, dataset)
    dataset = precompute.prompt_in_context_from_dataset(
        dataset, output_key="prompt", context_suffix="\n\n"
    )
    
    # Benchmark evaluation arguments 
    benchmark_kwargs: dict = {}
    benchmark_kwargs["emphasized_text"] = args.emphasized_text 
    benchmark_kwargs["batch_size"] = args.batch_size 
    benchmark_kwargs["max_new_tokens"] = args.max_new_tokens 
    benchmark_kwargs["entity_occurrence"] = 1
    benchmark_kwargs["add_few_shot"] = args.add_few_shot 
    benchmark_kwargs["few_shot_index"] = args.few_shot_index 
    if args.max_length is not None:
        benchmark_kwargs["max_length"] = args.max_length 
    tfidf_vectorizer = data.load_biosbias_tfidf_vectorizer()
    benchmark_kwargs["tfidf_vectorizer"] = tfidf_vectorizer
    # Set up output dir 
    output_dir = "{pasta}_{update_section}".format(
        pasta = f"pasta-{args.alpha}-{args.scale_position}" if args.apply_pasta else "pasta-none",
        update_section = f"update-{args.emphasized_text}" if args.emphasized_text else "update-none", 
    )
    if args.add_few_shot:
        output_dir = output_dir + "_few-%d-%s"%(args.add_few_shot, str(args.few_shot_index))
    evaluation_result_dir = ( 
        experiment.results_dir 
        / f"{args.task}" 
        / "{model}_prompt-{prompt_idx}_data-{example_subset}".format(
            model = args.model.split("/")[-1], 
            prompt_idx = f"{','.join([str(idx) for idx in args.prompt_idx])}" if args.prompt_idx else "all",
            example_subset = args.example_subset if args.example_subset else "all", 
            )
        / output_dir 
    )
    if args.pasta_head_config is not None:
        evaluation_result_dir = evaluation_result_dir / f"{args.pasta_head_config.split('/')[-1]}"

    evaluation_result_dir.mkdir(exist_ok=True, parents=True) 
    result_file = evaluation_result_dir / f"instruction_evaluation_{args.task}.json"
    metric_file = evaluation_result_dir / "metric_result.json" 

    if not result_file.exists() or args.overwrite_output_dir:
        logger.info("begin baseline")
        evluation_result = benchmarks.biasbios_instruction_evaluation(
            mt=mt,
            dataset=dataset,
            task=args.task, 
            prompt_idx=args.prompt_idx, 
            pasta_steerer=pasta_steerer, 
            device=device,
            desc="Instruction evaluation [LM]",
            **benchmark_kwargs,
        )
        logging.info(
            f"Evaluation complete! results:\n%s",
            json.dumps(evluation_result.metrics.to_dict(), indent=1),
        )
        tb_writter = SummaryWriter(log_dir=str(evaluation_result_dir.resolve()))

        metrics = evluation_result.metrics.to_dict() 
        tb_writter.add_scalar("top1_accuracy", metrics['top1_accuracy'], 1)
        tb_writter.add_scalar(f"top{metrics['k']}_accuracy", metrics['topk_accuracy'], 1)
        for key in ["mean", "std"]:
            tb_writter.add_scalar(f"fluency/{key}", metrics['fluency'][key], 1)
            tb_writter.add_scalar(f"consistency/{key}", metrics['consistency'][key], 1)
        instruction_evaluation_result = metrics['instruction_evaluation'] 
        for key in instruction_evaluation_result:
            tb_writter.add_scalar(f"instruction_evaluation/{key}", instruction_evaluation_result[key], 1)
        tb_writter.close()

        result_file.parent.mkdir(exist_ok=True, parents=True)
        with result_file.open("w") as handle:
            json.dump(evluation_result.to_dict(), handle)
        with metric_file.open("w") as handle:
            json.dump(metrics, handle) 
    else:
        logger.info(
            f"existing baseline results found at {result_file}; skipping"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="evaluate editor generation on bias dataset"
    )
    parser.add_argument(
        "--task", type=str, default="json", help="The name of evaluation task: [json|pronchange]."
    )
    parser.add_argument(
        "--prompt_idx", nargs="+", type=int, default=0, help="Which prompt template to apply for evaluation."
    )
    parser.add_argument(
        "--dataset_file", type=Path, help="path to dataset.", default=None, 
    )
    models.add_model_args(parser)
    precompute.add_preprocessing_args(parser)
    experiment_utils.add_experiment_args(parser)
    logging_utils.add_logging_args(parser)
    args = parser.parse_args()
    main(args)
