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
    """Run the evaluation for BiasBios prediction task."""
    experiment = experiment_utils.setup_experiment(args)
    logging_utils.configure(args=args)
    data.disable_caching()
    if args.debug: 
        experiment_utils.set_ipdb_trace()

    # Load the model and tokenizer 
    device = args.device or "cuda" if torch.cuda.is_available() else "cpu"
    mt = models.load_model(args.model, device=device, fp16=args.fp16)

    # Set up the PASTA steerer. 
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
    
    # Config benchmark arguments
    benchmark_kwargs: dict = {}
    benchmark_kwargs["batch_size"] = args.batch_size 
    benchmark_kwargs["max_length"] = args.max_length 
    benchmark_kwargs["max_new_tokens"] = args.max_new_tokens 
    benchmark_kwargs["add_few_shot"] = args.add_few_shot 
    benchmark_kwargs["few_shot_index"] = args.few_shot_index 
    benchmark_kwargs["add_marker"] = args.add_marker 
    # Here we follow the REMEDI repo and retain the code
    # to evaluation the consistency as well. 
    tfidf_vectorizer = data.load_biosbias_tfidf_vectorizer()
    benchmark_kwargs["tfidf_vectorizer"] = tfidf_vectorizer

    # Set up the output dir 
    output_dir = "{pasta}".format(
        pasta = f"pasta-{args.alpha}-{args.scale_position}" if args.apply_pasta else "baseline",
    )
    if args.add_few_shot:
        output_dir = output_dir + "_few-%d-%s"%(args.add_few_shot, str(args.few_shot_index))
    if args.add_marker:
        output_dir = output_dir + "_marker-%s"%(args.add_marker)
    result_output_dir = ( 
        experiment.results_dir 
        / f"{args.model.split('/')[-1]}_prefix"
        / output_dir 
    )
    if args.pasta_head_config is not None:
        result_output_dir = result_output_dir / f"{args.pasta_head_config.split('/')[-1]}"
    result_output_dir.mkdir(exist_ok=True, parents=True) 
    result_output_file = result_output_dir / "result.json"

    if not result_output_file.exists() or args.overwrite_output_dir:
        logger.info("begin evaluation")
        results = benchmarks.biasbios_prediction_evaluation(
            mt=mt,
            dataset=dataset,
            pasta_steerer=pasta_steerer, 
            device=device,
            desc="BiasBios Evaluation",
            **benchmark_kwargs,
        )
        logging.info(
            f"Evaluation complete! results:\n%s",
            json.dumps(results.metrics.to_dict(), indent=1),
        )
        # Readout the results
        tb_writter = SummaryWriter(log_dir=str(result_output_dir.resolve()))
        metrics = results.metrics.to_dict() 
        tb_writter.add_scalar("top1_accuracy", metrics['top1_accuracy'], 1)
        tb_writter.add_scalar(f"top{metrics['k']}_accuracy", metrics['topk_accuracy'], 1)
        for key in ["mean", "std"]:
            tb_writter.add_scalar(f"fluency/{key}", metrics['fluency'][key], 1)
            tb_writter.add_scalar(f"consistency/{key}", metrics['consistency'][key], 1)

        result_output_file.parent.mkdir(exist_ok=True, parents=True)
        with result_output_file.open("w") as handle:
            json.dump(results.to_dict(), handle)
        tb_writter.close()
    else:
        logger.info(
            f"existing results found at {result_output_file}; skipping"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluation model generation on BiasBios dataset."
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
