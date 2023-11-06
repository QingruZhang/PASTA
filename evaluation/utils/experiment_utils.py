"""Utilities for managing experiment runtimes and results."""
import argparse
import json
import logging
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from evaluation.utils import env_utils
from evaluation.utils.typing import PathLike

import numpy
import torch

logger = logging.getLogger(__name__)

DEFAULT_SEED = 123456


@dataclass(frozen=True)
class Experiment:
    """A configured experiment."""

    name: str
    results_dir: Path
    seed: int


def read_head_config(pasta_head_config):
    if '.json' in pasta_head_config:
        with open(pasta_head_config, 'r') as handle:
            head_config = json.load(handle) 
    elif '{' in pasta_head_config and '}' in pasta_head_config:
        head_config = json.loads(pasta_head_config)
    else:
        # head_config = [int(h) for h in pasta_head_config.strip().split(',')]
        raise ValueError("Incorrect format of head config.")
    return head_config


def set_seed(seed: int) -> None:
    """Globally set random seed."""
    logger.info("setting all seeds to %d", seed)
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)


def set_ipdb_trace():
    import ipdb 
    ipdb.set_trace()


def create_results_dir(
    experiment_name: str,
    root: PathLike | None = None,
    args: argparse.Namespace | None = None,
    args_file_name: str | None = None,
    clear_if_exists: bool = False,
) -> Path:
    """Create a directory for storing experiment results.

    Args:
        name: Experiment name.
        root: Root directory to store results in. Consults env if not set.
        args: If set, save the full argparse namespace as JSON.
        args_file: Save args file here.
        clear_if_exists: Clear the results dir if it already exists.

    Returns:
        The initialized results directory.

    """
    if root is None:
        root = env_utils.determine_results_dir()
    root = Path(root)

    results_dir = root / experiment_name
    results_dir = results_dir.resolve()

    if results_dir.exists():
        logger.info(f"rerunning experiment {experiment_name}")
        if clear_if_exists:
            logger.info(f"clearing previous results from {results_dir}")
            shutil.rmtree(results_dir)

    results_dir.mkdir(exist_ok=True, parents=True)
    if args is not None:
        if args_file_name is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            args_file_name = f"args-{timestamp}.json"
        args_file = results_dir / args_file_name
        logger.info(f"saving args to {args_file}")
        with args_file.open("w") as handle:
            json.dump({key: str(value) for key, value in vars(args).items()}, handle)

    return results_dir


def add_experiment_args(parser: argparse.ArgumentParser) -> None:
    """Add args common to all experiments.

    The args include:
        --experiment-name (-n): Requied, unique identifier for this experiment.
        --results-dir: Root directory containing all experiment folders.
        --clear-results-dir: If set, experiment-specific results directory is cleared.
        --args-file-name: Dump all args to this file; defaults to generated name.
        --seed: Random seed.

    """
    parser.add_argument(
        "--experiment_name",
        "-n",
        required=True,
        help="unique name for the experiment",
    )
    parser.add_argument(
        "--apply_pasta", 
        action="store_true",
        help="Whether to apply PASTA."
    )
    parser.add_argument(
        "--pasta_head_config", 
        type=str, default=None, 
        help="PASTA head config for steering.",
    )
    parser.add_argument(
        "--emphasized_text", 
        type=str, default=None, 
        help="Which textual segments to emphasize:[include|exclude]."
    )
    parser.add_argument(
        "--alpha", 
        type=float, default=None, 
        help="Scaling coefficient."
    )
    parser.add_argument(
        "--scale_position", 
        type=str, default=None, 
        help="Steer the selected section or others."
    )
    parser.add_argument("--add_unmediated_fact", type=bool, default=True, help="Present models both facts.")
    # parser.add_argument("--record_attn_score", nargs="+", type=str, default=None, help="")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--max_length", type=int, default=None, help="Max sequence length.")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="Max generation length.")
    parser.add_argument("--add_few_shot", type=int, default=None, help="Apply few-shot prompting.")
    parser.add_argument(
        "--few_shot_index", type=str, default=None, help="Sample index for few shots, e.g., 0,1,2."
    )
    parser.add_argument("--add_marker", type=str, default=None, help="Apply marked prompting.")

    parser.add_argument("--args-file-name", help="file name for args dump")
    parser.add_argument(
        "--example_subset", type=str, default=None, help="run on a subset of data"
    )
    parser.add_argument(
        "--results-dir", type=Path, help="root directory containing experiment results"
    )
    parser.add_argument(
        "--clear-results-dir",
        action="store_true",
        default=False,
        help="clear any old results and start anew",
    )
    parser.add_argument("--overwrite_output_dir", action="store_true", help="")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed")
    parser.add_argument("--debug", action="store_true", help="Whether to invoke ipdb debugging.")


def setup_experiment(args: argparse.Namespace) -> Experiment:
    """Configure experiment from the args."""
    experiment_name = args.experiment_name
    seed = args.seed

    logger.info(f"setting up experiment {experiment_name}")

    set_seed(seed)

    results_dir = create_results_dir(
        experiment_name,
        root=args.results_dir,
        args=args,
        args_file_name=args.args_file_name,
        clear_if_exists=args.clear_results_dir,
    )

    return Experiment(
        name=experiment_name,
        results_dir=results_dir,
        seed=seed,
    )
