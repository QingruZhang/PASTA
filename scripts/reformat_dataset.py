"""Reformat datasets.

This script handles reformatting and caching dataset from outside sources.
Callers usually should specify --dataset-file and point to the raw data
downloaded from its source.

Once the data is reformatted, you'll no longer have to specify --dataset-file
to any other scripts. The code will simply read it from the cache.
"""
import argparse
from pathlib import Path

from evaluation import data
from evaluation.utils import logging_utils


def main(args: argparse.Namespace) -> None:
    """Do the reformatting by loading the dataset once."""
    # data.disable_caching()
    logging_utils.configure(args=args)
    # data.load_dataset(args.dataset, file=args.dataset_file)

    file = data._reformat_bias_in_bios_file(
        args.biasbios_raw_path, bio_min_words=10, sent_min_words=3, 
        file_name=args.biasbios_save_file, sents_choice="all", 
        attr_sent_idx=0,
    )
    return file 



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="reformat datasets on disk")
    data.add_dataset_args(parser)
    parser.add_argument(
        "--biasbios_raw_path", type=Path, help="path to BiasBios raw file.", default=None, 
    )
    parser.add_argument(
        "--biasbios_save_file", type=Path, help="path to save the dataset.", default=None, 
    )
    logging_utils.add_logging_args(parser)
    args = parser.parse_args()
    main(args)
