"""Utilities to make using python logging module easier."""
import argparse
import logging
import sys
from typing import Any

DEFAULT_FORMAT = "%(asctime)s %(name)s %(levelname)-8s %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LEVEL = logging.INFO


def configure(args: argparse.Namespace | None = None, **kwargs: Any) -> None:
    """Configure logging globally with reasonable defaults."""
    if "level" in kwargs:
        level = kwargs.get("level", DEFAULT_LEVEL)
    elif args is not None:
        level = getattr(args, "log_level", DEFAULT_LEVEL)
    else:
        level = DEFAULT_LEVEL

    kwargs.setdefault("stream", sys.stdout)
    kwargs.setdefault("format", DEFAULT_FORMAT)
    kwargs.setdefault("datefmt", DEFAULT_DATEFMT)
    kwargs.setdefault("level", level)
    logging.basicConfig(**kwargs)


def add_logging_args(parser: argparse.ArgumentParser) -> None:
    """Add --verbose (-v) and --quiet (-q) args to the parser."""
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_const",
        dest="log_level",
        const=logging.DEBUG,
        default=logging.INFO,
        help="show all log statements",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_const",
        dest="log_level",
        const=logging.WARNING,
        help="hide all non-critical messages",
    )
