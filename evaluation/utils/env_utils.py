"""Tools for reading and controlling the runtime environment."""
import os
import pathlib

from evaluation.utils.typing import PathLike

ENV_DATA_DIR = "CM_DATA_DIR"
ENV_MODELS_DIR = "CM_MODELS_DIR"
ENV_RESULTS_DIR = "CM_RESULTS_DIR"

DEFAULT_DATA_DIR = "data"
DEFAULT_MODELS_DIR = "models"
DEFAULT_RESULTS_DIR = "results"


def maybe_relative_to_repo(path: PathLike) -> pathlib.Path:
    """Resolve the (potentially relative) path.

    Args:
        path (PathLike): The path to resolve. If the path is relative, it is
            assumed to be relative to the repository root. If this is already
            an absolute path, it is returned unchanged.

    Returns:
        pathlib.Path: The resolved path.

    """
    path = pathlib.Path(path)
    if path.is_absolute():
        return path
    return pathlib.Path(__file__).parents[2] / path


def read_path(name: str, default: PathLike) -> pathlib.Path:
    """Try to read a path from the env.

    Args:
        name (str): Name of the env variable to read.
        default (PathLike): The default path in case the environment variable
            does not exist. If relative, assumed to be relative from the
            repo root.

    Returns:
        PathLike: The path, if the variable could be read and/or
            if default was provided.

    """
    read = os.environ.get(name)
    path: PathLike = maybe_relative_to_repo(default) if read is None else read
    return pathlib.Path(path)


def determine_data_dir(default: PathLike = DEFAULT_DATA_DIR) -> pathlib.Path:
    """Return directory containing project datasets.

    Args:
        default (PathLike, optional): Default to use if CM_DATA_DIR env
            variable is not set. Defaults to './data'.

    Returns:
        pathlib.Path: Directory data is stored in.

    """
    return read_path(ENV_DATA_DIR, default)


def determine_models_dir(default: PathLike = DEFAULT_MODELS_DIR) -> pathlib.Path:
    """Return directory containing project models.

    Args:
        default (PathLike, optional): Default to use if CM_MODELS_DIR env
            variable is not set. Defaults to './models'.

    Returns:
        pathlib.Path: Directory data is stored in.

    """
    return read_path(ENV_MODELS_DIR, default)


def determine_results_dir(default: PathLike = DEFAULT_RESULTS_DIR) -> pathlib.Path:
    """Return directory containing results from any scripts.

    Args:
        default (PathLike, optional): Default to use if CM_RESULTS_DIR env
            variable is not set. Defaults to './results'.

    Returns:
        pathlib.Path: Directory results are stored in.

    """
    return read_path(ENV_RESULTS_DIR, default)
