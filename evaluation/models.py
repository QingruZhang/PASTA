"""Functions for loading and interacting with pretrained language models.

This module is designed to house all the annoying branching logic
that comes with supporting analysis of many slightly different model
implementations.
"""
import argparse
import logging
import json 
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Literal, Optional, Sequence, overload

from evaluation.utils import tokenizer_utils
from evaluation.utils.typing import Device, Model, Tokenizer

import torch
import transformers

logger = logging.getLogger(__name__)

GPT_J_NAME_SHORT = "gptj"  # A useful alias for the CLI.
GPT_J_NAME = "EleutherAI/gpt-j-6B"

GPT_NEO_X_NAME_SHORT = "neox"
GPT_NEO_X_NAME = "EleutherAI/gpt-neox-20b"


@dataclass(frozen=True)
class ModelAndTokenizer:
    """A pretrained model and its tokenizer."""

    model: Model
    tokenizer: Tokenizer

    def to_(self, device: Optional[Device]) -> None:
        """Send model to the device."""
        self.model.to(device)

    def eval_(self) -> None:
        """Set model to eval mode."""
        self.model.eval()


def unwrap_model(value: Model | ModelAndTokenizer) -> Model:
    """Unwrap the model if necessary."""
    if isinstance(value, ModelAndTokenizer):
        return value.model
    return value


def unwrap_tokenizer(tokenizer: ModelAndTokenizer | Tokenizer) -> Tokenizer:
    """Unwrap the tokenizer."""
    if isinstance(tokenizer, ModelAndTokenizer):
        return tokenizer.tokenizer
    return tokenizer


def determine_layers(model: ModelAndTokenizer | Model) -> tuple[int, ...]:
    """Return all hidden layer names for the given model."""
    model = unwrap_model(model)
    assert isinstance(model, Model)

    if isinstance(model, transformers.GPTNeoXForCausalLM):
        n_layer = model.config.num_hidden_layers
    else:
        n_layer = model.config.n_layer

    return (*range(n_layer),)


@overload
def determine_layer_paths(
    model: ModelAndTokenizer | Model,
    layers: Optional[Sequence[int]] = ...,
    *,
    return_dict: Literal[False] = ...,
) -> Sequence[str]:
    """Determine layer path for each layer."""
    ...


@overload
def determine_layer_paths(
    model: ModelAndTokenizer | Model,
    layers: Optional[Sequence[int]] = ...,
    *,
    return_dict: Literal[True],
) -> dict[int, str]:
    """Determine mapping from layer to layer path."""
    ...


def determine_layer_paths(
    model: ModelAndTokenizer | Model,
    layers: Optional[Sequence[int]] = None,
    *,
    return_dict: bool = False,
) -> Sequence[str] | dict[int, str]:
    """Determine the absolute paths to the given layers in the model.

    Args:
        model: The model.
        layers: The specific layer (numbers) to look at. Defaults to all of them.
        return_dict: If True, return mapping from layer to layer path,
            otherwise just return list of layer paths in same order as `layers`.

    Returns:
        Mapping from layer number to layer path.

    """
    model = unwrap_model(model)

    if layers is None:
        layers = determine_layers(model)

    assert isinstance(model, Model), type(model)

    layer_paths = {}
    for layer in layers:
        if isinstance(model, transformers.GPTNeoXForCausalLM):
            layer_path = f"gpt_neox.layers.{layer}"
        else:
            layer_path = f"transformer.h.{layer}"
        layer_paths[layer] = layer_path

    return layer_paths if return_dict else tuple(layer_paths[la] for la in layers)


def determine_hidden_size(model: ModelAndTokenizer | Model) -> int:
    """Determine hidden rep size for the model."""
    model = unwrap_model(model)
    return model.config.hidden_size


def determine_device(model: ModelAndTokenizer | Model) -> torch.device | None:
    """Determine device model is running on."""
    parameter = any_parameter(model)
    return parameter.device if parameter is not None else None


def determine_dtype(model: ModelAndTokenizer | Model) -> torch.dtype | None:
    """Determine dtype of model."""
    parameter = any_parameter(model)
    return parameter.dtype if parameter is not None else None


def any_parameter(model: ModelAndTokenizer | Model) -> torch.nn.Parameter | None:
    """Get any example parameter for the model."""
    model = unwrap_model(model)
    return next(iter(model.parameters()), None)


@contextmanager
def set_padding_side(
    tokenizer: Tokenizer | ModelAndTokenizer, padding_side: str = "right"
) -> Iterator[None]:
    """Wrap `tokenizer_utils.set_padding_side`."""
    tokenizer = unwrap_tokenizer(tokenizer)
    with tokenizer_utils.set_padding_side(tokenizer, padding_side=padding_side):
        yield


def map_to(
    orig: Any, device: Device | None = None, dtype: torch.dtype | None = None
) -> Any:
    """Map all tensors in the given value to the device.

    Args:
        orig: Any sequence of or mapping to tensors, or just a tensor.
        device: Device to send to.

    Returns:
        Same value, but with all tensors moved to the device.

    """
    if device is None and dtype is None:
        return orig

    result = orig
    if isinstance(orig, torch.Tensor):
        result = orig.to(device=device, dtype=dtype)
    elif isinstance(orig, dict):
        result = {
            key: map_to(value, device=device, dtype=dtype)
            for key, value in orig.items()
        }
    elif isinstance(orig, (list, tuple)):
        result = orig.__class__(
            map_to(value, device=device, dtype=dtype) for value in orig
        )
    assert isinstance(result, orig.__class__), f"{type(result)}/{type(orig)}"
    return result


def load_model(
    name: str, 
    device: Optional[Device] = None, 
    fp16: Optional[bool] = None, 
) -> ModelAndTokenizer:
    """Load the model given its string name.

    Args:
        name: Name of the model.
        device: If set, send model to this device. Defaults to CPU.
        fp16: Whether to use half precision. If not set, depends on model.

    Returns:
        ModelAndTokenizer: Loaded model and its tokenizer.

    """
    if name == GPT_J_NAME_SHORT:
        name = GPT_J_NAME
    elif name == GPT_NEO_X_NAME_SHORT:
        name = GPT_NEO_X_NAME

    is_gpt_j_variant = name == GPT_J_NAME or GPT_J_NAME_SHORT in name
    is_neo_x_variant = name == GPT_NEO_X_NAME or GPT_NEO_X_NAME_SHORT in name
    if fp16 is None:
        fp16 = is_gpt_j_variant or is_neo_x_variant
    torch_dtype = torch.float16 if fp16 else None
    model_kwargs: dict = dict(torch_dtype=torch_dtype)
    if name == GPT_J_NAME or GPT_J_NAME_SHORT in name:
        model_kwargs["low_cpu_mem_usage"] = True
        if fp16:
            model_kwargs["revision"] = "float16"
    if "llama" in name:
        model_kwargs["torch_dtype"] = torch.bfloat16 
    if device=="auto":
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = torch.bfloat16 

    logger.info(f"loading {name} (device={device}, fp16={fp16})")

    config = transformers.AutoConfig.from_pretrained(name)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        name, config=config, **model_kwargs
    )

    if is_neo_x_variant:
        model.to(torch_dtype)
    if device != "auto":
        model.to(device).eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    tokenizer.pad_token = tokenizer.eos_token

    return ModelAndTokenizer(model, tokenizer)


def add_model_args(parser: argparse.ArgumentParser) -> None:
    """Add args needed to load a model.

    The args include:
        --model: The language model to load, defaulting to GPT-J.
        --device: The device to send model and inputs to.
        --fp16: Whether to use half precision version of the model.
            Note this is used as `--fp16 False` since default value depends on
            which model we are loading.
    """
    parser.add_argument(
        "--model",
        "-m",
        default=GPT_J_NAME_SHORT,
        help="model to edit",
    )

    parser.add_argument("--device", help="device to train on")
    parser.add_argument("--fp16", type=bool, help="set whether to use fp16")
