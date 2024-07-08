"""PASTA Implementation"""
import torch
import abc, json
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, cast, overload, Tuple

import transformers 
from pastalib.utils import tokenizer_utils
from pastalib.utils.typing import (
    Model,
    Dataset,
    Device,
    ModelInput,
    ModelOutput,
    StrSequence,
    Tokenizer,
    TokenizerOffsetMapping,
)


class PASTA(abc.ABC):
    """
    Create PASTA to steer attentions of transformer models. 

    Args:
        model ([`transformers.PreTrainedModel`]): The model to be steered. 
        tokenizer ([`transformers.PreTrainedTokenizer`]): The model's tokenizer. 
        head_config (`dict`): The config to control which attention heads to be steered. 
        alpha (`float`): The scaling coefficient of attention steering. 
        scale_position (`str`): To upweight the scores of highlighted tokens (`include`), 
            or to downweight those of unselected tokens (`exclude`). 

    Returns:
        PASTA: PASTA steerer that can register the pre-forward hooks on target models. 

    """

    ATTN_MODULE_NAME = {
        "gptj": "transformer.h.{}.attn",
        "llama": "model.layers.{}.self_attn",
        "mistral": "model.layers.{}.self_attn",
        "gemma": "model.layers.{}.self_attn",
        "phi3mini": "model.layers.{}.self_attn"
    }
    ATTENTION_MASK_ARGIDX = {
        "gptj": 2, 
        "llama": 1, 
        "mistral": 1, 
        "gemma": 1,
    }
    def __init__(
        self, 
        model: Model, 
        tokenizer: Tokenizer, 
        head_config: dict|list|None = None, 
        alpha: float = 0.01, 
        scale_position: str = "exclude", 
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.setup_model(model)

        self.alpha = alpha
        self.scale_position = scale_position
        self.setup_head_config(head_config)

        assert self.scale_position in ['include', 'exclude', 'generation']
        assert self.alpha > 0

    def setup_model(self, model):
        """Obtain the model type and complete the configuration."""
        if isinstance(model, transformers.LlamaForCausalLM):
            self.model_name = "llama"
            self.num_attn_head = model.config.num_attention_heads
        elif isinstance(model, transformers.GPTJForCausalLM):
            self.model_name = "gptj"
            self.num_attn_head = model.config.n_head
        elif isinstance(model, transformers.MistralForCausalLM):
            self.model_name = "mistral"
            self.num_attn_head = model.config.num_attention_heads
        elif isinstance(model, transformers.GemmaForCausalLM):
            self.model_name = "gemma"
            self.num_attn_head = model.config.num_attention_heads
        elif model.__class__.__name__ == "Phi3ForCausalLM":
            self.model_name = "phi3"
        else:
            raise ValueError("Unimplemented Model Type.")
        
    def setup_head_config(self, head_config):
        """
        Config the attention heads to be steered.

        If `head_config` is `list` of layer index, PASTA will steer the entire layers. 
        """
        if isinstance(head_config, dict):
            self.head_config = {int(k):v for k,v in head_config.items()} 
            self.all_layers_idx = [int(key) for key in head_config]
        elif isinstance(head_config, list):
            self.all_layers_idx = [int(v) for v in head_config]
            self.head_config = {
                idx:list(range(self.num_attn_head)) for idx in self.all_layers_idx
            }
        else:
            raise ValueError(f"Incorrect head config: {head_config}")
    
    def _maybe_batch(self, text: str | StrSequence) -> StrSequence:
        """Batch the text if it is not already batched."""
        if isinstance(text, str):
            return [text]
        return text

    def token_ranges_from_batch(
        self,
        strings: str | StrSequence,
        substrings: str | StrSequence,
        offsets_mapping: Sequence[TokenizerOffsetMapping],
        occurrence: int = 0,
    ) -> torch.Tensor:
        """Return shape (batch_size, 2) tensor of token ranges for (str, substr) pairs."""
        strings = self._maybe_batch(strings)
        substrings = self._maybe_batch(substrings)
        if len(strings) != len(substrings):
            raise ValueError(
                f"got {len(strings)} strings but only {len(substrings)} substrings"
            )
        return torch.tensor(
            [
                tokenizer_utils.find_token_range(
                    string, substring, offset_mapping=offset_mapping, occurrence=occurrence
                )
                for string, substring, offset_mapping in zip(
                    strings, substrings, offsets_mapping
                )
            ]
        )

    def edit_attention_mask(
        self, 
        module: torch.nn.Module, 
        input_args: tuple,
        input_kwargs: dict, 
        head_idx: list[int],
        token_range: torch.Tensor, 
        input_len: int, 
    ):
        """
        The hook function registerred pre-forward for attention models. 

        Args: 
            module ([`torch.nn.Module`]): The registerred attention modules. 
            input_args (`tuple`): The positional arguments of forward function. 
            input_kwargs (`dict`): The keyword arguments of forward function. 
            head_idx (`list[int]`): The index of heads to be steered. 
            token_range (`torch.Tensor`): A B*2 tensor, 
                suggesting the index range of hightlight tokens.  
            input_len (`int`): The length L of inputs.

        Returns: 
            tuple, dict: return the modified `attention_mask`,
                while not changing other input arguments. 
        """
        if "attention_mask" in input_kwargs:
            attention_mask = input_kwargs['attention_mask'].clone()
        elif input_args is not None:
            arg_idx = self.ATTENTION_MASK_ARGIDX[self.model_name]
            attention_mask = input_args[arg_idx].clone()
        else:
            raise ValueError(f"Not found attention masks in {str(module)}")
        
        bsz, head_dim, tgt_len, src_len = attention_mask.size()
        dtype, device = attention_mask.dtype, attention_mask.device
        if head_dim != self.num_attn_head:
            attention_mask = attention_mask.expand(
                bsz, self.num_attn_head, tgt_len, src_len
            ).clone()
        scale_constant = torch.Tensor([self.alpha]).to(dtype).to(device).log()
        
        for bi, (ti,tj) in enumerate(token_range.tolist()):
            if self.scale_position == "include":
                attention_mask[bi, head_idx, :, ti:tj] += scale_constant
            else:
                attention_mask[bi, head_idx, :, :ti] += scale_constant
                attention_mask[bi, head_idx, :, tj:input_len] += scale_constant
        
        if self.model_name in ["llama", "mistral", "gemma", "phi3"]:
            attention_mask.old_size = attention_mask.size 
            attention_mask.size = lambda:(bsz, 1, tgt_len, src_len)
        
        if "attention_mask" in input_kwargs:
            input_kwargs['attention_mask'] = attention_mask 
            return input_args, input_kwargs
        else:
            return (input_args[:arg_idx], attention_mask, *input_args[arg_idx+1:]), input_kwargs

    def edit_multisection_attention(
        self, 
        module: torch.nn.Module, 
        input_args: tuple,
        input_kwargs: dict, 
        head_idx: list[int],
        token_ranges: list[torch.Tensor], 
        input_len: int, 
    ):
        """
        The hook function registerred pre-forward for attention models. 

        Args: 
            module ([`torch.nn.Module`]): The registerred attention modules. 
            input_args (`tuple`): The positional arguments of forward function. 
            input_kwargs (`dict`): The keyword arguments of forward function. 
            head_idx (`list[int]`): The index of heads to be steered. 
            token_ranges (`torch.Tensor`): A list of B*2 tensors, 
                suggesting the index range of hightlight tokens of multiple sections.  
            input_len (`int`): The length L of inputs.

        Returns: 
            tuple, dict: return the modified `attention_mask`,
                while not changing other input arguments. 
        """
        if "attention_mask" in input_kwargs:
            attention_mask = input_kwargs['attention_mask'].clone()
        elif input_args is not None:
            arg_idx = self.ATTENTION_MASK_ARGIDX[self.model_name]
            attention_mask = input_args[arg_idx].clone()
        else:
            raise ValueError(f"Not found attention masks in {str(module)}")
        
        bsz, head_dim, tgt_len, src_len = attention_mask.size()
        dtype, device = attention_mask.dtype, attention_mask.device
        if head_dim != self.num_attn_head:
            attention_mask = attention_mask.expand(
                bsz, self.num_attn_head, tgt_len, src_len
            ).clone()
        scale_constant = torch.Tensor([self.alpha]).to(dtype).to(device).log()
        
        for token_range in token_ranges:
            for bi, (ti,tj) in enumerate(token_range.tolist()):
                if self.scale_position == "include":
                    attention_mask[bi, head_idx, :, ti:tj] += scale_constant
                elif self.scale_position == "exclude":
                    attention_mask[bi, head_idx, :, :ti] += scale_constant
                    attention_mask[bi, head_idx, :, tj:input_len] += scale_constant
                elif self.scale_position == "generation":
                    attention_mask[bi, head_idx, :, :input_len] += scale_constant 
                else:
                    raise ValueError(f"Unexcepted {self.scale_position}.")
        if self.scale_position == "include":
            attention_mask[:, head_idx, :, :input_len] -= scale_constant
        
        if self.model_name in ["llama", "mistral"]:
            attention_mask.old_size = attention_mask.size 
            attention_mask.size = lambda:(bsz, 1, tgt_len, src_len)
        
        if "attention_mask" in input_kwargs:
            input_kwargs['attention_mask'] = attention_mask 
            return input_args, input_kwargs
        else:
            return (input_args[:arg_idx], attention_mask, input_args[arg_idx+1:]), input_kwargs


    @contextmanager
    def apply_steering(
        self, 
        model: Model, 
        strings: list, 
        substrings: list, 
        model_input: ModelInput, 
        offsets_mapping: Sequence[TokenizerOffsetMapping], 
        occurrence: int = 0,
    ):
        """
        The function of context manager to register the pre-forward hook on `model`. 

        Args:
            model ([`transformers.PreTrainedModel`]): The transformer model to be steered. 
            strings (`list[str]`): The input strings. 
            substrings (`list[list[str]]` or list[str]): The highlighted input spans for each string. 
            model_input (`transformers.BatchEncoding`): The batched model inputs. 
            offsets_mapping (`TokenizerOffsetMapping`): The offset mapping outputed by
                the tokenizer when encoding the `strings`. 
        """
        if isinstance(substrings[0], str):
            substrings = [substrings]

        token_ranges = []
        for sections in substrings:
            token_range = self.token_ranges_from_batch(
                strings, sections, offsets_mapping, occurrence=occurrence,
            )
            token_ranges.append(token_range)

        registered_hooks = []
        for layer_idx in self.all_layers_idx:
            name = self.ATTN_MODULE_NAME[self.model_name].format(layer_idx)
            module = model.get_submodule(name)
            # Prepare the hook function with partial arguments being fixed. 
            # Pass the head_idx, token_range, input_len for each attention module in advance. 
            hook_func = partial(
                self.edit_multisection_attention, 
                head_idx = self.head_config[layer_idx],
                token_ranges = token_ranges, 
                input_len = model_input['input_ids'].size(-1)
            )
            registered_hook = module.register_forward_pre_hook(hook_func, with_kwargs=True)
            registered_hooks.append(registered_hook)
        try:
            yield model
        except Exception as error:
            raise error
        finally:
            for registered_hook in registered_hooks:
                registered_hook.remove()
    

    def inputs_from_batch(
        self, 
        text: str | StrSequence,
        tokenizer: Tokenizer|None = None,
        device: Optional[Device] = None,
    ) -> tuple[ModelInput, Sequence[TokenizerOffsetMapping]]:
        """Precompute model inputs."""
        if tokenizer is None:
            tokenizer = self.tokenizer
        with tokenizer_utils.set_padding_side(tokenizer, padding_side="left"):
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=False,
                padding="longest",
                return_offsets_mapping=True,
            )
            offset_mapping = inputs.pop("offset_mapping")
        if device is not None:
            inputs = inputs.to(device)
        return inputs, offset_mapping

    @classmethod
    def load_head_config(cls, file:str|Path):
        """Load the `head_config` from JSON file."""
        with open(file, "r") as f:
            head_config = json.load(f)
        return head_config 

