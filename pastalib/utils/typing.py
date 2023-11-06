"""Some useful type aliases relevant to this project."""
import pathlib
from typing import Sequence

import datasets
import numpy
import torch
import transformers
import transformers.modeling_outputs

ArrayLike = list | tuple | numpy.ndarray | torch.Tensor
PathLike = str | pathlib.Path
Device = str | torch.device

# Throughout this codebase, we use HuggingFace model implementations
# as well as HuggingFace datasets.
Model = (
    transformers.GPTJForCausalLM
    | transformers.LlamaForCausalLM
    | transformers.GPT2LMHeadModel
    | transformers.GPTNeoXForCausalLM
)
Tokenizer = transformers.PreTrainedTokenizerFast
TokenizerOffsetMapping = Sequence[tuple[int, int]]
Dataset = datasets.arrow_dataset.Dataset | datasets.dataset_dict.DatasetDict
ModelInput = transformers.BatchEncoding
ModelOutput = transformers.modeling_outputs.CausalLMOutput
ModelGenerateOutput = transformers.generation.utils.GenerateOutput | torch.LongTensor

# All strings are also Sequence[str], so we have to distinguish that we
# mean lists or tuples of strings, or sets of strings, not other strings.
StrSequence = list[str] | tuple[str, ...]
