"""Logic for getting and mucking with model hidden representations."""
import argparse
from functools import partial
from typing import Any, Literal, Optional, Sequence, cast, overload

from evaluation import data, models
from evaluation.utils import tokenizer_utils
from evaluation.utils.typing import (
    Dataset,
    Device,
    ModelInput,
    ModelOutput,
    StrSequence,
    Tokenizer,
    TokenizerOffsetMapping,
)

import torch
from baukit import nethook


def _remove_sent_case(text: str) -> str:
    """Make the string NOT sentence case (first letter lowercase)."""
    return text[0].lower() + text[1:]


def _is_batched(text: str | StrSequence) -> bool:
    """Determine if text is batched or not."""
    return not isinstance(text, str)


def _maybe_batch(text: str | StrSequence) -> StrSequence:
    """Batch the text if it is not already batched."""
    if isinstance(text, str):
        return [text]
    return text


def _as_fp32(data: dict) -> dict:
    """Cast all top-level float tensor values to float32."""
    return {
        key: value.float()
        if isinstance(value, torch.Tensor) and value.dtype.is_floating_point
        else value
        for key, value in data.items()
    }


def _validate_lengths(lengths: torch.Tensor) -> None:
    """Validate sequence lengths tensor is correct shape."""
    if len(lengths.shape) != 1:
        raise ValueError(f"misshapen lengths: {lengths.shape}")


def _validate_token_ranges(
    token_ranges: torch.Tensor, batch_size: int | None = None
) -> None:
    """Validate token ranges are correct shape."""
    if len(token_ranges.shape) != 2 or token_ranges.shape[1] != 2:
        raise ValueError(f"misshapen token ranges: {token_ranges.shape}")
    if batch_size is not None and token_ranges.shape[0] != batch_size:
        raise ValueError(
            f"expected batch_size={batch_size}, got {token_ranges.shape[0]}"
        )


def inputs_from_batch(
    mt: models.ModelAndTokenizer,
    text: str | StrSequence,
    device: Optional[Device] = None,
) -> tuple[ModelInput, Sequence[TokenizerOffsetMapping]]:
    """Precompute model inputs."""
    inputs = mt.tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="longest",
        return_offsets_mapping=True,
    )
    offset_mapping = inputs.pop("offset_mapping")
    if device is not None:
        inputs = inputs.to(device)
    return inputs, offset_mapping


def last_token_index_from_batch(inputs: ModelInput) -> Sequence[int]:
    """Return index of last token for each item in batch, accounting for padding."""
    return inputs.attention_mask.sum(dim=-1).cpu() - 1


HiddensByLayer = dict[int, torch.Tensor]


@overload
def hiddens_from_batch(
    mt: models.ModelAndTokenizer,
    inputs: str | StrSequence | ModelInput,
    stop: Literal[True] = True,
    layers: Optional[Sequence[int]] = None,
    device: Optional[Device] = None,
) -> HiddensByLayer:
    ...


@overload
def hiddens_from_batch(
    mt: models.ModelAndTokenizer,
    inputs: str | StrSequence | ModelInput,
    stop: Literal[False],
    layers: Optional[Sequence[int]] = None,
    device: Optional[Device] = None,
) -> tuple[HiddensByLayer, ModelOutput]:
    ...


@torch.inference_mode()
def hiddens_from_batch(
    mt: models.ModelAndTokenizer,
    inputs: str | StrSequence | ModelInput,
    stop: bool = True,
    layers: Optional[Sequence[int]] = None,
    device: Optional[Device] = None,
) -> HiddensByLayer | tuple[HiddensByLayer, ModelOutput]:
    """Precomptue hidden reps.

    Args:
        mt: The model and tokenizer.
        inputs: The model inputs.
        stop: Stop computation after retrieving the hiddens.
        layers: Layers to compute hiddens for. Defaults to all.
        device: Send model and inputs to this device.

    Returns:
        Hidden reps mapped by layer and model output (if stop=False).

    """
    mt.to_(device)
    if isinstance(inputs, str | list | tuple):
        inputs, _ = inputs_from_batch(mt, inputs, device=device)
    if device is not None:
        inputs = inputs.to(device)
    outputs = None
    layer_paths = models.determine_layer_paths(mt, layers=layers, return_dict=True)
    with nethook.TraceDict(mt.model, layers=layer_paths.values(), stop=stop) as ret:
        outputs = mt.model(**inputs)

    hiddens_by_layer = {}
    for layer, layer_path in layer_paths.items():
        hiddens = ret[layer_path].output
        if isinstance(hiddens, tuple):
            hiddens = hiddens[0]
        assert isinstance(hiddens, torch.Tensor)
        hiddens_by_layer[layer] = hiddens.cpu()

    return hiddens_by_layer if stop else (hiddens_by_layer, outputs)


def token_ranges_from_batch(
    strings: str | StrSequence,
    substrings: str | StrSequence,
    offsets_mapping: Sequence[TokenizerOffsetMapping],
    occurrence: int = 0,
) -> torch.Tensor:
    """Return shape (batch_size, 2) tensor of token ranges for (str, substr) pairs."""
    strings = _maybe_batch(strings)
    substrings = _maybe_batch(substrings)
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


def last_token_ranges_from_batch(token_ranges: torch.Tensor) -> torch.Tensor:
    """Convert batch of token ranges to only include last token."""
    _validate_token_ranges(token_ranges)
    token_ranges = token_ranges.clone()
    token_ranges[:, 0] = token_ranges[:, 1] - 1
    return token_ranges


def negative_token_ranges_from_batch(
    token_ranges: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    """Convert positive token ranges to negative ones."""
    _validate_lengths(lengths)
    _validate_token_ranges(token_ranges, batch_size=len(lengths))
    return token_ranges.cpu() - lengths[:, None].cpu()


def first_token_ids_from_batch(
    mt: models.ModelAndTokenizer | Tokenizer, words: str | StrSequence, add_space: bool = True,
) -> torch.Tensor:
    """Return shape (batch_size,) int tensor with first token ID for each word."""
    words = _maybe_batch(words)
    tokenizer = models.unwrap_tokenizer(mt)
    # TODO(evandez): Centralize this spacing nonsense.
    if add_space:
        token_ids = tokenizer([" " + word for word in words], add_special_tokens=False)
    else:
        token_ids = tokenizer(words, add_special_tokens=False)
    return torch.tensor([ti[0] for ti in token_ids.input_ids])


def average_hiddens_from_batch(
    hiddens: torch.Tensor, ranges: Sequence[Sequence[int]] | torch.Tensor
) -> torch.Tensor:
    """Compute average hidden rep in given token ranges.

    Args:
        hiddens: Should have shape (batch_size, sequence_length, hidden_size)
        ranges: Token ranges.

    Returns:
        Shape (batch_size, hidden_size) tensor containing average hiddens.

    """
    if not isinstance(ranges, torch.Tensor):
        ranges = torch.tensor(ranges)
    if ranges.shape != (hiddens.shape[0], 2):
        raise ValueError(f"unexpected ranges shape: {ranges.shape}")

    averages = []
    for bi, (ti, tj) in enumerate(ranges.tolist()):
        average = hiddens[bi, ti:tj].mean(dim=0)
        averages.append(average)

    return torch.stack(averages)


def editor_inputs_from_batch(
    mt: models.ModelAndTokenizer,
    batch: data.ContextMediationInput,
    layers: Optional[Sequence[int]] = None,
    device: Optional[Device] = None,
    entity_occurrence_in_prompt: int = 0,
    return_token_ranges: bool = False,
    return_target_token_ids: bool = True,
    return_entity_hiddens: bool = False,
    return_attribute_hiddens: bool = False,
    fp32: bool = False,
    target_token_first_space: bool = True, 
) -> dict:
    """Precompute everything the editor model needs to run from the batch."""
    mt.model.to(device)

    # Pull out expected values.
    entities = _maybe_batch(batch["entity"])
    prompts = _maybe_batch(batch["prompt"])
    contexts = _maybe_batch(batch["context"])
    attributes = _maybe_batch(batch["attribute"])

    precomputed: dict = {}

    # Precompute inputs.
    prompt_inputs, prompt_offset_mapping = None, None
    context_inputs, context_offset_mapping = None, None
    entity_inputs, entity_offset_mapping = None, None
    with models.set_padding_side(mt, padding_side="right"):
        if return_token_ranges:
            prompt_inputs, prompt_offset_mapping = inputs_from_batch(mt, prompts)
        if return_attribute_hiddens or return_token_ranges:
            context_inputs, context_offset_mapping = inputs_from_batch(
                mt, contexts, device=device
            )
        if return_entity_hiddens or return_token_ranges:
            entity_inputs, entity_offset_mapping = inputs_from_batch(
                mt, entities, device=device
            )

    # Precompute token ranges if needed.
    if return_token_ranges or return_attribute_hiddens:
        assert prompt_inputs is not None and prompt_offset_mapping is not None
        assert context_inputs is not None and context_offset_mapping is not None
        assert entity_inputs is not None and entity_offset_mapping is not None
        for (
            key_string,
            key_substring,
            strings,
            substrings,
            inputs,
            offset_mapping,
            occurrence,
        ) in (
            (
                "entity",
                "entity",
                entities,
                entities,
                entity_inputs,
                entity_offset_mapping,
                0,
            ),
            (
                "prompt",
                "entity",
                prompts,
                entities,
                prompt_inputs,
                prompt_offset_mapping,
                entity_occurrence_in_prompt,
            ),
            (
                "context",
                "attribute",
                contexts,
                attributes,
                context_inputs,
                context_offset_mapping,
                0,
            ),
        ):
            lengths = inputs.attention_mask.sum(dim=-1).cpu()
            precomputed[f"{key_string}.length"] = lengths

            key = f"{key_string}.{key_substring}"

            key_tr_base = f"{key}.token_range"
            precomputed[key_tr_base] = tr = token_ranges_from_batch(
                strings,
                substrings,
                offsets_mapping=offset_mapping,
                occurrence=occurrence,
            )

            key_tr_neg = f"{key}.negative_token_range"
            precomputed[key_tr_neg] = negative_token_ranges_from_batch(tr, lengths)

            key_tr_base_last = f"{key_tr_base}.last"
            precomputed[key_tr_base_last] = ltr = last_token_ranges_from_batch(tr)

            key_tr_neg_last = f"{key_tr_neg}.last"
            precomputed[key_tr_neg_last] = negative_token_ranges_from_batch(
                ltr, lengths
            )

    # Precompute token IDs if needed.
    if return_target_token_ids:
        for target_key in ("target_mediated", "target_unmediated"):
            target = cast(str | list, batch.get(target_key))
            if target is None or any(t is None for t in target):
                continue
            precomputed[f"{target_key}.token_id"] = first_token_ids_from_batch(
                mt, target, target_token_first_space 
            )

    # Precompute average/last hidden reps if needed.
    for (
        key_string,
        key_substring,
        condition,
        inputs,
    ) in (
        ("entity", "entity", return_entity_hiddens, entity_inputs),
        ("context", "attribute", return_attribute_hiddens, context_inputs),
    ):
        if not condition:
            continue

        key = f"{key_string}.{key_substring}"

        key_token_range = f"{key}.token_range"
        token_ranges = precomputed[key_token_range]
        token_ranges_last = precomputed[f"{key_token_range}.last"]

        hiddens_by_layer = hiddens_from_batch(mt, inputs, layers=layers, device=device)
        for layer, hiddens in hiddens_by_layer.items():
            key_hiddens = f"{key}.hiddens.{layer}"
            key_hiddens_last = f"{key_hiddens}.last"
            precomputed[key_hiddens_last] = average_hiddens_from_batch(
                hiddens, token_ranges_last
            )

            key_hiddens_average = f"{key_hiddens}.average"
            precomputed[key_hiddens_average] = average_hiddens_from_batch(
                hiddens, token_ranges
            )

    if fp32:
        precomputed = _as_fp32(precomputed)

    return precomputed


def edit_input_string_attn_mask(
        strings: list, 
        substrings: list, 
        inputs: ModelInput,
        offset_mapping,
        attn_scale: float, 
        attn_scale_pos: str, 
        occurrence: int = 0,
        device: Optional[Device] = None,
    ):
    # Calculate the token ranges for attribute
    precomputed: dict = dict()
    key_tr_base = f"token_range"
    precomputed[key_tr_base] = tr = token_ranges_from_batch(
        strings,
        substrings,
        offsets_mapping=offset_mapping,
        occurrence=occurrence,
    )

    attention_mask = inputs['attention_mask'].float().clone()
    for bi, (ti, tj) in enumerate(tr.tolist()):
        if attn_scale_pos == "include":
            attention_mask[bi, ti:tj] = attention_mask[bi, ti:tj]*attn_scale
        elif attn_scale_pos == "exclude":
            attention_mask[bi, :ti] = attention_mask[bi, :ti]*attn_scale
            attention_mask[bi, tj:] = attention_mask[bi, tj:]*attn_scale
        else:
            raise ValueError(f"Unimplemented scale pos: {str(attn_scale_pos)}")
    inputs['attention_mask'] = attention_mask
        
    if device is not None:
        inputs = inputs.to(device)
    
    return inputs, precomputed


def edit_input_attn_mask_for_biosbias(
        batch: data.ContextMediationInput, 
        key_string: str,
        key_substring: str,
        inputs: ModelInput,
        offset_mapping,
        attn_scale: float, 
        attn_scale_pos: str, 
        occurrence: int = 0,
        device: Optional[Device] = None,
    ):
    strings = _maybe_batch(batch[key_string]) 
    substrings = _maybe_batch(batch[key_substring]) 
    precomputed: dict = dict()

    # Calculate the token ranges for attribute
    key = f"{key_string}.{key_substring}"
    key_tr_base = f"{key}.token_range"
    precomputed[key_tr_base] = tr = token_ranges_from_batch(
        strings,
        substrings,
        offsets_mapping=offset_mapping,
        occurrence=occurrence,
    )

    attention_mask = inputs['attention_mask'].float().clone()
    for bi, (ti, tj) in enumerate(tr.tolist()):
        if attn_scale_pos == "include":
            attention_mask[bi, ti:tj] = attention_mask[bi, ti:tj]*attn_scale
        elif attn_scale_pos == "exclude":
            attention_mask[bi, :ti] = attention_mask[bi, :ti]*attn_scale
            attention_mask[bi, tj:] = attention_mask[bi, tj:]*attn_scale
        else:
            raise ValueError(f"Unimplemented scale pos: {str(attn_scale_pos)}")
    inputs['attention_mask'] = attention_mask
        
    if device is not None:
        inputs = inputs.to(device)
    
    return inputs, precomputed


def edit_input_attn_mask_from_batch(
    mt: models.ModelAndTokenizer,
    batch: data.ContextMediationInput,
    attn_scale: float | None = None, 
    attn_scale_pos: str | None = None,  
    add_unmediated_fact: str|None = None, 
    add_marker: str|None = None,
    fewshot_examples: str|None = None, 
    device: Optional[Device] = None,
) -> dict:
    """Precompute everything the editor model needs to run from the batch."""
    mt.model.to(device)

    # Pull out expected values.
    prompts = _maybe_batch(batch["prompt"])
    contexts = _maybe_batch(batch["context"])
    attributes = _maybe_batch(batch["attribute"])
    targets_mediated = _maybe_batch(batch["target_mediated"])
    targets_unmediated = _maybe_batch(batch["target_unmediated"])

    if add_unmediated_fact is not None:
        new_prompts = []
        for prompt, context, target_mediated, target_unmediated in zip(
            prompts, contexts, targets_mediated, targets_unmediated
        ):
            unmediated_prefix = "Previously "
            mediated_prefix = "Currently "
            if add_unmediated_fact == "left":
                unmediated_fact = context.replace(target_mediated, target_unmediated)+". "
                new_prompt = f"{unmediated_prefix}{unmediated_fact}{mediated_prefix}{prompt}"
            elif add_unmediated_fact == "right":
                prompt = prompt.replace(target_mediated, target_unmediated)
                context = context.rstrip(". ") + ". "
                new_prompt = f"{mediated_prefix}{context}{unmediated_prefix}{prompt}"
            new_prompts.append(new_prompt)

        prompts = new_prompts

        if add_marker is not None:
            prompts = [
                prompt.replace(attr, add_marker+attr+add_marker) for prompt,attr in zip(prompts, attributes)
            ]
        
        if fewshot_examples is not None:
            prompts = [
                fewshot_examples + prompt for prompt in prompts
            ]

    # Precompute inputs.
    prompt_inputs, prompt_offset_mapping = None, None
    with models.set_padding_side(mt, padding_side="left"):
        prompt_inputs, prompt_offset_mapping = inputs_from_batch(mt, prompts)

    # precomputed: dict = {}
      
    def edit_attention_mask(
        key_string,
        key_substring,
        strings,
        substrings,
        inputs,
        attn_scale, 
        attn_scale_pos,
        offset_mapping,
        occurrence,
    ):
        precomputed: dict = {}
        lengths = inputs.attention_mask.sum(dim=-1).cpu()
        precomputed[f"{key_string}.length"] = lengths

        key = f"{key_string}.{key_substring}"
        key_tr_base = f"{key}.token_range"
        precomputed[key_tr_base] = tr = token_ranges_from_batch(
            strings,
            substrings,
            offsets_mapping=offset_mapping,
            occurrence=occurrence,
        )

        if isinstance(attn_scale, float):
            attention_mask = inputs['attention_mask'].float().clone()
            for bi, (ti, tj) in enumerate(tr.tolist()):
                if attn_scale_pos == "include":
                    attention_mask[bi, ti:tj] = attention_mask[bi, ti:tj]*attn_scale
                elif attn_scale_pos == "exclude":
                    attention_mask[bi, :ti] = attention_mask[bi, :ti]*attn_scale
                    attention_mask[bi, tj:] = attention_mask[bi, tj:]*attn_scale
                else:
                    raise ValueError(f"Unimplemented scale pos: {str(attn_scale_pos)}")
            inputs['attention_mask'] = attention_mask
            
        if device is not None:
            inputs = inputs.to(device)
        
        return inputs, precomputed
    
    new_inputs, precomputed = edit_attention_mask(
            "prompt",
            "attribute",
            prompts,
            attributes,
            prompt_inputs,
            attn_scale, 
            attn_scale_pos,
            prompt_offset_mapping,
            0,
        )
    precomputed["new_prompt"] = prompts 

    return new_inputs, precomputed


def prepare_counterfact_few_shot_examples(
        fewshot_samples, 
        add_unmediated_fact="left", unmediated_prefix="Previously ", mediated_prefix="Currently ",
        example_sep = "\n###\n", text_target_sep = " "
    ):
    labeled_examples = [] 
    for example in fewshot_samples:
        prompt = example['prompt']
        context = example['context']
        target_mediated = example['target_mediated']
        target_unmediated = example['target_unmediated']

        if add_unmediated_fact == "left":
            unmediated_fact = context.replace(target_mediated, target_unmediated)+". "
            new_prompt = f"{unmediated_prefix}{unmediated_fact}{mediated_prefix}{prompt}"
        elif add_unmediated_fact == "right":
            prompt = prompt.replace(target_mediated, target_unmediated)
            context = context.rstrip(". ") + ". "
            new_prompt = f"{mediated_prefix}{context}{unmediated_prefix}{prompt}"
        else:
            raise ValueError("Unexpected %s"%(add_unmediated_fact))

        # text = example[text_key]
        # entity = example['entity']
        # target = example['target_mediated'] 
        # text_label = text + text_target_sep + target 

        text_label = new_prompt + text_target_sep + target_mediated
        labeled_examples.append(text_label) 
    
    fewshot_examples = example_sep.join(labeled_examples) + example_sep
    return fewshot_examples 


def editor_inputs_from_dataset(
    mt: models.ModelAndTokenizer,
    dataset: Dataset,
    layers: Optional[Sequence[int]] = None,
    device: Optional[Device] = None,
    batch_size: int = 64,
    desc: str | None = "precompute editor inputs",
    **kwargs: Any,
) -> Dataset:
    """Precompute everything the editor model needs to train and run."""
    if "fp32" in kwargs:
        raise ValueError("cannot set fp32= because arrow datasets only support fp32")
    return dataset.map(
        partial(
            editor_inputs_from_batch,
            mt,
            layers=layers,
            device=device,
            fp32=True,
            **kwargs,
        ),
        batched=True,
        batch_size=batch_size,
        desc=desc,
        keep_in_memory=True,
        num_proc=1,
    )


def has_editor_inputs(batch: dict) -> bool:
    """Determine if editor inputs already precomputed."""
    return "prompt.entity.token_range" in batch  # Check for just one flag entry.


def prompt_in_context_from_sample(
    entity: str,
    prompt: str,
    context: str,
    context_prefix: str | None = None,
    context_suffix: str | None = None,
    prompt_prefix: str | None = None,
) -> str:
    """Compute prompt in context for the sample.

    The prompt in context is simply the "prompt" field in each sample prepended with
    the "context" field. This function tries to make the casing look sensible while
    also not changing the casing of any entity mention.

    Can optionally include prefixes for all contexts and/or for all prompts. This is
    useful for adding function or transition words between the prompt and context so
    that the language model can better reconcile the task.

    Args:
        entity: The entity.
        prompt: The prompt.
        context: The context.
        context_prefix: Prepend this to context.
        context_suffix: Append this to context.
        prompt_prefix: Prepend this to prompt, but after context.


    Returns:
        A single string with the context followed by the prompt.

    """
    if prompt_prefix is not None:
        if not prompt.startswith(entity):
            prompt = _remove_sent_case(prompt)
        prompt = f"{prompt_prefix}{prompt}"

    if context_prefix is not None:
        if not context.startswith(entity):
            context = _remove_sent_case(context)
        context = f"{context_prefix} {context}"

    # Always make sure context is a complete, period-ended sentence.
    context = context.rstrip(". ") + "."

    if context_suffix is not None:
        context = f"{context}{context_suffix}"
    else:
        context += " "

    prompt_in_context = f"{context}{prompt}"
    return prompt_in_context


def prompt_in_context_from_batch(
    batch: data.ContextMediationInput,
    output_key: str = "prompt_in_context",
    **kwargs: Any,
) -> dict:
    """Compute prompt in context from batch."""
    is_batched = _is_batched(batch["entity"])
    entities = _maybe_batch(batch["entity"])
    prompts = _maybe_batch(batch["prompt"])
    contexts = _maybe_batch(batch["context"])

    prompts_in_context = []
    for entity, prompt, context in zip(entities, prompts, contexts):
        prompt_in_context = prompt_in_context_from_sample(
            entity, prompt, context, **kwargs
        )
        prompts_in_context.append(prompt_in_context)

    return {output_key: prompts_in_context if is_batched else prompts_in_context[0]}


def prompt_in_context_from_dataset(
    dataset: Dataset, desc: str | None = "precompute prompt in context", **kwargs: Any
) -> Dataset:
    """Compute prompt in context for whole dataset."""
    return dataset.map(
        partial(prompt_in_context_from_batch, **kwargs),
        desc=desc,
        keep_in_memory=True,
    )


def has_prompt_in_context(batch: dict) -> bool:
    """Check if prompt_in_context has already been computed."""
    return "prompt_in_context" in batch


def entity_deltas_from_batch(
    mt: models.ModelAndTokenizer,
    batch: data.ContextMediationInput,
    layers: Optional[Sequence[int]] = None,
    device: Optional[Device] = None,
    fp32: bool = False,
    return_token_ranges: bool = True,
    return_deltas: bool = True,
    **kwargs: Any,
) -> dict:
    """Compute in-context entity delta for the batch.

    Keyword arguments are forwarded to `prompt_in_context_from_batch`.

    Args:
        mt: Model and tokenizer.
        batch: Context mediation-style batch.
        layers: Layers to compute deltas in. Defaults to all.
        device: Send model and inputs to this device.
        fp32: Force cast each tensor to fp32.
        return_token_ranges: Return entity token ranges for contextualied prompt.
        return_deltas: Return entity deltas for each layer.

    Returns:
        Deltas and related precomputed values.

    """
    precomputed: dict = prompt_in_context_from_batch(batch, **kwargs)

    entities = _maybe_batch(batch["entity"])
    prompts_in_context = precomputed["prompt_in_context"]

    inputs = None
    first_entity_token_ranges = None
    last_entity_token_ranges = None
    if return_token_ranges or return_deltas:
        with models.set_padding_side(mt, padding_side="right"):
            inputs, offset_mapping = inputs_from_batch(
                mt, prompts_in_context, device=device
            )
        first_entity_token_ranges = token_ranges_from_batch(
            prompts_in_context, entities, offset_mapping
        )
        last_entity_token_ranges = token_ranges_from_batch(
            prompts_in_context, entities, offset_mapping, occurrence=1
        )

    if return_token_ranges:
        assert first_entity_token_ranges is not None
        assert last_entity_token_ranges is not None
        for position, token_ranges in (
            ("first", first_entity_token_ranges),
            ("last", last_entity_token_ranges),
        ):
            key = f"prompt_in_context.entity.token_range.{position}"
            precomputed[key] = token_ranges

    if return_deltas:
        assert inputs is not None
        assert first_entity_token_ranges is not None
        assert last_entity_token_ranges is not None
        hiddens_by_layer = hiddens_from_batch(mt, inputs, layers=layers, device=device)
        for layer, hiddens in hiddens_by_layer.items():
            first_entity_hiddens = average_hiddens_from_batch(
                hiddens, first_entity_token_ranges
            )
            last_entity_hiddens = average_hiddens_from_batch(
                hiddens, last_entity_token_ranges
            )
            delta = last_entity_hiddens - first_entity_hiddens

            key = f"prompt_in_context.entity.delta.{layer}"
            precomputed[key] = delta

    if fp32:
        precomputed = _as_fp32(precomputed)

    return precomputed


def entity_deltas_from_dataset(
    mt: models.ModelAndTokenizer,
    dataset: Dataset,
    layers: Optional[Sequence[int]] = None,
    device: Optional[Device] = None,
    batch_size: int = 64,
    desc: str | None = "precompute entity deltas",
    **kwargs: Any,
) -> Dataset:
    """Precompute entity deltas in context for the whole dataset."""
    return dataset.map(
        partial(
            entity_deltas_from_batch,
            mt,
            layers=layers,
            device=device,
            fp32=True,
            **kwargs,
        ),
        batched=True,
        batch_size=batch_size,
        desc=desc,
        keep_in_memory=True,
        num_proc=1,
    )


def has_entity_deltas(batch: dict) -> bool:
    """Return True if the batch already has precomputed entity deltas."""
    return any("delta" in key for key in batch)


def classification_inputs_from_batch(
    mt: models.ModelAndTokenizer,
    batch: data.ContextMediationInput,
    layers: Optional[Sequence[int]] = None,
    device: Optional[Device] = None,
    fp32: bool = False,
    **kwargs: Any,
) -> dict:
    """Precompute classification inputs for the batch.

    An extension of `editor_inputs_from_batch` that additionally computes attribute
    directions for the unmediated case and entity representations in context.

    Args:
        mt: Model and tokenizer.
        batch: The batch.
        layers: Model layers to classify entities at.
        device: Send model and inputs to this device.
        fp32: Force cast each tensor to fp32.

    Returns:
        Batch with precomputed values.

    """
    precomputed: dict = {**batch}
    if not has_editor_inputs(precomputed):
        precomputed = editor_inputs_from_batch(
            mt, batch, layers=layers, device=device, fp32=fp32, **kwargs
        )
    if not has_prompt_in_context(precomputed):
        precomputed.update(prompt_in_context_from_batch(batch))

    entities = _maybe_batch(batch["entity"])
    prompts = _maybe_batch(batch["prompt"])
    prompts_in_context = cast(list[str], precomputed["prompt_in_context"])

    contexts_m = _maybe_batch(batch["context"])
    attributes_m = _maybe_batch(batch["attribute"])

    targets_m = batch["target_mediated"]
    if targets_m is None:
        raise ValueError("target_mediated cannot be None")
    targets_m = _maybe_batch(targets_m)
    targets_m_ids = precomputed["target_mediated.token_id"]

    targets_u = batch.get("target_unmediated")
    targets_u_ids = None
    contexts_u = None
    attributes_u = None
    if targets_u is not None and all(t is not None for t in targets_u):
        targets_u = _maybe_batch(targets_u)
        targets_u_ids = precomputed["target_unmediated.token_id"]

        precomputed["context_unmediated"] = contexts_u = [
            context.replace(target_m, target_u)
            for context, target_m, target_u in zip(contexts_m, targets_m, targets_u)
        ]
        precomputed["attribute_unmediated"] = attributes_u = [
            attribute.replace(target_m, target_u)
            for attribute, target_m, target_u in zip(attributes_m, targets_m, targets_u)
        ]

    for (
        key_string,
        key_substring,
        strings,
        substrings,
        occurrence,
        target_ids,
        comparator_ids,
    ) in (
        (
            "context_unmediated",
            "attribute_unmediated",
            contexts_u,
            attributes_u,
            0,
            None,
            None,
        ),
        (
            "prompt_in_context",
            "entity",
            prompts_in_context,
            entities,
            1,
            targets_m_ids,
            targets_u_ids,
        ),
        (
            "prompt",
            "entity",
            prompts,
            entities,
            0,
            targets_u_ids,
            targets_m_ids,
        ),
    ):
        if strings is None or substrings is None:
            continue

        with models.set_padding_side(mt, padding_side="left"):
            inputs, offsets_mapping = inputs_from_batch(mt, strings, device=device)

        trs_all = token_ranges_from_batch(
            strings, substrings, offsets_mapping, occurrence=occurrence
        )
        trs_last = last_token_ranges_from_batch(trs_all)

        # NOTE(evandez): This is an odd place to do this, but since we've already done
        # the work of processing the prompt and prompt in context, we may as well
        # record the target probabilities.

        if target_ids is not None or comparator_ids is not None:
            hiddens_by_layer, outputs = hiddens_from_batch(
                mt, inputs, layers=layers, device=device, stop=False
            )
            log_probs = torch.log_softmax(outputs.logits[:, -1], dim=-1)
            batch_idx = torch.arange(len(strings))
            if target_ids is not None:
                precomputed[f"{key_string}.target.logp"] = log_probs[
                    batch_idx, target_ids
                ]
            if comparator_ids is not None:
                precomputed[f"{key_string}.comparator.logp"] = log_probs[
                    batch_idx, comparator_ids
                ]
        else:
            hiddens_by_layer = hiddens_from_batch(
                mt, inputs, layers=layers, device=device
            )

        for layer, hiddens in hiddens_by_layer.items():
            key = f"{key_string}.{key_substring}.hiddens.{layer}"
            precomputed[f"{key}.average"] = average_hiddens_from_batch(hiddens, trs_all)
            precomputed[f"{key}.last"] = average_hiddens_from_batch(hiddens, trs_last)

    if fp32:
        precomputed = _as_fp32(precomputed)

    return precomputed


def classification_inputs_from_dataset(
    mt: models.ModelAndTokenizer,
    dataset: Dataset,
    layers: Optional[Sequence[int]] = None,
    device: Optional[Device] = None,
    batch_size: int = 64,
    desc: str | None = "precompute classification inputs",
    **kwargs: Any,
) -> Dataset:
    """Precompute classification inputs for the whole dataset."""
    return dataset.map(
        partial(
            classification_inputs_from_batch,
            mt,
            layers=layers,
            device=device,
            fp32=True,
            **kwargs,
        ),
        batched=True,
        batch_size=batch_size,
        desc=desc,
        keep_in_memory=True,
        num_proc=1,
    )


def has_classification_inputs(batch: dict) -> bool:
    """Determine if batch already has precomputed classification inputs."""
    return "context_unmediated" in batch


PromptKey = Literal["prompt", "prompt_in_context"]
TargetKey = Literal["target_mediated", "target_unmediated"]


@torch.inference_mode()
def model_predictions_from_batch(
    mt: models.ModelAndTokenizer,
    batch: dict,
    device: Device | None = None,
    return_top_k: int = 5,
    input_prompt_key: PromptKey = "prompt",
    input_target_key: TargetKey | None = "target_unmediated",
    input_comparator_key: TargetKey | None = "target_mediated",
    other_targets: StrSequence | None = None,
    other_targets_idx: Sequence[int] | None = None,
    output_correct_key: str = "model_correct",
    output_other_targets_key: str = "other_targets",
    output_top_tokens_key: str = "top_tokens",
) -> dict:
    """Precompute model predictions on prompt from the batch."""
    if other_targets is not None and other_targets_idx is None:
        other_targets_idx = first_token_ids_from_batch(mt, other_targets).tolist()

    prompts = batch[input_prompt_key]
    with models.set_padding_side(mt, padding_side="left"):
        inputs, _ = inputs_from_batch(mt, prompts, device=device)
    outputs = mt.model(**inputs)
    distribution = torch.log_softmax(outputs.logits[:, -1], dim=-1)

    precomputed = {}

    top_tokens_idx = distribution.topk(k=return_top_k, dim=-1).indices.tolist()
    top_tokens = tokenizer_utils.batch_convert_ids_to_tokens(
        top_tokens_idx, mt.tokenizer
    )
    precomputed[f"{input_prompt_key}.{output_top_tokens_key}"] = top_tokens

    batch_idx = torch.arange(len(prompts))
    if input_target_key:
        targets = batch[input_target_key]
        targets_token_idx = first_token_ids_from_batch(mt, targets)
        targets_log_p = distribution[batch_idx, targets_token_idx]
        precomputed[
            f"{input_prompt_key}.{input_target_key}.logp"
        ] = targets_log_p.tolist()

    if input_comparator_key is not None:
        comparators = batch[input_comparator_key]
        comparators_token_idx = first_token_ids_from_batch(mt, comparators)
        comparators_log_p = distribution[batch_idx, comparators_token_idx]
        precomputed[
            f"{input_prompt_key}.{input_comparator_key}.logp"
        ] = comparators_log_p.tolist()

    if other_targets_idx is not None:
        key = f"{input_prompt_key}.{output_other_targets_key}.logp"
        precomputed[key] = distribution[:, other_targets_idx].tolist()

    if input_target_key is not None and input_comparator_key is not None:
        precomputed[f"{input_prompt_key}.{output_correct_key}"] = targets_log_p.gt(
            comparators_log_p
        ).tolist()

    return precomputed


def model_predictions_from_dataset(
    mt: models.ModelAndTokenizer,
    dataset: Dataset,
    device: Optional[Device] = None,
    batch_size: int = 64,
    desc: str | None = "precompute model predictions",
    other_targets: StrSequence | None = None,
    other_targets_idx: Sequence[int] | None = None,
    **kwargs: Any,
) -> Dataset:
    """Precompute model predictions for the whole dataset."""
    if other_targets is not None and other_targets_idx is None:
        other_targets_idx = first_token_ids_from_batch(mt, other_targets).tolist()
    return dataset.map(
        partial(
            model_predictions_from_batch,
            mt,
            device=device,
            other_targets_idx=other_targets_idx,
            **kwargs,
        ),
        batched=True,
        batch_size=batch_size,
        desc=desc,
        keep_in_memory=True,
        num_proc=1,
    )


def add_preprocessing_args(parser: argparse.ArgumentParser) -> None:
    """Add common preprocessing args.

    The args include:
        --attribute-no-entity: When computing attribute rep, do not use entity.

    """
    parser.add_argument(
        "--attribute-no-entity",
        action="store_true",
        default=False,
        help="set context = attribute",
    )


def from_args(args: argparse.Namespace, dataset: Dataset) -> Dataset:
    """Apply all the preprocessing steps determined by the args."""
    if args.attribute_no_entity:
        dataset = dataset.map(
            lambda e: {"context": e["attribute"]}, desc="set context=attribute"
        )
    return dataset
