"""Utils for interacting with huggingface tokenizers."""
from contextlib import contextmanager
from typing import Any, Iterator, Optional, Sequence, Tuple

from evaluation.utils.typing import StrSequence, Tokenizer, TokenizerOffsetMapping


def find_token_range(
    string: str,
    substring: str,
    tokenizer: Optional[Tokenizer] = None,
    occurrence: int = 0,
    offset_mapping: Optional[TokenizerOffsetMapping] = None,
    **kwargs: Any,
) -> Tuple[int, int]:
    """Find index range of tokenized string containing tokens for substring.

    The kwargs are forwarded to the tokenizer.

    A simple example:

        string = 'The batman is the night.'
        substring = 'batman'
        tokenizer = ...

        # Example tokenization: ['the', 'bat', '##man', 'is', 'the', 'night']
        assert find_token_range(string, substring, tokenizer) == (1, 3)

    Args:
        string: The string.
        substring: The substring to find token range for.
        tokenizer: The tokenizer. If not set, offset_mapping must be.
        occurrence: The occurence of the substring to look for.
            Zero indexed. Defaults to 0, the first occurrence.
        offset_mapping: Precomputed offset mapping. If not set, tokenizer will be run.

    Raises:
        ValueError: If substring is not actually in string or if banned
            kwargs are specified.

    Returns:
        Tuple[int, int]: The start (inclusive) and end (exclusive) token idx.
    """
    if tokenizer is None and offset_mapping is None:
        raise ValueError("must set either tokenizer= or offset_mapping=")
    if "return_offsets_mapping" in kwargs:
        raise ValueError("cannot set return_offsets_mapping")
    if substring not in string:
        raise ValueError(f'"{substring}" not found in "{string}"')
    char_start = string.index(substring)
    for _ in range(occurrence):
        try:
            char_start = string.index(substring, char_start + 1)
        except ValueError as error:
            raise ValueError(
                f"could not find {occurrence + 1} occurrences "
                f'of "{substring} in "{string}"'
            ) from error
    char_end = char_start + len(substring)

    if offset_mapping is None:
        assert tokenizer is not None
        tokens = tokenizer(string, return_offsets_mapping=True, **kwargs)
        offset_mapping = tokens.offset_mapping

    token_start, token_end = None, None
    for index, (token_char_start, token_char_end) in enumerate(offset_mapping):
        if token_start is None:
            if token_char_start <= char_start and token_char_end >= char_start:
                token_start = index
        if token_end is None:
            if token_char_start <= char_end and token_char_end >= char_end:
                token_end = index
                break

    assert token_start is not None
    assert token_end is not None
    assert token_start <= token_end
    return (token_start, token_end + 1)


def batch_convert_ids_to_tokens(
    batch: Sequence[Sequence[int]], tokenizer: Tokenizer, **kwargs: Any
) -> Sequence[StrSequence]:
    """Call `convert_ids_to_tokens` on every sequence in the batch."""
    return [tokenizer.convert_ids_to_tokens(ids, **kwargs) for ids in batch]


@contextmanager
def set_padding_side(
    tokenizer: Tokenizer, padding_side: str = "right"
) -> Iterator[None]:
    """Temporarily set padding side for tokenizer.

    Useful for when you want to batch generate with causal LMs like GPT, as these
    require the padding to be on the left side in such settings but are much easier
    to mess around with when the padding, by default, is on the right.

    Example usage:
        tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
        with tokenizer_utils.set_padding_side(tokenizer, "left"):
            inputs = mt.tokenizer(...)
        # ... later
        model.generate(**inputs)

    """
    _padding_side = tokenizer.padding_side
    tokenizer.padding_side = padding_side
    yield
    tokenizer.padding_side = _padding_side
