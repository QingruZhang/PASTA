"""Heuristics and tools for cleaning language."""
import logging
from functools import cache

from nltk.corpus import cmudict

logger = logging.getLogger(__name__)


@cache
def _cmudict() -> dict[str, list]:
    """Read the nltk cmudict corpus."""
    return cmudict.dict()


def determine_article(word: str, default: str = "a") -> str:
    """Determine article that should precede text."""
    pronounciation = _cmudict().get(word)
    if not pronounciation:
        logger.debug(f"no pronounciation found for: {word}")
        return default
    first = pronounciation[0]
    while not isinstance(first, str) and first:
        first = first[0]
    is_vowel_sound = first[-1].isdigit()
    return "an" if is_vowel_sound else "a"
