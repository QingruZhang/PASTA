"""Metrics used for evaluating the editors.

Lots of this code is taken from the ROME CounterFact evaluation, for which the source
code can be found: https://github.com/kmeng01/rome/blob/main/experiments/py/eval_utils_counterfact.py
"""
import logging
from dataclasses import dataclass
from typing import Any, Sequence

from evaluation.utils.typing import ArrayLike, StrSequence

import nltk
import numpy as np
from dataclasses_json import DataClassJsonMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Metric(DataClassJsonMixin):
    """An aggregate metric."""

    mean: float
    std: float
    values: ArrayLike | None = None

    def without_values(self) -> "Metric":
        """Return the metric without the values stored."""
        return Metric(mean=self.mean, std=self.std)

    @staticmethod
    def aggregate(values: ArrayLike, store_values: bool = True) -> "Metric":
        """Aggregate mean/std of the values."""
        return Metric(
            np.mean(values), np.std(values), values=values if store_values else None
        )


@dataclass(frozen=True)
class EfficacyMetrics(DataClassJsonMixin):
    """Efficacy metrics."""

    score: Metric
    magnitude: Metric

    def without_values(self) -> "EfficacyMetrics":
        """Return the metrics without the values stored."""
        return EfficacyMetrics(
            score=self.score.without_values(),
            magnitude=self.magnitude.without_values(),
        )


def efficacy(
    p_targets: Sequence[ArrayLike],
    p_comparators: Sequence[ArrayLike],
    assume_log_probs: bool = True,
    store_values: bool = True,
) -> EfficacyMetrics:
    """Compute efficacy on metrics.

    Efficacy is determined by a score and a magnitude. The score is how often
    p(target) > p(comparator). The magnitude is the average p(target) - p(comparator).

    Inputs are two sequences. Each element should be one or more measurements of
    the probability (for e.g. different prompts). This function will first average
    across those inner lists, then average across the whole list.
    """
    _validate_same_length(p_targets=p_targets, p_comparators=p_comparators)

    scores, magnitudes = [], []
    for i, (p_target, p_comparator) in enumerate(zip(p_targets, p_comparators)):
        _validate_same_length(
            **{f"p_target_{i}": p_target, f"p_comparator_{i}": p_comparator}
        )

        if assume_log_probs:
            p_target = np.exp(p_target)
            p_comparator = np.exp(p_comparator)
        else:
            p_target = np.array(p_target)
            p_comparator = np.array(p_comparator)

        score = np.mean(p_target > p_comparator)
        scores.append(score)

        magnitude = np.mean(p_target - p_comparator)
        magnitudes.append(magnitude)

    return EfficacyMetrics(
        score=Metric.aggregate(scores, store_values=store_values),
        magnitude=Metric.aggregate(magnitudes, store_values=store_values),
    )


def average_tfidf_similarity(
    generations: Sequence[StrSequence],
    references: Sequence[StrSequence],
    tfidf_vectorizer: TfidfVectorizer,
    desc: str | None = None,
) -> Metric:
    """Compute TF-IDF similarity between generated text and reference texts.

    This is used downstream to compute the "consistency" and "essence" scores.
    """
    _validate_same_length(generations=generations, references=references)
    if desc is None:
        desc = "tfidf similarity"
    similarities = [
        tfidf_similarity(gs, rs, tfidf_vectorizer)
        for gs, rs in tqdm(list(zip(generations, references)), desc=desc)
    ]
    return Metric.aggregate(similarities)


def tfidf_similarity(
    source: str | StrSequence,
    reference: str | StrSequence,
    tfidf_vectorizer: TfidfVectorizer,
) -> float:
    """Return TfIdf similarity between the texts."""
    if isinstance(source, str):
        source = [source]
    if isinstance(reference, str):
        reference = [reference]
    sv, rv = tfidf_vectorizer.transform([" ".join(source), " ".join(reference)]).A
    return vector_similarity(sv, rv)


def vector_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) + eps) / (np.linalg.norm(b) + eps)


def average_weighted_n_gram_entropy(
    generations: Sequence[StrSequence], desc: str | None = None, **kwargs: Any
) -> Metric:
    """Compute fluency score.

    Args:
        generations: Any number of generated strings from the model.

    Returns:
        Average n-gram entropy across generations. Can specify multiple ns by
        setting the ns= kwarg. By default, uses 2-grams and 3-grams.

    """
    if desc is None:
        desc = "weighed n-gram entropy"
    entropies = [
        np.mean([weighted_n_gram_entropy(text, **kwargs) for text in texts])
        for texts in tqdm(generations, desc=desc)
    ]
    return Metric.aggregate(entropies)


def weighted_n_gram_entropy(
    texts: str | StrSequence,
    ns: Sequence[int] = (2, 3),
    weights: Sequence[float] = (2 / 3, 4 / 3),
) -> float:
    """Return weighted n-gram entropy for different values of n."""
    _validate_same_length(ns=ns, weights=weights)
    if isinstance(texts, str):
        texts = [texts]
    entropies = []
    for text in texts:
        entropies_by_n = np.array([n_gram_entropy(text, n) for n in ns])
        entropy = np.mean(entropies_by_n * np.array(weights))
        entropies.append(entropy)
    return np.mean(entropies)


def n_gram_entropy(text: str, n: int) -> float:
    """Return entropy of n-gram distribution in text."""
    counts = _n_gram_counts(text, n)
    dist = np.array([count for _, count in counts.items()], dtype=np.float32)
    dist /= dist.sum()
    entropy = np.sum(-dist * np.log(dist) / np.log(2))
    return entropy.item()


def _validate_same_length(**kwargs: Sequence | ArrayLike) -> None:
    """Validate all batch sizes are the same."""
    lengths = {key: len(seq) for key, seq in kwargs.items()}
    if len(set(lengths.values())) > 1:
        message = f"inconsistent batch sizes:" + "\n\t"
        message += "\n\t".join(f"{key}={length}" for key, length in lengths.items())
        raise ValueError(message)


def _n_gram_counts(text: str, n: int) -> dict[tuple[str, ...], int]:
    """Return the n-gram counts for the text."""
    tokens = nltk.word_tokenize(text)
    ngrams = nltk.ngrams(tokens, n)
    return nltk.FreqDist(ngrams)
