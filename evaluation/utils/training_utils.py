"""Utilities for training models."""
from typing import Sequence, Sized, cast

import torch
from torch.utils import data


class EarlyStopping:
    """Observes a numerical value and determines when it has not improved."""

    def __init__(self, patience: int = 4, decreasing: bool = True):
        """Initialize the early stopping tracker.

        Args:
            patience (int, optional): Allow tracked value to not improve over
                its best value this many times. Defaults to 4.
            decreasing (bool, optional): If True, the tracked value "improves"
                if it decreases. If False, it "improves" if it increases.
                Defaults to True.

        """
        self.patience = patience
        self.decreasing = decreasing
        self.best = float("inf") if decreasing else float("-inf")
        self.num_bad = 0

    def __call__(self, value: float) -> bool:
        """Considers the new tracked value and decides whether to stop.

        Args:
            value (float): The new tracked value.

        Returns:
            bool: True if patience has been exceeded.

        """
        improved = self.decreasing and value < self.best
        improved |= not self.decreasing and value > self.best
        if improved:
            self.best = value
            self.num_bad = 0
        else:
            self.num_bad += 1

        return self.num_bad > self.patience

    @property
    def improved(self) -> bool:
        """Check if the running value just improved."""
        return self.num_bad == 0


TrainValSplit = tuple[data.Subset, data.Subset]


def random_split(dataset: data.Dataset, hold_out: float = 0.1) -> TrainValSplit:
    """Randomly split the dataset into a train and val set.

    Args:
        dataset: The full dataset.
        hold_out: Fraction of data to hold out for the val set. Defaults to .1.

    Returns:
        The train and val sets.

    """
    if hold_out <= 0 or hold_out >= 1:
        raise ValueError(f"hold_out must be in (0, 1), got {hold_out}")

    size = len(cast(Sized, dataset))
    val_size = int(hold_out * size)
    train_size = size - val_size

    for name, size in (("train", train_size), ("val", val_size)):
        if size == 0:
            raise ValueError(
                f"hold_out={hold_out} causes {name} set size " "to be zero"
            )

    splits = data.random_split(dataset, (train_size, val_size))
    assert len(splits) == 2
    train, val = splits
    return train, val


def fixed_split(dataset: data.Dataset, indices: Sequence[int]) -> TrainValSplit:
    """Split dataset on the given indices.

    Args:
        dataset: The dataset to split.
        indices: Indices comprising the right split.

    Returns:
        The subset *not* for the indices, followed by the subset *for* the indices.

    """
    size = len(cast(Sized, dataset))
    for index in indices:
        if index < 0 or index >= size:
            raise IndexError(f"dataset index out of bounds: {index}")

    others = sorted(set(range(size)) - set(indices))
    if not others:
        raise ValueError("indices cover entire dataset; nothing to split!")

    return data.Subset(dataset, others), data.Subset(dataset, indices)


def cosine_similarity_float16(
    x1: torch.Tensor, x2: torch.Tensor, dim: int = -1, eps: float = 1e-5
) -> torch.Tensor:
    """Cosine similarity that is slightly more stable for float16 inputs.

    Prenormalizes each vector, which removes risk of overflow in numerator
    when norms are large.
    """
    x1 = x1 / x1.norm(dim=dim, keepdim=True)
    x2 = x2 / x2.norm(dim=dim, keepdim=True)
    return torch.nn.functional.cosine_similarity(x1, x2, dim=dim, eps=eps)
