"""
Author: Casey Sederman
Created: Tue Sep 26 09:49:11 2023
"""

from __future__ import annotations

import warnings
import numpy as np
import typing as t

from keras.utils import Sequence


from .base import Generator
from .sequences import ResponseSequence

if t.TYPE_CHECKING:
    from ..data.datasets import Dataset


__all__ = [
    "BatchedResponseGenerator",
]


class BatchedResponseGenerator(Generator):
    """"""

    def __init__(self, D: Dataset, batch_size: int) -> None:
        self.dataset = D
        self.batch_size = batch_size

    def num_batch_dims(self) -> int:
        return 1

    def flow(
        self,
        cell_ids: t.Iterable[t.Any],
        drug_ids: t.Iterable[t.Any],
        targets: t.Iterable[t.Any] | None = None,
        sample_weights: t.Iterable[t.Any] | None = None,
        shuffle: bool = False,
        seed: t.Any = None,
    ) -> Sequence:
        """"""
        # invalid_cells = self.dataset.cell_ids
        return ResponseSequence(
            self._encode_features,
            self.batch_size,
            cell_ids,
            drug_ids,
            targets=targets,
            sample_weights=sample_weights,
            shuffle=shuffle,
            seed=seed,
        )
