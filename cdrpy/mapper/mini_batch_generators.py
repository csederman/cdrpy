"""
Author: Casey Sederman
Created: Tue Sep 26 09:49:11 2023
"""

from __future__ import annotations

import warnings
import functools
import numpy as np
import typing as t

from keras.utils import Sequence


from .base import Generator
from .sequences import ResponseSequence

if t.TYPE_CHECKING:
    from ..datasets import Dataset


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
        drugs_first: bool = False,
        shuffle: bool = False,
        seed: t.Any = None,
    ) -> ResponseSequence:
        """"""
        encode_func = functools.partial(self._encode_features, drugs_first=drugs_first)

        return ResponseSequence(
            encode_func,
            self.batch_size,
            cell_ids,
            drug_ids,
            targets=targets,
            sample_weights=sample_weights,
            shuffle=shuffle,
            seed=seed,
        )

    def flow_from_dataset(self, D: Dataset, **kwargs) -> ResponseSequence:
        """"""
        return self.flow(
            cell_ids=D.cell_ids,
            drug_ids=D.drug_ids,
            targets=D.labels,
            **kwargs,
        )
