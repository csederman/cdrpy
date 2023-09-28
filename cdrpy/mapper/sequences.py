"""
Author: Casey Sederman
Created: Tue Sep 26 09:44:12 2023
"""

from __future__ import annotations

import warnings
import numpy as np
import typing as t

from keras.utils import Sequence
from collections import defaultdict

from ..core.utils import is_real_iterable
from ..core.random import random_state


__all__ = [
    "ResponseSequence",
]


class ResponseSequence(Sequence):
    """Keras-compatible data generator to use with the Keras
    methods :meth:`keras.Model.fit`, :meth:`keras.Model.evaluate`,
    and :meth:`keras.Model.predict`.

    Parameters
    ----------
        sample_function:
        batch_size:
        cell_ids:
        drug_ids:
        targets:
        shuffle:
        seed:
    """

    def __init__(
        self,
        encode_function: t.Callable,
        batch_size: int,
        cell_ids: t.Iterable[t.Any],
        drug_ids: t.Iterable[t.Any],
        targets: t.Iterable[t.Any] | None = None,
        sample_weights: t.Iterable[t.Any] | None = None,
        shuffle: bool = True,
        seed: t.Any = None,
    ) -> None:
        # check that cell_ids is an iterable
        if not is_real_iterable(cell_ids):
            raise TypeError(
                "cell_ids must be an iterable or a numpy array if cell ids."
            )

        # check that drug_ids is an iterable
        if not is_real_iterable(drug_ids):
            raise TypeError(
                "drug_ids must be an iterable or a numpy array if drug ids."
            )

        if len(cell_ids) != len(drug_ids):
            raise ValueError(
                "The length of cell_ids must be the same as the length of "
                "drug_ids"
            )

        if targets is not None:
            if not is_real_iterable(targets):
                raise TypeError(
                    "Targets must be None or an iterable or a numpy array."
                )
            if len(cell_ids) != len(targets):
                raise ValueError(
                    (
                        "The length of targets must be the same as the length "
                        "of cell_ids and the length of drug_ids"
                    )
                )
            self.targets = np.asanyarray(targets)
        else:
            self.targets = None

        if sample_weights is not None:
            if not is_real_iterable(sample_weights):
                raise TypeError(
                    "sample_weights must be None or an iterable or a numpy "
                    "array."
                )
            if len(cell_ids) != len(sample_weights):
                raise ValueError(
                    (
                        "The length of sample_weights must be the same as the "
                        "length of cell_ids and the length of drug_ids"
                    )
                )
            self.sample_weights = np.asanyarray(sample_weights)
        else:
            self.sample_weights = None

        if isinstance(encode_function, t.Callable):
            self._encode_function = encode_function
        else:
            raise TypeError(
                "({}) The sampling function expects a callable function.".format(
                    type(self).__name__
                )
            )

        self.cell_ids = np.asanyarray(cell_ids)
        self.drug_ids = np.asanyarray(drug_ids)
        self.data_size = len(cell_ids)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._rs, _ = random_state(seed)

        # initial shuffle
        self.on_epoch_end()

    def __len__(self):
        """Number of batches per epoch"""
        return int(np.ceil(self.data_size / self.batch_size))

    def __getitem__(self, batch_num: int):
        start_idx = self.batch_size * batch_num
        end_idx = start_idx + self.batch_size

        if start_idx >= self.data_size:
            raise IndexError("Mapper: batch_num larger than length of data")

        batch_indices = self.indices[start_idx:end_idx]
        batch_cell_ids = self.cell_ids[batch_indices]
        batch_drug_ids = self.drug_ids[batch_indices]

        batch_targets = (
            None if self.targets is None else self.targets[batch_indices]
        )

        batch_sample_weights = (
            None
            if self.sample_weights is None
            else self.sample_weights[batch_indices]
        )

        batch_feats = self._encode_function(batch_cell_ids, batch_drug_ids)
        return batch_feats, batch_targets, batch_sample_weights

    def on_epoch_end(self):
        """
        Shuffle all head (root) nodes at the end of each epoch
        """
        self.indices = list(range(self.data_size))
        if self.shuffle:
            self._rs.shuffle(self.indices)
