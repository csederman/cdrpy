"""
Author: Casey Sederman
Created: Tue Sep 26 09:44:15 2023
"""

from __future__ import annotations

import typing as t
import numpy as np

from abc import ABC, abstractmethod

if t.TYPE_CHECKING:
    from cdrpy.feat.encoders import Encoder


class Generator(ABC):
    """Generator for creating keras sequences as inputs for modeling."""

    def _encode_features(
        self, cell_ids: t.Iterable[t.Any], drug_ids: t.Iterable[t.Any]
    ) -> t.Any:
        """Encode the features"""

        def encode(e: Encoder, ids: t.Iterable[t.Any]) -> np.ndarray:
            return np.asanyarray(e.encode(ids))

        cell_feats = [encode(e, cell_ids) for e in self.dataset.cell_encoders]
        drug_feats = [encode(e, drug_ids) for e in self.dataset.drug_encoders]
        return cell_feats + drug_feats

    @abstractmethod
    def num_batch_dims(self):
        """
        Returns the number of batch dimensions in returned tensors (_not_ the batch size itself).

        For instance, for full batch methods like GCN, the feature has shape ``1 x number of nodes x
        feature size``, where the 1 is a "dummy" batch dimension and ``number of nodes`` is the real
        batch size (every node in the graph).
        """
        ...

    @abstractmethod
    def flow(self, *args, **kwargs):
        """
        Creates a Keras Sequence or similar input appropriate for CDRP models.
        """
        ...
