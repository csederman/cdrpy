"""
Feature encoders.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import typing as t

import tensorflow as tf

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path

from cdrpy.types import PathLike


D = t.TypeVar("D")


class Encoder(ABC, t.Generic[D]):
    """Interface for encoders."""

    def __init__(self, data: D, name: str | None = None) -> None:
        self.data = data
        self.name = name

    def __repr__(self) -> str:
        return "{}(name={}, n={})".format(
            self.__class__.__name__,
            repr(self.name),
            self.size,
        )

    @property
    @abstractmethod
    def size(self) -> int:
        pass

    @abstractmethod
    def encode(self, ids: t.Iterable[t.Any]) -> t.Iterable[t.Any]:
        pass

    @abstractmethod
    def encode_tf(self, ids: t.Iterable[t.Any]) -> tf.data.Dataset:
        pass


class DictEncoder(Encoder[dict]):
    """Encoder for data stored as dictionaries."""

    @property
    def size(self) -> int:
        return len(self.data)

    def encode(self, ids: t.Iterable[t.Any]) -> t.Iterable[t.Any]:
        """Encode features for the specified IDs."""
        return [self.data[k] for k in ids]

    def encode_tf(self, ids: t.Iterable[t.Any]) -> tf.data.Dataset:
        """Encode features as a `tf.data.Dataset` object."""
        return tf.data.Dataset.from_tensor_slices(
            self.encode(ids), name=self.name
        )


class PandasEncoder(Encoder[pd.DataFrame]):
    """Encoder for data stored as `pd.DataFrame` objects."""

    @property
    def size(self) -> int:
        return self.data.shape[0]

    def encode(self, ids: t.Iterable[t.Any]) -> pd.DataFrame:
        """Returns a dataframe of encoded values."""
        return self.data.loc[ids]

    def encode_tf(self, ids: t.Iterable[t.Any]) -> tf.data.Dataset:
        """Returns features as a `tf.data.Dataset` object."""
        arr = self.encode(ids).to_numpy()
        return tf.data.Dataset.from_tensor_slices(arr, name=self.name)

    @classmethod
    def from_csv(
        cls,
        file_path: PathLike | Path,
        index_col: int = 0,
        name: str | None = None,
        **kwargs,
    ) -> PandasEncoder:
        """Generates a `PandasEncoder` from a CSV file."""
        df = pd.read_csv(file_path, index_col=index_col, **kwargs)
        return cls(df, name=name)


class RepeatEncoder(Encoder[t.Any]):
    """Convenience class for encoding repeated values."""

    @property
    def size(self) -> int:
        return 1

    def encode(self, ids: t.Iterable[t.Any]) -> t.Any:
        return [self.data for _ in range(len(ids))]

    def encode_tf(self, ids: t.Iterable[t.Any]) -> tf.data.Dataset:
        """Return a `tf.data.RepeatDataset object`."""
        return tf.data.Dataset.from_tensor_slices(
            [self.data], name=self.name
        ).repeat(len(ids))
