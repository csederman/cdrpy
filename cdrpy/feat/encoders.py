"""
Feature encoders.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import typing as t

import tensorflow as tf

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path

from cdrpy.types import PathLike

# from cdrpy.util.decorators import unstable


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
    def get(self, id_: t.Any) -> t.Any:
        pass

    @abstractmethod
    def encode(self, ids: t.Iterable[t.Any]) -> t.Iterable[t.Any]:
        pass

    @abstractmethod
    def encode_tf(self, ids: t.Iterable[t.Any]) -> tf.data.Dataset:
        pass

    def to_pickle(self, file_path: PathLike | Path) -> None:
        """Save encoder data as pickle object."""
        with open(file_path, "wb") as fh:
            pickle.dump(self.data, fh)

    @classmethod
    def from_pickle(
        cls, file_path: PathLike | Path, name: str | None = None
    ) -> Encoder:
        """Load encoder from pickle file."""
        with open(file_path, "rb") as fh:
            data = pickle.load(fh)
        return cls(data, name=name)

    def get_config(self) -> dict[str, t.Any]:
        """Get encoder configuration."""
        return {"name": self.name, "class": self.__class__.__name__}


class DictEncoder(Encoder[dict]):
    """Encoder for data stored as dictionaries.

    FIXME: Add validation of shape the first time it is accessed and convert
        into a cached property
    """

    @property
    def size(self) -> int:
        return len(self.data)

    @property
    def dtype(self) -> t.Any:
        """Try and return the datatype."""
        # FIXME: I should add the option to pass these in during the init
        #   and only infer them if they are not set.
        first_key = list(self.data)[0]
        first_val = self.data[first_key]
        if hasattr(first_val, "dtype"):
            return first_val.dtype
        else:
            raise AttributeError(f"{type(first_val)} has no dtype attribute.")

    @property
    def shape(self) -> tuple[int, ...]:
        """Try and return the shape of the values."""
        first_key = list(self.data)[0]
        first_val = self.data[first_key]
        # assert all(isinstance(val, type(first_val)) for val in values)
        assert hasattr(first_val, "shape")
        assert all(val.shape == first_val.shape for val in self.data.values())
        return first_val.shape

    def get(self, id_: t.Any) -> t.Any:
        """Gets a single encoding."""
        return self.data[id_]

    def encode(self, ids: t.Iterable[t.Any]) -> list[t.Any]:
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
    def dtype(self) -> t.Any:
        return self.data.values.dtype

    @property
    def size(self) -> int:
        return self.data.shape[0]

    @property
    def shape(self) -> tuple[int]:
        return self.data.shape[1:]

    def get(self, id_: t.Any) -> np.ndarray:
        """Gets a single encoding."""
        return self.data.loc[id_].values

    def encode(self, ids: t.Iterable[t.Any]) -> list[np.ndarray]:
        """Returns a dataframe of encoded values."""
        return list(self.data.loc[ids].values)

    def encode_tf(self, ids: t.Iterable[t.Any]) -> tf.data.Dataset:
        """Returns features as a `tf.data.Dataset` object."""
        arr = self.encode(ids)
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

    def to_pickle(self, file_path: PathLike | Path, **kwargs) -> None:
        """Save encoder data as pickle object."""
        self.data.to_pickle(file_path, **kwargs)

    @classmethod
    def from_pickle(
        cls, file_path: PathLike | Path, name: str | None = None, **kwargs
    ) -> None:
        """Load encoder from pickle file."""
        df = pd.read_pickle(file_path, **kwargs)
        return cls(df, name=name)


class RepeatEncoder(Encoder[t.Any]):
    """Convenience class for encoding repeated values."""

    @property
    def dtype(self) -> t.Any:
        """Try and return the datatype."""
        # FIXME: I should add the option to pass these in during the init
        #   and only infer them if they are not set.
        if hasattr(self.data, "dtype"):
            return self.data.dtype
        else:
            raise AttributeError(f"{type(self.data)} has no dtype attribute.")

    @property
    def shape(self) -> tuple[int, ...]:
        if hasattr(self.data, "shape"):
            return self.data.shape
        else:
            raise AttributeError(f"{type(self.data)} has no shape attribute.")

    @property
    def size(self) -> int:
        return 1

    def get(self, id_: t.Any) -> t.Any:
        return self.data

    def encode(self, ids: t.Iterable[t.Any]) -> list[t.Any]:
        return [self.data for _ in range(len(ids))]

    def encode_tf(self, ids: t.Iterable[t.Any]) -> tf.data.Dataset:
        """Return a `tf.data.RepeatDataset object`."""
        return tf.data.Dataset.from_tensor_slices(
            [self.data], name=self.name
        ).repeat(len(ids))
