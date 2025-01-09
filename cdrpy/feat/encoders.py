"""
Feature encoders.
"""

from __future__ import annotations

import h5py
import copy

import numpy as np
import pandas as pd
import typing as t

import tensorflow as tf

from abc import ABC, abstractmethod
from pathlib import Path

from cdrpy.types import PathLike
from cdrpy.util import io

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
    def size(self) -> int: ...

    @abstractmethod
    def get(self, id_: t.Any) -> t.Any: ...

    @abstractmethod
    def keys(self) -> t.List[t.Any] | None: ...

    # @abstractmethod
    # def subset(self, ids: t.Iterable[t.Any]) -> Encoder[D]:
    #     ...

    @abstractmethod
    def encode(self, ids: t.Iterable[t.Any]) -> t.Iterable[t.Any]: ...

    @abstractmethod
    def merge(self, other: Encoder) -> Encoder: ...

    @abstractmethod
    def copy(self) -> Encoder: ...

    # @abstractmethod
    # def encode_tf(self, ids: t.Iterable[t.Any]) -> tf.data.Dataset:
    #     ...

    @abstractmethod
    def tf_signature(self) -> t.Any: ...

    # @abstractmethod
    # def save(self, file_or_group: h5py.File | h5py.Group, key: str) -> None:
    #     ...

    # @classmethod
    # @abstractmethod
    # def load(cls, file_or_group: h5py.File | h5py.Group, key: str) -> Encoder:
    #     ...


class DictEncoder(Encoder[dict]):
    """Encoder for data stored as dictionaries.

    FIXME: Add validation of shape the first time it is accessed and convert
        into a cached property
    """

    @property
    def size(self) -> int:
        """Returns the number of data points in the encoder."""
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
    def shape(self) -> t.Tuple[int, ...]:
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

    def keys(self) -> t.List[t.Any]:
        return list(self.data.keys())

    def encode(self, ids: t.Iterable[t.Any]) -> t.List[t.Any]:
        """Encode features for the specified IDs."""
        return [self.data[id_] for id_ in ids]

    def tf_signature(self) -> t.Any:
        return tf.TensorSpec(
            self.shape, dtype=tf.dtypes.as_dtype(self.dtype), name=self.name
        )

    def merge(self, other: DictEncoder, **kwargs) -> DictEncoder:
        """Returns a new DictEncoder with a union of keys."""
        assert isinstance(other, DictEncoder)
        assert self.shape == other.shape
        assert self.dtype == other.dtype

        # NOTE: we prioritize data already in self
        # NOTE: we don't check that data with the same key is identical
        data = {**other.data, **self.data}
        return DictEncoder(data, **kwargs)

    def copy(self) -> DictEncoder:
        return DictEncoder(copy.deepcopy(self.data), name=self.name)


class PandasEncoder(Encoder[pd.DataFrame]):
    """Encoder for data stored as `pd.DataFrame` objects."""

    @property
    def dtype(self) -> t.Any:
        return self.data.values.dtype

    @property
    def size(self) -> int:
        return self.data.shape[0]

    @property
    def shape(self) -> t.Tuple[int]:
        return self.data.shape[1:]

    def get(self, id_: t.Any) -> np.ndarray:
        """Gets a single encoding."""
        return self.data.loc[id_].values

    def keys(self) -> t.List[t.Any]:
        """Returns a list of lookup keys"""
        return self.data.index.to_list()

    def encode(self, ids: t.Iterable[t.Any]) -> t.List[np.ndarray]:
        """Returns a dataframe of encoded values."""
        return list(self.data.loc[ids].values)

    def tf_signature(self) -> t.Any:
        return tf.TensorSpec(
            self.shape, dtype=tf.dtypes.as_dtype(self.dtype), name=self.name
        )

    def merge(self, other: PandasEncoder, **kwargs) -> PandasEncoder:
        """Returns a new PandasEncoder with a union of keys."""
        assert isinstance(other, PandasEncoder)
        assert self.shape == other.shape
        assert self.dtype == other.dtype

        # NOTE: we prioritize data already in self
        # NOTE: we don't check that data with the same key is identical
        other_data = other.data[~other.data.index.isin(self.data.index)]
        data = pd.concat([self.data, other_data])
        return PandasEncoder(data, **kwargs)

    def copy(self) -> PandasEncoder:
        """Creates a copy of the encoder and it's data."""
        return PandasEncoder(self.data.copy(deep=True), name=self.name)

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

    def save(self, file_or_group: h5py.File | h5py.Group, key: str) -> None:
        """Saves the encoder in hdf5 format."""
        group = file_or_group.create_group(key)
        io.pandas_to_h5(group, self.data, index=True)
        if self.name is not None:
            group.attrs["name"] = self.name

    @classmethod
    def load(cls, file_or_group: h5py.File | h5py.Group, key: str) -> PandasEncoder:
        """Loads a PandasEncoder from an hdf5 file."""
        group = file_or_group[key]
        data = io.pandas_from_h5(group)
        return cls(data, name=group.attrs.get("name"))


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
    def shape(self) -> t.Tuple[int, ...]:
        if hasattr(self.data, "shape"):
            return self.data.shape
        else:
            raise AttributeError(f"{type(self.data)} has no shape attribute.")

    @property
    def size(self) -> int:
        return 1

    def get(self, id_: t.Any) -> t.Any:
        return self.data

    def keys(self) -> None:
        return None

    def encode(self, ids: t.Iterable[t.Any]) -> t.List[t.Any]:
        return [self.data for _ in range(len(ids))]

    def tf_signature(self) -> t.Any:
        return tf.TensorSpec(
            self.shape, dtype=tf.dtypes.as_dtype(self.dtype), name=self.name
        )

    def merge(self, other: RepeatEncoder, **kwargs) -> RepeatEncoder:
        """Returns a new RepeatEncoder with a union of keys."""
        assert isinstance(other, RepeatEncoder)
        assert self.shape == other.shape
        assert self.dtype == other.dtype
        # NOTE: we prioritize data already in self
        # NOTE: we don't check that data with the same key is identical
        return RepeatEncoder(self.data, **kwargs)

    def copy(self) -> RepeatEncoder:
        """Creates a copy of the enoder and it's data."""
        return RepeatEncoder(copy.deepcopy(self.data), name=self.name)


EncoderMapper = {
    PandasEncoder.__name__: PandasEncoder,
}
