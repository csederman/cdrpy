"""
Feature encoders.

FIXME: consider implementing these lookups as a keras embedding layer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import typing as t

import tensorflow as tf

from collections.abc import Mapping


K = t.TypeVar("K")
V = t.TypeVar("V")

ValidatorFunc = t.Callable[[Mapping[K, V]], Mapping[K, V]]


def validator(func: ValidatorFunc) -> ValidatorFunc:
    """"""
    if isinstance(func, property):
        func.fget.validator = True
    else:
        func.validator = True
    return func


class _ValidatorRegisteringType(type):
    """Metaclass for classes with validator registries."""

    def __init__(cls, name, bases, attrs) -> None:
        tagged_funcs = set()
        for name, method in attrs.items():
            if isinstance(method, property):
                func = method.fget
            else:
                func = method
            if hasattr(func, "validator"):
                tagged_funcs.add(name)

        @property
        def _validators(self) -> set[ValidatorFunc]:
            funcs = tagged_funcs.copy()
            try:
                funcs.update(super(cls, self)._validators)
            except AttributeError:
                pass
            return funcs

        cls._validators = _validators


class FeatureLookupBase(t.Generic[K, V], metaclass=_ValidatorRegisteringType):
    """Base class for dictionary lookup-based feature encoders.

    FIXME: type hinting for mapping since the types can change when we apply
    FIXME: add check that values are arrays
    FIXME: subclass dict instead of having a .mapping attribute

    Parameters
    ----------
        mapping: the mapping of key-value pairs for the lookup table.
        name: optional name of the feature encoder.
    """

    def __init__(
        self, mapping: Mapping[K, V], name: str | None = None
    ) -> None:
        self.mapping = self.validate(mapping)
        self.name = name

    def __repr__(self) -> str:
        return "{}(name={}, n={})".format(
            self.__class__.__name__,
            self.name,
            self.n_keys,
        )

    @property
    def n_keys(self) -> int:
        return len(self.mapping)

    def validate(self, mapping: Mapping[K, V]) -> None:
        """Validate mapping properties."""
        for method in self._validators:
            func = getattr(self, method)
            mapping = func(mapping)
        return mapping

    def map(self, func: t.Callable[[V], V]) -> FeatureLookupBase:
        """Apply a transformation to mapping values."""
        self.mapping = {k: func(v) for (k, v) in self.mapping.items()}
        return self

    def encode(self, keys: t.Iterable[K]) -> list[V]:
        """Encode features for the specified keys."""
        return [self.mapping[k] for k in keys]

    def encode_tf(self, keys: t.Iterable[K]) -> tf.data.Dataset:
        """Encode features as a `tf.data.Dataset` object."""
        return tf.data.Dataset.from_tensor_slices(self.encode(keys))


class ArrayFeatureLookup(FeatureLookupBase):
    """Lookup for array-based features."""

    def __repr__(self) -> str:
        return "{}(name={}, n={}, shape={})".format(
            self.__class__.__name__,
            self.name,
            self.n_keys,
            self.shape,
        )

    @validator
    def _check_dims(self, mapping: Mapping[K, V]) -> Mapping[K, V]:
        """Ensures that all arrays have the same dimension."""
        shapes = [arr.shape for arr in mapping.values()]
        assert len(set(shapes)) == 1
        return mapping

    @property
    def shape(self) -> tuple[int]:
        first_arr = list(self.mapping.values())[0]
        return first_arr.shape


class Array1DFeatureLookup(ArrayFeatureLookup):
    """Lookup for 1d array-based features."""

    @validator
    def _check_1d(self, mapping: Mapping[K, V]) -> Mapping[K, V]:
        """Ensures that all arrays are 1D."""
        assert all(arr.ndim == 1 for arr in mapping.values())
        return mapping

    @classmethod
    def from_pandas(
        cls, df: pd.DataFrame, *args, **kwargs
    ) -> Array1DFeatureLookup:
        """Create a dict encoder from a `pandas.DataFrame`."""
        return cls(dict(zip(df.index, df.values)), *args, **kwargs)


class RepeatEncoder:
    """"""

    def __init__(self, value: np.ndarray, name: str | None = None):
        self.value = value
        self.name = name

    @property
    def shape(self) -> tuple[int, ...]:
        return self.value.shape

    def encode(self, keys: t.Iterable[t.Any]) -> list[t.Any]:
        return [self.value for _ in range(len(keys))]

    def encode_tf(self, keys: t.Iterable[t.Any]) -> tf.data.Dataset:
        """"""
        n_keys = len(keys)
        tf_ds = tf.data.Dataset.from_tensor_slices([self.value])
        return tf_ds.repeat(n_keys)
