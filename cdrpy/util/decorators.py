"""
Function decorators.
"""

from __future__ import annotations

import functools
import warnings

import typing as t

R = t.TypeVar("R")
F = t.Callable[..., R]


def unstable(*warning_args, **warning_kwargs):
    """Warn users about potential unresolved issues with a function."""

    def decorator(func: F) -> F:
        def wrapper(*args, **kwargs) -> R:
            warnings.warn(*warning_args, **warning_kwargs)
            return func(*args, **kwargs)

        return t.cast(wrapper, functools.update_wrapper(wrapper, func))

    return decorator


class CDRPYException(Exception):
    """Base class for catching all cdrpy exceptions."""


class MissingEncoderException(CDRPYException):
    def __init__(self, msg: str) -> None:
        self.msg = msg


def check_encoders(func: F) -> F:
    """Require dataset encoders."""

    def wrapper(self, *args, **kwargs) -> R:
        if not self.cell_encoders:
            raise MissingEncoderException(
                "{} instance does not have any registered cell encoders".format(
                    self.__class__.name
                )
            )
        if not self.drug_encoders:
            raise MissingEncoderException(
                "{} instance does not have any registered drug encoders".format(
                    self.__class__.name
                )
            )
        return func(self, *args, **kwargs)

    return t.cast(wrapper, functools.update_wrapper(wrapper, func))
