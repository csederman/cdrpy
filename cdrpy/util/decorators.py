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
