"""
Utilities for model runners.
"""

from __future__ import annotations

import functools
import logging
import typing as t


log = logging.getLogger(__name__)

R = t.TypeVar("R")
F = t.Callable[..., R]


def pipeline_step(msg: str):
    def decorator(func: F) -> F:
        def wrapper(*args, **kwargs) -> R:
            log.info(msg)
            return func(*args, **kwargs)

        return t.cast(wrapper, functools.update_wrapper(wrapper, func))

    return decorator
