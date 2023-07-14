"""

"""


from __future__ import annotations

import pickle
import typing as t

from pathlib import Path

from .types import PathLike


K = t.TypeVar("K")
V = t.TypeVar("V")


def read_pickled_list(file_path: PathLike | Path) -> list[t.Any]:
    """"""
    with open(file_path, "rb") as fh:
        lst = pickle.load(fh)
    return lst


def read_pickled_dict(file_path: PathLike | Path) -> dict[t.Any, t.Any]:
    """Load pickled dict object from file."""
    with open(file_path, "rb") as fh:
        dct = pickle.load(fh)
    return dct


def read_list(file_path: PathLike | Path) -> list[str]:
    with open(file_path) as fh:
        lines = fh.read().splitlines()
    return lines
