"""

"""


from __future__ import annotations

import os
import pickle
import typing as t

from pathlib import Path

from .types import PathLike


def read_pickled_list(file_path: PathLike | Path) -> list[t.Any]:
    """"""
    with open(file_path, "rb") as fh:
        lst = pickle.load(fh)
    return lst
