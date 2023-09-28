"""

"""

from __future__ import annotations

import csv
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


def read_gmt(file_path: str | Path) -> dict[str, list[str]]:
    """Reads a `.gmt` file into a `dict` mapping gene sets to genes."""
    gs_dict = {}
    with open(file_path, "r") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            gs_dict[row[0]] = row[2:]
    return gs_dict


def read_txt(
    file_path: str | Path,
    skip_rows: int = 0,
    delimiter: str = "\t",
    n_cols: int | None = None,
) -> t.Generator[t.Any, None, None]:
    """"""
    with open(file_path, "r") as fh:
        reader = csv.reader(fh, delimiter=delimiter)
        for i, row in enumerate(reader):
            if i >= skip_rows and any(item for item in row):
                if n_cols is not None:
                    yield row[:n_cols]
                else:
                    yield row
