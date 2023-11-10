"""

"""

from __future__ import annotations

import csv
import h5py
import pickle

import numpy as np
import pandas as pd
import typing as t

from pathlib import Path

from .types import PathLike


K = t.TypeVar("K")
V = t.TypeVar("V")


def read_pickled_list(file_path: PathLike | Path) -> t.List[t.Any]:
    """"""
    with open(file_path, "rb") as fh:
        lst = pickle.load(fh)
    return lst


def read_pickled_dict(file_path: PathLike | Path) -> t.Dict[t.Any, t.Any]:
    """Load pickled dict object from file."""
    with open(file_path, "rb") as fh:
        dct = pickle.load(fh)
    return dct


def read_list(file_path: PathLike | Path) -> t.List[str]:
    with open(file_path) as fh:
        lines = fh.read().splitlines()
    return lines


def read_gmt(file_path: str | Path) -> t.Dict[str, t.List[str]]:
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


def pandas_to_h5(
    file_or_group: h5py.File | h5py.Group,
    df: pd.DataFrame,
    index: bool = False,
) -> None:
    """Saves a `pd.DataFrame` in h5 format."""
    file_or_group.attrs["cols"] = list(df.columns)
    for col_name, col_data in df.items():
        dt = col_data.dtype
        if dt == np.object_:
            dt = h5py.special_dtype(vlen=str)
        file_or_group.create_dataset(col_name, data=col_data.values, dtype=dt)

    if index:
        dt = df.index.dtype
        if dt == np.object_:
            dt = h5py.special_dtype(vlen=str)
        file_or_group.create_dataset("index", data=df.index.values, dtype=dt)


def pandas_from_h5(file_or_group: h5py.File | h5py.Group) -> pd.DataFrame:
    """Loads a `pd.DataFrame` from an h5 file."""
    data = {}
    for col in file_or_group.attrs["cols"]:
        dt = file_or_group[col].dtype
        if dt == np.object_:
            dt = str
        data[col] = np.array(file_or_group[col], dtype=dt)

    index = file_or_group.get("index")
    if index is not None:
        dt = index.dtype
        if dt == np.object_:
            dt = str
        index = np.array(index, dtype=dt)
    return pd.DataFrame(data, index=index)
