""""""

from __future__ import annotations

import pickle

import typing as t

from importlib import resources
from pathlib import Path

DATA_MODULE = "cdrpy.data.resources"


def load_pickled_data_resource(
    file_name: str | Path, *, data_module: str = DATA_MODULE
) -> t.Any:
    """"""
    file_path = resources.files(data_module).joinpath(file_name)
    with open(file_path, "rb") as compressed_file:
        obj = pickle.load(compressed_file)
    return obj
