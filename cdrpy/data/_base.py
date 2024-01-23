""""""

from __future__ import annotations

import pickle

import typing as t

from enum import Enum, auto
from importlib import resources
from pathlib import Path

DATA_MODULE = "cdrpy.data.resources"
GENE_DATA_MODULE = f"{DATA_MODULE}.genelists"


class Genelist(str, Enum):
    """Available cdrpy genelists."""

    CGC = "cgc"
    MCG = "mcg"
    LINCS = "lincs"
    HALLMARK = "hallmark"


GENELIST_FILES = {
    Genelist.CGC: "cgc.pkl",
    Genelist.MCG: "mcg.pkl",
    Genelist.LINCS: "lincs.pkl",
    Genelist.HALLMARK: "hallmark.pkl",
}


def load_pickled_data_resource(
    file_name: str | Path, *, data_module: str = DATA_MODULE
) -> t.Any:
    """"""
    file_path = resources.files(data_module).joinpath(file_name)
    with open(file_path, "rb") as compressed_file:
        obj = pickle.load(compressed_file)
    return obj


def load_genelist(genelist: Genelist) -> t.List[str]:
    """"""
    return load_pickled_data_resource(
        GENELIST_FILES[genelist], data_module=GENE_DATA_MODULE
    )
