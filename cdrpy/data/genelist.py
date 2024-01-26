"""Genelist data."""

from __future__ import annotations

import typing as t

from enum import Enum
from cdrpy.data._base import DATA_MODULE, load_pickled_data_resource


GENE_DATA_MODULE = f"{DATA_MODULE}.genelists"


class GenelistEnum(str, Enum):
    """Available cdrpy genelists."""

    CGC = "cgc"
    MCG = "mcg"
    LINCS = "lincs"
    HALLMARK = "hallmark"


GENELIST_FILES = {
    GenelistEnum.CGC: "cgc.pkl",
    GenelistEnum.MCG: "mcg.pkl",
    GenelistEnum.LINCS: "lincs.pkl",
    GenelistEnum.HALLMARK: "hallmark.pkl",
}


def load_genelist(genelist: GenelistEnum) -> t.List[str]:
    """"""
    return load_pickled_data_resource(
        GENELIST_FILES[genelist], data_module=GENE_DATA_MODULE
    )
