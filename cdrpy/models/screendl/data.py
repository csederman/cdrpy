"""
Data loading utilities for ScreenDL.
"""

from __future__ import annotations

import pandas as pd

from pathlib import Path

from cdrpy.util.types import PathLike
from cdrpy.feat.encoders import PandasEncoder


def load_cell_features(
    exp_path: PathLike | Path,
    mut_path: PathLike | Path | None = None,
    cnv_path: PathLike | Path | None = None,
) -> tuple[PandasEncoder, PandasEncoder | None, PandasEncoder | None]:
    """Load cell features for ScreenDL."""
    exp_mat = pd.read_csv(exp_path, index_col=0).astype("float32")
    exp_enc = PandasEncoder(exp_mat, name="cell_encoder")

    mut_enc = None
    if mut_path is not None:
        mut_mat = pd.read_csv(mut_path, index_col=0).astype("int32")
        mut_enc = PandasEncoder(mut_mat, name="mut_encoder")

    cnv_enc = None
    if cnv_path is not None:
        cnv_mat = pd.read_csv(cnv_path, index_col=0).astype("float32")
        cnv_enc = PandasEncoder(cnv_mat, name="cnv_encoder")

    return exp_enc, mut_enc, cnv_enc


def load_drug_features(mol_path: PathLike | Path) -> PandasEncoder:
    """Load drug features for ScreenDL."""
    mol_mat = pd.read_csv(mol_path, index_col=0).astype("int32")
    return PandasEncoder(mol_mat, name="drug_encoder")
