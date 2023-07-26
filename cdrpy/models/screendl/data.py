"""
Data loading utilities for ScreenDL.
"""

from __future__ import annotations

import pandas as pd

from pathlib import Path

from cdrpy.util.types import PathLike
from cdrpy.feat.encoders import PandasEncoder


def load_cell_features(exp_path: PathLike | Path) -> PandasEncoder:
    """Load cell features for ScreenDL."""
    exp_mat = pd.read_csv(exp_path, index_col=0).astype("float32")
    return PandasEncoder(exp_mat, name="cell_encoder")


def load_drug_features(mol_path: PathLike | Path) -> PandasEncoder:
    """Load drug features for ScreenDL."""
    mol_mat = pd.read_csv(mol_path, index_col=0).astype("int32")
    return PandasEncoder(mol_mat, name="drug_encoder")
