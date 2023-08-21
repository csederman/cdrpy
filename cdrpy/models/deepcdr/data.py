"""
DeepCDR data loading utilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pathlib import Path

from .util import normalize_adj

from cdrpy.feat.encoders import DictEncoder, PandasEncoder
from cdrpy.types import PathLike
from cdrpy.util.io import read_pickled_dict


DegList = list[np.int32]
AdjList = list[list[int]]
ConvMolFeat = tuple[np.ndarray, DegList, AdjList]


def _process_drug_feature(
    feat_mat: np.ndarray, adj_lst: AdjList, max_atoms: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Add padding and convert to `np.ndarray`.

    Parameters
    ----------
        feat_mat:
        adj_lst:
        max_atoms:

    Returns
    -------
        A tuple of (feature_matrix, adjacency_matrix).
    """
    n_atoms = len(adj_lst)
    assert feat_mat.shape[0] == n_atoms

    F = np.zeros((max_atoms, feat_mat.shape[-1]))
    F[: feat_mat.shape[0], :] = feat_mat
    A = np.zeros((max_atoms, max_atoms), dtype="float32")

    for i, nodes in enumerate(adj_lst):
        for n in nodes:
            A[i, int(n)] = 1

    assert np.allclose(A, A.T)

    A_vals = normalize_adj(A[:n_atoms, :n_atoms])
    A_pad = normalize_adj(A[n_atoms:, n_atoms:])

    A[:n_atoms, :n_atoms] = A_vals
    A[n_atoms:, n_atoms:] = A_pad

    return F, A


def load_drug_features(
    mol_path: PathLike | Path,
) -> tuple[DictEncoder, DictEncoder]:
    """Load drug chemical features."""
    drug_dict = read_pickled_dict(mol_path)
    drug_dict = {
        k: _process_drug_feature(F, A) for k, (F, _, A) in drug_dict.items()
    }
    drug_feat = {k: v[0].astype("float32") for k, v in drug_dict.items()}
    drug_adj = {k: v[1].astype("float32") for k, v in drug_dict.items()}
    return (
        DictEncoder(drug_feat, name="drug_feature_encoder"),
        DictEncoder(drug_adj, name="drug_adj_encoder"),
    )


def load_cell_features(
    exp_path: PathLike | Path,
    mut_path: PathLike | Path,
    methyl_path: PathLike | Path | None = None,
) -> tuple[PandasEncoder, DictEncoder, PandasEncoder | None]:
    """Load cell features for ScreenDL."""
    exp_mat = pd.read_csv(exp_path, index_col=0).astype("float32")
    mut_mat = pd.read_csv(mut_path, index_col=0).astype("int32")

    mut_dict = {}
    for cell_id in mut_mat.index:
        cell_mut = mut_mat.loc[cell_id].values.reshape(1, -1, 1)
        mut_dict[cell_id] = cell_mut

    methyl_enc = None
    if methyl_path is not None:
        methyl_mat = pd.read_csv(methyl_path, index_col=0).astype("float32")
        methyl_enc = PandasEncoder(methyl_mat, name="methyl_encoder")

    return (
        PandasEncoder(exp_mat, name="exp_encoder"),
        DictEncoder(mut_dict, name="mut_encoder"),
        methyl_enc,
    )
