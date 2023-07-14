"""
DualGCN data loading utilities.
"""

from __future__ import annotations

import typing as t

import networkx as nx
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path

from .util import normalize_adj

from cdrpy.feat.encoders import DictEncoder, RepeatEncoder
from cdrpy.util.validation import check_same_columns, check_same_indexes
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
    file_path: PathLike | Path,
) -> tuple[DictEncoder, DictEncoder]:
    """Load drug chemical features."""
    drug_dict = read_pickled_dict(file_path)
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
    cnv_path: PathLike | Path,
    ppi_path: PathLike | Path,
) -> tuple[DictEncoder, RepeatEncoder]:
    """Load cell line omics features.

    Parameters
    ----------
        exp_path:
        cnv_path:
        ppi_path:

    Returns
    -------
        A tuple of (omics_encoder, ppi_encoder).
    """

    # load the cell line omics data
    exp_mat = pd.read_csv(exp_path, index_col=0).astype("float32")
    cnv_mat = pd.read_csv(cnv_path, index_col=0).astype("float32")

    check_same_columns(exp_mat, cnv_mat)
    check_same_indexes(exp_mat, cnv_mat)

    # extract and reshape the omics features
    omics_dict = {}
    for cell_id in exp_mat.index:
        cell_exp = exp_mat.loc[cell_id].values.reshape(-1, 1)
        cell_cnv = cnv_mat.loc[cell_id].values.reshape(-1, 1)
        omics_dict[cell_id] = np.hstack((cell_exp, cell_cnv))

    n_genes = exp_mat.shape[1]
    gene2ind = dict(zip(exp_mat.columns, range(n_genes)))

    # load the PPI
    ppi_edges = pd.read_csv(ppi_path)
    ppi_graph = nx.Graph()
    ppi_graph.add_edges_from(zip(ppi_edges["gene_1"], ppi_edges["gene_2"]))

    # extract the PPI adjacency matrix
    ppi_adj = np.zeros((n_genes, n_genes))
    for gene_1, gene_2 in ppi_graph.edges:
        gene_1_ind = gene2ind[gene_1]
        gene_2_ind = gene2ind[gene_2]
        ppi_adj[gene_1_ind, gene_2_ind] = 1
        ppi_adj[gene_2_ind, gene_1_ind] = 1

    assert np.allclose(ppi_adj, ppi_adj.T)

    ppi_adj_norm = normalize_adj(ppi_adj).astype("float32")

    return (
        DictEncoder(omics_dict, name="omics_encoder"),
        RepeatEncoder(tf.sparse.from_dense(ppi_adj_norm), name="ppi_encoder"),
    )
