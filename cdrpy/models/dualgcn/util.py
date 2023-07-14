"""
DualGCN utilities.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def normalize_adj(A: np.ndarray) -> np.ndarray:
    """Normalize adjacency matrix."""
    A = A + np.eye(A.shape[0])
    diags = sp.diags(np.power(A.sum(axis=1), -0.5)).toarray()
    return A.dot(diags).transpose().dot(diags)
