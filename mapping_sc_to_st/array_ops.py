"""Small array utilities used across the pipeline.

Centralizing these avoids subtle drift (dtype/eps handling) across modules.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import issparse

def to_dense(X, *, dtype=None):
    """Convert sparse matrix to dense ndarray (copy only if needed)."""
    if issparse(X):
        X = X.toarray()
    X = np.asarray(X)
    if dtype is not None and X.dtype != dtype:
        X = X.astype(dtype, copy=False)
    return X

def row_l2_normalize(X: np.ndarray, *, eps: float = 1e-12):
    """Row-wise L2 normalization.

    Returns
    -------
    Xn : np.ndarray
        Normalized matrix.
    norms : np.ndarray
        Row norms (before normalization).
    """
    X = np.asarray(X)
    norms = np.linalg.norm(X, axis=1)
    denom = np.maximum(norms, eps)
    return (X / denom[:, None]), norms

def col_normalize(T: np.ndarray, *, eps: float = 1e-12):
    """Column-normalize a transport matrix so that each column sums to 1."""
    T = np.asarray(T)
    s = T.sum(axis=0)
    s = np.maximum(s, eps)
    return T / s[None, :]
