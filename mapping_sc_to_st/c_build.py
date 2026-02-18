# util/c_build.py
"""Utilities to build raw geometry matrices C_sc / C_st.

We keep these builders *solver-agnostic* and output **raw** C matrices.
Any solver-specific interpretation (FGW: distance/sim; FUGW: distance->sim)
is handled by :func:`util.transport_engine.run_transport`.

Supported raw builders
----------------------
1) Euclidean distance on an embedding/coordinate matrix.
2) kNN graph derived distance (dense fill or geodesic shortest-path).

These are especially useful for MERFISH-like irregular point sets on the ST side.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .array_ops import to_dense
from sklearn.metrics import pairwise_distances

def get_geom(
    adata,
    *,
    obsm_key: Optional[str] = None,
    obs_keys: Optional[Sequence[str]] = None,
    fallback_X: Optional[np.ndarray] = None,
    allow_X_fallback: bool = True,
) -> np.ndarray:
    """Fetch a geometry/embedding matrix from AnnData.

    Priority:
      1) adata.obsm[obsm_key] if provided and exists
      2) stacked adata.obs[obs_keys] if provided
      3) fallback_X if provided
      4) adata.X (only if allow_X_fallback=True)

    Notes:
      - Uses util_v2.array_ops.to_dense to safely handle sparse matrices.
    """
    if obsm_key is not None and obsm_key in getattr(adata, "obsm", {}):
        return to_dense(adata.obsm[obsm_key], dtype=float)

    if obs_keys is not None:
        cols = []
        for k in obs_keys:
            if k not in adata.obs:
                raise KeyError(f"obs key not found: {k}")
            cols.append(np.asarray(adata.obs[k], dtype=float))
        return np.stack(cols, axis=1)

    if fallback_X is not None:
        return to_dense(fallback_X, dtype=float)

    if allow_X_fallback:
        X = getattr(adata, "X", None)
        if X is None:
            raise ValueError("No geometry source found.")
        return to_dense(X, dtype=float)

    raise ValueError("No geometry source found.")

def _normalize_C(C: np.ndarray, normalize: Optional[str], eps: float = 1e-12) -> np.ndarray:
    if normalize in (None, "none"):
        return C
    vals = C[C > 0]
    if vals.size == 0:
        return C
    if normalize == "median":
        s = float(np.median(vals))
    elif normalize == "q95":
        s = float(np.quantile(vals, 0.95))
    else:
        raise ValueError("normalize must be 'median'|'q95'|None")
    if not np.isfinite(s) or s <= eps:
        s = 1.0
    return C / (s + eps)

def build_C_euclidean(
    geom: np.ndarray,
    *,
    metric: str = "euclidean",
    normalize: Optional[str] = "q95",
    eps: float = 1e-12,
) -> np.ndarray:
    """Dense pairwise distance C from an embedding/coordinate matrix."""
    X = np.asarray(geom, dtype=float)
    C = pairwise_distances(X, metric=metric)

    if normalize in (None, "none"):
        return C

    vals = C[C > 0]
    if vals.size == 0:
        return C
    if normalize == "median":
        s = float(np.median(vals))
    elif normalize == "q95":
        s = float(np.quantile(vals, 0.95))
    else:
        raise ValueError("normalize must be 'median'|'q95'|None")

    if not np.isfinite(s) or s <= eps:
        s = 1.0
    return C / (s + eps)

def _get_obsp_key(adata, key_added: str, kind: str) -> str:
    if key_added == "neighbors":
        return kind
    return f"{key_added}_{kind}"

def get_knn_distances_sparse(adata, *, key_added: str) -> "object":
    """Return sparse distance matrix produced by scanpy neighbors."""
    key = _get_obsp_key(adata, key_added, "distances")
    if key not in adata.obsp:
        raise KeyError(f"kNN distances not found in adata.obsp['{key}']")
    return adata.obsp[key]

def knn_distances_to_dense_C(D_sparse, *, fill_value="max", symmetrize=True):
    """Convert a sparse kNN distance matrix to a dense cost matrix C.

    Parameters
    ----------
    D_sparse:
        Sparse distance matrix (CSR/CSC). Only kNN edges have finite values.
    fill_value:
        Value to fill for missing edges. If "max", uses max observed distance.
    symmetrize:
        If True, makes C symmetric by taking min(C, C.T).

    Notes
    -----
    - Diagonal is set to 0.
    - Missing edges should NOT be left as 0 off-diagonal (would imply zero cost).
    """
    import numpy as _np
    from scipy.sparse import csr_matrix as _csr

    D = D_sparse
    if not hasattr(D, "tocsr"):
        D = _csr(D)
    D = D.tocsr()
    n = D.shape[0]
    C = _np.zeros((n, n), dtype=float)

    rows, cols = D.nonzero()
    C[rows, cols] = D.data

    if fill_value == "max":
        vmax = float(D.data.max()) if D.nnz > 0 else 1.0
        fill = vmax
    else:
        fill = float(fill_value)

    missing = (C == 0)
    _np.fill_diagonal(missing, False)
    C[missing] = fill

    if symmetrize:
        C = _np.minimum(C, C.T)

    _np.fill_diagonal(C, 0.0)
    return C

def knn_graph_geodesic_C(D_sparse, *, directed=False):
    """Graph geodesic (shortest-path) distances on a kNN graph.

    Returns a dense matrix; disconnected pairs are filled with max finite distance.
    """
    import numpy as _np
    from scipy.sparse import csr_matrix as _csr
    from scipy.sparse.csgraph import shortest_path as _shortest_path

    D = D_sparse
    if not hasattr(D, "tocsr"):
        D = _csr(D)
    D = D.tocsr()

    C = _shortest_path(D, directed=directed, unweighted=False)
    C = _np.asarray(C, dtype=float)

    finite = _np.isfinite(C)
    if _np.any(finite):
        mx = float(_np.max(C[finite]))
        C[~finite] = mx
    else:
        C[:] = 1.0

    _np.fill_diagonal(C, 0.0)
    return C

def build_C_from_knn(
    adata,
    *,
    idx: np.ndarray,
    key_added: str,
    mode: str = "geodesic",
    fill_value: str | float = "max",
    symmetrize: bool = True,
    directed: bool = False,
) -> np.ndarray:
    """Build dense C for a subset idx using a kNN distance graph.

    Parameters
    ----------
    mode:
      - "geodesic": shortest-path distances on the subgraph
      - "dense": directly densify kNN distances with missing filled
    """
    from scipy.sparse import csr_matrix

    idx = np.asarray(idx, dtype=int)
    D = get_knn_distances_sparse(adata, key_added=key_added)
 # subset rows/cols
    D_sub = D[idx][:, idx]
    if not hasattr(D_sub, "tocsr"):
        D_sub = csr_matrix(D_sub)
    D_sub = D_sub.tocsr()

    if mode == "dense":
        return knn_distances_to_dense_C(D_sub, fill_value=fill_value, symmetrize=symmetrize)
    if mode == "geodesic":
        return knn_graph_geodesic_C(D_sub, directed=directed)
    raise ValueError("mode must be 'dense'|'geodesic'")
