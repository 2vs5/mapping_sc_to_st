"""Anchor selection utilities (ST side).

This file intentionally stays small.

We keep anchor selection separate from any alignment logic so that:
- the update pipeline can stay focused on transport solving;
- alignment/QC implementations can evolve independently.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, List

import numpy as np

def _as_str(a) -> np.ndarray:
    return np.asarray(a, dtype=object).astype(str)

def _grid_ids(points: np.ndarray, bins: int = 10) -> np.ndarray:
    """Assign each point to a coarse grid cell (hash id)."""
    P = np.asarray(points, float)
    if P.ndim != 2 or P.shape[0] == 0:
        return np.zeros((P.shape[0],), dtype=np.int64)
    mins = P.min(axis=0)
    maxs = P.max(axis=0)
    span = np.maximum(maxs - mins, 1e-12)
    U = (P - mins) / span
    U = np.clip(U, 0.0, 0.999999)
    q = np.floor(U * int(max(1, bins))).astype(int)
    base = int(max(1, bins))
    h = np.zeros((P.shape[0],), dtype=np.int64)
    for d in range(q.shape[1]):
        h = h * base + q[:, d]
    return h

def _pick_diverse(idx_sorted: np.ndarray, grid: np.ndarray, k: int, min_per_grid: int = 1) -> np.ndarray:
    """Pick top-k indices while enforcing simple grid diversity."""
    idx_sorted = np.asarray(idx_sorted, dtype=int)
    grid = np.asarray(grid, dtype=np.int64)
    if k <= 0 or idx_sorted.size == 0:
        return np.array([], dtype=int)
    out: List[int] = []
    used: Dict[int, int] = {}
    for idx in idx_sorted:
        g = int(grid[idx])
        c = used.get(g, 0)
        if c < int(max(1, min_per_grid)):
            out.append(int(idx))
            used[g] = c + 1
            if len(out) >= k:
                return np.asarray(out, dtype=int)
 # fill remaining regardless of grid
    for idx in idx_sorted:
        if int(idx) in out:
            continue
        out.append(int(idx))
        if len(out) >= k:
            break
    return np.asarray(out, dtype=int)

def select_st_anchors_final(
    adata_st,
    *,
    st_geom_key: str,
    st_type_key: str,
    st_score_key: str,
    n_anchor_per_type: int = 50,
    n_anchor_global: int = 0,
    top_frac_per_type: Optional[float] = 0.25,
    min_per_type: int = 10,
    diversity_bins: int = 10,
    diversity_min_per_grid: int = 1,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Select ST anchor spot indices for FINAL.

    Strategy:
    - For each type t: take top candidates by score (optionally top fraction),
      then pick a diverse subset in the ST geometry space.
    - Optionally add global anchors from all spots.

    Returns
    -------
    idx_anchor : (K,) global ST indices
    anchors_by_type : dict type -> indices (subset of idx_anchor)
    """
    if st_type_key not in adata_st.obs:
        raise KeyError(f"missing adata_st.obs['{st_type_key}']")
    if st_score_key not in adata_st.obs:
        raise KeyError(f"missing adata_st.obs['{st_score_key}']")
    if st_geom_key not in adata_st.obsm:
        raise KeyError(f"missing adata_st.obsm['{st_geom_key}']")

    y = np.asarray(adata_st.obsm[st_geom_key], dtype=float)
    t = _as_str(adata_st.obs[st_type_key])
    s = np.asarray(adata_st.obs[st_score_key], dtype=float)

    grid = _grid_ids(y, bins=int(max(1, diversity_bins)))

    idx_anchor_all: List[int] = []
    anchors_by_type: Dict[str, np.ndarray] = {}
    for ct in np.unique(t):
        mask = (t == ct) & np.isfinite(s)
        idx = np.where(mask)[0]
        if idx.size < int(min_per_type):
            continue
 # sort by score desc
        idx_sorted = idx[np.argsort(-s[idx])]
        if top_frac_per_type is not None:
            m = int(max(1, round(float(top_frac_per_type) * idx_sorted.size)))
            idx_sorted = idx_sorted[:m]
        picked = _pick_diverse(
            idx_sorted,
            grid,
            k=int(n_anchor_per_type),
            min_per_grid=int(max(1, diversity_min_per_grid)),
        )
        anchors_by_type[str(ct)] = picked.astype(int)
        idx_anchor_all.extend([int(x) for x in picked])

 # global anchors
    if int(n_anchor_global) > 0:
        idx_sorted = np.argsort(-np.nan_to_num(s, nan=-np.inf))
        picked = _pick_diverse(
            idx_sorted,
            grid,
            k=int(n_anchor_global),
            min_per_grid=int(max(1, diversity_min_per_grid)),
        )
        idx_anchor_all.extend([int(x) for x in picked])

 # unique preserve order
    seen = set()
    uniq: List[int] = []
    for x in idx_anchor_all:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    idx_anchor = np.asarray(uniq, dtype=int)
    return idx_anchor, anchors_by_type
