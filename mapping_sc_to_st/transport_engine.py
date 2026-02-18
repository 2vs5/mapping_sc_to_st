# util/transport_engine.py
"""Common transport engine used by both pairwise refinement and final-global updates.

This module centralizes:

1) M construction (expression cost + optional anchor cost) using :func:`util.m_cost.build_M`.
2) C adaptation policy using :func:`util.geometry.adapt_C_for_fgw` and
   :func:`util.geometry.adapt_C_for_fugw`.
3) Solving via a solver object (typically :class:`util.fgw_solver.POTFGWSolver`).
4) Column-normalization of the transport plan to produce mixture weights W.

The goal is that higher-level algorithms (pairwise refinement, final-global update)
can share exactly the same transport core while differing only in:
- how they choose SC/ST indices (corridors/subsets)
- what features they use (genes/modules)
- how they build optional anchor costs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from sklearn.metrics import pairwise_distances

from .array_ops import col_normalize
from .geometry import adapt_C_for_fgw, adapt_C_for_fugw
from .m_cost import build_M

@dataclass
class TransportResult:
    """Output container for a transport solve."""

    T: np.ndarray
    W: np.ndarray
    M: np.ndarray
    log: Any
    C_sc: np.ndarray
    C_st: np.ndarray

def run_transport(
    *,
    X_sc_feat: np.ndarray,
    X_st_feat: np.ndarray,
    C_sc_raw: np.ndarray,
    C_st_raw: np.ndarray,
    solver,
 # M
    M_anchor: Optional[np.ndarray] = None,
    beta_expr: float = 0.7,
    M_norm: str = "mean",
 # C adapter policy
    C_kind_sc: str = "distance",
    C_kind_st: str = "distance",
    C_norm: str = "q95",
    C_clip_q: Optional[float] = None,
 # FUGW conversion (only used if solver.method indicates FUGW)
    fugw_to_similarity: bool = True,
    fugw_sim_method: str = "rbf",
    fugw_sigma: Optional[float] = None,
 # solver kwargs
    **solver_kwargs,
) -> TransportResult:
    """Solve a transport plan and return both T and column-normalized weights W.

    Notes
    -----
    - This function always builds M_expr from X_sc_feat/X_st_feat via squared L2.
    - Marginals are uniform by default (consistent with your current pipeline).
    - C adaptation is applied here so that pairwise/final-global share identical logic.
    """

    X_sc_feat = np.asarray(X_sc_feat, dtype=float)
    X_st_feat = np.asarray(X_st_feat, dtype=float)

 # 1) M (expression cost) + optional anchor cost
    M_expr = pairwise_distances(X_sc_feat, X_st_feat, metric="euclidean") ** 2
    M = build_M(M_expr, M_anchor=M_anchor, beta_expr=beta_expr, norm=M_norm)

 # 2) C adaptation based on solver family
    method = getattr(solver, "method", "fgw")
    method = str(method)
    if method in ("fugw", "fused_unbalanced_gromov_wasserstein"):
        C_sc = adapt_C_for_fugw(
            C_sc_raw,
            C_kind=C_kind_sc,
            to_similarity=fugw_to_similarity,
            sim_method=fugw_sim_method,
            sigma=fugw_sigma,
            norm=C_norm,
            clip_q=C_clip_q,
        )
        C_st = adapt_C_for_fugw(
            C_st_raw,
            C_kind=C_kind_st,
            to_similarity=fugw_to_similarity,
            sim_method=fugw_sim_method,
            sigma=fugw_sigma,
            norm=C_norm,
            clip_q=C_clip_q,
        )
    else:
        C_sc = adapt_C_for_fgw(C_sc_raw, C_kind=C_kind_sc, norm=C_norm, clip_q=C_clip_q)
        C_st = adapt_C_for_fgw(C_st_raw, C_kind=C_kind_st, norm=C_norm, clip_q=C_clip_q)

 # 3) uniform marginals
    n_sc = int(X_sc_feat.shape[0])
    n_st = int(X_st_feat.shape[0])
    p = np.ones(n_sc, dtype=float) / max(n_sc, 1)
    q = np.ones(n_st, dtype=float) / max(n_st, 1)

 # 4) solve
    T, logd = solver.solve(M, C_sc, C_st, p, q, **solver_kwargs)
    W = col_normalize(T)

    return TransportResult(T=T, W=W, M=M, log=logd, C_sc=C_sc, C_st=C_st)
