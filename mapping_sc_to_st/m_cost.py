# util/m_cost.py
"""
Unified feature-cost (M) construction utilities.

Policy
------
- Build raw expression cost M_expr (outside) and optional anchor cost M_anchor.
- Normalize EACH cost matrix before mixing (default: mean scaling).
- Mix as: M = beta_expr * M_expr_norm + (1-beta_expr) * M_anchor_norm.

This keeps beta_expr meaning consistent across pairwise / final-global / solvers.
"""
from __future__ import annotations
import numpy as np
from typing import Optional

def normalize_cost(M: np.ndarray, mode: str = "mean", eps: float = 1e-12) -> np.ndarray:
    M = np.asarray(M, dtype=float)
    if mode in (None, "none"):
        return M
    if mode == "mean":
        s = float(np.mean(M))
    elif mode == "median":
        s = float(np.median(M))
    elif mode == "fro":
        s = float(np.linalg.norm(M))
    else:
        raise ValueError("mode must be 'mean' | 'median' | 'fro' | 'none'")
    if not np.isfinite(s) or s <= eps:
        s = 1.0
    return M / (s + eps)

def build_M(
    M_expr: np.ndarray,
    *,
    M_anchor: Optional[np.ndarray] = None,
    beta_expr: float = 0.7,
    norm: str = "mean",
    eps: float = 1e-12,
) -> np.ndarray:
    M_expr_n = normalize_cost(M_expr, mode=norm, eps=eps)
    if M_anchor is None:
        return M_expr_n
    M_anchor_n = normalize_cost(M_anchor, mode=norm, eps=eps)
    beta = float(np.clip(beta_expr, 0.0, 1.0))
    return beta * M_expr_n + (1.0 - beta) * M_anchor_n
