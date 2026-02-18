# util/geometry.py
"""
Geometry (C) adapters for FGW/FUGW.

We keep C_raw in either "distance" (smaller=closer) or "similarity" (larger=closer).
FGW-family can accept either, but we scale-normalize for stability.
FUGW (POT fused_unbalanced_gromov_wasserstein) is typically expressed with similarity;
we therefore convert distance->similarity by default for that solver.
"""
from __future__ import annotations
import numpy as np
from typing import Optional

def _clip_quantile(C: np.ndarray, q: Optional[float]) -> np.ndarray:
    if q is None:
        return C
    q = float(q)
    if not (0.0 < q < 0.5):
        raise ValueError("clip_q must be in (0, 0.5)")
    lo, hi = np.quantile(C, [q, 1.0 - q])
    return np.clip(C, lo, hi)

def scale_normalize(C: np.ndarray, mode: str = "mean", eps: float = 1e-12) -> np.ndarray:
    C = np.asarray(C, dtype=float)
    if mode in (None, "none"):
        return C
    if mode == "mean":
        s = float(np.mean(C))
    elif mode == "median":
        s = float(np.median(C))
    elif mode == "q95":
        s = float(np.quantile(C, 0.95))
    else:
        raise ValueError("mode must be 'mean'|'median'|'q95'|'none'")
    if not np.isfinite(s) or s <= eps:
        s = 1.0
    return C / (s + eps)

def distance_to_similarity(D: np.ndarray, method: str = "rbf", sigma: Optional[float] = None, eps: float = 1e-12) -> np.ndarray:
    D = np.asarray(D, dtype=float)
    if method == "rbf":
        if sigma is None:
            sigma = float(np.median(D)) + eps
        sigma = float(sigma)
        if not np.isfinite(sigma) or sigma <= eps:
            sigma = 1.0
        return np.exp(-D / (sigma + eps))
    if method == "minmax":
        mn, mx = float(np.min(D)), float(np.max(D))
        return 1.0 - (D - mn) / (mx - mn + eps)
    raise ValueError("method must be 'rbf'|'minmax'")

def adapt_C_for_fgw(
    C_raw: np.ndarray,
    *,
    C_kind: str = "distance",
    norm: str = "mean",
    clip_q: Optional[float] = None,
    eps: float = 1e-12,
) -> np.ndarray:
    C = np.asarray(C_raw, dtype=float)
    C = _clip_quantile(C, clip_q)
 # FGW can use distance or similarity; keep as-is.
    return scale_normalize(C, mode=norm, eps=eps)

def adapt_C_for_fugw(
    C_raw: np.ndarray,
    *,
    C_kind: str = "distance",
    to_similarity: bool = True,
    sim_method: str = "rbf",
    sigma: Optional[float] = None,
    norm: str = "mean",
    clip_q: Optional[float] = None,
    eps: float = 1e-12,
) -> np.ndarray:
    C = np.asarray(C_raw, dtype=float)
    C = _clip_quantile(C, clip_q)
    ck = str(C_kind)
    if to_similarity and ck == "distance":
        C = distance_to_similarity(C, method=sim_method, sigma=sigma, eps=eps)
    return scale_normalize(C, mode=norm, eps=eps)
