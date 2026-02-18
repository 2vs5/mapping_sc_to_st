"""
Anchor-distance axis calibration (type-wise, Gaussian per axis).

This module implements the simplest alignment requested:

- Represent each ST spot / SC cell by a K-dim vector of distances to K anchors.
- For each cell type c and each anchor-axis k, match SOURCE -> TARGET by aligning
  mean and standard deviation (Gaussian assumption, independent axes).

Direction is configurable:
    - direction="sc_to_st":  SC phi -> ST phi (default; legacy behavior)
    - direction="st_to_sc":  ST phi -> SC phi

Notes
-----
- ST anchor: idx_anchor  ST spot ( anchor)
- SC anchor: weights_A  ST anchor SC pseudo center   anchor
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

# ----------------------------
# Small utilities
# ----------------------------
def _safe_mean_std(x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    mu = float(x.mean())
    sd = float(x.std(ddof=0))
    return mu, sd

def _to_np_str(arr) -> np.ndarray:
    return np.asarray(arr, dtype=object).astype(str)

# ----------------------------
# Phi builders
# ----------------------------
def compute_phi_st(
    *,
    st_geom_all: np.ndarray,
    idx_query: np.ndarray,
    idx_anchor: np.ndarray,
) -> np.ndarray:
    """Compute Phi_st[j,k] = || z_st[j] - z_st[anchor_k] ||.

    Returns float32 (n_query, K).
    """
    Z = np.asarray(st_geom_all, dtype=float)
    idx_query = np.asarray(idx_query, dtype=int)
    idx_anchor = np.asarray(idx_anchor, dtype=int)
    Q = Z[idx_query]
    A = Z[idx_anchor]
 # squared distances via norms + dot, then sqrt
    q2 = (Q**2).sum(axis=1, keepdims=True)
    a2 = (A**2).sum(axis=1, keepdims=True).T
    D2 = q2 + a2 - 2.0 * (Q @ A.T)
    np.maximum(D2, 0.0, out=D2)
    return np.sqrt(D2).astype(np.float32, copy=False)

def compute_phi_sc(
    *,
    sc_geom: np.ndarray,
    anchor_centers: np.ndarray,
) -> np.ndarray:
    """Compute Phi_sc[i,k] = || z_sc[i] - c_k ||, where c_k is pseudo anchor center.

    Returns float32 (n_sc, K).
    """
    Z = np.asarray(sc_geom, dtype=float)
    C = np.asarray(anchor_centers, dtype=float)
    z2 = (Z**2).sum(axis=1, keepdims=True)
    c2 = (C**2).sum(axis=1, keepdims=True).T
    D2 = z2 + c2 - 2.0 * (Z @ C.T)
    np.maximum(D2, 0.0, out=D2)
    return np.sqrt(D2).astype(np.float32, copy=False)

def compute_pseudo_anchor_centers_from_weights_A(
    *,
    sc_geom_all: np.ndarray,
    idx_anchor_st: np.ndarray,
    weights_A: Dict[int, Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute SC pseudo anchor centers for each ST anchor index.

    Parameters
    ----------
    sc_geom_all : (n_sc, d)
    idx_anchor_st : (K,) ST indices of anchor spots
    weights_A : dict st_idx -> {"sc_idx": array, "w": array}

    Returns
    -------
    centers : (K, d)
    valid   : (K,) bool

    Notes
    -----
    - We renormalize weights after filtering invalid indices/weights.
    - This function keeps sc_idx–w mapping intact.
    """
    Z = np.asarray(sc_geom_all, dtype=float)
    idx_anchor_st = np.asarray(idx_anchor_st, dtype=int)
    d = int(Z.shape[1])
    centers = np.zeros((idx_anchor_st.size, d), dtype=float)
    valid = np.zeros((idx_anchor_st.size,), dtype=bool)

    for k, st_idx in enumerate(idx_anchor_st):
        info = weights_A.get(int(st_idx), None)
        if info is None:
            continue
        sc_idx = np.asarray(info.get("sc_idx", []), dtype=int)
        w = np.asarray(info.get("w", []), dtype=float)
        if sc_idx.size == 0 or w.size == 0:
            continue
        m = int(min(sc_idx.size, w.size))
        sc_idx = sc_idx[:m]
        w = w[:m]
        mask = (sc_idx >= 0) & (sc_idx < Z.shape[0])
        mask &= np.isfinite(w) & (w > 0)
        if not np.any(mask):
            continue
        sc_idx = sc_idx[mask]
        w = w[mask]
        ws = float(w.sum())
        if not np.isfinite(ws) or ws <= 0:
            continue
        w = w / ws
        centers[k] = (w[:, None] * Z[sc_idx]).sum(axis=0)
        valid[k] = True

    return centers, valid

# ----------------------------
# Calibrator
# ----------------------------
@dataclass
class AxisGaussianCalibrator:
    """Type-wise, per-axis Gaussian calibrator (SOURCE -> TARGET).

    For each type c and axis k:
        x' = alpha[c,k] * x + beta[c,k]
    where alpha/beta are fitted to match mean/std from SOURCE to TARGET.
    """

    anchors: np.ndarray # (K,) anchor indices (ST; after valid filtering)
    types_seen: np.ndarray # (T,) str

 # global (fallback) per-axis transform
    alpha_global: np.ndarray # (K,)
    beta_global: np.ndarray # (K,)

 # type-specific override
    alpha_type: Dict[str, np.ndarray] # type -> (K,)
    beta_type: Dict[str, np.ndarray] # type -> (K,)

 # clipping bounds computed on TARGET (global/type-wise)
    clip_lo_type: Optional[Dict[str, np.ndarray]] = None # type -> (K,)
    clip_hi_type: Optional[Dict[str, np.ndarray]] = None # type -> (K,)
    clip_lo_global: Optional[np.ndarray] = None # (K,)
    clip_hi_global: Optional[np.ndarray] = None # (K,)

    eps: float = 1e-8
    min_n: int = 30
    shrink_k: int = 2000
    clip_q: Optional[Tuple[float, float]] = (0.01, 0.99)

 # metadata
    direction: str = "sc_to_st" # "sc_to_st" or "st_to_sc"
    source_name: str = "sc"
    target_name: str = "st"

def fit_axis_gaussian_calibrator(
    *,
    phi_st_train: np.ndarray,
    st_types_train: np.ndarray,
    phi_sc_train: np.ndarray,
    sc_types_train: np.ndarray,
    anchors: np.ndarray,
    direction: str = "sc_to_st",
    min_n: int = 30,
    eps: float = 1e-8,
    shrink_k: int = 2000,
    clip_q: Optional[Tuple[float, float]] = (0.01, 0.99),
) -> AxisGaussianCalibrator:
    """Fit per-type, per-axis alpha/beta to map SOURCE phi to TARGET phi.

    Parameters
    ----------
    phi_st_train : (n_st, K)
    st_types_train : (n_st,)
    phi_sc_train : (n_sc, K)
    sc_types_train : (n_sc,)
    anchors : (K,) ST anchor indices used to build the phi matrices (after valid filtering)
    direction : "sc_to_st" (default) or "st_to_sc"
        - sc_to_st: SOURCE=SC, TARGET=ST
        - st_to_sc: SOURCE=ST, TARGET=SC
    min_n : minimum samples per type on *both* sides to fit a type-specific map
    shrink_k : shrinkage strength toward global (higher => more shrink)
    clip_q : if not None, store per-axis quantile bounds from TARGET for clipping

    Returns
    -------
    AxisGaussianCalibrator : SOURCE -> TARGET transform
    """
    Phi_st = np.asarray(phi_st_train, dtype=float)
    Phi_sc = np.asarray(phi_sc_train, dtype=float)
    st_types_train = _to_np_str(st_types_train)
    sc_types_train = _to_np_str(sc_types_train)
    anchors = np.asarray(anchors, dtype=int)

    if Phi_st.ndim != 2 or Phi_sc.ndim != 2:
        raise ValueError("phi matrices must be 2D")
    if Phi_st.shape[1] != Phi_sc.shape[1]:
        raise ValueError("ST and SC phi must have same K (number of anchors)")

    direction = str(direction).lower().strip()
    if direction not in ("sc_to_st", "st_to_sc"):
        raise ValueError('direction must be "sc_to_st" or "st_to_sc"')

    if direction == "sc_to_st":
        Src = Phi_sc
        Tgt = Phi_st
        src_types = sc_types_train
        tgt_types = st_types_train
        source_name, target_name = "sc", "st"
    else:
        Src = Phi_st
        Tgt = Phi_sc
        src_types = st_types_train
        tgt_types = sc_types_train
        source_name, target_name = "st", "sc"

    K = int(Src.shape[1])

 # global stats per axis: align SOURCE -> TARGET
    mu_t = np.nanmean(Tgt, axis=0)
    sd_t = np.nanstd(Tgt, axis=0)
    mu_s = np.nanmean(Src, axis=0)
    sd_s = np.nanstd(Src, axis=0)

    alpha_g = sd_t / (sd_s + float(eps))
    beta_g = mu_t - alpha_g * mu_s

    clip_lo_g = clip_hi_g = None
    if clip_q is not None:
        q0, q1 = float(clip_q[0]), float(clip_q[1])
 # clipping bounds are always computed on TARGET distribution
        clip_lo_g = np.nanquantile(Tgt, q0, axis=0)
        clip_hi_g = np.nanquantile(Tgt, q1, axis=0)

    alpha_t: Dict[str, np.ndarray] = {}
    beta_t: Dict[str, np.ndarray] = {}
    clip_lo_t: Dict[str, np.ndarray] = {}
    clip_hi_t: Dict[str, np.ndarray] = {}

    types_seen = np.unique(np.concatenate([src_types, tgt_types]))
    for ct in types_seen:
        s_mask = (src_types == ct)
        t_mask = (tgt_types == ct)
        n_s = int(s_mask.sum())
        n_t = int(t_mask.sum())
        if n_s < int(min_n) or n_t < int(min_n):
            continue

        Src_c = Src[s_mask]
        Tgt_c = Tgt[t_mask]

        mu_t_c = np.nanmean(Tgt_c, axis=0)
        sd_t_c = np.nanstd(Tgt_c, axis=0)
        mu_s_c = np.nanmean(Src_c, axis=0)
        sd_s_c = np.nanstd(Src_c, axis=0)

        a = sd_t_c / (sd_s_c + float(eps))
        b = mu_t_c - a * mu_s_c

 # shrinkage toward global for stability:
 # target-side sample size as reliability proxy
        w = float(n_t) / float(n_t + int(max(1, shrink_k)))
        a = w * a + (1.0 - w) * alpha_g
        b = w * b + (1.0 - w) * beta_g

        alpha_t[str(ct)] = a.astype(np.float32)
        beta_t[str(ct)] = b.astype(np.float32)

        if clip_q is not None:
            q0, q1 = float(clip_q[0]), float(clip_q[1])
            clip_lo_t[str(ct)] = np.nanquantile(Tgt_c, q0, axis=0).astype(np.float32)
            clip_hi_t[str(ct)] = np.nanquantile(Tgt_c, q1, axis=0).astype(np.float32)

    return AxisGaussianCalibrator(
        anchors=anchors.astype(int),
        types_seen=types_seen.astype(object),
        alpha_global=alpha_g.astype(np.float32),
        beta_global=beta_g.astype(np.float32),
        alpha_type=alpha_t,
        beta_type=beta_t,
        clip_lo_type=(clip_lo_t if clip_q is not None else None),
        clip_hi_type=(clip_hi_t if clip_q is not None else None),
        clip_lo_global=(clip_lo_g.astype(np.float32) if clip_lo_g is not None else None),
        clip_hi_global=(clip_hi_g.astype(np.float32) if clip_hi_g is not None else None),
        eps=float(eps),
        min_n=int(min_n),
        shrink_k=int(shrink_k),
        clip_q=clip_q,
        direction=direction,
        source_name=source_name,
        target_name=target_name,
    )

def apply_axis_gaussian_calibrator_to_source(
    *,
    phi_source: np.ndarray,
    source_types: np.ndarray,
    model: AxisGaussianCalibrator,
    inplace: bool = False,
) -> np.ndarray:
    """Apply fitted calibrator (SOURCE -> TARGET) to a SOURCE phi matrix."""
    Src = phi_source if inplace else np.array(phi_source, dtype=np.float32, copy=True)
    source_types = _to_np_str(source_types)

    if Src.ndim != 2:
        raise ValueError("phi_source must be 2D")
    K = int(Src.shape[1])
    if int(model.alpha_global.shape[0]) != K:
        raise ValueError("model K mismatch")

 # global transform
    Out = Src * model.alpha_global[None, :] + model.beta_global[None, :]
    if model.clip_lo_global is not None and model.clip_hi_global is not None:
        Out = np.clip(Out, model.clip_lo_global[None, :], model.clip_hi_global[None, :])

 # type-specific override (compute from original SOURCE)
    for ct, a in model.alpha_type.items():
        mask = (source_types == str(ct))
        if not np.any(mask):
            continue
        b = model.beta_type[str(ct)]
        Out[mask] = Src[mask] * a[None, :] + b[None, :]
        if model.clip_lo_type is not None and model.clip_hi_type is not None:
            lo = model.clip_lo_type.get(str(ct), None)
            hi = model.clip_hi_type.get(str(ct), None)
            if lo is not None and hi is not None:
                Out[mask] = np.clip(Out[mask], lo[None, :], hi[None, :])

 # distances can't be negative
    np.maximum(Out, 0.0, out=Out)
    return Out

# Backward-compatible wrapper (legacy name kept):
def apply_axis_gaussian_calibrator(
    *,
    phi_sc: np.ndarray,
    sc_types: np.ndarray,
    model: AxisGaussianCalibrator,
    inplace: bool = False,
) -> np.ndarray:
    """Backward-compatible wrapper.

    Historically this applied to SC. If you fit direction="sc_to_st", this is correct.
    If you fit direction="st_to_sc", you should call apply_axis_gaussian_calibrator_to_source
    with phi_source=phi_st, source_types=st_types instead.
    """
    return apply_axis_gaussian_calibrator_to_source(
        phi_source=phi_sc,
        source_types=sc_types,
        model=model,
        inplace=inplace,
    )

# ----------------------------
# Cost
# ----------------------------
def compute_squared_cost_from_phi(
    *,
    phi_sc: np.ndarray,
    phi_st: np.ndarray,
) -> np.ndarray:
    """Return M(i,j) = ||phi_sc(i)-phi_st(j)||^2 as float32."""
    A = np.asarray(phi_sc, dtype=np.float32)
    B = np.asarray(phi_st, dtype=np.float32)
    a2 = np.sum(A**2, axis=1, dtype=np.float32)
    b2 = np.sum(B**2, axis=1, dtype=np.float32)
    G = (A @ B.T).astype(np.float32)
    M = a2[:, None] + b2[None, :] - 2.0 * G
    np.maximum(M, 0.0, out=M)
    return M

# ----------------------------
# Convenience: build phi + fit + apply (QC friendly)
# ----------------------------
def build_phi_before_after(
    *,
    adata_sc,
    adata_st,
    sc_type_key: str,
    st_type_key: str,
    sc_geom_key: str,
    st_geom_key: str,
    idx_anchor: np.ndarray,
    idx_st_train: np.ndarray,
    weights_A: dict, # st_idx -> {"sc_idx":..., "w":...}
    idx_sc_train: np.ndarray | None = None,
    direction: str = "sc_to_st",
    min_n: int = 30,
    eps: float = 1e-8,
    shrink_k: int = 2000,
    clip_q: tuple[float, float] | None = (0.01, 0.99),
):
    """
    Align SOURCE->TARGET according to `direction` and return phi for QC.

    Returns dict with keys:
      - phi_st_train
      - phi_sc_raw_train
      - phi_sc_aligned_train (if direction="sc_to_st" else None)
      - phi_st_aligned_train (if direction="st_to_sc" else None)
      - st_types_train, sc_types_train
      - idx_anchor_use, anchor_valid
      - model
      - idx_sc_train, idx_st_train
      - direction, source_name, target_name
    """
    idx_anchor = np.asarray(idx_anchor, dtype=int)
    idx_st_train = np.asarray(idx_st_train, dtype=int)

    Z_st = np.asarray(adata_st.obsm[st_geom_key])
    Z_sc = np.asarray(adata_sc.obsm[sc_geom_key])

 # 1) Build SC pseudo-anchor centers
    anchor_centers, anchor_valid = compute_pseudo_anchor_centers_from_weights_A(
        sc_geom_all=Z_sc,
        idx_anchor_st=idx_anchor,
        weights_A=weights_A,
    )
    if not np.any(anchor_valid):
        raise ValueError("No valid pseudo anchors. Check `weights_A` / `idx_anchor`.")

    idx_anchor_use = idx_anchor[anchor_valid]
    centers_use = anchor_centers[anchor_valid]

    phi_st_train = compute_phi_st(
        st_geom_all=Z_st,
        idx_query=idx_st_train,
        idx_anchor=idx_anchor_use,
    )

    if idx_sc_train is None:
        idx_sc_train = np.arange(Z_sc.shape[0], dtype=int)
    else:
        idx_sc_train = np.asarray(idx_sc_train, dtype=int)

    phi_sc_raw_train = compute_phi_sc(
        sc_geom=Z_sc[idx_sc_train],
        anchor_centers=centers_use,
    )

    st_types_train = _to_np_str(adata_st.obs[st_type_key].to_numpy()[idx_st_train])
    sc_types_train = _to_np_str(adata_sc.obs[sc_type_key].to_numpy()[idx_sc_train])

 # 3) Fit alignment calibrator
    model = fit_axis_gaussian_calibrator(
        phi_st_train=phi_st_train,
        st_types_train=st_types_train,
        phi_sc_train=phi_sc_raw_train,
        sc_types_train=sc_types_train,
        anchors=idx_anchor_use,
        direction=direction,
        min_n=min_n,
        eps=eps,
        shrink_k=shrink_k,
        clip_q=clip_q,
    )

 # 4) Apply to SOURCE phi depending on direction
    phi_sc_aligned_train = None
    phi_st_aligned_train = None

    if model.direction == "sc_to_st":
        phi_sc_aligned_train = apply_axis_gaussian_calibrator_to_source(
            phi_source=phi_sc_raw_train,
            source_types=sc_types_train,
            model=model,
            inplace=False,
        )
    else:
        phi_st_aligned_train = apply_axis_gaussian_calibrator_to_source(
            phi_source=phi_st_train,
            source_types=st_types_train,
            model=model,
            inplace=False,
        )

    return dict(
        phi_st_train=phi_st_train,
        phi_sc_raw_train=phi_sc_raw_train,
        phi_sc_aligned_train=phi_sc_aligned_train,
        phi_st_aligned_train=phi_st_aligned_train,
        st_types_train=st_types_train,
        sc_types_train=sc_types_train,
        idx_anchor_use=idx_anchor_use,
        anchor_valid=anchor_valid,
        model=model,
        idx_sc_train=idx_sc_train,
        idx_st_train=idx_st_train,
        direction=model.direction,
        source_name=model.source_name,
        target_name=model.target_name,
    )
