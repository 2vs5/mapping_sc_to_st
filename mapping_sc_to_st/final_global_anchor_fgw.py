from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import time
import numpy as np

from .array_ops import to_dense as _to_dense_arr
from .keys import (
    OBSM_ST_UMAP,
    OBS_GLOBAL_BEST_CELL,
    OBS_FINAL_SCORE,
    OBS_FINAL_TYPE,
    OBSM_SC_LATENT,
    OBS_FINAL_GLOBAL_TYPE,
    OBS_FINAL_GLOBAL_SCORE,
    UNS_GLOBAL_FINAL_WEIGHTS,
    UNS_FINAL_WEIGHTS,
)
from .c_build import build_C_euclidean, build_C_from_knn, _normalize_C
from .transport_engine import run_transport
from .precomp import mixture_scores_from_weights

from .anchors import select_st_anchors_final
from .alignment import (
    compute_phi_st,
    compute_phi_sc,
    compute_squared_cost_from_phi,
    fit_axis_gaussian_calibrator,
    apply_axis_gaussian_calibrator,
    compute_pseudo_anchor_centers_from_weights_A,
)

# direction-aware apply (requires the updated alignment.py I provided earlier)
try:
    from .alignment import apply_axis_gaussian_calibrator_to_source
except Exception: # pragma: no cover
    apply_axis_gaussian_calibrator_to_source = None

# ----------------------------
# helpers
# ----------------------------

def _to_dense(X, *, dtype=None):
    return _to_dense_arr(X, dtype=dtype)

def _infer_type_by_mass(W_sub: np.ndarray, sc_types: np.ndarray):
    sc_types = np.asarray(sc_types).astype(str)
    uniq = np.unique(sc_types)

    n_sc_rows, n_upd = W_sub.shape
    out_type = np.empty(n_upd, dtype=object)
    out_mass = np.zeros(n_upd, dtype=float)

    idx_by_t = {t: np.where(sc_types == t)[0] for t in uniq}

    for k in range(n_upd):
        w = W_sub[:, k]
        best_t = None
        best_m = -1.0
        for t in uniq:
            idx = idx_by_t[t]
            m = float(w[idx].sum())
            if m > best_m:
                best_m = m
                best_t = t
        out_type[k] = best_t
        out_mass[k] = best_m
    return out_type, out_mass

def _get_prev_weights_dict(adata_st, key: str):
    d = adata_st.uns.get(key, {})
    if d is None or not isinstance(d, dict):
        return {}
    return d

def _finite_stats(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    xf = x[np.isfinite(x)]
    if xf.size == 0:
        return {"mean": np.nan, "median": np.nan, "p95": np.nan, "max": np.nan, "min": np.nan, "n": 0}
    return {
        "mean": float(np.mean(xf)),
        "median": float(np.median(xf)),
        "p95": float(np.quantile(xf, 0.95)),
        "max": float(np.max(xf)),
        "min": float(np.min(xf)),
        "n": int(xf.size),
    }

def _fmt_stats(name: str, x: np.ndarray) -> str:
    s = _finite_stats(x)
    return (
        f"{name}: mean={s['mean']:.4g}, med={s['median']:.4g}, "
        f"p95={s['p95']:.4g}, min={s['min']:.4g}, max={s['max']:.4g}, n={s['n']}"
    )

# ----------------------------
# cache & robust parsing
# ----------------------------

def _convert_weights_B_to_A(
    weights_B: dict,
    *,
    n_sc: int,
    eps: float = 1e-12,
) -> dict:
    """
    Robust conversion to A-format for pseudo-centers.

    Accepts mixed per-ST dicts like:
      {st: {0:w0, 1:w1, ..., 'best_type': 'EVL', 'score': 0.7}}
    Or pure B-format:
      {st: {sc_idx: weight, ...}}
    Or already A-format:
      {st: {'sc_idx': [...], 'w': [...]}}

    Returns A-format:
      {st: {'sc_idx': np.ndarray(int), 'w': np.ndarray(float)}}
    """
    out: Dict[int, Dict[str, Any]] = {}
    if not isinstance(weights_B, dict) or len(weights_B) == 0:
        return out

    for st_k, info in weights_B.items():
        try:
            st_idx = int(st_k)
        except (TypeError, ValueError):
            continue
        if info is None:
            continue

 # Already A-format?
        if isinstance(info, dict) and ("sc_idx" in info) and ("w" in info):
            sc_idx = np.asarray(info["sc_idx"], dtype=np.int64)
            w = np.asarray(info["w"], dtype=np.float64)
            if sc_idx.size == 0 or w.size == 0:
                continue
            m = int(min(sc_idx.size, w.size))
            sc_idx = sc_idx[:m]
            w = w[:m]
            ok = (sc_idx >= 0) & (sc_idx < int(n_sc)) & np.isfinite(w) & (w > 0)
            sc_idx = sc_idx[ok]
            w = w[ok]
            if sc_idx.size == 0:
                continue
            s = float(w.sum())
            if not np.isfinite(s) or s <= 0:
                continue
            w = w / (s + float(eps))
            out[st_idx] = {"sc_idx": sc_idx, "w": w}
            continue

 # B-format or mixed dict: pull numeric keys only
        if isinstance(info, dict):
            sc_idx_list = []
            w_list = []
            for k, v in info.items():
                try:
                    sc_i = int(k)
                    ww = float(v)
                except (TypeError, ValueError):
                    continue
                sc_idx_list.append(sc_i)
                w_list.append(ww)
            if len(sc_idx_list) == 0:
                continue
            sc_idx = np.asarray(sc_idx_list, dtype=np.int64)
            w = np.asarray(w_list, dtype=np.float64)

            ok = (sc_idx >= 0) & (sc_idx < int(n_sc)) & np.isfinite(w) & (w > 0)
            if not np.any(ok):
                continue
            sc_idx = sc_idx[ok]
            w = w[ok]
            s = float(w.sum())
            if not np.isfinite(s) or s <= 0:
                continue
            w = w / (s + float(eps))
            out[st_idx] = {"sc_idx": sc_idx, "w": w}
            continue

    return out

def _get_or_build_weights_A_cache(
    adata_st,
    *,
    prev_weights_key: str,
    cache_key: str,
    n_sc: int,
    eps: float = 1e-12,
    force_rebuild: bool = False,
) -> Dict[int, Dict[str, Any]]:
    if (not force_rebuild) and (cache_key in adata_st.uns) and isinstance(adata_st.uns.get(cache_key, None), dict):
        return adata_st.uns[cache_key]

    weights_B = _get_prev_weights_dict(adata_st, str(prev_weights_key))
    weights_A = _convert_weights_B_to_A(weights_B, n_sc=int(n_sc), eps=float(eps))
    adata_st.uns[cache_key] = weights_A
    return weights_A

def _compute_M_anchor_axis_gauss_in_chunks(
    *,
    st_geom_all: np.ndarray,
    idx_update: np.ndarray,
    idx_anchor_use: np.ndarray,
    sc_geom_cand: np.ndarray,
    anchor_centers: np.ndarray,
    st_types_upd: np.ndarray,
    sc_types_cand: np.ndarray,
    use_alignment: bool,
    align_direction: str,
    align_min_per_type: int,
    align_eps: float,
    align_clip_q_low: float,
    align_clip_q_high: float,
    shrink_k: int,
    chunk: int = 2048,
) -> np.ndarray:
    """
    Chunked computation of M_anchor with optional direction-aware alignment.

    - align_direction="sc_to_st": align SC -> ST (default; )
    - align_direction="st_to_sc": align ST -> SC
    """
    idx_update = np.asarray(idx_update, dtype=int)
    idx_anchor_use = np.asarray(idx_anchor_use, dtype=int)

    align_direction = str(align_direction).lower().strip()
    if align_direction not in ("sc_to_st", "st_to_sc"):
        raise ValueError('align_direction must be "sc_to_st" or "st_to_sc"')

 # Precompute Phi_sc once (n_cand, K)
    Phi_sc_raw = compute_phi_sc(sc_geom=sc_geom_cand, anchor_centers=anchor_centers)

 # We'll build M_anchor in (n_cand, n_upd) via identity:
 # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    n_cand = int(Phi_sc_raw.shape[0])
    n_upd = int(idx_update.size)

 # baseline norms
    Phi_sc = Phi_sc_raw.astype(np.float32, copy=False)
    Phi_sc_norm2 = (Phi_sc ** 2).sum(axis=1).astype(np.float32)

    M_anchor = np.empty((n_cand, n_upd), dtype=np.float32)

    for start in range(0, n_upd, int(max(1, chunk))):
        end = int(min(n_upd, start + int(max(1, chunk))))
        idx_chunk = idx_update[start:end]

        Phi_st_raw = compute_phi_st(st_geom_all=st_geom_all, idx_query=idx_chunk, idx_anchor=idx_anchor_use)
        Phi_st = Phi_st_raw.astype(np.float32, copy=False)

        if use_alignment:
            model = fit_axis_gaussian_calibrator(
                phi_st_train=Phi_st_raw,
                st_types_train=st_types_upd[start:end],
                phi_sc_train=Phi_sc_raw,
                sc_types_train=sc_types_cand,
                anchors=idx_anchor_use,
                direction=align_direction,
                min_n=int(align_min_per_type),
                eps=float(align_eps),
                shrink_k=int(shrink_k),
                clip_q=(float(align_clip_q_low), float(align_clip_q_high)),
            )

            if str(model.direction) == "sc_to_st":
                Phi_sc = apply_axis_gaussian_calibrator(phi_sc=Phi_sc_raw, sc_types=sc_types_cand, model=model)
                Phi_sc = Phi_sc.astype(np.float32, copy=False)
                Phi_sc_norm2 = (Phi_sc ** 2).sum(axis=1).astype(np.float32)
                Phi_st = Phi_st_raw.astype(np.float32, copy=False)

            else: # st_to_sc
                if apply_axis_gaussian_calibrator_to_source is None:
                    raise ImportError(
                        "apply_axis_gaussian_calibrator_to_source is required for direction='st_to_sc'. "
                        "Update alignment.py to the direction-aware version."
                    )
                Phi_st = apply_axis_gaussian_calibrator_to_source(
                    phi_source=Phi_st_raw,
                    source_types=st_types_upd[start:end],
                    model=model,
                    inplace=False,
                ).astype(np.float32, copy=False)
                Phi_sc = Phi_sc_raw.astype(np.float32, copy=False)
                Phi_sc_norm2 = (Phi_sc ** 2).sum(axis=1).astype(np.float32)

        b2 = (Phi_st ** 2).sum(axis=1).astype(np.float32)
        G = (Phi_sc @ Phi_st.T).astype(np.float32)
        Mc = Phi_sc_norm2[:, None] + b2[None, :] - 2.0 * G
        np.maximum(Mc, 0.0, out=Mc)
        M_anchor[:, start:end] = Mc

    return M_anchor

def _compute_pairwise_like_score_and_imputed(
    *,
    W: np.ndarray,
    idx_sc_cand: np.ndarray,
    idx_update: np.ndarray,
    precomp: dict,
    cos_threshold=None,
):
    """
    pairwise   score .
     mixture_scores_from_weights 2 (imputed) ,
        W.T @ X_sc_exp .

    Returns
    -------
    score_sub : (n_upd,) float
    X_imputed_all : (n_upd, G) float32
    """
    out = mixture_scores_from_weights(
        W,
        idx_sc_cand,
        idx_update,
        precomp,
        gene_mask=None,
        cos_threshold=cos_threshold,
        n_jobs=1,
    )

    if isinstance(out, tuple) and len(out) >= 1:
        score_sub = np.asarray(out[0], dtype=float)
        X_hat = out[1] if len(out) > 1 else None
    else:
        score_sub = np.asarray(out, dtype=float)
        X_hat = None

    n_upd = int(idx_update.size)
    if score_sub.shape[0] != n_upd:
        raise ValueError(f"score_sub has wrong length: {score_sub.shape[0]} != {n_upd}")

    X_imputed_all = None
    if X_hat is not None:
        X_hat = np.asarray(X_hat)
        if X_hat.ndim == 2 and X_hat.shape[0] == n_upd:
            X_imputed_all = X_hat.astype(np.float32, copy=False)

    if X_imputed_all is None:
        X_sc_exp = np.asarray(precomp["X_sc_exp"], dtype=np.float32)
        X_imputed_all = (W.T @ X_sc_exp[idx_sc_cand]).astype(np.float32)

    return score_sub, X_imputed_all

def _store_imputation(
    *,
    adata_st,
    X_imputed_all: np.ndarray,
    idx_apply: np.ndarray,
    apply_sub: np.ndarray,
    n_st: int,
    obsm_key: str,
    uns_key: str,
    store_dense: bool,
    store_dict: bool,
):
    """
    apply  spot . (   )
    """
    if idx_apply.size > 0:
        X_imputed = X_imputed_all[apply_sub]
    else:
        X_imputed = np.zeros((0, X_imputed_all.shape[1]), dtype=np.float32)

    if bool(store_dense):
        adata_st.obsm[obsm_key] = np.zeros((n_st, X_imputed_all.shape[1]), dtype=np.float32)
        if idx_apply.size > 0:
            adata_st.obsm[obsm_key][idx_apply] = X_imputed

    if bool(store_dict):
        adata_st.uns[uns_key] = {int(idx_apply[ii]): X_imputed[ii] for ii in range(idx_apply.size)}

@dataclass
class GlobalAnchorFGWConfig:
    anchor_score_key: str = OBS_FINAL_SCORE
    update_score_key: str = OBS_FINAL_SCORE
    update_type_key: str = OBS_FINAL_TYPE

    tau_anchor: float = 0.75
    tau_update: float = 0.60

 # geometry / C builders
    sc_C_mode: str = "euclidean" # "euclidean" | "knn_dense" | "knn_geodesic"
    st_C_mode: str = "euclidean" # "euclidean" | "knn_dense" | "knn_geodesic"
    C_normalize: str = "q95" # for euclidean builder
    knn_sc_key_added: str = "neighbors_sc"
    knn_st_key_added: str = "neighbors_st"
    knn_fill_value: str = "max" # for knn_dense
    knn_symmetrize: bool = True
    knn_directed: bool = False # for knn_geodesic

    anchor_cap: int = 20000
    min_anchors: int = 10

 # --- stratified diverse anchor selection ---
    anchor_select_mode: str = "stratified_grid" # "threshold" | "stratified_grid"
    anchor_k_per_type: int = 30 # anchors per cell type (if available)
    anchor_grid_bins: int = 12 # grid resolution for diversity (spatial)
    anchor_min_per_type: int = 5 # ensure at least this many per type if possible

    sc_feat_key: Optional[str] = None
    st_feat_key: Optional[str] = None
    sc_geom_key: str = OBSM_SC_LATENT
    st_geom_key: str = OBSM_ST_UMAP

    beta_expr: float = 0.7

 # transport normalization / C adaptation (kept in sync with pairwise)
    M_norm: str = "mean"
    C_kind_sc: str = "distance"
    C_kind_st: str = "distance"
    C_norm: str = "mean"
    C_clip_q: Optional[float] = None

 # extra kwargs forwarded into solver.solve(...) via run_transport
    solver_kwargs: Optional[Dict[str, Any]] = None

 # allow anchors to be updated if they are low-score
    allow_anchor_update: bool = True

 # prev weights for pseudo-anchor
    prev_weights_key: str = UNS_FINAL_WEIGHTS

 # cache key for A-format conversion
    prev_weights_cache_key: str = UNS_FINAL_WEIGHTS + "__A_cache"

 # pseudo-center parallel compute
    centers_parallel: bool = True
    centers_n_jobs: int = 128
    centers_prefer: str = "threads"
    centers_batch_size: int = 256
    centers_force_rebuild_cache: bool = False

 # anchor cost mode + temperature for weighted
    anchor_sig_chunk: int = 2048 # chunk size over ST updates when building signature cost

 # diagnostics
    log_cost_stats: bool = True
    log_cost_col_stats: bool = False # if True, logs mean per column too (can be verbose)

 # --- FINAL-only alignment: type-wise, per-anchor-axis Gaussian calibration ---
    use_alignment: bool = True
    alignment_store_key: str = "final_axis_gauss_calib" # stored in adata_st.uns
    align_min_per_type: int = 30
    align_eps: float = 1e-8
    align_clip_q_low: float = 0.01
    align_clip_q_high: float = 0.99
    alignment_shrink_k: int = 20000 # shrink per-type scale toward global

 #  alignment direction
    alignment_direction: str = "sc_to_st" # "sc_to_st" | "st_to_sc"

 # outputs
    out_weights_key: str = UNS_GLOBAL_FINAL_WEIGHTS
    out_type_key: str = OBS_FINAL_GLOBAL_TYPE
    out_updated_mask_key: str = "final_global_anchor_fgw_updated"
    out_mass_key: str = "final_global_anchor_fgw_mass"

    out_score_key: str = OBS_FINAL_GLOBAL_SCORE
    out_score_raw_key: str = "final_anchor_fgw_score_raw"
    out_improved_mask_key: str = "final_anchor_fgw_improved"

    out_imputed_obsm_key: str = "X_imputed_final_global_anchor"
    out_imputed_uns_key: str = "imputed_final_global_anchor"

# ----------------------------
# main
# ----------------------------

def run_final_global_anchor_fgw_update(
    adata_sc,
    adata_st,
    *,
    sc_type_key: str = "cell_type",
    cfg: GlobalAnchorFGWConfig = GlobalAnchorFGWConfig(),
    solver=None,
    precomp: Dict[str, Any] = None,
    incoming_obj: Dict[str, Any] = None,
    use_incoming_genes: bool = True,
    cos_threshold=None,
    update_only_if_improved: bool = True,
    compute_imputation: bool = True,
    store_imputation_dense: bool = True,
    store_imputation_dict: bool = False,
    verbose: bool = True,
    logger=print,
) -> Dict[str, Any]:

    t_all = time.perf_counter()

    def _log(msg: str):
        if bool(verbose):
            logger(msg)

    def _sec(dt: float) -> str:
        return f"{dt*1000:.1f} ms" if dt < 1.0 else f"{dt:.2f} s"

    _log("[final_global_anchor_fgw] Step 0/12: Start + validate inputs")

    if solver is None:
        raise ValueError("solver is required (e.g., POTFGWSolver(...)).")
    if precomp is None:
        raise ValueError("precomp is required.")

    for k in ("X_sc_exp", "X_st_exp", "genes_union", "baseline_sc", "baseline_st", "G", "eps"):
        if k not in precomp:
            raise KeyError(f"[final_global_anchor_fgw] precomp missing '{k}'")

    if cfg.update_type_key not in adata_st.obs:
        raise KeyError(f"[final_global_anchor_fgw] missing adata_st.obs['{cfg.update_type_key}']")

    for k in (cfg.anchor_score_key, cfg.update_score_key, OBS_GLOBAL_BEST_CELL):
        if k not in adata_st.obs:
            raise KeyError(f"[final_global_anchor_fgw] missing adata_st.obs['{k}']")

    if cfg.st_geom_key not in adata_st.obsm:
        raise KeyError(f"[final_global_anchor_fgw] missing adata_st.obsm['{cfg.st_geom_key}']")

    if cfg.sc_geom_key is not None and cfg.sc_geom_key not in adata_sc.obsm:
        raise KeyError(f"[final_global_anchor_fgw] missing adata_sc.obsm['{cfg.sc_geom_key}']")

    if sc_type_key not in adata_sc.obs:
        raise KeyError(f"[final_global_anchor_fgw] missing adata_sc.obs['{sc_type_key}']")

    s_anchor = np.asarray(adata_st.obs[cfg.anchor_score_key], dtype=float)
    s_update = np.asarray(adata_st.obs[cfg.update_score_key], dtype=float)

    n_st = adata_st.n_obs
    n_sc = adata_sc.n_obs

 # geometry arrays used throughout (avoid repeated .obsm lookups and ensure dtype)
    st_geom_all = np.asarray(adata_st.obsm[str(cfg.st_geom_key)], dtype=float)
    sc_geom_all = np.asarray(adata_sc.obsm[str(cfg.sc_geom_key)], dtype=float)

    if st_geom_all.shape[0] != int(n_st):
        raise ValueError(f"[final_global_anchor_fgw] st_geom_all has wrong n_obs: {st_geom_all.shape[0]} != {n_st}")
    if sc_geom_all.shape[0] != int(n_sc):
        raise ValueError(f"[final_global_anchor_fgw] sc_geom_all has wrong n_obs: {sc_geom_all.shape[0]} != {n_sc}")

    _log(f"[final_global_anchor_fgw]  n_sc={n_sc}, n_st={n_st}")
    _log(f"[final_global_anchor_fgw]  tau_anchor={cfg.tau_anchor}, tau_update={cfg.tau_update}, beta_expr={cfg.beta_expr}")
    _log(f"[final_global_anchor_fgw]  allow_anchor_update={cfg.allow_anchor_update}")
    _log(f"[final_global_anchor_fgw]  prev_weights_key='{cfg.prev_weights_key}'")
    _log(f"[final_global_anchor_fgw]  alignment: use={cfg.use_alignment}, direction='{cfg.alignment_direction}'")

    _log("[final_global_anchor_fgw] Step 1/12: Initialize output columns (type snapshot -> working output)")
    adata_st.obs[cfg.out_type_key] = adata_st.obs[cfg.update_type_key].astype(object).copy()
    adata_st.obs[cfg.out_score_key] = np.asarray(adata_st.obs[cfg.update_score_key], dtype=float).copy()
    adata_st.obs[cfg.out_score_raw_key] = np.asarray(adata_st.obs[cfg.update_score_key], dtype=float).copy()

 # anchor indices
    _log("[final_global_anchor_fgw] Step 2/12: Select anchors")
    if str(cfg.anchor_select_mode) == "stratified_grid":
        idx_anchor, anchors_by_type = select_st_anchors_final(
            adata_st,
            st_geom_key=str(cfg.st_geom_key),
            st_type_key=str(cfg.update_type_key),
            st_score_key=str(cfg.anchor_score_key),
            n_anchor_per_type=int(cfg.anchor_k_per_type),
            top_frac_per_type=0.6,
            min_per_type=int(cfg.anchor_min_per_type),
            diversity_bins=int(cfg.anchor_grid_bins),
            diversity_min_per_grid=1,
        )
    else:
        idx_anchor = np.where((s_anchor >= float(cfg.tau_anchor)) & np.isfinite(s_anchor))[0].astype(int)
        anchors_by_type = {}

    if idx_anchor.size > int(cfg.anchor_cap):
        idx_anchor = idx_anchor[: int(cfg.anchor_cap)]

    if idx_anchor.size < int(cfg.min_anchors):
        raise ValueError(f"[final_global_anchor_fgw] too few anchors: {idx_anchor.size} < {cfg.min_anchors}")
    _log(f"[final_global_anchor_fgw]  anchors={idx_anchor.size}")

 # update indices
    _log("[final_global_anchor_fgw] Step 3/12: Select update spots")
    idx_update = np.where((s_update < float(cfg.tau_update)) & np.isfinite(s_update))[0].astype(int)
    if not bool(cfg.allow_anchor_update):
 # exclude anchors from updates
        m = np.ones(adata_st.n_obs, dtype=bool)
        m[idx_anchor] = False
        idx_update = idx_update[m[idx_update]]
    _log(f"[final_global_anchor_fgw]  updates={idx_update.size}")

    if idx_update.size == 0:
        _log("[final_global_anchor_fgw]  no updates -> return early")
        return {"idx_anchor": idx_anchor, "idx_update": idx_update}

 # candidate SC set (best-cells + all? currently use all)
    _log("[final_global_anchor_fgw] Step 4/12: Define SC candidates")
    idx_sc_cand = np.arange(n_sc, dtype=int)
    _log(f"[final_global_anchor_fgw]  sc_candidates={idx_sc_cand.size}")

 # features for expression cost
    _log("[final_global_anchor_fgw] Step 5/12: Prepare expression features (M_expr built in run_transport)")
    t = time.perf_counter()

    X_sc_feat_all = precomp["X_sc_exp"]
    X_st_feat_all = precomp["X_st_exp"]
    X_sc_feat = np.asarray(X_sc_feat_all[idx_sc_cand], dtype=np.float32)
    X_st_sub = np.asarray(X_st_feat_all[idx_update], dtype=np.float32)

 # (7) pseudo-anchor cost M_anchor (cached)
    t = time.perf_counter()
    _log("[final_global_anchor_fgw] Step 7/12: Build pseudo-anchor cost M_anchor from prev weights (cached A-format)")

    weights_A = _get_or_build_weights_A_cache(
        adata_st,
        prev_weights_key=cfg.prev_weights_key,
        cache_key=cfg.prev_weights_cache_key,
        n_sc=n_sc,
        eps=1e-12,
        force_rebuild=bool(cfg.centers_force_rebuild_cache),
    )

 # pseudo-centers
    anchor_centers, anchor_valid = compute_pseudo_anchor_centers_from_weights_A(
        sc_geom_all=sc_geom_all,
        idx_anchor_st=idx_anchor,
        weights_A=weights_A,
    )

    _log(f"[final_global_anchor_fgw]  valid pseudo-centers: {int(anchor_valid.sum())}/{idx_anchor.size}")

 # --- Anchor-distance signatures (phi) ---
    valid_loc = np.flatnonzero(anchor_valid)
    if valid_loc.size < int(max(1, cfg.min_anchors)):
        raise ValueError(
            f"[final_global_anchor_fgw] too few valid anchors with pseudo-centers: {int(valid_loc.size)} < {int(cfg.min_anchors)}"
        )

    idx_anchor_use = np.asarray(idx_anchor, dtype=int)[valid_loc]
    centers_use = np.asarray(anchor_centers, dtype=float)[valid_loc]

 # Phi_st for update spots (ST reference)
    Phi_st_upd = compute_phi_st(
        st_geom_all=st_geom_all,
        idx_query=idx_update,
        idx_anchor=idx_anchor_use,
    ) # (n_upd, K)

 # Phi_sc for SC candidates (raw, in SC geometry)
    Phi_sc_raw = compute_phi_sc(
        sc_geom=np.asarray(sc_geom_all[idx_sc_cand], dtype=float),
        anchor_centers=centers_use,
    ) # (n_sc_cand, K)

    sc_types_cand = np.asarray(adata_sc.obs[sc_type_key], dtype=object).astype(str)[idx_sc_cand]
    st_types_upd = np.asarray(adata_st.obs[cfg.update_type_key], dtype=object).astype(str)[idx_update]

 # direction-aware alignment (SOURCE -> TARGET)
    Phi_sc = Phi_sc_raw.astype(np.float32, copy=False)
    Phi_st = Phi_st_upd.astype(np.float32, copy=False)

    if bool(cfg.use_alignment):
        align_dir = str(cfg.alignment_direction).lower().strip()
        if align_dir not in ("sc_to_st", "st_to_sc"):
            raise ValueError('cfg.alignment_direction must be "sc_to_st" or "st_to_sc"')

        model_calib = fit_axis_gaussian_calibrator(
            phi_st_train=Phi_st_upd,
            st_types_train=st_types_upd,
            phi_sc_train=Phi_sc_raw,
            sc_types_train=sc_types_cand,
            anchors=idx_anchor_use,
            direction=align_dir, #
            min_n=int(cfg.align_min_per_type),
            eps=float(cfg.align_eps),
            shrink_k=int(cfg.alignment_shrink_k),
            clip_q=(float(cfg.align_clip_q_low), float(cfg.align_clip_q_high)),
        )

        if str(model_calib.direction) == "sc_to_st":
 # SC -> ST : transform SC
            Phi_sc = apply_axis_gaussian_calibrator(
                phi_sc=Phi_sc_raw,
                sc_types=sc_types_cand,
                model=model_calib,
            ).astype(np.float32, copy=False)
            Phi_st = Phi_st_upd.astype(np.float32, copy=False)

        else:
 # ST -> SC : transform ST
            if apply_axis_gaussian_calibrator_to_source is None:
                raise ImportError(
                    "apply_axis_gaussian_calibrator_to_source is required for direction='st_to_sc'. "
                    "Update alignment.py to the direction-aware version."
                )
            Phi_st = apply_axis_gaussian_calibrator_to_source(
                phi_source=Phi_st_upd,
                source_types=st_types_upd,
                model=model_calib,
                inplace=False,
            ).astype(np.float32, copy=False)
            Phi_sc = Phi_sc_raw.astype(np.float32, copy=False)

 # Store model for QC / reproducibility
        try:
            payload = {
                "direction": str(getattr(model_calib, "direction", "sc_to_st")),
                "source_name": str(getattr(model_calib, "source_name", "sc")),
                "target_name": str(getattr(model_calib, "target_name", "st")),
                "anchors": idx_anchor_use.astype(int).tolist(),
                "types_seen": np.asarray(model_calib.types_seen, dtype=object).tolist(),
                "alpha_global": model_calib.alpha_global.astype(np.float32),
                "beta_global": model_calib.beta_global.astype(np.float32),
                "alpha_type": {k: v.astype(np.float32) for k, v in model_calib.alpha_type.items()},
                "beta_type": {k: v.astype(np.float32) for k, v in model_calib.beta_type.items()},
                "clip_q": model_calib.clip_q,
            }
 # optional: store clipping bounds too (recommended)
            if getattr(model_calib, "clip_lo_global", None) is not None and getattr(model_calib, "clip_hi_global", None) is not None:
                payload["clip_lo_global"] = model_calib.clip_lo_global.astype(np.float32)
                payload["clip_hi_global"] = model_calib.clip_hi_global.astype(np.float32)
            if getattr(model_calib, "clip_lo_type", None) is not None and getattr(model_calib, "clip_hi_type", None) is not None:
                payload["clip_lo_type"] = {k: v.astype(np.float32) for k, v in model_calib.clip_lo_type.items()}
                payload["clip_hi_type"] = {k: v.astype(np.float32) for k, v in model_calib.clip_hi_type.items()}
            adata_st.uns[str(cfg.alignment_store_key)] = payload
        except Exception as e:
            _log(f"[final_global_anchor_fgw] warning: failed to store alignment calib in adata_st.uns: {e}")

 # Build M_anchor(i,j) = ||Phi_sc(i) - Phi_st(j)||^2
    M_anchor = compute_squared_cost_from_phi(phi_sc=Phi_sc, phi_st=Phi_st).astype(np.float32, copy=False)
    _log(f"[final_global_anchor_fgw]  M_anchor(axis_gauss) shape={M_anchor.shape} | time={_sec(time.perf_counter()-t)}")
    _log("[final_global_anchor_fgw] Step 8/12: (centralized) M will be built in run_transport")

    if bool(cfg.log_cost_stats):
        _log("[final_global_anchor_fgw]  cost stats (raw):")
        _log("  " + _fmt_stats("M_anchor(raw)", M_anchor))
        if bool(cfg.log_cost_col_stats):
            _log("  M_anchor(raw) col-mean: " + _fmt_stats("colmean", np.mean(M_anchor, axis=0)))
    _log(f"[final_global_anchor_fgw]  time={_sec(time.perf_counter()-t)}")

 # geometry
    t = time.perf_counter()
    _log("[final_global_anchor_fgw] Step 9/12: Build geometry costs C_sc, C_st (raw)")

    if str(cfg.sc_C_mode).startswith("knn"):
        mode = "geodesic" if str(cfg.sc_C_mode).endswith("geodesic") else "dense"
        C_sc_raw = build_C_from_knn(
            adata_sc,
            idx=idx_sc_cand,
            key_added=str(cfg.knn_sc_key_added),
            mode=mode,
            fill_value=str(cfg.knn_fill_value),
            symmetrize=bool(cfg.knn_symmetrize),
            directed=bool(cfg.knn_directed),
        )
    elif str(cfg.sc_C_mode) == "metacell_distance":
        C_sc_raw = _normalize_C(adata_sc.obsp['mf_distance_matrix'], normalize =None)

    else:
        C_sc_raw = build_C_euclidean(sc_geom_all[idx_sc_cand], metric="euclidean", normalize=None)

    if str(cfg.st_C_mode).startswith("knn"):
        mode = "geodesic" if str(cfg.st_C_mode).endswith("geodesic") else "dense"
        C_st_raw = build_C_from_knn(
            adata_st,
            idx=idx_update,
            key_added=str(cfg.knn_st_key_added),
            mode=mode,
            fill_value=str(cfg.knn_fill_value),
            symmetrize=bool(cfg.knn_symmetrize),
            directed=bool(cfg.knn_directed),
        )
    else:
        C_st_raw = build_C_euclidean(st_geom_all[idx_update], metric="euclidean", normalize=None)

    _log(f"[final_global_anchor_fgw]  C_sc_raw shape={C_sc_raw.shape}, C_st_raw shape={C_st_raw.shape} | time={_sec(time.perf_counter()-t)}")

    _log("[final_global_anchor_fgw] Step 10/12: Run transport (FGW)")

    rr = run_transport(
        solver=solver,
        X_sc_feat=X_sc_feat,
        X_st_feat=X_st_sub,
        C_sc_raw=C_sc_raw,
        C_st_raw=C_st_raw,
        M_anchor=M_anchor,
        beta_expr=float(cfg.beta_expr),
        M_norm=str(cfg.M_norm),
        C_kind_sc=str(cfg.C_kind_sc),
        C_kind_st=str(cfg.C_kind_st),
        C_norm=str(cfg.C_norm),
        C_clip_q=cfg.C_clip_q,
        **(cfg.solver_kwargs or {}),
    )

    W = rr.W
    _log(f"[final_global_anchor_fgw]  transport done: W shape={W.shape}")

    _log("[final_global_anchor_fgw] Step 11/12: Update types/scores based on W")
    sc_types_all = np.asarray(adata_sc.obs[sc_type_key], dtype=object).astype(str)
    new_type_sub, mass_sub = _infer_type_by_mass(W, sc_types_all[idx_sc_cand])

    score_sub, X_imputed_all = _compute_pairwise_like_score_and_imputed(
        W=W,
        idx_sc_cand=idx_sc_cand,
        idx_update=idx_update,
        precomp=precomp,
        cos_threshold=cos_threshold,
    )

    prev_score = np.asarray(adata_st.obs[cfg.update_score_key], dtype=float)
    improved_sub = score_sub > prev_score[idx_update]

    improved_mask = np.zeros(n_st, dtype=bool)
    improved_mask[idx_update] = improved_sub
    adata_st.obs[cfg.out_improved_mask_key] = improved_mask

    apply_sub = improved_sub if bool(update_only_if_improved) else np.ones(idx_update.size, dtype=bool)
    idx_apply = idx_update[apply_sub]

    adata_st.obs[cfg.out_mass_key] = 0.0
    if idx_apply.size > 0:
        adata_st.obs[cfg.out_mass_key].iloc[idx_apply] = mass_sub[apply_sub]
        adata_st.obs[cfg.out_type_key].iloc[idx_apply] = new_type_sub[apply_sub]

    raw_score = np.asarray(adata_st.obs[cfg.out_score_raw_key], dtype=float)
    raw_score2 = raw_score.copy()
    raw_score2[idx_update] = score_sub
    adata_st.obs[cfg.out_score_raw_key] = raw_score2

    if idx_apply.size > 0:
        adata_st.obs[cfg.out_score_key].iloc[idx_apply] = score_sub[apply_sub]

    updated_mask = np.zeros(n_st, dtype=bool)
    updated_mask[idx_apply] = True
    adata_st.obs[cfg.out_updated_mask_key] = updated_mask

    _log("[final_global_anchor_fgw] Step 12/12: Store weights + (optional) imputation")

    weights_dict = {}
    for kk, j in enumerate(idx_update):
        if not bool(apply_sub[kk]):
            continue
        col = W[:, kk]
        dd = {int(idx_sc_cand[ii]): float(col[ii]) for ii in range(col.size) if float(col[ii]) > 0.0}
        weights_dict[int(j)] = dd
    adata_st.uns[cfg.out_weights_key] = weights_dict

    if bool(compute_imputation):
        _store_imputation(
            adata_st=adata_st,
            X_imputed_all=X_imputed_all,
            idx_apply=idx_apply,
            apply_sub=apply_sub,
            n_st=n_st,
            obsm_key=cfg.out_imputed_obsm_key,
            uns_key=cfg.out_imputed_uns_key,
            store_dense=store_imputation_dense,
            store_dict=store_imputation_dict,
        )

    _log(f"[final_global_anchor_fgw] Done. total time={_sec(time.perf_counter()-t_all)}")

    return {
        "idx_anchor": idx_anchor,
        "idx_update": idx_update,
        "idx_sc_cand": idx_sc_cand,
        "W": W,
        "M_anchor": M_anchor,
        "C_sc_raw": C_sc_raw,
        "C_st_raw": C_st_raw,
        "out_weights_key": cfg.out_weights_key,
        "out_type_key": cfg.out_type_key,
        "out_score_key": cfg.out_score_key,
        "out_imputed_obsm_key": cfg.out_imputed_obsm_key if compute_imputation and store_imputation_dense else None,
        "out_imputed_uns_key": cfg.out_imputed_uns_key if compute_imputation and store_imputation_dict else None,
        "out_improved_mask_key": cfg.out_improved_mask_key,
        "out_updated_mask_key": cfg.out_updated_mask_key,
        "n_improved": int(improved_sub.sum()),
    }
