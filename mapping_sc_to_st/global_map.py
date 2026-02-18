# util/global_map.py
"""
Global 1:1 mapping (SC cell -> ST spot) using:
    sim(i, j) = 0.5 * cosine(exp_sc_i, exp_st_j)
             + 0.5 * corr(FC_sc_i, FC_st_j)

where corr is selectable:
  - "spearman": Spearman rank correlation on FC vectors
  - "pearson": Pearson correlation on FC vectors

Outputs (canonical keys; defined in util.keys):
- adata_st.obs[OBS_GLOBAL_SIM]
- adata_st.obs[OBS_GLOBAL_TYPE]
- adata_st.obs[OBS_GLOBAL_BEST_CELL]       (optional)
- adata_st.obs[OBS_GLOBAL_BEST_CELL_TYPE]  (optional)
- adata_st.uns[UNS_GLOBAL_GENES_UNION]

Anchors are assigned later in util/anchors.py.
"""

import hashlib
import warnings
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from joblib import Parallel, delayed

from .array_ops import to_dense as _to_dense
from .array_ops import row_l2_normalize as _row_l2_normalize

from .keys import (
    OBS_GLOBAL_SIM,
    OBS_GLOBAL_TYPE,
    OBS_GLOBAL_BEST_CELL,
    OBS_GLOBAL_BEST_CELL_IDX,
    OBS_GLOBAL_BEST_CELL_TYPE,
    UNS_GLOBAL_GENES_UNION,
 # unified weights container shared with pairwise refinement
    UNS_FINAL_WEIGHTS,
    OBS_GLOBAL_ANCHOR,
    OBS_GLOBAL_ANCHOR_TYPE,
    OBS_SC_IS_ANCHOR,
    OBS_SC_ANCHOR_N_SPOTS,
    OBS_SC_ANCHOR_MAX_SCORE,
)

def to_dense(X, *, dtype=None):
    """Backwards-compatible wrapper (prefer util.array_ops.to_dense)."""
    return _to_dense(X, dtype=dtype)

def _rank_center_std_per_row(X, ddof=1, eps=1e-12, *, dtype=np.float32):
    """
    For each row x:
      r = rankdata(x)
      rc = r - mean(r)
      rs = std(rc)
    Return:
      rc: (n, G)
      rs: (n,)
    """
    n, G = X.shape
    R = np.empty((n, G), dtype=dtype)
    for i in range(n):
        R[i] = rankdata(X[i]).astype(dtype, copy=False) # ties -> average ranks

    Rm = R.mean(axis=1, keepdims=True)
    Rc = R - Rm
    Rs = Rc.std(axis=1, ddof=ddof)
    Rs = np.where((Rs > eps) & np.isfinite(Rs), Rs, 1.0)
    return Rc.astype(dtype, copy=False), Rs.astype(dtype, copy=False)

def _center_std_per_row(X, ddof=1, eps=1e-12, *, dtype=np.float32):
    """For each row x, return centered values and row-wise std.

    Used for Pearson correlation on FC vectors.
    """
    X = np.asarray(X)
    Xm = X.mean(axis=1, keepdims=True)
    Xc = (X - Xm).astype(dtype, copy=False)
    Xs = Xc.std(axis=1, ddof=ddof)
    Xs = np.where((Xs > eps) & np.isfinite(Xs), Xs, 1.0)
    return Xc, Xs.astype(dtype, copy=False)

def _extract_genes_union(selected_genes, adata_sc, adata_st, sort_genes=True):
    genes = set()

    if isinstance(selected_genes, dict):
        for _, obj in selected_genes.items():
            if hasattr(obj, "index") and not callable(obj.index):
                genes.update(list(obj.index))
            elif isinstance(obj, (list, tuple, set, np.ndarray)):
                genes.update(list(obj))
            elif hasattr(obj, "tolist") and callable(obj.tolist):
                genes.update(list(obj.tolist()))
            else:
                raise TypeError(f"Unsupported selected_genes dict value type: {type(obj)}")
    elif isinstance(selected_genes, (list, tuple, set, np.ndarray)):
        genes.update(list(selected_genes))
    elif hasattr(selected_genes, "tolist") and callable(selected_genes.tolist):
        genes.update(list(selected_genes.tolist()))
    else:
        raise TypeError(f"Unsupported selected_genes type: {type(selected_genes)}")

 # overlap filter
    genes = [g for g in genes if (g in adata_sc.var_names) and (g in adata_st.var_names)]
    if len(genes) == 0:
        raise ValueError("[global_map] genes_union is empty after overlap filter.")

    if sort_genes:
        genes = np.array(sorted(genes), dtype=object)
    else:
        genes = np.array(list(genes), dtype=object)
    return genes

# ============================================================
# precomp
# ============================================================

def prepare_global_precomp(
    adata_sc,
    adata_st,
    selected_genes=None,
    *,
 # gene panel
    genes_union=None,
    groupby_sc="cell_type",
    sort_genes=True,

 # what to include
    include_X_st_exp=True,
    include_X_sc_fc=True,
    precompute_sc_rank=True,

 # numerics
    dtype=np.float32,
    eps=1e-12,

 # FC correlation
    corr_method: str = "spearman", # "spearman" | "pearson"

 # caching
    cache_key="st_rank_cache",
):
    """
    Unified global precomp (compatible with scoring-precomp).

    Always returns:
      genes_union
      X_sc_exp
      X_st_exp_normed
      baseline_sc
      baseline_st
      X_st_rank_center, X_st_rank_std
      sc_obs_names, sc_celltypes
      G

    Optionally returns:
      X_st_exp (include_X_st_exp=True)
      X_sc_fc  (include_X_sc_fc=True)
      X_sc_rank_center, X_sc_rank_std (precompute_sc_rank=True)
    """

 # -------------------------
 # -------------------------
    if genes_union is None:
        if selected_genes is None:
            raise ValueError("Provide selected_genes or genes_union.")
        genes_union = _extract_genes_union(
            selected_genes, adata_sc, adata_st, sort_genes=sort_genes
        )
    else:
        genes_union = np.asarray(list(map(str, genes_union)), dtype=object)
        if sort_genes:
            genes_union = np.asarray(sorted(genes_union), dtype=object)

    if len(genes_union) == 0:
        raise ValueError("[prepare_global_precomp] genes_union is empty.")

 # genes_list = list(map(str, genes_union.tolist()))
 # require_unique_list(genes_list, name="genes_union", where="[prepare_global_precomp]")
 # missing_sc = [g for g in genes_list if g not in adata_sc.var_names]
 # missing_st = [g for g in genes_list if g not in adata_st.var_names]
 # if missing_sc or missing_st:
 #     raise KeyError(f"missing genes: sc={len(missing_sc)}, st={len(missing_st)}")

 # -------------------------
 # 1) expression matrices
 # -------------------------
    X_sc_exp = to_dense(adata_sc[:, genes_union].X, dtype=dtype)
    X_st_exp = to_dense(adata_st[:, genes_union].X, dtype=dtype)

    X_st_exp_normed, _ = _row_l2_normalize(X_st_exp, eps=eps)

    baseline_sc = X_sc_exp.mean(axis=0)
    baseline_st = X_st_exp.mean(axis=0)

    corr_method = str(corr_method).lower().strip()
    if corr_method not in ("spearman", "pearson"):
        raise ValueError("corr_method must be 'spearman' | 'pearson'")

 # -------------------------
 # 2) ST FC-precompute (cached per gene panel)
 # -------------------------
    genes_sig = hashlib.sha1("|".join(map(str, genes_union)).encode("utf-8")).hexdigest()
    cache = adata_st.uns.setdefault(cache_key, {})

    hit = cache.get(genes_sig, None)
    if hit is not None:
        X_st_rank_center = hit.get("X_st_rank_center", None)
        X_st_rank_std = hit.get("X_st_rank_std", None)
        X_st_fc_center = hit.get("X_st_fc_center", None)
        X_st_fc_std = hit.get("X_st_fc_std", None)
    else:
        X_st_rank_center = X_st_rank_std = None
        X_st_fc_center = X_st_fc_std = None

    if corr_method == "spearman":
        if X_st_rank_center is None or X_st_rank_std is None:
            X_st_fc = X_st_exp - baseline_st[None, :]
            X_st_rank_center, X_st_rank_std = _rank_center_std_per_row(
                X_st_fc, ddof=1, eps=eps, dtype=dtype
            )
        cache[genes_sig] = {
            "genes_union": genes_union,
            "X_st_rank_center": X_st_rank_center,
            "X_st_rank_std": X_st_rank_std,
        }
    else: # pearson
        if X_st_fc_center is None or X_st_fc_std is None:
            X_st_fc = X_st_exp - baseline_st[None, :]
            X_st_fc_center, X_st_fc_std = _center_std_per_row(
                X_st_fc, ddof=1, eps=eps, dtype=dtype
            )
        cache[genes_sig] = {
            "genes_union": genes_union,
            "X_st_fc_center": X_st_fc_center,
            "X_st_fc_std": X_st_fc_std,
        }

 # -------------------------
 # 3) assemble output
 # -------------------------
    precomp = {
        "genes_union": genes_union,
        "X_sc_exp": X_sc_exp,
        "X_st_exp_normed": X_st_exp_normed,
        "baseline_sc": baseline_sc,
        "baseline_st": baseline_st,
        "corr_method": corr_method,
        "X_st_rank_center": X_st_rank_center,
        "X_st_rank_std": X_st_rank_std,
        "X_st_fc_center": X_st_fc_center,
        "X_st_fc_std": X_st_fc_std,
        "sc_obs_names": np.asarray(adata_sc.obs_names),
        "sc_celltypes": np.asarray(adata_sc.obs[groupby_sc]),
        "G": int(X_sc_exp.shape[1]),
        "eps": float(eps),
        "dtype": np.dtype(dtype).name,
    }

    if include_X_st_exp:
        precomp["X_st_exp"] = X_st_exp # (n_st, G)

    if include_X_sc_fc or precompute_sc_rank:
        X_sc_fc = X_sc_exp - baseline_sc[None, :]
        if include_X_sc_fc:
            precomp["X_sc_fc"] = X_sc_fc

        if precompute_sc_rank:
            X_sc_rank_center, X_sc_rank_std = _rank_center_std_per_row(
                X_sc_fc, ddof=1, eps=eps, dtype=dtype
            )
            precomp["X_sc_rank_center"] = X_sc_rank_center
            precomp["X_sc_rank_std"] = X_sc_rank_std

    return precomp

# ============================================================
# core: one-sc similarity
# ============================================================

def _sim_one_sc(i_sc, precomp, cos_threshold=None, eps=1e-12):
    """
    Return similarity vector (n_st,) for a single SC cell i_sc vs all ST spots.
    """
    exp_sc = precomp["X_sc_exp"][i_sc] # (G,)
    X_st_exp_normed = precomp["X_st_exp_normed"]
    G = precomp["G"]

    sc_norm = np.linalg.norm(exp_sc)
    if (not np.isfinite(sc_norm)) or (sc_norm <= eps):
        cos_all = np.zeros(X_st_exp_normed.shape[0], dtype=float)
    else:
        cos_all = X_st_exp_normed @ (exp_sc / sc_norm)

    corr_method = str(precomp.get("corr_method", "spearman")).lower().strip()

 # cosine-only fast path: if threshold wipes all, we can return quickly
    if cos_threshold is not None:
        idx = np.where(cos_all >= float(cos_threshold))[0]
        if idx.size == 0:
            return 0.5 * cos_all, None
    else:
        idx = None

    if corr_method == "spearman":
 # --- Spearman(FC) via centered ranks dot-product ---
        st_center = precomp["X_st_rank_center"] if idx is None else precomp["X_st_rank_center"][idx]
        st_std = precomp["X_st_rank_std"] if idx is None else precomp["X_st_rank_std"][idx]

 # SC rank center/std (precomputed if available)
        if "X_sc_rank_center" in precomp:
            sc_center = precomp["X_sc_rank_center"][i_sc]
            sc_std = precomp["X_sc_rank_std"][i_sc]
        else:
            fc_sc = precomp["X_sc_fc"][i_sc]
            r = rankdata(fc_sc)
            sc_center = r - r.mean()
            sc_std = sc_center.std(ddof=1)
            if (not np.isfinite(sc_std)) or (sc_std <= eps):
                sc_std = 1.0

        numer = st_center @ sc_center
        denom = (G - 1) * sc_std * st_std

        corr_sel = np.zeros_like(numer, dtype=float)
        m = (denom != 0) & np.isfinite(denom)
        corr_sel[m] = numer[m] / denom[m]
        corr_sel[~np.isfinite(corr_sel)] = 0.0

    elif corr_method == "pearson":
 # --- Pearson(FC) via centered FC dot-product ---
        st_center = precomp["X_st_fc_center"] if idx is None else precomp["X_st_fc_center"][idx]
        st_std = precomp["X_st_fc_std"] if idx is None else precomp["X_st_fc_std"][idx]

        fc_sc = precomp["X_sc_fc"][i_sc]
        sc_center = fc_sc - fc_sc.mean()
        sc_std = sc_center.std(ddof=1)
        if (not np.isfinite(sc_std)) or (sc_std <= eps):
            sc_std = 1.0

        numer = st_center @ sc_center
        denom = (G - 1) * sc_std * st_std

        corr_sel = np.zeros_like(numer, dtype=float)
        m = (denom != 0) & np.isfinite(denom)
        corr_sel[m] = numer[m] / denom[m]
        corr_sel[~np.isfinite(corr_sel)] = 0.0

    else:
        raise ValueError("precomp['corr_method'] must be 'spearman'|'pearson'")

    corr_all = np.zeros_like(cos_all, dtype=float)
    if idx is None:
        corr_all = corr_sel
    else:
        corr_all[idx] = corr_sel

    return 0.5 * cos_all + 0.5 * corr_all, None

def _sim_block_sc_batch(idxs_sc, precomp, cos_threshold=None, eps=1e-12):
    """Compute similarity block for a batch of SC indices.

    Parameters
    ----------
    idxs_sc
        1D array of SC indices (length B).

    Returns
    -------
    sim : np.ndarray
        Similarity matrix with shape (B, n_st).

    Notes
    -----
    This function is used only when `use_vectorized_batch=True` in
    `run_global_1to1_mapping`. It is *not* the default to minimize any risk
    of result changes due to floating-point reduction order.
    """
    idxs_sc = np.asarray(idxs_sc, dtype=int)
    X_sc_exp = precomp["X_sc_exp"][idxs_sc] # (B, G)
    X_st_exp_normed = precomp["X_st_exp_normed"] # (n_st, G)
    G = precomp["G"]

 # --- cosine(exp) ---
    sc_norms = np.linalg.norm(X_sc_exp, axis=1)
    good = (np.isfinite(sc_norms)) & (sc_norms > eps)
    X_sc_exp_scaled = np.zeros_like(X_sc_exp, dtype=np.asarray(X_sc_exp).dtype)
    if np.any(good):
        X_sc_exp_scaled[good] = X_sc_exp[good] / sc_norms[good, None]
 # (B, n_st)
    cos_block = X_sc_exp_scaled @ X_st_exp_normed.T

    corr_method = str(precomp.get("corr_method", "spearman")).lower().strip()

    if corr_method == "spearman":
 # --- Spearman(FC) via centered ranks dot-product ---
 # Use precomputed SC ranks if available (recommended).
        if ("X_sc_rank_center" in precomp) and ("X_sc_rank_std" in precomp):
            sc_center = precomp["X_sc_rank_center"][idxs_sc] # (B, G)
            sc_std = precomp["X_sc_rank_std"][idxs_sc] # (B,)
        else:
 # Fallback: compute per-row ranks (slower). Keeps behavior consistent.
            X_sc_fc = precomp["X_sc_fc"][idxs_sc]
            sc_center, sc_std = _rank_center_std_per_row(X_sc_fc, dtype=X_sc_fc.dtype)

        st_center = precomp["X_st_rank_center"] # (n_st, G)
        st_std = precomp["X_st_rank_std"] # (n_st,)

        numer = st_center @ sc_center.T
        denom = (G - 1) * (st_std[:, None] * sc_std[None, :])

    elif corr_method == "pearson":
 # --- Pearson(FC) via centered FC dot-product ---
        X_sc_fc = precomp["X_sc_fc"][idxs_sc]
        sc_center, sc_std = _center_std_per_row(X_sc_fc, dtype=X_sc_fc.dtype)

        st_center = precomp["X_st_fc_center"] # (n_st, G)
        st_std = precomp["X_st_fc_std"] # (n_st,)

        numer = st_center @ sc_center.T
        denom = (G - 1) * (st_std[:, None] * sc_std[None, :])

    else:
        raise ValueError("precomp['corr_method'] must be 'spearman'|'pearson'")

    corr = np.zeros_like(numer, dtype=float)
    m = (denom != 0) & np.isfinite(denom)
    corr[m] = numer[m] / denom[m]
    corr[~np.isfinite(corr)] = 0.0
    corr_block = corr.T # (B, n_st)

    if cos_threshold is not None:
        mask = cos_block < float(cos_threshold)
        if np.any(mask):
            corr_block = corr_block.copy()
            corr_block[mask] = 0.0

    return 0.5 * cos_block + 0.5 * corr_block

# ============================================================
# public API
# ============================================================

def run_global_1to1_mapping(
    adata_sc,
    adata_st,
    selected_genes,
    *,
    groupby_sc="cell_type",
    n_jobs=8,
    cos_threshold=None,
    type_threshold=None,
    uncertain_label="uncertain",
    store_best_cell=True,
    sort_genes=True,
    precompute_sc_rank=True,
    corr_method: str = "spearman", # "spearman" | "pearson"
    sc_batch_size=2048,
    eps=1e-12,
    use_vectorized_batch=True,
):
    """
    Compute global mapping (top-1 SC per ST) WITHOUT storing full S.

    Writes:
      adata_st.obs[OBS_GLOBAL_SIM]
      adata_st.obs[OBS_GLOBAL_TYPE]
      (optional) adata_st.obs[OBS_GLOBAL_BEST_CELL]
      (optional) adata_st.obs[OBS_GLOBAL_BEST_CELL_TYPE]
      adata_st.uns[UNS_GLOBAL_GENES_UNION]
    """
    precomp = prepare_global_precomp(
        adata_sc,
        adata_st,
        selected_genes,
        groupby_sc=groupby_sc,
        sort_genes=sort_genes,
        precompute_sc_rank=precompute_sc_rank,
        eps=eps,
        corr_method=corr_method,
    )

    n_sc = precomp["X_sc_exp"].shape[0]
    n_st = precomp["X_st_exp_normed"].shape[0]

 # Streaming top-1 over SC cells without materializing (n_sc x n_st) in memory.

 # Fast path default (numerical equivalence)
 # ----------------------------------------
 # By default we use the vectorized batch implementation (matrix
 # multiplications) for speed. This is *numerically* equivalent to the
 # original per-SC loop, but floating-point rounding can differ slightly due
 # to different accumulation/reduction order (typically ~1e-7–1e-5, depending
 # on BLAS and dtype).

 # If you ever need strict bitwise equivalence to the historical behavior,
 # set `use_vectorized_batch=False` to use the original per-SC joblib loop.
    best_score = np.full(n_st, -np.inf, dtype=float)
    best_sc_idx = np.zeros(n_st, dtype=int)

    sc_batch_size = int(sc_batch_size) if sc_batch_size is not None else n_sc
    sc_batch_size = max(1, min(sc_batch_size, n_sc))

    for start in range(0, n_sc, sc_batch_size):
        end = min(n_sc, start + sc_batch_size)
        idxs = np.arange(start, end, dtype=int)

        if use_vectorized_batch:
 # Fast path (default): compute a full (batch x n_st) similarity
 # block using matrix multiplications, then stream-update the top-1.

 # NOTE: Because matmul changes accumulation order vs per-row loops,
 # tiny floating-point differences are possible. For strict bitwise
 # equivalence, set use_vectorized_batch=False.
            sim_block = _sim_block_sc_batch(
                idxs,
                precomp,
                cos_threshold=cos_threshold,
                eps=eps,
            )
 # Update streaming top-1
 # Find, for each ST column, the best row in this batch
            j_best = np.argmax(sim_block, axis=0)
            batch_best = sim_block[j_best, np.arange(n_st)]
            better = batch_best > best_score
            if np.any(better):
                best_score[better] = batch_best[better]
                best_sc_idx[better] = idxs[j_best[better]]
        else:
 # Compatibility path: per-SC loop with joblib parallelism.
            rows = Parallel(n_jobs=n_jobs)(
                delayed(_sim_one_sc)(int(i), precomp, cos_threshold=cos_threshold, eps=eps)[0]
                for i in idxs
            )
            for i, sim_row in zip(idxs, rows):
                better = sim_row > best_score
                if np.any(better):
                    best_score[better] = sim_row[better]
                    best_sc_idx[better] = int(i)

    sc_obs_names = precomp["sc_obs_names"]
    sc_celltypes = precomp["sc_celltypes"]

    best_cell = sc_obs_names[best_sc_idx].astype(object)
    best_type_raw = sc_celltypes[best_sc_idx].astype(object)

    best_type = best_type_raw.copy()
    if type_threshold is not None:
        ok = best_score >= float(type_threshold)
        best_type[~ok] = uncertain_label

 # store canonical outputs
    adata_st.obs[OBS_GLOBAL_SIM] = best_score
    adata_st.obs[OBS_GLOBAL_TYPE] = best_type

    if store_best_cell:
        adata_st.obs[OBS_GLOBAL_BEST_CELL] = best_cell
 # IMPORTANT: store integer SC indices explicitly for downstream steps
 # (e.g., final_global_anchor_fgw), to avoid ambiguity with obs_names.
        adata_st.obs[OBS_GLOBAL_BEST_CELL_IDX] = best_sc_idx.astype(int)
        adata_st.obs[OBS_GLOBAL_BEST_CELL_TYPE] = best_type_raw

 # ------------------------------------------------------------
 # Unified weights container (from the very first step)
 # ------------------------------------------------------------
 # Store global 1:1 mapping as a degenerate mixture:
 #   sc_idx = [best_sc_idx[j]]
 #   w      = [1.0]
 # This makes downstream steps (pairwise merge, final-global update)
 # able to reuse a single .uns weights dict consistently.
        try:
            n_st = int(adata_st.n_obs)
            best_sc_idx_int = np.asarray(best_sc_idx, dtype=int)
            g_score = np.asarray(best_score, dtype=float)
            g_type = np.asarray(best_type, dtype=object)
            weights0 = {}
            for j in range(n_st):
                sc_i = int(best_sc_idx_int[j])
                weights0[int(j)] = dict(
                    best_type=g_type[j],
                    model="global_1to1",
                    sc_idx=np.asarray([sc_i], dtype=int),
                    w=np.asarray([1.0], dtype=float),
                    score=float(g_score[j]),
                )
            adata_st.uns[UNS_FINAL_WEIGHTS] = weights0
        except Exception as e:
            warnings.warn(
                f"[global_map] skipped initializing {UNS_FINAL_WEIGHTS!r} due to: {type(e).__name__}: {e}",
                RuntimeWarning,
            )
            pass

    adata_st.uns[UNS_GLOBAL_GENES_UNION] = precomp["genes_union"].astype(str).tolist()

    return {
        "precomp": precomp,
        "best_sc_idx": best_sc_idx,
        "best_score": best_score,
        "best_cell": best_cell,
        "best_type_raw": best_type_raw,
        "best_type": best_type,
    }

def assign_anchors_from_global(
    adata_st,
    *,
    out_prefix="global_sc", # kept for compatibility; canonical keys used from keys.py
    sim_threshold=None,
    top_frac=None,
    type_key=None, # default: OBS_GLOBAL_TYPE
    best_cell_key=None, # default: OBS_GLOBAL_BEST_CELL
    best_cell_type_key=None, # default: OBS_GLOBAL_BEST_CELL_TYPE
    write_to_obs=True,

    adata_sc=None,
    sc_geom_obsm_key=None,
    out_anchor_obsm_key="X_global_sc_anchor",
    write_anchor_obsm=False,
):
    """
    Assign anchors using stored global mapping results.
    ...
    """

    sim_key = OBS_GLOBAL_SIM
    type_key = type_key if type_key is not None else OBS_GLOBAL_TYPE
    best_cell_key = best_cell_key if best_cell_key is not None else OBS_GLOBAL_BEST_CELL
    best_cell_type_key = best_cell_type_key if best_cell_type_key is not None else OBS_GLOBAL_BEST_CELL_TYPE

    for k in (sim_key, type_key, best_cell_key, best_cell_type_key):
        if k not in adata_st.obs:
            raise KeyError(f"[assign_anchors_from_global] missing adata_st.obs['{k}']")

    best_score = np.asarray(adata_st.obs[sim_key], dtype=float)
    spot_type = np.asarray(adata_st.obs[type_key]).astype(object)
    best_cell = np.asarray(adata_st.obs[best_cell_key]).astype(object)
    best_cell_type = np.asarray(adata_st.obs[best_cell_type_key]).astype(object)

    n_st = adata_st.n_obs
    valid = np.ones(n_st, dtype=bool)

 # (A) type-wise top fraction
    if top_frac is not None:
        tf = float(top_frac)
        if not (0.0 < tf <= 1.0):
            raise ValueError("top_frac must be in (0, 1].")
        valid[:] = False

        s_type = pd.Series(spot_type)
        is_ok_type = pd.notna(s_type.values)
        for ct in pd.unique(s_type[is_ok_type]):
            idx = np.where(s_type.values == ct)[0]
            if idx.size == 0:
                continue
            thr = np.quantile(best_score[idx], 1.0 - tf)
            valid[idx] = best_score[idx] >= thr

 # (B) global threshold
    if sim_threshold is not None:
        valid &= (best_score >= float(sim_threshold))

    anchor_cell = best_cell.copy()
    anchor_type = best_cell_type.copy()
    anchor_cell[~valid] = None
    anchor_type[~valid] = None

    if write_to_obs:
        adata_st.obs[OBS_GLOBAL_ANCHOR] = anchor_cell
        adata_st.obs[OBS_GLOBAL_ANCHOR_TYPE] = anchor_type

    if write_anchor_obsm:
        if adata_sc is None:
            raise ValueError("write_anchor_obsm=True adata_sc .")
        if sc_geom_obsm_key is None:
            raise ValueError("write_anchor_obsm=True sc_geom_obsm_key .")
        if sc_geom_obsm_key not in adata_sc.obsm:
            raise KeyError(f"adata_sc.obsm['{sc_geom_obsm_key}'] not found.")

        X_sc = np.asarray(adata_sc.obsm[sc_geom_obsm_key], dtype=float) # (n_sc, d)
        d = X_sc.shape[1]

 # sc cell id -> row index
        sc_map = {cid: i for i, cid in enumerate(adata_sc.obs_names)}

        X_anchor = np.full((adata_st.n_obs, d), np.nan, dtype=float)

        for j, cid in enumerate(anchor_cell):
            if cid is None:
                continue
            i = sc_map.get(cid, None)
            if i is None:
                continue
            X_anchor[j] = X_sc[i]

        adata_st.obsm[out_anchor_obsm_key] = X_anchor

    return valid

def write_sc_anchor_stats(
    adata_sc,
    adata_st,
    *,
    anchor_key=OBS_GLOBAL_ANCHOR,
    score_key=OBS_GLOBAL_SIM,
):
    """
    Optional: write SC-side anchor usage stats.

    Requires:
      adata_st.obs[anchor_key] contains sc obs_names (strings) or None
      adata_st.obs[score_key] contains anchor scores

    Writes into adata_sc.obs (canonical keys from util.keys):
      OBS_SC_IS_ANCHOR
      OBS_SC_ANCHOR_N_SPOTS
      OBS_SC_ANCHOR_MAX_SCORE
    """
    if anchor_key not in adata_st.obs:
        raise KeyError(f"[write_sc_anchor_stats] missing adata_st.obs['{anchor_key}']")
    if score_key not in adata_st.obs:
        raise KeyError(f"[write_sc_anchor_stats] missing adata_st.obs['{score_key}']")

    anchors = np.asarray(adata_st.obs[anchor_key]).astype(object)
    scores = np.asarray(adata_st.obs[score_key], dtype=float)

    sc_names = np.asarray(adata_sc.obs_names).astype(object)
    name_to_idx = {name: i for i, name in enumerate(sc_names)}
    n_sc = adata_sc.n_obs

    is_anchor = np.zeros(n_sc, dtype=bool)
    n_spots = np.zeros(n_sc, dtype=int)
    max_score = np.zeros(n_sc, dtype=float)

 # iterate only over valid anchors
    idx_st = np.where(pd.notna(anchors))[0]
    for j in idx_st:
        a = anchors[j]
        i = name_to_idx.get(a, None)
        if i is None:
            continue
        is_anchor[i] = True
        n_spots[i] += 1
        s = scores[j]
        if np.isfinite(s) and s > max_score[i]:
            max_score[i] = float(s)

    adata_sc.obs[OBS_SC_IS_ANCHOR] = is_anchor
    adata_sc.obs[OBS_SC_ANCHOR_N_SPOTS] = n_spots
    adata_sc.obs[OBS_SC_ANCHOR_MAX_SCORE] = max_score

    return {
        "n_sc": int(n_sc),
        "n_anchor_sc": int(is_anchor.sum()),
        "n_anchor_spots": int(len(idx_st)),
    }
