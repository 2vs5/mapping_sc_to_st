from __future__ import annotations

from collections import defaultdict
import numpy as np

from scipy.spatial import cKDTree

from .prep import _get_spatial_coords as _get_spatial_coords
from .keys import (
    OBS_GLOBAL_ANCHOR,
    OBS_GLOBAL_SIM,
    OBSM_ST_SPATIAL_NORMED,
    OBSM_SC_LATENT,
    OBSM_SC_ST_REP,
    OBS_SC_HAS_REP,
    UNS_SC_KNN_EDGES,
    UNS_TYPEPAIR_RESULTS,
    UNS_TYPEPAIR_PAIRS,
    UNS_TYPEPAIR_PAIRS_ALL,
)

# -----------------------------
# helpers
# -----------------------------

def _names_to_indices(adata, names_raw):
    """Map obs_names (strings) -> integer indices; -1 if missing."""
    obs_names = np.asarray(adata.obs_names).astype(object)
    name2idx = {name: i for i, name in enumerate(obs_names)}
    idx = np.full(len(names_raw), -1, dtype=np.int64)
    for j, a in enumerate(names_raw):
        if a is None or (isinstance(a, float) and np.isnan(a)):
            continue
        i = name2idx.get(a, None)
        if i is None:
            continue
        idx[j] = int(i)
    return idx

def _ckdtree_query(tree: cKDTree, X: np.ndarray, k: int, workers: int = 1):
    """
    SciPy version compatibility:
    - Newer SciPy: query(..., workers=)
    - Older SciPy: query(..., n_jobs=) or no parallel arg
    """
    try:
        return tree.query(X, k=k, workers=workers)
    except TypeError:
        try:
            return tree.query(X, k=k, n_jobs=workers)
        except TypeError:
            return tree.query(X, k=k)

# -----------------------------
# 1) SC -> representative ST coord (streaming, memory-safe)
# -----------------------------

def build_anchor_representatives_streaming(
    adata_main, # typically adata_sc
    adata_ref, # typically adata_st (anchors + spatial)
    *,
    anchor_key=OBS_GLOBAL_ANCHOR,
    score_key=OBS_GLOBAL_SIM,
    ref_spatial_key=OBSM_ST_SPATIAL_NORMED,
    xy_cols=None,
    method="best", # "best" | "weighted_mean" | "mean"
    min_score=None,
    out_obsm_key=OBSM_SC_ST_REP,
    out_obs_valid_key=OBS_SC_HAS_REP,
    dtype=np.float32,
):
    """
    Build representative coordinates in adata_main based on anchors in adata_ref.

    - adata_ref.obs[anchor_key] contains adata_main.obs_names (e.g., ST spot -> SC cell name)
    - adata_ref.obsm[ref_spatial_key] provides spatial coordinates

    Output:
      adata_main.obsm[out_obsm_key] : (n_main, d) representative coords
      adata_main.obs[out_obs_valid_key] : bool, has representative
    """
    if anchor_key not in adata_ref.obs:
        raise KeyError(f"{anchor_key} not found in adata_ref.obs.")
    if ref_spatial_key not in adata_ref.obsm:
        raise KeyError(f"{ref_spatial_key} not found in adata_ref.obsm.")

    X_ref = _get_spatial_coords(adata_ref, spatial_key=ref_spatial_key, xy_cols=xy_cols)
    X_ref = np.asarray(X_ref, dtype=dtype, order="C")

    n_main = adata_main.n_obs
    d = X_ref.shape[1]

    anchors_raw = np.asarray(adata_ref.obs[anchor_key], dtype=object)
    idx_main = _names_to_indices(adata_main, anchors_raw)

    scores = None
    if score_key is not None and score_key in adata_ref.obs:
        scores = np.asarray(adata_ref.obs[score_key], dtype=np.float32)

    valid = (idx_main >= 0) & (idx_main < n_main)
    if scores is not None and min_score is not None:
        valid &= (scores >= float(min_score))

    rep = np.full((n_main, d), np.nan, dtype=dtype)
    has = np.zeros(n_main, dtype=bool)

    if method == "best":
        best = np.full(n_main, -np.inf, dtype=np.float32)
        for j in np.where(valid)[0]:
            i = int(idx_main[j])
            s = float(scores[j]) if scores is not None else 0.0
            if s > best[i]:
                best[i] = s
                rep[i] = X_ref[j]
                has[i] = True

    elif method == "mean":
        sum_xy = np.zeros((n_main, d), dtype=np.float64)
        cnt = np.zeros(n_main, dtype=np.int64)
        for j in np.where(valid)[0]:
            i = int(idx_main[j])
            sum_xy[i] += X_ref[j]
            cnt[i] += 1
        ok = cnt > 0
        rep[ok] = (sum_xy[ok] / cnt[ok, None]).astype(dtype)
        has[ok] = True

    elif method == "weighted_mean":
        sum_wxy = np.zeros((n_main, d), dtype=np.float64)
        sum_w = np.zeros(n_main, dtype=np.float64)
        cnt = np.zeros(n_main, dtype=np.int64)

        for j in np.where(valid)[0]:
            i = int(idx_main[j])
            if scores is None:
                w = 1.0
            else:
                w = float(scores[j])
                if w < 0:
                    w = 0.0
            sum_wxy[i] += X_ref[j] * w
            sum_w[i] += w
            cnt[i] += 1

        ok = cnt > 0
        ok_w = ok & (sum_w > 0)
        ok_0 = ok & (sum_w <= 0)

        rep[ok_w] = (sum_wxy[ok_w] / sum_w[ok_w, None]).astype(dtype)

        if np.any(ok_0):
 # fallback to mean for those with zero total weight
            sum_xy = np.zeros((n_main, d), dtype=np.float64)
            cnt2 = np.zeros(n_main, dtype=np.int64)
            for j in np.where(valid)[0]:
                i = int(idx_main[j])
                if ok_0[i]:
                    sum_xy[i] += X_ref[j]
                    cnt2[i] += 1
            rep[ok_0] = (sum_xy[ok_0] / np.maximum(cnt2[ok_0], 1)[:, None]).astype(dtype)

        has[ok] = True

    else:
        raise ValueError("method must be 'best', 'weighted_mean', or 'mean'.")

    adata_main.obsm[out_obsm_key] = rep
    adata_main.obs[out_obs_valid_key] = has
    return rep, has

# -----------------------------
# -----------------------------

def build_latent_knn_edges_ckdtree(
    adata,
    *,
    latent_key,
    k=30,
    out_uns_key=UNS_SC_KNN_EDGES,
    subset_mask=None, # optional boolean mask (compute kNN only on subset)
    workers=1,
    dtype=np.float32,
):
    """
    Any AnnData -> kNN edge list (directed) using SciPy cKDTree.

    - Uses adata.obsm[latent_key] as coordinates in latent space
    - Stores src/dst into adata.uns[out_uns_key]

    subset_mask:
      - If provided, compute neighbors only among subset.
      - Edges will be in the full obs index space (src/dst are original indices).
    """
    if latent_key not in adata.obsm:
        raise KeyError(f"{latent_key} not found in adata.obsm.")

    Z = np.asarray(adata.obsm[latent_key], dtype=dtype)
    Z = np.ascontiguousarray(Z)

    n = Z.shape[0]
    k = int(k)
    if n == 0:
        raise ValueError("adata has 0 observations.")
    if k <= 0:
        raise ValueError("k must be >= 1.")

    if not np.isfinite(Z).all():
        raise ValueError("latent contains NaN/Inf.")

    if subset_mask is None:
        idx = np.arange(n, dtype=np.int64)
    else:
        subset_mask = np.asarray(subset_mask, dtype=bool)
        idx = np.where(subset_mask)[0].astype(np.int64)

    if idx.size == 0:
        raise ValueError("subset_mask selects 0 observations.")
    if k >= idx.size:
        raise ValueError(f"k={k} must be < selected_n={idx.size} (self removed).")

    Z_sub = Z[idx] # (n_sub, d)

    tree = cKDTree(Z_sub)
 # query returns (dist, nn_idx_in_subspace) with self included as first neighbor
    _, nbr = _ckdtree_query(tree, Z_sub, k=k + 1, workers=int(workers))
    nbr = np.asarray(nbr, dtype=np.int64)[:, 1:] # remove self -> shape (n_sub, k)

    src_sub = np.repeat(np.arange(idx.size, dtype=np.int64), k)
    dst_sub = nbr.reshape(-1).astype(np.int64)

    src = idx[src_sub]
    dst = idx[dst_sub]

    adata.uns[out_uns_key] = {
        "src": src.astype(np.int64, copy=False),
        "dst": dst.astype(np.int64, copy=False),
        "k": k,
        "latent_key": latent_key,
        "backend": "cKDTree",
        "workers": int(workers),
        "n_obs": int(n),
        "subset_n": int(idx.size),
        "dtype": str(Z.dtype),
    }
    return src, dst

# -----------------------------
# 3) Type-pair proximity summary in spatial coordinates
# -----------------------------

def summarize_typepair_spatial_proximity(
    adata,
    *,
    cell_type_key="cell_type",
    coord_key,
    knn_edges_key=UNS_SC_KNN_EDGES,
    min_edges=30,
    quantiles=(0.1, 0.5, 0.9),
    undirected=True,
):
    """
    Aggregate spatial distances along kNN edges by (type_a, type_b).

    Required:
      - adata.obs[cell_type_key]
      - adata.obsm[coord_key] : (n_obs, d) spatial coordinates
      - adata.uns[knn_edges_key]["src","dst"]
    """
    if cell_type_key not in adata.obs:
        raise KeyError(f"{cell_type_key} not found in adata.obs.")
    if coord_key not in adata.obsm:
        raise KeyError(f"{coord_key} not found in adata.obsm.")
    if knn_edges_key not in adata.uns:
        raise KeyError(f"{knn_edges_key} not found in adata.uns.")

    cell_types_str = np.asarray(adata.obs[cell_type_key]).astype(str)
    uniq = np.unique(cell_types_str)
    type_to_id = {t: i for i, t in enumerate(uniq)}
    id_to_type = {i: t for t, i in type_to_id.items()}
    type_ids = np.array([type_to_id[t] for t in cell_types_str], dtype=np.int64)

    coords = np.asarray(adata.obsm[coord_key], dtype=np.float32)
    coords = np.ascontiguousarray(coords)

    edges = adata.uns[knn_edges_key]
    src = np.asarray(edges["src"], dtype=np.int64)
    dst = np.asarray(edges["dst"], dtype=np.int64)

    A = coords[src]
    B = coords[dst]
    valid = np.isfinite(A).all(axis=1) & np.isfinite(B).all(axis=1)

    src_v = src[valid]
    dst_v = dst[valid]
    d = np.sqrt(((A[valid] - B[valid]) ** 2).sum(axis=1))

    buckets = defaultdict(list)
    if undirected:
        for i, j, dist in zip(src_v, dst_v, d):
            ti, tj = type_ids[i], type_ids[j]
            if ti <= tj:
                buckets[(ti, tj)].append(float(dist))
            else:
                buckets[(tj, ti)].append(float(dist))
    else:
        for i, j, dist in zip(src_v, dst_v, d):
            buckets[(type_ids[i], type_ids[j])].append(float(dist))

    results = []
    for (ti, tj), vals in buckets.items():
        vals = np.asarray(vals, dtype=np.float32)
        if vals.size < int(min_edges):
            continue
        row = {
            "type_a": id_to_type[ti],
            "type_b": id_to_type[tj],
            "n_edges": int(vals.size),
            "mean_d": float(vals.mean()),
            "median_d": float(np.median(vals)),
        }
        for q in quantiles:
            row[f"q{int(q*100):02d}_d"] = float(np.quantile(vals, q))
        results.append(row)

    results.sort(key=lambda r: (r["median_d"], r["mean_d"], -r["n_edges"]))
    return results

def select_pairs_by_proximity_top_frac(results, *, top_frac=0.01, remove_self=True):
    """
    Select top fraction by median distance (closest first).
    """
    if results is None or len(results) == 0:
        return [], [], []

    rows = []
    filtered_results = []
    for r in results:
        a = str(r["type_a"])
        b = str(r["type_b"])
        if remove_self and (a == b):
            continue
        rows.append((a, b, float(r["median_d"]), int(r.get("n_edges", 0))))
        filtered_results.append(r)

    rows.sort(key=lambda x: (x[2], -x[3]))
    k = max(1, int(np.ceil(len(rows) * float(top_frac))))
    keep = rows[:k]

    pairs = [tuple(sorted((a, b))) for (a, b, _, _) in keep]
    pairs_unique = []
    seen = set()
    for p in pairs:
        if p not in seen:
            seen.add(p)
            pairs_unique.append(p)

    return filtered_results, pairs_unique, keep

# -----------------------------
# 4) End-to-end runner (SC or ST)
# -----------------------------

def run_proximity_pair_selection_ckdtree(
    adata_main, # SC or ST
    adata_ref=None, # if SC: pass ST here; if ST: keep None
    *,
 # graph in latent (on adata_main)
    latent_key=OBSM_SC_LATENT,
    k=30,
    workers=1,

 # spatial distances on adata_main using coord_key_used
 # - ST pipeline: pass coord_key explicitly (e.g. OBSM_ST_SPATIAL_NORMED)
 # - SC pipeline: coord_key defaults to OBSM_SC_ST_REP (built from adata_ref)
    coord_key=None,

 # labels for type-pair aggregation (must exist on adata_main)
    cell_type_key="cell_type",

 # SC representative building (only used if adata_ref is not None)
    anchor_key=OBS_GLOBAL_ANCHOR,
    score_key=OBS_GLOBAL_SIM,
    ref_spatial_key=OBSM_ST_SPATIAL_NORMED,
    xy_cols=None,
    rep_method="best",
    rep_min_score=None,
    rep_out_obsm_key=OBSM_SC_ST_REP,
    rep_out_valid_key=OBS_SC_HAS_REP,

 # summarize/select
    min_edges=30,
    quantiles=(0.1, 0.5, 0.9),
    top_frac=0.01,
    undirected=True,

 # store
    knn_edges_key=UNS_SC_KNN_EDGES,
    store_to_uns=True,
    store_target="ref",
):
    """
    Usage:
      - SC pipeline:
          run_proximity_pair_selection_ckdtree(
              adata_main=adata_sc,
              adata_ref=adata_st,
              latent_key=OBSM_SC_LATENT,
          )
        (SC coords are built into adata_sc.obsm[OBSM_SC_ST_REP] and used for distances)

      - ST pipeline:
          run_proximity_pair_selection_ckdtree(
              adata_main=adata_st,
              adata_ref=None,
              latent_key="X_scvi",
              coord_key=OBSM_ST_SPATIAL_NORMED,
          )
    """
 # 1) coordinate space for distances
    if adata_ref is not None:
        build_anchor_representatives_streaming(
            adata_main,
            adata_ref,
            anchor_key=anchor_key,
            score_key=score_key,
            ref_spatial_key=ref_spatial_key,
            xy_cols=xy_cols,
            method=rep_method,
            min_score=rep_min_score,
            out_obsm_key=rep_out_obsm_key,
            out_obs_valid_key=rep_out_valid_key,
        )
        coord_key_used = rep_out_obsm_key if coord_key is None else coord_key

        subset_mask = np.asarray(adata_main.obs[rep_out_valid_key], dtype=bool)

    else:
        if coord_key is None:
            raise ValueError("For adata_ref=None (ST pipeline), coord_key must be provided.")
        coord_key_used = coord_key
        subset_mask = None

 # 2) kNN graph via cKDTree
    build_latent_knn_edges_ckdtree(
        adata_main,
        latent_key=latent_key,
        k=k,
        workers=workers,
        subset_mask=subset_mask,
        out_uns_key=knn_edges_key,
    )

 # 3) summarize + select
    results = summarize_typepair_spatial_proximity(
        adata_main,
        cell_type_key=cell_type_key,
        coord_key=coord_key_used,
        knn_edges_key=knn_edges_key,
        min_edges=min_edges,
        quantiles=quantiles,
        undirected=undirected,
    )
    filtered_results, pairs_selected, keep_rows = select_pairs_by_proximity_top_frac(
        results, top_frac=top_frac, remove_self=True
    )

 # 4) store
    if store_to_uns:
        if store_target == "main":
            target = adata_main
        elif store_target == "ref":
            target = adata_ref if adata_ref is not None else adata_main
        else:
            raise ValueError("store_target must be 'main' or 'ref'.")

        target.uns[UNS_TYPEPAIR_RESULTS] = results
        target.uns[UNS_TYPEPAIR_PAIRS] = pairs_selected
        target.uns[UNS_TYPEPAIR_PAIRS_ALL] = [
            tuple(sorted((str(r["type_a"]), str(r["type_b"]))))
            for r in results
            if str(r["type_a"]) != str(r["type_b"])
        ]
        target.uns["typepair_proximity_meta"] = {
            "backend": "cKDTree",
            "latent_key": latent_key,
            "k": int(k),
            "workers": int(workers),
            "coord_key": coord_key_used,
            "cell_type_key": cell_type_key,
            "min_edges": int(min_edges),
            "top_frac": float(top_frac),
            "ran_with_ref": bool(adata_ref is not None),
            "knn_edges_key": knn_edges_key,
        }

    return {
        "results": filtered_results,
        "pairs_selected": pairs_selected,
        "keep_rows": keep_rows,
        "coord_key_used": coord_key_used,
        "knn_edges_key_used": knn_edges_key,
    }
