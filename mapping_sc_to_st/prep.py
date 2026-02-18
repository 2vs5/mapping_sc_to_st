"""
Preparation / ensure utilities.

This module groups "misc but essential" preparation steps:
- Ensure ST has raw spatial coords and (optionally) normalized spatial coords
- Ensure SC/ST has scVI latent embedding (optional)
- Ensure SC/ST has NMF embedding (optional)
- Ensure UMAP exists (for SC and/or ST)
- Ensure common keys exist and have correct dtypes

Design:
- "ensure_*" functions are idempotent: calling twice should not break results.
- By default, they DO NOT overwrite existing representations unless overwrite=True.
"""

from __future__ import annotations

import numpy as np
from .c_build import knn_distances_to_dense_C, knn_graph_geodesic_C # re-export # noqa: F401
from .c_build import get_geom # re-export # noqa: F401

try:
    import scanpy as sc
except ImportError: # pragma: no cover
    sc = None

try:
    import scvi
except ImportError: # pragma: no cover
    scvi = None

try:
    import scipy.sparse as sp
except ImportError: # pragma: no cover
    sp = None

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import NMF as SkNMF

from .keys import (
    OBSM_ST_SPATIAL,
    OBSM_ST_SPATIAL_NORMED,
    UNS_ST_SPATIAL_NORM_INFO,
    OBSM_SC_LATENT,
)

# ============================================================
# Spatial helpers
# ============================================================

def _get_spatial_coords(adata_st, spatial_key=OBSM_ST_SPATIAL_NORMED, xy_cols=None):
    if spatial_key in adata_st.obsm:
        return np.asarray(adata_st.obsm[spatial_key], dtype=float)
    if xy_cols is not None:
        return np.asarray(adata_st.obs[list(xy_cols)], dtype=float)
    raise ValueError(
        f"Spatial coordinates not found. Provide adata_st.obsm['{spatial_key}'] "
        "or specify xy_cols from adata_st.obs."
    )

def _median_knn_distance(X, k=10):
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X)
    dist, _ = nn.kneighbors(X)
    return float(np.median(dist[:, 1:]))

def ensure_spatial_normed(
    adata_st,
    *,
    spatial_key=OBSM_ST_SPATIAL,
    xy_cols=None,
    k=10,
    out_key=OBSM_ST_SPATIAL_NORMED,
    info_key=UNS_ST_SPATIAL_NORM_INFO,
    overwrite=False,
):
    """
    Ensure normalized spatial coordinates exist in adata_st.obsm[out_key].

    - Uses median kNN distance as robust scale:
        spatial_normed = spatial / median_knn_dist
    - Stores metadata in adata_st.uns[info_key].

    Returns
    -------
    scale : float
        The median kNN distance used for normalization.
    """
    if sc is None:
        raise ImportError("scanpy is required for ensure_spatial_normed")

    if (not overwrite) and (out_key in adata_st.obsm):
        info = adata_st.uns.get(info_key, {})
        return float(info.get("scale_median_knn", np.nan))

    X = _get_spatial_coords(adata_st, spatial_key=spatial_key, xy_cols=xy_cols)
    scale = _median_knn_distance(X, k=k)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid spatial scale computed: {scale}")

    adata_st.obsm[out_key] = (X / scale).astype(float, copy=False)
    adata_st.uns[info_key] = {
        "spatial_key": spatial_key if spatial_key in adata_st.obsm else None,
        "xy_cols": list(xy_cols) if xy_cols is not None else None,
        "k": int(k),
        "scale_median_knn": float(scale),
    }
    return float(scale)

def ensure_spatial_raw(
    adata_st,
    *,
    spatial_key=OBSM_ST_SPATIAL,
    xy_cols=None,
):
    """
    Ensure raw spatial coords exist at adata_st.obsm[spatial_key].

    If missing and xy_cols is provided, it copies obs[xy_cols] into obsm[spatial_key].
    """
    if sc is None:
        raise ImportError("scanpy is required for ensure_spatial_raw")

    if spatial_key in adata_st.obsm:
        return

    if xy_cols is None:
        raise ValueError(f"adata_st.obsm['{spatial_key}'] missing and xy_cols not provided.")

    X = np.asarray(adata_st.obs[list(xy_cols)], dtype=float)
    adata_st.obsm[spatial_key] = X.astype(float, copy=False)

# ============================================================
# scVI helpers (works for SC or ST)
# ============================================================

def ensure_scvi_latent(
    adata,
    *,
    latent_key=OBSM_SC_LATENT,
    batch_key=None,
    n_latent=30,
    max_epochs=200,
    overwrite=False,
    model_info_key="scvi_info",
    random_seed=0,
    counts_layer="counts",
 # --- NEW ---
    model_device = 0,
    accelerator="auto", # "gpu" / "cpu" / "auto"
    devices="auto",
    **trainer_kwargs,
):
    if scvi is None:
        raise ImportError("scvi-tools is required for ensure_scvi_latent")

    if (not overwrite) and (latent_key in adata.obsm):
        return None

    scvi.settings.seed = int(random_seed)

    layer_used = counts_layer if (counts_layer is not None and counts_layer in adata.layers) else None
    scvi.model.SCVI.setup_anndata(adata, batch_key=batch_key, layer=layer_used)

    model = scvi.model.SCVI(adata, n_latent=int(n_latent))
    model.to_device("cpu")
    
    

    model.train(
        max_epochs=int(max_epochs),
        accelerator=accelerator,
        devices=devices,
        **trainer_kwargs,
    )

    Z = model.get_latent_representation()
    adata.obsm[latent_key] = Z.astype(np.float32, copy=False)
    adata.uns[model_info_key] = {
        "batch_key": batch_key,
        "n_latent": int(n_latent),
        "max_epochs": int(max_epochs),
        "random_seed": int(random_seed),
        "layer_used": layer_used,
        "accelerator": accelerator,
        "devices": devices,
    }
    return model

# ============================================================
# NMF helpers (SC/ST)
# ============================================================

def _get_matrix_for_factorization(adata, *, layer=None):
    """
    Return X matrix for factorization (NMF, etc.)
    - If layer is provided and exists, use adata.layers[layer]
    - else use adata.X
    """
    if (layer is not None) and (layer in adata.layers):
        return adata.layers[layer], layer
    return adata.X, None

def _clip_negative_to_zero(X):
    """
    NMF requires non-negative matrix.
    If negatives exist, clip to zero (safe fallback).
    """
    if sp is not None and sp.issparse(X):
        if X.nnz == 0:
            return X
        X2 = X.tocsr(copy=True)
        neg = X2.data < 0
        if np.any(neg):
            X2.data[neg] = 0.0
        return X2

    X = np.asarray(X)
    if np.min(X) < 0:
        X = np.maximum(X, 0.0)
    return X

def ensure_nmf(
    adata,
    *,
    n_components=30,
    layer=None,
    key_added="X_nmf",
    overwrite=False,
    random_state=0,
    max_iter=400,
    init="nndsvda",
    model_info_key="nmf_info",
    verbose=True,
):
    """
    Ensure NMF embedding exists in adata.obsm[key_added].

    Notes
    -----
    - Requires non-negative matrix; negatives are clipped to 0 as a safe fallback.
    - Stores lightweight metadata in adata.uns[model_info_key].
    - Does NOT overwrite unless overwrite=True.
    """
    if (not overwrite) and (key_added in adata.obsm):
        if verbose:
            print(f"[prep] NMF exists: obsm['{key_added}']")
        return

    X, layer_used = _get_matrix_for_factorization(adata, layer=layer)
    X = _clip_negative_to_zero(X)

    nmf = SkNMF(
        n_components=int(n_components),
        init=init,
        random_state=int(random_state),
        max_iter=int(max_iter),
    )

    W = nmf.fit_transform(X)
    adata.obsm[key_added] = W.astype(np.float32, copy=False)

    adata.uns[model_info_key] = {
        "key_added": key_added,
        "layer_used": layer_used,
        "n_components": int(n_components),
        "random_state": int(random_state),
        "max_iter": int(max_iter),
        "init": init,
        "components_shape": tuple(getattr(nmf, "components_", np.empty((0, 0))).shape),
        "reconstruction_err": float(getattr(nmf, "reconstruction_err_", np.nan)),
    }

    if verbose:
        print(
            f"[prep] NMF computed: obsm['{key_added}'] shape={adata.obsm[key_added].shape}, layer={layer_used}"
        )

# ============================================================
# UMAP helpers
# ============================================================

def ensure_umap(
    adata,
    *,
    use_rep=None,
    n_neighbors=15,
    min_dist=0.5,
    metric="euclidean",
    key_added="X_umap",
    overwrite=False,
    pca_if_needed=True,
    pca_key="X_pca",
    n_pcs=50,
):
    """
    Ensure UMAP exists for an AnnData.

    Typical usage:
      - SC: use_rep='X_scvi' or 'X_pca'
      - ST: use_rep='X_pca' (unless you have a special representation)

    Notes
    -----
    - scanpy stores UMAP at adata.obsm['X_umap'] by default.
    - We keep key_added for flexibility, but default aligns with scanpy.
    """
    if sc is None:
        raise ImportError("scanpy is required for ensure_umap")

    if (not overwrite) and (key_added in adata.obsm):
        return

 # scanpy uses 'X_umap' by default; we will copy if key_added differs.
    if (not overwrite) and ("X_umap" in adata.obsm) and (key_added != "X_umap"):
        adata.obsm[key_added] = adata.obsm["X_umap"].copy()
        return

    if (not overwrite) and ("X_umap" in adata.obsm) and (key_added == "X_umap"):
        return

 # Ensure neighbors graph exists
    if use_rep is None:
        if pca_if_needed and (pca_key not in adata.obsm):
            sc.pp.pca(adata, n_comps=int(n_pcs))
        use_rep = pca_key if pca_key in adata.obsm else None

    sc.pp.neighbors(
        adata,
        use_rep=use_rep,
        n_neighbors=int(n_neighbors),
        metric=metric,
    )
    sc.tl.umap(adata, min_dist=float(min_dist))

    if key_added != "X_umap":
        adata.obsm[key_added] = adata.obsm["X_umap"].copy()

# ============================================================
# PCA / Neighbors / Graph utilities
# ============================================================

def ensure_sc_pca(
    adata_sc,
    *,
    n_comps=50,
    use_highly_variable=True,
    svd_solver="arpack",
    pca_key="X_pca",
    overwrite=False,
    verbose=True,
):
    """
    Ensure SC PCA embedding in `adata_sc.obsm[pca_key]`.

    Notes:
      - Scanpy writes to `obsm["X_pca"]` by default; we mirror/copy to `pca_key`.
      - Assumes `adata_sc.X` is already suitable (typically log1p normalized).
    """
    if sc is None:
        raise ImportError("scanpy is required for ensure_sc_pca")

    if (not overwrite) and (pca_key in adata_sc.obsm):
        if verbose:
            print(f"[prep] SC PCA exists: obsm['{pca_key}']")
        return

    sc.tl.pca(
        adata_sc,
        n_comps=int(n_comps),
        use_highly_variable=bool(use_highly_variable),
        svd_solver=svd_solver,
    )
    adata_sc.obsm[pca_key] = np.asarray(adata_sc.obsm["X_pca"])
    if verbose:
        print(f"[prep] SC PCA computed: obsm['{pca_key}'] shape={adata_sc.obsm[pca_key].shape}")

def ensure_neighbors(
    adata,
    *,
    use_rep="X_pca",
    n_neighbors=15,
    metric="euclidean",
    key_added="neighbors",
    overwrite=False,
    verbose=True,
):
    """
    Ensure kNN graph for `adata` using scanpy neighbors.

    - If `key_added == "neighbors"` (default), scanpy writes:
        uns["neighbors"], obsp["distances"], obsp["connectivities"]
      else writes:
        uns[key_added], obsp[f"{key_added}_distances"], obsp[f"{key_added}_connectivities"].
    """
    if sc is None:
        raise ImportError("scanpy is required for ensure_neighbors")

    if not overwrite:
        if key_added == "neighbors":
            if (
                ("neighbors" in adata.uns)
                and ("distances" in adata.obsp)
                and ("connectivities" in adata.obsp)
            ):
                if verbose:
                    print("[prep] neighbors exists (default keys)")
                return
        else:
            dkey = f"{key_added}_distances"
            ckey = f"{key_added}_connectivities"
            if (key_added in adata.uns) and (dkey in adata.obsp) and (ckey in adata.obsp):
                if verbose:
                    print(f"[prep] neighbors exists (key_added='{key_added}')")
                return

    sc.pp.neighbors(
        adata,
        n_neighbors=int(n_neighbors),
        use_rep=use_rep,
        metric=metric,
        key_added=key_added,
    )
    if verbose:
        print(
            f"[prep] neighbors computed: use_rep={use_rep}, n_neighbors={n_neighbors}, "
            f"metric={metric}, key_added={key_added}"
        )

def get_knn_distance_matrix(
    adata,
    *,
    key_added="neighbors",
):
    """
    Return the sparse kNN distance matrix from `.obsp`, according to `key_added`.
    """
    if key_added == "neighbors":
        return adata.obsp["distances"]
    return adata.obsp[f"{key_added}_distances"]

# ============================================================
# Combined convenience
# ============================================================

def ensure_all_basic(
    adata_sc,
    adata_st,
    *,
 # spatial
    st_xy_cols=None,
    make_spatial_normed=True,
    spatial_k=10,
 # scvi SC
    make_scvi=False,
    scvi_batch_key=None,
    scvi_n_latent=30,
    scvi_max_epochs=200,
    scvi_counts_layer="counts",

 # scvi ST
    make_scvi_st=False,
    scvi_st_batch_key=None,
    scvi_st_n_latent=30,
    scvi_st_max_epochs=200,
    scvi_st_counts_layer="counts",
    st_latent_key="X_scvi",
    
# parameters for scvi
    model_device = 0,
    accelerator = 'auto',
    devices = 'auto',
 # NMF (NEW)
    make_nmf_sc=False,
    nmf_sc_n_components=30,
    nmf_sc_layer=None,
    nmf_sc_key="X_nmf",
    make_nmf_st=False,
    nmf_st_n_components=30,
    nmf_st_layer=None,
    nmf_st_key="X_nmf",
 # sc pca / knn
    make_pca_sc=False,
    sc_pca_n_comps=50,
    sc_pca_use_hvg=True,
    make_knn_sc=False,
    knn_sc_use_rep="X_pca",
    knn_sc_n_neighbors=15,
    knn_sc_metric="euclidean",
    knn_sc_key_added="neighbors_sc",
 # st knn (spatial graph)
    make_knn_st=False,
    knn_st_use_rep="spatial_normed",
    knn_st_n_neighbors=15,
    knn_st_metric="euclidean",
    knn_st_key_added="neighbors_st",
 # umap
    make_umap_sc=False,
    make_umap_st=False,
    umap_neighbors=15,
    umap_min_dist=0.5,
    overwrite=False,
):
    """
    One-call convenience to ensure common prerequisites.

    Typical recommendation:
      - ensure raw spatial always exists
      - ensure spatial_normed exists (for geometry/proximity/FGW)
      - ensure scVI latent exists for SC neighbor graph (and optionally for ST)
      - ensure NMF embeddings optionally (SC and/or ST)
      - ensure UMAP for SC (for plotting); ST UMAP optional

    Returns
    -------
    info : dict
        What was created / scale values.
    """
    if sc is None:
        raise ImportError("scanpy is required for ensure_all_basic")

    info = {}

 # ---- ST spatial
    ensure_spatial_raw(adata_st, xy_cols=st_xy_cols)
    if make_spatial_normed:
        scale = ensure_spatial_normed(
            adata_st,
            k=spatial_k,
            overwrite=overwrite,
        )
        info["spatial_norm_scale"] = scale
        

 # ---- SC scVI
    if make_scvi:
        print('start make_scvi')
        model = ensure_scvi_latent(
            adata_sc,
            latent_key=OBSM_SC_LATENT,
            batch_key=scvi_batch_key,
            n_latent=scvi_n_latent,
            max_epochs=scvi_max_epochs,
            overwrite=overwrite,
            model_info_key="scvi_info_sc",
            counts_layer=scvi_counts_layer,
            model_device = 0,
            accelerator = accelerator,
            devices = devices
        )
        info["scvi_trained_sc"] = model is not None

 # ---- ST scVI
    if make_scvi_st:
        model_st = ensure_scvi_latent(
            adata_st,
            latent_key=st_latent_key,
            batch_key=scvi_st_batch_key,
            n_latent=scvi_st_n_latent,
            max_epochs=scvi_st_max_epochs,
            overwrite=overwrite,
            model_info_key="scvi_info_st",
            counts_layer=scvi_st_counts_layer,
            model_device = 0,
            accelerator = accelerator,
            devices = devices
        )
        info["scvi_trained_st"] = model_st is not None

 # ---- NMF SC
    if make_nmf_sc:
        ensure_nmf(
            adata_sc,
            n_components=nmf_sc_n_components,
            layer=nmf_sc_layer,
            key_added=nmf_sc_key,
            overwrite=overwrite,
            model_info_key="nmf_info_sc",
            verbose=True,
        )
        info["nmf_sc"] = True

 # ---- NMF ST
    if make_nmf_st:
        ensure_nmf(
            adata_st,
            n_components=nmf_st_n_components,
            layer=nmf_st_layer,
            key_added=nmf_st_key,
            overwrite=overwrite,
            model_info_key="nmf_info_st",
            verbose=True,
        )
        info["nmf_st"] = True

 # ---- SC PCA
    if make_pca_sc:
        ensure_sc_pca(
            adata_sc,
            n_comps=sc_pca_n_comps,
            use_highly_variable=sc_pca_use_hvg,
            pca_key="X_pca",
            overwrite=overwrite,
            verbose=True,
        )
        info["pca_sc"] = True

 # ---- SC kNN graph
    if make_knn_sc:
        ensure_neighbors(
            adata_sc,
            use_rep=knn_sc_use_rep,
            n_neighbors=knn_sc_n_neighbors,
            metric=knn_sc_metric,
            key_added=knn_sc_key_added,
            overwrite=overwrite,
            verbose=True,
        )
        info["knn_sc"] = True

 # ---- ST kNN graph (typically on spatial_normed)
    if make_knn_st:
        if knn_st_use_rep == "spatial_normed" and (OBSM_ST_SPATIAL_NORMED not in adata_st.obsm):
            ensure_spatial_normed(adata_st, k=spatial_k, overwrite=overwrite)

        ensure_neighbors(
            adata_st,
            use_rep=(OBSM_ST_SPATIAL_NORMED if knn_st_use_rep == "spatial_normed" else knn_st_use_rep),
            n_neighbors=knn_st_n_neighbors,
            metric=knn_st_metric,
            key_added=knn_st_key_added,
            overwrite=overwrite,
            verbose=True,
        )
        info["knn_st"] = True

 # ---- UMAP SC
    if make_umap_sc:
        rep_sc = OBSM_SC_LATENT if (OBSM_SC_LATENT in adata_sc.obsm) else ("X_nmf" if "X_nmf" in adata_sc.obsm else None)
        ensure_umap(
            adata_sc,
            use_rep=rep_sc,
            n_neighbors=umap_neighbors,
            min_dist=umap_min_dist,
            overwrite=overwrite,
        )
        info["umap_sc"] = True

 # ---- UMAP ST
    if make_umap_st:
        rep_st = st_latent_key if (st_latent_key in adata_st.obsm) else ("X_nmf" if "X_nmf" in adata_st.obsm else None)
        ensure_umap(
            adata_st,
            use_rep=rep_st,
            n_neighbors=umap_neighbors,
            min_dist=umap_min_dist,
            overwrite=overwrite,
        )
        info["umap_st"] = True

    return info

def filter_small_celltypes(
    adata,
    groupby="cell_type",
    min_cells=20,
    verbose=True,
):
    """
    Remove cell types with fewer than `min_cells` cells.
    """
    counts = adata.obs[groupby].value_counts()
    small_groups = counts[counts < min_cells].index.tolist()

    if verbose:
        print("too small groups:", small_groups)

    mask = ~adata.obs[groupby].isin(small_groups)
    adata_filt = adata[mask].copy()
    return adata_filt, small_groups
