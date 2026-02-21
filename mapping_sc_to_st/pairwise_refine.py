"""
Pairwise refinement module (FGW + mixture scoring), stepwise and pluggable.

What this module does
---------------------
Given a candidate cell-type pair (ct1, ct2) and a corridor of ST spots whose
global type is ct1 or ct2:

1) Build two directional gene sets (ct1-focused and ct2-focused):
      G_ct1 = genes_union - bleeding(ct1 -> ct2)   (or other policy)
      G_ct2 = genes_union - bleeding(ct2 -> ct1)
2) Solve two FGW problems to obtain transport plans T1, T2
3) Column-normalize to mixture weights W1, W2
4) For each ST spot, score the mixture against its own ST expression
5) Choose the better *model* (direction) per spot, then decide final type
   using weight-mass (ct1-mass vs ct2-mass) with a tie-break.

Important
---------
- This function refines within a pair only. Merging multiple pairs is done by util.merge.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from .prep import _get_spatial_coords as _get_spatial_coords

from .keys import (
    OBS_GLOBAL_TYPE,
    OBS_GLOBAL_ANCHOR,
    OBS_FINAL_TYPE,
    OBS_FINAL_SCORE,
    OBS_FINAL_ANCHOR_TYPE,
    OBS_FINAL_ANCHOR
)
from .fgw_solver import build_anchor_cost_pair
from .transport_engine import run_transport
from .c_build import get_geom as _get_geom, build_C_euclidean, build_C_from_knn, _normalize_C
from .precomp import mixture_scores_from_weights

def _subset_indices_by_types(labels, ct1, ct2):
    labels = np.asarray(labels, dtype=object)
    m = (labels == ct1) | (labels == ct2)
    return np.where(m)[0], m

def refine_pair_ct1_ct2(
    adata_sc,
    adata_st,
    *,
    ct1: str,
    ct2: str,
    bleeding_obj=None, # output of compute_bleeding_directional
    incoming_obj=None, # output of compute_incoming_bleeding (optional)
    bleeding_mode="bleeding", # "bleeding" | "incoming" | "none"
    groupby_sc="cell_type",
    global_type_key=OBS_GLOBAL_TYPE,
 # geometry for FGW (C_sc, C_st)
    sc_geom_obsm_key="X_scvi",
    st_geom_obsm_key="spatial_normed",
    sc_geom_obs_keys=None,
    st_geom_obs_keys=None,
 # geometry builder options
    sc_C_mode="euclidean", # "euclidean" | "knn_dense" | "knn_geodesic"
    st_C_mode="euclidean", # "euclidean" | "knn_dense" | "knn_geodesic"
    sc_knn_key_added="neighbors_sc",
    st_knn_key_added="neighbors_st",
    knn_fill_value="max",
    knn_directed=False,
 # anchor cost
    use_anchor_cost=True,
    anchor_key=OBS_GLOBAL_ANCHOR,
    beta_expr=1,

 # transport normalization / C adaptation (kept in sync with final-global)
    M_norm: str = "mean",
    C_kind_sc: str = "distance",
    C_kind_st: str = "distance",
    C_norm: str = "q95",
    C_clip_q=None,
 # solver
    solver=None,
    solver_kwargs=None,
    alpha_fgw=None, # deprecated (use solver.alpha)
 # scoring
    precomp=None, # output of util.precomp.prepare_precomp_for_scoring
    cos_threshold=None,
    score_n_jobs=1,
 # decisions
    mass_tie_eps=0,
):
    """
    Returns
    -------
    out : dict with keys
      - mask_pair: (n_st,) bool
      - refined_type: (n_st,) object (only for pair spots, else None)
      - refined_score: (n_st,) float  (only for pair spots, else 0)
      - weights_dict: dict[int st_idx -> dict(...)]
      - debug: dict(...)
    """
    if solver is None:
        raise ValueError("solver is required (e.g., util.POTFGWSolver(...)).")
    solver_kwargs = {} if solver_kwargs is None else dict(solver_kwargs)
    if alpha_fgw is not None:
 # keep compatibility; prefer solver.alpha
        solver_kwargs.setdefault("alpha", float(alpha_fgw))

    ct1 = str(ct1); ct2 = str(ct2)

 # corridor indices
    idx_st_pair, mask_pair = _subset_indices_by_types(adata_st.obs[global_type_key], ct1, ct2)
    if idx_st_pair.size == 0:
        return dict(mask_pair=mask_pair, refined_type=np.array([None]*adata_st.n_obs, object),
                    refined_score=np.zeros(adata_st.n_obs, float), weights_dict={}, debug={"reason":"no_st_pair"})

    idx_sc_pair, _ = _subset_indices_by_types(adata_sc.obs[groupby_sc], ct1, ct2)
    if idx_sc_pair.size == 0:
        return dict(mask_pair=mask_pair, refined_type=np.array([None]*adata_st.n_obs, object),
                    refined_score=np.zeros(adata_st.n_obs, float), weights_dict={}, debug={"reason":"no_sc_pair"})

    genes_union = np.asarray(precomp["genes_union"], dtype=object)

 # ------------------------------------------------------------
 # Gene panel policy (fixed: Policy B)
 # ---------------------------------
 #   base panel = genes_union (global selected genes)
 #   remove only bleeding / incoming genes
 #   (directional for 'bleeding', per-target for 'incoming')

 # Rationale:
 #   - keeps score scale comparable to the global stage
 #   - keeps feature dimension stable across pairs

 # NOTE: bleeding_mode names are kept for API compatibility.
 # ------------------------------------------------------------
    if bleeding_mode == "none":
        mask_g1 = np.ones(len(genes_union), dtype=bool)
        mask_g2 = np.ones(len(genes_union), dtype=bool)

    elif bleeding_mode == "incoming":
        if incoming_obj is None:
            raise ValueError("bleeding_mode='incoming' requires incoming_obj.")
        incoming = incoming_obj.get("incoming_per_target", {}) or {}
        inc1 = set(incoming.get(ct1, set()) or [])
        inc2 = set(incoming.get(ct2, set()) or [])
        mask_g1 = ~np.isin(genes_union, list(inc1))
        mask_g2 = ~np.isin(genes_union, list(inc2))

    elif bleeding_mode == "bleeding":
        if bleeding_obj is None:
            raise ValueError("bleeding_mode='bleeding' requires bleeding_obj.")
        bleed = bleeding_obj.get("bleeding_pairs", {}) or {}
        b12 = set(bleed.get((ct1, ct2), set()) or [])
        b21 = set(bleed.get((ct2, ct1), set()) or [])
        mask_g1 = ~np.isin(genes_union, list(b12))
        mask_g2 = ~np.isin(genes_union, list(b21))

    else:
        raise ValueError("bleeding_mode must be 'bleeding' | 'incoming' | 'none'.")
    if int(mask_g1.sum()) < 3 or int(mask_g2.sum()) < 3:
        return dict(mask_pair=mask_pair, refined_type=np.array([None]*adata_st.n_obs, object),
                    refined_score=np.zeros(adata_st.n_obs, float), weights_dict={}, debug={"reason":"too_few_genes",
                    "n_g1":int(mask_g1.sum()), "n_g2":int(mask_g2.sum())})

 # features
    X_sc_all = precomp["X_sc_exp"]
    X_st_all = precomp["X_st_exp"]
    X_sc_pair_g1 = X_sc_all[idx_sc_pair][:, mask_g1]
    X_st_pair_g1 = X_st_all[idx_st_pair][:, mask_g1]
    X_sc_pair_g2 = X_sc_all[idx_sc_pair][:, mask_g2]
    X_st_pair_g2 = X_st_all[idx_st_pair][:, mask_g2]

 # geometry for C matrices (raw)
    sc_geom_all = _get_geom(adata_sc, obsm_key=sc_geom_obsm_key, obs_keys=sc_geom_obs_keys, fallback_X=X_sc_all)
    st_geom_all = _get_geom(adata_st, obsm_key=st_geom_obsm_key, obs_keys=st_geom_obs_keys, fallback_X=X_st_all)

    if sc_C_mode == "euclidean":
        C_sc_raw = build_C_euclidean(sc_geom_all[idx_sc_pair], metric="euclidean", normalize=None)

    elif sc_C_mode == 'metacell_distance':
        Cfull = adata_sc.obsp["mf_distance_matrix"]
        Csub = Cfull[idx_sc_pair, :][:, idx_sc_pair]
        C_sc_raw = _normalize_C(np.asarray(Csub, dtype=float), normalize = None)

    elif sc_C_mode in ("knn_dense", "knn_geodesic"):
        C_sc_raw = build_C_from_knn(
            adata_sc,
            idx=idx_sc_pair,
            key_added=sc_knn_key_added,
            mode=("dense" if sc_C_mode == "knn_dense" else "geodesic"),
            fill_value=knn_fill_value,
            directed=knn_directed,
        )
    else:
        raise ValueError("sc_C_mode must be 'euclidean'|'knn_dense'|'knn_geodesic'")

    if st_C_mode == "euclidean":
        C_st_raw = build_C_euclidean(st_geom_all[idx_st_pair], metric="euclidean", normalize=None)
    elif st_C_mode in ("knn_dense", "knn_geodesic"):
        C_st_raw = build_C_from_knn(
            adata_st,
            idx=idx_st_pair,
            key_added=st_knn_key_added,
            mode=("dense" if st_C_mode == "knn_dense" else "geodesic"),
            fill_value=knn_fill_value,
            directed=knn_directed,
        )
    else:
        raise ValueError("st_C_mode must be 'euclidean'|'knn_dense'|'knn_geodesic'")

 # anchor cost (shared for both directions)
    M_anchor = None
    if use_anchor_cost:
        M_anchor = build_anchor_cost_pair(
            adata_sc, adata_st,
            idx_sc_pair=idx_sc_pair,
            idx_st_pair=idx_st_pair,
            sc_geom_all=sc_geom_all,
            st_geom_all=st_geom_all,
            anchor_key=anchor_key,
        )

 # FGW solve (two directions)
    rr1 = run_transport(
        X_sc_feat=X_sc_pair_g1,
        X_st_feat=X_st_pair_g1,
        C_sc_raw=C_sc_raw,
        C_st_raw=C_st_raw,
        solver=solver,
        M_anchor=M_anchor,
        beta_expr=beta_expr,
        M_norm=str(M_norm),
        C_kind_sc=str(C_kind_sc),
        C_kind_st=str(C_kind_st),
        C_norm=str(C_norm),
        C_clip_q=C_clip_q,
        **solver_kwargs,
    )
    rr2 = run_transport(
        X_sc_feat=X_sc_pair_g2,
        X_st_feat=X_st_pair_g2,
        C_sc_raw=C_sc_raw,
        C_st_raw=C_st_raw,
        solver=solver,
        M_anchor=M_anchor,
        beta_expr=beta_expr,
        M_norm=str(M_norm),
        C_kind_sc=str(C_kind_sc),
        C_kind_st=str(C_kind_st),
        C_norm=str(C_norm),
        C_clip_q=C_clip_q,
        **solver_kwargs,
    )

    W1 = rr1.W
    W2 = rr2.W

 # mixture scoring
    score1, _ = mixture_scores_from_weights(W1, idx_sc_pair, idx_st_pair, precomp, gene_mask=mask_g1, cos_threshold=cos_threshold, n_jobs=score_n_jobs)
    score2, _ = mixture_scores_from_weights(W2, idx_sc_pair, idx_st_pair, precomp, gene_mask=mask_g2, cos_threshold=cos_threshold, n_jobs=score_n_jobs)

    use_model1 = score1 >= score2
    best_score_pair = np.where(use_model1, score1, score2)

 # decide type by mass(W) within ct1/ct2 among idx_sc_pair
    sc_types_pair = np.asarray(adata_sc.obs[groupby_sc], dtype=object)[idx_sc_pair]
    mask_ct1 = (sc_types_pair == ct1)
    mask_ct2 = (sc_types_pair == ct2)

    weights_dict = {}
    best_type_pair = np.empty(len(idx_st_pair), dtype=object)

    for j_loc, st_idx in enumerate(idx_st_pair):
        Wj = (W1[:, j_loc] if use_model1[j_loc] else W2[:, j_loc])
        mass1 = float(Wj[mask_ct1].sum())
        mass2 = float(Wj[mask_ct2].sum())

        if abs(mass1 - mass2) <= float(mass_tie_eps):
 # tie -> fallback to model scores
            if score1[j_loc] > score2[j_loc]:
                best_ct = ct1
            elif score2[j_loc] > score1[j_loc]:
                best_ct = ct2
            else:
                best_ct = None
        else:
            best_ct = ct1 if mass1 > mass2 else ct2

        best_type_pair[j_loc] = best_ct

        weights_dict[int(st_idx)] = dict(
            best_type=best_ct,
            model=("ct1_model" if use_model1[j_loc] else "ct2_model"),
            sc_idx=np.asarray(idx_sc_pair, dtype=int).copy(),
            w=np.asarray(Wj, dtype=float).copy(),
            mass_ct1=mass1,
            mass_ct2=mass2,
            score_ct1_model=float(score1[j_loc]),
            score_ct2_model=float(score2[j_loc]),
        )

    refined_type = np.array([None] * adata_st.n_obs, dtype=object)
    refined_score = np.zeros(adata_st.n_obs, dtype=float)
    refined_type[idx_st_pair] = best_type_pair
    refined_score[idx_st_pair] = best_score_pair

    return dict(
        mask_pair=mask_pair,
        refined_type=refined_type,
        refined_score=refined_score,
        weights_dict=weights_dict,
        debug=dict(
            idx_st_pair=idx_st_pair,
            idx_sc_pair=idx_sc_pair,
            n_g1=int(mask_g1.sum()),
            n_g2=int(mask_g2.sum()),

        )
    )

def refine_single_ct(
    adata_sc,
    adata_st,
    *,
    ct: str,
    groupby_sc="cell_type",
    global_type_key=OBS_GLOBAL_TYPE,
 # geometry for FGW (C_sc, C_st)
    sc_geom_obsm_key="X_scvi",
    st_geom_obsm_key="spatial_normed",
    sc_geom_obs_keys=None,
    st_geom_obs_keys=None,
 # anchor cost
    use_anchor_cost=False,
    anchor_key=OBS_GLOBAL_ANCHOR,
    beta_expr=0.7,
 # solver
    solver=None,
    solver_kwargs=None,
    alpha_fgw=None, # deprecated (use solver.alpha)
 # scoring
    precomp=None, # output of util.precomp.prepare_precomp_for_scoring
    cos_threshold=None,
    score_n_jobs=1,
 # optional decision
    min_score=None, # e.g. 0.6; if provided, below -> None
    M_norm: str = "mean",
    C_kind_sc: str = "distance",
    C_kind_st: str = "distance",
    C_norm: str = "q95",
    C_clip_q=None,
):
    """
    Single-type refinement (no bleeding/incoming; one FGW solve).

    Returns (same schema as refine_pair_ct1_ct2)
    -------
    out : dict with keys
      - mask_pair: (n_st,) bool             (here: mask_ct)
      - refined_type: (n_st,) object        (only for ct spots, else None)
      - refined_score: (n_st,) float        (only for ct spots, else 0)
      - weights_dict: dict[int st_idx -> dict(...)]
      - debug: dict(...)
    """
    if solver is None:
        raise ValueError("solver is required (e.g., util.POTFGWSolver(...)).")
    if precomp is None:
        raise ValueError("precomp is required (prepare_precomp_for_scoring output).")

    solver_kwargs = {} if solver_kwargs is None else dict(solver_kwargs)
    if alpha_fgw is not None:
        solver_kwargs.setdefault("alpha", float(alpha_fgw))

    ct = str(ct)

 # ST subset: only this ct from global mapping
    idx_st_ct, mask_ct = _subset_indices_by_types(adata_st.obs[global_type_key], ct, ct)
 # NOTE: if your _subset_indices_by_types expects (ct1, ct2) and returns union,
 # passing (ct, ct) should give only ct; if not, replace with direct mask:
 # mask_ct = np.asarray(adata_st.obs[global_type_key], object) == ct
 # idx_st_ct = np.where(mask_ct)[0]

    if idx_st_ct.size == 0:
        return dict(
            mask_pair=mask_ct,
            refined_type=np.array([None]*adata_st.n_obs, object),
            refined_score=np.zeros(adata_st.n_obs, float),
            weights_dict={},
            debug={"reason": "no_st_ct"},
        )

 # SC subset: only this ct
    idx_sc_ct, _ = _subset_indices_by_types(adata_sc.obs[groupby_sc], ct, ct)
 # or direct mask similarly if needed

    if idx_sc_ct.size == 0:
        return dict(
            mask_pair=mask_ct,
            refined_type=np.array([None]*adata_st.n_obs, object),
            refined_score=np.zeros(adata_st.n_obs, float),
            weights_dict={},
            debug={"reason": "no_sc_ct"},
        )

 # gene panel: fixed (no bleeding/incoming)
    genes_union = np.asarray(precomp["genes_union"], dtype=object)
    if genes_union.size < 3:
        return dict(
            mask_pair=mask_ct,
            refined_type=np.array([None]*adata_st.n_obs, object),
            refined_score=np.zeros(adata_st.n_obs, float),
            weights_dict={},
            debug={"reason": "too_few_genes", "n_genes": int(genes_union.size)},
        )

 # features
    X_sc_all = precomp["X_sc_exp"]
    X_st_all = precomp["X_st_exp"]
    X_sc_ct = X_sc_all[idx_sc_ct] # (n_sc_ct, G)
    X_st_ct = X_st_all[idx_st_ct] # (n_st_ct, G)

 # geometry for C matrices
    sc_geom_all = _get_geom(
        adata_sc, obsm_key=sc_geom_obsm_key, obs_keys=sc_geom_obs_keys, fallback_X=X_sc_all
    )
    st_geom_all = _get_geom(
        adata_st, obsm_key=st_geom_obsm_key, obs_keys=st_geom_obs_keys, fallback_X=X_st_all
    )

    if C_kind_sc == 'distnace':
        C_sc_raw = build_C_euclidean(sc_geom_all[idx_sc_ct], metric="euclidean", normalize= None)

    elif C_kind_sc == 'metacell_distnace':
        C_sc_raw = _normalize_C(adata_sc.obsp['mf_distance_matrix'], normalize = None)

    C_st_raw = build_C_euclidean(st_geom_all[idx_st_ct], metric="euclidean", normalize= None)

 # optional anchor cost (corridor = this ct spots)
    M_anchor = None
    if use_anchor_cost:
        M_anchor = build_anchor_cost_pair(
            adata_sc, adata_st,
            idx_sc_pair=idx_sc_ct,
            idx_st_pair=idx_st_ct,
            sc_geom_all=sc_geom_all,
            st_geom_all=st_geom_all,
            anchor_key=anchor_key,
        )

    rr = run_transport(
        X_sc_feat=X_sc_ct,
        X_st_feat=X_st_ct,
        C_sc_raw=C_sc_raw,
        C_st_raw=C_st_raw,
        solver=solver,
        M_anchor=M_anchor,
        beta_expr=beta_expr,
        M_norm=str(M_norm),
        C_kind_sc=str(C_kind_sc),
        C_kind_st=str(C_kind_st),
        C_norm=str(C_norm),
        C_clip_q=C_clip_q,
        **solver_kwargs,
    )
    W = rr.W

 # mixture scoring (single)
    score_ct, _ = mixture_scores_from_weights(
        W, idx_sc_ct, idx_st_ct, precomp,
        cos_threshold=cos_threshold, n_jobs=score_n_jobs
    )

 # final assignment: ct (optionally drop low-score)
    best_type_ct = np.array([ct] * len(idx_st_ct), dtype=object)
    if min_score is not None:
        best_type_ct = np.where(score_ct >= float(min_score), best_type_ct, None)

 # store weights per st
    weights_dict = {}
    for j_loc, st_idx in enumerate(idx_st_ct):
        weights_dict[int(st_idx)] = dict(
            best_type=(best_type_ct[j_loc]),
            model="single_ct",
            sc_idx=np.asarray(idx_sc_ct, dtype=int).copy(),
            w=np.asarray(W[:, j_loc], dtype=float).copy(),
            score=float(score_ct[j_loc]),
        )

    refined_type = np.array([None] * adata_st.n_obs, dtype=object)
    refined_score = np.zeros(adata_st.n_obs, dtype=float)
    refined_type[idx_st_ct] = best_type_ct
    refined_score[idx_st_ct] = score_ct

    return dict(
        mask_pair=mask_ct,
        refined_type=refined_type,
        refined_score=refined_score,
        weights_dict=weights_dict,
        debug=dict(
            idx_st_ct=np.asarray(idx_st_ct, dtype=int),
            idx_sc_ct=np.asarray(idx_sc_ct, dtype=int),
            n_genes=int(genes_union.size),
        ),
    )

def refine_pair_ct1_ct2_selected_genes(
    adata_sc,
    adata_st,
    *,
    ct1: str,
    ct2: str,
    selected_genes, # dict[ct -> DataFrame (genes in .index) or iterable[str]]
    bleeding_obj=None, # output of compute_bleeding_directional
    incoming_obj=None, # output of compute_incoming_bleeding (optional)
    bleeding_mode="bleeding", # "bleeding" | "incoming" | "none"
    groupby_sc="cell_type",
    global_type_key=OBS_GLOBAL_TYPE,
 # geometry for FGW (C_sc, C_st)
    sc_geom_obsm_key="X_scvi",
    st_geom_obsm_key="spatial_normed",
    sc_geom_obs_keys=None,
    st_geom_obs_keys=None,
 # geometry builder options (kept identical to refine_pair_ct1_ct2)
    sc_C_mode="euclidean", # "euclidean" | "knn_dense" | "knn_geodesic" | "metacell_distance"
    st_C_mode="euclidean", # "euclidean" | "knn_dense" | "knn_geodesic"
    sc_knn_key_added="neighbors_sc",
    st_knn_key_added="neighbors_st",
    knn_fill_value="max",
    knn_directed=False,
 # anchor cost
    use_anchor_cost=True,
    anchor_key=OBS_GLOBAL_ANCHOR,
    beta_expr=1,
 # transport normalization / C adaptation (kept in sync with final-global)
    M_norm: str = "mean",
    C_kind_sc: str = "distance",
    C_kind_st: str = "distance",
    C_norm: str = "q95",
    C_clip_q=None,
 # solver
    solver=None,
    solver_kwargs=None,
    alpha_fgw=None, # deprecated (use solver.alpha)
 # scoring
    precomp=None, # output of util.precomp.prepare_precomp_for_scoring
    cos_threshold=None,
    score_n_jobs=1,
 # decisions
    mass_tie_eps=0,
 # selected-genes specific
    min_genes=3,
):
    """
    Concept
    -------
    refine_pair_ct1_ct2  / ,
      DEG  selected_genes(CT ) .

    Gene panel policy
    -----------------
      - ct1 : selected_genes[ct1]
      - ct2 : selected_genes[ct2]
      - bleeding_mode    bleeding/incoming
      - precomp["genes_union"]   X_sc_exp/X_st_exp

    Returns
    -------
    out : dict with keys (refine_pair_ct1_ct2 )
      - mask_pair: (n_st,) bool
      - refined_type: (n_st,) object (only for pair spots, else None)
      - refined_score: (n_st,) float  (only for pair spots, else 0)
      - weights_dict: dict[int st_idx -> dict(...)]
      - debug: dict(...)
    """
    if solver is None:
        raise ValueError("solver is required (e.g., util.POTFGWSolver(...)).")
    if precomp is None:
        raise ValueError("precomp is required (prepare_precomp_for_scoring output).")
    if selected_genes is None:
        raise ValueError("selected_genes is required.")

    solver_kwargs = {} if solver_kwargs is None else dict(solver_kwargs)
    if alpha_fgw is not None:
 # keep compatibility; prefer solver.alpha
        solver_kwargs.setdefault("alpha", float(alpha_fgw))

    ct1 = str(ct1)
    ct2 = str(ct2)

 # ------------------------------------------------------------
 # 0) corridor indices
 # ------------------------------------------------------------
    idx_st_pair, mask_pair = _subset_indices_by_types(adata_st.obs[global_type_key], ct1, ct2)
    if idx_st_pair.size == 0:
        return dict(
            mask_pair=mask_pair,
            refined_type=np.array([None] * adata_st.n_obs, dtype=object),
            refined_score=np.zeros(adata_st.n_obs, dtype=float),
            weights_dict={},
            debug={"reason": "no_st_pair"},
        )

    idx_sc_pair, _ = _subset_indices_by_types(adata_sc.obs[groupby_sc], ct1, ct2)
    if idx_sc_pair.size == 0:
        return dict(
            mask_pair=mask_pair,
            refined_type=np.array([None] * adata_st.n_obs, dtype=object),
            refined_score=np.zeros(adata_st.n_obs, dtype=float),
            weights_dict={},
            debug={"reason": "no_sc_pair"},
        )

    genes_union = np.asarray(precomp["genes_union"], dtype=object)

 # ------------------------------------------------------------
 # 1) gene panels from selected_genes (ct-specific)
 # ------------------------------------------------------------
    def _genes_for_ct(ct):
        if ct not in selected_genes:
            return None
        obj = selected_genes[ct]
 # pandas DataFrame/Series: genes in index
        if hasattr(obj, "index"):
            return list(map(str, list(obj.index)))
 # iterable[str]
        return list(map(str, list(obj)))

    g1_raw = _genes_for_ct(ct1)
    g2_raw = _genes_for_ct(ct2)
    if g1_raw is None or g2_raw is None:
        return dict(
            mask_pair=mask_pair,
            refined_type=np.array([None] * adata_st.n_obs, dtype=object),
            refined_score=np.zeros(adata_st.n_obs, dtype=float),
            weights_dict={},
            debug={
                "reason": "ct_not_in_selected_genes",
                "missing": [c for c in [ct1, ct2] if c not in selected_genes],
            },
        )

    G1 = set(g1_raw)
    G2 = set(g2_raw)

 # ------------------------------------------------------------
 # ------------------------------------------------------------
    if bleeding_mode == "none":
        pass

    elif bleeding_mode == "incoming":
        if incoming_obj is None:
            raise ValueError("bleeding_mode='incoming' requires incoming_obj.")
        incoming = incoming_obj.get("incoming_per_target", {}) or {}
        inc1 = set(map(str, incoming.get(ct1, set()) or []))
        inc2 = set(map(str, incoming.get(ct2, set()) or []))
        G1 = G1 - inc1
        G2 = G2 - inc2

    elif bleeding_mode == "bleeding":
        if bleeding_obj is None:
            raise ValueError("bleeding_mode='bleeding' requires bleeding_obj.")
        bleed = bleeding_obj.get("bleeding_pairs", {}) or {}
        b12 = set(map(str, bleed.get((ct1, ct2), set()) or []))
        b21 = set(map(str, bleed.get((ct2, ct1), set()) or []))
        G1 = G1 - b12
        G2 = G2 - b21

    else:
        raise ValueError("bleeding_mode must be 'bleeding' | 'incoming' | 'none'.")

    mask_g1 = np.isin(genes_union, list(G1))
    mask_g2 = np.isin(genes_union, list(G2))

    n_g1 = int(mask_g1.sum())
    n_g2 = int(mask_g2.sum())
    if n_g1 < int(min_genes) or n_g2 < int(min_genes):
        return dict(
            mask_pair=mask_pair,
            refined_type=np.array([None] * adata_st.n_obs, dtype=object),
            refined_score=np.zeros(adata_st.n_obs, dtype=float),
            weights_dict={},
            debug={"reason": "too_few_genes", "n_g1": n_g1, "n_g2": n_g2},
        )

 # ------------------------------------------------------------
 # 3) features (direction-specific masks)
 # ------------------------------------------------------------
    X_sc_all = precomp["X_sc_exp"]
    X_st_all = precomp["X_st_exp"]

    X_sc_pair_g1 = X_sc_all[idx_sc_pair][:, mask_g1]
    X_st_pair_g1 = X_st_all[idx_st_pair][:, mask_g1]
    X_sc_pair_g2 = X_sc_all[idx_sc_pair][:, mask_g2]
    X_st_pair_g2 = X_st_all[idx_st_pair][:, mask_g2]

 # ------------------------------------------------------------
 # ------------------------------------------------------------
    sc_geom_all = _get_geom(adata_sc, obsm_key=sc_geom_obsm_key, obs_keys=sc_geom_obs_keys, fallback_X=X_sc_all)
    st_geom_all = _get_geom(adata_st, obsm_key=st_geom_obsm_key, obs_keys=st_geom_obs_keys, fallback_X=X_st_all)

    if sc_C_mode == "euclidean":
        C_sc_raw = build_C_euclidean(sc_geom_all[idx_sc_pair], metric="euclidean", normalize= None)

    elif sc_C_mode == "metacell_distance":
        Cfull = adata_sc.obsp["mf_distance_matrix"]
        Csub = Cfull[idx_sc_pair, :][:, idx_sc_pair]
        C_sc_raw = _normalize_C(np.asarray(Csub, dtype=float), normalize= None)

    elif sc_C_mode in ("knn_dense", "knn_geodesic"):
        C_sc_raw = build_C_from_knn(
            adata_sc,
            idx=idx_sc_pair,
            key_added=sc_knn_key_added,
            mode=("dense" if sc_C_mode == "knn_dense" else "geodesic"),
            fill_value=knn_fill_value,
            directed=knn_directed,
        )
    else:
        raise ValueError("sc_C_mode must be 'euclidean'|'knn_dense'|'knn_geodesic'|'metacell_distance'")

    if st_C_mode == "euclidean":
        C_st_raw = build_C_euclidean(st_geom_all[idx_st_pair], metric="euclidean", normalize= None)
    elif st_C_mode in ("knn_dense", "knn_geodesic"):
        C_st_raw = build_C_from_knn(
            adata_st,
            idx=idx_st_pair,
            key_added=st_knn_key_added,
            mode=("dense" if st_C_mode == "knn_dense" else "geodesic"),
            fill_value=knn_fill_value,
            directed=knn_directed,
        )
    else:
        raise ValueError("st_C_mode must be 'euclidean'|'knn_dense'|'knn_geodesic'")

 # ------------------------------------------------------------
 # 5) anchor cost (shared for both directions)
 # ------------------------------------------------------------
    M_anchor = None
    if use_anchor_cost:
        M_anchor = build_anchor_cost_pair(
            adata_sc,
            adata_st,
            idx_sc_pair=idx_sc_pair,
            idx_st_pair=idx_st_pair,
            sc_geom_all=sc_geom_all,
            st_geom_all=st_geom_all,
            anchor_key=anchor_key,
        )

 # ------------------------------------------------------------
 # ------------------------------------------------------------
    rr1 = run_transport(
        X_sc_feat=X_sc_pair_g1,
        X_st_feat=X_st_pair_g1,
        C_sc_raw=C_sc_raw,
        C_st_raw=C_st_raw,
        solver=solver,
        M_anchor=M_anchor,
        beta_expr=beta_expr,
        M_norm=str(M_norm),
        C_kind_sc=str(C_kind_sc),
        C_kind_st=str(C_kind_st),
        C_norm=str(C_norm),
        C_clip_q=C_clip_q,
        **solver_kwargs,
    )
    rr2 = run_transport(
        X_sc_feat=X_sc_pair_g2,
        X_st_feat=X_st_pair_g2,
        C_sc_raw=C_sc_raw,
        C_st_raw=C_st_raw,
        solver=solver,
        M_anchor=M_anchor,
        beta_expr=beta_expr,
        M_norm=str(M_norm),
        C_kind_sc=str(C_kind_sc),
        C_kind_st=str(C_kind_st),
        C_norm=str(C_norm),
        C_clip_q=C_clip_q,
        **solver_kwargs,
    )

    W1 = rr1.W
    W2 = rr2.W

 # ------------------------------------------------------------
 # ------------------------------------------------------------
    score1, _ = mixture_scores_from_weights(
        W1,
        idx_sc_pair,
        idx_st_pair,
        precomp,
        gene_mask=mask_g1,
        cos_threshold=cos_threshold,
        n_jobs=score_n_jobs,
    )
    score2, _ = mixture_scores_from_weights(
        W2,
        idx_sc_pair,
        idx_st_pair,
        precomp,
        gene_mask=mask_g2,
        cos_threshold=cos_threshold,
        n_jobs=score_n_jobs,
    )

    use_model1 = score1 >= score2
    best_score_pair = np.where(use_model1, score1, score2)

 # ------------------------------------------------------------
 # 8) decide type by mass(W) within ct1/ct2 among idx_sc_pair
 # ------------------------------------------------------------
    sc_types_pair = np.asarray(adata_sc.obs[groupby_sc], dtype=object)[idx_sc_pair]
    mask_ct1 = (sc_types_pair == ct1)
    mask_ct2 = (sc_types_pair == ct2)

    weights_dict = {}
    best_type_pair = np.empty(len(idx_st_pair), dtype=object)

    for j_loc, st_idx in enumerate(idx_st_pair):
        Wj = (W1[:, j_loc] if use_model1[j_loc] else W2[:, j_loc])
        mass1 = float(Wj[mask_ct1].sum())
        mass2 = float(Wj[mask_ct2].sum())

        if abs(mass1 - mass2) <= float(mass_tie_eps):
 # tie -> fallback to model scores
            if score1[j_loc] > score2[j_loc]:
                best_ct = ct1
            elif score2[j_loc] > score1[j_loc]:
                best_ct = ct2
            else:
                best_ct = None
        else:
            best_ct = ct1 if mass1 > mass2 else ct2

        best_type_pair[j_loc] = best_ct

        weights_dict[int(st_idx)] = dict(
            best_type=best_ct,
            model=("ct1_model" if use_model1[j_loc] else "ct2_model"),
            sc_idx=np.asarray(idx_sc_pair, dtype=int).copy(),
            w=np.asarray(Wj, dtype=float).copy(),
            mass_ct1=mass1,
            mass_ct2=mass2,
            score_ct1_model=float(score1[j_loc]),
            score_ct2_model=float(score2[j_loc]),
        )

 # ------------------------------------------------------------
 # 9) restore to full n_st arrays (same output contract)
 # ------------------------------------------------------------
    refined_type = np.array([None] * adata_st.n_obs, dtype=object)
    refined_score = np.zeros(adata_st.n_obs, dtype=float)
    refined_type[idx_st_pair] = best_type_pair
    refined_score[idx_st_pair] = best_score_pair

    return dict(
        mask_pair=mask_pair,
        refined_type=refined_type,
        refined_score=refined_score,
        weights_dict=weights_dict,
        debug=dict(
            idx_st_pair=idx_st_pair,
            idx_sc_pair=idx_sc_pair,
            n_g1=n_g1,
            n_g2=n_g2,
            bleeding_mode=str(bleeding_mode),
            sc_C_mode=str(sc_C_mode),
            st_C_mode=str(st_C_mode),
        ),
    )

def assign_anchors_from_pairwise(
    adata_st,
    *,
    sim_threshold=None,
    top_frac=1,
    type_key=OBS_FINAL_TYPE,
    score_key=OBS_FINAL_SCORE,
    out_anchor_type_key=OBS_FINAL_ANCHOR_TYPE,
    out_anchor_key=OBS_FINAL_ANCHOR,
    write_to_obs=True,
    invalid_value=None, # what to write for invalid anchors (None is safest)
):
    """
    Assign anchors using stored *pairwise* mapping results.

    Rules
    -----
    valid_anchor(j) =
        (typewise-top-frac rule if top_frac is not None)
        AND
        (score >= sim_threshold if sim_threshold is not None)

    Notes
    -----
    - Pairwise outputs typically have:
        - final_pairwise_type  (per-spot predicted type)
        - final_pairwise_score (per-spot score)
      but NOT a stable "best_cell" id.
    - Therefore this function stores anchor *type* by default.
      (out_anchor_key is written as None for valid rows unless you customize later.)

    Parameters
    ----------
    adata_st : AnnData
    sim_threshold : float or None
    top_frac : float in (0, 1] or None
        If provided, within each predicted type group, keep top top_frac fraction by score.
    type_key / score_key : str
        Where pairwise outputs live in adata_st.obs.
    out_anchor_type_key / out_anchor_key : str
        Where to write anchor assignments in adata_st.obs.
    write_to_obs : bool
    invalid_value : object
        Value to write for invalid anchors (default None).

    Returns
    -------
    valid : np.ndarray[bool], shape (n_st,)
    """
    for k in (type_key, score_key):
        if k not in adata_st.obs:
            raise KeyError(f"[assign_anchors_from_pairwise] missing adata_st.obs['{k}']")

    score = np.asarray(adata_st.obs[score_key], dtype=float)
    spot_type = np.asarray(adata_st.obs[type_key]).astype(object)

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
            thr = np.quantile(score[idx], 1.0 - tf)
            valid[idx] = score[idx] >= thr

 # (B) absolute threshold
    if sim_threshold is not None:
        valid &= (score >= float(sim_threshold))

 # Build outputs
    anchor_type = spot_type.copy()
    anchor_type[~valid] = invalid_value

 # pairwise usually doesn't have "best_cell"; keep a placeholder column for compatibility
    anchor_cell = np.full(n_st, invalid_value, dtype=object)
 # if you later decide to store some per-spot representative id, you can fill anchor_cell[valid] there.

    if write_to_obs:
        adata_st.obs[out_anchor_type_key] = anchor_type
        adata_st.obs[out_anchor_key] = anchor_cell

    return valid

def select_type_pairs_from_A(
    A: pd.DataFrame,
    *,
    drop_self: bool = True,
    undirected: bool = True,
    top_n: int | None = None,
    top_frac: float | None = None,
    min_weight: float | None = None,
) -> pd.DataFrame:
    """
    A: type x type adjacency (DataFrame),   ()
    : ['type_a','type_b','weight'] DataFrame ()

      :
      1) min_weight
      2) top_n  top_frac  (   top_n )
    """
    if not isinstance(A, pd.DataFrame):
        raise TypeError("A must be a pandas DataFrame with index/columns as type names.")

    types = list(A.index.astype(str))
    M = A.to_numpy(dtype=float, copy=True)

    if drop_self:
        np.fill_diagonal(M, 0.0)

    rows, cols = np.nonzero(M)
    w = M[rows, cols]
    if w.size == 0:
        return pd.DataFrame(columns=["type_a", "type_b", "weight"])

    df = pd.DataFrame({
        "type_a": [types[i] for i in rows],
        "type_b": [types[j] for j in cols],
        "weight": w.astype(float),
    })

    if undirected:
        key = df.apply(lambda r: tuple(sorted((r["type_a"], r["type_b"]))), axis=1)
        df["pair"] = key
        df = (df.groupby("pair", as_index=False)["weight"].sum()
                .rename(columns={"pair": "pair"}))
        df["type_a"] = df["pair"].apply(lambda p: p[0])
        df["type_b"] = df["pair"].apply(lambda p: p[1])
        df = df.drop(columns=["pair"])

    if min_weight is not None:
        df = df[df["weight"] >= float(min_weight)]

    df = df.sort_values("weight", ascending=False).reset_index(drop=True)

 # top_n / top_frac
    if top_n is not None:
        df = df.head(int(top_n)).reset_index(drop=True)
    elif top_frac is not None:
        k = max(1, int(np.ceil(len(df) * float(top_frac))))
        df = df.head(k).reset_index(drop=True)

    return df


def _ckdtree_query(tree: cKDTree, X: np.ndarray, k: int, workers: int = 1):
    """SciPy version compatibility for cKDTree.query parallel arg."""
    try:
        return tree.query(X, k=k, workers=workers)
    except TypeError:
        try:
            return tree.query(X, k=k, n_jobs=workers)
        except TypeError:
            return tree.query(X, k=k)


def build_st_pairs_from_knn(
    adata_st,
    *,
    ct_key: str = OBS_GLOBAL_TYPE,  # usually "global_sc_type"
    spatial_key_prefer=("spatial_normed", "spatial"),
    xy_cols=None,
    k: int = 10,
    workers: int = 1,
    strip_types: bool = True,
    drop_self: bool = True,
):
    """Build ST-only cell-type pairs from spatial kNN.

    Notes
    -----
    - Uses *spot-level* spatial kNN based on `spatial_key_prefer`.
    - Produces *undirected* type pairs by normalizing each directed edge
      (A->B) to an unordered pair (min(A,B), max(A,B)).
    - Weight is the *symmetric sum* of directed neighbor counts.

    Returns
    -------
    pairs_df_st : pd.DataFrame
        Columns: ["ct1", "ct2", "weight"], with ct1 < ct2.
    centroid_df : pd.DataFrame
        Index: cell_type; columns: ["x", "y", "n_spots"], where (x,y) is mean coord.
    """
    if ct_key not in adata_st.obs:
        raise KeyError(f"'{ct_key}' not found in adata_st.obs")

    k = int(k)
    if k <= 0:
        raise ValueError("k must be >= 1")
    if adata_st.n_obs <= 1:
        raise ValueError("adata_st must contain at least 2 observations")

    # choose spatial key
    spatial_key = None
    for cand in spatial_key_prefer:
        if cand in adata_st.obsm:
            spatial_key = cand
            break
    if spatial_key is None:
        spatial_key = spatial_key_prefer[0] if len(spatial_key_prefer) else "spatial_normed"

    X = _get_spatial_coords(adata_st, spatial_key=spatial_key, xy_cols=xy_cols)
    X = np.asarray(X, dtype=np.float32)
    X = np.ascontiguousarray(X)
    if not np.isfinite(X).all():
        raise ValueError(f"Spatial coordinates contain NaN/Inf (key='{spatial_key}').")

    # labels
    ct = adata_st.obs[ct_key].astype(str)
    if strip_types:
        ct = ct.str.strip()
    ct = ct.to_numpy(dtype=object)

    # centroid per type (mean)
    df_tmp = pd.DataFrame({"ct": ct, "x": X[:, 0].astype(float), "y": X[:, 1].astype(float)})
    g = df_tmp.groupby("ct", sort=True)
    centroid_df = pd.DataFrame({
        "x": g["x"].mean(),
        "y": g["y"].mean(),
        "n_spots": g.size().astype(int),
    })
    centroid_df.index.name = "cell_type"

    # kNN (query k+1 then drop self)
    n = X.shape[0]
    if k >= n:
        raise ValueError(f"k={k} must be < n_obs={n} (self removed).")
    tree = cKDTree(X)
    _, nn = _ckdtree_query(tree, X, k=k + 1, workers=workers)
    nn = np.asarray(nn, dtype=np.int64)

    # Expect self in first column; if not, we'll filter self explicitly.
    nn = nn[:, 1:]
    src = np.repeat(np.arange(n, dtype=np.int64), k)
    dst = nn.reshape(-1)

    # safeguard (in case self wasn't first)
    mask_not_self = dst != src
    if not np.all(mask_not_self):
        src = src[mask_not_self]
        dst = dst[mask_not_self]

    ct_src = ct[src]
    ct_dst = ct[dst]

    # undirected normalization (ct1 < ct2 lexicographically)
    le = ct_src <= ct_dst
    ct1 = np.where(le, ct_src, ct_dst)
    ct2 = np.where(le, ct_dst, ct_src)

    if drop_self:
        m = ct1 != ct2
        ct1 = ct1[m]
        ct2 = ct2[m]

    pairs_df_st = pd.DataFrame({"ct1": ct1, "ct2": ct2})
    if pairs_df_st.empty:
        pairs_df_st = pd.DataFrame(columns=["ct1", "ct2", "weight"])
        return pairs_df_st, centroid_df

    pairs_df_st["weight"] = 1
    pairs_df_st = (
        pairs_df_st.groupby(["ct1", "ct2"], sort=True, as_index=False)["weight"]
        .sum()
        .sort_values(["weight", "ct1", "ct2"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
    pairs_df_st["weight"] = pairs_df_st["weight"].astype(int)

    return pairs_df_st, centroid_df


def finalize_pairs(
    pairs_df_st: pd.DataFrame,
    *,
    threshold: int,
    exclude_sc: bool = False,
    sc_pair_csv_path: str | None = None,
    sc_pair_cols=("cell_type1", "cell_type2"),
    strip_types: bool = True,
):
    """Apply weight threshold and optionally remove SC/metacell pairs from a CSV.

    Returns only the final pair table (ct1, ct2, weight).
    """
    if not isinstance(pairs_df_st, pd.DataFrame):
        raise TypeError("pairs_df_st must be a pandas DataFrame")
    for col in ("ct1", "ct2", "weight"):
        if col not in pairs_df_st.columns:
            raise KeyError(f"pairs_df_st must contain column '{col}'")

    thr = int(threshold)
    out = pairs_df_st.copy()
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce")
    out = out.dropna(subset=["ct1", "ct2", "weight"])
    out = out[out["weight"] >= thr]
    out["ct1"] = out["ct1"].astype(str)
    out["ct2"] = out["ct2"].astype(str)
    if strip_types:
        out["ct1"] = out["ct1"].str.strip()
        out["ct2"] = out["ct2"].str.strip()

    # ensure undirected normalization
    le = out["ct1"].to_numpy(object) <= out["ct2"].to_numpy(object)
    ct1 = np.where(le, out["ct1"].to_numpy(object), out["ct2"].to_numpy(object))
    ct2 = np.where(le, out["ct2"].to_numpy(object), out["ct1"].to_numpy(object))
    out["ct1"] = ct1
    out["ct2"] = ct2
    out = out[out["ct1"] != out["ct2"]]

    if exclude_sc:
        if sc_pair_csv_path is None:
            raise ValueError("exclude_sc=True requires sc_pair_csv_path")
        c1, c2 = sc_pair_cols
        sc_df = pd.read_csv(sc_pair_csv_path)
        if c1 not in sc_df.columns or c2 not in sc_df.columns:
            raise KeyError(f"SC CSV must contain columns {sc_pair_cols}")

        s1 = sc_df[c1].astype(str)
        s2 = sc_df[c2].astype(str)
        if strip_types:
            s1 = s1.str.strip()
            s2 = s2.str.strip()

        le2 = s1 <= s2
        sc_ct1 = np.where(le2, s1.to_numpy(object), s2.to_numpy(object))
        sc_ct2 = np.where(le2, s2.to_numpy(object), s1.to_numpy(object))
        m = sc_ct1 != sc_ct2
        sc_pairs = set(zip(sc_ct1[m].tolist(), sc_ct2[m].tolist()))

        if sc_pairs:
            keep_mask = ~out.apply(lambda r: (r["ct1"], r["ct2"]) in sc_pairs, axis=1)
            out = out.loc[keep_mask]

    out["weight"] = out["weight"].astype(int)
    out = out.sort_values(["weight", "ct1", "ct2"], ascending=[False, True, True]).reset_index(drop=True)
    return out[["ct1", "ct2", "weight"]]