"""
Unpaired single-type refinement

After pairwise refinement + merge (final_pairwise_*), some cell types may have
never appeared in any candidate pair (ct1, ct2). For those *unpaired* types,
we still allow mixture refinement (FGW-based) but WITHOUT competition:

- Keep the cell type fixed (do NOT change final type).
- Solve FGW between SC cells of that type and ST spots currently assigned to that type.
- Build mixture per spot and re-score (same score function as pairwise pipeline).
- Update only those spots whose new score improves the current final score
  (spot-wise accept/reject), optionally with a delta margin.

Bleeding subtraction is optional here: by default we keep full genes_union,
    but if incoming_bleeding_obj is provided we remove incoming-bleeding genes for stability.
"""

from __future__ import annotations

import numpy as np
from .array_ops import col_normalize

from .keys import (
    OBS_FINAL_TYPE,
    OBS_FINAL_SCORE,
    OBS_GLOBAL_TYPE,
    OBS_GLOBAL_SIM,
    UNS_FINAL_WEIGHTS,
)
from .fgw_solver import build_intra_dist, compute_fgw_plan
from .precomp import mixture_scores_from_weights
from .prep import get_geom

def _get_geom(adata, *, obsm_key=None, obs_keys=None, fallback_X=None):
    """Backwards-compatible alias (prefer util.prep.get_geom)."""
    return get_geom(adata, obsm_key=obsm_key, obs_keys=obs_keys, fallback_X=fallback_X)

def run_unpaired_single_type_refinement(
    adata_sc,
    adata_st,
    *,
    pairs, # list[(ct1,ct2)] or list[dict] with keys type_a/type_b
    selected_genes, # dict[type -> list[str]]
    precomp, # util.precomp.prepare_precomp_for_scoring output
    incoming_bleeding_obj=None, # optional: use genes_union - incoming_bleeding(ct)
    cell_type_key_sc: str = "cell_type",
    final_type_key: str = OBS_FINAL_TYPE,
    final_score_key: str = OBS_FINAL_SCORE,
    global_type_key: str = OBS_GLOBAL_TYPE,
    global_score_key: str = OBS_GLOBAL_SIM,
    delta_accept: float = 0.0,

 # geometry
    sc_geom_obsm_key: str | None = None,
    sc_geom_obs_keys=None, # tuple/list of obs column names
    st_geom_obsm_key: str | None = None,
    st_geom_obs_keys=None, # tuple/list of obs column names
    st_spatial_key: str = "spatial",
    fallback_to_X_sc: bool = True,
    fallback_to_X_st: bool = True,

 # FGW / scoring
    solver=None, # instance of POTFGWSolver (util.ot_solvers)
    solver_kwargs=None,
    beta_expr: float = 0.7, # for API symmetry (anchor mix), unused when M_anchor=None
    cos_threshold=None,
    score_n_jobs: int = 1,
    min_genes: int = 5,
):
    """
    Returns dict:
      - unpaired_types
      - accepted_mask (n_st,)
      - n_updated
    """
    if solver is None:
        raise ValueError("solver must be provided (e.g., POTFGWSolver).")
    if solver_kwargs is None:
        solver_kwargs = {}

 # paired types from `pairs`
    paired = set()
    for p in pairs:
        if isinstance(p, dict):
            a = p.get("type_a", None) or p.get("ct1", None) or p.get("type1", None)
            b = p.get("type_b", None) or p.get("ct2", None) or p.get("type2", None)
            if a is not None: paired.add(str(a))
            if b is not None: paired.add(str(b))
        else:
            a, b = p
            paired.add(str(a)); paired.add(str(b))

    all_types = list(np.unique(np.asarray(adata_sc.obs[cell_type_key_sc], dtype=object)))
    unpaired_types = [c for c in all_types if c not in paired]

    genes_union = np.asarray(precomp["genes_union"], dtype=object)
    X_sc_all = precomp["X_sc_exp"]
    X_st_all = precomp["X_st_exp"]

    sc_geom_all = _get_geom(
        adata_sc,
        obsm_key=sc_geom_obsm_key,
        obs_keys=sc_geom_obs_keys,
        fallback_X=(X_sc_all if fallback_to_X_sc else None),
    )

    if st_geom_obsm_key is None and st_geom_obs_keys is None:
        st_geom_obsm_key = st_spatial_key
    st_geom_all = _get_geom(
        adata_st,
        obsm_key=st_geom_obsm_key,
        obs_keys=st_geom_obs_keys,
        fallback_X=(X_st_all if fallback_to_X_st else None),
    )

 # current final arrays
    if final_type_key in adata_st.obs:
        cur_type = np.asarray(adata_st.obs[final_type_key], dtype=object)
    else:
        cur_type = np.asarray(adata_st.obs[global_type_key], dtype=object)

    if final_score_key in adata_st.obs:
        cur_score = np.asarray(adata_st.obs[final_score_key], dtype=float)
    else:
        cur_score = np.asarray(adata_st.obs[global_score_key], dtype=float)

    weights_store = adata_st.uns.get(UNS_FINAL_WEIGHTS, {}) or {}
    accepted = np.zeros(adata_st.n_obs, dtype=bool)
    n_updated = 0

    sc_labels = np.asarray(adata_sc.obs[cell_type_key_sc], dtype=object)

    for ct in unpaired_types:
        idx_st = np.where(cur_type == ct)[0]
        if idx_st.size == 0:
            continue
        idx_sc = np.where(sc_labels == ct)[0]
        if idx_sc.size == 0:
            continue

 # Gene panel policy (user-intended):
 #   base panel = genes_union (global selected genes)
 #   remove only incoming-bleeding genes for this (fixed) type, if provided
        if incoming_bleeding_obj is not None:
            incoming = incoming_bleeding_obj.get("incoming_per_target", {}) or {}
            inc = set(incoming.get(ct, set()) or [])
            mask_g = ~np.isin(genes_union, list(inc))
        else:
 # fallback to the original behavior if no incoming object is provided
            genes_ct = selected_genes.get(ct, None)
            if genes_ct is None:
                continue
            mask_g = np.isin(genes_union, np.asarray(genes_ct, dtype=object))

        if mask_g.sum() < int(min_genes):
            continue

        X_sc_feat = X_sc_all[idx_sc][:, mask_g]
        X_st_feat = X_st_all[idx_st][:, mask_g]

        C_sc = build_intra_dist(sc_geom_all[idx_sc], metric="euclidean")
        C_st = build_intra_dist(st_geom_all[idx_st], metric="euclidean")

        T, log, M = compute_fgw_plan(
            X_sc_feat, X_st_feat, C_sc, C_st,
            solver=solver,
            M_anchor=None,
            beta_expr=beta_expr,
            **solver_kwargs,
        )

        W = col_normalize(T)

        scores_new, _ = mixture_scores_from_weights(
            W, idx_sc, idx_st, precomp,
            cos_threshold=cos_threshold,
            n_jobs=score_n_jobs,
        )

        for j_loc, st_idx in enumerate(idx_st):
            s_new = float(scores_new[j_loc])
            if s_new > float(cur_score[st_idx]) + float(delta_accept):
                accepted[st_idx] = True
                cur_score[st_idx] = s_new
                n_updated += 1
                weights_store[int(st_idx)] = dict(
                    best_type=str(ct),
                    model="single_type_unpaired",
                    sc_idx=np.asarray(idx_sc, dtype=int).copy(),
                    w=np.asarray(W[:, j_loc], dtype=float).copy(),
                    score_single=float(s_new),
                )

    adata_st.obs[final_score_key] = cur_score
    adata_st.uns[UNS_FINAL_WEIGHTS] = weights_store

    return dict(
        unpaired_types=unpaired_types,
        accepted_mask=accepted,
        n_updated=int(n_updated),
    )