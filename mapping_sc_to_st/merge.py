"""
Merge utilities for combining multiple pairwise refinements into final labels.

Your core policy
----------------
- Start from global mapping results.
- For each ST spot, if any pairwise refinement improves the score, take the best.
- Store weights for the chosen refinement (so downstream steps can use them).
"""

from __future__ import annotations
import numpy as np
import copy #  added

from .keys import (
    OBS_GLOBAL_SIM,
    OBS_GLOBAL_TYPE,
    OBS_FINAL_TYPE,
    OBS_FINAL_SCORE,
    UNS_FINAL_WEIGHTS,
    UNS_FINAL_PAIR_WEIGHTS,
)

def merge_refinements(
    adata_st,
    refinements,
    *,
    global_score_key=OBS_GLOBAL_SIM,
    global_type_key=OBS_GLOBAL_TYPE,
    final_type_key=OBS_FINAL_TYPE,
    final_score_key=OBS_FINAL_SCORE,
    overwrite=True,
    only_when_improved=True,
):
    """
    Parameters
    ----------
    refinements : list[dict]
      Each dict is output of util.pairwise_refine.refine_pair_ct1_ct2.

    Writes
    ------
    adata_st.obs[final_type_key]
    adata_st.obs[final_score_key]
    adata_st.uns[UNS_FINAL_WEIGHTS]      : dict[st_idx -> weights_dict_entry]
    adata_st.uns[UNS_FINAL_PAIR_WEIGHTS] : list of per-pair weight dicts (for debugging)

    Backup (optional)
    -----------------
    If overwrite=True and backup_prev=True, the previous "final" state is saved as:
      adata_st.obs[f"{backup_name}_{final_type_key}"]
      adata_st.obs[f"{backup_name}_{final_score_key}"]
      adata_st.uns[f"{backup_name}_{UNS_FINAL_WEIGHTS}"]
      adata_st.uns[f"{backup_name}_{UNS_FINAL_PAIR_WEIGHTS}"]

    Returns
    -------
    out : dict with final arrays + stored weights
    """
    n_st = adata_st.n_obs
    global_score = np.asarray(adata_st.obs[global_score_key], dtype=float)
    global_type = np.asarray(adata_st.obs[global_type_key], dtype=object)

    best_score = global_score.copy()
    best_type = global_type.copy()

 # ------------------------------------------------------------
 # Unified weights container
 # ------------------------------------------------------------
 # Prefer already-initialized weights (from global_map). If missing, we
 # rebuild a degenerate 1:1 mixture using global best SC indices.
    best_weights = {}
    cached = adata_st.uns.get(UNS_FINAL_WEIGHTS, None)
    if isinstance(cached, dict) and len(cached) > 0:
        best_weights = dict(cached) # shallow copy (OK; we overwrite entries by spot)
    else:
        if "global_sc_best_cell_idx" in adata_st.obs:
            best_sc_idx = np.asarray(adata_st.obs["global_sc_best_cell_idx"], dtype=int)
            for j in range(n_st):
                sc_i = int(best_sc_idx[j])
                best_weights[int(j)] = dict(
                    best_type=best_type[j],
                    model="global_1to1",
                    sc_idx=np.asarray([sc_i], dtype=int),
                    w=np.asarray([1.0], dtype=float),
                    score=float(best_score[j]),
                )

    per_pair_weights = []

    for ref in refinements:
        if ref is None:
            continue
        r_type = np.asarray(ref.get("refined_type"), dtype=object)
        r_score = np.asarray(ref.get("refined_score"), dtype=float)
        wdict = ref.get("weights_dict", {})
        per_pair_weights.append(wdict)

 # update only where this refinement applies
        mask = np.asarray(ref.get("mask_pair"), dtype=bool)
        if mask.shape[0] != n_st:
 # fallback: infer from non-None type
            mask = (r_type != None)

        cand_better = r_score > best_score
        if only_when_improved:
            upd = mask & cand_better
        else:
            upd = mask

        idx = np.where(upd)[0]
        for j in idx:
 # allow None -> skip
            if r_type[j] is None:
                continue
            best_score[j] = float(r_score[j])
            best_type[j] = r_type[j]
            if int(j) in wdict:
                best_weights[int(j)] = wdict[int(j)]

    if overwrite or (final_type_key not in adata_st.obs):
        adata_st.obs[final_type_key] = best_type
    if overwrite or (final_score_key not in adata_st.obs):
        adata_st.obs[final_score_key] = best_score

    adata_st.uns[UNS_FINAL_WEIGHTS] = best_weights
    adata_st.uns[UNS_FINAL_PAIR_WEIGHTS] = per_pair_weights

    return dict(
        final_type=best_type,
        final_score=best_score,
        weights=best_weights,
    )
