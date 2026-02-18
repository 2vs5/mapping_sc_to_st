import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.sparse import issparse

from . import keys as K

# =========================
# Keys (single source of truth)
# =========================
# Default cell-type key: prefer final global type (if present), otherwise fall back.
DEFAULT_CELL_TYPE_KEY = getattr(K, "OBS_FINAL_GLOBAL_TYPE", K.OBS_FINAL_TYPE)

OBS_GLOBAL_BEST_CELL_IDX = K.OBS_GLOBAL_BEST_CELL_IDX
OBS_GLOBAL_SIM = K.OBS_GLOBAL_SIM

# (Stage1 cell-type from 1:1 best cell)
OBS_GLOBAL_BEST_CELL_TYPE = getattr(K, "OBS_GLOBAL_BEST_CELL_TYPE", "global_sc_best_cell_type")

OBS_STAGE2_FINAL_SCORE = K.OBS_FINAL_SCORE
UNS_STAGE2_FINAL_WEIGHTS = K.UNS_FINAL_WEIGHTS

OBSM_IMPUTED_FINAL_GLOBAL_ANCHOR = K.OBSM_IMPUTED_FINAL_GLOBAL_ANCHOR
UNS_GLOBAL_GENES_UNION = K.UNS_GLOBAL_GENES_UNION
OBS_FGW_UPDATED_MASK = K.OBS_FGW_UPDATED_MASK


# =========================
# Helpers
# =========================
def _to_dense(X):
    return X.toarray() if issparse(X) else np.asarray(X)

def _safe_int(x):
    try:
        if x is None:
            return None
        if isinstance(x, float) and np.isnan(x):
            return None
        return int(x)
    except Exception:
        return None

def _gene_wise_corr(X_true, X_pred, genes, min_n=3):
    rows = []
    for gi, g in enumerate(genes):
        a = X_true[:, gi]
        b = X_pred[:, gi]
        m = np.isfinite(a) & np.isfinite(b)
        n = int(m.sum())
        if n < min_n or np.nanstd(a[m]) == 0 or np.nanstd(b[m]) == 0:
            r = np.nan
        else:
            r = pearsonr(a[m], b[m])[0]
        rows.append({"gene": g, "corr": r, "n_used": n})
    return pd.DataFrame(rows)

def get_clean_deg_genes_for_target(deg_results, bleeding_results, c_tgt):
    """
    CleanDEG(ct) = DEG(ct) - Bleeding(ct)
    deg_results[ct]: DataFrame(index=genes) or list-like
    bleeding_results[ct]: list/set-like
    """
    deg_obj = deg_results.get(c_tgt, None)
    if deg_obj is None:
        return []
    deg_genes = list(deg_obj.index) if hasattr(deg_obj, "index") else list(deg_obj)
    bleed = set(list(bleeding_results.get(c_tgt, [])))
    return [g for g in deg_genes if g not in bleed]

def _predict_stage1_bestcell(adata_sc, adata_st, idx_st, genes, best_cell_idx_key):
    best_idx_all = np.asarray(adata_st.obs[best_cell_idx_key])
    X1 = np.full((len(idx_st), len(genes)), np.nan, dtype=np.float32)

    valid_pos = []
    sc_rows = []
    for pos, j in enumerate(idx_st):
        bi = _safe_int(best_idx_all[int(j)])
        if bi is None or bi < 0 or bi >= adata_sc.n_obs:
            continue
        valid_pos.append(pos)
        sc_rows.append(bi)

    if not sc_rows:
        return X1

    X_sc = _to_dense(adata_sc[np.asarray(sc_rows, dtype=int), genes].X).astype(np.float32, copy=False)
    X1[np.asarray(valid_pos, dtype=int)] = X_sc
    return X1

def _predict_stage2_mixture_Aformat(adata_sc, adata_st, idx_st, genes, weight_key):
    """
    adata_st.uns[weight_key][spot_idx] = {"sc_idx":[...], "w":[...]}
    Returns Xmix with NaN where not available.
    """
    wdict = adata_st.uns.get(weight_key, {})
    Xmix = np.full((len(idx_st), len(genes)), np.nan, dtype=np.float32)

    for pos, j in enumerate(idx_st):
        info = wdict.get(int(j), None)
        if not isinstance(info, dict) or ("sc_idx" not in info) or ("w" not in info):
            continue

        sc_idx = np.asarray(info["sc_idx"], dtype=int)
        w = np.asarray(info["w"], dtype=float)
        if sc_idx.size == 0 or w.size == 0:
            continue

        m = int(min(sc_idx.size, w.size))
        sc_idx = sc_idx[:m]
        w = w[:m]

        ok = (sc_idx >= 0) & (sc_idx < adata_sc.n_obs)
        sc_idx = sc_idx[ok]
        w = w[ok]
        if sc_idx.size == 0:
            continue

        s = w.sum()
        if s > 0:
            w = w / s

        X_sc = _to_dense(adata_sc[sc_idx, genes].X).astype(np.float32, copy=False)
        Xmix[pos] = (w.astype(np.float32) @ X_sc)

    return Xmix


# =========================================================
# 1) Cell-type-wise (CleanDEG) correlation with policy chain
#    (기존: 하나의 cell_type_key로 고정해서 stage1/2/3 비교)
# =========================================================
def compute_ct_gene_corr_policy(
    adata_sc,
    adata_st,
    deg_results,
    bleeding_results,
    *,
    cell_type_key=DEFAULT_CELL_TYPE_KEY,
    best_cell_idx_key=OBS_GLOBAL_BEST_CELL_IDX,
    stage2_weight_key=UNS_STAGE2_FINAL_WEIGHTS,
    use_stage2_score_gate=False,
    stage2_score_key=OBS_STAGE2_FINAL_SCORE,
    stage1_score_key=OBS_GLOBAL_SIM,
    fgw_obsm_key=OBSM_IMPUTED_FINAL_GLOBAL_ANCHOR,
    fgw_panel_genes_key=UNS_GLOBAL_GENES_UNION,
    fgw_updated_mask_key=OBS_FGW_UPDATED_MASK,
    min_spots=5,
    min_genes=5,
    min_n_corr=3,
):
    # key checks
    for k in [cell_type_key, best_cell_idx_key, fgw_updated_mask_key]:
        if k not in adata_st.obs:
            raise KeyError(f"adata_st.obs missing: {k}")
    for k in [stage2_weight_key, fgw_panel_genes_key]:
        if k not in adata_st.uns:
            raise KeyError(f"adata_st.uns missing: {k}")
    if fgw_obsm_key not in adata_st.obsm:
        raise KeyError(f"adata_st.obsm missing: {fgw_obsm_key}")

    # FGW panel mapping
    X_fgw_all = np.asarray(adata_st.obsm[fgw_obsm_key])
    panel_genes = list(adata_st.uns[fgw_panel_genes_key])
    if X_fgw_all.shape[1] != len(panel_genes):
        raise ValueError(f"{fgw_obsm_key}.shape[1] != len({fgw_panel_genes_key})")

    panel_pos = {g: i for i, g in enumerate(panel_genes)}
    panel_set = set(panel_genes)

    sc_set = set(adata_sc.var_names)
    st_set = set(adata_st.var_names)

    # stage2 score gate
    if use_stage2_score_gate:
        if stage2_score_key not in adata_st.obs or stage1_score_key not in adata_st.obs:
            raise KeyError("score gate on but score keys missing in adata_st.obs")
        s1 = np.asarray(adata_st.obs[stage1_score_key], dtype=float)
        s2 = np.asarray(adata_st.obs[stage2_score_key], dtype=float)
        improved = np.isfinite(s1) & np.isfinite(s2) & (s2 > s1)
    else:
        improved = None

    upd_all = np.asarray(adata_st.obs[fgw_updated_mask_key], dtype=bool)
    ct_all = np.asarray(adata_st.obs[cell_type_key], dtype=object)

    out_rows = []

    for ct in pd.unique(ct_all):
        if ct is None or (isinstance(ct, float) and np.isnan(ct)):
            continue

        idx = np.where(ct_all == ct)[0]
        if idx.size < min_spots:
            continue

        # genes for this CT: CleanDEG ∩ (panel ∩ sc ∩ st)
        clean = get_clean_deg_genes_for_target(deg_results, bleeding_results, ct)
        genes = [g for g in clean if (g in panel_set) and (g in sc_set) and (g in st_set)]
        if len(genes) < min_genes:
            continue

        # observed
        X_st = _to_dense(adata_st[idx, genes].X).astype(np.float32, copy=False)

        # stage1
        X1 = _predict_stage1_bestcell(adata_sc, adata_st, idx, genes, best_cell_idx_key)

        # stage2 mix + policy overwrite
        X2_mix = _predict_stage2_mixture_Aformat(adata_sc, adata_st, idx, genes, stage2_weight_key)
        X2 = X1.copy()
        has_mix = np.isfinite(X2_mix).any(axis=1)
        if improved is not None:
            has_mix = has_mix & improved[idx]
        X2[has_mix] = X2_mix[has_mix]

        # final overwrite updated only (policy)
        Xf = X2.copy()
        upd_local = upd_all[idx]
        if upd_local.any():
            cols = np.array([panel_pos[g] for g in genes], dtype=int)
            Xfgw_sub = X_fgw_all[idx][:, cols].astype(np.float32, copy=False)
            Xf[upd_local] = Xfgw_sub[upd_local]

        # correlations
        df1 = _gene_wise_corr(X_st, X1, genes, min_n=min_n_corr).rename(
            columns={"corr": "corr_stage1", "n_used": "n_used_stage1"}
        )
        df2 = _gene_wise_corr(X_st, X2, genes, min_n=min_n_corr).rename(
            columns={"corr": "corr_stage2", "n_used": "n_used_stage2"}
        )
        dff = _gene_wise_corr(X_st, Xf, genes, min_n=min_n_corr).rename(
            columns={"corr": "corr_fgw", "n_used": "n_used_fgw"}
        )

        df = df1.merge(df2, on="gene", how="outer").merge(dff, on="gene", how="outer")
        df["cell_type"] = ct
        df["delta_2_1"] = df["corr_stage2"] - df["corr_stage1"]
        df["delta_3_2"] = df["corr_fgw"] - df["corr_stage2"]
        df["delta_3_1"] = df["corr_fgw"] - df["corr_stage1"]
        out_rows.append(df)

    if not out_rows:
        return pd.DataFrame(columns=[
            "cell_type", "gene",
            "corr_stage1", "n_used_stage1",
            "corr_stage2", "n_used_stage2",
            "corr_fgw", "n_used_fgw",
            "delta_2_1", "delta_3_2", "delta_3_1",
        ])

    return pd.concat(out_rows, ignore_index=True)


# =========================================================
# 1b) NEW: Stage별 cell type 기준을 따로 적용하는 버전
#    - stage1: 1:1에서 만든 cell type 기준
#    - stage2: stage2가 예측한 cell type 기준
#    - stage3: stage3 최종 cell type 기준
#
# 반환은 "long format":
#   [stage, cell_type, gene, corr, n_used]
# =========================================================
def compute_ct_gene_corr_policy_stage_specific(
    adata_sc,
    adata_st,
    deg_results,
    bleeding_results,
    *,
    stage1_cell_type_key=OBS_GLOBAL_BEST_CELL_TYPE,
    stage2_cell_type_key=K.OBS_FINAL_TYPE,
    stage3_cell_type_key=getattr(K, "OBS_FINAL_GLOBAL_TYPE", K.OBS_FINAL_TYPE),
    best_cell_idx_key=OBS_GLOBAL_BEST_CELL_IDX,
    stage2_weight_key=UNS_STAGE2_FINAL_WEIGHTS,
    use_stage2_score_gate=False,
    stage2_score_key=OBS_STAGE2_FINAL_SCORE,
    stage1_score_key=OBS_GLOBAL_SIM,
    fgw_obsm_key=OBSM_IMPUTED_FINAL_GLOBAL_ANCHOR,
    fgw_panel_genes_key=UNS_GLOBAL_GENES_UNION,
    fgw_updated_mask_key=OBS_FGW_UPDATED_MASK,
    min_spots=5,
    min_genes=5,
    min_n_corr=3,
):
    # key checks
    need_obs = [best_cell_idx_key, fgw_updated_mask_key,
                stage1_cell_type_key, stage2_cell_type_key, stage3_cell_type_key]
    for k in need_obs:
        if k not in adata_st.obs:
            raise KeyError(f"adata_st.obs missing: {k}")
    for k in [stage2_weight_key, fgw_panel_genes_key]:
        if k not in adata_st.uns:
            raise KeyError(f"adata_st.uns missing: {k}")
    if fgw_obsm_key not in adata_st.obsm:
        raise KeyError(f"adata_st.obsm missing: {fgw_obsm_key}")

    # FGW panel mapping
    X_fgw_all = np.asarray(adata_st.obsm[fgw_obsm_key])
    panel_genes = list(adata_st.uns[fgw_panel_genes_key])
    if X_fgw_all.shape[1] != len(panel_genes):
        raise ValueError(f"{fgw_obsm_key}.shape[1] != len({fgw_panel_genes_key})")

    panel_pos = {g: i for i, g in enumerate(panel_genes)}
    panel_set = set(panel_genes)

    sc_set = set(adata_sc.var_names)
    st_set = set(adata_st.var_names)

    # stage2 score gate
    if use_stage2_score_gate:
        if stage2_score_key not in adata_st.obs or stage1_score_key not in adata_st.obs:
            raise KeyError("score gate on but score keys missing in adata_st.obs")
        s1 = np.asarray(adata_st.obs[stage1_score_key], dtype=float)
        s2 = np.asarray(adata_st.obs[stage2_score_key], dtype=float)
        improved = np.isfinite(s1) & np.isfinite(s2) & (s2 > s1)
    else:
        improved = None

    upd_all = np.asarray(adata_st.obs[fgw_updated_mask_key], dtype=bool)

    ct_stage = {
        "stage1_1to1": np.asarray(adata_st.obs[stage1_cell_type_key], dtype=object),
        "stage2_pairwise": np.asarray(adata_st.obs[stage2_cell_type_key], dtype=object),
        "stage3_global": np.asarray(adata_st.obs[stage3_cell_type_key], dtype=object),
    }

    out = []

    for stage_name, ct_all in ct_stage.items():
        for ct in pd.unique(ct_all):
            if ct is None or (isinstance(ct, float) and np.isnan(ct)):
                continue

            idx = np.where(ct_all == ct)[0]
            if idx.size < min_spots:
                continue

            # genes for this CT: CleanDEG ∩ (panel ∩ sc ∩ st)
            clean = get_clean_deg_genes_for_target(deg_results, bleeding_results, ct)
            genes = [g for g in clean if (g in panel_set) and (g in sc_set) and (g in st_set)]
            if len(genes) < min_genes:
                continue

            # observed
            X_st = _to_dense(adata_st[idx, genes].X).astype(np.float32, copy=False)

            # stage1 prediction (1:1 copy)
            X1 = _predict_stage1_bestcell(adata_sc, adata_st, idx, genes, best_cell_idx_key)

            if stage_name == "stage1_1to1":
                Xpred = X1
            else:
                # stage2 prediction
                X2_mix = _predict_stage2_mixture_Aformat(adata_sc, adata_st, idx, genes, stage2_weight_key)
                X2 = X1.copy()
                has_mix = np.isfinite(X2_mix).any(axis=1)
                if improved is not None:
                    has_mix = has_mix & improved[idx]
                X2[has_mix] = X2_mix[has_mix]

                if stage_name == "stage2_pairwise":
                    Xpred = X2
                else:
                    # stage3 prediction = stage2 + overwrite updated spots with FGW imputation
                    Xf = X2.copy()
                    upd_local = upd_all[idx]
                    if upd_local.any():
                        cols = np.array([panel_pos[g] for g in genes], dtype=int)
                        Xfgw_sub = X_fgw_all[idx][:, cols].astype(np.float32, copy=False)
                        Xf[upd_local] = Xfgw_sub[upd_local]
                    Xpred = Xf

            df = _gene_wise_corr(X_st, Xpred, genes, min_n=min_n_corr)
            df["stage"] = stage_name
            df["cell_type"] = ct
            out.append(df)

    if not out:
        return pd.DataFrame(columns=["stage", "cell_type", "gene", "corr", "n_used"])
    return pd.concat(out, ignore_index=True)


# =========================================================
# 2) All-spots (Panel-genes, NO DEG) correlation with policy chain
# =========================================================
def compute_all_spots_gene_corr_policy_panel(
    adata_sc,
    adata_st,
    *,
    # NEW
    bleeding_results=None,                 # dict: {cell_type: iterable of bleeding genes}
    cell_type_key=DEFAULT_CELL_TYPE_KEY,   # spot-level cell-type key in adata_st.obs
    exclude_bleeding_ct_in_all=True,       # if True, exclude bleeding CTs per gene in all-spots eval
    best_cell_idx_key=OBS_GLOBAL_BEST_CELL_IDX,
    stage2_weight_key=UNS_STAGE2_FINAL_WEIGHTS,
    use_stage2_score_gate=False,
    stage2_score_key=OBS_STAGE2_FINAL_SCORE,
    stage1_score_key=OBS_GLOBAL_SIM,
    fgw_obsm_key=OBSM_IMPUTED_FINAL_GLOBAL_ANCHOR,
    fgw_panel_genes_key=UNS_GLOBAL_GENES_UNION,
    fgw_updated_mask_key=OBS_FGW_UPDATED_MASK,
    min_n_corr=3,
):
    # -----------------------------
    # Validate required keys
    # -----------------------------
    for k in [best_cell_idx_key, fgw_updated_mask_key]:
        if k not in adata_st.obs:
            raise KeyError(f"adata_st.obs missing: {k}")
    for k in [stage2_weight_key, fgw_panel_genes_key]:
        if k not in adata_st.uns:
            raise KeyError(f"adata_st.uns missing: {k}")
    if fgw_obsm_key not in adata_st.obsm:
        raise KeyError(f"adata_st.obsm missing: {fgw_obsm_key}")

    if exclude_bleeding_ct_in_all and cell_type_key not in adata_st.obs:
        raise KeyError(f"adata_st.obs missing: {cell_type_key}")

    # -----------------------------
    # Evaluation genes: panel ∩ sc ∩ st
    # -----------------------------
    panel_genes = list(adata_st.uns[fgw_panel_genes_key])
    sc_set = set(adata_sc.var_names)
    st_set = set(adata_st.var_names)
    genes = [g for g in panel_genes if (g in sc_set) and (g in st_set)]
    if len(genes) == 0:
        return pd.DataFrame(columns=[
            "cell_type", "gene",
            "corr_stage1", "n_used_stage1",
            "corr_stage2", "n_used_stage2",
            "corr_fgw", "n_used_fgw",
            "delta_2_1", "delta_3_2", "delta_3_1"
        ])

    # -----------------------------
    # FGW panel column mapping
    # -----------------------------
    X_fgw_all = np.asarray(adata_st.obsm[fgw_obsm_key])
    if X_fgw_all.shape[1] != len(panel_genes):
        raise ValueError(f"{fgw_obsm_key}.shape[1] != len({fgw_panel_genes_key})")
    panel_pos = {g: i for i, g in enumerate(panel_genes)}
    cols = np.array([panel_pos[g] for g in genes], dtype=int)

    # -----------------------------
    # Optional stage2 score gating:
    # apply stage2 mixture only where stage2 score > stage1 score
    # -----------------------------
    if use_stage2_score_gate:
        if stage2_score_key not in adata_st.obs or stage1_score_key not in adata_st.obs:
            raise KeyError("score gate on but score keys missing in adata_st.obs")
        s1 = np.asarray(adata_st.obs[stage1_score_key], dtype=float)
        s2 = np.asarray(adata_st.obs[stage2_score_key], dtype=float)
        improved = np.isfinite(s1) & np.isfinite(s2) & (s2 > s1)
    else:
        improved = None

    idx_all = np.arange(adata_st.n_obs)

    # -----------------------------
    # Observed ST expression
    # -----------------------------
    X_st = _to_dense(adata_st[:, genes].X).astype(np.float32, copy=False)

    # -----------------------------
    # Stage1 prediction: best-cell copy (1:1)
    # -----------------------------
    X1 = _predict_stage1_bestcell(adata_sc, adata_st, idx_all, genes, best_cell_idx_key)

    # -----------------------------
    # Stage2 prediction: mixture overwrite on top of stage1
    # -----------------------------
    X2_mix = _predict_stage2_mixture_Aformat(adata_sc, adata_st, idx_all, genes, stage2_weight_key)
    X2 = X1.copy()
    has_mix = np.isfinite(X2_mix).any(axis=1)
    if improved is not None:
        has_mix = has_mix & improved
    X2[has_mix] = X2_mix[has_mix]

    # -----------------------------
    # Stage3 (FGW) final prediction:
    # overwrite stage2 only on updated spots
    # -----------------------------
    Xf = X2.copy()
    upd = np.asarray(adata_st.obs[fgw_updated_mask_key], dtype=bool)
    if upd.any():
        Xfgw_sub = X_fgw_all[:, cols].astype(np.float32, copy=False)
        Xf[upd] = Xfgw_sub[upd]

    # -----------------------------
    # NEW: Build gene -> bleeding cell-types mapping
    # -----------------------------
    if exclude_bleeding_ct_in_all:
        if bleeding_results is None:
            raise ValueError("bleeding_results must be provided when exclude_bleeding_ct_in_all=True.")

        spot_ct = np.asarray(adata_st.obs[cell_type_key], dtype=object)

        # Reverse map: gene -> set(cell_types where this gene is considered bleeding)
        gene_to_bleeding_cts = {}
        for ct, glist in bleeding_results.items():
            for g in list(glist):
                gene_to_bleeding_cts.setdefault(g, set()).add(ct)
    else:
        gene_to_bleeding_cts = {}

    # -----------------------------
    # Gene-wise correlation (gene-specific mask)
    # -----------------------------
    rows = []
    for gi, g in enumerate(genes):
        a1 = X1[:, gi]
        a2 = X2[:, gi]
        af = Xf[:, gi]
        y = X_st[:, gi]

        # For each gene, exclude spots whose cell-type is in that gene's bleeding CT set
        if exclude_bleeding_ct_in_all:
            excluded_cts = gene_to_bleeding_cts.get(g, set())
            m_ct = ~pd.Series(spot_ct).isin(excluded_cts).to_numpy()
        else:
            excluded_cts = set()
            m_ct = np.ones(adata_st.n_obs, dtype=bool)

        # Stage1 correlation
        m1 = m_ct & np.isfinite(y) & np.isfinite(a1)
        n1 = int(m1.sum())
        if n1 < min_n_corr or np.nanstd(y[m1]) == 0 or np.nanstd(a1[m1]) == 0:
            r1 = np.nan
        else:
            r1 = pearsonr(y[m1], a1[m1])[0]

        # Stage2 correlation
        m2 = m_ct & np.isfinite(y) & np.isfinite(a2)
        n2 = int(m2.sum())
        if n2 < min_n_corr or np.nanstd(y[m2]) == 0 or np.nanstd(a2[m2]) == 0:
            r2 = np.nan
        else:
            r2 = pearsonr(y[m2], a2[m2])[0]

        # Stage3(FGW) correlation
        mf = m_ct & np.isfinite(y) & np.isfinite(af)
        nf = int(mf.sum())
        if nf < min_n_corr or np.nanstd(y[mf]) == 0 or np.nanstd(af[mf]) == 0:
            rf = np.nan
        else:
            rf = pearsonr(y[mf], af[mf])[0]

        rows.append({
            "cell_type": "all_spots",
            "gene": g,
            "corr_stage1": r1,
            "n_used_stage1": n1,
            "corr_stage2": r2,
            "n_used_stage2": n2,
            "corr_fgw": rf,
            "n_used_fgw": nf,
            "delta_2_1": (r2 - r1) if np.isfinite(r2) and np.isfinite(r1) else np.nan,
            "delta_3_2": (rf - r2) if np.isfinite(rf) and np.isfinite(r2) else np.nan,
            "delta_3_1": (rf - r1) if np.isfinite(rf) and np.isfinite(r1) else np.nan,
            # Debug/tracing field (can be removed if not needed)
            "n_excluded_ct": len(excluded_cts),
        })

    return pd.DataFrame(rows)



# =========================================================
# 3) Convenience wrapper: ct + all_spots and concat (기존 고정 CT)
# =========================================================
def compute_ct_and_all_gene_corr(
    adata_sc,
    adata_st,
    deg_results,
    bleeding_results,
    *,
    min_spots=5,
    min_genes=5,
    min_n_corr=3,
    use_stage2_score_gate=False,
):
    df_ct = compute_ct_gene_corr_policy(
        adata_sc, adata_st,
        deg_results, bleeding_results,
        min_spots=min_spots,
        min_genes=min_genes,
        min_n_corr=min_n_corr,
        use_stage2_score_gate=use_stage2_score_gate,
    )
    df_all = compute_all_spots_gene_corr_policy_panel(
        adata_sc, adata_st,
        bleeding_results=bleeding_results,   
        cell_type_key=DEFAULT_CELL_TYPE_KEY,
        exclude_bleeding_ct_in_all=True,     
        min_n_corr=min_n_corr,
        use_stage2_score_gate=use_stage2_score_gate,
    )
    return pd.concat([df_ct, df_all], ignore_index=True)


# =========================================================
# 3b) NEW wrapper: stage-specific CT(long) + all_spots(wide) concat
#     (둘 포맷이 달라서 보통은 따로 쓰는 걸 추천)
# =========================================================
def compute_ct_stage_specific_and_all_gene_corr(
    adata_sc,
    adata_st,
    deg_results,
    bleeding_results,
    *,
    min_spots=5,
    min_genes=5,
    min_n_corr=3,
    use_stage2_score_gate=False,
    stage1_cell_type_key=OBS_GLOBAL_BEST_CELL_TYPE,
    stage2_cell_type_key=K.OBS_FINAL_TYPE,
    stage3_cell_type_key=getattr(K, "OBS_FINAL_GLOBAL_TYPE", K.OBS_FINAL_TYPE),
):
    df_ct_stage = compute_ct_gene_corr_policy_stage_specific(
        adata_sc, adata_st,
        deg_results, bleeding_results,
        min_spots=min_spots,
        min_genes=min_genes,
        min_n_corr=min_n_corr,
        use_stage2_score_gate=use_stage2_score_gate,
        stage1_cell_type_key=stage1_cell_type_key,
        stage2_cell_type_key=stage2_cell_type_key,
        stage3_cell_type_key=stage3_cell_type_key,
    )
    df_all = compute_all_spots_gene_corr_policy_panel(
        adata_sc, adata_st,
        bleeding_results=bleeding_results,   
        cell_type_key=DEFAULT_CELL_TYPE_KEY,
        exclude_bleeding_ct_in_all=True,     
        min_n_corr=min_n_corr,
        use_stage2_score_gate=use_stage2_score_gate,
    )
    return df_ct_stage, df_all


def extract_expected_vs_res_per_spot_gene(
    adata_sc,
    adata_st,
    *,
    deg_by_cell_type=None,                 # dict: {cell_type: iterable of DEG genes}
    bleeding_results=None,                 # dict: {cell_type: iterable of bleeding genes}
    cell_type_key=DEFAULT_CELL_TYPE_KEY,
    exclude_bleeding_ct_in_all=True,
    best_cell_idx_key=OBS_GLOBAL_BEST_CELL_IDX,
    stage2_weight_key=UNS_STAGE2_FINAL_WEIGHTS,
    use_stage2_score_gate=False,
    stage2_score_key=OBS_STAGE2_FINAL_SCORE,
    stage1_score_key=OBS_GLOBAL_SIM,
    fgw_obsm_key=OBSM_IMPUTED_FINAL_GLOBAL_ANCHOR,
    fgw_panel_genes_key=UNS_GLOBAL_GENES_UNION,
    fgw_updated_mask_key=OBS_FGW_UPDATED_MASK,
    return_filtered_only=False,            # True면 bleeding 제외된 행만 반환
):
    # -----------------------------
    # Validate keys
    # -----------------------------
    for k in [best_cell_idx_key, fgw_updated_mask_key, cell_type_key]:
        if k not in adata_st.obs:
            raise KeyError(f"adata_st.obs missing: {k}")
    for k in [stage2_weight_key, fgw_panel_genes_key]:
        if k not in adata_st.uns:
            raise KeyError(f"adata_st.uns missing: {k}")
    if fgw_obsm_key not in adata_st.obsm:
        raise KeyError(f"adata_st.obsm missing: {fgw_obsm_key}")

    # -----------------------------
    # genes = panel ∩ sc ∩ st
    # -----------------------------
    panel_genes = list(adata_st.uns[fgw_panel_genes_key])
    sc_set = set(adata_sc.var_names)
    st_set = set(adata_st.var_names)
    genes = [g for g in panel_genes if (g in sc_set) and (g in st_set)]
    if len(genes) == 0:
        return pd.DataFrame(columns=[
            "spot_idx", "spot_id", "cell_type", "gene",
            "is_deg_for_celltype",
            "is_bleeding_excluded",
            "obs_expr", "pred_stage1_expr", "pred_stage2_expr", "pred_fgw_expr"
        ])

    # FGW column mapping
    X_fgw_all = np.asarray(adata_st.obsm[fgw_obsm_key])
    if X_fgw_all.shape[1] != len(panel_genes):
        raise ValueError(f"{fgw_obsm_key}.shape[1] != len({fgw_panel_genes_key})")
    panel_pos = {g: i for i, g in enumerate(panel_genes)}
    cols = np.array([panel_pos[g] for g in genes], dtype=int)

    # optional stage2 gate
    if use_stage2_score_gate:
        if stage2_score_key not in adata_st.obs or stage1_score_key not in adata_st.obs:
            raise KeyError("score gate on but score keys missing in adata_st.obs")
        s1 = np.asarray(adata_st.obs[stage1_score_key], dtype=float)
        s2 = np.asarray(adata_st.obs[stage2_score_key], dtype=float)
        improved = np.isfinite(s1) & np.isfinite(s2) & (s2 > s1)
    else:
        improved = None

    idx_all = np.arange(adata_st.n_obs)

    # observed / predicted matrices
    X_st = _to_dense(adata_st[:, genes].X).astype(np.float32, copy=False)                 # expected
    X1 = _predict_stage1_bestcell(adata_sc, adata_st, idx_all, genes, best_cell_idx_key) # res1
    X2_mix = _predict_stage2_mixture_Aformat(adata_sc, adata_st, idx_all, genes, stage2_weight_key)
    X2 = X1.copy()
    has_mix = np.isfinite(X2_mix).any(axis=1)
    if improved is not None:
        has_mix = has_mix & improved
    X2[has_mix] = X2_mix[has_mix]

    Xf = X2.copy()
    upd = np.asarray(adata_st.obs[fgw_updated_mask_key], dtype=bool)
    if upd.any():
        Xfgw_sub = X_fgw_all[:, cols].astype(np.float32, copy=False)
        Xf[upd] = Xfgw_sub[upd]

    # gene -> bleeding cell types
    if exclude_bleeding_ct_in_all:
        if bleeding_results is None:
            raise ValueError("bleeding_results must be provided when exclude_bleeding_ct_in_all=True.")
        gene_to_bleeding_cts = {}
        for ct, glist in bleeding_results.items():
            for g in list(glist):
                gene_to_bleeding_cts.setdefault(g, set()).add(ct)
    else:
        gene_to_bleeding_cts = {}

    # cell type -> DEG set
    if deg_by_cell_type is None:
        deg_by_cell_type = {}
    deg_sets = {ct: set(gs) for ct, gs in deg_by_cell_type.items()}

    spot_ct = np.asarray(adata_st.obs[cell_type_key], dtype=object)
    spot_ids = np.asarray(adata_st.obs_names, dtype=object)

    rows = []
    n_spots = adata_st.n_obs
    n_genes = len(genes)

    for si in range(n_spots):
        ct = spot_ct[si]
        deg_set_ct = deg_sets.get(ct, set())

        for gi in range(n_genes):
            g = genes[gi]

            # bleeding exclusion flag (gene-specific)
            excluded_cts = gene_to_bleeding_cts.get(g, set())
            is_bleeding_excluded = (ct in excluded_cts)

            if return_filtered_only and is_bleeding_excluded:
                continue

            rows.append({
                "spot_idx": si,
                "spot_id": spot_ids[si],
                "cell_type": ct,
                "gene": g,
                "is_deg_for_celltype": (g in deg_set_ct),
                "is_bleeding_excluded": is_bleeding_excluded,
                "obs_expr": float(X_st[si, gi]) if np.isfinite(X_st[si, gi]) else np.nan,
                "pred_stage1_expr": float(X1[si, gi]) if np.isfinite(X1[si, gi]) else np.nan,
                "pred_stage2_expr": float(X2[si, gi]) if np.isfinite(X2[si, gi]) else np.nan,
                "pred_fgw_expr": float(Xf[si, gi]) if np.isfinite(Xf[si, gi]) else np.nan,
            })

    return pd.DataFrame(rows)