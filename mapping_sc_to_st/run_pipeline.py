
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Iterable, Optional

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import tqdm
from scipy.stats import spearmanr



from .global_map import run_global_1to1_mapping
from .genes import select_genes_per_celltype
from .pairwise_refine import select_type_pairs_from_A, refine_pair_ct1_ct2, refine_single_ct, build_st_pairs_from_knn, finalize_pairs
from .bleeding import compute_directional_bleeding, compute_incoming_union
from .merge import merge_refinements
from .anchors import select_st_anchors_final
from .alignment import build_phi_before_after
from .fgw_solver import POTFGWSolver
from .array_ops import to_dense as _to_dense
from .gene_correlation import compute_ct_stage_specific_and_all_gene_corr, _predict_stage1_bestcell, _predict_stage2_mixture_Aformat
from .final_global_anchor_fgw import GlobalAnchorFGWConfig, run_final_global_anchor_fgw_update
from . import keys as K

OBS_GLOBAL_TYPE = K.OBS_GLOBAL_TYPE

DEFAULT_CELL_TYPE_KEY = getattr(K, "OBS_FINAL_GLOBAL_TYPE", K.OBS_FINAL_TYPE)
OBS_FINAL_GLOBAL_TYPE = K.OBS_FINAL_GLOBAL_TYPE

OBS_GLOBAL_BEST_CELL_IDX = K.OBS_GLOBAL_BEST_CELL_IDX
OBS_GLOBAL_SIM = K.OBS_GLOBAL_SIM


# (Stage1 cell-type from 1:1 best cell)
OBS_GLOBAL_BEST_CELL_TYPE = getattr(K, "OBS_GLOBAL_BEST_CELL_TYPE", "global_sc_best_cell_type")

OBS_STAGE2_FINAL_SCORE = K.OBS_FINAL_SCORE
UNS_STAGE2_FINAL_WEIGHTS = K.UNS_FINAL_WEIGHTS
UNS_FINAL_WEIGHTS = K.UNS_FINAL_WEIGHTS
UNS_GLOBAL_FINAL_WEIGHTS = K.UNS_GLOBAL_FINAL_WEIGHTS

OBSM_IMPUTED_FINAL_GLOBAL_ANCHOR = K.OBSM_IMPUTED_FINAL_GLOBAL_ANCHOR
UNS_GLOBAL_GENES_UNION = K.UNS_GLOBAL_GENES_UNION
OBS_FGW_UPDATED_MASK = K.OBS_FGW_UPDATED_MASK



# ------------------------------------------------------------
# Internal helpers (CV + evaluation)
# ------------------------------------------------------------

def to_dense(X, *, dtype=None):
    """Backwards-compatible wrapper (prefer util.array_ops.to_dense)."""
    return _to_dense(X, dtype=dtype)

def _genes_union_from_by_ct(selected_genes_by_ct: Dict[str, pd.DataFrame]) -> List[str]:
    return sorted({str(g) for df in selected_genes_by_ct.values() for g in df.index.tolist()})

def _genes_union_from_by_ct_v2(selected_genes_by_ct: Dict[str, List[str]]) -> List[str]:
    return sorted({str(g) for g_list in selected_genes_by_ct.values() for g in g_list})

def _weights_dict_to_csr(weights_dict: Dict[int, Dict[str, Any]], *, n_sc: int, n_st: int):
    """Convert {st_idx -> {sc_idx,w}} to CSR (n_st x n_sc)."""
    from scipy.sparse import csr_matrix

    indptr = np.zeros(n_st + 1, dtype=int)
    indices: List[int] = []
    data: List[float] = []

    for j in range(n_st):
        entry = weights_dict.get(int(j), None)
        if not isinstance(entry, dict):
            indptr[j + 1] = indptr[j]
            continue
        sc_idx = np.asarray(entry.get("sc_idx", []), dtype=int).ravel()
        w = np.asarray(entry.get("w", []), dtype=float).ravel()
        if sc_idx.size == 0 or w.size == 0:
            indptr[j + 1] = indptr[j]
            continue
        m = int(min(sc_idx.size, w.size))
        sc_idx = sc_idx[:m]
        w = w[:m]
        ok = (sc_idx >= 0) & (sc_idx < n_sc)
        sc_idx = sc_idx[ok]
        w = w[ok]
        if sc_idx.size == 0:
            indptr[j + 1] = indptr[j]
            continue
        s = float(np.sum(w))
        if s > 0:
            w = w / s
        indices.extend(sc_idx.tolist())
        data.extend(w.tolist())
        indptr[j + 1] = indptr[j] + sc_idx.size

    return csr_matrix((np.asarray(data, float), np.asarray(indices, int), indptr), shape=(n_st, n_sc))


def _merge_stage2_stage3_weights(
    stage2: Optional[Dict[int, Dict[str, Any]]],
    stage3_updated: Optional[Dict[int, Dict[str, Any]]],
) -> Dict[int, Dict[str, Any]]:
    """Policy weights: start from stage2 (all spots), overwrite with stage3 updated spots."""
    w = dict(stage2 or {})
    if isinstance(stage3_updated, dict) and len(stage3_updated) > 0:
        for k, v in stage3_updated.items():
            w[int(k)] = v
    return w


def _gene_wise_pearson_corr(X_true: np.ndarray, X_pred: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Vectorized Pearson per gene (column-wise) over spots."""
    A = np.asarray(X_true, dtype=float)
    B = np.asarray(X_pred, dtype=float)
    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: true={A.shape}, pred={B.shape}")

    A0 = A - np.nanmean(A, axis=0, keepdims=True)
    B0 = B - np.nanmean(B, axis=0, keepdims=True)

    num = np.nansum(A0 * B0, axis=0)
    den = np.sqrt(np.nansum(A0 * A0, axis=0) * np.nansum(B0 * B0, axis=0))
    den = np.where(np.isfinite(den) & (den > eps), den, np.nan)
    r = num / den
    r[~np.isfinite(r)] = np.nan
    return r


def _normalize_selected_genes_by_ct(selected_genes_by_ct: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Ensure dict[ct -> DataFrame(index=genes as str)]."""
    norm: Dict[str, pd.DataFrame] = {}
    for ct, df in (selected_genes_by_ct or {}).items():
        if df is None:
            norm[str(ct)] = pd.DataFrame(index=[])
        elif isinstance(df, pd.DataFrame):
            norm[str(ct)] = pd.DataFrame(index=list(map(str, df.index.tolist())))
        else:
            norm[str(ct)] = pd.DataFrame(index=list(map(str, list(df))))
    return norm


def split_selected_genes_by_ct(
    selected_genes_by_ct: Dict[str, pd.DataFrame],
    *,
    n_splits: int = 5,
    test_frac: float = 0.2,
    seed: int = 0,
    min_train_per_ct: int = 20,
    mode: str = "kfold",
) -> List[Dict[str, Any]]:
    """Split selected genes for CV.

    Default behavior is **true K-fold** over the *unique* selected gene union.
    That means each unique gene appears in the test set exactly once across folds
    (disjoint test sets). This addresses the issue where per-celltype sampling +
    union can create very large test sets.

    Parameters
    ----------
    selected_genes_by_ct
        dict[ct -> DataFrame(index=genes)]. Genes may overlap across cell types.
    n_splits
        Number of folds (K).
    test_frac
        Only used when mode="mc" (Monte-Carlo / repeated holdout). For mode="kfold",
        the implied test fraction is 1/n_splits.
    min_train_per_ct
        For mode="mc" we enforce it. For mode="kfold" we do **not** break disjointness;
        instead we emit warnings if some cell types end up with too few train genes.
    mode
        "kfold" (default) or "mc".

    Returns
    -------
    splits
        List of dicts with keys:
          - split_id
          - train_by_ct (dict[ct -> DF(index=train genes)])
          - test_genes (list[str]; disjoint across folds for mode="kfold")
          - test_by_ct (dict[ct -> DF(index=test genes)])
          - warnings (list[str])
    """

    mode = str(mode).lower().strip()
    if int(n_splits) < 2:
        raise ValueError("n_splits must be >= 2")

    # Normalize genes as strings
    cts = list(selected_genes_by_ct.keys())
    
    try:
        ct_genes: Dict[str, List[str]] = {
            str(ct): list(map(str, selected_genes_by_ct[ct].index.tolist())) for ct in cts
        }
    except:
        ct_genes = selected_genes_by_ct
    
    genes_union = sorted({g for gl in ct_genes.values() for g in gl})
    if len(genes_union) < int(n_splits):
        raise ValueError(
            f"Not enough unique genes ({len(genes_union)}) to split into {n_splits} folds."
        )

    rng = np.random.default_rng(int(seed))
    splits: List[Dict[str, Any]] = []

    if mode == "kfold":
        # Disjoint fold assignment over the *unique gene union*
        perm = rng.permutation(len(genes_union))
        buckets = np.array_split(np.asarray(genes_union, dtype=object)[perm], int(n_splits))

        for k, bucket in enumerate(buckets):
            test_genes = list(map(str, bucket.tolist()))
            test_set = set(test_genes)

            train_by_ct: Dict[str, pd.DataFrame] = {}
            test_by_ct: Dict[str, pd.DataFrame] = {}
            warnings: List[str] = []

            for ct, gl in ct_genes.items():
                gl_set = set(gl)
                g_test = sorted(gl_set & test_set)
                g_train = sorted(gl_set - test_set)

                test_by_ct[ct] = pd.DataFrame(index=g_test)
                train_by_ct[ct] = pd.DataFrame(index=g_train)

                if len(g_train) < int(min_train_per_ct):
                    warnings.append(
                        f"ct={ct!r} has only {len(g_train)} train genes (<{min_train_per_ct}) in fold {k}."
                    )

            splits.append(
                {
                    "split_id": int(k),
                    "train_by_ct": train_by_ct,
                    "test_by_ct": test_by_ct,
                    "test_genes": sorted(test_set),
                    "warnings": warnings,
                }
            )

        return splits

    if mode != "mc":
        raise ValueError("mode must be 'kfold' or 'mc'")

    # Legacy: per-celltype Monte-Carlo / repeated holdout (can yield large test unions)
    if not (0.0 < float(test_frac) < 1.0):
        raise ValueError("test_frac must be in (0,1).")

    rng_master = np.random.default_rng(int(seed))
    for s in range(int(n_splits)):
        rng_s = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
        train_by_ct: Dict[str, pd.DataFrame] = {}
        test_by_ct: Dict[str, pd.DataFrame] = {}
        test_union: set[str] = set()
        warnings: List[str] = []

        for ct in cts:
            genes = ct_genes[str(ct)]
            if len(genes) == 0:
                train_by_ct[str(ct)] = pd.DataFrame(index=[])
                test_by_ct[str(ct)] = pd.DataFrame(index=[])
                continue

            perm = rng_s.permutation(len(genes))
            n_test = int(np.floor(float(test_frac) * len(genes)))
            n_test = min(n_test, max(0, len(genes) - int(min_train_per_ct)))
            test_idx = perm[:n_test]
            train_idx = perm[n_test:]
            g_test = sorted({genes[i] for i in test_idx})
            g_train = sorted({genes[i] for i in train_idx})
            train_by_ct[str(ct)] = pd.DataFrame(index=g_train)
            test_by_ct[str(ct)] = pd.DataFrame(index=g_test)
            test_union.update(g_test)

        splits.append(
            {
                "split_id": int(s),
                "train_by_ct": train_by_ct,
                "test_by_ct": test_by_ct,
                "test_genes": sorted(test_union),
                "warnings": warnings,
            }
        )

    return splits


# ------------------------------------------------------------
# A_df (metanode graph adjacency) helper
# ------------------------------------------------------------

def build_A_df_from_metanode_graph(
    adata_sc,
    *,
    metacell_graph_path: str,
    cell_type_color: "pd.DataFrame",
    adata_st=None,
    ref_ct_key: str = OBS_GLOBAL_TYPE,
    graph_key: str = "mf_graph",
    ct_key: str = "cell_type",
    obsm_key: str = "X_umap",
    obsm_dims: Tuple[int, int] = (0, 1),
    ensure_umap: bool = True,
    umap_use_rep: str = "X_scvi",
    store_uns_key: str = "A_df",
    verbose: bool = True,
    **plot_kwargs,
) -> "pd.DataFrame":
    """
    Build the metanode graph adjacency table (A_df) used to select type-pairs for pairwise refinement.

    Notes
    -----
    - If `ref_ct_key` refers to a label stored on ST (e.g. global 1:1 output),
      make sure you've run the global 1:1 mapping first so `adata_st.obs[ref_ct_key]` exists.
    - This wraps `util_v2.fig.plot_celltype_metanode_graph` but closes the figure immediately
      (no plotting side effects in notebooks unless you explicitly display it).
    """
    if ensure_umap and (obsm_key is not None) and (obsm_key not in getattr(adata_sc, "obsm", {})):
        try:
            import scanpy as sc  # type: ignore
        except Exception as e:
            raise ImportError(
                "scanpy is required to compute UMAP automatically. "
                "Either install scanpy or set ensure_umap=False and provide obsm_key."
            ) from e

        if umap_use_rep not in adata_sc.obsm:
            raise KeyError(
                f"obsm['{obsm_key}'] is missing and umap_use_rep='{umap_use_rep}' not found in adata_sc.obsm. "
                "Provide a valid representation (e.g. 'X_scvi') or compute UMAP upstream."
            )
        if verbose:
            print(f"[build_A_df] '{obsm_key}' missing -> computing neighbors/umap with use_rep='{umap_use_rep}'")
        sc.pp.neighbors(adata_sc, use_rep=umap_use_rep)
        sc.tl.umap(adata_sc)

    from .fig import plot_celltype_metanode_graph

    fig, ax, A_df, _mgraph_df = plot_celltype_metanode_graph(
        adata=adata_sc,
        cell_type_color=cell_type_color,
        excel_path=metacell_graph_path,
        adata_type_ref=adata_st,
        ref_ct_key=str(ref_ct_key),
        use_ref_intersection=True,
        graph_key=graph_key,
        ct_key=ct_key,
        obsm_key=obsm_key,
        #obsm_dims=obsm_dims,
        verbose=verbose,
        #**plot_kwargs,
    )

    try:
        import matplotlib.pyplot as plt  # type: ignore
        plt.close(fig)
    except Exception:
        pass

    if store_uns_key:
        try:
            adata_sc.uns[store_uns_key] = A_df
        except Exception:
            pass

    return A_df


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

@dataclass
class PipelineConfig:
    """
    Minimal config mirroring the example notebook structure.

    This is intentionally "pipeline-ish" (not strictly CV).
    """
    # gene selection
    deg_method: str = "wilcoxon"
    deg_pval_cutoff: float = 0.1
    deg_logfc_cutoff: float = 1.5
    hvg_n_top_genes: int = 40
    hvg_flavor: str = "seurat_v3"
    mode: str = "union"  # union | intersection

    # global map
    groupby_sc: str = "cell_type"
    n_jobs: int = 128
    cos_threshold: float = 0.5
    corr_method: str = "pearson"

    # pairwise refine
    pair_k = 10
    pair_threshold = 200
    top_n_pairs: int = 1000
    st_sim_min: float = 0.3
    min_cells_sc: int = 2
    min_cells_st: int = 20
    tau_sc_pos: float = 1.0
    tau_st_flat: float = 0.2

    # solver (pairwise + final)
    solver_method_stage2 = "entropic_fgw"
    solver_method_stage3 = "entropic_fgw"
    solver_alpha: float = 0.5
    solver_loss_fun: str = "square_loss"
    solver_epsilon_stage2: float = 0.1
    solver_epsilon_stage3: float = 0.15

    # anchors
    n_anchor_per_type: int = 30
    top_frac_per_type: float = 0.3
    min_per_type: int = 2
    diversity_bins: int = 12
    diversity_min_per_grid: int = 1
    st_geom_key: str = "X_scvi"
    st_type_key_stage2: str = "final_pairwise_type"
    st_score_key_stage2: str = "final_pairwise_score"

    # alignment + final update
    direction_alignment_build: str = "sc_to_st"
    align_min_per_type: int = 30
    alignment_shrink_k: int = 1

    # final global anchor fgw
    tau_anchor_percentile: float = 50.0
    tau_update: float = 0.9
    beta_expr_stage3: float = 0.3
    update_only_if_improved: bool = True
    use_incoming_genes: bool = True

    # imputation
    compute_imputation: bool = True
    store_imputation_dense: bool = True
    store_imputation_dict: bool = True


# ------------------------------------------------------------
# Pipeline runner
# ------------------------------------------------------------

def run_pipeline(
    adata_sc,
    adata_st,
    adata_mc,
    *,
    cfg: PipelineConfig = PipelineConfig(),
    # gene inputs
    selected_genes_by_ct: Optional[Dict[str, pd.DataFrame]] = None,
    # 1:1 controls
    precomp: Optional[Dict[str, Any]] = None,
    skip_global_1to1: bool = False,
    # A_df controls
    A_df: Optional[pd.DataFrame] = None,
    auto_build_A_df: bool = False,
    recompute_A_df: bool = False,
    metacell_graph_path: Optional[str] = None,
    cell_type_color: Optional[pd.DataFrame] = None,
    # misc
    return_intermediates: bool = True,
    verbose: bool = True,
    **A_df_kwargs,
) -> Dict[str, Any]:
    """
    Run the example notebook pipeline end-to-end (no plotting).

    Design goals
    ------------
    - `selected_genes_by_ct` is the canonical gene container (dict[ct -> DF(index=genes)]).
      This matches bleeding's expectations.
    - `skip_global_1to1=True` allows CV to precompute global 1:1 outside and pass `precomp`.
      In this mode we REQUIRE that adata_st already contains global 1:1 obs keys.

    A_df handling
    ------------
    - If `A_df` is provided, use it.
    - Else if `adata_sc.uns["A_df"]` exists and not `recompute_A_df`, reuse it.
    - Else if `auto_build_A_df` is True, build it inside (requires metacell_graph_path + cell_type_color).
    - Else raise.

    Returns
    -------
    out : dict containing key outputs + (optionally) intermediates.
    """
    out: Dict[str, Any] = {"cfg": asdict(cfg)}

    # 1) gene selection (or provided)
    if selected_genes_by_ct is None:
        selected_genes_by_ct, deg_results, hvg_results = select_genes_per_celltype(
            adata_sc,
            groupby=cfg.groupby_sc,
            deg_method=cfg.deg_method,
            deg_n_genes=None,
            deg_pval_cutoff=cfg.deg_pval_cutoff,
            deg_logfc_cutoff=cfg.deg_logfc_cutoff,
            hvg_n_top_genes=cfg.hvg_n_top_genes,
            hvg_flavor=cfg.hvg_flavor,
            hvg_layer=None,
            mode=cfg.mode,
        )
    else:
        # normalize keys and indices to str
        norm = {}
        for ct, df in selected_genes_by_ct.items():
            if df is None:
                norm[str(ct)] = pd.DataFrame(index=[])
            elif isinstance(df, pd.DataFrame):
                norm[str(ct)] = pd.DataFrame(index=list(map(str, df.index.tolist())))
            else:
                # allow list-like as a convenience
                norm[str(ct)] = pd.DataFrame(index=list(map(str, list(df))))
        selected_genes_by_ct = norm
        deg_results, hvg_results = None, None
        
        
    out["selected_genes_by_ct"] = {k: list(v.index.astype(str)) for k, v in selected_genes_by_ct.items()}
    out["deg_results"] = deg_results
    out["hvg_results"] = hvg_results

    # union panel for global map / precomp
    genes_union = sorted({str(g) for df in selected_genes_by_ct.values() for g in df.index.tolist()})
    out["genes_union_n"] = int(len(genes_union))
    
    adata_sc = adata_sc[:, genes_union].copy()
    adata_st = adata_st[:, genes_union].copy()

    # 2) global 1:1 mapping (or reuse from CV)
    if not skip_global_1to1:
        S = run_global_1to1_mapping(
            adata_sc,
            adata_st,
            selected_genes_by_ct,  # supports dict
            groupby_sc=cfg.groupby_sc,
            n_jobs=cfg.n_jobs,
            cos_threshold=cfg.cos_threshold,
            uncertain_label="uncertain",
            use_vectorized_batch=True,
            corr_method=cfg.corr_method,
        )
        precomp = S["precomp"]
        out["global_map"] = {
            "best_score": S.get("best_score", None),
            "best_sc_idx": S.get("best_sc_idx", None),
        }
    else:
        # CV path: require precomp + global obs keys
        if precomp is None:
            raise ValueError("skip_global_1to1=True requires a precomputed `precomp`.")
        for k in (OBS_GLOBAL_TYPE, OBS_GLOBAL_SIM):
            if k not in adata_st.obs:
                raise KeyError(
                    f"skip_global_1to1=True expects adata_st.obs[{k!r}] to exist. "
                    "Run global 1:1 mapping in the CV driver first."
                )
        out["global_map"] = {"skipped": True}

    # 3) A_df (candidate type-pair graph)
    pairs_df_st, centroid_df = build_st_pairs_from_knn(
        adata_st, ct_key="global_sc_type", k=cfg.pair_k
    )
    
    pairs_df_final = finalize_pairs(
        pairs_df_st, threshold=cfg.pair_threshold, exclude_sc=False, sc_pair_csv_path=metacell_graph_path
    )

    # 4) choose candidate type pairs
    pairs_selected = list(pairs_df_final[["ct1", "ct2"]].itertuples(index=False, name=None))
    out["pairs_selected"] = pairs_selected

    # 5) bleeding + incoming union
    bleed = compute_directional_bleeding(
        adata_sc,
        adata_st,
        selected_genes=selected_genes_by_ct,   # MUST be dict[ct->DF(index=genes)]
        groupby_sc=cfg.groupby_sc,
        st_type_key=OBS_GLOBAL_TYPE,
        st_sim_key=OBS_GLOBAL_SIM,
        pairs_selected=pairs_selected,
        st_sim_min=cfg.st_sim_min,
        genes_use=None,
        min_cells_sc=cfg.min_cells_sc,
        min_cells_st=cfg.min_cells_st,
        eps=1e-6,
        tau_sc_pos=cfg.tau_sc_pos,
        tau_st_flat=cfg.tau_st_flat,
    )
    incoming_genes = compute_incoming_union(bleed)
    out["incoming_genes"] = incoming_genes

    # 6) pairwise refine + merge
    solver2 = POTFGWSolver(
        method=cfg.solver_method_stage2,
        alpha=cfg.solver_alpha,
        loss_fun=cfg.solver_loss_fun,
        epsilon=cfg.solver_epsilon_stage2,
    )

    refinements = []
    for ct1, ct2 in pairs_selected:
        rr = refine_pair_ct1_ct2(
            adata_sc,
            adata_st,
            ct1=ct1,
            ct2=ct2,
            bleeding_obj=bleed,
            bleeding_mode="bleeding",
            groupby_sc=cfg.groupby_sc,
            sc_geom_obsm_key=cfg.st_geom_key,
            st_geom_obsm_key=cfg.st_geom_key,
            use_anchor_cost=False,
            anchor_key="X_global_sc_anchor",
            beta_expr=0.7,
            solver=solver2,
            precomp=precomp,
            cos_threshold=cfg.cos_threshold,
            score_n_jobs=cfg.n_jobs,
            mass_tie_eps=0,
        )
        refinements.append(rr)

    used = set([a for a, b in pairs_selected] + [b for a, b in pairs_selected])
    all_ct = set(adata_st.obs[OBS_GLOBAL_TYPE].astype(str).unique().tolist())
    for ct in sorted(all_ct - used):
        rr = refine_single_ct(
            adata_sc,
            adata_st,
            ct=ct,
            groupby_sc=cfg.groupby_sc,
            sc_geom_obsm_key=cfg.st_geom_key,
            st_geom_obsm_key=cfg.st_geom_key,
            C_kind_sc="distnace",
            use_anchor_cost=False,
            beta_expr=0.7,
            solver=solver2,
            precomp=precomp,
            cos_threshold=cfg.cos_threshold,
            score_n_jobs=cfg.n_jobs,
        )
        refinements.append(rr)

    merge_result = merge_refinements(
        adata_st,
        refinements,
        overwrite=True,
        only_when_improved=True,
    )
    out["pairwise_merge"] = merge_result

    # 7) anchors
    idx_anchor, anchors_by_type = select_st_anchors_final(
        adata_st,
        n_anchor_per_type=cfg.n_anchor_per_type,
        top_frac_per_type=cfg.top_frac_per_type,
        min_per_type=cfg.min_per_type,
        diversity_bins=cfg.diversity_bins,
        diversity_min_per_grid=cfg.diversity_min_per_grid,
        st_geom_key=cfg.st_geom_key,
        st_type_key=cfg.st_type_key_stage2,
        st_score_key=cfg.st_score_key_stage2,
    )
    out["anchors"] = {"idx_anchor": idx_anchor, "anchors_by_type": anchors_by_type}

    # 8) alignment (phi build)
    idx_st_train = np.flatnonzero(np.asarray(adata_st.obs[cfg.st_score_key_stage2], float) > -np.inf)
    _ = build_phi_before_after(
        adata_sc=adata_sc,
        adata_st=adata_st,
        sc_type_key=cfg.groupby_sc,
        st_type_key=cfg.st_type_key_stage2,
        sc_geom_key=cfg.st_geom_key,
        st_geom_key=cfg.st_geom_key,
        idx_anchor=idx_anchor,
        idx_st_train=idx_st_train,
        weights_A=adata_st.uns[cfg.st_score_key_stage2.replace("score", "weights")],
        direction=cfg.direction_alignment_build,
        min_n=cfg.align_min_per_type,
        shrink_k=cfg.alignment_shrink_k,
        clip_q=(0.01, 0.99),
    )

    # 9) final update (global anchor FGW)
    solver3 = POTFGWSolver(
        method=cfg.solver_method_stage3,
        alpha=cfg.solver_alpha,
        loss_fun=cfg.solver_loss_fun,
        epsilon=cfg.solver_epsilon_stage3,
    )

    tau_anchor = np.percentile(np.asarray(adata_st.obs[cfg.st_score_key_stage2], float), cfg.tau_anchor_percentile)
    cfg3 = GlobalAnchorFGWConfig(
        tau_anchor=float(tau_anchor),
        tau_update=float(cfg.tau_update),
        beta_expr=float(cfg.beta_expr_stage3),
        update_type_key=cfg.st_type_key_stage2,
        update_score_key=cfg.st_score_key_stage2,
        use_alignment=True,
        anchor_k_per_type=cfg.n_anchor_per_type,
        align_min_per_type=cfg.align_min_per_type,
        alignment_shrink_k=cfg.alignment_shrink_k,
        alignment_direction="st_to_sc",
        C_norm="q95",
        sc_geom_key=cfg.st_geom_key,
        st_geom_key=cfg.st_geom_key,
        out_weights_key=UNS_GLOBAL_FINAL_WEIGHTS,
    )

    fg_out = run_final_global_anchor_fgw_update(
        adata_sc,
        adata_st,
        sc_type_key=cfg.groupby_sc,
        cfg=cfg3,
        solver=solver3,
        precomp=precomp,
        incoming_obj=incoming_genes,
        use_incoming_genes=cfg.use_incoming_genes,
        cos_threshold=cfg.cos_threshold,
        update_only_if_improved=cfg.update_only_if_improved,
        compute_imputation=cfg.compute_imputation,
        store_imputation_dense=cfg.store_imputation_dense,
        store_imputation_dict=cfg.store_imputation_dict,
        verbose=verbose,
        logger=print if verbose else (lambda *_: None),
    )
    out["final_global_anchor_fgw"] = fg_out

    if not return_intermediates:
        out = {"cfg": out["cfg"]}

    return out, adata_st


# ------------------------------------------------------------
# CV runner (one-call notebook API)
# ------------------------------------------------------------

def _to_gene_set(obj) -> set:
    if obj is None:
        return set()
    if isinstance(obj, pd.DataFrame):
        if obj.index is not None and len(obj.index) > 0:
            return set(map(str, obj.index))
        for c in ["gene", "genes", "names", "Gene"]:
            if c in obj.columns:
                return set(map(str, obj[c].astype(str).values))
        if obj.shape[1] > 0:
            return set(map(str, obj.iloc[:, 0].astype(str).values))
        return set()
    if isinstance(obj, pd.Series):
        return set(map(str, obj.astype(str).values))
    if isinstance(obj, (list, tuple, set, np.ndarray)):
        return set(map(str, obj))
    return set()


def run_cv(
    adata_sc_full,
    adata_st_full,
    cal_adata_st=None,            # full 결과가 반영된 ST AnnData (옵션)
    full_gene_use_result=None,    # run_pipeline(full) 결과 dict (옵션)
    df_ct_stage=None,             # full 기준 df_ct_stage (옵션)
    *,
    cfg: PipelineConfig = PipelineConfig(),
    n_splits: int = 5,
    test_frac: float = 0.2,
    seed: int = 0,
    min_train_per_ct: int = 20,
    metacell_graph_path: str,
    cell_type_color: pd.DataFrame,
    # ct별 계산 제어
    final_ct_key: Optional[str] = None,  # None이면 K.OBS_FINAL_GLOBAL_TYPE -> K.OBS_FINAL_TYPE
    min_spots_per_ct: int = 3,
    agg_cv_gene: str = "median",         # "median" or "mean"
    verbose: bool = True,
    **pipeline_kwargs,
):
    """
    CV + cell-type specific (DEG_ct - bleeding_ct) gene-wise corr 계산.

    Returns
    -------
    fold_df : DataFrame
    gene_df : DataFrame or None
    test_gene_df : DataFrame or None
    df_ct_stage : DataFrame
        full 결과 기반 ct/stage corr (compute_ct_stage_specific_and_all_gene_corr 결과)
    full_gene_use_result : dict
    ct_gene_df : DataFrame
        fold별 ct-gene corr
        columns: [fold, cell_type, gene, cor, n_used]
    ct_gene_cv : DataFrame
        fold 통합 ct-gene corr
        columns: [cell_type, gene, cor_cv, n_folds_used, n_used_median]
    ct_summary_cv : DataFrame
        ct 요약
    """

    # -----------------------------
    # 0) 기준 ST 객체
    # -----------------------------
    if cal_adata_st is None:
        cal_adata_st = adata_st_full.copy()

    # -----------------------------
    # 1) full 결과 / gene sets 확보
    # -----------------------------
    if full_gene_use_result is not None:
        selected_genes_by_ct = full_gene_use_result["selected_genes_by_ct"]
        _deg = full_gene_use_result["deg_results"]
        _hvg = full_gene_use_result.get("hvg_results", None)
        incoming_genes = full_gene_use_result["incoming_genes"]['incoming_per_target'] 

    else:
        has_full = (
            isinstance(cal_adata_st.uns.get(UNS_FINAL_WEIGHTS, None), dict)
            and len(cal_adata_st.uns.get(UNS_FINAL_WEIGHTS, {})) > 0
        )

        if has_full:
            # full dict 없으면 최소 재구성
            selected_genes_by_ct, _deg, _hvg = select_genes_per_celltype(
                adata_sc_full,
                groupby=cfg.groupby_sc,
                deg_method=cfg.deg_method,
                deg_n_genes=None,
                deg_pval_cutoff=cfg.deg_pval_cutoff,
                deg_logfc_cutoff=cfg.deg_logfc_cutoff,
                hvg_n_top_genes=cfg.hvg_n_top_genes,
                hvg_flavor=cfg.hvg_flavor,
                hvg_layer=None,
                mode=cfg.mode,
            )

            A_df = build_A_df_from_metanode_graph(
                adata_sc_full,
                metacell_graph_path=metacell_graph_path,
                cell_type_color=cell_type_color,
                adata_st=cal_adata_st,
                ref_ct_key=OBS_GLOBAL_TYPE,
                verbose=verbose,
                **pipeline_kwargs,
            )
            pairs_df = select_type_pairs_from_A(A_df, top_n=cfg.top_n_pairs, undirected=True, drop_self=True)
            pairs_selected = list(pairs_df[["type_a", "type_b"]].itertuples(index=False, name=None))

            bleed = compute_directional_bleeding(
                adata_sc_full,
                cal_adata_st,
                selected_genes=selected_genes_by_ct,
                groupby_sc=cfg.groupby_sc,
                st_type_key=OBS_GLOBAL_TYPE,
                st_sim_key=OBS_GLOBAL_SIM,
                pairs_selected=pairs_selected,
                st_sim_min=cfg.st_sim_min,
                genes_use=None,
                min_cells_sc=cfg.min_cells_sc,
                min_cells_st=cfg.min_cells_st,
                eps=1e-6,
                tau_sc_pos=cfg.tau_sc_pos,
                tau_st_flat=cfg.tau_st_flat,
            )
            incoming_genes = compute_incoming_union(bleed)

            full_gene_use_result = dict(
                selected_genes_by_ct=selected_genes_by_ct,
                deg_results=_deg,
                hvg_results=_hvg,
                incoming_genes=incoming_genes,
            )

        else:
            # full 1회
            full_gene_use_result = run_pipeline(
                adata_sc_full,
                cal_adata_st,
                cfg=cfg,
                selected_genes_by_ct=None,
                auto_build_A_df=True,
                recompute_A_df=True,
                metacell_graph_path=metacell_graph_path,
                cell_type_color=cell_type_color,
                return_intermediates=False,
                verbose=False,
                **pipeline_kwargs,
            )
            selected_genes_by_ct = full_gene_use_result["selected_genes_by_ct"]
            _deg = full_gene_use_result["deg_results"]
            _hvg = full_gene_use_result.get("hvg_results", None)
            incoming_genes = full_gene_use_result["incoming_genes"]['incoming_per_target'] 


    # -----------------------------
    # 2) full 기준 df_ct_stage
    # -----------------------------
    if df_ct_stage is None:
        df_ct_stage, _ = compute_ct_stage_specific_and_all_gene_corr(
            adata_sc=adata_sc_full,
            adata_st=cal_adata_st,
            deg_results=_deg,
            bleeding_results=incoming_genes,
            min_spots=1,
            min_genes=1,
            min_n_corr=1,
        )

    # -----------------------------
    # 3) ct 라벨/ct별 eval gene 준비
    # -----------------------------
    if final_ct_key is None:
        final_ct_key = OBS_FINAL_GLOBAL_TYPE
    if final_ct_key not in cal_adata_st.obs:
        raise KeyError(f"cal_adata_st.obs missing final cell-type key: {final_ct_key}")

    ct_all = np.asarray(cal_adata_st.obs[final_ct_key], dtype=object)

    ct_to_eval_genes: Dict[str, set] = {}
    for ct in pd.unique(ct_all):
        if ct is None or (isinstance(ct, float) and np.isnan(ct)):
            continue
        ct = str(ct)
        deg_set = _to_gene_set(_deg.get(ct))
        bleed_set = _to_gene_set(incoming_genes.get(ct, []))
        ct_to_eval_genes[ct] = deg_set - bleed_set

    # -----------------------------
    # 4) CV split
    # -----------------------------
    splits = split_selected_genes_by_ct(
        selected_genes_by_ct,
        n_splits=n_splits,
        test_frac=test_frac,
        seed=seed,
        min_train_per_ct=min_train_per_ct,
    )

    fold_rows: List[Dict[str, Any]] = []
    gene_rows: List[pd.DataFrame] = []
    test_gene_rows: List[pd.DataFrame] = []
    ct_gene_rows: List[pd.DataFrame] = []

    sc_gene_set = set(map(str, adata_sc_full.var_names))
    st_gene_set = set(map(str, cal_adata_st.var_names))

    # -----------------------------
    # 5) fold loop
    # -----------------------------
    for sp in splits:
        fold = int(sp["split_id"])
        train_by_ct = sp["train_by_ct"]
        test_genes = list(map(str, sp["test_genes"]))
        train_union = _genes_union_from_by_ct(train_by_ct)

        if verbose:
            print(
                f"[CV] fold {fold}: train_union={len(train_union)} "
                f"test_union={len(test_genes)} (ct={len(train_by_ct)})"
            )

        ad_sc_train = adata_sc_full[:, train_union].copy()
        ad_st_train = cal_adata_st[:, train_union].copy()

        _ = run_pipeline(
            ad_sc_train,
            ad_st_train,
            cfg=cfg,
            selected_genes_by_ct=train_by_ct,
            auto_build_A_df=True,
            recompute_A_df=True,
            metacell_graph_path=metacell_graph_path,
            cell_type_color=cell_type_color,
            return_intermediates=False,
            verbose=False,
            **pipeline_kwargs,
        )

        w_stage2 = ad_st_train.uns.get(UNS_FINAL_WEIGHTS, None)
        w_stage3 = ad_st_train.uns.get(UNS_GLOBAL_FINAL_WEIGHTS, None)
        if not isinstance(w_stage2, dict) or len(w_stage2) == 0:
            raise KeyError(
                f"CV fold {fold}: missing stage2 weights in adata_st.uns[{UNS_FINAL_WEIGHTS!r}]."
            )
        w_eval = _merge_stage2_stage3_weights(w_stage2, w_stage3)

        genes_eval = [g for g in test_genes if (g in sc_gene_set) and (g in st_gene_set)]
        eval_set = set(genes_eval)

        test_gene_rows.append(
            pd.DataFrame(
                {
                    "fold": fold,
                    "gene": test_genes,
                    "is_eval": [g in eval_set for g in test_genes],
                }
            )
        )

        n_total = int(len(test_genes))
        n_used = int(len(genes_eval))
        if n_used == 0:
            fold_rows.append(
                dict(
                    fold=fold,
                    n_test_total=n_total,
                    n_test_used=0,
                    cor_mean=np.nan,
                    cor_median=np.nan,
                    cor_q10=np.nan,
                    cor_pos_frac=np.nan,
                )
            )
            continue

        X_true = _to_dense(cal_adata_st[:, genes_eval].X)             # (n_st, n_gene_eval)
        X_sc = _to_dense(adata_sc_full[:, genes_eval].X)              # (n_sc, n_gene_eval)
        W = _weights_dict_to_csr(w_eval, n_sc=adata_sc_full.n_obs, n_st=cal_adata_st.n_obs)
        X_pred = np.asarray(W @ X_sc, dtype=float)                    # (n_st, n_gene_eval)

        # fold 전체 요약(기존 유지)
        cor = _gene_wise_pearson_corr(X_true, X_pred)
        cor_valid = cor[np.isfinite(cor)]
        fold_rows.append(
            dict(
                fold=fold,
                n_test_total=n_total,
                n_test_used=int(cor_valid.size),
                cor_mean=float(np.nanmean(cor_valid)) if cor_valid.size else np.nan,
                cor_median=float(np.nanmedian(cor_valid)) if cor_valid.size else np.nan,
                cor_q10=float(np.nanquantile(cor_valid, 0.10)) if cor_valid.size else np.nan,
                cor_pos_frac=float(np.mean(cor_valid > 0)) if cor_valid.size else np.nan,
            )
        )
        gene_rows.append(pd.DataFrame({"fold": fold, "gene": genes_eval, "cor": cor}))

        # -----------------------------
        # 핵심: ct별 (DEG_ct - bleeding_ct)만 계산
        # -----------------------------
        g2j = {g: j for j, g in enumerate(genes_eval)}

        for ct in pd.unique(ct_all):
            if ct is None or (isinstance(ct, float) and np.isnan(ct)):
                continue
            ct = str(ct)

            idx_ct = np.where(ct_all == ct)[0]
            if idx_ct.size < min_spots_per_ct:
                continue

            genes_use = [g for g in genes_eval if g in ct_to_eval_genes.get(ct, set())]
            if len(genes_use) == 0:
                continue

            cols = np.array([g2j[g] for g in genes_use], dtype=int)

            Xt = X_true[idx_ct][:, cols]
            Xp = X_pred[idx_ct][:, cols]

            cor_ct = _gene_wise_pearson_corr(Xt, Xp)

            ct_gene_rows.append(
                pd.DataFrame(
                    {
                        "fold": fold,
                        "cell_type": ct,
                        "gene": genes_use,
                        "corr": cor_ct,
                        "n_used": int(idx_ct.size),
                    }
                )
            )

    # -----------------------------
    # 6) finalize
    # -----------------------------
    fold_df = pd.DataFrame(fold_rows)
    gene_df = pd.concat(gene_rows, ignore_index=True) if gene_rows else None
    test_gene_df = pd.concat(test_gene_rows, ignore_index=True) if test_gene_rows else None

    ct_gene_df = (
        pd.concat(ct_gene_rows, ignore_index=True)
        if ct_gene_rows
        else pd.DataFrame(columns=["fold", "cell_type", "gene", "cor", "n_used"])
    )

    # fold 통합 ct-gene
    if len(ct_gene_df) > 0:
        if agg_cv_gene == "mean":
            ct_gene_cv = (
                ct_gene_df.groupby(["cell_type", "gene"], as_index=False)
                .agg(
                    cor_cv=("cor", "mean"),
                    n_folds_used=("cor", "size"),
                    n_used_median=("n_used", "median"),
                )
            )
        else:
            ct_gene_cv = (
                ct_gene_df.groupby(["cell_type", "gene"], as_index=False)
                .agg(
                    cor_cv=("cor", "median"),
                    n_folds_used=("cor", "size"),
                    n_used_median=("n_used", "median"),
                )
            )

        ct_summary_cv = (
            ct_gene_cv.groupby("cell_type")["cor_cv"]
            .agg(
                n_genes="size",
                cor_mean="mean",
                cor_median="median",
                cor_q10=lambda x: float(np.nanquantile(np.asarray(x), 0.10)),
                cor_pos_frac=lambda x: float(np.mean(np.asarray(x) > 0)),
            )
            .reset_index()
        )
    else:
        ct_gene_cv = pd.DataFrame(columns=["cell_type", "gene", "cor_cv", "n_folds_used", "n_used_median"])
        ct_summary_cv = pd.DataFrame(columns=["cell_type", "n_genes", "cor_mean", "cor_median", "cor_q10", "cor_pos_frac"])

    return (
        fold_df,
        gene_df,
        test_gene_df,
        df_ct_stage,
        full_gene_use_result,
        ct_gene_df,
        ct_gene_cv,
        ct_summary_cv,
    )


def _to_gene_set(obj) -> set:
    if obj is None:
        return set()
    if isinstance(obj, pd.DataFrame):
        if obj.index is not None and len(obj.index) > 0:
            return set(map(str, obj.index))
        for c in ["gene", "genes", "names", "Gene"]:
            if c in obj.columns:
                return set(map(str, obj[c].astype(str).values))
        if obj.shape[1] > 0:
            return set(map(str, obj.iloc[:, 0].astype(str).values))
        return set()
    if isinstance(obj, pd.Series):
        return set(map(str, obj.astype(str).values))
    if isinstance(obj, (list, tuple, set, np.ndarray)):
        return set(map(str, obj))
    return set()


def _safe_pearson_1d(x, y):
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return np.nan
    x = x[m]
    y = y[m]
    if np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _safe_spearman_1d(x, y):
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return np.nan
    x = x[m]
    y = y[m]
    if np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan
    r = spearmanr(x, y, nan_policy="omit").correlation
    return float(r) if np.isfinite(r) else np.nan


def run_loo(
    adata_sc_full,
    adata_st_full,
    cal_adata_st=None,
    full_gene_use_result=None,
    df_all=None,
    *,
    cfg: PipelineConfig = PipelineConfig(),
    n_genes_sample: int = 30,
    seed: int = 0,
    metacell_graph_path: str,
    cell_type_color: pd.DataFrame,
    final_ct_key: Optional[str] = None,
    min_spots_per_ct: int = 3,   # (유지) 현재 계산에는 직접 사용 안 함
    save_dir: str = "./loo_out",
    run_name: str = "loo_run",
    save_each_gene: bool = True,
    verbose: bool = True,
    save_expr_long: bool = True,   # 추가: gene별 expression long 저장 여부
    **pipeline_kwargs,
):

    if cal_adata_st is None:
        cal_adata_st = adata_st_full.copy()

    # -----------------------------
    # 0) 저장 경로 먼저 준비 (버그 수정)
    # -----------------------------
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # 1) full 결과 / gene sets 확보
    # -----------------------------
    if full_gene_use_result is not None:
        selected_genes_by_ct = full_gene_use_result["selected_genes_by_ct"]
        _deg = full_gene_use_result["deg_results"]
        _hvg = full_gene_use_result.get("hvg_results", None)

        # incoming_genes 포맷 normalize
        _inc = full_gene_use_result["incoming_genes"]
        if isinstance(_inc, dict) and ("incoming_per_target" in _inc):
            incoming_genes = _inc["incoming_per_target"]
        else:
            incoming_genes = _inc

    else:
        has_full = (
            isinstance(cal_adata_st.uns.get(UNS_FINAL_WEIGHTS, None), dict)
            and len(cal_adata_st.uns.get(UNS_FINAL_WEIGHTS, {})) > 0
        )

        if has_full:
            selected_genes_by_ct, _deg, _hvg = select_genes_per_celltype(
                adata_sc_full,
                groupby=cfg.groupby_sc,
                deg_method=cfg.deg_method,
                deg_n_genes=None,
                deg_pval_cutoff=cfg.deg_pval_cutoff,
                deg_logfc_cutoff=cfg.deg_logfc_cutoff,
                hvg_n_top_genes=cfg.hvg_n_top_genes,
                hvg_flavor=cfg.hvg_flavor,
                hvg_layer=None,
                mode=cfg.mode,
            )
            
            
            pairs_df_st, centroid_df = build_st_pairs_from_knn(
                cal_adata_st, ct_key="global_sc_type", k=cfg.pair_k
            )

            pairs_df_final = finalize_pairs(
                pairs_df_st, threshold=cfg.pair_threshold, exclude_sc=False, sc_pair_csv_path=metacell_graph_path
            )

            # 4) choose candidate type pairs
            pairs_selected = list(pairs_df_final[["ct1", "ct2"]].itertuples(index=False, name=None))
            

            bleed = compute_directional_bleeding(
                adata_sc_full,
                cal_adata_st,
                selected_genes=selected_genes_by_ct,
                groupby_sc=cfg.groupby_sc,
                st_type_key=OBS_GLOBAL_TYPE,
                st_sim_key=OBS_GLOBAL_SIM,
                pairs_selected=pairs_selected,
                st_sim_min=cfg.st_sim_min,
                genes_use=None,
                min_cells_sc=cfg.min_cells_sc,
                min_cells_st=cfg.min_cells_st,
                eps=1e-6,
                tau_sc_pos=cfg.tau_sc_pos,
                tau_st_flat=cfg.tau_st_flat,
            )
            incoming_genes = compute_incoming_union(bleed)

            full_gene_use_result = dict(
                selected_genes_by_ct=selected_genes_by_ct,
                deg_results=_deg,
                hvg_results=_hvg,
                incoming_genes=incoming_genes,
            )
        else:
            full_gene_use_result = run_pipeline(
                adata_sc_full,
                cal_adata_st,
                cfg=cfg,
                selected_genes_by_ct=None,
                auto_build_A_df=True,
                recompute_A_df=True,
                metacell_graph_path=metacell_graph_path,
                cell_type_color=cell_type_color,
                return_intermediates=False,
                verbose=False,
                **pipeline_kwargs,
            )
            selected_genes_by_ct = full_gene_use_result["selected_genes_by_ct"]
            _deg = full_gene_use_result["deg_results"]
            _hvg = full_gene_use_result.get("hvg_results", None)

            _inc = full_gene_use_result["incoming_genes"]
            if isinstance(_inc, dict) and ("incoming_per_target" in _inc):
                incoming_genes = _inc["incoming_per_target"]
            else:
                incoming_genes = _inc

    # -----------------------------
    # 2) full 기준 df_all
    # -----------------------------
    if df_all is None:
        _, df_all = compute_ct_stage_specific_and_all_gene_corr(
            adata_sc=adata_sc_full,
            adata_st=cal_adata_st,
            deg_results=_deg,
            bleeding_results=incoming_genes,
            min_spots=1,
            min_genes=1,
            min_n_corr=1,
        )
        pd.to_pickle(df_all, out_dir / f"{run_name}_full_data.pkl")

    # -----------------------------
    # 3) gene_pool (selected_genes 기반)
    # -----------------------------
    sc_gene_set = set(map(str, adata_sc_full.var_names))
    st_gene_set = set(map(str, adata_st_full.var_names))
    try:
        selected_genes = _genes_union_from_by_ct_v2(selected_genes_by_ct)
        gene_pool = [g for g in selected_genes if (g in sc_gene_set and g in st_gene_set)]
    except Exception:
        selected_genes = _genes_union_from_by_ct(selected_genes_by_ct)
        gene_pool = [g for g in selected_genes if (g in sc_gene_set and g in st_gene_set)]

    if len(gene_pool) == 0:
        raise ValueError("selected_genes 기반 공통 gene_pool이 비어 있습니다.")
        

    if n_genes_sample < 1:
        raise ValueError("n_genes_sample must be >= 1")
    if n_genes_sample > len(gene_pool):
        n_genes_sample = len(gene_pool)

    rng = np.random.default_rng(seed)
    sampled_genes = list(rng.choice(gene_pool, size=n_genes_sample, replace=False))

    # -----------------------------
    # 4) 평가 cell type key
    # -----------------------------
    if final_ct_key is None:
        if OBS_FINAL_GLOBAL_TYPE in cal_adata_st.obs.columns:
            ct_key = OBS_FINAL_GLOBAL_TYPE
        elif OBS_FINAL_TYPE in cal_adata_st.obs.columns:
            ct_key = OBS_FINAL_TYPE
        elif OBS_GLOBAL_TYPE in cal_adata_st.obs.columns:
            ct_key = OBS_GLOBAL_TYPE
        else:
            raise KeyError("ST obs에서 cell type key를 찾지 못했습니다. final_ct_key를 지정하세요.")
    else:
        ct_key = final_ct_key
        if ct_key not in cal_adata_st.obs.columns:
            raise KeyError(f"'{ct_key}' not found in cal_adata_st.obs")

    spot_ct = cal_adata_st.obs[ct_key].astype(str).values
    spot_ids = cal_adata_st.obs_names.astype(str).values

    # gene -> bleeding CT 집합
    gene_to_bleeding_cts = {}
    for ct, genes in incoming_genes.items():
        for g in _to_gene_set(genes):
            gene_to_bleeding_cts.setdefault(str(g), set()).add(str(ct))

    # -----------------------------
    # 5) 기본 저장
    # -----------------------------
    pd.to_pickle(sampled_genes, out_dir / f"{run_name}_sampled_genes.pkl")

    rows_global = []   # corr 결과 (gene별)
    rows_expr = []     # expr 결과 (spot-gene long)
    n_total = len(sampled_genes)

    # -----------------------------
    # 6) gene-wise LOO
    # -----------------------------
    for i, g in enumerate(sampled_genes, 1):
        if i < 406:
            continue
        
        
        train_genes = [x for x in gene_pool if x != g]
        new_selected_genes_by_ct = {k: list(set(v) - {g}) for k, v in selected_genes_by_ct.items()}

        row_global_g = dict(
            gene=g,
            pearson_excl_bleeding=np.nan,
            spearman_excl_bleeding=np.nan,
            n_valid_spots=0,
            n_total_spots=int(cal_adata_st.n_obs),
            excluded_bleeding_cts=[],
            status="init",
        )
        rows_expr_g = []

        if len(train_genes) == 0:
            row_global_g["status"] = "empty_train_genes"
        else:
            ad_sc_train = adata_sc_full[:, train_genes].copy()
            ad_st_train = adata_st_full[:, train_genes].copy()

            _, ad_st_train = run_pipeline(
                ad_sc_train,
                ad_st_train,
                cfg=cfg,
                selected_genes_by_ct=new_selected_genes_by_ct,
                auto_build_A_df=True,
                recompute_A_df=True,
                metacell_graph_path=metacell_graph_path,
                cell_type_color=cell_type_color,
                return_intermediates=False,
                verbose=False,
                **pipeline_kwargs,
            )

            missing_keys = []
            if OBS_GLOBAL_BEST_CELL_IDX not in ad_st_train.obs:
                missing_keys.append(f"obs:{OBS_GLOBAL_BEST_CELL_IDX}")
            if OBS_FGW_UPDATED_MASK not in ad_st_train.obs:
                missing_keys.append(f"obs:{OBS_FGW_UPDATED_MASK}")
            if UNS_STAGE2_FINAL_WEIGHTS not in ad_st_train.uns:
                missing_keys.append(f"uns:{UNS_STAGE2_FINAL_WEIGHTS}")
            if UNS_GLOBAL_GENES_UNION not in ad_st_train.uns:
                missing_keys.append(f"uns:{UNS_GLOBAL_GENES_UNION}")
            if OBSM_IMPUTED_FINAL_GLOBAL_ANCHOR not in ad_st_train.obsm:
                missing_keys.append(f"obsm:{OBSM_IMPUTED_FINAL_GLOBAL_ANCHOR}")

            if len(missing_keys) > 0:
                row_global_g["status"] = "missing_keys_for_Xf:" + ",".join(missing_keys)

            elif (g not in adata_sc_full.var_names) or (g not in cal_adata_st.var_names):
                row_global_g["status"] = "gene_not_in_sc_or_st"

            else:
                idx_all = np.arange(cal_adata_st.n_obs)

                # true
                X_true = _to_dense(cal_adata_st[:, [g]].X).reshape(-1).astype(np.float32, copy=False)

                # stage1
                X1 = _predict_stage1_bestcell(
                    adata_sc=adata_sc_full,
                    adata_st=ad_st_train,
                    idx_st=idx_all,
                    genes=[g],
                    best_cell_idx_key=OBS_GLOBAL_BEST_CELL_IDX,
                ).reshape(-1)

                # stage2
                X2_mix = _predict_stage2_mixture_Aformat(
                    adata_sc=adata_sc_full,
                    adata_st=ad_st_train,
                    idx_st=idx_all,
                    genes=[g],
                    weight_key=UNS_STAGE2_FINAL_WEIGHTS,
                ).reshape(-1)

                X2 = X1.copy()
                has_mix = np.isfinite(X2_mix)
                X2[has_mix] = X2_mix[has_mix]

                # stage3 FGW overwrite
                Xf = X2.copy()
                upd = np.asarray(ad_st_train.obs[OBS_FGW_UPDATED_MASK], dtype=bool)

                panel_genes = list(map(str, ad_st_train.uns[UNS_GLOBAL_GENES_UNION]))
                if str(g) in panel_genes:
                    col = panel_genes.index(str(g))
                    X_fgw_all = np.asarray(ad_st_train.obsm[OBSM_IMPUTED_FINAL_GLOBAL_ANCHOR])

                    if X_fgw_all.shape[1] != len(panel_genes):
                        row_global_g["status"] = "fgw_shape_mismatch"
                        X_pred = X2
                    else:
                        Xfg = X_fgw_all[:, col].astype(np.float32, copy=False)
                        Xf[upd] = Xfg[upd]
                        X_pred = Xf
                else:
                    X_pred = X2

                # corr + expr (bleeding flag)
                if row_global_g.get("status", "init") in ["init", "ok", "nan_corr", "too_few_spots"]:
                    bleeding_cts = sorted(list(gene_to_bleeding_cts.get(str(g), set())))
                    is_bleeding = pd.Series(spot_ct).isin(bleeding_cts).values
                    include_mask = ~is_bleeding

                    row_global_g["excluded_bleeding_cts"] = bleeding_cts
                    row_global_g["n_valid_spots"] = int(include_mask.sum())

                    # corr: bleeding 제외
                    if include_mask.sum() >= 2:
                        rp = _safe_pearson_1d(X_true[include_mask], X_pred[include_mask])
                        rs = _safe_spearman_1d(X_true[include_mask], X_pred[include_mask])

                        row_global_g["pearson_excl_bleeding"] = rp
                        row_global_g["spearman_excl_bleeding"] = rs
                        if np.isfinite(rp) or np.isfinite(rs):
                            row_global_g["status"] = "ok"
                        else:
                            row_global_g["status"] = "nan_corr"
                    else:
                        row_global_g["status"] = "too_few_spots"

                    # expr 결과: 전체 spot 저장 + bleeding 표시
                    if save_expr_long:
                        rows_expr_g = [
                            dict(
                                gene=str(g),
                                spot_id=str(spot_ids[k]),
                                cell_type=str(spot_ct[k]),
                                expr_true=float(X_true[k]) if np.isfinite(X_true[k]) else np.nan,
                                expr_pred=float(X_pred[k]) if np.isfinite(X_pred[k]) else np.nan,
                                is_bleeding=bool(is_bleeding[k]),
                                excluded_in_corr=bool(is_bleeding[k]),
                            )
                            for k in range(len(spot_ids))
                        ]

        # 누적
        rows_global.append(row_global_g)
        if save_expr_long and len(rows_expr_g) > 0:
            rows_expr.extend(rows_expr_g)

        # gene별 저장
        if save_each_gene:
            df_g_global = pd.DataFrame([row_global_g])
            pd.to_pickle(df_g_global, out_dir / f"{run_name}_gene_{i:04d}_{g}_corr.pkl")

            if save_expr_long:
                df_g_expr = pd.DataFrame(rows_expr_g)
                pd.to_pickle(df_g_expr, out_dir / f"{run_name}_gene_{i:04d}_{g}_expr.pkl")

        if verbose and (i % 10 == 0 or i == n_total):
            print(f"[run_loo] {i}/{n_total} genes done")

    # -----------------------------
    # 7) 최종 통합 저장/반환
    # -----------------------------
    df_corr = pd.DataFrame(rows_global)
    pd.to_pickle(df_corr, out_dir / f"{run_name}_corr_excl_bleeding.pkl")

    if save_expr_long:
        df_expr = pd.DataFrame(rows_expr)
        pd.to_pickle(df_expr, out_dir / f"{run_name}_sc_st_expr_with_bleeding_flag.pkl")
    else:
        df_expr = pd.DataFrame()

    if verbose:
        print(f"[run_loo] saved: {out_dir / f'{run_name}_corr_excl_bleeding.pkl'}")
        if save_expr_long:
            print(f"[run_loo] saved: {out_dir / f'{run_name}_sc_st_expr_with_bleeding_flag.pkl'}")
        print(f"[run_loo] corr rows={len(df_corr)}, expr rows={len(df_expr)}")

    return df_corr, df_expr

