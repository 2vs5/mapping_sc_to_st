# util/genes.py
"""
Gene utilities: DEG / HVG / gene-set selection per cell type.

Style rules
-----------
- This module ONLY handles gene-level computation.
- No filtering, no geometry, no spatial logic.
- Returns plain Python objects (dict / list / DataFrame).
"""

from __future__ import annotations

import warnings
import pandas as pd
try:
    import scanpy as sc
except ImportError: # pragma: no cover
    sc = None

# ============================================================
# Group order helper
# ============================================================

def get_groups_in_stable_order(adata, groupby: str):
    """
    Return group labels in a stable order.
    - categorical: category order
    - otherwise: sorted unique values
    """
    obs_col = adata.obs[groupby]
    if str(obs_col.dtype) == "category":
        return list(obs_col.cat.categories.astype(str))
    return sorted(pd.unique(obs_col.dropna()).astype(str))

# ============================================================
# DEG
# ============================================================

def compute_deg_per_celltype(
    adata,
    *,
    groupby: str = "cell_type",
    method: str = "wilcoxon",
    n_genes: Optional[int] = None,
    pval_cutoff: float = 0.05,
    logfc_cutoff: Optional[float] = 0.25,
    key_added: Optional[str] = None,
    use_raw: bool = False,
):
    """
    Compute DEGs per cell type using scanpy.rank_genes_groups.

    Returns
    -------
    deg_results : dict[str, pd.DataFrame]
        index = gene names
    """
    if groupby not in adata.obs:
        raise KeyError(f"[compute_deg_per_celltype] '{groupby}' not in adata.obs")

    rg_kwargs = dict(
        groupby=groupby,
        method=method,
        n_genes=n_genes,
        use_raw=use_raw,
    )
    if key_added is not None:
        rg_kwargs["key_added"] = key_added

    sc.tl.rank_genes_groups(adata, **rg_kwargs)

    deg_results: Dict[str, pd.DataFrame] = {}
    groups = get_groups_in_stable_order(adata, groupby)

    for ct in groups:
        if key_added is None:
            df = sc.get.rank_genes_groups_df(adata, group=ct)
        else:
            df = sc.get.rank_genes_groups_df(adata, group=ct, key=key_added)

        pcol = "pvals_adj" if "pvals_adj" in df.columns else "pvals"
        df = df[df[pcol] < float(pval_cutoff)]

        if logfc_cutoff is not None and "logfoldchanges" in df.columns:
            df = df[df["logfoldchanges"].abs() > float(logfc_cutoff)]

        df = df.set_index("names")
        df.index = df.index.astype(str)
        deg_results[str(ct)] = df

    return deg_results

# ============================================================
# HVG (per cell type)
# ============================================================

def compute_hvg_per_celltype(
    adata,
    *,
    groupby: str = "cell_type",
    n_top_genes: int = 2000,
    flavor: str = "seurat_v3",
    layer: Optional[str] = None,
    warn_if_layer_none: bool = True,
):
    """
    Compute HVGs separately for each cell type.

    Returns
    -------
    hvg_results : dict[str, list[str]]
    """
    if groupby not in adata.obs:
        raise KeyError(f"[compute_hvg_per_celltype] '{groupby}' not in adata.obs")

    if layer is None and warn_if_layer_none:
        warnings.warn(
            "HVGs computed from adata.X (layer=None). "
            "Ensure preprocessing consistency.",
            RuntimeWarning,
        )

    hvg_results: Dict[str, List[str]] = {}
    groups = get_groups_in_stable_order(adata, groupby)

    for ct in groups:
        ad_ct = adata[adata.obs[groupby].astype(str) == str(ct)].copy()

        sc.pp.highly_variable_genes(
            ad_ct,
            n_top_genes=int(n_top_genes),
            flavor=flavor,
            subset=False,
            layer=layer,
        )

        hvgs = ad_ct.var_names[ad_ct.var["highly_variable"]].astype(str).tolist()
        hvg_results[str(ct)] = hvgs

    return hvg_results

# ============================================================
# Combine DEG + HVG
# ============================================================

def select_genes_per_celltype(
    adata,
    *,
    groupby: str = "cell_type",

 # DEG
    deg_method: str = "wilcoxon",
    deg_n_genes: Optional[int] = None,
    deg_pval_cutoff: float = 0.05,
    deg_logfc_cutoff: Optional[float] = 0.25,
    deg_key_added: Optional[str] = None,
    deg_use_raw: bool = False,

 # HVG
    hvg_n_top_genes: int = 2000,
    hvg_flavor: str = "seurat_v3",
    hvg_layer: Optional[str] = None,
    hvg_warn_if_layer_none: bool = True,

 # selection
    mode: str = "union", # "union" | "intersection"
    min_selected_genes: int = 0,
):
    """
    Combine DEG and HVG to produce final gene sets per cell type.
    """
    deg_results = compute_deg_per_celltype(
        adata,
        groupby=groupby,
        method=deg_method,
        n_genes=deg_n_genes,
        pval_cutoff=deg_pval_cutoff,
        logfc_cutoff=deg_logfc_cutoff,
        key_added=deg_key_added,
        use_raw=deg_use_raw,
    )

    hvg_results = compute_hvg_per_celltype(
        adata,
        groupby=groupby,
        n_top_genes=hvg_n_top_genes,
        flavor=hvg_flavor,
        layer=hvg_layer,
        warn_if_layer_none=hvg_warn_if_layer_none,
    )

    selected: Dict[str, List[str]] = {}
    groups = get_groups_in_stable_order(adata, groupby)

    for ct in groups:
        ct = str(ct)
        deg_genes = set(deg_results.get(ct, pd.DataFrame()).index)
        hvg_genes = set(hvg_results.get(ct, []))

        if mode == "intersection":
            genes = deg_genes & hvg_genes
            if min_selected_genes and len(genes) < min_selected_genes:
                genes = deg_genes | hvg_genes
        elif mode == "union":
            genes = deg_genes | hvg_genes
        else:
            raise ValueError(f"Unknown mode: {mode}")

        selected[ct] = sorted(map(str, genes))

    selected_genes = {k: pd.DataFrame(index = v)for k, v in selected.items()}

    return selected_genes, deg_results, hvg_results
