"""
Visualization helpers for the sc->ST mapping pipeline.

Design goals
------------
- Pure helper functions (no side effects unless requested).
- Uses matplotlib + networkx. (Scanpy is optional and only required for UMAP plotting.)
- Works even when some cell types have no edges: keep isolated nodes.
- Uses provided `cell_type_color` (pandas DataFrame with columns: ['cell_type','color'])
  or a dict {cell_type: color}.

Main figures you asked for previously
-------------------------------------
1) Type-pair proximity graph (from run_proximity_pair_selection results)
   - node color: cell_type_color
   - remove self edges
   - edge width: function of weight (e.g., inverse distance or a provided "weight" field)
   - edge label: numeric value (median_d_st or other)

2) Score distributions (hist / boxplots)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import networkx as nx
from matplotlib.lines import Line2D
import scanpy as sc
import scipy.sparse as sp
from matplotlib.collections import LineCollection
import seaborn as sns

from .keys import OBS_GLOBAL_SIM, OBS_FINAL_TYPE, OBS_FINAL_SCORE

def sim_score_box_plot(
    adata,
    pairs, # [("type_key","score_key"), ...] or [{"type_key":..,"score_key":..,"label":..}, ...]
    *,
    type_mode="intersection", # "intersection" | "union"
    include_all=True,
    all_label="all",
    strip_types=True,
    dropna=True,
    drop_uncertain=False,
    uncertain_label="uncertain",
    ct_order="sorted", # "sorted" | "appearance" | list
    rotate_xticks=45,
    figsize=(20, 6),
    showfliers=True,
    legend_out=True, # place legend outside
    legend_ncol=1,
    bottom_margin=0.30, # bottom margin for long labels
    right_margin=0.82, # right margin when legend_out
    show=True,
):
    """
    Compare up to 3 scores per cell type using a boxplot with hue.
    Optionally include an 'all' group per score label.
    Returns: df_long, fig, ax
      - df_long columns: ["type", "label", "value"]
    """

    if not isinstance(pairs, (list, tuple)) or len(pairs) == 0:
        raise ValueError("`pairs` must contain at least one item.")
    if len(pairs) > 3:
        raise ValueError("`pairs` supports up to 3 items.")

 # normalize + validate
    norm = []
    for p in pairs:
        if isinstance(p, dict):
            tk = p["type_key"]
            sk = p["score_key"]
            label = p.get("label", sk)
        else:
            tk, sk = p
            label = sk
        if tk not in adata.obs:
            raise KeyError(f"'{tk}' not found in adata.obs")
        if sk not in adata.obs:
            raise KeyError(f"'{sk}' not found in adata.obs")
        norm.append((tk, sk, label))

 # build long df
    chunks = []
    for tk, sk, label in norm:
        df = adata.obs[[tk, sk]].copy()
        df.columns = ["type", "value"]
        df["label"] = label

        df["type"] = df["type"].astype(str)
        if strip_types:
            df["type"] = df["type"].str.strip()

        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        if dropna:
            df = df.dropna(subset=["type", "value"])
        if drop_uncertain:
            df = df[df["type"] != uncertain_label]

        if include_all:
            df_all = df[["value"]].copy()
            df_all["type"] = all_label
            df_all["label"] = label
            df = pd.concat([df_all, df], ignore_index=True)

        chunks.append(df)

    df_long = pd.concat(chunks, ignore_index=True)
    if df_long.empty:
        raise ValueError("No data after filtering.")

    labels = [x[2] for x in norm]

    type_sets = [set(df_long.loc[df_long["label"] == lb, "type"].unique()) for lb in labels]
    if type_mode == "intersection":
        types = sorted(set.intersection(*type_sets))
    elif type_mode == "union":
        types = sorted(set.union(*type_sets))
    else:
        raise ValueError("`type_mode` must be 'intersection' or 'union'.")

    if include_all and all_label not in types:
        types = [all_label] + types
    elif include_all and types[0] != all_label:
        types = [all_label] + [t for t in types if t != all_label]

 # order
    types_set = set(types)
    if isinstance(ct_order, (list, tuple, np.ndarray)):
        order = [t for t in ct_order if t in types_set]
        if include_all and all_label in types_set and all_label not in order:
            order = [all_label] + order
    elif ct_order == "appearance":
        first = labels[0]
        seen = df_long.loc[df_long["label"] == first, "type"].tolist()
        order, s = [], set()
        for t in seen:
            if t in types_set and t not in s:
                order.append(t); s.add(t)
        if include_all and all_label in types_set:
            order = [all_label] + [t for t in order if t != all_label]
        for t in types:
            if t not in s:
                order.append(t)
    else:
        order = types

 # plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(
        data=df_long[df_long["type"].isin(order)],
        x="type",
        y="value",
        hue="label",
        order=order,
        dodge=True,
        showfliers=showfliers,
        linewidth=1.0,
        ax=ax,
    )

    ax.grid(False)
    ax.set_xlabel("")
    ax.set_ylabel("score")
    ax.set_title("Scores by cell type")

    ax.tick_params(axis="x", labelrotation=rotate_xticks)
    for lab in ax.get_xticklabels():
        lab.set_ha("right")

 # legend
    if legend_out:
        ax.legend(
            title=None,
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            ncol=legend_ncol,
        )
        fig.subplots_adjust(bottom=bottom_margin, right=right_margin)
    else:
        ax.legend(title=None, frameon=False, loc="best")
        fig.subplots_adjust(bottom=bottom_margin)

    if show:
        plt.show()

    return df_long, fig, ax

def plot_umap_with_category_colors(
    adata,
    color_key,
    cell_type_color,
    cell_type_col="cell_type",
    color_col="color",
    default_color="#808080",
    subset_mask=None,
    subset_query=None,
    inplace_subset=False,
    umap_kwargs=None,
    debug=True,
    fig_size = (10,10),
):
    """
    Plot UMAP with a categorical obs column using a user-provided color mapping.

    This function:
      1) optionally subsets adata
      2) ensures adata.obs[color_key] is categorical and removes unused categories
      3) builds a palette aligned to the category order
      4) writes colors into adata.uns[f"{color_key}_colors"] (Scanpy convention)
      5) calls sc.pl.umap(...)

    Args:
        adata: AnnData
        color_key: str
            obs column name to color by (e.g., "global_sc_type" or "global_sc_anchor_type")
        cell_type_color: pandas.DataFrame
            Must contain columns [cell_type_col, color_col], e.g. ["cell_type","color"]
        cell_type_col: str
            Column name in cell_type_color for category labels
        color_col: str
            Column name in cell_type_color for hex colors
        default_color: str
            Color used when a category is missing in cell_type_color
        subset_mask: array-like bool (optional)
            Boolean mask to subset rows. Mutually exclusive with subset_query.
        subset_query: str (optional)
            Query string for adata.obs.eval/query, e.g. "batch == 'E1'".
        inplace_subset: bool
            If True, modifies input adata by subsetting in place. If False, uses a copy.
        umap_kwargs: dict (optional)
            Extra kwargs forwarded to sc.pl.umap (size, alpha, legend_loc, etc.)
        debug: bool
            Print missing categories and summary.

    Returns:
        adata_used: AnnData
            The AnnData object actually plotted (subset copy if requested).
    """
    try:
        import scanpy as sc
    except ImportError as e:
        raise ImportError(
            "plot_umap_with_category_colors requires scanpy. Install scanpy to use this function."
        ) from e

    sc.set_figure_params(figsize=fig_size, dpi=120)

    if umap_kwargs is None:
        umap_kwargs = {}

 # --- subset handling ---
    if subset_mask is not None and subset_query is not None:
        raise ValueError("Provide only one of subset_mask or subset_query.")

    if subset_query is not None:
 # use pandas eval on obs; safer to use .query if possible
        mask = adata.obs.eval(subset_query)
        mask = np.asarray(mask, dtype=bool)
    elif subset_mask is not None:
        mask = np.asarray(subset_mask, dtype=bool)
        if mask.shape[0] != adata.n_obs:
            raise ValueError(f"subset_mask length {mask.shape[0]} != adata.n_obs {adata.n_obs}")
    else:
        mask = None

    if mask is not None:
        if inplace_subset:
            adata._inplace_subset_obs(mask)
            adata_used = adata
        else:
            adata_used = adata[mask].copy()
    else:
        adata_used = adata

 # --- check key exists ---
    if color_key not in adata_used.obs:
        raise KeyError(f"adata.obs does not contain '{color_key}'")

 # --- enforce categorical + remove unused ---
    adata_used.obs[color_key] = adata_used.obs[color_key].astype("category")
    adata_used.obs[color_key] = adata_used.obs[color_key].cat.remove_unused_categories()

 # --- build color map from DF ---
    if cell_type_col not in cell_type_color.columns or color_col not in cell_type_color.columns:
        raise KeyError(
            f"cell_type_color must have columns '{cell_type_col}' and '{color_col}'. "
            f"Got {list(cell_type_color.columns)}"
        )

    color_map = dict(zip(cell_type_color[cell_type_col], cell_type_color[color_col]))

 # --- align palette to category order ---
    cats = adata_used.obs[color_key].cat.categories
    missing = [c for c in cats if c not in color_map]

    if debug:
        print(f"[plot_umap_with_category_colors] key='{color_key}'")
        print(f"  n_obs={adata_used.n_obs}, n_categories={len(cats)}")
        print("  missing in color_map:", missing)

    adata_used.uns[f"{color_key}_colors"] = [color_map.get(c, default_color) for c in cats]

 # --- plot ---
    sc.pl.umap(
        adata_used,
        color=color_key,
        **umap_kwargs,
    )

def plot_umap_st_cell_type(
    adata,
    cell_type_color,
    keys=("global_sc_type", "global_sc_anchor_type"),
    subset_mask=None,
    subset_query=None,
    inplace_subset=False,
    umap_kwargs_common=None,
    debug=True,
    fig_size=(12, 5),
    wspace=0.25,
):
    """
    Plot multiple `.obs` keys in a single 1xN figure.
    The figure is created once (index shown once).
    Returns: adata_used
    """

    if umap_kwargs_common is None:
        umap_kwargs_common = dict(size=40, alpha=0.8, legend_loc="right margin", frameon=False)

 # subset once
    if subset_query is not None:
        mask = np.asarray(adata.obs.eval(subset_query), dtype=bool)
    elif subset_mask is not None:
        mask = np.asarray(subset_mask, dtype=bool)
        if mask.shape[0] != adata.n_obs:
            raise ValueError("subset_mask length mismatch.")
    else:
        mask = None

    if mask is not None:
        if inplace_subset:
            adata._inplace_subset_obs(mask)
            adata_used = adata
        else:
            adata_used = adata[mask].copy()
    else:
        adata_used = adata

    keys = [k for k in keys if k in adata_used.obs]
    if len(keys) == 0:
        if debug:
            print("[plot_umap_global_types_row] no valid keys to plot.")
        return adata_used

 # Create axes horizontally in one figure
    n = len(keys)
    sc.set_figure_params(dpi=120)
    fig, axes = plt.subplots(1, n, figsize=fig_size, squeeze=False)
    axes = axes[0]
    plt.subplots_adjust(wspace=wspace)

    for i, (ax, k) in enumerate(zip(axes, keys)):
        is_last = (i == len(keys) - 1)

        kwargs = {**umap_kwargs_common, "show": False, "ax": ax}

 #  Show legend only on the last panel
        if not is_last:
            kwargs["legend_loc"] = None
            kwargs["frameon"] = False

        plot_umap_with_category_colors(
            adata_used,
            color_key=k,
            cell_type_color=cell_type_color,
            subset_mask=None,
            subset_query=None,
            inplace_subset=True,
            umap_kwargs=kwargs,
            debug=debug,
        )

 #  If a legend is created, keep it only on the last panel
        leg = ax.get_legend()
        if (leg is not None) and (not is_last):
            leg.remove()

        ax.set_title(k)
    return adata_used

def plot_distribution_of_pairwise_cell_type_distance(
    pair_info,
    distance_type="median_d",
    bins=30,
    use_selected_threshold="max_selected", # "max_selected" | "quantile_of_all"
    cut_off_val=0.2,
    figsize=(6, 4),
):
    """
    Plot histogram of distances over ALL type-pairs (pair_info['results']),
    and draw a red vertical line indicating the boundary corresponding to pairs_selected.

    - If use_selected_threshold="max_selected":
        red line = max distance among selected pairs (pairs_selected).
        This acts like an empirical cutoff: <= line ~ selected.
    - If use_selected_threshold="quantile_of_all":
        red line = quantile of ALL distances (legacy behavior).
    """
    import numpy as np
    import matplotlib.pyplot as plt

 # --- all distances (histogram population) ---
    all_d = np.array(
        [r[distance_type] for r in pair_info["results"] if np.isfinite(r.get(distance_type, np.nan))],
        dtype=float
    )
    if all_d.size == 0:
        raise ValueError("No finite distances found in pair_info['results'].")

    cutoff = None

    if use_selected_threshold == "max_selected":
 # build lookup from pair -> distance
        res_by_pair = {}
        for r in pair_info.get("results", []) or []:
            a = str(r.get("type_a"))
            b = str(r.get("type_b"))
            key = tuple(sorted((a, b)))
            val = r.get(distance_type, np.nan)
            if np.isfinite(val):
 # keep the smallest if duplicates exist
                if key not in res_by_pair or float(val) < float(res_by_pair[key]):
                    res_by_pair[key] = float(val)

        selected = pair_info.get("pairs_selected", []) or []
        sel_d = []
        for a, b in selected:
            key = tuple(sorted((str(a), str(b))))
            if key in res_by_pair:
                sel_d.append(res_by_pair[key])

        sel_d = np.asarray(sel_d, dtype=float)
        sel_d = sel_d[np.isfinite(sel_d)]

        if sel_d.size == 0:
            raise ValueError("No finite distances found for pairs_selected. Check pair_info content.")

        cutoff = float(np.max(sel_d)) # boundary between selected and the rest (empirical)

    elif use_selected_threshold == "quantile_of_all":
        cutoff = float(np.quantile(all_d, float(cut_off_val)))

    else:
        raise ValueError("use_selected_threshold must be 'max_selected' or 'quantile_of_all'.")

 # --- plot ---
    plt.figure(figsize=figsize)
    plt.hist(all_d, bins=int(bins))
    plt.axvline(cutoff, color="red", linestyle="--", label=f"selected cutoff: {cutoff:.3f}")
    plt.xlabel(f"{distance_type}")
    plt.ylabel("Number of cell-type pairs")
    plt.title(f"Distribution of {distance_type} (ALL type pairs)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

def plot_fc_scatter_by_pair(
    bleeding_obj,
    deg_results,
    *,
    tau_sc_pos=1.0,
    tau_st_flat=0.2,
    use_abs_st=True,
    require_deg_src=True,
    pairs=None,
    ncols=4,
    max_genes_per_pair=None,
    point_size=8,
    alpha=0.35,
):
    genes = list(bleeding_obj["genes"])
    logFC_sc = bleeding_obj["logFC_sc"]
    logFC_st = bleeding_obj["logFC_st"]

    if pairs is None:
        pairs = list(logFC_sc.keys())
    else:
        pairs = [(str(a), str(b)) for a, b in pairs if (str(a), str(b)) in logFC_sc]

    if len(pairs) == 0:
        raise ValueError("No pairs to plot (check pairs argument vs bleeding_obj keys).")

    n = len(pairs)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 3.6*nrows), squeeze=False)

    for k, (src, tgt) in enumerate(pairs):
        ax = axes[k // ncols][k % ncols]

        lsc = np.asarray(logFC_sc[(src, tgt)], dtype=float)
        lst = np.asarray(logFC_st[(src, tgt)], dtype=float)
        y = np.abs(lst) if use_abs_st else lst

 # DEG(src) mask
        deg_src = set(deg_results.get(src, pd.DataFrame()).index.tolist())
        is_deg_src = np.array([g in deg_src for g in genes], dtype=bool)
        keep = is_deg_src if require_deg_src else np.ones(len(genes), dtype=bool)

        idx = np.where(keep)[0]
        if max_genes_per_pair is not None and len(idx) > int(max_genes_per_pair):
            rng = np.random.default_rng(0)
            idx = rng.choice(idx, size=int(max_genes_per_pair), replace=False)

        x = lsc[idx]
        yy = y[idx]

 # bleeding decision
        if use_abs_st:
            bleed = (x > float(tau_sc_pos)) & (yy < float(tau_st_flat))
        else:
            bleed = (x > float(tau_sc_pos)) & (yy < float(tau_st_flat))

 # background points
        ax.scatter(x[~bleed], yy[~bleed], s=point_size, alpha=alpha, rasterized=True)

 # highlight bleeding points
        ax.scatter(x[bleed], yy[bleed], s=point_size*1.2, alpha=min(0.9, alpha+0.2),
                   color="crimson", rasterized=True)

 # threshold
        ax.axvline(tau_sc_pos, ls="--", lw=1.2, color="crimson")
        ax.axhline(tau_st_flat, ls="--", lw=1.2, color="crimson")

        ax.set_title(f"{src} -> {tgt}\n(n={len(idx)}, bleed={int(bleed.sum())})")
        ax.set_xlabel("logFC_sc")
        ax.set_ylabel("|logFC_st|" if use_abs_st else "logFC_st")

 # clear unused subplots
    for kk in range(n, nrows*ncols):
        axes[kk // ncols][kk % ncols].axis("off")

    fig.suptitle("QC: gene-level SC vs ST contrast by (src -> tgt)", y=1.01, fontsize=14)
    plt.tight_layout()
    return fig

def umap_or_pca_2d(X, random_state=0):
    """
    Return a 2D embedding: UMAP if available, otherwise PCA.
    """
    try:
        import umap
        reducer = umap.UMAP(
            n_neighbors=30,
            min_dist=0.3,
            metric="euclidean",
            random_state=random_state,
        )
        return reducer.fit_transform(X)
    except Exception:
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=random_state).fit_transform(X)

def plot_phi_umap_compare(
    *,
 # --- phi inputs ---
    phi_st_train,
    phi_sc_raw_train,
 # aligned (either is fine)
    phi_sc_aligned_train=None,
    phi_st_aligned_train=None,

    phi_st_all=None,

 # --- types ---
    st_types_train=None,
    sc_types_train=None,
    st_types_all=None,

 # --- plotting options ---
    cell_type_color=None,
    max_points=50000,
    random_state=0,
    method="auto",
    default_color="#BDBDBD",
    legend_max=30,
    figsize=(14, 10),
    show_type_legend=False,

    st_view="train", # "train" | "all"

    direction="sc_to_st", # "sc_to_st" | "st_to_sc"
):
    """
      (UMAP/PCA) 3  .

    direction="sc_to_st":
        - ST (train/all), SC raw, SC aligned  (  )
    direction="st_to_sc":
        - ST (train/all), SC raw, ST aligned  (  ST )

    Notes
    -----
    - direction  aligned   .
      sc_to_st: phi_sc_aligned_train
      st_to_sc: phi_st_aligned_train
    """

    direction = str(direction).lower().strip()
    if direction not in ("sc_to_st", "st_to_sc"):
        raise ValueError('direction must be "sc_to_st" or "st_to_sc"')

 # ----------------------------
 # helpers
 # ----------------------------
    def _to_color_map(ctc):
        if ctc is None:
            return None
        if isinstance(ctc, dict):
            return {str(k): str(v) for k, v in ctc.items()}
        try:
            import pandas as pd
            if isinstance(ctc, pd.DataFrame):
                if not {"cell_type", "color"}.issubset(ctc.columns):
                    raise ValueError("cell_type_color DataFrame ['cell_type','color']  .")
                return dict(zip(ctc["cell_type"].astype(str), ctc["color"].astype(str)))
        except Exception:
            pass
        raise TypeError("cell_type_color dict  (cell_type,color) DataFrame .")

    color_map = _to_color_map(cell_type_color)

    def _colors_for_types(types_arr):
        if types_arr is None or color_map is None:
            return None
        types_arr = np.asarray(types_arr).astype(str)
        return [color_map.get(t, default_color) for t in types_arr]

    def _add_type_legend(ax, uniq_types, title="cell_type"):
        if (not show_type_legend) or (color_map is None):
            return
        uniq_types = [str(t) for t in uniq_types]
        if len(uniq_types) > legend_max:
            return
        handles = [
            Line2D([0], [0], marker="o", linestyle="",
                   markerfacecolor=color_map.get(t, default_color),
                   markeredgecolor="none", label=t, markersize=6)
            for t in uniq_types
        ]
        ax.legend(handles=handles, title=title,
                  bbox_to_anchor=(1.02, 1), loc="upper left",
                  frameon=False)

    def _subsample(X, types, nmax):
        X = np.asarray(X)
        n = X.shape[0]
        if types is not None:
            types = np.asarray(types)
        if n <= nmax:
            return X, types
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=nmax, replace=False)
        return (X[idx], None) if types is None else (X[idx], types[idx])

    def _embed_2d(X, method_, random_state_):
        method_ = method_.lower()
        if method_ == "umap":
            import umap
            return umap.UMAP(
                n_components=2,
                random_state=random_state_,
                n_neighbors=30,
                min_dist=0.3,
            ).fit_transform(X)
        if method_ == "pca":
            from sklearn.decomposition import PCA
            return PCA(n_components=2, random_state=random_state_).fit_transform(X)
        raise ValueError(f"Unknown method: {method_}")

    def _resolve_methods(method_):
        m = str(method_).lower()
        if m in ("umap", "pca"):
            return [m]
        if m == "both":
            try:
                import umap # noqa
                return ["umap", "pca"]
            except Exception:
                return ["pca"]
        if m == "auto":
            try:
                import umap # noqa
                return ["umap"]
            except Exception:
                return ["pca"]
        raise ValueError('method "auto" | "umap" | "pca" | "both"   .')

 # ----------------------------
 # choose ST view (train vs all)
 # ----------------------------
    if st_view == "all":
        if phi_st_all is None:
            raise ValueError("st_view='all' phi_st_all   .")
        phi_st_use = phi_st_all
        st_types_use = st_types_all
    elif st_view == "train":
        phi_st_use = phi_st_train
        st_types_use = st_types_train
    else:
        raise ValueError("st_view 'train'  'all'  .")

 # ----------------------------
 # pick which aligned cloud to plot, based on direction
 # ----------------------------
    if direction == "sc_to_st":
        if phi_sc_aligned_train is None:
            raise ValueError('direction="sc_to_st" phi_sc_aligned_train  .')
        third_name = "SC aligned"
        third_phi = phi_sc_aligned_train
        third_types = sc_types_train
    else:
        if phi_st_aligned_train is None:
            raise ValueError('direction="st_to_sc" phi_st_aligned_train  .')
        third_name = "ST aligned"
        third_phi = phi_st_aligned_train
        third_types = st_types_train

 # subsample each cloud independently
    phi_st_s, st_t_s = _subsample(phi_st_use, st_types_use, max_points)
    phi_raw_s, sc_t_s = _subsample(phi_sc_raw_train, sc_types_train, max_points)
    phi_thd_s, thd_t_s = _subsample(third_phi, third_types, max_points)

 # concatenate for a single embedding fit
    X_all = np.vstack([phi_st_s, phi_raw_s, phi_thd_s])
    n1 = phi_st_s.shape[0]
    n2 = phi_raw_s.shape[0]

    methods = _resolve_methods(method)

    for m in methods:
        emb = _embed_2d(X_all, m, random_state)
        E_st = emb[:n1]
        E_raw = emb[n1:n1 + n2]
        E_thd = emb[n1 + n2:]

        fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
        ax00, ax01 = axes[0, 0], axes[0, 1]
        ax10, ax11 = axes[1, 0], axes[1, 1]

 # 00: overlay
        ax00.scatter(E_st[:, 0], E_st[:, 1], s=3, alpha=0.5, label=f"ST ({st_view})")
        ax00.scatter(E_raw[:, 0], E_raw[:, 1], s=3, alpha=0.5, label="SC raw")
        ax00.scatter(E_thd[:, 0], E_thd[:, 1], s=3, alpha=0.5, label=third_name)
        ax00.set_title(f"ST({st_view}) vs SC raw vs {third_name} [{m.upper()}] ({direction})")
        ax00.set_xlabel("dim1"); ax00.set_ylabel("dim2")
        ax00.grid(False)
        ax00.legend(frameon=False)

 # legend candidates
        uniq_for_legend = None
        if st_t_s is not None:
            uniq_for_legend = np.unique(np.asarray(st_t_s).astype(str))
        elif sc_t_s is not None:
            uniq_for_legend = np.unique(np.asarray(sc_t_s).astype(str))
        elif thd_t_s is not None:
            uniq_for_legend = np.unique(np.asarray(thd_t_s).astype(str))

 # 01: ST colored
        if st_t_s is not None:
            ax01.scatter(E_st[:, 0], E_st[:, 1], s=3, alpha=0.7, c=_colors_for_types(st_t_s))
            ax01.set_title(f"ST ({st_view}, colored by type) [{m.upper()}]")
            ax01.set_xlabel("dim1"); ax01.set_ylabel("dim2")
            ax01.grid(False)
            if uniq_for_legend is not None:
                _add_type_legend(ax01, uniq_for_legend)
        else:
            ax01.set_axis_off()

 # 10: SC raw colored
        if sc_t_s is not None:
            ax10.scatter(E_raw[:, 0], E_raw[:, 1], s=3, alpha=0.7, c=_colors_for_types(sc_t_s))
            ax10.set_title(f"SC raw (colored by type) [{m.upper()}]")
            ax10.set_xlabel("dim1"); ax10.set_ylabel("dim2")
            ax10.grid(False)
            if uniq_for_legend is not None:
                _add_type_legend(ax10, uniq_for_legend)
        else:
            ax10.set_axis_off()

 # 11: third cloud colored (SC aligned or ST aligned)
        if thd_t_s is not None:
            ax11.scatter(E_thd[:, 0], E_thd[:, 1], s=3, alpha=0.7, c=_colors_for_types(thd_t_s))
            ax11.set_title(f"{third_name} (colored by type) [{m.upper()}]")
            ax11.set_xlabel("dim1"); ax11.set_ylabel("dim2")
            ax11.grid(False)
            if uniq_for_legend is not None:
                _add_type_legend(ax11, uniq_for_legend)
        else:
            ax11.set_axis_off()

        plt.show()

def plot_axis_histograms(
    *,
    phi_st_train,
    phi_sc_raw_train,
    phi_sc_aligned_train,
    axes_to_plot=(0, 1, 2),
    bins=60,
):
    """
    (axis)    ST vs SC(raw) vs SC(aligned)   .
    """
    K = phi_st_train.shape[1]
    axes_to_plot = [a for a in axes_to_plot if 0 <= a < K]
    for k in axes_to_plot:
        plt.figure()
        plt.hist(phi_st_train[:, k], bins=bins, alpha=0.5, label="ST train")
        plt.hist(phi_sc_raw_train[:, k], bins=bins, alpha=0.5, label="SC raw")
        plt.hist(phi_sc_aligned_train[:, k], bins=bins, alpha=0.5, label="SC aligned")
        plt.title(f"Axis {k} distribution (phi)")
        plt.xlabel("distance to anchor")
        plt.ylabel("count")
        plt.legend()
        plt.show()

def apply_obs_colors_from_df(
    adata,
    obs_key: str,
    color_df: pd.DataFrame,
    key_col: str = "cell_type",
    color_col: str = "color",
    make_categorical: bool = True,
    fill_missing: str = "lightgray",
):
    """
    color_df (key_col, color_col)  adata.uns[f"{obs_key}_colors"] .
    Scanpy obs_key categorical  categories   colors list .
    """
    if obs_key not in adata.obs:
        raise KeyError(f"adata.obs['{obs_key}']  .")

    if not isinstance(color_df, pd.DataFrame):
        raise TypeError("color_df pandas DataFrame .")

    if key_col not in color_df.columns or color_col not in color_df.columns:
        raise KeyError(f"color_df '{key_col}', '{color_col}'  .")

    if make_categorical and not pd.api.types.is_categorical_dtype(adata.obs[obs_key]):
        adata.obs[obs_key] = adata.obs[obs_key].astype("category")

    mapping = dict(zip(color_df[key_col].astype(str), color_df[color_col].astype(str)))

    if pd.api.types.is_categorical_dtype(adata.obs[obs_key]):
        cats = list(adata.obs[obs_key].cat.categories.astype(str))
    else:
        cats = list(pd.unique(adata.obs[obs_key].astype(str)))

    colors = [mapping.get(c, fill_missing) for c in cats]
    adata.uns[f"{obs_key}_colors"] = colors

    missing = [c for c in cats if c not in mapping]
    extra = [c for c in mapping.keys() if c not in set(cats)]
    return {"categories": cats, "missing_in_df": missing, "extra_in_df": extra}

def plot_rep_umap_pca(
    adata,
    rep_key: str = "X_scvi",
    color_key: str = "cell_type",
    color_df: pd.DataFrame | None = None,
    df_key_col: str = "cell_type",
    df_color_col: str = "color",
    neighbors_k: int = 15,
    metric: str = "euclidean",
    umap_min_dist: float = 0.3,
    pca_components: int = 2,
    do_umap: bool = True,
    do_pca: bool = True,
    show: bool = True,
    neighbors_key: str | None = None,
    umap_basis: str | None = None,

    figsize: tuple[float, float] = (10, 10),
    point_size: float = 60.0,
):
    if rep_key not in adata.obsm:
        raise KeyError(f"adata.obsm['{rep_key}']  .")

    diag = None
    if color_df is not None:
        diag = apply_obs_colors_from_df(
            adata,
            obs_key=color_key,
            color_df=color_df,
            key_col=df_key_col,
            color_col=df_color_col,
        )

    if neighbors_key is None:
        neighbors_key = f"neighbors_{rep_key}"
    if umap_basis is None:
        umap_basis = f"umap_{rep_key}"
    umap_obsm_key = f"X_{umap_basis}"

    if do_umap:
        sc.pp.neighbors(
            adata,
            use_rep=rep_key,
            n_neighbors=neighbors_k,
            metric=metric,
            key_added=neighbors_key,
        )
        sc.tl.umap(
            adata,
            min_dist=umap_min_dist,
            neighbors_key=neighbors_key,
            key_added=umap_obsm_key,
        )

        fig, ax = plt.subplots(figsize=figsize)
        sc.pl.embedding(
            adata,
            basis=umap_basis,
            color=[color_key],
            size=point_size,
            ax=ax,
            show=show,
        )

    if do_pca:
        X = adata.obsm[rep_key]
        pca = PCA(n_components=pca_components, random_state=0)
        adata.obsm[f"X_{rep_key}_pca{pca_components}"] = pca.fit_transform(X)

        fig, ax = plt.subplots(figsize=figsize)
        sc.pl.scatter(
            adata,
            basis=f"{rep_key}_pca{pca_components}",
            color=[color_key],
            size=point_size,
            ax=ax,
            show=show,
        )

    return diag

def _make_color_map(cell_type_color: pd.DataFrame,
                    ct_col="cell_type", color_col="color") -> dict:
    ctc = cell_type_color[[ct_col, color_col]].dropna().copy()
    ctc[ct_col] = ctc[ct_col].astype(str)
    ctc[color_col] = ctc[color_col].astype(str)
    return dict(zip(ctc[ct_col], ctc[color_col]))

def plot_celltype_metanode_graph(
    adata,
    cell_type_color: pd.DataFrame,
    excel_path: str,
    *,
 # link existing reference (e.g., adata_st)
    adata_type_ref=None,
    ref_ct_key="cell_type",
    use_ref_intersection=True,

 # csv columns
    ct1_col="cell_type1",
    ct2_col="cell_type2",

 # graph/keys
    graph_key="mf_graph",
    ct_key="cell_type",

    x_key="x",
    y_key="y",
    obsm_key=None,
    obsm_dims=(0, 1),

    default_color="#BDBDBD",
    max_type_edges=2000,
    line_alpha=0.35,
    line_width_min=0.3,
    line_width_max=3.0,
    node_size=160,
    drop_self=True,
    side_legend=True,
    legend_ncol=2,
    legend_fontsize=9,
    legend_marker_size=7,
    strip_types=True,
    verbose=True,
):
    mgraph = pd.read_csv(excel_path)

    if ct1_col not in mgraph.columns or ct2_col not in mgraph.columns:
        raise ValueError(
            f" '{ct1_col}', '{ct2_col}'  .  : {list(mgraph.columns)}"
        )

    t1 = mgraph[ct1_col].astype(str)
    t2 = mgraph[ct2_col].astype(str)
    if strip_types:
        t1 = t1.str.strip()
        t2 = t2.str.strip()

    allowed_types = pd.Index(pd.unique(pd.concat([t1, t2], ignore_index=True)))
    allowed_types = allowed_types[allowed_types.notna() & (allowed_types != "nan")]

    if len(allowed_types) == 0:
        raise ValueError("  cell type  .")

 # --- 1b) intersect with existing reference types ---
    if adata_type_ref is not None and use_ref_intersection:
        ref_types = pd.Index(adata_type_ref.obs[ref_ct_key].astype(str))
        if strip_types:
            ref_types = ref_types.str.strip()
        ref_types = ref_types.unique()
        allowed_types = allowed_types.intersection(ref_types)

        if len(allowed_types) == 0:
            raise ValueError(
                "/CSV  ref   0.\n"
                f"- ref_ct_key='{ref_ct_key}' \n"
                f"-  /(strip_types) "
            )

    if verbose:
        print(f"[INFO] allowed_types count = {len(allowed_types)}")

    if (x_key in adata.obs.columns) and (y_key in adata.obs.columns):
        xy = np.c_[adata.obs[x_key].to_numpy(dtype=float),
                   adata.obs[y_key].to_numpy(dtype=float)]
    else:
        if obsm_key is None:
            for cand in ["X_umap", "spatial", "X_tsne", "X_pca"]:
                if cand in adata.obsm.keys():
                    obsm_key = cand
                    break
        if obsm_key is None or obsm_key not in adata.obsm.keys():
            raise KeyError(
                f"adata.obs '{x_key}','{y_key}' , obsm_key  .\n"
                f"  obsm keys: {list(adata.obsm.keys())}\n"
                f": obsm_key='X_umap'  'spatial'  ."
            )
        X = np.asarray(adata.obsm[obsm_key])
        d0, d1 = obsm_dims
        if X.ndim != 2 or X.shape[1] <= max(d0, d1):
            raise ValueError(f"obsm['{obsm_key}'] shape={X.shape}  dims={obsm_dims}   .")
        xy = np.c_[X[:, d0].astype(float), X[:, d1].astype(float)]
        if verbose:
            print(f"[INFO] coords from obsm['{obsm_key}'] dims={obsm_dims}")

    ct_all = adata.obs[ct_key].astype(str)
    if strip_types:
        ct_all = ct_all.str.strip()
    ct_all = ct_all.to_numpy()

    keep = np.isin(ct_all, allowed_types.to_numpy())
    if keep.sum() == 0:
        raise ValueError(
            "   cell type .\n"
            f"- ct_key='{ct_key}' \n"
            f"-   adata   "
        )

    xy = xy[keep]
    ct_sub = ct_all[keep]

    ct = pd.Categorical(ct_sub) # observed only
    types = list(ct.categories)
    n = len(ct)
    k = len(types)

    if verbose:
        print(f"[INFO] kept cells = {n}, kept types = {k}")

    color_map = _make_color_map(cell_type_color)

    G = adata.obsp[graph_key]
    if not sp.isspmatrix(G):
        G = sp.csr_matrix(G)
    G = G.tocsr()

    idx = np.where(keep)[0]
    Gs = G[idx][:, idx].tocsr()

    onehot = sp.csr_matrix(
        (np.ones(n, dtype=np.float32), (np.arange(n), ct.codes)),
        shape=(n, k)
    )
    A = (onehot.T @ Gs @ onehot).toarray().astype(np.float64)
    if drop_self:
        np.fill_diagonal(A, 0.0)

 # --- 8) centroid ---
    type_xy = np.zeros((k, 2), dtype=float)
    for i in range(k):
        m = (ct.codes == i)
        type_xy[i, 0] = xy[m, 0].mean()
        type_xy[i, 1] = xy[m, 1].mean()

 # --- 9) edges ---
    src, dst = np.nonzero(A)
    w = A[src, dst]
    if w.size == 0:
        raise ValueError("-    0. (   sparse  )")

    if w.size > max_type_edges:
        idx2 = np.argpartition(-w, max_type_edges - 1)[:max_type_edges]
        src, dst, w = src[idx2], dst[idx2], w[idx2]

    w_min, w_max = float(w.min()), float(w.max())
    if w_max == w_min:
        widths = np.full_like(w, (line_width_min + line_width_max) / 2.0, dtype=float)
    else:
        widths = line_width_min + (w - w_min) * (line_width_max - line_width_min) / (w_max - w_min)

    segments = np.stack([type_xy[src], type_xy[dst]], axis=1)

 # --- 10) plot ---
    fig, ax = plt.subplots(figsize=(16, 6))
    lc = LineCollection(segments, linewidths=widths, alpha=line_alpha)
    ax.add_collection(lc)

    ax.grid(False)

    for i, t in enumerate(types):
        ax.scatter(
            type_xy[i, 0], type_xy[i, 1],
            s=node_size,
            c=color_map.get(str(t), default_color),
            edgecolors="black",
            linewidths=0.5,
            zorder=3
        )

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(bottom=False, left=False)

    ax.set_xlabel("")
    ax.set_ylabel("")

 #ax.set_xlabel(x_key if (x_key in adata.obs.columns) else "dim1")
 #ax.set_ylabel(y_key if (y_key in adata.obs.columns) else "dim2")
 #ax.set_title("Cell-type metanode graph (filtered by file cell types" +
 #             (", intersect ref)" if (adata_type_ref is not None and use_ref_intersection) else ")"))

 # legend
    if side_legend:
        handles, labels = [], []
        for t in types:
            col = color_map.get(str(t), default_color)
            h = ax.scatter([], [], s=legend_marker_size**2, c=col)
            handles.append(h)
            labels.append(str(t))

        ax.legend(
            handles, labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=legend_fontsize,
            ncol=legend_ncol,
            scatterpoints=1,
            handletextpad=0.6,
        )
        plt.tight_layout(rect=[0, 0, 0.80, 1])
    else:
        plt.tight_layout()

    A_df = pd.DataFrame(A, index=types, columns=types)
    return fig, ax, A_df, mgraph

def plot_gene_wise_corr_stage_specific_boxplot(
    df_ct_stage: pd.DataFrame,
    df_all_wide: pd.DataFrame | None = None,
    *,
    # df_ct_stage (long) columns
    stage_col="stage",
    cell_type_col="cell_type",
    value_col="corr",
    gene_col="gene",
    # df_all_wide (wide) columns
    col_stage1="corr_stage1",
    col_stage2="corr_stage2",
    col_stage3="corr_fgw",
    # stage ordering / labels
    stage_order=("stage1_1to1", "stage2_pairwise", "stage3_global"),
    stage_labels=("corr_stage1", "corr_stage2", "corr_fgw"),
    title="Gene-wise Correlation per Cell Type (Stage-specific grouping) + all_spots",
    figsize=(20, 7),
    include_all=True,
    all_labels=("all", "all_spots"),
    min_genes_per_ct=5,
    order_by="median_stage3",  # "name" | "median_stage1" | "median_stage2" | "median_stage3"
    rotate_xticks=45,
    show=True,
):

    # -----------------
    # 0) validate df_ct_stage (long)
    # -----------------
    req_long = {cell_type_col, stage_col, value_col}
    miss_long = req_long - set(df_ct_stage.columns)
    if miss_long:
        raise KeyError(f"df_ct_stage missing: {miss_long}")

    df = df_ct_stage.copy()

    # -----------------
    # 1) optionally append df_all_wide as all_spots (convert wide -> long)
    # -----------------
    if df_all_wide is not None:
        need_wide = {cell_type_col, gene_col, col_stage1, col_stage2, col_stage3}
        miss_wide = need_wide - set(df_all_wide.columns)
        if miss_wide:
            raise KeyError(f"df_all_wide missing: {miss_wide}")

        df_all_long = pd.melt(
            df_all_wide,
            id_vars=[cell_type_col, gene_col],
            value_vars=[col_stage1, col_stage2, col_stage3],
            var_name="_var",
            value_name=value_col,
        )
        var_to_stage = {col_stage1: stage_order[0], col_stage2: stage_order[1], col_stage3: stage_order[2]}
        df_all_long[stage_col] = df_all_long["_var"].map(var_to_stage)
        df_all_long = df_all_long.drop(columns=["_var"])

        # 맞추기 (df_ct_stage에 gene_col이 없을 수도 있으니)
        if gene_col not in df.columns and gene_col in df_all_long.columns:
            df[gene_col] = np.nan

        # concat
        common_cols = sorted(set(df.columns).union(df_all_long.columns))
        df = pd.concat(
            [df.reindex(columns=common_cols), df_all_long.reindex(columns=common_cols)],
            ignore_index=True,
        )

    # -----------------
    # 2) filtering / cleaning
    # -----------------
    if not include_all:
        df = df[~df[cell_type_col].isin(all_labels)].copy()

    df = df[df[stage_col].isin(stage_order)].copy()
    df = df[np.isfinite(df[value_col])].copy()

    # min genes per ct
    if min_genes_per_ct is not None and min_genes_per_ct > 1:
        if gene_col in df.columns:
            ct_counts = df.groupby(cell_type_col)[gene_col].nunique()
        else:
            ct_counts = df.groupby(cell_type_col).size()
        keep = ct_counts[ct_counts >= min_genes_per_ct].index
        df = df[df[cell_type_col].isin(keep)].copy()

    # stage label
    stage_map = dict(zip(stage_order, stage_labels))
    df["variable"] = df[stage_col].map(stage_map).fillna(df[stage_col].astype(str))

    # -----------------
    # 3) cell-type ordering
    # -----------------
    if order_by == "name":
        order = sorted(df[cell_type_col].unique(), key=lambda x: str(x))
    else:
        med = (
            df.pivot_table(index=cell_type_col, columns=stage_col, values=value_col, aggfunc="median")
            .reindex(columns=list(stage_order))
        )
        tmp = pd.DataFrame({
            "s1": med[stage_order[0]] if stage_order[0] in med.columns else np.nan,
            "s2": med[stage_order[1]] if stage_order[1] in med.columns else np.nan,
            "s3": med[stage_order[2]] if stage_order[2] in med.columns else np.nan,
        })
        if order_by == "median_stage1":
            order = tmp.sort_values("s1", ascending=False).index.tolist()
        elif order_by == "median_stage2":
            order = tmp.sort_values("s2", ascending=False).index.tolist()
        elif order_by == "median_stage3":
            order = tmp.sort_values("s3", ascending=False).index.tolist()
        else:
            raise ValueError(f"order_by: {order_by}")

    if include_all:
        tail = [ct for ct in order if ct in all_labels]
        head = [ct for ct in order if ct not in all_labels]
        order = head + tail

    # -----------------
    # 4) plot
    # -----------------
    plt.figure(figsize=figsize)
    ax = sns.boxplot(
        data=df,
        x=cell_type_col,
        y=value_col,
        hue="variable",
        order=order,
        hue_order=[stage_map.get(s, s) for s in stage_order],
    )
    ax.grid(False)
    ax.legend(
            title=None,
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            ncol=1,
        )
    
    ax.set_title(title)
    ax.set_xlabel("Cell type")
    ax.set_ylabel("Pearson correlation")
    plt.xticks(rotation=rotate_xticks, ha="right")
    plt.tight_layout()

    if show:
        plt.show()

    return ax.figure, ax, df



def _palette_from_cell_type_color(cell_type_color: pd.DataFrame):
    """
    cell_type_color: columns ['cell_type','color']  dict
    """
    if cell_type_color is None:
        return None
    if not isinstance(cell_type_color, pd.DataFrame):
        raise TypeError("cell_type_color must be a pandas.DataFrame")
    if ("cell_type" not in cell_type_color.columns) or ("color" not in cell_type_color.columns):
        raise ValueError("cell_type_color must have columns ['cell_type','color']")
    return dict(zip(cell_type_color["cell_type"].astype(str), cell_type_color["color"].astype(str)))

def plot_umap_st_types_and_anchors(
    adata_st,
    cell_type_color: pd.DataFrame,
    *,
    st_type_key: str,
    idx_anchor: np.ndarray,
    anchors_by_type: dict,
    anchor_type_key: str = "final_anchor_type",
    global_anchor_label: str = "global_anchor",
    include_global_anchor_label: bool = False,
    keys_to_plot=None,
    fig_size=(10, 10),
    dpi=120,
    umap_kwargs_common=None,
    debug=True,
):
    """
    - adata_st.obs anchor_type_key  /
    - UMAP anchor_type_key 1
      (anchor   NA   )
    """

    if st_type_key not in adata_st.obs:
        raise KeyError(f"missing adata_st.obs['{st_type_key}']")
    if "X_umap" not in adata_st.obsm:
        raise KeyError("missing adata_st.obsm['X_umap'] (UMAP    )")

    sc.set_figure_params(figsize=fig_size, dpi=dpi)

    if umap_kwargs_common is None:
        umap_kwargs_common = dict(size=40, alpha=0.85, legend_loc="right margin", frameon=False)

 # --- palette dict
    pal_map = _palette_from_cell_type_color(cell_type_color)

    anc = pd.Series(pd.NA, index=adata_st.obs_names, dtype="object")

    for ct, idxs in anchors_by_type.items():
        idxs = np.asarray(idxs, dtype=int)
        anc.iloc[idxs] = str(ct)

    idx_anchor = np.asarray(idx_anchor, dtype=int)
    not_filled = anc.iloc[idx_anchor].isna().to_numpy()
    if np.any(not_filled):
        missing_idxs = idx_anchor[not_filled]
        if include_global_anchor_label:
            anc.iloc[missing_idxs] = str(global_anchor_label)
        else:
            anc.iloc[missing_idxs] = adata_st.obs[st_type_key].astype(str).iloc[missing_idxs].to_numpy()

    adata_st.obs[anchor_type_key] = anc

    adata_st.obs[anchor_type_key] = adata_st.obs[anchor_type_key].astype("category")

    def _palette_list_for(col):
        if pal_map is None:
            return None
        if not pd.api.types.is_categorical_dtype(adata_st.obs[col]):
            return None
        cats = list(adata_st.obs[col].cat.categories)
        return [pal_map.get(str(c), "#BDBDBD") for c in cats]

 # =========================
 # =========================
    k = anchor_type_key

    if k not in adata_st.obs:
        if debug:
            print(f"[plot_umap_st_types_and_anchors] skip '{k}' (not in adata_st.obs)")
        return adata_st

    palette = _palette_list_for(k)
    sc.pl.umap(
        adata_st,
        color=k,
        palette=palette,
        **umap_kwargs_common,
    )

    return adata_st

def plot_CV_test_vs_all_gene_boxplots(
    gene_df: pd.DataFrame,
    df_all: pd.DataFrame | None = None,
    *,
    gene_col="gene",
    fold_col="fold",
    fold_corr_col="cor",
    all_stage1_col="corr_stage1",
    all_stage2_col="corr_stage2",
    all_fgw_col="corr_fgw",
    labels=("cv_test", "all_stage1", "all_stage2", "all_fgw"),
    title="Gene-wise Correlation: Full-Gene Training vs CV Test Gene set",
    figsize=(10, 5),
    rotate_xticks=30,
    show=True,
):
    need = {gene_col, fold_col, fold_corr_col}
    miss = need - set(gene_df.columns)
    if miss:
        raise KeyError(f"gene_df missing columns: {miss}")

    df_cv = gene_df[[gene_col, fold_corr_col]].copy()
    df_cv = df_cv.rename(columns={fold_corr_col: "value"})
    df_cv["group"] = labels[0]

    df_all_long = None
    if df_all is not None:
        if gene_col not in df_all.columns:
            raise KeyError(f"df_all missing column: {gene_col}")

        stage_cols, stage_groups = [], []
        if all_stage1_col in df_all.columns:
            stage_cols.append(all_stage1_col); stage_groups.append(labels[1])
        if all_stage2_col in df_all.columns:
            stage_cols.append(all_stage2_col); stage_groups.append(labels[2])
        if all_fgw_col in df_all.columns:
            stage_cols.append(all_fgw_col); stage_groups.append(labels[3])

        if all_stage2_col not in df_all.columns:
            raise KeyError(f"df_all must contain '{all_stage2_col}'")

        common_genes = set(df_cv[gene_col]).intersection(set(df_all[gene_col]))
        df_cv = df_cv[df_cv[gene_col].isin(common_genes)].copy()
        df_all_sub = df_all[df_all[gene_col].isin(common_genes)].copy()

        df_all_long = pd.melt(
            df_all_sub,
            id_vars=[gene_col],
            value_vars=stage_cols,
            var_name="_stage",
            value_name="value",
        )
        stage_map = {c: g for c, g in zip(stage_cols, stage_groups)}
        df_all_long["group"] = df_all_long["_stage"].map(stage_map).fillna(df_all_long["_stage"])
        df_all_long = df_all_long[[gene_col, "group", "value"]]

    if df_all_long is not None:
        df_plot = pd.concat([df_cv[[gene_col, "group", "value"]], df_all_long], ignore_index=True)
        order = [labels[0]] + [g for g in [labels[1], labels[2], labels[3]] if g in df_plot["group"].unique()]
    else:
        df_plot = df_cv[[gene_col, "group", "value"]].copy()
        order = [labels[0]]

    df_plot = df_plot[np.isfinite(df_plot["value"])].copy()

    plt.figure(figsize=figsize)
    ax = sns.boxplot(data=df_plot, x="group", y="value", order=order)
    ax.grid(False)
    plt.grid(False)
    ax.set_xlabel("")
    ax.set_ylabel("Correlation")
    ax.set_title(title)
    plt.xticks(rotation=rotate_xticks, ha="right")
    plt.tight_layout()

    if show:
        plt.show()

    return ax.figure, ax, df_plot

def plot_fold_summary_mean_median_boxplot(
    fold_df: pd.DataFrame,
    *,
    fold_col: str = "fold",
    mean_col: str = "cor_mean",
    median_col: str = "cor_median",
    title: str = "Fold summary: mean vs median",
    figsize: tuple = (6, 4),
    rotate_xticks: int = 0,
    show: bool = True,
):
    """
    Boxplot of fold-level summary metrics (mean and median) across folds.

    Parameters
    ----------
    fold_df : pd.DataFrame
        Must contain columns [fold_col, mean_col, median_col].
    fold_col : str
        Fold identifier column name.
    mean_col : str
        Column name for per-fold mean correlation.
    median_col : str
        Column name for per-fold median correlation.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    rotate_xticks : int
        Rotation angle for x tick labels.
    show : bool
        If True, calls plt.show().

    Returns
    -------
    fig, ax, df_long
        Matplotlib figure/axes and the melted long-form DataFrame used for plotting.
    """
    required = {fold_col, mean_col, median_col}
    missing = required - set(fold_df.columns)
    if missing:
        raise KeyError(f"fold_df missing columns: {missing}")

    df_long = fold_df.melt(
        id_vars=[fold_col],
        value_vars=[mean_col, median_col],
        var_name="metric",
        value_name="value",
    )

    fig = plt.figure(figsize=figsize)
    ax = sns.boxplot(data=df_long, x="metric", y="value")

    # Disable grid lines (both seaborn and matplotlib)
    ax.grid(False)
    plt.grid(False)

    ax.set_xlabel("")
    ax.set_ylabel("Correlation")
    ax.set_title(title)

    if rotate_xticks:
        plt.xticks(rotation=rotate_xticks, ha="right")

    plt.tight_layout()

    if show:
        plt.show()

    return fig, ax, df_long

def plot_phi_umap(
    *,
    adata_sc,
    adata_st,
    idx_anchor,
    cell_type_color,
    sc_type_key="cell_type",
    st_type_key="final_pairwise_type",
    sc_geom_key="X_scvi",
    st_geom_key="X_scvi",
    score_key="final_pairwise_score",
    weights_key_in_uns="final_pairwise_weights",
    direction="sc_to_st",
    min_n=30,
    shrink_k=1,
    clip_q=(0.01, 0.99),
    st_view="all",
    method="auto",
    train_score_percentile=0.0,
):
    """
    One-shot helper: build phi (before/after), compute phi for all ST points, then call plot_phi_umap_compare.

    Parameters
    ----------
    adata_sc, adata_st : AnnData
    idx_anchor : array-like of int
        Anchor indices (ST indices if direction uses ST anchors; use whatever your alignment module expects).
    cell_type_color : pd.DataFrame or mapping
        Passed through to plotting.
    train_score_percentile : float
        Percentile threshold for selecting ST training indices based on `score_key`.
        0.0 means "strictly greater than min" (usually equivalent to excluding NaNs/very low values only).

    Returns
    -------
    out : dict
        Output dictionary from util_v2.alignment.build_phi_before_after, plus phi_st_all and st_types_all.
    """
    from .alignment import build_phi_before_after, compute_phi_st, _to_np_str

    # Select ST training indices by score threshold
    scores = adata_st.obs[score_key].to_numpy()
    thr = np.percentile(scores[np.isfinite(scores)], train_score_percentile) if np.isfinite(scores).any() else np.nan
    idx_st_train_mask = np.isfinite(scores) & (scores > thr)
    idx_st_train = np.flatnonzero(idx_st_train_mask)

    # Build phi before/after alignment on the training subset
    out = build_phi_before_after(
        adata_sc=adata_sc,
        adata_st=adata_st,
        sc_type_key=sc_type_key,
        st_type_key=st_type_key,
        sc_geom_key=sc_geom_key,
        st_geom_key=st_geom_key,
        idx_anchor=idx_anchor,
        idx_st_train=idx_st_train,
        weights_A=adata_st.uns[weights_key_in_uns],
        direction=direction,
        min_n=min_n,
        shrink_k=shrink_k,
        clip_q=clip_q,
    )

    # Compute phi for all ST points (for "all" view)
    Z_st = np.asarray(adata_st.obsm[st_geom_key])
    idx_all_st = np.arange(Z_st.shape[0], dtype=int)

    phi_st_all = compute_phi_st(
        st_geom_all=Z_st,
        idx_query=idx_all_st,
        idx_anchor=idx_anchor,
    )

    st_types_all = _to_np_str(adata_st.obs[st_type_key].to_numpy())

    # Plot
    fig = plot_phi_umap_compare(
        phi_st_train=out["phi_st_train"],
        phi_sc_raw_train=out["phi_sc_raw_train"],
        phi_sc_aligned_train=out["phi_sc_aligned_train"],
        st_types_train=out["st_types_train"],
        sc_types_train=out["sc_types_train"],
        phi_st_aligned_train=out.get("phi_st_aligned_train", None),
        direction=direction,
        st_view=st_view,
        phi_st_all=phi_st_all,
        st_types_all=st_types_all,
        cell_type_color=cell_type_color,
        method=method,
    )

    # Attach extra outputs for convenience
    out["idx_st_train"] = idx_st_train
    out["phi_st_all"] = phi_st_all
    out["st_types_all"] = st_types_all
    out["fig"] = fig
    return out


# =====================================================================
# ST spatial kNN pair workflow figures
# =====================================================================

def plot_pair_weight_hist(
    pairs_df: pd.DataFrame,
    *,
    threshold: float,
    bins="auto",
    figsize=(7, 4),
    title="ST type-pair weight distribution",
    xlabel="weight",
    ylabel="#pairs",
    show: bool = True,
):
    """Plot a histogram of pair weights with a red vertical threshold line.

    The x-axis is weight and the y-axis is the number of pairs in each bin.
    """
    if not isinstance(pairs_df, pd.DataFrame):
        raise TypeError("pairs_df must be a pandas DataFrame")
    if "weight" not in pairs_df.columns:
        raise KeyError("pairs_df must contain column 'weight'")

    w = pd.to_numeric(pairs_df["weight"], errors="coerce").dropna().to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(w, bins=bins)
    ax.axvline(float(threshold), color="red", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if show:
        plt.show()
    return fig, ax


def plot_typepair_graph_spatial(
    pairs_df: pd.DataFrame,
    centroid_df: pd.DataFrame,
    cell_type_color: dict,
    *,
    ct1_col="ct1",
    ct2_col="ct2",
    weight_col="weight",
    figsize=(14, 5),
    node_size=140,
    node_edgewidth=0.8,
    edge_color="#4C8DFF",  
    edge_alpha=0.25,
    edge_width_min=0.3,
    edge_width_max=2.5,
    show_labels=False,
    legend=True,
    legend_ncol=2,
    legend_fontsize=10,
    legend_title=None,
):
    # --- validate ---
    
    if isinstance(cell_type_color, pd.DataFrame):
        if "cell_type" not in cell_type_color.columns or "color" not in cell_type_color.columns:
            raise KeyError("cell_type_color DF must have columns ['cell_type','color']")
        color_map = dict(
            zip(
                cell_type_color["cell_type"].astype(str).str.strip(),
                cell_type_color["color"].astype(str).str.strip()
            )
        )
        
    elif isinstance(cell_type_color, dict):
        color_map = {str(k).strip(): str(v).strip() for k, v in cell_type_color.items()}
    else:
        raise TypeError("cell_type_color must be dict or DataFrame with columns ['cell_type','color']")
        
    for c in (ct1_col, ct2_col, weight_col):
        if c not in pairs_df.columns:
            raise KeyError(f"pairs_df must contain '{c}'")
    for c in ("x", "y"):
        if c not in centroid_df.columns:
            raise KeyError("centroid_df must contain columns 'x','y'")

    nodes = [str(x).strip() for x in centroid_df.index.astype(str)]
    pos_xy = centroid_df.loc[nodes, ["x", "y"]].to_numpy(float)
    pos = {ct: (pos_xy[i, 0], pos_xy[i, 1]) for i, ct in enumerate(nodes)}
    node_set = set(nodes)

    # --- edges ---
    e = pairs_df[[ct1_col, ct2_col, weight_col]].copy()
    e[ct1_col] = e[ct1_col].astype(str)
    e[ct2_col] = e[ct2_col].astype(str)
    e[weight_col] = pd.to_numeric(e[weight_col], errors="coerce")
    e = e.dropna(subset=[weight_col])
    e = e[e[ct1_col].isin(node_set) & e[ct2_col].isin(node_set)]

    w = e[weight_col].to_numpy(float)
    if len(w) > 0:
        wmin, wmax = float(w.min()), float(w.max())
        if wmax == wmin:
            lw = np.full_like(w, (edge_width_min + edge_width_max) / 2)
        else:
            lw = edge_width_min + (w - wmin) / (wmax - wmin) * (edge_width_max - edge_width_min)
    else:
        lw = np.array([])

    segments = []
    for a, b in e[[ct1_col, ct2_col]].itertuples(index=False, name=None):
        xa, ya = pos[a]
        xb, yb = pos[b]
        segments.append([(xa, ya), (xb, yb)])

    # --- figure/axes ---
    fig, ax = plt.subplots(figsize=figsize)


    fig.subplots_adjust(right=0.72)

    # draw edges
    if segments:
        lc = LineCollection(
            segments,
            linewidths=lw,
            colors=edge_color,
            alpha=edge_alpha,
            zorder=1,
        )
        ax.add_collection(lc)

    # draw nodes
    node_colors = [color_map.get(ct, "#BDBDBD") for ct in nodes]
    ax.scatter(
        pos_xy[:, 0],
        pos_xy[:, 1],
        s=node_size,
        c=node_colors,
        edgecolors='none',
        linewidths=0,
        zorder=2,
    )

    # optional labels
    if show_labels:
        for ct in nodes:
            x, y = pos[ct]
            ax.text(x, y, ct, fontsize=9, ha="center", va="center", zorder=3)


    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="datalim")

    # legend on the right
    if legend:
        handles = [
            Line2D([0], [0], marker="o", linestyle="",
                   markerfacecolor=color_map.get(ct, "#BDBDBD"),
                   markeredgecolor='none',
                   markeredgewidth=3,
                   markersize=7,
                   label=ct)
            for ct in nodes
        ]
        ax.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            ncol=legend_ncol,
            fontsize=legend_fontsize,
            title=legend_title,
        )

    return fig, ax