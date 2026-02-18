# util/bleeding_utils.py
"""
Directional bleeding + incoming union utilities.

Updated concept (FC-collapse; selected pairs supported)
------------------------------------------------------
We compare within-dataset logFC for directed cell-type pairs (c_src -> c_tgt):

  logFC_sc(g; src,tgt) = log((mu_sc[src,g] + eps) / (mu_sc[tgt,g] + eps))
  logFC_st(g; src,tgt) = log((mu_st[src,g] + eps) / (mu_st[tgt,g] + eps))

SC-marker-like mask:
  Marker_sc(src->tgt,g) = (logFC_sc > tau_sc_pos) AND (g in DEG(src))

Bleeding gene (src->tgt) definition (FC-collapse):
  - g is marker-like in SC for src vs tgt (same mask)
  - but ST contrast is weak (near-zero) within ST:
        |logFC_st| < tau_st_flat        (default; use_abs_st=True)
    (option) directional-only:
        logFC_st < tau_st_flat          (use_abs_st=False)

Pair restriction:
  - If pairs_selected is provided, we evaluate only those directed pairs (src,tgt).
  - Otherwise we evaluate all directed pairs permutations(ct_valid, 2).

Incoming union (target-side)
----------------------------
Default incoming is derived from bleeding collapse itself (layer-robust):
  Incoming(src->tgt) := bleeding_pairs(src->tgt)
  Incoming_union(tgt) = union_{src!=tgt} Incoming(src->tgt)

Legacy cross-layer rule is still available via incoming_mode="cross_layer":
  B_in(tgt,g) = log((mu_st[tgt,g]+eps)/(mu_sc[tgt,g]+eps)) > tau_in

Outputs are dictionaries of sets and per-(src,tgt) arrays.
No OT/FGW here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import permutations

from .array_ops import to_dense as _to_dense

# ============================================================
# 0) basic helpers
# ============================================================

def to_dense(X):
    """Backwards-compatible wrapper (prefer util.array_ops.to_dense)."""
    return _to_dense(X)

def _deg_union_genes(selected_genes, genes_use=None):
    """
    selected_genes: dict[ct -> DataFrame with gene names in index]
    """
    gset = set()
    for _, df in selected_genes.items():
        if df is None:
            continue
        gset.update(list(df.index))
    if genes_use is not None:
        gset &= set(genes_use)
    return gset

def _valid_celltypes_from_sc_st(
    adata_sc,
    adata_st,
    *,
    groupby_sc="cell_type",
    st_type_key="global_sc_type",
    st_sim_key="global_sc_sim",
    st_sim_min=0.3,
    min_cells_sc=20,
    min_cells_st=20,
):
    sc_ct = np.asarray(adata_sc.obs[groupby_sc]).astype(object)

    st_ct = np.asarray(adata_st.obs[st_type_key]).astype(object)
    st_sim = np.asarray(adata_st.obs[st_sim_key], dtype=float)
    st_high = st_sim >= float(st_sim_min)

 # candidates
    ct_sc_all = pd.unique(pd.Series(sc_ct)[pd.notna(sc_ct)]).tolist()
    ct_st_all = pd.unique(pd.Series(st_ct)[st_high & pd.notna(st_ct)]).tolist()

 # filter by counts
    ct_valid_sc = []
    for ct in ct_sc_all:
        if int(np.sum(sc_ct == ct)) >= int(min_cells_sc):
            ct_valid_sc.append(str(ct))

    ct_valid_st = []
    for ct in ct_st_all:
        if int(np.sum((st_ct == ct) & st_high)) >= int(min_cells_st):
            ct_valid_st.append(str(ct))

    ct_valid = sorted(set(ct_valid_sc) & set(ct_valid_st))
    return ct_valid, st_high

def _normalize_pairs_selected(pairs_selected, ct_valid, *, bidirectional=True):
    """
    Normalize and filter user-provided pairs.

    Parameters
    ----------
    pairs_selected:
        iterable of pairs. Can be undirected (orderless) pairs.
    ct_valid:
        valid cell types list
    bidirectional:
        if True, expand each undirected pair {a,b} into (a,b) and (b,a)

    Returns
    -------
    list[(src,tgt)] as strings, deduplicated, src!=tgt, both in ct_valid.
    """
    if pairs_selected is None:
        return None

    ct_valid_set = set(map(str, ct_valid))
    out = set()

    for a, b in pairs_selected:
        x = str(a)
        y = str(b)
        if x == y:
            continue
        if (x not in ct_valid_set) or (y not in ct_valid_set):
            continue

        out.add((x, y))
        if bidirectional:
            out.add((y, x))

 # stable order (optional)
    return sorted(out)

# ============================================================
# 1) directional bleeding (core)  [FC-collapse + selected pairs]
# ============================================================

def compute_directional_bleeding(
    adata_sc,
    adata_st,
    *,
    selected_genes,
    pairs_selected=None,
    groupby_sc="cell_type",
    st_type_key="global_sc_type",
    st_sim_key="global_sc_sim",
    st_sim_min=0.3,
    genes_use=None,
    min_cells_sc=20,
    min_cells_st=20,
    eps=1e-6,
    tau_sc_pos=1.0,
    tau_st_flat=0.2,
    use_abs_st=True,
):
    """
    Compute directional logFC, SC-marker mask, and bleeding gene sets (FC-collapse).

    Pair restriction:
      - If pairs_selected is None: evaluate all permutations(ct_valid, 2)
      - Else: evaluate only provided (src,tgt) pairs (after filtering to ct_valid)

    Bleeding rule (src->tgt):
      marker_mask = (logFC_sc > tau_sc_pos) & (g in DEG(src))
      bleed_mask  = marker_mask & (|logFC_st| < tau_st_flat)   if use_abs_st
                 or marker_mask & (logFC_st < tau_st_flat)     if not use_abs_st

    Returns dict with:
      - genes: list[str] (global gene panel used)
      - celltypes_valid: list[str]
      - pairs_used: list[(src,tgt)] or None (if full permutations)
      - mu_sc: dict[ct -> (G,)]
      - mu_st: dict[ct -> (G,)]
      - logFC_sc: dict[(src,tgt) -> (G,)]
      - logFC_st: dict[(src,tgt) -> (G,)]
      - is_sc_marker: dict[(src,tgt) -> (G,) bool]
      - bleeding_pairs: dict[(src,tgt) -> set[str]]
      - params: dict
    """
    if selected_genes is None:
        raise ValueError("[compute_directional_bleeding] selected_genes required.")

 # valid types and ST high-sim mask
    ct_valid, st_high = _valid_celltypes_from_sc_st(
        adata_sc, adata_st,
        groupby_sc=groupby_sc,
        st_type_key=st_type_key,
        st_sim_key=st_sim_key,
        st_sim_min=st_sim_min,
        min_cells_sc=min_cells_sc,
        min_cells_st=min_cells_st,
    )
    if len(ct_valid) < 2:
        raise ValueError("[compute_directional_bleeding] Need >=2 valid cell types.")

 # normalize/filter selected pairs
    pairs_used = _normalize_pairs_selected(pairs_selected, ct_valid, bidirectional=True)
    if pairs_selected is not None and (pairs_used is None or len(pairs_used) == 0):
        raise ValueError(
            "[compute_directional_bleeding] pairs_selected was provided but no pairs remained "
            "after filtering to valid cell types."
        )

 # gene panel
    deg_union = _deg_union_genes(selected_genes, genes_use=genes_use)
    genes = [
        g for g in deg_union
        if (g in adata_sc.var_names) and (g in adata_st.var_names)
    ]
    if len(genes) == 0:
        raise ValueError("[compute_directional_bleeding] No overlapping genes after filters.")
    genes = list(map(str, genes))

 # expression
    X_sc = to_dense(adata_sc[:, genes].X).astype(float, copy=False)
    X_st = to_dense(adata_st[:, genes].X).astype(float, copy=False)

    sc_ct = np.asarray(adata_sc.obs[groupby_sc]).astype(object)
    st_ct = np.asarray(adata_st.obs[st_type_key]).astype(object)

 # means
    mu_sc = {}
    for ct in ct_valid:
        m = (sc_ct == ct)
        mu_sc[ct] = X_sc[m].mean(axis=0)

    mu_st = {}
    for ct in ct_valid:
        m = (st_ct == ct) & st_high
        mu_st[ct] = X_st[m].mean(axis=0)

 # choose which pairs to evaluate
    if pairs_used is None:
        pair_iter = permutations(ct_valid, 2)
    else:
        pair_iter = pairs_used

 # per directed pair
    logFC_sc = {}
    logFC_st = {}
    is_sc_marker = {}
    bleeding_pairs = {}

    for c_src, c_tgt in pair_iter:
        lsc = np.log((mu_sc[c_src] + eps) / (mu_sc[c_tgt] + eps)) # (G,)
        lst = np.log((mu_st[c_src] + eps) / (mu_st[c_tgt] + eps)) # (G,)

        logFC_sc[(c_src, c_tgt)] = lsc
        logFC_st[(c_src, c_tgt)] = lst

        deg_src = set(selected_genes.get(c_src, pd.DataFrame()).index.tolist())
        mask_deg_src = np.array([g in deg_src for g in genes], dtype=bool)

        marker_mask = (lsc > float(tau_sc_pos)) & mask_deg_src
        is_sc_marker[(c_src, c_tgt)] = marker_mask

        if use_abs_st:
            flat_mask = np.abs(lst) < float(tau_st_flat)
        else:
            flat_mask = lst < float(tau_st_flat)

        bleed_mask = marker_mask & flat_mask
        bleeding_pairs[(c_src, c_tgt)] = (
            {genes[i] for i in np.where(bleed_mask)[0]}
            if np.any(bleed_mask)
            else set()
        )

    return {
        "genes": genes,
        "celltypes_valid": ct_valid,
        "pairs_used": pairs_used, # None means "full permutations"
        "mu_sc": mu_sc,
        "mu_st": mu_st,
        "logFC_sc": logFC_sc,
        "logFC_st": logFC_st,
        "is_sc_marker": is_sc_marker,
        "bleeding_pairs": bleeding_pairs,
        "params": dict(
            method="fc_collapse",
            st_sim_min=float(st_sim_min),
            tau_sc_pos=float(tau_sc_pos),
            tau_st_flat=float(tau_st_flat),
            use_abs_st=bool(use_abs_st),
            eps=float(eps),
            min_cells_sc=int(min_cells_sc),
            min_cells_st=int(min_cells_st),
            pairs_selected_provided=bool(pairs_selected is not None),
        ),
    }

# ============================================================
# 2) incoming union (target-side)
# ============================================================

def compute_incoming_union(
    bleeding_obj,
    *,
    incoming_mode="from_collapse",
    tau_in=0.5,
    eps=1e-6,
):
    """
    Compute incoming genes union per target type.

    incoming_mode
    -------------
    - "from_collapse" (default; layer-robust):
        Incoming(src->tgt) := bleeding_pairs(src->tgt)
        incoming_union(tgt) = union_{src!=tgt} bleeding_pairs(src->tgt)
      (No cross-layer mu_st/mu_sc direct comparison.)

    - "cross_layer" (legacy; your previous rule):
        B_in(tgt,g) = log((mu_st[tgt,g]+eps)/(mu_sc[tgt,g]+eps))
        Incoming(src->tgt) = marker_mask(src->tgt) & (B_in(tgt) > tau_in)

    Returns dict:
      - incoming_pairs: dict[(src,tgt)->set[str]]
      - incoming_per_target: dict[tgt->set[str]]
      - (optional) B_in: dict[tgt->(G,)] when incoming_mode="cross_layer"
      - genes, celltypes_valid, params
    """
    genes = list(bleeding_obj["genes"])
    celltypes_valid = list(bleeding_obj["celltypes_valid"])

    incoming_pairs = {}
    incoming_per_target = {c: set() for c in celltypes_valid}

    if incoming_mode == "from_collapse":
        bleed_pairs = bleeding_obj.get("bleeding_pairs", {})
        for (c_src, c_tgt), gset in bleed_pairs.items():
            gset = set(gset) if gset is not None else set()
            incoming_pairs[(c_src, c_tgt)] = gset
            incoming_per_target[str(c_tgt)].update(gset)

        return {
            "incoming_pairs": incoming_pairs,
            "incoming_per_target": incoming_per_target,
            "genes": genes,
            "celltypes_valid": celltypes_valid,
            "params": dict(incoming_mode="from_collapse"),
        }

    if incoming_mode == "cross_layer":
        mu_sc = bleeding_obj["mu_sc"]
        mu_st = bleeding_obj["mu_st"]
        is_sc_marker = bleeding_obj["is_sc_marker"]

 # target-only incoming score
        B_in = {}
        for c_tgt in celltypes_valid:
            B_in[c_tgt] = np.log((np.asarray(mu_st[c_tgt]) + eps) / (np.asarray(mu_sc[c_tgt]) + eps))

        for (c_src, c_tgt), marker_mask in is_sc_marker.items():
            marker_mask = np.asarray(marker_mask, dtype=bool)
            b = B_in[c_tgt]
            incoming_mask = marker_mask & (b > float(tau_in))
            if np.any(incoming_mask):
                incoming = {genes[i] for i in np.where(incoming_mask)[0]}
                incoming_pairs[(c_src, c_tgt)] = incoming
                incoming_per_target[c_tgt].update(incoming)
            else:
                incoming_pairs[(c_src, c_tgt)] = set()

        return {
            "incoming_pairs": incoming_pairs,
            "incoming_per_target": incoming_per_target,
            "B_in": B_in,
            "genes": genes,
            "celltypes_valid": celltypes_valid,
            "params": dict(incoming_mode="cross_layer", tau_in=float(tau_in), eps=float(eps)),
        }

    raise ValueError("incoming_mode must be 'from_collapse' or 'cross_layer'.")

# ============================================================
# 3) clean DEG helper
# ============================================================

def clean_deg_for_target(selected_genes, incoming_union_obj, c_tgt: str):
    """
    Return DEG(target) - incoming_union(target).
    """
    if c_tgt not in selected_genes:
        return set()
    deg_genes = set(selected_genes[c_tgt].index.tolist())
    incoming = set(incoming_union_obj.get("incoming_per_target", {}).get(c_tgt, set()))
    return deg_genes - incoming

# ============================================================
# 4) QC summary (optional)
# ============================================================

def summarize_bleeding(bleeding_obj, incoming_union_obj=None, top_n=20):
    """
    Make a human-readable summary DataFrame:
      - per directed pair: #bleeding genes
      - (optional) per target: #incoming union genes

    Returns
    -------
    df_pairs : pd.DataFrame
    df_incoming : pd.DataFrame or None
    """
    rows = []
    for (src, tgt), genes in bleeding_obj["bleeding_pairs"].items():
        rows.append({"src": src, "tgt": tgt, "n_bleeding": int(len(genes))})
    df_pairs = pd.DataFrame(rows).sort_values(["n_bleeding", "src", "tgt"], ascending=[False, True, True])

    df_in = None
    if incoming_union_obj is not None:
        rows2 = []
        for tgt, gset in incoming_union_obj["incoming_per_target"].items():
            rows2.append({"tgt": tgt, "n_incoming_union": int(len(gset))})
        df_in = pd.DataFrame(rows2).sort_values("n_incoming_union", ascending=False)

 # optional truncate display
    if top_n is not None:
        df_pairs = df_pairs.head(int(top_n))
        if df_in is not None:
            df_in = df_in.head(int(top_n))

    return df_pairs, df_in

# ------------------------------------------------------------
# Public names expected by util.__init__
# ------------------------------------------------------------

def compute_bleeding_directional(*args, **kwargs):
    """Alias of compute_directional_bleeding."""
    return compute_directional_bleeding(*args, **kwargs)

def compute_incoming_bleeding(bleeding_obj, *args, **kwargs):
    """Alias of compute_incoming_union."""
    return compute_incoming_union(bleeding_obj, *args, **kwargs)

def get_clean_deg_genes_for_target(selected_genes, incoming_or_bleeding_obj, c_tgt: str, *, mode="incoming"):
    """Return a 'clean' DEG gene set for c_tgt.

    Parameters
    ----------
    mode:
      - 'incoming': DEG(c_tgt) - incoming_union(c_tgt)
      - 'bleeding': DEG(c_tgt) - union_{src!=tgt} bleeding(src->tgt)  (target-side incoming from bleeding)

    Notes
    -----
    In your current pipeline you typically use incoming-union subtraction.
    """
    if mode == "incoming":
        return clean_deg_for_target(selected_genes, incoming_or_bleeding_obj, c_tgt)

    if mode == "bleeding":
 # target-side union of bleeding genes that 'enter' the target
        bleed_pairs = incoming_or_bleeding_obj.get("bleeding_pairs", {})
        incoming = set()
        for (src, tgt), gset in bleed_pairs.items():
            if str(tgt) == str(c_tgt):
                incoming |= set(gset)
        if c_tgt not in selected_genes:
            return set()
        deg = set(selected_genes[c_tgt].index.tolist())
        return deg - incoming

    raise ValueError("mode must be 'incoming' or 'bleeding'.")
