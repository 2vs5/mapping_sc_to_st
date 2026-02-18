"""
Precomputation for mixture scoring (used in pairwise FGW refinement).

We keep this separate from util.global_map.prepare_global_precomp because:
- pairwise refinement reuses the same "cosine(exp) + Spearman(FC)" score
- but needs efficient scoring of MANY *mixture* vectors against their own ST spots

Core idea
---------
We keep the *same score definition* as global mapping:
  score = 0.5 * cosine(exp) + 0.5 * corr(FC)

where corr is selectable per-run:
  - "spearman": uses rank-centered FC
  - "pearson": uses mean-centered FC
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

def _score_vec_to_one_st(exp_vec, st_idx, precomp, *, cos_threshold=None, corr_method=None):
    """
    Score a single expression vector against ONE ST spot.
    Uses the same definition as global mapping:
      score = 0.5*cosine(exp) + 0.5*corr(FC)
    """
    exp_vec = np.asarray(exp_vec, dtype=float)
    G = precomp["G"]
    eps = precomp["eps"]

 # cosine
    st_normed = precomp["X_st_exp_normed"][st_idx]
    nrm = np.linalg.norm(exp_vec)
    if (not np.isfinite(nrm)) or (nrm <= eps):
        cos = 0.0
    else:
        cos = float(np.dot(st_normed, exp_vec / nrm))

    if cos_threshold is not None and cos < float(cos_threshold):
        return 0.5 * cos

    corr_method = str(corr_method or precomp.get("corr_method", "spearman")).lower().strip()
    if corr_method not in ("spearman", "pearson"):
        raise ValueError("corr_method must be 'spearman' | 'pearson'")

    fc = exp_vec - precomp["baseline_sc"]

    if corr_method == "spearman":
        r = rankdata(fc)
        sc_center = r - r.mean()
    else:
        sc_center = fc - fc.mean()

    std_sc = sc_center.std(ddof=1)
    if (not np.isfinite(std_sc)) or (std_sc <= eps):
        corr = 0.0
    else:
        if corr_method == "spearman":
            st_center = precomp["X_st_rank_center"][st_idx]
            st_std = precomp["X_st_rank_std"][st_idx]
        else:
 # pearson requires precomputed ST FC center/std
            st_center = precomp["X_st_fc_center"][st_idx]
            st_std = precomp["X_st_fc_std"][st_idx]

        numer = float(np.dot(st_center, sc_center))
        denom = float((G - 1) * std_sc * st_std)
        corr = 0.0 if denom == 0 or (not np.isfinite(denom)) else numer / denom
        if not np.isfinite(corr):
            corr = 0.0

    return 0.5 * cos + 0.5 * float(corr)

def mixture_scores_from_weights(
    W, # (n_sc_pair, n_st_pair) column-normalized
    idx_sc_pair, # global sc indices (n_sc_pair,)
    idx_st_pair, # global st indices (n_st_pair,)
    precomp,
    *,
    corr_method=None, # None -> use precomp['corr_method']
    gene_mask=None, # (G,) bool mask for scoring genes (optional)
    cos_threshold=None,
    n_jobs=1, # kept for API compatibility
    use_vectorized=True,
    strict_subset_spearman=True, # if True, compute ST stats on the subset panel (recommended)
):
    """
    Score each ST spot j in corridor using mixture expression:
      x_hat_j = sum_i W_{i,j} * x_sc_i

    Score = 0.5*cosine(expr) + 0.5*corr(FC)

    - cosine: compare L2-normalized ST expression vs L2-normalized mixture expression
    - corr(FC):
        - spearman: correlation between centered ranks of FC vectors
        - pearson: correlation between mean-centered FC vectors

    If gene_mask is provided and strict_subset_spearman=True:
      - The ST FC statistics are computed *exactly* on the subset panel (recommended).
        This avoids shape issues and is mathematically consistent for both spearman/pearson.

    Requirements in precomp:
      - genes_union, X_sc_exp, X_st_exp_normed, baseline_sc, eps, G
      - If strict_subset_spearman=True or gene_mask is not None:
          precomp["X_st_exp"] and precomp["baseline_st"] must exist.
    """
    idx_sc_pair = np.asarray(idx_sc_pair, dtype=int)
    idx_st_pair = np.asarray(idx_st_pair, dtype=int)

    if not use_vectorized:
        raise NotImplementedError("use_vectorized=False not supported in this implementation.")

    eps = float(precomp.get("eps", 1e-12))

 # ---- 0) Build mixture on FULL panel (stable) ----
    X_sc_full = precomp["X_sc_exp"][idx_sc_pair] # (n_sc_pair, G)
    X_hat_full = (W.T @ X_sc_full) # (n_st_pair, G)

 # ---- 1) Apply subset for scoring (optional) ----
    if gene_mask is not None:
        gene_mask = np.asarray(gene_mask, dtype=bool)
        G = int(len(precomp["genes_union"]))
        if gene_mask.ndim != 1 or gene_mask.shape[0] != G:
            raise ValueError(f"gene_mask must be 1D bool of length G={G}. Got {gene_mask.shape}.")
        G_eff = int(gene_mask.sum())
        if G_eff < 3:
            return np.zeros(len(idx_st_pair), dtype=float), X_hat_full[:, gene_mask]

        X_hat = X_hat_full[:, gene_mask] # (n_st_pair, G_eff)
        st_normed_sub = precomp["X_st_exp_normed"][idx_st_pair][:, gene_mask] # (n_st_pair, G_eff)
        baseline_sc = precomp["baseline_sc"][gene_mask] # (G_eff,)

 # for strict ST ranks on subset, we need X_st_exp and baseline_st
        if strict_subset_spearman:
            if "X_st_exp" not in precomp or "baseline_st" not in precomp:
                raise KeyError("strict_subset_spearman=True requires precomp['X_st_exp'] and precomp['baseline_st'] "
                               "(create precomp with include_X_st_exp=True).")
            X_st_exp_sub_all = precomp["X_st_exp"][idx_st_pair][:, gene_mask] # (n_st_pair, G_eff)
            baseline_st_sub = precomp["baseline_st"][gene_mask] # (G_eff,)
        else:
            st_center_all = precomp["X_st_rank_center"][idx_st_pair][:, gene_mask]
            st_std_all = precomp["X_st_rank_std"][idx_st_pair] # (n_st_pair,)

    else:
        G_eff = int(precomp["G"])
        X_hat = X_hat_full # (n_st_pair, G)
        st_normed_sub = precomp["X_st_exp_normed"][idx_st_pair] # (n_st_pair, G)
        baseline_sc = precomp["baseline_sc"] # (G,)

        if strict_subset_spearman:
            if "X_st_exp" not in precomp or "baseline_st" not in precomp:
                raise KeyError("strict_subset_spearman=True requires precomp['X_st_exp'] and precomp['baseline_st'] "
                               "(create precomp with include_X_st_exp=True).")
            X_st_exp_sub_all = precomp["X_st_exp"][idx_st_pair] # (n_st_pair, G)
            baseline_st_sub = precomp["baseline_st"] # (G,)
        else:
            st_center_all = precomp["X_st_rank_center"][idx_st_pair]
            st_std_all = precomp["X_st_rank_std"][idx_st_pair] # (n_st_pair,)

    n_st_pair = X_hat.shape[0]

 # ---- 2) cosine on subset panel ----
    norms = np.linalg.norm(X_hat, axis=1)
    ok = (np.isfinite(norms)) & (norms > eps)

    X_hat_normed = np.zeros_like(X_hat, dtype=np.float32)
    X_hat_normed[ok] = (X_hat[ok] / norms[ok, None]).astype(np.float32, copy=False)

    cos = np.sum(st_normed_sub * X_hat_normed, axis=1, dtype=np.float64)
    cos = np.where(np.isfinite(cos), cos, 0.0).astype(np.float64, copy=False)

    if cos_threshold is not None:
        need = cos >= float(cos_threshold)
    else:
        need = np.ones(n_st_pair, dtype=bool)

    scores = 0.5 * cos

    corr_method = str(corr_method or precomp.get("corr_method", "spearman")).lower().strip()
    if corr_method not in ("spearman", "pearson"):
        raise ValueError("corr_method must be 'spearman' | 'pearson'")

 # ---- 3) corr on FC ----
    if np.any(need):
 # mixture FC (subset)
        fc_sc = (X_hat[need] - baseline_sc[None, :]) # (n_need, G_eff)

        if corr_method == "spearman":
 # rank each row (mixture)
            try:
                r_sc = rankdata(fc_sc, axis=1).astype(np.float64, copy=False)
            except TypeError:
                r_sc = np.vstack([rankdata(row) for row in fc_sc]).astype(np.float64, copy=False)
            sc_center = r_sc - r_sc.mean(axis=1, keepdims=True)
        else:
            sc_center = fc_sc - fc_sc.mean(axis=1, keepdims=True)

        std_sc = sc_center.std(axis=1, ddof=1)
        std_sc = np.where((std_sc > eps) & np.isfinite(std_sc), std_sc, 1.0)

        if strict_subset_spearman:
 # strict: compute ST stats on subset panel for the chosen corr_method
            st_fc = (X_st_exp_sub_all[need] - baseline_st_sub[None, :]) # (n_need, G_eff)
            if corr_method == "spearman":
                try:
                    r_st = rankdata(st_fc, axis=1).astype(np.float64, copy=False)
                except TypeError:
                    r_st = np.vstack([rankdata(row) for row in st_fc]).astype(np.float64, copy=False)
                st_center = r_st - r_st.mean(axis=1, keepdims=True)
            else:
                st_center = st_fc - st_fc.mean(axis=1, keepdims=True)

            std_st = st_center.std(axis=1, ddof=1)
            std_st = np.where((std_st > eps) & np.isfinite(std_st), std_st, 1.0)

            numer = np.sum(st_center * sc_center, axis=1, dtype=np.float64)
            denom = (G_eff - 1) * std_sc * std_st

        else:
 # approx: slice precomputed ST center/std
            if corr_method == "spearman":
                st_center_sub = st_center_all[need].astype(np.float64, copy=False)
                std_st = np.asarray(st_std_all[need], dtype=np.float64)
            else:
                st_center_sub = precomp["X_st_fc_center"][idx_st_pair][need].astype(np.float64, copy=False)
                std_st = np.asarray(precomp["X_st_fc_std"][idx_st_pair][need], dtype=np.float64)
            numer = np.sum(st_center_sub * sc_center, axis=1, dtype=np.float64)
            denom = (G_eff - 1) * std_sc * std_st

        corr = np.zeros_like(numer, dtype=np.float64)
        good = (denom != 0) & np.isfinite(denom)
        corr[good] = numer[good] / denom[good]
        corr = np.where(np.isfinite(corr), corr, 0.0)

        scores[need] = 0.5 * cos[need] + 0.5 * corr

    return scores.astype(float, copy=False), X_hat

# ============================================================
# Gene-subset precomp (for bleeding/incoming removal)
# ============================================================

def build_precomp_subset(precomp: dict, genes_subset, *, include_X_st_exp=True):
    """
    Build a gene-subset precomp dict from a 'full' precomp.

    Requirements
    ------------
    The full precomp should include:
      - genes_union (list/array of gene names aligned to X_* matrices)
      - X_sc_exp (n_sc, G)
      - X_st_exp (n_st, G)  (required if include_X_st_exp=True)
      - baseline_sc (G,)
      - baseline_st (G,) if present

    This function recomputes ST rank-centered FC statistics on the subset so that
    Spearman(mixt, st_j) remains correct under gene removal.

    Returns
    -------
    sub_precomp : dict with the same core keys but restricted to genes_subset.
    """
    genes_union = np.asarray(precomp["genes_union"], dtype=object)
    genes_subset = np.asarray(list(genes_subset), dtype=object)
    gene_mask = np.isin(genes_union, genes_subset)
    if gene_mask.sum() < 1:
        raise ValueError("genes_subset has zero overlap with precomp['genes_union'].")

    sub = {}
    sub["genes_union"] = genes_union[gene_mask].copy()
    sub["G"] = int(gene_mask.sum())

 # slice expression
    sub["X_sc_exp"] = np.asarray(precomp["X_sc_exp"], dtype=float)[:, gene_mask]
    if include_X_st_exp:
        if "X_st_exp" not in precomp:
            raise KeyError("precomp missing X_st_exp; create full precomp with include_X_st_exp=True.")
        sub["X_st_exp"] = np.asarray(precomp["X_st_exp"], dtype=float)[:, gene_mask]

 # baselines (if present)
    if "baseline_sc" in precomp:
        sub["baseline_sc"] = np.asarray(precomp["baseline_sc"], dtype=float)[gene_mask]
    if "baseline_st" in precomp:
        sub["baseline_st"] = np.asarray(precomp["baseline_st"], dtype=float)[gene_mask]

 # normalize ST exp vectors for cosine term (same as full precomp logic)
    if "X_st_exp_normed" in precomp:
        sub["X_st_exp_normed"] = np.asarray(precomp["X_st_exp_normed"], dtype=float)[:, gene_mask]
    elif include_X_st_exp:
        X = sub["X_st_exp"]
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        sub["X_st_exp_normed"] = X / nrm

 # recompute ST FC ranks on subset
    if include_X_st_exp and ("baseline_st" in sub):
        X_fc = sub["X_st_exp"] - sub["baseline_st"][None, :]
    else:
 # fallback: if baseline_st missing, rank raw exp
        X_fc = sub["X_st_exp"] if include_X_st_exp else np.asarray(precomp["X_st_exp"], dtype=float)[:, gene_mask]

 # rank transform per ST spot
    X_rank = np.vstack([rankdata(row) for row in X_fc]) # (n_st, G_sub)
    X_center = X_rank - X_rank.mean(axis=1, keepdims=True)
    X_std = X_center.std(axis=1)
    X_std[X_std == 0] = 1.0
    sub["X_st_rank_center"] = X_center
    sub["X_st_rank_std"] = X_std

    return sub

class PrecompSubsetCache:
    """
    Lightweight cache for gene-subset precomp dicts.

    Key: tuple(sorted(genes_subset)) is used; for large subsets consider hashing upstream.
    """
    def __init__(self, max_items=64):
        self.max_items = int(max_items)
        self._cache = {}

    def get(self, precomp_full: dict, genes_subset, *, include_X_st_exp=True):
        key = tuple(sorted(map(str, genes_subset)))
        if key in self._cache:
            return self._cache[key]
        sub = build_precomp_subset(precomp_full, genes_subset, include_X_st_exp=include_X_st_exp)
        if len(self._cache) >= self.max_items:
 # drop an arbitrary item (simple)
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = sub
        return sub
