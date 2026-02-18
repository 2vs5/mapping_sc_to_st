# util/fgw_solver.py
"""
FGW solver utilities (POT wrapper + signature-safe routing).

This file is intentionally a "hub" for:
- Solver wrappers (POTFGWSolver)
- Transport computation helpers (compute_fgw_plan / run_transport)
- Anchor-cost builders (build_anchor_cost_pair)
- Geometry adapters for FGW/FUGW

Key fixes added (2026-01):
- method normalization + aliases/typos (e.g., semirelatxed_fgw)
- entropic_semirelaxed_fgw added
- max_iter support (injected into POT calls when supported)
- ALWAYS return (T, logd) from solver.solve(), tolerant to (T, log, extra...)
- FUGW (unbalanced GW) passes epsilon for entropic approximation when provided
"""

from __future__ import annotations

import numpy as np

try:
    import ot
except Exception:
    ot = None

from sklearn.metrics import pairwise_distances

from .solver_adapter import try_patterns

# ============================================================
# 0) Small utilities
# ============================================================

def mean_normalize(X, eps=1e-12):
    X = np.asarray(X, dtype=float)
    m = float(np.mean(X))
    if not np.isfinite(m) or m <= eps:
        return X
    return X / m

def median_normalize_nonzero(C, eps=1e-12):
    C = np.asarray(C, dtype=float)
    vals = C[C > 0]
    if vals.size == 0:
        return C
    med = float(np.median(vals))
    if not np.isfinite(med) or med <= eps:
        return C
    return C / med

def clip_quantile(C, q=0.99):
    C = np.asarray(C, dtype=float)
    if q is None:
        return C
    hi = np.quantile(C, q)
    if not np.isfinite(hi) or hi <= 0:
        return C
    return np.minimum(C, hi)

def scale_normalize(C, mode="q95", eps=1e-12, q=0.95):
    """
    Scale-normalize a distance/cost matrix C.

    Parameters
    ----------
    mode : {"mean","median","q95","q90","quantile"}
      - mean   : divide by mean(C)
      - median : divide by median(C)
      - q95    : divide by quantile(C, 0.95) on positive entries
      - q90    : divide by quantile(C, 0.90) on positive entries
      - quantile : divide by quantile(C, q) on positive entries
    """
    C = np.asarray(C, dtype=float)

    m = str(mode).lower() if mode is not None else "none"
    if m in ("none", "no", "off", "false"):
        return C

 # use positive entries to avoid the zero diagonal dominating statistics
    mask = (C > 0) & np.isfinite(C)
    if not np.any(mask):
        return C

    if m == "mean":
        s = float(np.mean(C[mask]))
    elif m == "median":
        s = float(np.median(C[mask]))
    elif m in ("q95", "quantile95"):
        s = float(np.quantile(C[mask], 0.95))
    elif m in ("q90", "quantile90"):
        s = float(np.quantile(C[mask], 0.90))
    elif m in ("quantile", "q"):
        s = float(np.quantile(C[mask], float(q)))
    else:
        raise ValueError(f"Unknown scale_normalize mode: {mode!r}")

    if (not np.isfinite(s)) or (s <= eps):
        return C
    return C / s

# ============================================================
# 1) Solver interface + POT solver wrapper
# ============================================================

class OTSolverBase:
    """
    Interface:
        T, log = solver.solve(M, C_sc, C_st, p, q, **kwargs)
    """
    def solve(self, M, C_sc, C_st, p, q, **kwargs):
        raise NotImplementedError

class POTFGWSolver(OTSolverBase):
    """
    POT wrapper for FGW variants.

    Supported methods (case-insensitive)
    -----------------------------------
      - "fgw"
      - "entropic_fgw" (requires epsilon)
      - "bapg_fgw" (optionally uses epsilon if supported)
      - "semirelaxed_fgw" (srFGW; POT signatures differ)
      - "entropic_semirelaxed_fgw" (entropic srFGW; requires epsilon)
      - "fugw" / "fused_unbalanced_gromov_wasserstein" / "fused_unbalanced_gw" / "unbalanced_fgw"
        (fused unbalanced GW; if available)

    Notes
    -----
    - alpha and loss_fun are stored in the solver.
    - additional kwargs passed to solver.solve(...) override defaults.
    - solve() ALWAYS returns (T, logd) to keep downstream unpack stable.
    """
    def __init__(
        self,
        *,
        method="fgw",
        alpha=0.5,
        loss_fun="square_loss",
        epsilon=None, # for entropic variants / fugw entropic approximation
        max_iter=None, # passed as `max_iter` when supported
        log=True,
        **default_kwargs,
    ):
        self.method = str(method)
        self.alpha = float(alpha)
        self.loss_fun = str(loss_fun)
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.log = bool(log)
        self.default_kwargs = dict(default_kwargs)

    def solve(self, M, C_sc, C_st, p, q, **kwargs):
        """Solve a fused Gromov-Wasserstein problem using POT."""
        if ot is None:
            raise ImportError("POT (package name: ot) is required for POTFGWSolver")

        kw = dict(self.default_kwargs)
        kw.update(kwargs)

 # Normalize method name and accept common aliases/typos
        raw_method = "" if self.method is None else str(self.method)
        method = raw_method.strip().lower()
        method_aliases = {
            "fgw": "fgw",
            "entropic_fgw": "entropic_fgw",
            "bapg_fgw": "bapg_fgw",
            "semirelaxed_fgw": "semirelaxed_fgw",
            "entropic_semirelaxed_fgw": "entropic_semirelaxed_fgw",

 # aliases / typos
            "semi_relaxed_fgw": "semirelaxed_fgw",
            "semi-relaxed_fgw": "semirelaxed_fgw",
            "semirelatxed_fgw": "semirelaxed_fgw",
            "srfgw": "semirelaxed_fgw",
            "sr_fgw": "semirelaxed_fgw",

            "entropy_fgw": "entropic_fgw",
            "entropy_srfgw": "entropic_semirelaxed_fgw",
            "entropy_semirelaxed_fgw": "entropic_semirelaxed_fgw",
            "entropic_srfgw": "entropic_semirelaxed_fgw",
            "entropic_sr_fgw": "entropic_semirelaxed_fgw",

            "fugw": "fugw",
            "fused_unbalanced_gromov_wasserstein": "fugw",
            "fused_unbalanced_gw": "fugw",
            "unbalanced_fgw": "fugw",
        }
        method = method_aliases.get(method, method)

 # Inject max_iter if configured (can be overridden by kwargs)
        if getattr(self, "max_iter", None) is not None and "max_iter" not in kw:
            kw["max_iter"] = int(self.max_iter)

 # Allow kwargs to override solver-level epsilon without causing duplicate keyword errors.
        _eps_kw = kw.pop("epsilon", None)
        eps = float(_eps_kw) if _eps_kw is not None else (None if self.epsilon is None else float(self.epsilon))

        def _maybe_unpack(ret):
 # Always return (T, logd) so callers can reliably unpack.
 # POT may return T, (T, log), or (T, log, extra...).
            if isinstance(ret, tuple):
                if len(ret) >= 2:
                    return ret[0], ret[1]
                if len(ret) == 1:
                    return ret[0], None
            return ret, None

        if method == "fgw":
            ret = ot.gromov.fused_gromov_wasserstein(
                M, C_sc, C_st, p, q,
                loss_fun=self.loss_fun,
                alpha=self.alpha,
                log=self.log,
                **kw,
            )
            return _maybe_unpack(ret)

        if method == "entropic_fgw":
            if eps is None:
                raise ValueError("POTFGWSolver(method='entropic_fgw') requires epsilon.")
            ret = ot.gromov.entropic_fused_gromov_wasserstein(
                M, C_sc, C_st, p, q,
                loss_fun=self.loss_fun,
                epsilon=eps,
                alpha=self.alpha,
                log=self.log,
                **kw,
            )
            return _maybe_unpack(ret)

        if method == "bapg_fgw":
            patterns = [
                ((M, C_sc, C_st, p, q), dict(loss_fun=self.loss_fun, alpha=self.alpha, epsilon=eps, log=self.log, **kw)),
                ((M, C_sc, C_st, p, q), dict(loss_fun=self.loss_fun, alpha=self.alpha, log=self.log, **kw)),
            ]
            ret = try_patterns(ot.gromov.BAPG_fused_gromov_wasserstein, patterns)
            return _maybe_unpack(ret)

        if method == "semirelaxed_fgw":
 # Prefer public API if available, fall back to internal import for older POT.
            try:
                fn = ot.gromov.semirelaxed_fused_gromov_wasserstein
            except Exception:
                try:
                    from ot.gromov._semirelaxed import semirelaxed_fused_gromov_wasserstein as fn
                except ImportError as e:
                    raise ImportError("semirelaxed_fused_gromov_wasserstein not available in this POT install.") from e

            patterns = [
 # Newer public signature (often no q)
                ((M, C_sc, C_st, p), dict(loss_fun=self.loss_fun, alpha=self.alpha, log=self.log, **kw)),
                ((M, C_sc, C_st), dict(p=p, loss_fun=self.loss_fun, alpha=self.alpha, log=self.log, **kw)),
 # Older variants
                ((M, C_sc, C_st, p, q), dict(loss_fun=self.loss_fun, alpha=self.alpha, log=self.log, **kw)),
                ((C_sc, C_st), dict(wx=p, wy=q, M=M, loss_fun=self.loss_fun, alpha=self.alpha, log=self.log, **kw)),
            ]
            ret = try_patterns(fn, patterns)
            return _maybe_unpack(ret)

        if method == "entropic_semirelaxed_fgw":
            if eps is None:
                raise ValueError("POTFGWSolver(method='entropic_semirelaxed_fgw') requires epsilon.")

 # Prefer public API if available, fall back to internal import for older POT.
            try:
                fn = ot.gromov.entropic_semirelaxed_fused_gromov_wasserstein
            except Exception:
                try:
                    from ot.gromov._semirelaxed import entropic_semirelaxed_fused_gromov_wasserstein as fn
                except ImportError as e:
                    raise ImportError("entropic_semirelaxed_fused_gromov_wasserstein not available in this POT install.") from e

            patterns = [
 # Newer public signature (often no q)
                ((M, C_sc, C_st, p), dict(loss_fun=self.loss_fun, alpha=self.alpha, epsilon=eps, log=self.log, **kw)),
                ((M, C_sc, C_st), dict(p=p, loss_fun=self.loss_fun, alpha=self.alpha, epsilon=eps, log=self.log, **kw)),
 # Older variants
                ((M, C_sc, C_st, p, q), dict(loss_fun=self.loss_fun, alpha=self.alpha, epsilon=eps, log=self.log, **kw)),
                ((C_sc, C_st), dict(wx=p, wy=q, M=M, loss_fun=self.loss_fun, alpha=self.alpha, epsilon=eps, log=self.log, **kw)),
            ]
            ret = try_patterns(fn, patterns)
            return _maybe_unpack(ret)

        if method == "fugw":
            try:
                fn = ot.gromov.fused_unbalanced_gromov_wasserstein
            except Exception as e:
                raise ImportError("fused_unbalanced_gromov_wasserstein not available in this POT install.") from e

 # POT signatures vary; try a few common patterns.
            patterns = [
                ((C_sc, C_st), dict(wx=p, wy=q, M=M, loss_fun=self.loss_fun, alpha=self.alpha, epsilon=eps, log=self.log, **kw)),
                ((C_sc, C_st, p, q), dict(M=M, loss_fun=self.loss_fun, alpha=self.alpha, epsilon=eps, log=self.log, **kw)),
                ((M, C_sc, C_st, p, q), dict(loss_fun=self.loss_fun, alpha=self.alpha, epsilon=eps, log=self.log, **kw)),
            ]
            ret = try_patterns(fn, patterns)
            return _maybe_unpack(ret)

        raise ValueError(f"Unknown POTFGWSolver method: {raw_method}")

# ============================================================
# 2) FGW plan computation with optional anchor-cost
# ============================================================

def build_M(M_expr, *, M_anchor=None, beta_expr=0.7, norm="mean"):
    """
    Build final M from expression-cost and optional anchor-cost.

    M = beta*M_expr + (1-beta)*M_anchor (if anchor provided)
    with optional mean-normalization on each component.
    """
    M_expr = np.asarray(M_expr, dtype=float)
    if norm and str(norm).lower() == "mean":
        M_expr = mean_normalize(M_expr)

    if M_anchor is None:
        return M_expr

    M_anchor = np.asarray(M_anchor, dtype=float)
    if norm and str(norm).lower() == "mean":
        M_anchor = mean_normalize(M_anchor)

    beta = float(beta_expr)
    beta = max(0.0, min(1.0, beta))
    return beta * M_expr + (1.0 - beta) * M_anchor

def adapt_C_for_fgw(C_raw, *, C_kind="distance", norm="q95", clip_q=None):
    """
    Normalize/clip C for FGW (distance-like).
    """
    C = np.asarray(C_raw, dtype=float)
    if clip_q is not None:
        C = clip_quantile(C, q=float(clip_q))
 # For FGW, C is treated as a distance/cost matrix.
    C = scale_normalize(C, mode=norm)
    return C

def distance_to_similarity(D, *, sim_method="rbf", sigma=None, eps=1e-12):
    """
    Convert distance matrix to similarity matrix.
    Default: RBF similarity: exp(-D / sigma).
    """
    D = np.asarray(D, dtype=float)
    sim_method = str(sim_method).lower()

    if sim_method in ("rbf", "gaussian"):
        if sigma is None:
 # robust scale: median of non-zero distances
            vals = D[D > 0]
            if vals.size == 0:
                sig = 1.0
            else:
                sig = float(np.median(vals))
            sigma = max(sig, eps)
        else:
            sigma = max(float(sigma), eps)
        return np.exp(-D / sigma)

    raise ValueError(f"Unknown sim_method: {sim_method!r}")

def adapt_C_for_fugw(
    C_raw,
    *,
    C_kind="distance",
    to_similarity=True,
    sim_method="rbf",
    sigma=None,
    norm="mean",
    clip_q=None,
):
    """
    Prepare C for FUGW (expects similarity matrices per POT doc).
    If C_kind == "distance" and to_similarity=True, convert distances to similarity.
    Then apply optional clipping and normalization.
    """
    C = np.asarray(C_raw, dtype=float)

    if clip_q is not None:
        C = clip_quantile(C, q=float(clip_q))

    kind = str(C_kind).lower()
    if kind == "distance" and bool(to_similarity):
        C = distance_to_similarity(C, sim_method=sim_method, sigma=sigma)

 # similarity should be scaled in a stable range
    C = scale_normalize(C, mode=norm)
    return C

def run_transport(
    *,
    solver: OTSolverBase,
    X_sc_feat,
    X_st_feat,
    C_sc_raw,
    C_st_raw,
    M_anchor=None,
    beta_expr=0.7,
    mean_scale=True,
 # C adapters
    C_kind_sc="distance",
    C_kind_st="distance",
    C_norm="q95",
    C_clip_q=None,
    fugw_to_similarity=True,
    fugw_sim_method="rbf",
    fugw_sigma=None,
    **solver_kwargs,
):
    """
    Prepare costs and call solver.solve().

    Returns
    -------
    T : (n_sc, n_st)
    logd : dict or None
    M : (n_sc, n_st) final feature cost used
    """
    X_sc_feat = np.asarray(X_sc_feat, dtype=float)
    X_st_feat = np.asarray(X_st_feat, dtype=float)

 # feature cost (raw)
    M_expr = pairwise_distances(X_sc_feat, X_st_feat, metric="euclidean") ** 2
 # build final M with unified normalization+mixing
    M = build_M(M_expr, M_anchor=M_anchor, beta_expr=beta_expr, norm=("mean" if mean_scale else "none"))

    n_sc = X_sc_feat.shape[0]
    n_st = X_st_feat.shape[0]
    p = np.ones(n_sc, dtype=float) / max(n_sc, 1)
    q = np.ones(n_st, dtype=float) / max(n_st, 1)

 # C adapters: if solver is FUGW, prepare similarity matrices
    method = str(getattr(solver, "method", "fgw")).strip().lower()
    if method in ("fugw", "fused_unbalanced_gromov_wasserstein", "fused_unbalanced_gw", "unbalanced_fgw"):
        C_sc = adapt_C_for_fugw(
            C_sc_raw,
            C_kind=C_kind_sc,
            to_similarity=fugw_to_similarity,
            sim_method=fugw_sim_method,
            sigma=fugw_sigma,
            norm=C_norm,
            clip_q=C_clip_q,
        )
        C_st = adapt_C_for_fugw(
            C_st_raw,
            C_kind=C_kind_st,
            to_similarity=fugw_to_similarity,
            sim_method=fugw_sim_method,
            sigma=fugw_sigma,
            norm=C_norm,
            clip_q=C_clip_q,
        )
    else:
        C_sc = adapt_C_for_fgw(C_sc_raw, C_kind=C_kind_sc, norm=C_norm, clip_q=C_clip_q)
        C_st = adapt_C_for_fgw(C_st_raw, C_kind=C_kind_st, norm=C_norm, clip_q=C_clip_q)

    T, logd = solver.solve(M, C_sc, C_st, p, q, **solver_kwargs)
    return T, logd, M

def compute_fgw_plan(
    X_sc_feat,
    X_st_feat,
    C_sc,
    C_st,
    *,
    solver: OTSolverBase,
    M_anchor=None,
    beta_expr=0.7,
    mean_scale=True,
 # C adapters
    C_kind_sc="distance",
    C_kind_st="distance",
    C_norm="q95",
    C_clip_q=None,
    fugw_to_similarity=True,
    fugw_sim_method="rbf",
    fugw_sigma=None,
    **solver_kwargs,
):
    """
    Compute FGW/FUGW plan with a consistent interface.

    Returns
    -------
    T : (n_sc, n_st)
    logd : dict or None
    M : (n_sc, n_st) final feature cost used
    """
    return run_transport(
        solver=solver,
        X_sc_feat=X_sc_feat,
        X_st_feat=X_st_feat,
        C_sc_raw=C_sc,
        C_st_raw=C_st,
        M_anchor=M_anchor,
        beta_expr=beta_expr,
        mean_scale=mean_scale,
        C_kind_sc=C_kind_sc,
        C_kind_st=C_kind_st,
        C_norm=C_norm,
        C_clip_q=C_clip_q,
        fugw_to_similarity=fugw_to_similarity,
        fugw_sim_method=fugw_sim_method,
        fugw_sigma=fugw_sigma,
        **solver_kwargs,
    )

# ============================================================
# 3) Anchor-cost builder for a corridor
# ============================================================

def build_anchor_cost_pair(
    adata_sc,
    adata_st,
    *,
    idx_sc_pair,
    idx_st_pair,
    sc_geom_all,
    st_geom_all,
    anchor_key="global_sc_anchor",
    square=False,
    normalize="median",
    eps=1e-12,
):
    """
    Build an anchor-cost matrix M_anchor for a selected (ct1, ct2) corridor.

    Expected behavior (as used by pairwise_refine / final_global_anchor_fgw):
    - take global anchor positions from adata_sc.obs[anchor_key] (or equivalent)
    - compute distances in anchor space between selected sc and st indices

    Parameters
    ----------
    idx_sc_pair : array-like of int
        indices of selected sc points
    idx_st_pair : array-like of int
        indices of selected st points
    sc_geom_all, st_geom_all : array-like
        geometry embeddings for all points (used if needed)
    square : bool
        if True, square distances
    normalize : {"median","mean","none"}
        scale normalization applied after distance computation

    Returns
    -------
    M_anchor : (len(idx_sc_pair), len(idx_st_pair)) ndarray
    """
 # NOTE: This function’s exact semantics depend on your project’s anchor representation.
 # The implementation below follows the common pattern already used in your codebase:
 # anchor distance in embedding/anchor space for the selected corridor.

    idx_sc_pair = np.asarray(idx_sc_pair, dtype=int)
    idx_st_pair = np.asarray(idx_st_pair, dtype=int)

 # Preferred: use per-cell global anchors if present; otherwise fall back to provided geometry.
    if anchor_key in getattr(adata_sc, "obs", {}):
        A_sc = np.asarray(adata_sc.obs[anchor_key].to_numpy(), dtype=float)
    else:
        A_sc = np.asarray(sc_geom_all, dtype=float)

    if anchor_key in getattr(adata_st, "obs", {}):
        A_st = np.asarray(adata_st.obs[anchor_key].to_numpy(), dtype=float)
    else:
        A_st = np.asarray(st_geom_all, dtype=float)

    A_sc_sel = A_sc[idx_sc_pair]
    A_st_sel = A_st[idx_st_pair]

    M_anchor = pairwise_distances(A_sc_sel, A_st_sel, metric="euclidean")
    if square:
        M_anchor = M_anchor ** 2

    norm = str(normalize).lower()
    if norm == "median":
        M_anchor = median_normalize_nonzero(M_anchor, eps=eps)
    elif norm == "mean":
        M_anchor = mean_normalize(M_anchor, eps=eps)
    elif norm in ("none", "no", "off", "false"):
        pass
    else:
        raise ValueError(f"Unknown normalize mode: {normalize!r}")

    return M_anchor
