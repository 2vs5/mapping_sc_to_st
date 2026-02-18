"""Utility package for scRNA-seq -> Spatial mapping pipeline.

Design goals
------------
- Keep `import util` lightweight (no heavy deps such as scanpy at import time).
- Expose keys eagerly (tiny, used everywhere).
- Provide lazy access to submodules via attribute access: `util.prep`, `util.pairwise_refine`, etc.
"""
from __future__ import annotations

import importlib
from typing import Any

# Keys (single source of truth)
from .keys import * # noqa: F401,F403

# Submodules that may be heavy; loaded lazily.
_LAZY = {
    "array_ops",
    "bleeding",
    "fgw_solver",
    "final_global_anchor_fgw",
    "global_map",
    "merge",
    "pairwise_refine",
    "precomp",
    "prep",
    "proximity_pairs",
    "unpaired_single_refine",
    "fig",
    "alignment",
    "anchors",
    "genes",
    "m_cost",
    "geometry",
    "solver_adapter",
    "c_build",
    "transport_engine",
    "gene_correlation",
    "run_pipeline",
}

def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")