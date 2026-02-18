# util/solver_adapter.py
"""
Signature-safe calling helpers for POT solvers.

POT changes signatures across versions (e.g., semirelaxed uses p,q vs wx,wy).
These helpers let us attempt multiple patterns while passing only supported kwargs.
"""
from __future__ import annotations
import inspect
from typing import Any, Callable, Dict, List, Tuple

def call_with_supported_kwargs(fn: Callable, *args, **kwargs):
    sig = inspect.signature(fn)
    supported = {}
    for k, v in kwargs.items():
        if v is None:
            continue
        if k in sig.parameters:
            supported[k] = v
    return fn(*args, **supported)

def try_patterns(fn: Callable, patterns: List[Tuple[Tuple[Any, ...], Dict[str, Any]]]):
    last_err = None
    for args, kwargs in patterns:
        try:
            return call_with_supported_kwargs(fn, *args, **kwargs)
        except TypeError as e:
            last_err = e
            continue
    if last_err is not None:
        raise last_err
    raise TypeError("No patterns provided")
