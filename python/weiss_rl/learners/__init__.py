"""Learner-side update logic."""

from __future__ import annotations

from importlib import import_module


def __getattr__(name: str):
    if name == "impala_learner":
        module = import_module(".impala", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
