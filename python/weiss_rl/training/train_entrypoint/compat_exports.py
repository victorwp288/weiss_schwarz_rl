"""Compatibility facade for training entrypoint export installation."""

from __future__ import annotations

# ruff: noqa: F401
from weiss_rl.training.train_entrypoint.namespace import (
    COMPAT_EXPORT_FAMILIES as _EXPORT_FAMILIES,
)
from weiss_rl.training.train_entrypoint.namespace import (
    install_train_entrypoint_compat_exports,
)

__all__ = ["_EXPORT_FAMILIES", "install_train_entrypoint_compat_exports"]
