"""Compatibility facade for training entrypoint phase helpers."""

from __future__ import annotations

# ruff: noqa: F401,I001

from weiss_rl.training.train_entrypoint.cli_phase import (
    require_explicit_resume_geometry,
    resolve_train_cli_state,
)
from weiss_rl.training.train_entrypoint.execution_phase import execute_train_run
from weiss_rl.training.train_entrypoint.manifest_phase import prepare_train_manifest_state
from weiss_rl.training.train_entrypoint.startup_phase import prepare_train_startup_state
from weiss_rl.training.train_entrypoint.state import (
    TrainCliState,
    TrainManifestState,
    TrainStartupState,
)
