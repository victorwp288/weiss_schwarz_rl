"""Compatibility facade for checkpoint, learner, and snapshot wrapper installers."""

from __future__ import annotations

from weiss_rl.training.train_entrypoint.checkpoint_io_wrappers import install_checkpoint_io_wrappers
from weiss_rl.training.train_entrypoint.checkpoint_wrappers import install_checkpoint_wrappers
from weiss_rl.training.train_entrypoint.learner_wrappers import install_learner_wrappers
from weiss_rl.training.train_entrypoint.snapshot_wrappers import install_snapshot_wrappers

__all__ = [
    "install_checkpoint_io_wrappers",
    "install_checkpoint_wrappers",
    "install_learner_wrappers",
    "install_snapshot_wrappers",
]
