"""Checkpoint-related wrapper composition for the training entrypoint facade."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from typing import Any

from weiss_rl.training.train_entrypoint_checkpoint_io_wrappers import install_checkpoint_io_wrappers
from weiss_rl.training.train_entrypoint_learner_wrappers import install_learner_wrappers
from weiss_rl.training.train_entrypoint_snapshot_wrappers import install_snapshot_wrappers


def install_checkpoint_wrappers(
    namespace: MutableMapping[str, Any],
    *,
    entrypoint_api: Callable[[], Any],
) -> None:
    install_checkpoint_io_wrappers(namespace, entrypoint_api=entrypoint_api)
    install_learner_wrappers(namespace, entrypoint_api=entrypoint_api)
    install_snapshot_wrappers(namespace, entrypoint_api=entrypoint_api)


__all__ = ["install_checkpoint_wrappers"]
