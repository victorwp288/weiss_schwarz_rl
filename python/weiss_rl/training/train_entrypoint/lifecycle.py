"""Runtime lifecycle wrapper installation for the training entrypoint facade."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from typing import Any

from weiss_rl.training.train_entrypoint.dev_eval_wrappers import install_dev_eval_wrappers
from weiss_rl.training.train_entrypoint.lifecycle_checkpoint_wrappers import (
    install_best_checkpoint_wrappers,
    install_current_checkpoint_wrapper,
    install_promotion_wrapper,
)
from weiss_rl.training.train_entrypoint.lifecycle_training_wrapper import install_minimal_training_wrapper


def install_script_wrappers(
    namespace: MutableMapping[str, Any],
    *,
    entrypoint_api: Callable[[], Any],
) -> None:
    install_current_checkpoint_wrapper(namespace, entrypoint_api=entrypoint_api)
    install_dev_eval_wrappers(namespace, entrypoint_api=entrypoint_api)
    install_best_checkpoint_wrappers(namespace, entrypoint_api=entrypoint_api)
    install_promotion_wrapper(namespace, entrypoint_api=entrypoint_api)


__all__ = [
    "install_best_checkpoint_wrappers",
    "install_current_checkpoint_wrapper",
    "install_dev_eval_wrappers",
    "install_minimal_training_wrapper",
    "install_promotion_wrapper",
    "install_script_wrappers",
]
