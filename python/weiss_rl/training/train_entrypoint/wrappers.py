"""Compose private training entrypoint wrappers onto the facade namespace."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from typing import Any

from weiss_rl.training.train_entrypoint.checkpoints import install_checkpoint_wrappers
from weiss_rl.training.train_entrypoint.lifecycle import install_minimal_training_wrapper, install_script_wrappers
from weiss_rl.training.train_entrypoint.metadata_wrappers import install_metadata_wrappers
from weiss_rl.training.train_entrypoint.runner_wrappers import install_runner_wrappers


def install_train_entrypoint_wrappers(
    namespace: MutableMapping[str, Any],
    *,
    entrypoint_api: Callable[[], Any],
    periodic_dev_eval_runner_base: type[Any],
    promotion_gate_runner_base: type[Any],
    random_legal_policy_id: str,
) -> None:
    install_runner_wrappers(
        namespace,
        entrypoint_api=entrypoint_api,
        periodic_dev_eval_runner_base=periodic_dev_eval_runner_base,
        promotion_gate_runner_base=promotion_gate_runner_base,
        random_legal_policy_id=random_legal_policy_id,
    )
    install_metadata_wrappers(namespace, entrypoint_api=entrypoint_api)
    install_checkpoint_wrappers(namespace, entrypoint_api=entrypoint_api)
    install_script_wrappers(namespace, entrypoint_api=entrypoint_api)
    install_minimal_training_wrapper(namespace, entrypoint_api=entrypoint_api)
