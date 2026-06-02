"""Runner class wrapper installation for the training entrypoint facade."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from typing import Any

from weiss_rl.training.train_entrypoint.runner_hooks import bind_periodic_dev_eval_runner, bind_promotion_gate_runner


def install_runner_wrappers(
    namespace: MutableMapping[str, Any],
    *,
    entrypoint_api: Callable[[], Any],
    periodic_dev_eval_runner_base: type[Any],
    promotion_gate_runner_base: type[Any],
    random_legal_policy_id: str,
) -> None:
    namespace["_PeriodicDevEvalRunner"] = bind_periodic_dev_eval_runner(entrypoint_api, periodic_dev_eval_runner_base)
    namespace["_PromotionGateRunner"] = bind_promotion_gate_runner(
        entrypoint_api,
        promotion_gate_runner_base,
        random_legal_policy_id=random_legal_policy_id,
    )
