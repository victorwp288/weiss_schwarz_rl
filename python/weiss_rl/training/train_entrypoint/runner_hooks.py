"""Runner class binding helpers for the training entrypoint compatibility facade."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

EntrypointApiProvider = Callable[[], Any]


def bind_periodic_dev_eval_runner(
    api_provider: EntrypointApiProvider,
    base_cls: type[Any],
) -> type[Any]:
    class _PeriodicDevEvalRunner(base_cls):
        def __init__(
            self,
            *,
            stack: Any,
            model: Any,
            opponent_policy_id: str,
            observation_dim: int,
            action_dim: int,
            pass_action_id: int,
            artifact_dir: Path,
            focal_policy_id: str,
            require_sorted_legal_ids: bool,
            opponent_model: Any | None = None,
            heuristic_policy: Any | None = None,
        ) -> None:
            super().__init__(
                stack=stack,
                model=model,
                opponent_policy_id=opponent_policy_id,
                observation_dim=observation_dim,
                action_dim=action_dim,
                pass_action_id=pass_action_id,
                artifact_dir=artifact_dir,
                focal_policy_id=focal_policy_id,
                require_sorted_legal_ids=require_sorted_legal_ids,
                build_eval_env=api_provider()._build_ids_eval_env,
                opponent_model=opponent_model,
                heuristic_policy=heuristic_policy,
            )

    return _PeriodicDevEvalRunner


def bind_promotion_gate_runner(
    api_provider: EntrypointApiProvider,
    base_cls: type[Any],
    *,
    random_legal_policy_id: str,
) -> type[Any]:
    class _PromotionGateRunner(base_cls):
        def __init__(
            self,
            *,
            stack: Any,
            focal_policy_id: str,
            focal_model: Any,
            anchor_models: dict[str, Any],
            heuristic_policies: dict[str, Any],
            observation_dim: int,
            action_dim: int,
            pass_action_id: int,
            artifact_dir: Path,
            require_sorted_legal_ids: bool,
        ) -> None:
            super().__init__(
                stack=stack,
                focal_policy_id=focal_policy_id,
                focal_model=focal_model,
                anchor_models=anchor_models,
                heuristic_policies=heuristic_policies,
                observation_dim=observation_dim,
                action_dim=action_dim,
                pass_action_id=pass_action_id,
                artifact_dir=artifact_dir,
                require_sorted_legal_ids=require_sorted_legal_ids,
                build_eval_env=api_provider()._build_ids_eval_env,
                random_legal_policy_id=random_legal_policy_id,
            )

    return _PromotionGateRunner
