"""Periodic dev-eval and checkpoint-guard orchestration for training."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import torch

from weiss_rl.config import StackConfig
from weiss_rl.core.simulator_contract import SimulatorContract
from weiss_rl.diagnostics.tensorboard_logger import TensorBoardLogger
from weiss_rl.training.checkpointing.periodic_dev_eval_confirmatory import (
    PeriodicDevEvalEffectiveSummary,
    checkpoint_tracker_best_record,
    maybe_run_confirmatory_dev_eval,
)
from weiss_rl.training.checkpointing.periodic_dev_eval_guard import (
    CheckpointGuardApplicationResult,
    apply_periodic_dev_eval_checkpoint_guard,
)


@dataclass(frozen=True, slots=True)
class TrainingPeriodicDevEvalHooks:
    should_run_periodic_dev_eval: Any
    run_periodic_dev_eval: Any
    slug_policy_id: Any
    load_checkpoint_tracker: Any
    confirmatory_dev_eval_request: Any
    periodic_dev_eval_schedule: Any
    expand_periodic_dev_eval_paired_seeds: Any
    ensure_current_checkpoint: Any
    publish_checkpoint_aliases: Any
    maybe_log_structured_mainmove_guard: Any
    maybe_rollback_to_best_checkpoint: Any


@dataclass(frozen=True, slots=True)
class PeriodicDevEvalGuardResult:
    last_dev_eval_summary: Mapping[str, Any] | None
    last_dev_eval_update_count: int | None
    last_checkpoint_guard_rollback_update: int | None
    stop_requested: bool


def maybe_run_periodic_dev_eval_and_checkpoint_guard(
    *,
    learner: Any,
    model: Any,
    stack: StackConfig,
    contract: SimulatorContract,
    artifacts: Any,
    training_paths: Any,
    runtime: Any,
    device: torch.device,
    spec_hash256: str,
    algorithm: Any,
    latest_metrics: dict[str, float],
    last_dev_eval_summary: Mapping[str, Any] | None,
    last_dev_eval_update_count: int | None,
    last_checkpoint_guard_rollback_update: int | None,
    run_id256: str,
    config_hash256: str,
    tensorboard_logger: TensorBoardLogger | None,
    hooks: TrainingPeriodicDevEvalHooks,
) -> PeriodicDevEvalGuardResult:
    update_count = int(learner.update_count)
    result = PeriodicDevEvalGuardResult(
        last_dev_eval_summary=last_dev_eval_summary,
        last_dev_eval_update_count=last_dev_eval_update_count,
        last_checkpoint_guard_rollback_update=last_checkpoint_guard_rollback_update,
        stop_requested=False,
    )
    if not hooks.should_run_periodic_dev_eval(stack, update_count=update_count):
        return result

    summary_payload = run_base_periodic_dev_eval(
        hooks=hooks,
        stack=stack,
        contract=contract,
        artifacts=artifacts,
        training_paths=training_paths,
        learner=learner,
        device=device,
        run_id256=run_id256,
        config_hash256=config_hash256,
        spec_hash256=spec_hash256,
    )
    effective = maybe_run_confirmatory_dev_eval(
        hooks=hooks,
        stack=stack,
        learner=learner,
        summary_payload=summary_payload,
        contract=contract,
        artifacts=artifacts,
        training_paths=training_paths,
        device=device,
        run_id256=run_id256,
        config_hash256=config_hash256,
        spec_hash256=spec_hash256,
        update_count=update_count,
    )
    guard_result = apply_periodic_dev_eval_checkpoint_guard(
        hooks=hooks,
        stack=stack,
        learner=learner,
        model=model,
        artifacts=artifacts,
        training_paths=training_paths,
        runtime=runtime,
        device=device,
        spec_hash256=spec_hash256,
        algorithm=algorithm,
        latest_metrics=latest_metrics,
        effective_summary=effective.summary,
        last_checkpoint_guard_rollback_update=last_checkpoint_guard_rollback_update,
        run_id256=run_id256,
        tensorboard_logger=tensorboard_logger,
        update_count=update_count,
    )

    return PeriodicDevEvalGuardResult(
        last_dev_eval_summary=effective.summary,
        last_dev_eval_update_count=update_count,
        last_checkpoint_guard_rollback_update=guard_result.next_rollback_update,
        stop_requested=guard_result.stop_requested,
    )


def run_base_periodic_dev_eval(
    *,
    hooks: TrainingPeriodicDevEvalHooks,
    stack: StackConfig,
    contract: SimulatorContract,
    artifacts: Any,
    training_paths: Any,
    learner: Any,
    device: torch.device,
    run_id256: str,
    config_hash256: str,
    spec_hash256: str,
) -> Mapping[str, Any]:
    summary_payload = hooks.run_periodic_dev_eval(
        stack=stack,
        contract=contract,
        artifacts=artifacts,
        training_paths=training_paths,
        learner=learner,
        device=device,
        run_id256=run_id256,
        config_hash256=config_hash256,
        spec_hash256=spec_hash256,
    )
    anchor_keys = sorted(cast(dict[str, Any], summary_payload["anchor_scores"]).keys())
    opponent_fragment = f" opponent={hooks.slug_policy_id(anchor_keys[0])}" if anchor_keys else ""
    print(
        "Periodic dev eval: "
        f"update={learner.update_count}{opponent_fragment} "
        f"aggregate={summary_payload['aggregate_score']:.4f} "
        f"anchors={','.join(anchor_keys)}"
    )
    return cast(Mapping[str, Any], summary_payload)


__all__ = [
    "CheckpointGuardApplicationResult",
    "PeriodicDevEvalEffectiveSummary",
    "PeriodicDevEvalGuardResult",
    "TrainingPeriodicDevEvalHooks",
    "apply_periodic_dev_eval_checkpoint_guard",
    "checkpoint_tracker_best_record",
    "maybe_run_periodic_dev_eval_and_checkpoint_guard",
    "maybe_run_confirmatory_dev_eval",
    "run_base_periodic_dev_eval",
]
