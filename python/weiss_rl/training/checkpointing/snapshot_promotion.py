"""Periodic checkpoint snapshot promotion phase for training."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import torch

from weiss_rl.config import StackConfig
from weiss_rl.core.simulator_contract import SimulatorContract
from weiss_rl.diagnostics.tensorboard_logger import TensorBoardLogger
from weiss_rl.training.checkpointing.io import checkpoint_path_for_update


class CheckpointPromotionLearner(Protocol):
    update_count: int
    model: Any

    def get_policy_version(self) -> int: ...


class CheckpointPromotionArtifacts(Protocol):
    run_dir: Path


class CheckpointPromotionPaths(Protocol):
    checkpoints_dir: Path


class CheckpointPromotionRuntime(Protocol):
    def refresh_opponent_pool(self) -> None: ...


@dataclass(frozen=True, slots=True)
class TrainingCheckpointPromotionHooks:
    write_checkpoint: Any
    publish_checkpoint_aliases: Any
    maybe_log_structured_mainmove_guard: Any
    persist_snapshot_registry_entry: Any
    run_snapshot_promotion_gate: Any


@dataclass(frozen=True, slots=True)
class CheckpointSnapshotWriteResult:
    checkpoint_path: Path
    tracker_payload: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class SnapshotRegistryCandidate:
    policy_id: str


@dataclass(frozen=True, slots=True)
class SnapshotPromotionGateResult:
    candidate_policy_id: str
    promotion_passed: bool


def league_reference_update_from_metrics(latest_metrics: Mapping[str, float]) -> int | None:
    if "league_effective_update" not in latest_metrics:
        return None
    return int(latest_metrics["league_effective_update"])


def maybe_checkpoint_and_promote_snapshot(
    *,
    learner: CheckpointPromotionLearner,
    stack: StackConfig,
    contract: SimulatorContract,
    artifacts: CheckpointPromotionArtifacts,
    training_paths: CheckpointPromotionPaths,
    runtime: CheckpointPromotionRuntime,
    device: torch.device,
    spec_hash256: str,
    algorithm: Any,
    latest_metrics: dict[str, float],
    last_dev_eval_summary: Mapping[str, Any] | None,
    checkpoint_interval_updates: int,
    run_id256: str,
    config_hash256: str,
    tensorboard_logger: TensorBoardLogger | None,
    hooks: TrainingCheckpointPromotionHooks,
) -> Mapping[str, Any] | None:
    update_count = int(learner.update_count)
    if update_count % int(checkpoint_interval_updates) != 0:
        return None

    snapshot = write_checkpoint_snapshot_and_aliases(
        hooks=hooks,
        learner=learner,
        stack=stack,
        artifacts=artifacts,
        training_paths=training_paths,
        device=device,
        spec_hash256=spec_hash256,
        algorithm=algorithm,
        latest_metrics=latest_metrics,
        last_dev_eval_summary=last_dev_eval_summary,
        tensorboard_logger=tensorboard_logger,
        update_count=update_count,
    )
    candidate = persist_checkpoint_snapshot_candidate(
        hooks=hooks,
        learner=learner,
        stack=stack,
        artifacts=artifacts,
        training_paths=training_paths,
        checkpoint_path=snapshot.checkpoint_path,
        config_hash256=config_hash256,
        device=device,
        update_count=update_count,
    )
    run_snapshot_candidate_promotion_gate(
        hooks=hooks,
        learner=learner,
        stack=stack,
        contract=contract,
        artifacts=artifacts,
        training_paths=training_paths,
        runtime=runtime,
        latest_metrics=latest_metrics,
        candidate=candidate,
        update_count=update_count,
        run_id256=run_id256,
        config_hash256=config_hash256,
        spec_hash256=spec_hash256,
    )
    return snapshot.tracker_payload


def write_checkpoint_snapshot_and_aliases(
    *,
    hooks: TrainingCheckpointPromotionHooks,
    learner: CheckpointPromotionLearner,
    stack: StackConfig,
    artifacts: CheckpointPromotionArtifacts,
    training_paths: CheckpointPromotionPaths,
    device: torch.device,
    spec_hash256: str,
    algorithm: Any,
    latest_metrics: dict[str, float],
    last_dev_eval_summary: Mapping[str, Any] | None,
    tensorboard_logger: TensorBoardLogger | None,
    update_count: int,
) -> CheckpointSnapshotWriteResult:
    checkpoint_path = checkpoint_path_for_update(training_paths.checkpoints_dir, update_count=update_count)
    hooks.write_checkpoint(
        checkpoint_path=checkpoint_path,
        learner=learner,
        stack=stack,
        device=device,
        spec_hash256=spec_hash256,
        algorithm=algorithm,
    )
    tracker_payload = hooks.publish_checkpoint_aliases(
        stack=stack,
        training_paths=training_paths,
        artifacts=artifacts,
        checkpoint_path=checkpoint_path,
        learner=learner,
        latest_metrics=latest_metrics,
    )
    hooks.maybe_log_structured_mainmove_guard(
        training_paths=training_paths,
        learner=learner,
        latest_metrics=latest_metrics,
        dev_eval_summary=last_dev_eval_summary,
    )
    if tensorboard_logger is not None:
        tensorboard_logger.log_checkpoint_tracker(tracker_payload, step=update_count)

    return CheckpointSnapshotWriteResult(
        checkpoint_path=checkpoint_path,
        tracker_payload=cast(Mapping[str, Any], tracker_payload),
    )


def persist_checkpoint_snapshot_candidate(
    *,
    hooks: TrainingCheckpointPromotionHooks,
    learner: CheckpointPromotionLearner,
    stack: StackConfig,
    artifacts: CheckpointPromotionArtifacts,
    training_paths: CheckpointPromotionPaths,
    checkpoint_path: Path,
    config_hash256: str,
    device: torch.device,
    update_count: int,
) -> SnapshotRegistryCandidate:
    if learner.model is None:
        raise RuntimeError("Cannot persist a snapshot registry entry without a learner model")
    candidate_policy_id = hooks.persist_snapshot_registry_entry(
        stack=stack,
        training_paths=training_paths,
        run_dir=artifacts.run_dir,
        checkpoint_path=checkpoint_path,
        model_state_dict=learner.model.state_dict(),
        config_hash256=config_hash256,
        device=device,
        update=update_count,
        policy_version=int(learner.get_policy_version()),
        model=learner.model,
    )
    return SnapshotRegistryCandidate(policy_id=str(candidate_policy_id))


def run_snapshot_candidate_promotion_gate(
    *,
    hooks: TrainingCheckpointPromotionHooks,
    learner: CheckpointPromotionLearner,
    stack: StackConfig,
    contract: SimulatorContract,
    artifacts: CheckpointPromotionArtifacts,
    training_paths: CheckpointPromotionPaths,
    runtime: CheckpointPromotionRuntime,
    latest_metrics: Mapping[str, float],
    candidate: SnapshotRegistryCandidate,
    update_count: int,
    run_id256: str,
    config_hash256: str,
    spec_hash256: str,
) -> SnapshotPromotionGateResult:
    runtime.refresh_opponent_pool()
    promotion_passed = hooks.run_snapshot_promotion_gate(
        stack=stack,
        contract=contract,
        artifacts=artifacts,
        training_paths=training_paths,
        learner=learner,
        candidate_policy_id=candidate.policy_id,
        update_count=update_count,
        league_reference_update=league_reference_update_from_metrics(latest_metrics),
        policy_version=int(learner.get_policy_version()),
        run_id256=run_id256,
        config_hash256=config_hash256,
        spec_hash256=spec_hash256,
    )
    if promotion_passed:
        runtime.refresh_opponent_pool()
    return SnapshotPromotionGateResult(
        candidate_policy_id=candidate.policy_id,
        promotion_passed=bool(promotion_passed),
    )


__all__ = [
    "CheckpointPromotionArtifacts",
    "CheckpointPromotionLearner",
    "CheckpointPromotionPaths",
    "CheckpointPromotionRuntime",
    "CheckpointSnapshotWriteResult",
    "SnapshotPromotionGateResult",
    "SnapshotRegistryCandidate",
    "TrainingCheckpointPromotionHooks",
    "league_reference_update_from_metrics",
    "maybe_checkpoint_and_promote_snapshot",
    "persist_checkpoint_snapshot_candidate",
    "run_snapshot_candidate_promotion_gate",
    "write_checkpoint_snapshot_and_aliases",
]
