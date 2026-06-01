"""Final checkpoint selection phase for minimal training."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import torch

from weiss_rl.config import StackConfig
from weiss_rl.diagnostics.tensorboard_logger import TensorBoardLogger


class FinalCheckpointLearner(Protocol):
    update_count: int


class FinalCheckpointArtifacts(Protocol):
    run_dir: Path


class FinalCheckpointPaths(Protocol):
    checkpoints_dir: Path


@dataclass(frozen=True, slots=True)
class TrainingFinalCheckpointHooks:
    ensure_current_checkpoint: Any
    publish_checkpoint_aliases: Any
    maybe_finalize_from_best_checkpoint: Any
    load_checkpoint_tracker: Any


@dataclass(frozen=True, slots=True)
class FinalCheckpointPublication:
    checkpoint_path: Path
    dev_eval_summary: Mapping[str, Any] | None
    tracker_payload: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class FinalCheckpointSelection:
    tracker_payload: Mapping[str, Any]
    guard_event: Mapping[str, Any] | None


def final_dev_eval_summary_for_update(
    *,
    last_dev_eval_summary: Mapping[str, Any] | None,
    last_dev_eval_update_count: int | None,
    learner_update_count: int,
) -> Mapping[str, Any] | None:
    if last_dev_eval_update_count != int(learner_update_count):
        return None
    return last_dev_eval_summary


def finalize_training_checkpoint_selection(
    *,
    learner: FinalCheckpointLearner,
    stack: StackConfig,
    artifacts: FinalCheckpointArtifacts,
    training_paths: FinalCheckpointPaths,
    runtime: Any,
    device: torch.device,
    spec_hash256: str,
    algorithm: Any,
    latest_metrics: dict[str, float],
    last_dev_eval_summary: Mapping[str, Any] | None,
    last_dev_eval_update_count: int | None,
    tensorboard_logger: TensorBoardLogger | None,
    hooks: TrainingFinalCheckpointHooks,
) -> Mapping[str, Any]:
    update_count = int(learner.update_count)
    publication = publish_final_checkpoint_aliases(
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
        last_dev_eval_update_count=last_dev_eval_update_count,
        update_count=update_count,
    )
    selection = select_final_checkpoint_tracker_payload(
        hooks=hooks,
        learner=learner,
        stack=stack,
        artifacts=artifacts,
        training_paths=training_paths,
        runtime=runtime,
        device=device,
        spec_hash256=spec_hash256,
        algorithm=algorithm,
        latest_metrics=latest_metrics,
        publication=publication,
    )
    if tensorboard_logger is not None:
        tensorboard_logger.log_checkpoint_tracker(selection.tracker_payload, step=update_count)
    return selection.tracker_payload


def publish_final_checkpoint_aliases(
    *,
    hooks: TrainingFinalCheckpointHooks,
    learner: FinalCheckpointLearner,
    stack: StackConfig,
    artifacts: FinalCheckpointArtifacts,
    training_paths: FinalCheckpointPaths,
    device: torch.device,
    spec_hash256: str,
    algorithm: Any,
    latest_metrics: dict[str, float],
    last_dev_eval_summary: Mapping[str, Any] | None,
    last_dev_eval_update_count: int | None,
    update_count: int,
) -> FinalCheckpointPublication:
    final_checkpoint_path = hooks.ensure_current_checkpoint(
        training_paths=training_paths,
        learner=learner,
        stack=stack,
        device=device,
        spec_hash256=spec_hash256,
        algorithm=algorithm,
    )
    final_dev_eval_summary = final_dev_eval_summary_for_update(
        last_dev_eval_summary=last_dev_eval_summary,
        last_dev_eval_update_count=last_dev_eval_update_count,
        learner_update_count=update_count,
    )
    tracker_payload = hooks.publish_checkpoint_aliases(
        stack=stack,
        training_paths=training_paths,
        artifacts=artifacts,
        checkpoint_path=final_checkpoint_path,
        learner=learner,
        latest_metrics=latest_metrics,
        dev_eval_summary=final_dev_eval_summary,
    )
    return FinalCheckpointPublication(
        checkpoint_path=final_checkpoint_path,
        dev_eval_summary=final_dev_eval_summary,
        tracker_payload=cast(Mapping[str, Any], tracker_payload),
    )


def select_final_checkpoint_tracker_payload(
    *,
    hooks: TrainingFinalCheckpointHooks,
    learner: FinalCheckpointLearner,
    stack: StackConfig,
    artifacts: FinalCheckpointArtifacts,
    training_paths: FinalCheckpointPaths,
    runtime: Any,
    device: torch.device,
    spec_hash256: str,
    algorithm: Any,
    latest_metrics: dict[str, float],
    publication: FinalCheckpointPublication,
) -> FinalCheckpointSelection:
    finalize_guard_event = hooks.maybe_finalize_from_best_checkpoint(
        stack=stack,
        training_paths=training_paths,
        artifacts=artifacts,
        runtime=runtime,
        learner=learner,
        device=device,
        spec_hash256=spec_hash256,
        algorithm=algorithm,
        latest_metrics=latest_metrics,
        dev_eval_summary=publication.dev_eval_summary,
    )
    if finalize_guard_event is not None:
        print_final_checkpoint_selection_event(finalize_guard_event)
        tracker_payload = hooks.load_checkpoint_tracker(training_paths)
        return FinalCheckpointSelection(
            tracker_payload=cast(Mapping[str, Any], tracker_payload),
            guard_event=cast(Mapping[str, Any], finalize_guard_event),
        )

    return FinalCheckpointSelection(
        tracker_payload=publication.tracker_payload,
        guard_event=None,
    )


def print_final_checkpoint_selection_event(finalize_guard_event: Mapping[str, Any]) -> None:
    print(
        "Checkpoint guard final selection: "
        f"update={finalize_guard_event['update_count']} "
        f"best_update={finalize_guard_event['best_update_count']} "
        f"current_score={float(finalize_guard_event['current_score']):.4f} "
        f"best_score={float(finalize_guard_event['best_score']):.4f}"
    )


__all__ = [
    "FinalCheckpointArtifacts",
    "FinalCheckpointLearner",
    "FinalCheckpointPaths",
    "FinalCheckpointPublication",
    "FinalCheckpointSelection",
    "TrainingFinalCheckpointHooks",
    "final_dev_eval_summary_for_update",
    "finalize_training_checkpoint_selection",
    "print_final_checkpoint_selection_event",
    "publish_final_checkpoint_aliases",
    "select_final_checkpoint_tracker_payload",
]
