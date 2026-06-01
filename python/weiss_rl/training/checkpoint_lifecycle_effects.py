from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from weiss_rl.training.snapshots import demote_registry_champions_newer_than


class CheckpointGuardEffectPaths(Protocol):
    @property
    def best_checkpoint_path(self) -> Path: ...

    @property
    def snapshots_dir(self) -> Path: ...


class CheckpointGuardEffectRuntime(Protocol):
    def maybe_publish_snapshot(self, **kwargs: Any) -> Mapping[str, Any]: ...

    def reset_outcome_tracker(self) -> None: ...

    def refresh_opponent_pool(self) -> None: ...


@dataclass(frozen=True, slots=True)
class CheckpointGuardRestoreEffects:
    best_checkpoint_path: Path
    demoted_champions: Sequence[str]
    publish_metrics: Mapping[str, Any]


def restore_best_checkpoint_state(
    *,
    training_paths: CheckpointGuardEffectPaths,
    best_update_count: int,
    restore_checkpoint: Any,
) -> tuple[Path, Sequence[str]]:
    best_checkpoint_path = training_paths.best_checkpoint_path
    restore_checkpoint(best_checkpoint_path, restore_counters=False)
    demoted_champions = demote_registry_champions_newer_than(
        training_paths,
        update_count=int(best_update_count),
    )
    return best_checkpoint_path, demoted_champions


def refresh_checkpoint_guard_runtime(runtime: CheckpointGuardEffectRuntime) -> None:
    runtime.reset_outcome_tracker()
    runtime.refresh_opponent_pool()


def apply_rollback_to_best_effects(
    *,
    training_paths: CheckpointGuardEffectPaths,
    runtime: CheckpointGuardEffectRuntime,
    learner_model: Any,
    learner_update_count: int,
    best_update_count: int,
    restore_checkpoint: Any,
) -> CheckpointGuardRestoreEffects:
    best_checkpoint_path, demoted_champions = restore_best_checkpoint_state(
        training_paths=training_paths,
        best_update_count=best_update_count,
        restore_checkpoint=restore_checkpoint,
    )
    publish_metrics = runtime.maybe_publish_snapshot(
        learner_model=learner_model,
        learner_update_count=int(learner_update_count),
        force=True,
    )
    refresh_checkpoint_guard_runtime(runtime)
    return CheckpointGuardRestoreEffects(
        best_checkpoint_path=best_checkpoint_path,
        demoted_champions=demoted_champions,
        publish_metrics=publish_metrics,
    )


def apply_finalize_to_best_effects(
    *,
    training_paths: CheckpointGuardEffectPaths,
    runtime: CheckpointGuardEffectRuntime,
    best_update_count: int,
    restore_checkpoint: Any,
) -> CheckpointGuardRestoreEffects:
    best_checkpoint_path, demoted_champions = restore_best_checkpoint_state(
        training_paths=training_paths,
        best_update_count=best_update_count,
        restore_checkpoint=restore_checkpoint,
    )
    refresh_checkpoint_guard_runtime(runtime)
    return CheckpointGuardRestoreEffects(
        best_checkpoint_path=best_checkpoint_path,
        demoted_champions=demoted_champions,
        publish_metrics={},
    )


__all__ = [
    "CheckpointGuardEffectPaths",
    "CheckpointGuardEffectRuntime",
    "CheckpointGuardRestoreEffects",
    "apply_finalize_to_best_effects",
    "apply_rollback_to_best_effects",
    "refresh_checkpoint_guard_runtime",
    "restore_best_checkpoint_state",
]
