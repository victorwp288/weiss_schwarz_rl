"""Best-checkpoint callback assembly for the path-based training entrypoint."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RollbackToBestCheckpointRequest:
    stack: Any
    training_paths: Any
    artifacts: Any
    runtime: Any
    learner: Any
    model: Any
    device: Any
    spec_hash256: str
    algorithm: str
    latest_metrics: Mapping[str, float] | None
    dev_eval_summary: Mapping[str, Any] | None
    last_rollback_update: int | None


@dataclass(frozen=True)
class FinalizeFromBestCheckpointRequest:
    stack: Any
    training_paths: Any
    artifacts: Any
    runtime: Any
    learner: Any
    device: Any
    spec_hash256: str
    algorithm: str
    latest_metrics: Mapping[str, float] | None
    dev_eval_summary: Mapping[str, Any] | None


def maybe_rollback_to_best_checkpoint_with_script_hooks(api: Any, request: RollbackToBestCheckpointRequest) -> Any:
    stack = request.stack
    learner = request.learner
    device = request.device
    spec_hash256 = request.spec_hash256
    algorithm = request.algorithm
    return api.maybe_rollback_to_best_checkpoint(
        stack=stack,
        training_paths=request.training_paths,
        run_dir=request.artifacts.run_dir,
        runtime=request.runtime,
        learner=learner,
        learner_model=request.model,
        latest_metrics=request.latest_metrics,
        dev_eval_summary=request.dev_eval_summary,
        last_rollback_update=request.last_rollback_update,
        restore_checkpoint=lambda checkpoint_path, *, restore_counters: api._restore_learner_from_checkpoint(
            checkpoint_path=checkpoint_path,
            learner=learner,
            stack=stack,
            device=device,
            expected_spec_hash256=spec_hash256,
            algorithm=algorithm,
            restore_counters=restore_counters,
        ),
        write_checkpoint=lambda checkpoint_path: api._write_checkpoint(
            checkpoint_path=checkpoint_path,
            learner=learner,
            stack=stack,
            device=device,
            spec_hash256=spec_hash256,
            algorithm=algorithm,
        ),
    )


def maybe_finalize_from_best_checkpoint_with_script_hooks(api: Any, request: FinalizeFromBestCheckpointRequest) -> Any:
    stack = request.stack
    training_paths = request.training_paths
    learner = request.learner
    device = request.device
    spec_hash256 = request.spec_hash256
    algorithm = request.algorithm
    return api.maybe_finalize_from_best_checkpoint(
        stack=stack,
        training_paths=training_paths,
        run_dir=request.artifacts.run_dir,
        runtime=request.runtime,
        learner=learner,
        latest_metrics=request.latest_metrics,
        dev_eval_summary=request.dev_eval_summary,
        restore_checkpoint=lambda checkpoint_path, *, restore_counters: api._restore_learner_from_checkpoint(
            checkpoint_path=checkpoint_path,
            learner=learner,
            stack=stack,
            device=device,
            expected_spec_hash256=spec_hash256,
            algorithm=algorithm,
            restore_counters=restore_counters,
        ),
        ensure_current_checkpoint=lambda: api._ensure_current_checkpoint(
            training_paths=training_paths,
            learner=learner,
            stack=stack,
            device=device,
            spec_hash256=spec_hash256,
            algorithm=algorithm,
        ),
    )
