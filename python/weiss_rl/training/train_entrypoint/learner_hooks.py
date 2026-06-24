"""Learner construction and warmstart hooks for the training entrypoint facade."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from pathlib import Path
from typing import Any

from weiss_rl.training.train_entrypoint.checkpoint_io_hooks import (
    initialize_learner_from_checkpoint_with_entrypoint_hooks,
    restore_learner_from_checkpoint_with_entrypoint_hooks,
)
from weiss_rl.training.train_entrypoint.checkpoint_requests import (
    BuildTrainingLearnerRequest,
    InitializeLearnerCheckpointRequest,
    RestoreLearnerCheckpointRequest,
    StructuredWarmstartRequest,
)


def build_training_learner_with_entrypoint_hooks(api: Any, request: BuildTrainingLearnerRequest) -> Any:
    return api.build_training_learner(
        algorithm=request.algorithm,
        model=request.model,
        compiled_model=request.compiled_model,
        training_config=request.training_config,
        training_paths=request.training_paths,
        pass_action_id=request.pass_action_id,
        checkpoint_interval_updates=request.checkpoint_interval_updates,
    )


def run_structured_warmstart_with_entrypoint_hooks(api: Any, request: StructuredWarmstartRequest) -> dict[str, float]:
    return api.run_structured_warmstart(
        learner=request.learner,
        runtime=request.runtime,
        algorithm=request.algorithm,
        training_config=request.training_config,
        rewards_config=request.rewards_config,
        training_paths=request.training_paths,
        tensorboard_logger=request.tensorboard_logger,
        start_time=request.start_time,
        profile_timers=request.profile_timers,
        actor_torch_threads=request.actor_torch_threads,
        learner_torch_threads=request.learner_torch_threads,
    )


def install_learner_wrappers(
    namespace: MutableMapping[str, Any],
    *,
    entrypoint_api: Callable[[], Any],
) -> None:
    def _restore_learner_from_checkpoint(
        *,
        checkpoint_path: Path,
        learner: Any,
        stack: Any,
        device: Any,
        expected_spec_hash256: str,
        algorithm: str,
        restore_counters: bool = True,
    ) -> Any:
        return restore_learner_from_checkpoint_with_entrypoint_hooks(
            entrypoint_api(),
            RestoreLearnerCheckpointRequest(
                checkpoint_path=checkpoint_path,
                learner=learner,
                stack=stack,
                device=device,
                expected_spec_hash256=expected_spec_hash256,
                algorithm=algorithm,
                restore_counters=restore_counters,
            ),
        )

    def _initialize_learner_from_checkpoint(
        *,
        checkpoint_path: Path,
        learner: Any,
        device: Any,
        expected_spec_hash256: str,
        algorithm: str,
    ) -> Any:
        return initialize_learner_from_checkpoint_with_entrypoint_hooks(
            entrypoint_api(),
            InitializeLearnerCheckpointRequest(
                checkpoint_path=checkpoint_path,
                learner=learner,
                device=device,
                expected_spec_hash256=expected_spec_hash256,
                algorithm=algorithm,
            ),
        )

    def _build_training_learner(
        *,
        algorithm: str,
        model: Any,
        compiled_model: Any,
        training_config: Any,
        training_paths: Any,
        pass_action_id: int,
        checkpoint_interval_updates: int,
    ) -> Any:
        return build_training_learner_with_entrypoint_hooks(
            entrypoint_api(),
            BuildTrainingLearnerRequest(
                algorithm=algorithm,
                model=model,
                compiled_model=compiled_model,
                training_config=training_config,
                training_paths=training_paths,
                pass_action_id=pass_action_id,
                checkpoint_interval_updates=checkpoint_interval_updates,
            ),
        )

    def _run_structured_warmstart(
        *,
        learner: Any,
        runtime: Any,
        algorithm: str,
        training_config: Any,
        rewards_config: Any,
        training_paths: Any,
        tensorboard_logger: Any | None,
        start_time: float,
        profile_timers: bool = False,
        actor_torch_threads: int | None = None,
        learner_torch_threads: int | None = None,
    ) -> dict[str, float]:
        return run_structured_warmstart_with_entrypoint_hooks(
            entrypoint_api(),
            StructuredWarmstartRequest(
                learner=learner,
                runtime=runtime,
                algorithm=algorithm,
                training_config=training_config,
                rewards_config=rewards_config,
                training_paths=training_paths,
                tensorboard_logger=tensorboard_logger,
                start_time=start_time,
                profile_timers=profile_timers,
                actor_torch_threads=actor_torch_threads,
                learner_torch_threads=learner_torch_threads,
            ),
        )

    namespace.update(
        {
            "_restore_learner_from_checkpoint": _restore_learner_from_checkpoint,
            "_initialize_learner_from_checkpoint": _initialize_learner_from_checkpoint,
            "_build_training_learner": _build_training_learner,
            "_run_structured_warmstart": _run_structured_warmstart,
        }
    )


__all__ = [
    "build_training_learner_with_entrypoint_hooks",
    "initialize_learner_from_checkpoint_with_entrypoint_hooks",
    "install_learner_wrappers",
    "restore_learner_from_checkpoint_with_entrypoint_hooks",
    "run_structured_warmstart_with_entrypoint_hooks",
]
