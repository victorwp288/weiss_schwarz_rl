"""Runtime lifecycle wrapper installation for the training entrypoint facade."""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any

from weiss_rl.training.minimal.entrypoint_hooks import (
    MinimalTrainingEntryRequest,
    run_minimal_training_with_script_hooks,
)
from weiss_rl.training.script_entrypoint_best_checkpoint_hooks import (
    FinalizeFromBestCheckpointRequest,
    RollbackToBestCheckpointRequest,
    maybe_finalize_from_best_checkpoint_with_script_hooks,
    maybe_rollback_to_best_checkpoint_with_script_hooks,
)
from weiss_rl.training.script_entrypoint_current_checkpoint_hooks import (
    EnsureCurrentCheckpointRequest,
    ensure_current_checkpoint_with_script_hooks,
)
from weiss_rl.training.script_entrypoint_dev_eval_hooks import (
    PeriodicDevEvalOpponentsRequest,
    PeriodicDevEvalRequest,
    StallMonitorRequest,
    periodic_dev_eval_opponents_with_script_hooks,
    run_periodic_dev_eval_with_script_hooks,
    update_stall_monitor_with_script_hooks,
)
from weiss_rl.training.script_entrypoint_promotion_hooks import (
    SnapshotPromotionGateRequest,
    run_snapshot_promotion_gate_with_script_hooks,
)


def install_current_checkpoint_wrapper(
    namespace: MutableMapping[str, Any],
    *,
    entrypoint_api: Callable[[], Any],
) -> None:
    def _ensure_current_checkpoint(
        *,
        training_paths: Any,
        learner: Any,
        stack: Any,
        device: Any,
        spec_hash256: str | None = None,
        algorithm: str | None = None,
    ) -> Path:
        return ensure_current_checkpoint_with_script_hooks(
            entrypoint_api(),
            EnsureCurrentCheckpointRequest(
                training_paths=training_paths,
                learner=learner,
                stack=stack,
                device=device,
                spec_hash256=spec_hash256,
                algorithm=algorithm,
            ),
        )

    namespace["_ensure_current_checkpoint"] = _ensure_current_checkpoint


def install_dev_eval_wrappers(
    namespace: MutableMapping[str, Any],
    *,
    entrypoint_api: Callable[[], Any],
) -> None:
    def _periodic_dev_eval_opponents(
        *,
        stack: Any,
        contract: Any,
        run_dir: Path,
        observation_dim: int,
        action_dim: int,
    ) -> list[tuple[str, str, Any | None, Any | None]]:
        return periodic_dev_eval_opponents_with_script_hooks(
            entrypoint_api(),
            PeriodicDevEvalOpponentsRequest(
                stack=stack,
                contract=contract,
                run_dir=run_dir,
                observation_dim=observation_dim,
                action_dim=action_dim,
            ),
        )

    def _update_stall_monitor(
        *,
        stack: Any,
        training_paths: Any,
        update_count: int,
        summary_payload: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        return update_stall_monitor_with_script_hooks(
            entrypoint_api(),
            StallMonitorRequest(
                stack=stack,
                training_paths=training_paths,
                update_count=update_count,
                summary_payload=summary_payload,
            ),
        )

    def _run_periodic_dev_eval(
        *,
        stack: Any,
        contract: Any,
        artifacts: Any,
        training_paths: Any,
        learner: Any,
        device: Any,
        run_id256: str,
        config_hash256: str,
        spec_hash256: str,
        artifact_dir_name: str = "dev_eval",
        artifact_scope: str = "periodic_dev_eval",
        paired_seeds_override: Sequence[int] | None = None,
        persist_summary: bool = True,
        update_stall_monitor: bool = True,
    ) -> dict[str, Any]:
        return run_periodic_dev_eval_with_script_hooks(
            entrypoint_api(),
            PeriodicDevEvalRequest(
                stack=stack,
                contract=contract,
                artifacts=artifacts,
                training_paths=training_paths,
                learner=learner,
                device=device,
                run_id256=run_id256,
                config_hash256=config_hash256,
                spec_hash256=spec_hash256,
                artifact_dir_name=artifact_dir_name,
                artifact_scope=artifact_scope,
                paired_seeds_override=paired_seeds_override,
                persist_summary=persist_summary,
                update_stall_monitor=update_stall_monitor,
            ),
        )

    namespace.update(
        {
            "_periodic_dev_eval_opponents": _periodic_dev_eval_opponents,
            "_update_stall_monitor": _update_stall_monitor,
            "_run_periodic_dev_eval": _run_periodic_dev_eval,
        }
    )


def install_best_checkpoint_wrappers(
    namespace: MutableMapping[str, Any],
    *,
    entrypoint_api: Callable[[], Any],
) -> None:
    def _maybe_rollback_to_best_checkpoint(
        *,
        stack: Any,
        training_paths: Any,
        artifacts: Any,
        runtime: Any,
        learner: Any,
        model: Any,
        device: Any,
        spec_hash256: str,
        algorithm: str,
        latest_metrics: Mapping[str, float] | None,
        dev_eval_summary: Mapping[str, Any] | None,
        last_rollback_update: int | None,
    ) -> dict[str, Any] | None:
        return maybe_rollback_to_best_checkpoint_with_script_hooks(
            entrypoint_api(),
            RollbackToBestCheckpointRequest(
                stack=stack,
                training_paths=training_paths,
                artifacts=artifacts,
                runtime=runtime,
                learner=learner,
                model=model,
                device=device,
                spec_hash256=spec_hash256,
                algorithm=algorithm,
                latest_metrics=latest_metrics,
                dev_eval_summary=dev_eval_summary,
                last_rollback_update=last_rollback_update,
            ),
        )

    def _maybe_finalize_from_best_checkpoint(
        *,
        stack: Any,
        training_paths: Any,
        artifacts: Any,
        runtime: Any,
        learner: Any,
        device: Any,
        spec_hash256: str,
        algorithm: str,
        latest_metrics: Mapping[str, float] | None,
        dev_eval_summary: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        return maybe_finalize_from_best_checkpoint_with_script_hooks(
            entrypoint_api(),
            FinalizeFromBestCheckpointRequest(
                stack=stack,
                training_paths=training_paths,
                artifacts=artifacts,
                runtime=runtime,
                learner=learner,
                device=device,
                spec_hash256=spec_hash256,
                algorithm=algorithm,
                latest_metrics=latest_metrics,
                dev_eval_summary=dev_eval_summary,
            ),
        )

    namespace.update(
        {
            "_maybe_rollback_to_best_checkpoint": _maybe_rollback_to_best_checkpoint,
            "_maybe_finalize_from_best_checkpoint": _maybe_finalize_from_best_checkpoint,
        }
    )


def install_promotion_wrapper(
    namespace: MutableMapping[str, Any],
    *,
    entrypoint_api: Callable[[], Any],
) -> None:
    def _run_snapshot_promotion_gate(
        *,
        stack: Any,
        contract: Any,
        artifacts: Any,
        training_paths: Any,
        learner: Any,
        candidate_policy_id: str,
        update_count: int,
        league_reference_update: int | None,
        policy_version: int,
        run_id256: str,
        config_hash256: str,
        spec_hash256: str,
    ) -> bool | None:
        return run_snapshot_promotion_gate_with_script_hooks(
            entrypoint_api(),
            SnapshotPromotionGateRequest(
                stack=stack,
                contract=contract,
                artifacts=artifacts,
                training_paths=training_paths,
                learner=learner,
                candidate_policy_id=candidate_policy_id,
                update_count=update_count,
                league_reference_update=league_reference_update,
                policy_version=policy_version,
                run_id256=run_id256,
                config_hash256=config_hash256,
                spec_hash256=spec_hash256,
            ),
        )

    namespace["_run_snapshot_promotion_gate"] = _run_snapshot_promotion_gate


def install_minimal_training_wrapper(
    namespace: MutableMapping[str, Any],
    *,
    entrypoint_api: Callable[[], Any],
) -> None:
    def _run_minimal_training(
        *,
        stack: Any,
        contract: Any,
        artifacts: Any,
        num_envs: int,
        unroll_length: int,
        max_updates: int,
        profile: str,
        device: Any,
        seed: int,
        checkpoint_interval_updates: int,
        run_id256: str,
        config_hash256: str,
        spec_hash256: str,
        runtime_mode: Any,
        b1_baseline_run_dir: Path | None,
        seed_snapshot_run_dir: Path | None = None,
        profile_timers: bool = False,
        torch_profiler: bool = False,
        resume_checkpoint_path: Path | None = None,
        init_from_checkpoint_path: Path | None = None,
        init_schedule_offset_override_updates: int | None = None,
        tensorboard_logger: Any | None = None,
    ) -> dict[str, float]:
        return run_minimal_training_with_script_hooks(
            entrypoint_api(),
            MinimalTrainingEntryRequest(
                stack=stack,
                contract=contract,
                artifacts=artifacts,
                num_envs=num_envs,
                unroll_length=unroll_length,
                max_updates=max_updates,
                profile=profile,
                device=device,
                seed=seed,
                checkpoint_interval_updates=checkpoint_interval_updates,
                run_id256=run_id256,
                config_hash256=config_hash256,
                spec_hash256=spec_hash256,
                runtime_mode=runtime_mode,
                b1_baseline_run_dir=b1_baseline_run_dir,
                seed_snapshot_run_dir=seed_snapshot_run_dir,
                profile_timers=profile_timers,
                torch_profiler=torch_profiler,
                resume_checkpoint_path=resume_checkpoint_path,
                init_from_checkpoint_path=init_from_checkpoint_path,
                init_schedule_offset_override_updates=init_schedule_offset_override_updates,
                tensorboard_logger=tensorboard_logger,
            ),
        )

    namespace["_run_minimal_training"] = _run_minimal_training


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
