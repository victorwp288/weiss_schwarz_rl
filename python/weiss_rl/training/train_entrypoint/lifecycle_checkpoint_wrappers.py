"""Checkpoint and promotion lifecycle wrappers for the training entrypoint."""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping
from pathlib import Path
from typing import Any

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
from weiss_rl.training.script_entrypoint_promotion_hooks import (
    SnapshotPromotionGateRequest,
    run_snapshot_promotion_gate_with_script_hooks,
)


def install_current_checkpoint_wrapper(
    namespace: MutableMapping[str, Any],
    *,
    entrypoint_api: Callable[[], Any],
) -> None:
    """Install the current-checkpoint helper expected by the legacy facade."""

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


def install_best_checkpoint_wrappers(
    namespace: MutableMapping[str, Any],
    *,
    entrypoint_api: Callable[[], Any],
) -> None:
    """Install rollback/finalization helpers for best-checkpoint selection."""

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
    """Install the promotion-gate helper used by checkpoint publication."""

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


__all__ = [
    "install_best_checkpoint_wrappers",
    "install_current_checkpoint_wrapper",
    "install_promotion_wrapper",
]
