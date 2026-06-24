"""Script-compatible dev-eval wrapper installation for training entrypoints."""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any

from weiss_rl.training.script_entrypoint_dev_eval_hooks import (
    PeriodicDevEvalOpponentsRequest,
    PeriodicDevEvalRequest,
    StallMonitorRequest,
    periodic_dev_eval_opponents_with_script_hooks,
    run_periodic_dev_eval_with_script_hooks,
    update_stall_monitor_with_script_hooks,
)


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


__all__ = ["install_dev_eval_wrappers"]
