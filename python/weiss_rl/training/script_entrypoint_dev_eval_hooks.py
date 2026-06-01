"""Dev-eval callback assembly for the path-based training entrypoint."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PeriodicDevEvalOpponentsRequest:
    stack: Any
    contract: Any
    run_dir: Path
    observation_dim: int
    action_dim: int


@dataclass(frozen=True)
class StallMonitorRequest:
    stack: Any
    training_paths: Any
    update_count: int
    summary_payload: Mapping[str, Any]


@dataclass(frozen=True)
class PeriodicDevEvalRequest:
    stack: Any
    contract: Any
    artifacts: Any
    training_paths: Any
    learner: Any
    device: Any
    run_id256: str
    config_hash256: str
    spec_hash256: str
    artifact_dir_name: str = "dev_eval"
    artifact_scope: str = "periodic_dev_eval"
    paired_seeds_override: Sequence[int] | None = None
    persist_summary: bool = True
    update_stall_monitor: bool = True


def periodic_dev_eval_opponents_with_script_hooks(api: Any, request: PeriodicDevEvalOpponentsRequest) -> Any:
    return api.periodic_dev_eval_opponents(
        stack=request.stack,
        contract=request.contract,
        run_dir=request.run_dir,
        observation_dim=request.observation_dim,
        action_dim=request.action_dim,
        load_snapshot_model=api._load_snapshot_eval_model,
        build_heuristic_policy=api._build_heuristic_public_policy,
    )


def update_stall_monitor_with_script_hooks(api: Any, request: StallMonitorRequest) -> Any:
    return api._update_stall_monitor_impl(
        stack=request.stack,
        training_paths=request.training_paths,
        update_count=request.update_count,
        summary_payload=request.summary_payload,
    )


def run_periodic_dev_eval_with_script_hooks(
    api: Any,
    request: PeriodicDevEvalRequest | None = None,
    **kwargs: Any,
) -> Any:
    if request is None:
        request = PeriodicDevEvalRequest(
            stack=kwargs["stack"],
            contract=kwargs["contract"],
            artifacts=kwargs["artifacts"],
            training_paths=kwargs["training_paths"],
            learner=kwargs["learner"],
            device=kwargs["device"],
            run_id256=kwargs["run_id256"],
            config_hash256=kwargs["config_hash256"],
            spec_hash256=kwargs["spec_hash256"],
            artifact_dir_name=kwargs.get("artifact_dir_name", "dev_eval"),
            artifact_scope=kwargs.get("artifact_scope", "periodic_dev_eval"),
            paired_seeds_override=kwargs.get("paired_seeds_override"),
            persist_summary=kwargs.get("persist_summary", True),
            update_stall_monitor=kwargs.get(
                "update_stall_monitor_enabled",
                kwargs.get("update_stall_monitor", True),
            ),
        )
    return api.run_periodic_dev_eval(
        stack=request.stack,
        contract=request.contract,
        artifacts=request.artifacts,
        training_paths=request.training_paths,
        learner=request.learner,
        device=request.device,
        run_id256=request.run_id256,
        config_hash256=request.config_hash256,
        spec_hash256=request.spec_hash256,
        runner_cls=api._PeriodicDevEvalRunner,
        ensure_current_checkpoint_fn=api._ensure_current_checkpoint,
        current_focal_policy_id_fn=api._current_focal_policy_id,
        artifact_dir_name=request.artifact_dir_name,
        artifact_scope=request.artifact_scope,
        paired_seeds_override=request.paired_seeds_override,
        persist_summary=request.persist_summary,
        update_stall_monitor_enabled=request.update_stall_monitor,
        spec_dimensions_fn=api._spec_dimensions,
        clone_cpu_eval_model_fn=api._clone_cpu_eval_model,
        periodic_dev_eval_opponents_fn=api._periodic_dev_eval_opponents,
        persist_summary_fn=api._persist_periodic_dev_eval_summary,
        update_stall_monitor_fn=api._update_stall_monitor,
        write_json_fn=api._write_json,
    )
