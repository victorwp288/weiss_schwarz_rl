"""Registry and run-metadata hook assembly for the training entrypoint facade."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class PersistSnapshotRegistryEntryRequest:
    stack: Any
    training_paths: Any
    run_dir: Path
    checkpoint_path: Path
    model_state_dict: dict[str, Any]
    config_hash256: str
    device: torch.device
    update: int
    policy_version: int
    model: Any | None = None


def persist_snapshot_registry_entry_with_entrypoint_hooks(
    api: Any, request: PersistSnapshotRegistryEntryRequest
) -> str:
    guidance_payload = api._model_guidance_payload(request.model)
    return str(
        api.persist_snapshot_registry_entry(
            stack=request.stack,
            training_paths=request.training_paths,
            run_dir=request.run_dir,
            checkpoint_path=request.checkpoint_path,
            model_state_dict=request.model_state_dict,
            config_hash256=request.config_hash256,
            device=request.device,
            update=request.update,
            policy_version=request.policy_version,
            public_heuristic_logit_bias_scale=guidance_payload.get("public_heuristic_logit_bias_scale"),
            public_heuristic_actor_logit_bias_scale=guidance_payload.get("public_heuristic_actor_logit_bias_scale"),
        )
    )


def repo_root_with_entrypoint_hooks(api: Any, *, source_file: Path) -> Path:
    return Path(api.repo_root(source_file))


def git_output_with_entrypoint_hooks(api: Any, *, args: list[str], source_file: Path) -> str:
    return str(api.git_output(args, cwd=repo_root_with_entrypoint_hooks(api, source_file=source_file)))


def git_commit_with_entrypoint_hooks(api: Any, *, source_file: Path) -> str:
    return str(api.git_commit(cwd=repo_root_with_entrypoint_hooks(api, source_file=source_file)))


def git_dirty_with_entrypoint_hooks(api: Any, *, source_file: Path) -> bool:
    return bool(api.git_dirty(cwd=repo_root_with_entrypoint_hooks(api, source_file=source_file)))


def experiment_role_with_entrypoint_hooks(stack: Any) -> str:
    experiment = stack.config.experiment
    return "" if experiment is None else str(experiment.role).strip()
