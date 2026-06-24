"""State objects passed between training entrypoint stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TrainCliState:
    run_label: str
    num_envs: int
    unroll_length: int
    max_updates: int
    runtime_mode: Any
    stack: Any
    training_config: Any | None
    manifest_only_reason: str | None
    public_demo_enabled: bool
    resume_run_dir: Path | None
    resume_checkpoint_path: Path | None
    init_from_checkpoint_path: Path | None
    init_schedule_offset_override_updates: int | None


@dataclass(frozen=True, slots=True)
class TrainStartupState:
    cli: TrainCliState
    simulator_contract: Any | None
    spec_bundle: dict[str, Any]
    spec_hash256: str
    simulator_info: dict[str, Any]
    config_hash256: str
    git_commit: str
    start_nonce: str
    run_id256: str
    run_id64: str
    run_dir_name: str
    resume_artifacts: Any | None


@dataclass(frozen=True, slots=True)
class TrainManifestState:
    artifacts: Any
    manifest: Any
    device: Any
    profile: str
    seed: int
    policy_set_selection_details: dict[str, Any]
    tensorboard_logger: Any
    run_summary_payload: dict[str, Any]
    determinism_payload: dict[str, Any]
    environment_payload: dict[str, Any]


__all__ = ["TrainCliState", "TrainManifestState", "TrainStartupState"]
