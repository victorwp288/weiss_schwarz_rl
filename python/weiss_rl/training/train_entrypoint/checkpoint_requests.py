"""Request objects for training checkpoint and snapshot entrypoint hooks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class WriteCheckpointRequest:
    checkpoint_path: Path
    learner: Any
    stack: Any
    device: torch.device
    spec_hash256: str | None = None
    algorithm: str | None = None


@dataclass(frozen=True)
class BuildCheckpointRecordRequest:
    alias_name: str
    alias_path: Path
    source_checkpoint_path: Path
    artifacts: Any
    learner: Any
    metric_kind: str | None = None
    metric_value: float | None = None


@dataclass(frozen=True)
class PublishCheckpointAliasesRequest:
    stack: Any
    training_paths: Any
    artifacts: Any
    checkpoint_path: Path
    learner: Any
    latest_metrics: Mapping[str, float] | None
    dev_eval_summary: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class RestoreLearnerCheckpointRequest:
    checkpoint_path: Path
    learner: Any
    stack: Any
    device: torch.device
    expected_spec_hash256: str
    algorithm: str
    restore_counters: bool = True


@dataclass(frozen=True)
class InitializeLearnerCheckpointRequest:
    checkpoint_path: Path
    learner: Any
    device: torch.device
    expected_spec_hash256: str
    algorithm: str


@dataclass(frozen=True)
class BuildTrainingLearnerRequest:
    algorithm: str
    model: Any
    compiled_model: Any
    training_config: Any
    training_paths: Any
    pass_action_id: int
    checkpoint_interval_updates: int


@dataclass(frozen=True)
class StructuredWarmstartRequest:
    learner: Any
    runtime: Any
    algorithm: str
    training_config: Any
    rewards_config: Any
    training_paths: Any
    tensorboard_logger: Any
    start_time: float
    profile_timers: bool = False
    actor_torch_threads: int | None = None
    learner_torch_threads: int | None = None


@dataclass(frozen=True)
class BuildHeuristicPublicPolicyRequest:
    spec_bundle: Mapping[str, object]
    scoring_profile: str


@dataclass(frozen=True)
class ImportNoLeagueBaselineAnchorRequest:
    training_paths: Any
    run_dir: Path
    baseline_run_dir: Path
    expected_model_state_dict: dict[str, Any]
    expected_config_canonical: dict[str, Any] | None
    expected_spec_hash256: str | None


@dataclass(frozen=True)
class ValidateSeedSnapshotImportContractRequest:
    source_run_dir: Path
    payload: dict[str, Any]
    expected_model_state_dict: dict[str, Any]
    expected_config_canonical: dict[str, Any] | None
    expected_spec_hash256: str | None


@dataclass(frozen=True)
class SeedSnapshotPolicyIdRequest:
    source_run_dir: Path
    source_policy_id: str


@dataclass(frozen=True)
class ImportSeedSnapshotPoolRequest:
    stack: Any
    training_paths: Any
    run_dir: Path
    seed_snapshot_run_dir: Path
    expected_model_state_dict: dict[str, Any]
    expected_config_canonical: dict[str, Any] | None
    expected_spec_hash256: str | None


@dataclass(frozen=True)
class EnsureNoLeagueBaselineAnchorRequest:
    stack: Any
    training_paths: Any
    run_dir: Path
    learner: Any
    device: torch.device
    config_hash256: str
    spec_hash256: str | None = None
    baseline_run_dir: Path | None = None
    permit_current_run_alias: bool = False
    source_checkpoint_path: Path | None = None
    update: int | None = None


@dataclass(frozen=True)
class LoadSnapshotEvalModelRequest:
    run_dir: Path
    snapshot_path: str
    observation_dim: int
    action_dim: int
    stack: Any
    observation_spec: dict[str, Any] | None = None
    spec_bundle: dict[str, Any] | None = None


__all__ = [
    "BuildCheckpointRecordRequest",
    "BuildHeuristicPublicPolicyRequest",
    "BuildTrainingLearnerRequest",
    "EnsureNoLeagueBaselineAnchorRequest",
    "ImportNoLeagueBaselineAnchorRequest",
    "ImportSeedSnapshotPoolRequest",
    "InitializeLearnerCheckpointRequest",
    "LoadSnapshotEvalModelRequest",
    "PublishCheckpointAliasesRequest",
    "RestoreLearnerCheckpointRequest",
    "SeedSnapshotPolicyIdRequest",
    "StructuredWarmstartRequest",
    "ValidateSeedSnapshotImportContractRequest",
    "WriteCheckpointRequest",
]
