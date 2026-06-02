from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import torch

from weiss_rl.training.checkpointing.aliases import LearnerRecordSource
from weiss_rl.training.checkpointing.load import (
    load_initialization_checkpoint_contract,
    load_resume_checkpoint_contract,
)
from weiss_rl.training.checkpointing.restore import (
    ResumeCheckpoint,
    apply_minimal_checkpoint_initialization,
    apply_minimal_checkpoint_resume_state,
)
from weiss_rl.training.checkpointing.write import minimal_train_checkpoint_payload_from_learner


class CheckpointWritePaths(Protocol):
    @property
    def checkpoints_dir(self) -> Path: ...


class CheckpointLearner(Protocol):
    update_count: int
    policy_version: int
    total_samples_processed: int
    start_time: float
    model: Any
    optimizer: Any
    _grad_scaler: Any

    def get_policy_version(self) -> int: ...

    def _optimizer_for_step(self) -> Any: ...

    def policy_anchor_state_dict(self) -> dict[str, Any] | None: ...

    def load_policy_anchor_state_dict(self, state_dict: Mapping[str, Any] | None) -> None: ...

    def reset_policy_anchor_to_current_model(self) -> None: ...


def write_scalars_record(
    *,
    scalars_path: Path,
    learner: LearnerRecordSource,
    metrics: dict[str, float],
    start_time: float,
) -> dict[str, Any]:
    wall_clock_seconds = time.time() - start_time
    record = {
        "update_count": int(learner.update_count),
        "policy_version": int(learner.get_policy_version()),
        "wall_clock_seconds": wall_clock_seconds,
        "wall_clock_ms": int(wall_clock_seconds * 1000),
        **metrics,
    }
    with scalars_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def current_focal_policy_id(*, learner: LearnerRecordSource) -> str:
    return f"train_u{int(learner.update_count)}_p{int(learner.get_policy_version())}"


def checkpoint_path_for_update(checkpoints_dir: Path, *, update_count: int) -> Path:
    return checkpoints_dir / f"checkpoint_{update_count}.pt"


def ensure_current_checkpoint(
    *,
    training_paths: CheckpointWritePaths,
    learner: LearnerRecordSource,
    write_checkpoint: Any,
) -> Path:
    checkpoint_path = checkpoint_path_for_update(
        training_paths.checkpoints_dir,
        update_count=int(learner.update_count),
    )
    if checkpoint_path.is_file():
        return checkpoint_path

    write_checkpoint(checkpoint_path)
    return checkpoint_path


def write_minimal_train_checkpoint(
    *,
    checkpoint_path: Path,
    learner: CheckpointLearner,
    device: torch.device,
    config_hash256: str,
    spec_hash256: str | None = None,
    algorithm: str | None = None,
    recurrent_core: object = None,
    guidance_payload: dict[str, float] | None = None,
) -> dict[str, Any]:
    payload = minimal_train_checkpoint_payload_from_learner(
        learner=learner,
        device=device,
        config_hash256=config_hash256,
        spec_hash256=spec_hash256,
        algorithm=algorithm,
        recurrent_core=recurrent_core,
        guidance_payload=guidance_payload,
    )
    torch.save(payload, checkpoint_path)
    return payload


def restore_minimal_train_checkpoint(
    *,
    checkpoint_path: Path,
    learner: CheckpointLearner,
    device: torch.device,
    expected_config_hash: str,
    expected_spec_hash256: str,
    algorithm: str,
    restore_model_guidance: Any,
    allow_config_mismatch: bool = False,
    restore_counters: bool = True,
) -> ResumeCheckpoint:
    contract = load_resume_checkpoint_contract(
        checkpoint_path=checkpoint_path,
        device=device,
        expected_config_hash=expected_config_hash,
        expected_spec_hash256=expected_spec_hash256,
        algorithm=algorithm,
        allow_config_mismatch=allow_config_mismatch,
    )
    return apply_minimal_checkpoint_resume_state(
        checkpoint_path=checkpoint_path,
        learner=learner,
        contract=contract,
        restore_model_guidance=restore_model_guidance,
        restore_counters=bool(restore_counters),
    )


def initialize_model_from_checkpoint(
    *,
    checkpoint_path: Path,
    learner: CheckpointLearner,
    device: torch.device,
    expected_spec_hash256: str,
    algorithm: str,
    restore_model_guidance: Any,
) -> ResumeCheckpoint:
    """Load model weights/guidance from a checkpoint without resuming counters or optimizer state."""

    contract = load_initialization_checkpoint_contract(
        checkpoint_path=checkpoint_path,
        device=device,
        expected_spec_hash256=expected_spec_hash256,
        algorithm=algorithm,
    )
    return apply_minimal_checkpoint_initialization(
        checkpoint_path=checkpoint_path,
        learner=learner,
        contract=contract,
        restore_model_guidance=restore_model_guidance,
    )
