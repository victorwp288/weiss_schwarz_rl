from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.training.checkpointing.restore_state import (
    CheckpointCounterState,
    apply_checkpoint_resume_counters,
    checkpoint_counter_state_from_payload,
    reset_policy_anchor_to_initialized_model,
    restore_checkpoint_grad_scaler_state,
    restore_checkpoint_model_and_guidance,
    restore_checkpoint_optimizer_state,
    restore_checkpoint_policy_anchor_state,
)

MINIMAL_TRAIN_CHECKPOINT_FORMAT = "minimal_train_checkpoint_v1"


@dataclass(frozen=True, slots=True)
class CheckpointPayloadContract:
    payload: dict[str, Any]
    model_state_dict: dict[str, Any]
    config_hash_mismatch: bool
    expected_config_hash: str
    payload_config_hash: str


@dataclass(frozen=True, slots=True)
class ResumeCheckpoint:
    checkpoint_path: Path
    update_count: int
    policy_version: int
    total_samples_processed: int
    init_schedule_offset_updates: int = 0


def validate_checkpoint_payload_contract(
    payload: object,
    *,
    checkpoint_path: Path,
    expected_config_hash: str,
    expected_spec_hash256: str,
    algorithm: str,
    allow_config_mismatch: bool = False,
) -> CheckpointPayloadContract:
    if not isinstance(payload, dict):
        raise RuntimeError(f"checkpoint payload must be a dict: {checkpoint_path}")
    if str(payload.get("format", "")).strip() != MINIMAL_TRAIN_CHECKPOINT_FORMAT:
        raise RuntimeError(f"unsupported checkpoint format in {checkpoint_path}")
    payload_config_hash = str(payload.get("config_hash256", "")).strip().lower()
    config_hash_mismatch = payload_config_hash != expected_config_hash
    if config_hash_mismatch and not allow_config_mismatch:
        raise RuntimeError(
            f"checkpoint config hash mismatch for {checkpoint_path}: "
            f"expected {expected_config_hash}, got {payload_config_hash}"
        )
    payload_spec_hash = payload.get("spec_hash256")
    if payload_spec_hash is not None and str(payload_spec_hash).strip().lower() != expected_spec_hash256:
        raise RuntimeError(
            f"checkpoint spec hash mismatch for {checkpoint_path}: "
            f"expected {expected_spec_hash256}, got {payload_spec_hash}"
        )
    payload_algorithm = payload.get("algorithm")
    if payload_algorithm is not None and str(payload_algorithm).strip() and str(payload_algorithm).strip() != algorithm:
        raise RuntimeError(
            f"checkpoint algorithm mismatch for {checkpoint_path}: expected {algorithm}, got {payload_algorithm}"
        )
    model_state_dict = payload.get("model_state_dict")
    if not isinstance(model_state_dict, dict):
        raise RuntimeError(f"checkpoint is missing a model_state_dict: {checkpoint_path}")
    return CheckpointPayloadContract(
        payload=payload,
        model_state_dict=model_state_dict,
        config_hash_mismatch=config_hash_mismatch,
        expected_config_hash=expected_config_hash,
        payload_config_hash=payload_config_hash,
    )


def warn_if_config_hash_mismatch_allowed(contract: CheckpointPayloadContract) -> None:
    if not contract.config_hash_mismatch:
        return
    print(
        "Warning: allowing checkpoint config hash mismatch because "
        "WEISS_RL_ALLOW_RESUME_CONFIG_MISMATCH=1: "
        f"expected {contract.expected_config_hash}, got {contract.payload_config_hash}",
        file=sys.stderr,
    )


def apply_minimal_checkpoint_resume_state(
    *,
    checkpoint_path: Path,
    learner: Any,
    contract: CheckpointPayloadContract,
    restore_model_guidance: Any,
    restore_counters: bool,
) -> ResumeCheckpoint:
    payload = contract.payload
    restore_checkpoint_model_and_guidance(
        checkpoint_path=checkpoint_path,
        learner=learner,
        model_state_dict=contract.model_state_dict,
        restore_model_guidance=restore_model_guidance,
        context_kind="resume",
        payload=payload,
    )
    restore_checkpoint_policy_anchor_state(
        learner=learner,
        checkpoint_path=checkpoint_path,
        anchor_state=payload.get("policy_anchor_model_state_dict"),
    )
    restore_checkpoint_optimizer_state(learner=learner, optimizer_state_dict=payload.get("optimizer_state_dict"))
    restore_checkpoint_grad_scaler_state(learner=learner, grad_scaler_state_dict=payload.get("grad_scaler_state_dict"))
    restored_counters = apply_checkpoint_resume_counters(
        learner=learner,
        payload=payload,
        restore_counters=bool(restore_counters),
    )
    return _resume_checkpoint_from_counters(checkpoint_path=checkpoint_path, counters=restored_counters)


def apply_minimal_checkpoint_initialization(
    *,
    checkpoint_path: Path,
    learner: Any,
    contract: CheckpointPayloadContract,
    restore_model_guidance: Any,
) -> ResumeCheckpoint:
    payload = contract.payload
    restore_checkpoint_model_and_guidance(
        checkpoint_path=checkpoint_path,
        learner=learner,
        model_state_dict=contract.model_state_dict,
        restore_model_guidance=restore_model_guidance,
        context_kind="init",
        payload=payload,
    )
    reset_policy_anchor_to_initialized_model(learner)
    source_counters = checkpoint_counter_state_from_payload(payload)
    return _resume_checkpoint_from_counters(checkpoint_path=checkpoint_path, counters=source_counters)


def _resume_checkpoint_from_counters(
    *,
    checkpoint_path: Path,
    counters: CheckpointCounterState,
) -> ResumeCheckpoint:
    return ResumeCheckpoint(
        checkpoint_path=checkpoint_path.resolve(),
        update_count=counters.update_count,
        policy_version=counters.policy_version,
        total_samples_processed=counters.total_samples_processed,
        init_schedule_offset_updates=counters.init_schedule_offset_updates,
    )
