from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from weiss_rl.training.checkpointing.restore import (
    CheckpointPayloadContract,
    validate_checkpoint_payload_contract,
    warn_if_config_hash_mismatch_allowed,
)


@dataclass(frozen=True, slots=True)
class CheckpointContractLoadRequest:
    checkpoint_path: Path
    device: torch.device
    expected_config_hash: str
    expected_spec_hash256: str
    algorithm: str
    allow_config_mismatch: bool = False


def load_checkpoint_payload(
    *,
    checkpoint_path: Path,
    device: torch.device,
) -> object:
    return torch.load(checkpoint_path, map_location=device, weights_only=False)


def load_minimal_train_checkpoint_contract(
    request: CheckpointContractLoadRequest,
) -> CheckpointPayloadContract:
    payload = load_checkpoint_payload(
        checkpoint_path=request.checkpoint_path,
        device=request.device,
    )
    return validate_checkpoint_payload_contract(
        payload,
        checkpoint_path=request.checkpoint_path,
        expected_config_hash=request.expected_config_hash,
        expected_spec_hash256=request.expected_spec_hash256,
        algorithm=request.algorithm,
        allow_config_mismatch=request.allow_config_mismatch,
    )


def load_resume_checkpoint_contract(
    *,
    checkpoint_path: Path,
    device: torch.device,
    expected_config_hash: str,
    expected_spec_hash256: str,
    algorithm: str,
    allow_config_mismatch: bool,
) -> CheckpointPayloadContract:
    contract = load_minimal_train_checkpoint_contract(
        CheckpointContractLoadRequest(
            checkpoint_path=checkpoint_path,
            device=device,
            expected_config_hash=expected_config_hash,
            expected_spec_hash256=expected_spec_hash256,
            algorithm=algorithm,
            allow_config_mismatch=allow_config_mismatch,
        )
    )
    warn_if_config_hash_mismatch_allowed(contract)
    return contract


def load_initialization_checkpoint_contract(
    *,
    checkpoint_path: Path,
    device: torch.device,
    expected_spec_hash256: str,
    algorithm: str,
) -> CheckpointPayloadContract:
    return load_minimal_train_checkpoint_contract(
        CheckpointContractLoadRequest(
            checkpoint_path=checkpoint_path,
            device=device,
            expected_config_hash="",
            expected_spec_hash256=expected_spec_hash256,
            algorithm=algorithm,
            allow_config_mismatch=True,
        )
    )


__all__ = [
    "CheckpointContractLoadRequest",
    "load_checkpoint_payload",
    "load_initialization_checkpoint_contract",
    "load_minimal_train_checkpoint_contract",
    "load_resume_checkpoint_contract",
]
