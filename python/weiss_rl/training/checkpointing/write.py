from __future__ import annotations

from typing import Any

import torch

from weiss_rl.training.checkpointing.restore import MINIMAL_TRAIN_CHECKPOINT_FORMAT


def build_minimal_train_checkpoint_payload(
    *,
    update_count: int,
    policy_version: int,
    device: str,
    config_hash256: str,
    spec_hash256: str | None,
    algorithm: str | None,
    recurrent_core: object,
    total_samples_processed: int,
    init_schedule_offset_updates: int,
    model_state_dict: dict[str, Any],
    policy_anchor_model_state_dict: dict[str, Any] | None,
    guidance_payload: dict[str, float],
    optimizer_state_dict: object,
    grad_scaler_state_dict: object,
) -> dict[str, Any]:
    return {
        "format": MINIMAL_TRAIN_CHECKPOINT_FORMAT,
        "update_count": int(update_count),
        "policy_version": int(policy_version),
        "device": str(device),
        "config_hash256": config_hash256,
        "spec_hash256": spec_hash256,
        "algorithm": algorithm,
        "recurrent_core": recurrent_core,
        "total_samples_processed": int(total_samples_processed),
        "init_schedule_offset_updates": int(init_schedule_offset_updates),
        "model_state_dict": model_state_dict,
        "policy_anchor_model_state_dict": policy_anchor_model_state_dict,
        **guidance_payload,
        "optimizer_state_dict": optimizer_state_dict,
        "grad_scaler_state_dict": grad_scaler_state_dict,
    }


def minimal_train_checkpoint_payload_from_learner(
    *,
    learner: Any,
    device: torch.device,
    config_hash256: str,
    spec_hash256: str | None = None,
    algorithm: str | None = None,
    recurrent_core: object = None,
    guidance_payload: dict[str, float] | None = None,
) -> dict[str, Any]:
    if learner.model is None:
        raise RuntimeError("Cannot write a checkpoint without a learner model")
    return build_minimal_train_checkpoint_payload(
        update_count=int(learner.update_count),
        policy_version=int(learner.get_policy_version()),
        device=str(device),
        config_hash256=config_hash256,
        spec_hash256=spec_hash256,
        algorithm=algorithm,
        recurrent_core=recurrent_core,
        total_samples_processed=int(getattr(learner, "total_samples_processed", 0)),
        init_schedule_offset_updates=int(getattr(learner, "init_schedule_offset_updates", 0)),
        model_state_dict=learner.model.state_dict(),
        policy_anchor_model_state_dict=_policy_anchor_model_state_dict(learner),
        guidance_payload={} if guidance_payload is None else guidance_payload,
        optimizer_state_dict=_optimizer_state_dict(learner),
        grad_scaler_state_dict=_grad_scaler_state_dict(learner),
    )


def _policy_anchor_model_state_dict(learner: Any) -> dict[str, Any] | None:
    policy_anchor_state_fn = getattr(learner, "policy_anchor_state_dict", None)
    return None if not callable(policy_anchor_state_fn) else policy_anchor_state_fn()


def _optimizer_state_dict(learner: Any) -> object:
    optimizer = getattr(learner, "optimizer", None)
    return None if optimizer is None else optimizer.state_dict()


def _grad_scaler_state_dict(learner: Any) -> object:
    grad_scaler = getattr(learner, "_grad_scaler", None)
    return None if grad_scaler is None else grad_scaler.state_dict()
