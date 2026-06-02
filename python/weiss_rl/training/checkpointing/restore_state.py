from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from weiss_rl.models.state_dict_compat import load_model_state_dict_with_context_compat


@dataclass(frozen=True, slots=True)
class CheckpointCounterState:
    update_count: int
    policy_version: int
    total_samples_processed: int
    init_schedule_offset_updates: int


def checkpoint_counter_state_from_payload(payload: Mapping[str, Any]) -> CheckpointCounterState:
    return CheckpointCounterState(
        update_count=int(payload.get("update_count", 0)),
        policy_version=int(payload.get("policy_version", 0)),
        total_samples_processed=int(payload.get("total_samples_processed", 0)),
        init_schedule_offset_updates=int(payload.get("init_schedule_offset_updates", 0)),
    )


def checkpoint_counter_state_from_learner(
    learner: Any,
    *,
    init_schedule_offset_updates: int,
) -> CheckpointCounterState:
    return CheckpointCounterState(
        update_count=int(learner.update_count),
        policy_version=int(learner.policy_version),
        total_samples_processed=int(learner.total_samples_processed),
        init_schedule_offset_updates=int(init_schedule_offset_updates),
    )


def restore_checkpoint_model_and_guidance(
    *,
    checkpoint_path: Path,
    learner: Any,
    model_state_dict: dict[str, Any],
    restore_model_guidance: Any,
    context_kind: str,
    payload: Mapping[str, Any],
) -> None:
    if learner.model is None:
        raise RuntimeError(f"checkpoint is missing a model_state_dict: {checkpoint_path}")
    load_model_state_dict_with_context_compat(
        learner.model,
        model_state_dict,
        context=f"checkpoint {context_kind} {checkpoint_path}",
    )
    restore_model_guidance(learner.model, payload)


def restore_checkpoint_policy_anchor_state(
    *,
    learner: Any,
    checkpoint_path: Path,
    anchor_state: object,
) -> None:
    load_policy_anchor_state_fn = getattr(learner, "load_policy_anchor_state_dict", None)
    if not callable(load_policy_anchor_state_fn):
        return
    if anchor_state is not None and not isinstance(anchor_state, Mapping):
        raise RuntimeError(f"checkpoint policy_anchor_model_state_dict must be a dict: {checkpoint_path}")
    load_policy_anchor_state_fn(cast(Mapping[str, Any] | None, anchor_state))


def restore_checkpoint_optimizer_state(*, learner: Any, optimizer_state_dict: object) -> None:
    if optimizer_state_dict is None:
        return
    optimizer = learner._optimizer_for_step()
    optimizer.load_state_dict(optimizer_state_dict)


def restore_checkpoint_grad_scaler_state(*, learner: Any, grad_scaler_state_dict: object) -> None:
    grad_scaler = getattr(learner, "_grad_scaler", None)
    if grad_scaler_state_dict is not None and grad_scaler is not None:
        grad_scaler.load_state_dict(grad_scaler_state_dict)


def apply_checkpoint_resume_counters(
    *,
    learner: Any,
    payload: Mapping[str, Any],
    restore_counters: bool,
) -> CheckpointCounterState:
    payload_counters = checkpoint_counter_state_from_payload(payload)
    if restore_counters:
        learner.update_count = payload_counters.update_count
        learner.policy_version = payload_counters.policy_version
        learner.total_samples_processed = payload_counters.total_samples_processed
        learner.start_time = time.time()
    learner.init_schedule_offset_updates = payload_counters.init_schedule_offset_updates
    if restore_counters:
        return payload_counters
    return checkpoint_counter_state_from_learner(
        learner,
        init_schedule_offset_updates=payload_counters.init_schedule_offset_updates,
    )


def reset_policy_anchor_to_initialized_model(learner: Any) -> None:
    reset_policy_anchor_fn = getattr(learner, "reset_policy_anchor_to_current_model", None)
    if callable(reset_policy_anchor_fn):
        reset_policy_anchor_fn()
        return
    load_policy_anchor_state_fn = getattr(learner, "load_policy_anchor_state_dict", None)
    if callable(load_policy_anchor_state_fn):
        load_policy_anchor_state_fn(None)


__all__ = [
    "CheckpointCounterState",
    "apply_checkpoint_resume_counters",
    "checkpoint_counter_state_from_learner",
    "checkpoint_counter_state_from_payload",
    "reset_policy_anchor_to_initialized_model",
    "restore_checkpoint_grad_scaler_state",
    "restore_checkpoint_model_and_guidance",
    "restore_checkpoint_optimizer_state",
    "restore_checkpoint_policy_anchor_state",
]
