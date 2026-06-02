from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.runtime.components.legal_batching import concatenate_legal_actions


@dataclass(frozen=True, slots=True)
class RequiredTimeMajorFieldSpec:
    field_name: str


@dataclass(frozen=True, slots=True)
class OptionalTimeMajorFieldSpec:
    field_name: str
    missing_fill_value: Any


@dataclass(frozen=True, slots=True)
class RuntimeFieldAssemblySpec:
    action_dim: int
    required_time_major_fields: tuple[RequiredTimeMajorFieldSpec, ...] = (
        RequiredTimeMajorFieldSpec("obs"),
        RequiredTimeMajorFieldSpec("actions"),
        RequiredTimeMajorFieldSpec("rewards"),
        RequiredTimeMajorFieldSpec("terminated"),
        RequiredTimeMajorFieldSpec("truncated"),
        RequiredTimeMajorFieldSpec("to_play_seat"),
        RequiredTimeMajorFieldSpec("behavior_logp"),
        RequiredTimeMajorFieldSpec("values"),
        RequiredTimeMajorFieldSpec("policy_train_mask"),
    )
    optional_time_major_fields: tuple[OptionalTimeMajorFieldSpec, ...] = (
        OptionalTimeMajorFieldSpec("opponent_context_index", 0),
        OptionalTimeMajorFieldSpec("teacher_family", -1),
        OptionalTimeMajorFieldSpec("teacher_slot", -1),
        OptionalTimeMajorFieldSpec("teacher_move_source", -1),
        OptionalTimeMajorFieldSpec("teacher_attack_type", -1),
        OptionalTimeMajorFieldSpec("teacher_action", -1),
        OptionalTimeMajorFieldSpec("teacher_valid", False),
        OptionalTimeMajorFieldSpec("trajectory_retention_valid", False),
    )


@dataclass(frozen=True, slots=True)
class RuntimeBatchFields:
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    to_play_seat: np.ndarray
    behavior_logp: np.ndarray
    values: np.ndarray
    initial_hidden_state: np.ndarray
    policy_train_mask: np.ndarray
    opponent_context_index: np.ndarray | None
    teacher_family: np.ndarray | None
    teacher_slot: np.ndarray | None
    teacher_move_source: np.ndarray | None
    teacher_attack_type: np.ndarray | None
    teacher_action: np.ndarray | None
    teacher_valid: np.ndarray | None
    trajectory_retention_valid: np.ndarray | None
    legal_actions: LegalActionBatch
    legal_mask: np.ndarray | None


@dataclass(frozen=True, slots=True)
class TimeMajorBatchLayout:
    field_name: str
    time_steps: int
    batch_offsets: tuple[int, ...]
    total_batch: int

    def bounds_for_index(self, index: int) -> tuple[int, int]:
        return self.batch_offsets[index], self.batch_offsets[index + 1]


@dataclass(frozen=True, slots=True)
class BatchMajorLayout:
    field_name: str
    batch_offsets: tuple[int, ...]
    total_batch: int

    def bounds_for_index(self, index: int) -> tuple[int, int]:
        return self.batch_offsets[index], self.batch_offsets[index + 1]


def time_major_batch_layout(unrolls: Sequence[Any], field_name: str) -> TimeMajorBatchLayout:
    if not unrolls:
        raise ValueError("unrolls must be non-empty")
    template = np.asarray(getattr(unrolls[0], field_name))
    offsets = [0]
    for unroll in unrolls:
        value = np.asarray(getattr(unroll, field_name))
        offsets.append(offsets[-1] + int(value.shape[1]))
    return TimeMajorBatchLayout(
        field_name=str(field_name),
        time_steps=int(template.shape[0]),
        batch_offsets=tuple(offsets),
        total_batch=int(offsets[-1]),
    )


def batch_major_layout(unrolls: Sequence[Any], field_name: str) -> BatchMajorLayout:
    if not unrolls:
        raise ValueError("unrolls must be non-empty")
    offsets = [0]
    for unroll in unrolls:
        value = np.asarray(getattr(unroll, field_name))
        offsets.append(offsets[-1] + int(value.shape[0]))
    return BatchMajorLayout(
        field_name=str(field_name),
        batch_offsets=tuple(offsets),
        total_batch=int(offsets[-1]),
    )


def _require_time_major_shape(
    value: np.ndarray,
    *,
    field_name: str,
    time_steps: int,
    batch_width: int,
) -> None:
    if value.shape[0] != time_steps or value.shape[1] != batch_width:
        raise ValueError(f"{field_name} must have leading shape ({time_steps}, {batch_width}), got {value.shape[:2]}")


def concat_time_major_field_with_layout(
    unrolls: Sequence[Any],
    field_name: str,
    *,
    layout: TimeMajorBatchLayout,
) -> np.ndarray:
    if not unrolls:
        raise ValueError("unrolls must be non-empty")
    template = np.asarray(getattr(unrolls[0], field_name))
    result = np.empty((template.shape[0], layout.total_batch, *template.shape[2:]), dtype=template.dtype)
    for index, unroll in enumerate(unrolls):
        start, stop = layout.bounds_for_index(index)
        value = np.asarray(getattr(unroll, field_name))
        _require_time_major_shape(
            value,
            field_name=field_name,
            time_steps=int(template.shape[0]),
            batch_width=stop - start,
        )
        result[:, start:stop, ...] = value
    return result


def concat_optional_time_major_field(
    unrolls: Sequence[Any],
    field_name: str,
    *,
    missing_fill_value: Any,
    layout: TimeMajorBatchLayout | None = None,
) -> np.ndarray | None:
    present_values = [
        getattr(unroll, field_name, None) for unroll in unrolls if getattr(unroll, field_name, None) is not None
    ]
    if not present_values:
        return None
    effective_layout = time_major_batch_layout(unrolls, "obs") if layout is None else layout
    template = np.asarray(present_values[0])
    result = np.empty((template.shape[0], effective_layout.total_batch, *template.shape[2:]), dtype=template.dtype)
    for index, unroll in enumerate(unrolls):
        start, stop = effective_layout.bounds_for_index(index)
        raw_value = getattr(unroll, field_name, None)
        if raw_value is None:
            value = np.full(
                (template.shape[0], stop - start, *template.shape[2:]),
                missing_fill_value,
                dtype=template.dtype,
            )
        else:
            value = np.asarray(raw_value, dtype=template.dtype)
            _require_time_major_shape(
                value,
                field_name=field_name,
                time_steps=int(template.shape[0]),
                batch_width=stop - start,
            )
        result[:, start:stop, ...] = value
    return result


def concat_time_major_field(unrolls: Sequence[Any], field_name: str) -> np.ndarray:
    if not unrolls:
        raise ValueError("unrolls must be non-empty")
    layout = time_major_batch_layout(unrolls, field_name)
    return concat_time_major_field_with_layout(unrolls, field_name, layout=layout)


def concat_batch_major_field_with_layout(
    unrolls: Sequence[Any],
    field_name: str,
    *,
    layout: BatchMajorLayout,
) -> np.ndarray:
    if not unrolls:
        raise ValueError("unrolls must be non-empty")
    template = np.asarray(getattr(unrolls[0], field_name))
    result = np.empty((layout.total_batch, *template.shape[1:]), dtype=template.dtype)
    for index, unroll in enumerate(unrolls):
        start, stop = layout.bounds_for_index(index)
        value = np.asarray(getattr(unroll, field_name))
        if value.shape[0] != stop - start:
            raise ValueError(f"{field_name} must have batch width {stop - start}, got {value.shape[0]}")
        result[start:stop, ...] = value
    return result


def concat_batch_major_field(unrolls: Sequence[Any], field_name: str) -> np.ndarray:
    if not unrolls:
        raise ValueError("unrolls must be non-empty")
    layout = batch_major_layout(unrolls, field_name)
    return concat_batch_major_field_with_layout(unrolls, field_name, layout=layout)


def concat_required_time_major_fields(
    unrolls: Sequence[Any],
    specs: Sequence[RequiredTimeMajorFieldSpec],
) -> dict[str, np.ndarray]:
    if not specs:
        return {}
    layout = time_major_batch_layout(unrolls, specs[0].field_name)
    return {
        spec.field_name: concat_time_major_field_with_layout(unrolls, spec.field_name, layout=layout) for spec in specs
    }


def concat_optional_time_major_fields(
    unrolls: Sequence[Any],
    specs: Sequence[OptionalTimeMajorFieldSpec],
) -> dict[str, np.ndarray | None]:
    layout = time_major_batch_layout(unrolls, "obs") if specs else None
    return {
        spec.field_name: concat_optional_time_major_field(
            unrolls,
            spec.field_name,
            missing_fill_value=spec.missing_fill_value,
            layout=layout,
        )
        for spec in specs
    }


def concat_runtime_batch_fields(
    unrolls: Sequence[Any],
    *,
    action_dim: int,
    record_batch_timer_ms: Callable[[str, float], None] | None,
) -> RuntimeBatchFields:
    spec = RuntimeFieldAssemblySpec(action_dim=int(action_dim))
    concat_started = time.perf_counter()
    required = concat_required_time_major_fields(unrolls, spec.required_time_major_fields)
    initial_hidden_state = concat_batch_major_field_with_layout(
        unrolls,
        "initial_hidden_state",
        layout=batch_major_layout(unrolls, "initial_hidden_state"),
    )
    optional = concat_optional_time_major_fields(unrolls, spec.optional_time_major_fields)
    legal_actions = concatenate_legal_actions(unrolls, action_space=spec.action_dim)
    if record_batch_timer_ms is not None:
        record_batch_timer_ms("legal_concatenation", time.perf_counter() - concat_started)
    return RuntimeBatchFields(
        obs=required["obs"],
        actions=required["actions"],
        rewards=required["rewards"],
        terminated=required["terminated"],
        truncated=required["truncated"],
        to_play_seat=required["to_play_seat"],
        behavior_logp=required["behavior_logp"],
        values=required["values"],
        initial_hidden_state=initial_hidden_state,
        policy_train_mask=required["policy_train_mask"],
        opponent_context_index=optional["opponent_context_index"],
        teacher_family=optional["teacher_family"],
        teacher_slot=optional["teacher_slot"],
        teacher_move_source=optional["teacher_move_source"],
        teacher_attack_type=optional["teacher_attack_type"],
        teacher_action=optional["teacher_action"],
        teacher_valid=optional["teacher_valid"],
        trajectory_retention_valid=optional["trajectory_retention_valid"],
        legal_actions=legal_actions,
        legal_mask=None if legal_actions.mask is None else legal_actions.mask,
    )


def base_runtime_learner_payload(
    *,
    fields: RuntimeBatchFields,
    rewards: np.ndarray,
    discounts: np.ndarray,
    reset_before_step: np.ndarray,
) -> dict[str, Any]:
    return {
        "obs": fields.obs,
        "actions": fields.actions,
        "legal_actions": fields.legal_actions,
        "legal_mask": fields.legal_mask,
        "legal_action_meta": fields.legal_actions.meta,
        "to_play_seat": fields.to_play_seat,
        "actor": fields.to_play_seat,
        "initial_hidden_state": fields.initial_hidden_state,
        "rewards": rewards,
        "discounts": discounts,
        "reset_before_step": reset_before_step,
        "policy_train_mask": fields.policy_train_mask,
        "opponent_context_index": fields.opponent_context_index,
        "teacher_family": fields.teacher_family,
        "teacher_slot": fields.teacher_slot,
        "teacher_move_source": fields.teacher_move_source,
        "teacher_attack_type": fields.teacher_attack_type,
        "teacher_action": fields.teacher_action,
        "teacher_valid": fields.teacher_valid,
        "trajectory_retention_valid": fields.trajectory_retention_valid,
    }
