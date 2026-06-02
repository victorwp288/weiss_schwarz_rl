from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from weiss_rl.runtime.components.legal_batching import (
    optional_legal_action_meta,
    require_ids_offsets,
    require_mask,
)
from weiss_rl.runtime.components.teacher_labels import TeacherLabelArrays


class PackedStepLegal(NamedTuple):
    legal_ids: np.ndarray
    legal_offsets: np.ndarray
    legal_action_meta: np.ndarray | None
    reward_legal_ids: np.ndarray
    reward_legal_offsets: np.ndarray
    reward_legal_meta: np.ndarray | None
    teacher_labels: TeacherLabelArrays


class MaskStepLegal(NamedTuple):
    legal_mask: np.ndarray
    reward_legal_mask: np.ndarray
    teacher_labels: TeacherLabelArrays


class PackedArrayStepLegal(NamedTuple):
    legal_ids: np.ndarray
    legal_offsets: np.ndarray
    legal_action_meta: np.ndarray | None
    reward_legal_ids: np.ndarray
    reward_legal_offsets: np.ndarray
    reward_legal_meta: np.ndarray | None
    teacher_labels: TeacherLabelArrays | None


@dataclass(frozen=True, slots=True)
class StepLegalCaptureInputs:
    focal_rows: np.ndarray
    obs_step: np.ndarray
    counters: dict[str, int]


@dataclass(frozen=True, slots=True)
class PackedLegalSurface:
    legal_ids: np.ndarray
    legal_offsets: np.ndarray
    legal_action_meta: np.ndarray | None
    decision_kind: np.ndarray


@dataclass(frozen=True, slots=True)
class PackedLegalStorage:
    packed_ids: list[np.ndarray]
    packed_meta: list[np.ndarray]
    packed_offsets: list[np.ndarray]


@dataclass(frozen=True, slots=True)
class MaskLegalSurface:
    legal_mask: np.ndarray
    decision_kind: np.ndarray


@dataclass(frozen=True, slots=True)
class MaskLegalStorage:
    mask_steps: list[np.ndarray]


def _timed_teacher_labels_from_ids(
    *,
    capture: StepLegalCaptureInputs,
    surface: PackedLegalSurface,
    teacher_labels_from_ids: Callable[..., TeacherLabelArrays] | None,
) -> TeacherLabelArrays | None:
    if teacher_labels_from_ids is None:
        return None
    teacher_started = time.perf_counter()
    teacher_labels = teacher_labels_from_ids(
        focal_rows=capture.focal_rows,
        decision_kind=surface.decision_kind,
        obs_step=capture.obs_step,
        legal_ids=surface.legal_ids,
        legal_offsets=surface.legal_offsets,
        legal_action_meta=surface.legal_action_meta,
        counters=capture.counters,
    )
    capture.counters["teacher_label_ms"] += int((time.perf_counter() - teacher_started) * 1000.0)
    return teacher_labels


def _timed_teacher_labels_from_mask(
    *,
    capture: StepLegalCaptureInputs,
    surface: MaskLegalSurface,
    teacher_labels_from_mask: Callable[..., TeacherLabelArrays],
    reward_legal_mask: np.ndarray,
) -> TeacherLabelArrays:
    teacher_started = time.perf_counter()
    teacher_labels = teacher_labels_from_mask(
        focal_rows=capture.focal_rows,
        decision_kind=surface.decision_kind,
        obs_step=capture.obs_step,
        legal_mask=reward_legal_mask,
        counters=capture.counters,
    )
    capture.counters["teacher_label_ms"] += int((time.perf_counter() - teacher_started) * 1000.0)
    return teacher_labels


def _append_packed_legal_storage(
    *,
    surface: PackedLegalSurface,
    storage: PackedLegalStorage,
) -> None:
    offset_base = int(storage.packed_offsets[-1][-1])
    storage.packed_ids.append(np.array(surface.legal_ids, dtype=np.uint32, copy=True))
    if surface.legal_action_meta is not None:
        storage.packed_meta.append(np.array(surface.legal_action_meta, dtype=np.uint16, copy=True))
    storage.packed_offsets.append(np.array(surface.legal_offsets[1:] + offset_base, dtype=np.uint32, copy=True))


def capture_packed_surface_step_legal(
    *,
    capture: StepLegalCaptureInputs,
    surface: PackedLegalSurface,
    storage: PackedLegalStorage,
    teacher_labels_from_ids: Callable[..., TeacherLabelArrays] | None,
) -> PackedArrayStepLegal:
    teacher_labels = _timed_teacher_labels_from_ids(
        capture=capture,
        surface=surface,
        teacher_labels_from_ids=teacher_labels_from_ids,
    )
    capture.counters["packed_candidate_count"] += int(surface.legal_ids.shape[0])

    reward_legal_ids = np.array(surface.legal_ids, dtype=np.int64, copy=True)
    reward_legal_offsets = np.array(surface.legal_offsets, dtype=np.int64, copy=True)
    reward_legal_meta = (
        None if surface.legal_action_meta is None else np.asarray(surface.legal_action_meta, dtype=np.uint16)
    )
    _append_packed_legal_storage(surface=surface, storage=storage)
    return PackedArrayStepLegal(
        legal_ids=surface.legal_ids,
        legal_offsets=surface.legal_offsets,
        legal_action_meta=surface.legal_action_meta,
        reward_legal_ids=reward_legal_ids,
        reward_legal_offsets=reward_legal_offsets,
        reward_legal_meta=reward_legal_meta,
        teacher_labels=teacher_labels,
    )


def capture_mask_surface_step_legal(
    *,
    capture: StepLegalCaptureInputs,
    surface: MaskLegalSurface,
    storage: MaskLegalStorage,
    teacher_labels_from_mask: Callable[..., TeacherLabelArrays],
) -> MaskStepLegal:
    reward_legal_mask = np.array(surface.legal_mask, dtype=np.bool_, copy=True)
    teacher_labels = _timed_teacher_labels_from_mask(
        capture=capture,
        surface=surface,
        teacher_labels_from_mask=teacher_labels_from_mask,
        reward_legal_mask=reward_legal_mask,
    )
    storage.mask_steps.append(reward_legal_mask)
    return MaskStepLegal(
        legal_mask=surface.legal_mask,
        reward_legal_mask=reward_legal_mask,
        teacher_labels=teacher_labels,
    )


def capture_packed_step_legal(
    *,
    batch: Any,
    focal_rows: np.ndarray,
    obs_step: np.ndarray,
    counters: dict[str, int],
    ensure_legal_action_meta: Callable[[np.ndarray, np.ndarray | None], np.ndarray | None],
    teacher_labels_from_ids: Callable[..., TeacherLabelArrays],
    packed_ids: list[np.ndarray],
    packed_meta: list[np.ndarray],
    packed_offsets: list[np.ndarray],
) -> PackedStepLegal:
    legal_ids, legal_offsets = require_ids_offsets(batch)
    legal_action_meta = ensure_legal_action_meta(legal_ids, optional_legal_action_meta(batch))
    captured = capture_packed_surface_step_legal(
        capture=StepLegalCaptureInputs(
            focal_rows=focal_rows,
            obs_step=obs_step,
            counters=counters,
        ),
        surface=PackedLegalSurface(
            legal_ids=np.asarray(legal_ids, dtype=np.uint32),
            legal_offsets=np.asarray(legal_offsets, dtype=np.uint32),
            legal_action_meta=None if legal_action_meta is None else np.asarray(legal_action_meta, dtype=np.uint16),
            decision_kind=np.asarray(batch.decision_kind, dtype=np.int32),
        ),
        storage=PackedLegalStorage(
            packed_ids=packed_ids,
            packed_meta=packed_meta,
            packed_offsets=packed_offsets,
        ),
        teacher_labels_from_ids=teacher_labels_from_ids,
    )
    assert captured.teacher_labels is not None
    return PackedStepLegal(
        legal_ids=captured.legal_ids,
        legal_offsets=captured.legal_offsets,
        legal_action_meta=captured.legal_action_meta,
        reward_legal_ids=captured.reward_legal_ids,
        reward_legal_offsets=captured.reward_legal_offsets,
        reward_legal_meta=captured.reward_legal_meta,
        teacher_labels=captured.teacher_labels,
    )


def capture_packed_array_step_legal(
    *,
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    legal_action_meta: np.ndarray | None,
    decision_kind: np.ndarray,
    focal_rows: np.ndarray,
    obs_step: np.ndarray,
    counters: dict[str, int],
    teacher_labels_from_ids: Callable[..., TeacherLabelArrays] | None,
    packed_ids: list[np.ndarray],
    packed_meta: list[np.ndarray],
    packed_offsets: list[np.ndarray],
) -> PackedArrayStepLegal:
    return capture_packed_surface_step_legal(
        capture=StepLegalCaptureInputs(
            focal_rows=focal_rows,
            obs_step=obs_step,
            counters=counters,
        ),
        surface=PackedLegalSurface(
            legal_ids=np.asarray(legal_ids, dtype=np.uint32),
            legal_offsets=np.asarray(legal_offsets, dtype=np.uint32),
            legal_action_meta=None if legal_action_meta is None else np.asarray(legal_action_meta, dtype=np.uint16),
            decision_kind=np.asarray(decision_kind, dtype=np.int32),
        ),
        storage=PackedLegalStorage(
            packed_ids=packed_ids,
            packed_meta=packed_meta,
            packed_offsets=packed_offsets,
        ),
        teacher_labels_from_ids=teacher_labels_from_ids,
    )


def capture_mask_step_legal(
    *,
    batch: Any,
    focal_rows: np.ndarray,
    obs_step: np.ndarray,
    counters: dict[str, int],
    teacher_labels_from_mask: Callable[..., TeacherLabelArrays],
    mask_steps: list[np.ndarray],
) -> MaskStepLegal:
    legal_mask = require_mask(batch)
    return capture_mask_surface_step_legal(
        capture=StepLegalCaptureInputs(
            focal_rows=focal_rows,
            obs_step=obs_step,
            counters=counters,
        ),
        surface=MaskLegalSurface(
            legal_mask=np.asarray(legal_mask, dtype=np.bool_),
            decision_kind=np.asarray(batch.decision_kind, dtype=np.int32),
        ),
        storage=MaskLegalStorage(mask_steps=mask_steps),
        teacher_labels_from_mask=teacher_labels_from_mask,
    )
