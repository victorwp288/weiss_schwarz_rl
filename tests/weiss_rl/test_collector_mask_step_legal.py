from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from weiss_rl.runtime.components.collection.collector_step_legal import (
    MaskLegalStorage,
    MaskLegalSurface,
    StepLegalCaptureInputs,
    capture_mask_step_legal,
    capture_mask_surface_step_legal,
)

from .collector_step_legal_test_support import teacher_label_arrays


def test_capture_mask_step_legal_copies_mask_and_threads_teacher_inputs() -> None:
    mask = np.asarray(
        [
            [True, False, True],
            [False, True, True],
        ],
        dtype=np.bool_,
    )
    batch = SimpleNamespace(mask=mask, decision_kind=np.asarray([1, 2], dtype=np.int32))
    counters = {"teacher_label_ms": 0}
    mask_steps: list[np.ndarray] = []
    teacher_masks: list[np.ndarray] = []

    def teacher_labels_from_mask(**kwargs):
        teacher_masks.append(np.asarray(kwargs["legal_mask"], dtype=np.bool_).copy())
        return teacher_label_arrays(2)

    captured = capture_mask_step_legal(
        batch=batch,
        focal_rows=np.asarray([True, True], dtype=np.bool_),
        obs_step=np.zeros((2, 3), dtype=np.float32),
        counters=counters,
        teacher_labels_from_mask=teacher_labels_from_mask,
        mask_steps=mask_steps,
    )

    assert len(teacher_masks) == 1
    assert teacher_masks[0].tolist() == mask.tolist()
    assert captured.legal_mask.tolist() == mask.tolist()
    assert captured.reward_legal_mask.tolist() == mask.tolist()
    assert captured.teacher_labels[5].tolist() == [True, True]
    assert len(mask_steps) == 1
    assert mask_steps[0].tolist() == mask.tolist()

    mask[0, 0] = False
    assert captured.reward_legal_mask.tolist() == [[True, False, True], [False, True, True]]
    assert mask_steps[0].tolist() == [[True, False, True], [False, True, True]]


def test_capture_mask_surface_step_legal_is_canonical_mask_storage_and_teacher_path() -> None:
    counters = {"teacher_label_ms": 0}
    mask_steps: list[np.ndarray] = []
    legal_mask = np.asarray([[True, False], [False, True]], dtype=np.bool_)
    teacher_inputs: dict[str, np.ndarray] = {}

    def fake_teacher(**kwargs):
        teacher_inputs["focal_rows"] = np.asarray(kwargs["focal_rows"], dtype=np.bool_).copy()
        teacher_inputs["decision_kind"] = np.asarray(kwargs["decision_kind"], dtype=np.int32).copy()
        teacher_inputs["legal_mask"] = np.asarray(kwargs["legal_mask"], dtype=np.bool_).copy()
        return teacher_label_arrays(2)

    captured = capture_mask_surface_step_legal(
        capture=StepLegalCaptureInputs(
            focal_rows=np.asarray([True, False], dtype=np.bool_),
            obs_step=np.zeros((2, 3), dtype=np.float32),
            counters=counters,
        ),
        surface=MaskLegalSurface(
            legal_mask=legal_mask,
            decision_kind=np.asarray([6, 7], dtype=np.int32),
        ),
        storage=MaskLegalStorage(mask_steps=mask_steps),
        teacher_labels_from_mask=fake_teacher,
    )

    assert teacher_inputs["focal_rows"].tolist() == [True, False]
    assert teacher_inputs["decision_kind"].tolist() == [6, 7]
    assert teacher_inputs["legal_mask"].tolist() == [[True, False], [False, True]]
    assert captured.legal_mask is legal_mask
    assert captured.reward_legal_mask.tolist() == [[True, False], [False, True]]
    assert mask_steps[0].tolist() == [[True, False], [False, True]]

    legal_mask[0, 0] = False
    assert captured.reward_legal_mask.tolist() == [[True, False], [False, True]]
    assert mask_steps[0].tolist() == [[True, False], [False, True]]
