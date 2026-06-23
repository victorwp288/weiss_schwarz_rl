from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from weiss_rl.runtime.components.collector_step_legal import (
    PackedLegalStorage,
    PackedLegalSurface,
    StepLegalCaptureInputs,
    capture_packed_array_step_legal,
    capture_packed_step_legal,
    capture_packed_surface_step_legal,
)

from .collector_step_legal_test_support import teacher_label_arrays


def test_capture_packed_step_legal_copies_storage_and_threads_teacher_inputs() -> None:
    batch = SimpleNamespace(
        ids_offsets=(
            np.asarray([10, 11, 20], dtype=np.uint32),
            np.asarray([0, 2, 3], dtype=np.uint32),
        ),
        legal_action_meta=np.asarray([[1, 0], [2, 0], [3, 0]], dtype=np.uint16),
        decision_kind=np.asarray([1, 9], dtype=np.int32),
    )
    counters = {"teacher_label_ms": 0, "packed_candidate_count": 0}
    packed_ids: list[np.ndarray] = []
    packed_meta: list[np.ndarray] = []
    packed_offsets = [np.asarray([5], dtype=np.uint32)]
    teacher_calls: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]] = []

    def teacher_labels_from_ids(**kwargs):
        teacher_calls.append(
            (
                np.asarray(kwargs["focal_rows"], dtype=np.bool_).copy(),
                np.asarray(kwargs["legal_ids"], dtype=np.uint32).copy(),
                np.asarray(kwargs["legal_offsets"], dtype=np.uint32).copy(),
                None
                if kwargs["legal_action_meta"] is None
                else np.asarray(kwargs["legal_action_meta"], dtype=np.uint16).copy(),
            )
        )
        kwargs["counters"]["teacher_tactical_row_count"] = kwargs["counters"].get("teacher_tactical_row_count", 0) + 2
        return teacher_label_arrays(2)

    captured = capture_packed_step_legal(
        batch=batch,
        focal_rows=np.asarray([True, False], dtype=np.bool_),
        obs_step=np.zeros((2, 3), dtype=np.float32),
        counters=counters,
        ensure_legal_action_meta=lambda _ids, meta: meta,
        teacher_labels_from_ids=teacher_labels_from_ids,
        packed_ids=packed_ids,
        packed_meta=packed_meta,
        packed_offsets=packed_offsets,
    )

    assert len(teacher_calls) == 1
    assert teacher_calls[0][0].tolist() == [True, False]
    assert teacher_calls[0][1].tolist() == [10, 11, 20]
    assert teacher_calls[0][2].tolist() == [0, 2, 3]
    assert teacher_calls[0][3] is not None
    assert counters["packed_candidate_count"] == 3
    assert counters["teacher_tactical_row_count"] == 2
    assert captured.reward_legal_ids.dtype == np.int64
    assert captured.reward_legal_ids.tolist() == [10, 11, 20]
    assert captured.reward_legal_offsets.dtype == np.int64
    assert captured.reward_legal_offsets.tolist() == [0, 2, 3]
    assert captured.reward_legal_meta is not None
    assert captured.teacher_labels[0].tolist() == [0, 1]
    assert packed_ids[0].dtype == np.uint32
    assert packed_ids[0].tolist() == [10, 11, 20]
    assert packed_meta[0].tolist() == [[1, 0], [2, 0], [3, 0]]
    assert packed_offsets[1].tolist() == [7, 8]

    batch.ids_offsets[0][0] = 99
    assert packed_ids[0].tolist() == [10, 11, 20]
    assert captured.reward_legal_ids.tolist() == [10, 11, 20]


def test_capture_packed_array_step_legal_supports_optional_teacher_labels() -> None:
    counters = {"teacher_label_ms": 0, "packed_candidate_count": 0}
    packed_ids: list[np.ndarray] = []
    packed_meta: list[np.ndarray] = []
    packed_offsets = [np.asarray([4], dtype=np.uint32)]
    captured_inputs: dict[str, np.ndarray | None] = {}

    def fake_teacher(**kwargs):
        captured_inputs["decision_kind"] = np.asarray(kwargs["decision_kind"], dtype=np.int32).copy()
        captured_inputs["legal_ids"] = np.asarray(kwargs["legal_ids"], dtype=np.uint32).copy()
        captured_inputs["legal_offsets"] = np.asarray(kwargs["legal_offsets"], dtype=np.uint32).copy()
        captured_inputs["legal_action_meta"] = np.asarray(kwargs["legal_action_meta"], dtype=np.uint16).copy()
        return teacher_label_arrays(2)

    captured = capture_packed_array_step_legal(
        legal_ids=np.asarray([5, 6, 7], dtype=np.int64),
        legal_offsets=np.asarray([0, 2, 3], dtype=np.int64),
        legal_action_meta=np.asarray([[1, 0], [2, 0], [3, 0]], dtype=np.int64),
        decision_kind=np.asarray([10, 11], dtype=np.int64),
        focal_rows=np.asarray([True, False], dtype=np.bool_),
        obs_step=np.zeros((2, 4), dtype=np.float32),
        counters=counters,
        teacher_labels_from_ids=fake_teacher,
        packed_ids=packed_ids,
        packed_meta=packed_meta,
        packed_offsets=packed_offsets,
    )

    assert captured.teacher_labels is not None
    assert counters["packed_candidate_count"] == 3
    assert captured_inputs["decision_kind"].tolist() == [10, 11]
    assert captured_inputs["legal_ids"].tolist() == [5, 6, 7]
    assert captured_inputs["legal_offsets"].tolist() == [0, 2, 3]
    assert captured_inputs["legal_action_meta"] is not None
    assert captured.reward_legal_ids.dtype == np.int64
    assert captured.reward_legal_offsets.dtype == np.int64
    assert captured.reward_legal_meta is not None
    assert packed_ids[0].dtype == np.uint32
    assert packed_ids[0].tolist() == [5, 6, 7]
    assert packed_meta[0].dtype == np.uint16
    assert packed_offsets[1].tolist() == [6, 7]

    captured.reward_legal_ids[0] = 99
    assert packed_ids[0].tolist() == [5, 6, 7]

    no_teacher_counters = {"teacher_label_ms": 0, "packed_candidate_count": 0}
    no_teacher_offsets = [np.asarray([0], dtype=np.uint32)]
    no_teacher = capture_packed_array_step_legal(
        legal_ids=np.asarray([1], dtype=np.uint32),
        legal_offsets=np.asarray([0, 1], dtype=np.uint32),
        legal_action_meta=None,
        decision_kind=np.asarray([0], dtype=np.int32),
        focal_rows=np.asarray([True], dtype=np.bool_),
        obs_step=np.zeros((1, 4), dtype=np.float32),
        counters=no_teacher_counters,
        teacher_labels_from_ids=None,
        packed_ids=[],
        packed_meta=[],
        packed_offsets=no_teacher_offsets,
    )

    assert no_teacher.teacher_labels is None
    assert no_teacher.legal_action_meta is None
    assert no_teacher_counters == {"teacher_label_ms": 0, "packed_candidate_count": 1}
    assert no_teacher_offsets[1].tolist() == [1]


def test_capture_packed_surface_step_legal_is_canonical_storage_and_reward_copy_path() -> None:
    counters = {"teacher_label_ms": 0, "packed_candidate_count": 0}
    packed_ids: list[np.ndarray] = []
    packed_meta: list[np.ndarray] = []
    packed_offsets = [np.asarray([3], dtype=np.uint32)]
    teacher_inputs: dict[str, np.ndarray | None] = {}
    legal_ids = np.asarray([2, 4, 6], dtype=np.uint32)
    legal_offsets = np.asarray([0, 1, 3], dtype=np.uint32)
    legal_meta = np.asarray([[9], [8], [7]], dtype=np.uint16)

    def fake_teacher(**kwargs):
        teacher_inputs["focal_rows"] = np.asarray(kwargs["focal_rows"], dtype=np.bool_).copy()
        teacher_inputs["decision_kind"] = np.asarray(kwargs["decision_kind"], dtype=np.int32).copy()
        teacher_inputs["legal_ids"] = np.asarray(kwargs["legal_ids"], dtype=np.uint32).copy()
        teacher_inputs["legal_offsets"] = np.asarray(kwargs["legal_offsets"], dtype=np.uint32).copy()
        teacher_inputs["legal_action_meta"] = np.asarray(kwargs["legal_action_meta"], dtype=np.uint16).copy()
        return teacher_label_arrays(2)

    captured = capture_packed_surface_step_legal(
        capture=StepLegalCaptureInputs(
            focal_rows=np.asarray([True, False], dtype=np.bool_),
            obs_step=np.zeros((2, 4), dtype=np.float32),
            counters=counters,
        ),
        surface=PackedLegalSurface(
            legal_ids=legal_ids,
            legal_offsets=legal_offsets,
            legal_action_meta=legal_meta,
            decision_kind=np.asarray([4, 5], dtype=np.int32),
        ),
        storage=PackedLegalStorage(
            packed_ids=packed_ids,
            packed_meta=packed_meta,
            packed_offsets=packed_offsets,
        ),
        teacher_labels_from_ids=fake_teacher,
    )

    assert counters["packed_candidate_count"] == 3
    assert captured.teacher_labels is not None
    assert teacher_inputs["focal_rows"].tolist() == [True, False]
    assert teacher_inputs["decision_kind"].tolist() == [4, 5]
    assert teacher_inputs["legal_ids"].tolist() == [2, 4, 6]
    assert teacher_inputs["legal_offsets"].tolist() == [0, 1, 3]
    assert teacher_inputs["legal_action_meta"] is not None
    assert captured.reward_legal_ids.dtype == np.int64
    assert captured.reward_legal_offsets.dtype == np.int64
    assert captured.reward_legal_meta is not None
    assert packed_ids[0].dtype == np.uint32
    assert packed_ids[0].tolist() == [2, 4, 6]
    assert packed_meta[0].tolist() == [[9], [8], [7]]
    assert packed_offsets[1].tolist() == [4, 6]

    legal_ids[0] = 99
    legal_offsets[1] = 99
    legal_meta[0, 0] = 99
    assert captured.reward_legal_ids.tolist() == [2, 4, 6]
    assert captured.reward_legal_offsets.tolist() == [0, 1, 3]
    assert packed_ids[0].tolist() == [2, 4, 6]
    assert packed_offsets[1].tolist() == [4, 6]
    assert packed_meta[0].tolist() == [[9], [8], [7]]
