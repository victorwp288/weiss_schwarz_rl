from __future__ import annotations

import numpy as np
from weiss_rl.runtime.components.teacher_labels import (
    PUBLIC_TEACHER_DECISION_KINDS,
    teacher_labels_from_ids,
    teacher_labels_from_mask,
)

from .runtime_test_support import _teacher_test_catalog


def test_teacher_labels_from_ids_routes_public_decision_kinds_and_updates_counters() -> None:
    action_catalog = _teacher_test_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    attack_type_index = {name: index for index, name in enumerate(action_catalog.attack_type_names)}
    calls: list[dict[str, object]] = []

    def select_actions_from_ids(**kwargs: object) -> np.ndarray:
        calls.append(dict(kwargs))
        legal_ids = np.asarray(kwargs["legal_ids"], dtype=np.uint32)
        legal_offsets = np.asarray(kwargs["legal_offsets"], dtype=np.int64)
        return np.asarray(
            [int(legal_ids[int(legal_offsets[row_index])]) for row_index in np.asarray(kwargs["row_indices"]).tolist()],
            dtype=np.int64,
        )

    counters = {"teacher_tactical_row_count": 0}
    labels = teacher_labels_from_ids(
        focal_rows=np.asarray([True, True, True, False, True], dtype=np.bool_),
        decision_kind=np.asarray([1, 5, 8, 1, 0], dtype=np.int32),
        obs_step=np.zeros((5, 4), dtype=np.float32),
        legal_ids=np.asarray([0, 40, 14, 40, 39, 40, 40, 40, 40, 40], dtype=np.uint32),
        legal_offsets=np.asarray([0, 2, 4, 6, 8, 10], dtype=np.uint32),
        legal_action_meta=None,
        counters=counters,
        guidance_active=True,
        teacher_policy="teacher",
        action_catalog=action_catalog,
        family_index=family_index,
        attack_type_index=attack_type_index,
        select_actions_from_ids=select_actions_from_ids,
    )

    teacher_family, teacher_slot, _teacher_move_source, teacher_attack_type, teacher_action, teacher_valid = labels
    assert PUBLIC_TEACHER_DECISION_KINDS == frozenset({1, 2, 3, 4, 5, 6, 7, 8})
    assert counters["teacher_tactical_row_count"] == 3
    assert len(calls) == 1
    assert np.asarray(calls[0]["row_indices"]).tolist() == [0, 1, 2]
    assert calls[0]["heuristic_policy"] == "teacher"
    assert teacher_valid.tolist() == [True, True, True, False, False]
    assert teacher_family.tolist() == [
        family_index["main_play_character"],
        family_index["attack"],
        family_index["climax_play"],
        -1,
        -1,
    ]
    assert teacher_slot.tolist() == [0, 1, -1, -1, -1]
    assert teacher_attack_type.tolist() == [-1, 1, -1, -1, -1]
    assert teacher_action.tolist() == [0, 14, 39, -1, -1]


def test_teacher_labels_from_mask_routes_public_decision_kinds_and_updates_counters() -> None:
    action_catalog = _teacher_test_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    attack_type_index = {name: index for index, name in enumerate(action_catalog.attack_type_names)}
    calls: list[dict[str, object]] = []

    def select_actions_from_mask(**kwargs: object) -> np.ndarray:
        calls.append(dict(kwargs))
        return np.asarray([0, 14], dtype=np.int64)

    counters = {"teacher_tactical_row_count": 0}
    labels = teacher_labels_from_mask(
        focal_rows=np.asarray([True, False, True], dtype=np.bool_),
        decision_kind=np.asarray([2, 3, 7], dtype=np.int32),
        obs_step=np.zeros((3, 4), dtype=np.float32),
        legal_mask=np.ones((3, action_catalog.action_space_size), dtype=np.bool_),
        counters=counters,
        guidance_active=True,
        teacher_policy="teacher",
        action_catalog=action_catalog,
        family_index=family_index,
        attack_type_index=attack_type_index,
        select_actions_from_mask=select_actions_from_mask,
    )

    teacher_family, teacher_slot, _teacher_move_source, teacher_attack_type, teacher_action, teacher_valid = labels
    assert counters["teacher_tactical_row_count"] == 2
    assert len(calls) == 1
    assert np.asarray(calls[0]["row_indices"]).tolist() == [0, 2]
    assert calls[0]["heuristic_policy"] == "teacher"
    assert teacher_valid.tolist() == [True, False, True]
    assert teacher_family.tolist() == [family_index["main_play_character"], -1, family_index["attack"]]
    assert teacher_slot.tolist() == [0, -1, 1]
    assert teacher_attack_type.tolist() == [-1, -1, 1]
    assert teacher_action.tolist() == [0, -1, 14]


def test_teacher_labels_from_ids_and_mask_return_sentinels_when_inactive_or_missing_policy() -> None:
    action_catalog = _teacher_test_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    attack_type_index = {name: index for index, name in enumerate(action_catalog.attack_type_names)}
    calls = 0

    def fail_selector(**_kwargs: object) -> np.ndarray:
        nonlocal calls
        calls += 1
        raise AssertionError("selector should not be called")

    ids_labels = teacher_labels_from_ids(
        focal_rows=np.asarray([True], dtype=np.bool_),
        decision_kind=np.asarray([1], dtype=np.int32),
        obs_step=np.zeros((1, 4), dtype=np.float32),
        legal_ids=np.asarray([0], dtype=np.uint32),
        legal_offsets=np.asarray([0, 1], dtype=np.uint32),
        legal_action_meta=None,
        counters={"teacher_tactical_row_count": 0},
        guidance_active=False,
        teacher_policy="teacher",
        action_catalog=action_catalog,
        family_index=family_index,
        attack_type_index=attack_type_index,
        select_actions_from_ids=fail_selector,
    )
    mask_labels = teacher_labels_from_mask(
        focal_rows=np.asarray([True], dtype=np.bool_),
        decision_kind=np.asarray([1], dtype=np.int32),
        obs_step=np.zeros((1, 4), dtype=np.float32),
        legal_mask=np.ones((1, action_catalog.action_space_size), dtype=np.bool_),
        counters={"teacher_tactical_row_count": 0},
        guidance_active=True,
        teacher_policy=None,
        action_catalog=action_catalog,
        family_index=family_index,
        attack_type_index=attack_type_index,
        select_actions_from_mask=fail_selector,
    )

    assert calls == 0
    assert ids_labels[-1].tolist() == [False]
    assert mask_labels[-1].tolist() == [False]
