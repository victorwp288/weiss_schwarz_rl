from __future__ import annotations

import numpy as np
import pytest

from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.runtime_components.teacher_heuristic_mixin import QueueRuntimeTeacherHeuristicMixin
from weiss_rl.runtime_components.teacher_labels import (
    PUBLIC_TEACHER_DECISION_KINDS,
    selected_teacher_label_profile,
    teacher_guidance_active_for_collection,
    teacher_label_arrays,
    teacher_labels_from_actions,
    teacher_labels_from_ids,
    teacher_labels_from_mask,
)


def _teacher_test_catalog() -> ActionCatalog:
    return ActionCatalog.from_spec_bundle(
        {
            "action": {
                "action_encoding_version": 1,
                "action_space_size": 41,
                "pass_action_id": 40,
                "constants": [["MAX_HAND", 2], ["MAX_STAGE", 5], ["ATTACK_SLOT_COUNT", 3]],
                "families": [
                    {"name": "main_play_character", "base": 0, "count": 10},
                    {"name": "attack", "base": 10, "count": 9},
                    {"name": "main_move", "base": 19, "count": 20},
                    {"name": "climax_play", "base": 39, "count": 1},
                    {"name": "pass", "base": 40, "count": 1},
                ],
                "attack_type_encoding": [["frontal", 0], ["direct", 1], ["side", 2]],
            }
        }
    )


def test_teacher_guidance_active_for_collection_preserves_aux_mode_rules() -> None:
    assert not teacher_guidance_active_for_collection(
        enabled=False,
        teacher_aux_mode="always",
        warmstart_updates=3,
        current_learner_update=0,
    )
    assert not teacher_guidance_active_for_collection(
        enabled=True,
        teacher_aux_mode="off",
        warmstart_updates=3,
        current_learner_update=0,
    )
    assert teacher_guidance_active_for_collection(
        enabled=True,
        teacher_aux_mode="always",
        warmstart_updates=0,
        current_learner_update=99,
    )
    assert teacher_guidance_active_for_collection(
        enabled=True,
        teacher_aux_mode="warmstart_only",
        warmstart_updates=2,
        current_learner_update=1,
    )
    assert not teacher_guidance_active_for_collection(
        enabled=True,
        teacher_aux_mode="warmstart_only",
        warmstart_updates=2,
        current_learner_update=2,
    )


def test_selected_teacher_label_profile_tracks_cycle_mode_until_end_update() -> None:
    profiles = ("base", "aggressive", "control")

    assert selected_teacher_label_profile(profiles, profile_mode="cycle", update_count=0, end_updates=3) == "base"
    assert selected_teacher_label_profile(profiles, profile_mode="cycle", update_count=1, end_updates=3) == "aggressive"
    assert selected_teacher_label_profile(profiles, profile_mode="cycle", update_count=2, end_updates=3) == "control"
    assert selected_teacher_label_profile(profiles, profile_mode="cycle", update_count=4, end_updates=3) == "base"
    assert selected_teacher_label_profile(profiles, profile_mode="mixture", update_count=1, end_updates=-1) == "base"

    with pytest.raises(ValueError, match="unsupported profiles"):
        selected_teacher_label_profile(("unknown",), profile_mode="cycle", update_count=0, end_updates=-1)


def test_teacher_heuristic_mixin_selects_profiled_teacher_policy_for_labels() -> None:
    class _Runtime(QueueRuntimeTeacherHeuristicMixin):
        _teacher_policy = "base-policy"
        _teacher_policy_by_profile = {
            "base": "base-policy",
            "aggressive": "aggressive-policy",
            "control": "control-policy",
        }
        _teacher_label_profiles = ("base", "aggressive", "control")
        _teacher_label_profile_mode = "cycle"
        _teacher_label_profiles_end_updates = 150
        _current_learner_update = 2

    assert _Runtime()._teacher_label_policy_for_current_update() == "control-policy"


def test_teacher_label_arrays_use_sentinel_values_and_bool_validity() -> None:
    teacher_family, teacher_slot, teacher_move_source, teacher_attack_type, teacher_action, teacher_valid = (
        teacher_label_arrays(2)
    )

    assert teacher_family.dtype == np.int32
    assert teacher_valid.dtype == np.bool_
    assert teacher_family.tolist() == [-1, -1]
    assert teacher_slot.tolist() == [-1, -1]
    assert teacher_move_source.tolist() == [-1, -1]
    assert teacher_attack_type.tolist() == [-1, -1]
    assert teacher_action.tolist() == [-1, -1]
    assert teacher_valid.tolist() == [False, False]


def test_teacher_labels_from_actions_decode_family_specific_fields() -> None:
    action_catalog = _teacher_test_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    attack_type_index = {name: index for index, name in enumerate(action_catalog.attack_type_names)}
    main_move_to_slot_2 = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if ((decoded := action_catalog.decode(action_id)).family == "main_move" and decoded.to_slot == 2)
    )

    teacher_family, teacher_slot, teacher_move_source, teacher_attack_type, teacher_action, teacher_valid = (
        teacher_labels_from_actions(
            row_indices=np.array([0, 1, 2, 3], dtype=np.int64),
            chosen_actions=np.array([0, 14, main_move_to_slot_2, 40], dtype=np.int64),
            num_rows=4,
            guidance_active=True,
            action_catalog=action_catalog,
            family_index=family_index,
            attack_type_index=attack_type_index,
        )
    )

    assert teacher_valid.tolist() == [True, True, True, True]
    assert teacher_family.tolist() == [
        family_index["main_play_character"],
        family_index["attack"],
        family_index["main_move"],
        family_index["pass"],
    ]
    assert teacher_slot.tolist() == [0, 1, 2, -1]
    assert teacher_move_source.tolist() == [-1, -1, 0, -1]
    assert teacher_attack_type.tolist() == [-1, 1, -1, -1]
    assert teacher_action.tolist() == [0, 14, main_move_to_slot_2, 40]


def test_teacher_labels_from_actions_returns_sentinels_when_inactive() -> None:
    teacher_family, teacher_slot, teacher_move_source, teacher_attack_type, teacher_action, teacher_valid = (
        teacher_labels_from_actions(
            row_indices=np.array([0], dtype=np.int64),
            chosen_actions=np.array([0], dtype=np.int64),
            num_rows=1,
            guidance_active=False,
            action_catalog=_teacher_test_catalog(),
            family_index={},
            attack_type_index={},
        )
    )

    assert teacher_valid.tolist() == [False]
    assert teacher_family.tolist() == [-1]
    assert teacher_slot.tolist() == [-1]
    assert teacher_move_source.tolist() == [-1]
    assert teacher_attack_type.tolist() == [-1]
    assert teacher_action.tolist() == [-1]


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
