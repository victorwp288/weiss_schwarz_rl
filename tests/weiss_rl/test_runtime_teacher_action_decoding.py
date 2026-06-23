from __future__ import annotations

import numpy as np
from weiss_rl.runtime.components.teacher_labels import (
    teacher_label_arrays,
    teacher_labels_from_actions,
)

from .runtime_test_support import _teacher_test_catalog


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
