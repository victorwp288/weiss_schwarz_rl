from __future__ import annotations

from typing import Any, cast

import numpy as np
from weiss_rl.runtime import QueueRuntime

from .runtime_test_support import _make_teacher_runtime_adapter


def test_teacher_guidance_adapter_respects_warmstart_only_collection_window() -> None:
    runtime = _make_teacher_runtime_adapter(aux_mode="warmstart_only", warmstart_updates=1, learner_update=0)

    assert QueueRuntime._teacher_guidance_active_for_collection(runtime) is True

    runtime_any = cast(Any, runtime)
    runtime_any._current_learner_update = 1

    assert QueueRuntime._teacher_guidance_active_for_collection(runtime) is False


def test_teacher_action_adapter_decodes_family_specific_fields() -> None:
    runtime = _make_teacher_runtime_adapter()
    runtime_any = cast(Any, runtime)
    action_catalog = runtime_any._teacher_action_catalog
    main_move_to_slot_2 = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if ((decoded := action_catalog.decode(action_id)).family == "main_move" and decoded.to_slot == 2)
    )

    teacher_family, teacher_slot, teacher_move_source, teacher_attack_type, teacher_action, teacher_valid = (
        QueueRuntime._teacher_labels_from_actions(
            runtime,
            row_indices=np.array([0, 1, 2, 3], dtype=np.int64),
            chosen_actions=np.array([0, 14, main_move_to_slot_2, 40], dtype=np.int64),
            num_rows=4,
        )
    )

    assert teacher_valid.tolist() == [True, True, True, True]
    assert teacher_family.tolist() == [
        runtime_any._teacher_family_index["main_play_character"],
        runtime_any._teacher_family_index["attack"],
        runtime_any._teacher_family_index["main_move"],
        runtime_any._teacher_family_index["pass"],
    ]
    assert teacher_slot.tolist() == [0, 1, 2, -1]
    assert teacher_move_source.tolist() == [-1, -1, 0, -1]
    assert teacher_attack_type.tolist() == [-1, 1, -1, -1]
    assert teacher_action.tolist() == [0, 14, main_move_to_slot_2, 40]


def test_teacher_action_adapter_returns_sentinels_after_warmstart_only_phase() -> None:
    runtime = _make_teacher_runtime_adapter(aux_mode="warmstart_only", warmstart_updates=1, learner_update=1)

    teacher_family, teacher_slot, teacher_move_source, teacher_attack_type, teacher_action, teacher_valid = (
        QueueRuntime._teacher_labels_from_actions(
            runtime,
            row_indices=np.array([0], dtype=np.int64),
            chosen_actions=np.array([0], dtype=np.int64),
            num_rows=1,
        )
    )

    assert teacher_valid.tolist() == [False]
    assert teacher_family.tolist() == [-1]
    assert teacher_slot.tolist() == [-1]
    assert teacher_move_source.tolist() == [-1]
    assert teacher_attack_type.tolist() == [-1]
    assert teacher_action.tolist() == [-1]


def test_teacher_ids_adapter_covers_public_decision_kinds_beyond_tactical_subset() -> None:
    runtime = _make_teacher_runtime_adapter()
    runtime_any = cast(Any, runtime)
    runtime_any._heuristic_public_actions_from_ids = lambda **kwargs: np.asarray(
        [
            int(kwargs["legal_ids"][int(kwargs["legal_offsets"][row_index])])
            for row_index in np.asarray(kwargs["row_indices"], dtype=np.int64).tolist()
        ],
        dtype=np.int64,
    )

    legal_ids = np.asarray([0, 40, 14, 40, 39, 40, 40, 40], dtype=np.uint32)
    legal_offsets = np.asarray([0, 2, 4, 6, 8], dtype=np.uint32)
    counters = {"teacher_tactical_row_count": 0}

    teacher_family, teacher_slot, teacher_move_source, teacher_attack_type, teacher_action, teacher_valid = (
        QueueRuntime._teacher_labels_from_ids(
            runtime,
            focal_rows=np.asarray([True, True, True, True], dtype=np.bool_),
            decision_kind=np.asarray([1, 5, 8, 0], dtype=np.int32),
            obs_step=np.zeros((4, 4), dtype=np.float32),
            legal_ids=legal_ids,
            legal_offsets=legal_offsets,
            legal_action_meta=None,
            counters=counters,
        )
    )

    assert teacher_valid.tolist() == [True, True, True, False]
    assert teacher_family.tolist() == [
        runtime_any._teacher_family_index["main_play_character"],
        runtime_any._teacher_family_index["attack"],
        runtime_any._teacher_family_index["climax_play"],
        -1,
    ]
    assert teacher_slot.tolist() == [0, 1, -1, -1]
    assert teacher_move_source.tolist() == [-1, -1, -1, -1]
    assert teacher_attack_type.tolist() == [-1, 1, -1, -1]
    assert teacher_action.tolist() == [0, 14, 39, -1]
    assert counters["teacher_tactical_row_count"] == 3
