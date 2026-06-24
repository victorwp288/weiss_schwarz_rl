from __future__ import annotations

import numpy as np
from weiss_rl.runtime.components.actions.action_surface import (
    filter_main_move_only_rows_to_pass_from_ids,
    filter_mulligan_select_after_select_from_ids,
    filter_pass_when_attack_available_from_ids,
)


def test_filter_mulligan_select_after_select_keeps_initial_select_surface() -> None:
    obs = np.zeros((1, 8), dtype=np.int32)
    obs[0, 6] = -1
    legal_ids = np.array([0, 1, 2], dtype=np.uint32)
    legal_offsets = np.array([0, 3], dtype=np.uint32)
    legal_action_meta = np.array(
        [
            [3, 65535, 65535, 65535],
            [4, 0, 65535, 65535],
            [4, 1, 65535, 65535],
        ],
        dtype=np.uint16,
    )

    result = filter_mulligan_select_after_select_from_ids(
        obs=obs,
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        legal_action_meta=legal_action_meta,
        last_action_arg0_index=6,
        mulligan_confirm_family_id=3,
        mulligan_select_family_id=4,
    )

    assert result.filtered_rows == 0
    assert result.filtered_actions == 0
    assert result.legal_ids.tolist() == [0, 1, 2]
    assert result.legal_offsets.tolist() == [0, 3]


def test_filter_mulligan_select_after_select_forces_confirm_after_prior_select() -> None:
    obs = np.zeros((2, 8), dtype=np.int32)
    obs[:, 6] = np.array([0, -1], dtype=np.int32)
    legal_ids = np.array([0, 1, 2, 0, 1], dtype=np.uint32)
    legal_offsets = np.array([0, 3, 5], dtype=np.uint32)
    legal_action_meta = np.array(
        [
            [3, 65535, 65535, 65535],
            [4, 0, 65535, 65535],
            [4, 1, 65535, 65535],
            [3, 65535, 65535, 65535],
            [4, 0, 65535, 65535],
        ],
        dtype=np.uint16,
    )

    result = filter_mulligan_select_after_select_from_ids(
        obs=obs,
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        legal_action_meta=legal_action_meta,
        last_action_arg0_index=6,
        mulligan_confirm_family_id=3,
        mulligan_select_family_id=4,
    )

    assert result.filtered_rows == 1
    assert result.filtered_actions == 2
    assert result.legal_ids.tolist() == [0, 0, 1]
    assert result.legal_offsets.tolist() == [0, 1, 3]
    assert result.legal_action_meta is not None
    assert result.legal_action_meta[:, 0].tolist() == [3, 3, 4]


def test_filter_main_move_only_rows_to_pass_keeps_only_pass_when_no_productive_alternative() -> None:
    legal_ids = np.array([5, 10, 11, 5, 10, 20, 5, 20], dtype=np.uint32)
    legal_offsets = np.array([0, 3, 6, 8], dtype=np.uint32)
    legal_action_meta = np.array(
        [
            [1, 65535, 65535, 65535],
            [2, 0, 1, 65535],
            [2, 1, 2, 65535],
            [1, 65535, 65535, 65535],
            [2, 0, 1, 65535],
            [3, 0, 65535, 65535],
            [1, 65535, 65535, 65535],
            [3, 0, 65535, 65535],
        ],
        dtype=np.uint16,
    )

    result = filter_main_move_only_rows_to_pass_from_ids(
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        legal_action_meta=legal_action_meta,
        pass_action_id=5,
        main_move_family_id=2,
    )

    assert result.filtered_rows == 1
    assert result.filtered_actions == 2
    assert result.legal_ids.tolist() == [5, 5, 10, 20, 5, 20]
    assert result.legal_offsets.tolist() == [0, 1, 4, 6]
    assert result.legal_action_meta is not None
    assert result.legal_action_meta[:, 0].tolist() == [1, 1, 2, 3, 1, 3]


def test_filter_main_move_only_rows_to_pass_can_allow_selected_rows() -> None:
    legal_ids = np.array([5, 10, 11, 5, 12, 13], dtype=np.uint32)
    legal_offsets = np.array([0, 3, 6], dtype=np.uint32)
    legal_action_meta = np.array(
        [
            [1, 65535, 65535, 65535],
            [2, 0, 1, 65535],
            [2, 1, 2, 65535],
            [1, 65535, 65535, 65535],
            [2, 0, 2, 65535],
            [2, 1, 3, 65535],
        ],
        dtype=np.uint16,
    )

    result = filter_main_move_only_rows_to_pass_from_ids(
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        legal_action_meta=legal_action_meta,
        pass_action_id=5,
        main_move_family_id=2,
        allow_main_move_only_rows=np.array([True, False], dtype=np.bool_),
    )

    assert result.filtered_rows == 1
    assert result.filtered_actions == 2
    assert result.legal_ids.tolist() == [5, 10, 11, 5]
    assert result.legal_offsets.tolist() == [0, 3, 4]
    assert result.legal_action_meta is not None
    assert result.legal_action_meta[:, 0].tolist() == [1, 2, 2, 1]


def test_filter_pass_when_attack_available_removes_pass_only_on_attack_rows() -> None:
    legal_ids = np.array([5, 30, 31, 5, 20, 5], dtype=np.uint32)
    legal_offsets = np.array([0, 3, 5, 6], dtype=np.uint32)
    legal_action_meta = np.array(
        [
            [1, 65535, 65535, 65535],
            [4, 0, 0, 65535],
            [4, 1, 0, 65535],
            [1, 65535, 65535, 65535],
            [3, 0, 65535, 65535],
            [1, 65535, 65535, 65535],
        ],
        dtype=np.uint16,
    )

    result = filter_pass_when_attack_available_from_ids(
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        legal_action_meta=legal_action_meta,
        pass_action_id=5,
        attack_family_id=4,
    )

    assert result.filtered_rows == 1
    assert result.filtered_actions == 1
    assert result.legal_ids.tolist() == [30, 31, 5, 20, 5]
    assert result.legal_offsets.tolist() == [0, 2, 4, 5]
    assert result.legal_action_meta is not None
    assert result.legal_action_meta[:, 0].tolist() == [4, 4, 1, 3, 1]
