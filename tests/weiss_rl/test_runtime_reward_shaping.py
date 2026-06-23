from __future__ import annotations

import numpy as np
import pytest
from weiss_rl.runtime.components.reward_shaping import (
    apply_collector_reward_shaping,
    apply_mulligan_select_with_confirm_penalty,
    apply_pass_with_nonpass_penalty,
    mulligan_select_with_confirm_penalty_mask_from_ids,
    pass_penalty_ignored_alternative_family_ids,
    pass_with_nonpass_penalty_mask_from_ids,
    pass_with_nonpass_penalty_mask_from_mask,
)


def test_pass_with_nonpass_penalty_mask_from_ids_only_flags_pass_rows_with_real_alternatives() -> None:
    actions = np.array([5, 1, 5, 5], dtype=np.int64)
    legal_ids = np.array([5, 7, 1, 5], dtype=np.uint32)
    legal_offsets = np.array([0, 2, 3, 4, 4], dtype=np.uint32)

    mask = pass_with_nonpass_penalty_mask_from_ids(
        actions,
        legal_ids,
        legal_offsets,
        pass_action_id=5,
    )

    assert mask.tolist() == [True, False, False, False]


def test_pass_with_nonpass_penalty_ignores_main_move_only_alternatives_from_meta() -> None:
    actions = np.array([5, 5, 5], dtype=np.int64)
    legal_ids = np.array([5, 7, 5, 8, 9, 5, 9], dtype=np.uint32)
    legal_offsets = np.array([0, 2, 5, 7], dtype=np.uint32)
    pass_family = 1
    main_move_family = 2
    play_family = 3
    legal_action_meta = np.array(
        [
            [pass_family, 0, 0, 0],
            [main_move_family, 0, 0, 0],
            [pass_family, 0, 0, 0],
            [main_move_family, 0, 0, 0],
            [play_family, 0, 0, 0],
            [pass_family, 0, 0, 0],
            [play_family, 0, 0, 0],
        ],
        dtype=np.uint16,
    )

    mask = pass_with_nonpass_penalty_mask_from_ids(
        actions,
        legal_ids,
        legal_offsets,
        pass_action_id=5,
        legal_action_meta=legal_action_meta,
        ignored_alternative_family_ids=(main_move_family,),
    )

    assert mask.tolist() == [False, True, True]


def test_pass_penalty_ignored_alternative_family_ids_uses_main_move_when_present() -> None:
    assert pass_penalty_ignored_alternative_family_ids({"pass": 1, "main_move": 7}) == (7,)
    assert pass_penalty_ignored_alternative_family_ids({"pass": 1}) == ()


def test_pass_with_nonpass_penalty_mask_from_mask_only_flags_pass_rows_with_real_alternatives() -> None:
    actions = np.array([2, 2, 1], dtype=np.int64)
    legal_mask = np.array(
        [
            [True, False, True],
            [False, False, True],
            [False, True, True],
        ],
        dtype=np.bool_,
    )

    mask = pass_with_nonpass_penalty_mask_from_mask(
        actions,
        legal_mask,
        pass_action_id=2,
    )

    assert mask.tolist() == [True, False, False]


def test_apply_pass_with_nonpass_penalty_returns_shaped_copy_and_counter_units() -> None:
    rewards = np.array([0.0, 1.0, -1.0], dtype=np.float32)
    actions = np.array([5, 5, 3], dtype=np.int64)
    legal_ids = np.array([5, 8, 5, 3, 5], dtype=np.uint32)
    legal_offsets = np.array([0, 2, 3, 5], dtype=np.uint32)

    shaped, count, total_micros = apply_pass_with_nonpass_penalty(
        rewards,
        actions,
        pass_action_id=5,
        penalty=0.02,
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
    )

    assert shaped.tolist() == pytest.approx([-0.02, 1.0, -1.0])
    assert rewards.tolist() == pytest.approx([0.0, 1.0, -1.0])
    assert count == 1
    assert total_micros == 20_000


def test_mulligan_select_with_confirm_penalty_mask_only_flags_select_when_confirm_legal() -> None:
    actions = np.array([1, 0, 2, 1], dtype=np.int64)
    legal_ids = np.array([0, 1, 0, 1, 1, 2, 1], dtype=np.uint32)
    legal_offsets = np.array([0, 2, 4, 6, 7], dtype=np.uint32)
    legal_action_meta = np.array(
        [
            [3, 0, 0, 0],
            [4, 0, 0, 0],
            [3, 0, 0, 0],
            [4, 0, 0, 0],
            [4, 0, 0, 0],
            [5, 0, 0, 0],
            [4, 0, 0, 0],
        ],
        dtype=np.uint16,
    )

    mask = mulligan_select_with_confirm_penalty_mask_from_ids(
        actions,
        legal_ids,
        legal_offsets,
        legal_action_meta,
        mulligan_confirm_family_id=3,
        mulligan_select_family_id=4,
    )

    assert mask.tolist() == [True, False, False, False]


def test_apply_mulligan_select_with_confirm_penalty_returns_shaped_copy_and_counter_units() -> None:
    rewards = np.array([0.0, 1.0, -1.0], dtype=np.float32)
    actions = np.array([1, 0, 2], dtype=np.int64)
    legal_ids = np.array([0, 1, 0, 1, 1, 2], dtype=np.uint32)
    legal_offsets = np.array([0, 2, 4, 6], dtype=np.uint32)
    legal_action_meta = np.array(
        [
            [3, 0, 0, 0],
            [4, 0, 0, 0],
            [3, 0, 0, 0],
            [4, 0, 0, 0],
            [4, 0, 0, 0],
            [5, 0, 0, 0],
        ],
        dtype=np.uint16,
    )

    shaped, count, total_micros = apply_mulligan_select_with_confirm_penalty(
        rewards,
        actions,
        penalty=0.02,
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        legal_action_meta=legal_action_meta,
        mulligan_confirm_family_id=3,
        mulligan_select_family_id=4,
    )

    assert shaped.tolist() == pytest.approx([-0.02, 1.0, -1.0])
    assert rewards.tolist() == pytest.approx([0.0, 1.0, -1.0])
    assert count == 1
    assert total_micros == 20_000


def test_apply_collector_reward_shaping_updates_ids_penalty_counters() -> None:
    counters = {
        "pass_with_nonpass_penalty_count": 0,
        "pass_with_nonpass_penalty_total_micros": 0,
        "mulligan_select_with_confirm_penalty_count": 0,
        "mulligan_select_with_confirm_penalty_total_micros": 0,
    }
    rewards = np.array([0.0, 1.0, -1.0], dtype=np.float32)
    actions = np.array([5, 1, 5], dtype=np.int64)
    legal_ids = np.array([5, 8, 0, 1, 5, 7], dtype=np.uint32)
    legal_offsets = np.array([0, 2, 4, 6], dtype=np.uint32)
    legal_action_meta = np.array(
        [
            [1, 0, 0, 0],
            [2, 0, 0, 0],
            [3, 0, 0, 0],
            [4, 0, 0, 0],
            [1, 0, 0, 0],
            [2, 0, 0, 0],
        ],
        dtype=np.uint16,
    )

    shaped = apply_collector_reward_shaping(
        rewards,
        actions,
        counters=counters,
        pass_action_id=5,
        pass_with_nonpass_penalty=0.02,
        mulligan_select_with_confirm_penalty=0.03,
        action_family_index={"main_move": 2, "mulligan_confirm": 3, "mulligan_select": 4},
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        legal_action_meta=legal_action_meta,
    )

    assert shaped.tolist() == pytest.approx([0.0, 0.97, -1.0])
    assert rewards.tolist() == pytest.approx([0.0, 1.0, -1.0])
    assert counters["pass_with_nonpass_penalty_count"] == 0
    assert counters["pass_with_nonpass_penalty_total_micros"] == 0
    assert counters["mulligan_select_with_confirm_penalty_count"] == 1
    assert counters["mulligan_select_with_confirm_penalty_total_micros"] == 30_000


def test_apply_collector_reward_shaping_updates_mask_penalty_counters_without_mulligan() -> None:
    counters = {
        "pass_with_nonpass_penalty_count": 0,
        "pass_with_nonpass_penalty_total_micros": 0,
        "mulligan_select_with_confirm_penalty_count": 0,
        "mulligan_select_with_confirm_penalty_total_micros": 0,
    }

    shaped = apply_collector_reward_shaping(
        np.array([0.0, 1.0], dtype=np.float32),
        np.array([2, 1], dtype=np.int64),
        counters=counters,
        pass_action_id=2,
        pass_with_nonpass_penalty=0.05,
        mulligan_select_with_confirm_penalty=0.07,
        legal_mask=np.array(
            [
                [False, True, True],
                [False, True, True],
            ],
            dtype=np.bool_,
        ),
    )

    assert shaped.tolist() == pytest.approx([-0.05, 1.0])
    assert counters["pass_with_nonpass_penalty_count"] == 1
    assert counters["pass_with_nonpass_penalty_total_micros"] == 50_000
    assert counters["mulligan_select_with_confirm_penalty_count"] == 0
    assert counters["mulligan_select_with_confirm_penalty_total_micros"] == 0
