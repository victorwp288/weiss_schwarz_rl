from __future__ import annotations

import numpy as np
import pytest
from weiss_rl.actors.action_accounting import make_actor_action_accounting, record_actor_actions
from weiss_rl.actors.action_selection import ActorActionSelection


def test_record_actor_actions_updates_ids_counters_and_shapes_rewards() -> None:
    accounting = make_actor_action_accounting(num_envs=2)
    selection = ActorActionSelection(
        actions=np.array([5, 1], dtype=np.int64),
        logp=np.zeros((2,), dtype=np.float32),
        entropy=np.zeros((2,), dtype=np.float32),
        legal_ids=np.array([5, 8, 0, 1], dtype=np.int64),
        legal_offsets=np.array([0, 2, 4], dtype=np.int64),
    )

    shaped = record_actor_actions(
        accounting=accounting,
        layout_name="i16_legal_ids",
        selection=selection,
        rewards=np.array([0.0, 1.0], dtype=np.float32),
        pass_action_id=5,
        pass_with_nonpass_penalty=0.02,
        main_move_action=np.array([False, False], dtype=np.bool_),
    )

    assert shaped.tolist() == pytest.approx([-0.02, 1.0])
    assert accounting.counters["total_actions"] == 2
    assert accounting.counters["pass_actions"] == 1
    assert accounting.counters["pass_with_nonpass_available"] == 1
    assert accounting.counters["pass_with_nonpass_penalty_count"] == 1
    assert accounting.counters["pass_with_nonpass_penalty_total_micros"] == 20_000


def test_record_actor_actions_updates_mask_counters_and_uses_main_move_flag() -> None:
    accounting = make_actor_action_accounting(num_envs=2)
    selection = ActorActionSelection(
        actions=np.array([2, 1], dtype=np.int64),
        logp=np.zeros((2,), dtype=np.float32),
        entropy=np.zeros((2,), dtype=np.float32),
        legal_mask=np.array(
            [
                [True, True, True],
                [False, True, True],
            ],
            dtype=np.bool_,
        ),
    )

    shaped = record_actor_actions(
        accounting=accounting,
        layout_name="mask",
        selection=selection,
        rewards=np.array([0.0, 1.0], dtype=np.float32),
        pass_action_id=2,
        pass_with_nonpass_penalty=0.05,
        main_move_action=np.array([False, True], dtype=np.bool_),
    )

    assert shaped.tolist() == pytest.approx([-0.05, 1.0])
    assert accounting.counters["total_actions"] == 2
    assert accounting.counters["pass_actions"] == 1
    assert accounting.counters["main_move_actions"] == 1
    assert accounting.counters["max_consecutive_main_moves"] == 1
    assert accounting.counters["pass_with_nonpass_penalty_count"] == 1
    assert accounting.counters["pass_with_nonpass_penalty_total_micros"] == 50_000


def test_record_actor_actions_rejects_missing_layout_legality() -> None:
    accounting = make_actor_action_accounting(num_envs=1)
    selection = ActorActionSelection(
        actions=np.array([0], dtype=np.int64),
        logp=np.zeros((1,), dtype=np.float32),
        entropy=np.zeros((1,), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="legal_ids"):
        record_actor_actions(
            accounting=accounting,
            layout_name="i16_legal_ids",
            selection=selection,
            rewards=np.array([0.0], dtype=np.float32),
            pass_action_id=5,
            pass_with_nonpass_penalty=0.02,
        )
