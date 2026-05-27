from __future__ import annotations

import numpy as np
import pytest

from weiss_rl.diagnostics.action_diagnostics import (
    make_action_sequence_state,
    update_action_summary_from_ids,
)


def test_action_summary_prefers_simulator_main_move_flags_for_streaks() -> None:
    counters = {
        "total_actions": 0,
        "pass_actions": 0,
        "main_move_actions": 0,
        "pass_with_nonpass_available": 0,
        "max_consecutive_main_moves": 0,
    }
    state = make_action_sequence_state(1)
    legal_ids = np.array([51, 402], dtype=np.int64)
    legal_offsets = np.array([0, 2], dtype=np.int64)

    update_action_summary_from_ids(
        counters=counters,
        state=state,
        actions=np.array([402], dtype=np.int64),
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        main_move_action=np.array([False], dtype=np.bool_),
    )
    assert counters["main_move_actions"] == 0
    assert counters["max_consecutive_main_moves"] == 0

    update_action_summary_from_ids(
        counters=counters,
        state=state,
        actions=np.array([402], dtype=np.int64),
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        main_move_action=np.array([True], dtype=np.bool_),
    )
    update_action_summary_from_ids(
        counters=counters,
        state=state,
        actions=np.array([402], dtype=np.int64),
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        main_move_action=np.array([True], dtype=np.bool_),
    )

    assert counters["main_move_actions"] == 2
    assert counters["max_consecutive_main_moves"] == 2


def test_action_summary_validates_main_move_flag_shape() -> None:
    with pytest.raises(ValueError, match="main_move_action must have shape"):
        update_action_summary_from_ids(
            counters={},
            state=make_action_sequence_state(1),
            actions=np.array([402], dtype=np.int64),
            legal_ids=np.array([402], dtype=np.int64),
            legal_offsets=np.array([0, 1], dtype=np.int64),
            main_move_action=np.array([True, False], dtype=np.bool_),
        )
