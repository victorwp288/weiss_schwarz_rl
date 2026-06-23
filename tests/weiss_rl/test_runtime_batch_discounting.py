from __future__ import annotations

import numpy as np
import pytest
from weiss_rl.runtime.components.batching import (
    actor_perspective_discounts,
    gae_advantages,
)


def test_gae_advantages_matches_manual_discounted_deltas() -> None:
    rewards = np.asarray([[1.0], [0.5]], dtype=np.float32)
    values = np.asarray([[0.2], [0.3]], dtype=np.float32)
    discounts = np.asarray([[1.0], [0.0]], dtype=np.float32)
    bootstrap = np.asarray([0.4], dtype=np.float32)

    advantages = gae_advantages(
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap,
        discounts=discounts,
        gae_lambda=0.95,
    )

    expected_last = 0.5 - 0.3
    expected_first = (1.0 + 0.3 - 0.2) + (0.95 * expected_last)
    assert advantages[:, 0].tolist() == pytest.approx([expected_first, expected_last])


def test_actor_perspective_discounts_flip_when_next_value_is_opponent_perspective() -> None:
    done = np.asarray([[False], [False], [False]], dtype=np.bool_)
    to_play_seat = np.asarray([[0], [1], [1]], dtype=np.int64)
    bootstrap_actor = np.asarray([0], dtype=np.int64)

    discounts = actor_perspective_discounts(
        done=done,
        to_play_seat=to_play_seat,
        bootstrap_actor=bootstrap_actor,
        gamma=0.99,
    )

    assert discounts[:, 0].tolist() == pytest.approx([-0.99, 0.99, -0.99])


def test_actor_perspective_discounts_ignore_invalid_bootstrap_actor_on_done_rows() -> None:
    discounts = actor_perspective_discounts(
        done=np.asarray([[True]], dtype=np.bool_),
        to_play_seat=np.asarray([[0]], dtype=np.int64),
        bootstrap_actor=np.asarray([-1], dtype=np.int64),
        gamma=0.99,
    )

    assert discounts[:, 0].tolist() == pytest.approx([0.0])
