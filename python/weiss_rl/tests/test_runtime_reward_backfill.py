from __future__ import annotations

import numpy as np
import pytest

from weiss_rl.runtime_components.reward_backfill import (
    apply_runtime_reward_backfills,
    apply_terminal_outcome_backfill,
    apply_terminal_outcome_trace_backfill,
)


def test_terminal_outcome_backfill_skips_rows_without_prior_train_target() -> None:
    rewards = np.asarray([[-1.0], [0.0], [-1.0]], dtype=np.float32)
    done = np.asarray([[True], [False], [True]], dtype=np.bool_)
    policy_train_mask = np.asarray([[False], [True], [False]], dtype=np.bool_)

    shaped, backfill_count, total_micros, trace_count, trace_total_micros = apply_terminal_outcome_backfill(
        rewards=rewards,
        done=done,
        policy_train_mask=policy_train_mask,
        reward=0.5,
    )

    assert shaped[:, 0].tolist() == pytest.approx([-1.0, 0.5, -1.0])
    assert backfill_count == 1
    assert total_micros == 500_000
    assert trace_count == 0
    assert trace_total_micros == 0


def test_terminal_outcome_trace_backfill_resets_suffix_at_episode_boundary() -> None:
    rewards = np.asarray([[0.0], [-1.0], [0.0], [-1.0]], dtype=np.float32)
    done = np.asarray([[False], [True], [False], [True]], dtype=np.bool_)
    policy_train_mask = np.asarray([[True], [False], [True], [False]], dtype=np.bool_)

    shaped, _, _, trace_count, trace_total_micros = apply_terminal_outcome_trace_backfill(
        rewards=rewards,
        done=done,
        policy_train_mask=policy_train_mask,
        reward=0.25,
    )

    assert shaped[:, 0].tolist() == pytest.approx([0.25, -1.0, 0.25, -1.0])
    assert trace_count == 2
    assert trace_total_micros == 500_000


def test_runtime_reward_backfills_report_combined_metrics() -> None:
    rewards = np.asarray([[0.0], [-1.0]], dtype=np.float32)
    done = np.asarray([[False], [True]], dtype=np.bool_)
    policy_train_mask = np.asarray([[True], [False]], dtype=np.bool_)

    shaped, metrics = apply_runtime_reward_backfills(
        rewards=rewards,
        done=done,
        policy_train_mask=policy_train_mask,
        terminal_outcome_backfill_reward=0.5,
        terminal_outcome_trace_backfill_reward=0.25,
    )

    assert shaped[:, 0].tolist() == pytest.approx([0.75, -1.0])
    assert metrics.outcome_count == 1
    assert metrics.outcome_total_micros == 500_000
    assert metrics.trace_count == 1
    assert metrics.trace_total_micros == 250_000


def test_terminal_outcome_backfill_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="identical time-major shapes"):
        apply_terminal_outcome_backfill(
            rewards=np.zeros((2, 1), dtype=np.float32),
            done=np.zeros((2, 2), dtype=np.bool_),
            policy_train_mask=np.zeros((2, 1), dtype=np.bool_),
            reward=1.0,
        )
