from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.runtime.components.batching import build_impala_learner_batch


def test_build_impala_batch_backfills_terminal_outcome_to_last_train_row() -> None:
    unroll = SimpleNamespace(
        obs=np.zeros((4, 1, 1), dtype=np.float32),
        actions=np.zeros((4, 1), dtype=np.int64),
        rewards=np.asarray([[0.0], [-1.0], [0.0], [-1.0]], dtype=np.float32),
        terminated=np.asarray([[False], [True], [False], [True]], dtype=np.bool_),
        truncated=np.zeros((4, 1), dtype=np.bool_),
        to_play_seat=np.asarray([[0], [1], [0], [1]], dtype=np.int64),
        behavior_logp=np.zeros((4, 1), dtype=np.float32),
        values=np.zeros((4, 1), dtype=np.float32),
        bootstrap_value=np.zeros((1,), dtype=np.float32),
        bootstrap_obs=np.zeros((1, 1), dtype=np.float32),
        bootstrap_actor=np.zeros((1,), dtype=np.int64),
        initial_hidden_state=np.zeros((1, 1), dtype=np.float32),
        final_hidden_state=np.zeros((1, 1), dtype=np.float32),
        policy_train_mask=np.asarray([[True], [False], [True], [False]], dtype=np.bool_),
        legal_actions=LegalActionBatch.from_mask(np.ones((4, 1, 1), dtype=np.bool_)),
        teacher_family=None,
        teacher_slot=None,
        teacher_move_source=None,
        teacher_attack_type=None,
        teacher_action=None,
        teacher_valid=None,
        trajectory_retention_valid=None,
    )

    batch = build_impala_learner_batch(
        [unroll],
        action_dim=1,
        gamma=1.0,
        truncation_reward=0.0,
        truncation_bootstrap_value=False,
        vtrace_rho_bar=1.0,
        vtrace_c_bar=1.0,
        terminal_outcome_backfill_reward=1.0,
    )

    assert batch["rewards"][:, 0].tolist() == pytest.approx([1.0, -1.0, 1.0, -1.0])
    assert batch["terminal_outcome_backfill_count"] == 2
    assert batch["terminal_outcome_backfill_total_micros"] == 2_000_000


def test_build_impala_batch_trace_backfills_terminal_outcome_to_episode_suffix() -> None:
    unroll = SimpleNamespace(
        obs=np.zeros((5, 1, 1), dtype=np.float32),
        actions=np.zeros((5, 1), dtype=np.int64),
        rewards=np.asarray([[0.0], [0.0], [-1.0], [0.0], [-1.0]], dtype=np.float32),
        terminated=np.asarray([[False], [False], [True], [False], [True]], dtype=np.bool_),
        truncated=np.zeros((5, 1), dtype=np.bool_),
        to_play_seat=np.asarray([[0], [0], [1], [0], [0]], dtype=np.int64),
        behavior_logp=np.zeros((5, 1), dtype=np.float32),
        values=np.zeros((5, 1), dtype=np.float32),
        bootstrap_value=np.zeros((1,), dtype=np.float32),
        bootstrap_obs=np.zeros((1, 1), dtype=np.float32),
        bootstrap_actor=np.zeros((1,), dtype=np.int64),
        initial_hidden_state=np.zeros((1, 1), dtype=np.float32),
        final_hidden_state=np.zeros((1, 1), dtype=np.float32),
        policy_train_mask=np.asarray([[True], [True], [False], [True], [True]], dtype=np.bool_),
        legal_actions=LegalActionBatch.from_mask(np.ones((5, 1, 1), dtype=np.bool_)),
        teacher_family=None,
        teacher_slot=None,
        teacher_move_source=None,
        teacher_attack_type=None,
        teacher_action=None,
        teacher_valid=None,
        trajectory_retention_valid=None,
    )

    batch = build_impala_learner_batch(
        [unroll],
        action_dim=1,
        gamma=1.0,
        truncation_reward=0.0,
        truncation_bootstrap_value=False,
        vtrace_rho_bar=1.0,
        vtrace_c_bar=1.0,
        terminal_outcome_trace_backfill_reward=0.25,
    )

    assert batch["rewards"][:, 0].tolist() == pytest.approx([0.25, 0.25, -1.0, -0.25, -1.0])
    assert batch["terminal_outcome_trace_backfill_count"] == 3
    assert batch["terminal_outcome_trace_backfill_total_micros"] == 750_000
