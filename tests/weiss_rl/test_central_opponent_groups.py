from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from weiss_rl.runtime.components.opponents.central_opponent_groups import group_central_opponent_rows
from weiss_rl.runtime.components.policy_ids import MIRROR_OPPONENT_POLICY_ID


def _actor(*, focal: list[int], policies: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        focal_seat_by_env=np.asarray(focal, dtype=np.int64),
        opponent_policy_id_by_env=np.asarray(policies, dtype=object),
    )


def test_group_central_opponent_rows_skips_focal_and_mirror_rows() -> None:
    actor = _actor(
        focal=[0, 0, 1, 1],
        policies=["policy_b", MIRROR_OPPONENT_POLICY_ID, "policy_a", "policy_b"],
    )
    batch = SimpleNamespace(name="batch")
    obs_step = np.arange(12, dtype=np.float32).reshape(4, 3)
    actor_step = np.asarray([0, 1, 0, 0], dtype=np.int64)
    logits_out = np.zeros((4, 5), dtype=np.float32)
    values_out = np.zeros((4,), dtype=np.float32)

    groups = group_central_opponent_rows(
        actors=[actor],
        batches=[batch],
        obs_steps=[obs_step],
        actor_steps=[actor_step],
        logits_outs=[logits_out],
        values_outs=[values_out],
    )

    assert sorted(groups) == ["policy_a", "policy_b"]
    assert len(groups["policy_a"]) == 1
    assert groups["policy_a"][0].row_indices.tolist() == [2]
    assert groups["policy_b"][0].row_indices.tolist() == [3]
    assert groups["policy_b"][0].actor is actor
    assert groups["policy_b"][0].batch is batch
    assert groups["policy_b"][0].obs_step is obs_step
    assert groups["policy_b"][0].actor_step is actor_step
    assert groups["policy_b"][0].logits_out is logits_out
    assert groups["policy_b"][0].values_out is values_out


def test_group_central_opponent_rows_preserves_actor_entry_order_within_policy() -> None:
    actor_a = _actor(focal=[0, 0], policies=["policy_x", "policy_x"])
    actor_b = _actor(focal=[1, 1], policies=["policy_x", "policy_y"])

    groups = group_central_opponent_rows(
        actors=[actor_a, actor_b],
        batches=[SimpleNamespace(name="a"), SimpleNamespace(name="b")],
        obs_steps=[np.zeros((2, 1), dtype=np.float32), np.ones((2, 1), dtype=np.float32)],
        actor_steps=[np.asarray([1, 1], dtype=np.int64), np.asarray([0, 0], dtype=np.int64)],
        logits_outs=[None, None],
        values_outs=[np.zeros((2,), dtype=np.float32), np.zeros((2,), dtype=np.float32)],
    )

    assert [entry.actor for entry in groups["policy_x"]] == [actor_a, actor_b]
    assert groups["policy_x"][0].row_indices.tolist() == [0, 1]
    assert groups["policy_x"][1].row_indices.tolist() == [0]
    assert groups["policy_y"][0].actor is actor_b
    assert groups["policy_y"][0].row_indices.tolist() == [1]


def test_group_central_opponent_rows_returns_empty_when_all_rows_are_focal_or_mirror() -> None:
    actor = _actor(focal=[0, 1], policies=[MIRROR_OPPONENT_POLICY_ID, "policy_x"])

    groups = group_central_opponent_rows(
        actors=[actor],
        batches=[object()],
        obs_steps=[np.zeros((2, 1), dtype=np.float32)],
        actor_steps=[np.asarray([0, 1], dtype=np.int64)],
        logits_outs=[None],
        values_outs=[np.zeros((2,), dtype=np.float32)],
    )

    assert groups == {}


def test_group_central_opponent_rows_preserves_strict_zip_mismatch_failure() -> None:
    actor = _actor(focal=[0], policies=["policy_x"])

    with pytest.raises(ValueError):
        group_central_opponent_rows(
            actors=[actor],
            batches=[],
            obs_steps=[np.zeros((1, 1), dtype=np.float32)],
            actor_steps=[np.asarray([1], dtype=np.int64)],
            logits_outs=[None],
            values_outs=[np.zeros((1,), dtype=np.float32)],
        )
