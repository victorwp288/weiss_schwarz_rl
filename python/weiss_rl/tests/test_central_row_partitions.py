from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from weiss_rl.runtime.components.central_row_partitions import partition_central_actor_rows
from weiss_rl.runtime.components.policy_ids import MIRROR_OPPONENT_POLICY_ID


def _actor(*, focal: list[int], policies: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        focal_seat_by_env=np.asarray(focal, dtype=np.int64),
        opponent_policy_id_by_env=np.asarray(policies, dtype=object),
    )


def test_partition_central_actor_rows_splits_focal_heuristic_mirror_and_residual_rows() -> None:
    actor = _actor(
        focal=[0, 0, 1, 1, 0],
        policies=[
            "unused_focal",
            "B2 HeuristicPublic",
            MIRROR_OPPONENT_POLICY_ID,
            "policy_snapshot",
            "also_focal",
        ],
    )

    partitions = partition_central_actor_rows(
        actors=[actor],
        actor_steps=[np.asarray([0, 1, 0, 0, 0], dtype=np.int64)],
        heuristic_policy_ids=["B2 HeuristicPublic"],
        fuse_mirror_policy_rows=True,
    )

    partition = partitions.entries[0]
    assert partition.focal_rows.tolist() == [0, 4]
    assert partition.heuristic_rows.tolist() == [1]
    assert partition.mirror_rows.tolist() == [2]
    assert partition.residual_rows.tolist() == [3]
    assert partition.sampled_policy_rows.tolist() == [0, 4, 2]
    assert partition.opponent_row_count == 3
    assert [rows.tolist() for rows in partitions.sampled_policy_rows_by_actor] == [[0, 4, 2]]
    assert [rows.tolist() for rows in partitions.heuristic_rows_by_actor] == [[1]]
    assert [rows.tolist() for rows in partitions.residual_rows_by_actor] == [[3]]


def test_partition_central_actor_rows_can_leave_mirror_rows_for_opponent_routing() -> None:
    actor = _actor(
        focal=[0, 0, 0],
        policies=["unused_focal", MIRROR_OPPONENT_POLICY_ID, "policy_snapshot"],
    )

    partition = partition_central_actor_rows(
        actors=[actor],
        actor_steps=[np.asarray([0, 1, 1], dtype=np.int64)],
        heuristic_policy_ids=[],
        fuse_mirror_policy_rows=False,
    ).entries[0]

    assert partition.focal_rows.tolist() == [0]
    assert partition.mirror_rows.tolist() == [1]
    assert partition.residual_rows.tolist() == [2]
    assert partition.sampled_policy_rows.tolist() == [0]
    assert partition.opponent_row_count == 2


def test_partition_central_actor_rows_preserves_actor_order_and_empty_rows() -> None:
    actor_a = _actor(focal=[0, 1], policies=["a", "b"])
    actor_b = _actor(focal=[0, 0], policies=["policy_h", MIRROR_OPPONENT_POLICY_ID])

    partitions = partition_central_actor_rows(
        actors=[actor_a, actor_b],
        actor_steps=[
            np.asarray([0, 1], dtype=np.int64),
            np.asarray([1, 1], dtype=np.int64),
        ],
        heuristic_policy_ids=["policy_h"],
        fuse_mirror_policy_rows=True,
    )

    first, second = partitions.entries
    assert first.focal_rows.tolist() == [0, 1]
    assert first.heuristic_rows.tolist() == []
    assert first.mirror_rows.tolist() == []
    assert first.residual_rows.tolist() == []
    assert first.sampled_policy_rows.tolist() == [0, 1]
    assert second.focal_rows.tolist() == []
    assert second.heuristic_rows.tolist() == [0]
    assert second.mirror_rows.tolist() == [1]
    assert second.residual_rows.tolist() == []
    assert second.sampled_policy_rows.tolist() == [1]


def test_partition_central_actor_rows_preserves_strict_zip_mismatch_failure() -> None:
    actor = _actor(focal=[0], policies=["policy"])

    with pytest.raises(ValueError):
        partition_central_actor_rows(
            actors=[actor],
            actor_steps=[],
            heuristic_policy_ids=[],
            fuse_mirror_policy_rows=True,
        )
