from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np

from weiss_rl.runtime.components.policy_ids import MIRROR_OPPONENT_POLICY_ID


class CentralActorRowPartition(NamedTuple):
    focal_rows: np.ndarray
    heuristic_rows: np.ndarray
    mirror_rows: np.ndarray
    residual_rows: np.ndarray
    sampled_policy_rows: np.ndarray

    @property
    def opponent_row_count(self) -> int:
        return int(self.heuristic_rows.shape[0] + self.mirror_rows.shape[0] + self.residual_rows.shape[0])


class CentralActorRowPartitions(NamedTuple):
    entries: list[CentralActorRowPartition]

    @property
    def focal_rows_by_actor(self) -> list[np.ndarray]:
        return [entry.focal_rows for entry in self.entries]

    @property
    def heuristic_rows_by_actor(self) -> list[np.ndarray]:
        return [entry.heuristic_rows for entry in self.entries]

    @property
    def residual_rows_by_actor(self) -> list[np.ndarray]:
        return [entry.residual_rows for entry in self.entries]

    @property
    def sampled_policy_rows_by_actor(self) -> list[np.ndarray]:
        return [entry.sampled_policy_rows for entry in self.entries]


def partition_central_actor_rows(
    *,
    actors: Sequence[Any],
    actor_steps: Sequence[np.ndarray],
    heuristic_policy_ids: Sequence[str],
    fuse_mirror_policy_rows: bool,
    mirror_policy_id: str = MIRROR_OPPONENT_POLICY_ID,
) -> CentralActorRowPartitions:
    heuristic_policy_id_set = {str(policy_id) for policy_id in heuristic_policy_ids}
    entries: list[CentralActorRowPartition] = []
    for actor, actor_step in zip(actors, actor_steps, strict=True):
        actor_step_array = np.asarray(actor_step, dtype=np.int64)
        focal_seat_by_env = np.asarray(actor.focal_seat_by_env, dtype=np.int64)
        focal_rows = np.flatnonzero(actor_step_array == focal_seat_by_env)
        opponent_rows = np.flatnonzero(actor_step_array != focal_seat_by_env)
        if opponent_rows.size == 0:
            heuristic_rows = np.zeros((0,), dtype=np.int64)
            mirror_rows = np.zeros((0,), dtype=np.int64)
            residual_rows = np.zeros((0,), dtype=np.int64)
        else:
            opponent_policy_ids = np.asarray(actor.opponent_policy_id_by_env[opponent_rows], dtype=object)
            heuristic_mask = (
                np.isin(opponent_policy_ids, tuple(heuristic_policy_id_set))
                if heuristic_policy_id_set
                else np.zeros(opponent_policy_ids.shape, dtype=np.bool_)
            )
            mirror_mask = opponent_policy_ids == str(mirror_policy_id)
            heuristic_rows = opponent_rows[heuristic_mask]
            mirror_rows = opponent_rows[mirror_mask]
            residual_rows = opponent_rows[~(heuristic_mask | mirror_mask)]
        sampled_policy_rows = (
            np.concatenate((focal_rows, mirror_rows), axis=0).astype(np.int64, copy=False)
            if bool(fuse_mirror_policy_rows) and mirror_rows.size > 0
            else focal_rows
        )
        entries.append(
            CentralActorRowPartition(
                focal_rows=focal_rows,
                heuristic_rows=heuristic_rows,
                mirror_rows=mirror_rows,
                residual_rows=residual_rows,
                sampled_policy_rows=sampled_policy_rows,
            )
        )
    return CentralActorRowPartitions(entries=entries)
