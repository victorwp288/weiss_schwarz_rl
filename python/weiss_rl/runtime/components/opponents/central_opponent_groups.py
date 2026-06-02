from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np

from weiss_rl.runtime.components.policy_ids import MIRROR_OPPONENT_POLICY_ID


class CentralOpponentEntry(NamedTuple):
    actor: Any
    batch: Any
    row_indices: np.ndarray
    obs_step: np.ndarray
    actor_step: np.ndarray
    logits_out: np.ndarray | None
    values_out: np.ndarray


def group_central_opponent_rows(
    *,
    actors: Sequence[Any],
    batches: Sequence[Any],
    obs_steps: Sequence[np.ndarray],
    actor_steps: Sequence[np.ndarray],
    logits_outs: Sequence[np.ndarray | None],
    values_outs: Sequence[np.ndarray],
) -> dict[str, list[CentralOpponentEntry]]:
    policy_groups: dict[str, list[CentralOpponentEntry]] = {}
    for actor, batch, obs_step, actor_step, logits_out, values_out in zip(
        actors,
        batches,
        obs_steps,
        actor_steps,
        logits_outs,
        values_outs,
        strict=True,
    ):
        focal_rows = actor_step == actor.focal_seat_by_env
        opponent_indices = np.flatnonzero(~focal_rows)
        if opponent_indices.size == 0:
            continue
        for policy_id in sorted({str(actor.opponent_policy_id_by_env[index]) for index in opponent_indices.tolist()}):
            if policy_id == MIRROR_OPPONENT_POLICY_ID:
                continue
            policy_rows = opponent_indices[actor.opponent_policy_id_by_env[opponent_indices] == policy_id]
            if not policy_rows.size:
                continue
            policy_groups.setdefault(policy_id, []).append(
                CentralOpponentEntry(
                    actor=actor,
                    batch=batch,
                    row_indices=policy_rows,
                    obs_step=obs_step,
                    actor_step=actor_step,
                    logits_out=logits_out,
                    values_out=values_out,
                )
            )
    return policy_groups
