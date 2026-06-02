"""Pending-unroll selection helpers for queue runtime batches."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def actor_id_is_diverse_lane(*, actor_id: int, diverse_opponent_actor_count: int) -> bool:
    return int(actor_id) < int(diverse_opponent_actor_count)


def pending_unroll_is_diverse_lane(item: Any, *, diverse_opponent_actor_count: int) -> bool:
    return actor_id_is_diverse_lane(
        actor_id=int(item.actor_id),
        diverse_opponent_actor_count=diverse_opponent_actor_count,
    )


def pending_diverse_unroll_count(
    pending_unrolls: Sequence[Any],
    *,
    diverse_opponent_actor_count: int,
) -> int:
    return int(
        sum(
            1
            for item in pending_unrolls
            if pending_unroll_is_diverse_lane(
                item,
                diverse_opponent_actor_count=diverse_opponent_actor_count,
            )
        )
    )


def diverse_batch_target_count(
    *,
    batch_size: int,
    diverse_opponent_actor_count: int,
    diverse_opponent_batch_fraction: float,
) -> int:
    if int(batch_size) <= 0:
        return 0
    if int(diverse_opponent_actor_count) <= 0:
        return 0
    fraction = max(0.0, min(1.0, float(diverse_opponent_batch_fraction)))
    if fraction <= 0.0:
        return 0
    target = int(np.ceil(float(batch_size) * fraction))
    return max(1, min(int(batch_size), target))


def select_pending_unrolls(
    pending_unrolls: Sequence[Any],
    *,
    batch_size: int,
    mode: str,
    diverse_opponent_actor_count: int,
    diverse_opponent_batch_fraction: float,
) -> list[Any]:
    if str(mode) != "train_ordered":
        pending = list(pending_unrolls)
        diverse_target = diverse_batch_target_count(
            batch_size=batch_size,
            diverse_opponent_actor_count=diverse_opponent_actor_count,
            diverse_opponent_batch_fraction=diverse_opponent_batch_fraction,
        )
        if diverse_target <= 0:
            return pending[:batch_size]
        diverse_pending = [
            item
            for item in pending
            if pending_unroll_is_diverse_lane(
                item,
                diverse_opponent_actor_count=diverse_opponent_actor_count,
            )
        ]
        regular_pending = [
            item
            for item in pending
            if not pending_unroll_is_diverse_lane(
                item,
                diverse_opponent_actor_count=diverse_opponent_actor_count,
            )
        ]
        selected_diverse = diverse_pending[:diverse_target]
        selected = list(selected_diverse)
        remaining = int(batch_size) - len(selected)
        if remaining > 0:
            selected.extend(regular_pending[:remaining])
        if len(selected) < int(batch_size):
            diverse_cursor = len(selected_diverse)
            selected.extend(diverse_pending[diverse_cursor : diverse_cursor + (int(batch_size) - len(selected))])
        return selected[:batch_size]

    ordered = sorted(
        pending_unrolls,
        key=lambda item: (item.behavior_policy_version, item.unroll_seq, item.actor_id),
    )
    if not ordered:
        raise RuntimeError("train_ordered selection requires at least one pending unroll")
    oldest_version = int(ordered[0].behavior_policy_version)
    ordered_selected: list[Any] = []
    current_group: list[Any] = []
    current_seq: int | None = None
    for item in ordered:
        if int(item.behavior_policy_version) != oldest_version:
            break
        if current_seq is None or int(item.unroll_seq) == current_seq:
            current_group.append(item)
            current_seq = int(item.unroll_seq)
            continue
        if len(ordered_selected) + len(current_group) > int(batch_size):
            break
        ordered_selected.extend(current_group)
        current_group = [item]
        current_seq = int(item.unroll_seq)
    if current_group and len(ordered_selected) + len(current_group) <= int(batch_size):
        ordered_selected.extend(current_group)
    if not ordered_selected:
        raise RuntimeError("train_ordered selection could not produce a same-version batch")
    return ordered_selected
