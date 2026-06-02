"""Actor scheduling helpers for queue runtime collection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def next_actor_batch(
    actors: Sequence[T],
    *,
    next_actor_index: int,
    count: int,
) -> tuple[list[T], int]:
    """Select a round-robin actor batch and return the next cursor."""

    if count <= 0:
        return [], int(next_actor_index)
    actor_total = len(actors)
    if actor_total <= 0:
        return [], 0

    actor_batch: list[T] = []
    cursor = int(next_actor_index) % actor_total
    batch_size = min(int(count), actor_total)
    for _ in range(batch_size):
        actor_batch.append(actors[cursor])
        cursor = (cursor + 1) % actor_total
    return actor_batch, cursor
