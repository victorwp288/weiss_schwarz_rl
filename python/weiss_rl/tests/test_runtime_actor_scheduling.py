from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from weiss_rl.runtime import QueueRuntime
from weiss_rl.runtime_components.actor_scheduling import next_actor_batch


def _actor(actor_id: int) -> SimpleNamespace:
    return SimpleNamespace(actor_id=actor_id)


def test_next_actor_batch_returns_empty_without_advancing_for_non_positive_count() -> None:
    actors = [_actor(0), _actor(1)]

    selected, cursor = next_actor_batch(actors, next_actor_index=1, count=0)

    assert selected == []
    assert cursor == 1


def test_next_actor_batch_wraps_cursor_and_caps_at_actor_count() -> None:
    actors = [_actor(0), _actor(1), _actor(2)]

    selected, cursor = next_actor_batch(actors, next_actor_index=2, count=5)

    assert [actor.actor_id for actor in selected] == [2, 0, 1]
    assert cursor == 2


def test_next_actor_batch_handles_empty_actor_sequence() -> None:
    selected, cursor = next_actor_batch(cast(list[SimpleNamespace], []), next_actor_index=4, count=3)

    assert selected == []
    assert cursor == 0


def test_queue_runtime_next_actor_batch_preserves_cursor_state() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._actors = [_actor(0), _actor(1), _actor(2)]
    runtime_any._next_actor_index = 1

    selected = QueueRuntime._next_actor_batch(runtime, 2)

    assert [actor.actor_id for actor in selected] == [1, 2]
    assert runtime_any._next_actor_index == 0
