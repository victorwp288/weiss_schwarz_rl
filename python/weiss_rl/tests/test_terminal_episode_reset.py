from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import weiss_rl.runtime_components.terminal_episode_reset as terminal_reset_module
from weiss_rl.runtime_components.counters import collector_counter_template
from weiss_rl.runtime_components.terminal_episode_reset import reset_terminal_episode_rows


def test_reset_terminal_episode_rows_preserves_terminal_flow_order(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    counters = collector_counter_template()
    actor = SimpleNamespace(name="actor")
    next_batch = SimpleNamespace(name="next")
    action_sequence_state = object()
    reset_batch = object()
    done = np.asarray([True, False, True], dtype=np.bool_)
    acting_seat = np.asarray([0, 1, 1], dtype=np.int64)

    def fake_accumulate_timeout_counters(**kwargs) -> None:
        calls.append(("timeouts", kwargs["done"].copy()))
        assert kwargs["counters"] is counters
        assert kwargs["batch"] is next_batch

    def fake_update_outcomes(**kwargs) -> None:
        calls.append(("outcomes", kwargs["done"].copy()))
        assert kwargs["actor"] is actor
        assert kwargs["acting_seat"] is acting_seat
        assert kwargs["terminal_batch"] is next_batch
        assert kwargs["counters"] is counters

    def fake_assign_episode_roles(actor_arg, done_arg, *, counters: dict[str, int]) -> None:
        calls.append(("roles", done_arg.copy()))
        assert actor_arg is actor
        assert counters is counters_ref

    def fake_reset_actor_hidden_for_done(**kwargs):
        calls.append(("hidden", kwargs["done"].copy()))
        assert kwargs["actor"] is actor
        return SimpleNamespace(done=kwargs["done"].copy(), done_count=2)

    def fake_reset_action_sequence_state(state, done_arg) -> None:
        calls.append(("action_state", done_arg.copy()))
        assert state is action_sequence_state

    def fake_reset_done_rows(actor_arg, done_arg):
        calls.append(("reset_rows", done_arg.copy()))
        assert actor_arg is actor
        return reset_batch

    counters_ref = counters
    monkeypatch.setattr(terminal_reset_module, "accumulate_timeout_counters", fake_accumulate_timeout_counters)
    monkeypatch.setattr(terminal_reset_module, "reset_actor_hidden_for_done", fake_reset_actor_hidden_for_done)
    monkeypatch.setattr(terminal_reset_module, "reset_action_sequence_state", fake_reset_action_sequence_state)

    returned = reset_terminal_episode_rows(
        actor=actor,
        next_batch=next_batch,
        acting_seat=acting_seat,
        done=done,
        counters=counters,
        timeout_limits={"max_decisions": None, "max_ticks": None, "max_no_progress_decisions": None},
        action_sequence_state=action_sequence_state,
        device="cpu",
        update_outcomes=fake_update_outcomes,
        assign_episode_roles=fake_assign_episode_roles,
        reset_done_rows=fake_reset_done_rows,
    )

    assert returned is reset_batch
    assert [name for name, _ in calls] == [
        "timeouts",
        "outcomes",
        "roles",
        "hidden",
        "action_state",
        "reset_rows",
    ]
    assert all(np.array_equal(done_arg, done) for _, done_arg in calls)
    assert counters["actor_done_reset_ms"] >= 0
