"""Terminal episode reset flow shared by runtime collectors."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from weiss_rl.diagnostics.action_diagnostics import reset_action_sequence_state
from weiss_rl.runtime.components.counters import accumulate_timeout_counters
from weiss_rl.runtime.components.done_resets import reset_actor_hidden_for_done


def reset_terminal_episode_rows(
    *,
    actor: Any,
    next_batch: Any,
    acting_seat: np.ndarray,
    done: np.ndarray,
    counters: dict[str, int],
    timeout_limits: Any,
    action_sequence_state: Any,
    device: Any,
    update_outcomes: Callable[..., None],
    assign_episode_roles: Callable[..., None],
    reset_done_rows: Callable[..., Any],
) -> Any:
    done_array = np.asarray(done, dtype=np.bool_)
    accumulate_timeout_counters(
        counters=counters,
        batch=next_batch,
        done=done,
        timeout_limits=timeout_limits,
    )
    update_outcomes(
        actor=actor,
        acting_seat=acting_seat,
        terminal_batch=next_batch,
        done=done_array,
        counters=counters,
    )
    reset_started = time.perf_counter()
    assign_episode_roles(actor, done_array, counters=counters)
    reset_result = reset_actor_hidden_for_done(actor=actor, done=done_array, device=device)
    reset_action_sequence_state(action_sequence_state, reset_result.done)
    reset_batch = reset_done_rows(actor, reset_result.done)
    counters["actor_done_reset_ms"] += int((time.perf_counter() - reset_started) * 1000.0)
    return reset_batch
