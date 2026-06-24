"""Convert terminal environment rows into evaluation result records."""

from __future__ import annotations

from weiss_rl.core.termination_reason import classify_episode_end_reason
from weiss_rl.eval.simulator.records import GameResult
from weiss_rl.eval.simulator.terminal_step import MISSING as _MISSING
from weiss_rl.eval.simulator.terminal_step import optional_step_scalar as _optional_step_scalar
from weiss_rl.eval.simulator.terminal_step import (
    required_step_scalar_with_fallback as _required_step_scalar_with_fallback,
)
from weiss_rl.eval.simulator.terminal_step import step_scalar as _step_scalar
from weiss_rl.eval.simulator.terminal_step import winner_seat_from_terminal_step as _winner_seat_from_terminal_step


def game_result_from_step(
    step: object,
    *,
    env_index: int = 0,
    acting_seat: int | None = None,
    episode_seed: int | None = None,
    max_decisions: int | None = None,
    max_ticks: int | None = None,
    max_no_progress_decisions: int | None = None,
) -> GameResult:
    """Decode one environment row into an evaluation result.

    Prefer explicit terminal winner metadata when the step exposes it. Otherwise
    decisive terminated rows are inferred from reward sign relative to the
    acting seat, and a terminated zero reward is treated as a draw fallback.
    That zero-reward draw fallback matches the locked thesis configs and should
    be revisited if terminal shaping semantics change.

    Some minimal terminal step objects omit context fields such as acting seat
    or episode seed; callers may supply those explicitly when unavailable on
    the observed row.
    """
    reward = _step_scalar(step, ("reward", "rewards"), env_index=env_index, cast_fn=float)
    terminated = _step_scalar(step, ("terminated",), env_index=env_index, cast_fn=bool)
    truncated = _step_scalar(step, ("truncated",), env_index=env_index, cast_fn=bool)
    engine_status = _step_scalar(step, ("engine_status",), env_index=env_index, cast_fn=int)
    decision_count = _optional_step_scalar(step, ("decision_count",), env_index=env_index)
    tick_count = _optional_step_scalar(step, ("tick_count",), env_index=env_index)
    no_progress_count = _optional_step_scalar(step, ("no_progress_count",), env_index=env_index)
    decision_count_i = 0 if decision_count is None else int(decision_count)
    tick_count_i = 0 if tick_count is None else int(tick_count)
    no_progress_count_i = 0 if no_progress_count is None else int(no_progress_count)
    resolved_episode_seed = _required_step_scalar_with_fallback(
        step,
        ("episode_seed",),
        env_index=env_index,
        cast_fn=int,
        fallback=_MISSING if episode_seed is None else episode_seed,
        fallback_name="episode_seed",
    )
    simulator_episode_key = _optional_step_scalar(step, ("episode_key",), env_index=env_index)

    winner_seat = _winner_seat_from_terminal_step(
        step,
        env_index=env_index,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        acting_seat=acting_seat,
    )
    termination_reason = classify_episode_end_reason(
        terminated=terminated,
        truncated=truncated,
        engine_status=engine_status,
        decision_count=decision_count_i,
        tick_count=tick_count_i,
        no_progress_count=no_progress_count_i,
        max_decisions=max_decisions,
        max_ticks=max_ticks,
        max_no_progress_decisions=max_no_progress_decisions,
    )

    return GameResult(
        episode_seed=resolved_episode_seed,
        terminated=terminated,
        truncated=truncated,
        winner_seat=winner_seat,
        engine_status=engine_status,
        decision_count=decision_count_i,
        tick_count=tick_count_i,
        no_progress_count=no_progress_count_i,
        termination_reason=termination_reason,
        simulator_episode_key=simulator_episode_key,
    )


__all__ = ["game_result_from_step"]
