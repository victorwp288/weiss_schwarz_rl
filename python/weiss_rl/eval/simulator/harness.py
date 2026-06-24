"""Deterministic evaluation harness and pinned sampling helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from weiss_rl.eval.sampling import action_sampling as _action_sampling
from weiss_rl.eval.sampling.action_sampling import (
    _legal_probs_for_cdf as _legal_probs_for_cdf,
)
from weiss_rl.eval.sampling.action_sampling import (
    _normalize_cdf_probs as _normalize_cdf_probs,
)
from weiss_rl.eval.sampling.action_sampling import (
    eval_sampler_logp_from_legal_ids as eval_sampler_logp_from_legal_ids,
)
from weiss_rl.eval.sampling.action_sampling import (
    eval_sampler_logp_from_mask as eval_sampler_logp_from_mask,
)
from weiss_rl.eval.sampling.action_sampling import (
    select_action_argmax_pinned as select_action_argmax_pinned,
)
from weiss_rl.eval.simulator.completed_games import (
    outcome_for_focal as outcome_for_focal,
)
from weiss_rl.eval.simulator.completed_games import (
    record_completed_game as record_completed_game,
)
from weiss_rl.eval.simulator.completed_games import (
    resolve_eval_episode_key as resolve_eval_episode_key,
)
from weiss_rl.eval.simulator.completed_games import (
    resolve_eval_episode_key256 as resolve_eval_episode_key256,
)
from weiss_rl.eval.simulator.completed_games import (
    summarize_game_records as summarize_game_records,
)
from weiss_rl.eval.simulator.completed_games import (
    summarize_pair_outcomes as summarize_pair_outcomes,
)
from weiss_rl.eval.simulator.completed_games import (
    write_episodes_jsonl as write_episodes_jsonl,
)
from weiss_rl.eval.simulator.engine_faults import abort_on_engine_fault_eval as _abort_on_engine_fault_eval
from weiss_rl.eval.simulator.records import (
    EvalGameRecord as EvalGameRecord,
)
from weiss_rl.eval.simulator.records import (
    EvalGameRunner as EvalGameRunner,
)
from weiss_rl.eval.simulator.records import (
    EvalRunResult as EvalRunResult,
)
from weiss_rl.eval.simulator.records import (
    GameResult as GameResult,
)
from weiss_rl.eval.simulator.records import (
    MatchupSummary as MatchupSummary,
)
from weiss_rl.eval.simulator.records import (
    OutcomeToken as OutcomeToken,
)
from weiss_rl.eval.simulator.records import (
    ReplaySampleResult as ReplaySampleResult,
)
from weiss_rl.eval.simulator.records import (
    ScheduledGame as ScheduledGame,
)
from weiss_rl.eval.simulator.seat_swapped import build_seat_swapped_schedule as build_seat_swapped_schedule
from weiss_rl.eval.simulator.seat_swapped import run_seat_swapped_matchup as run_seat_swapped_matchup
from weiss_rl.eval.simulator.terminal_result import game_result_from_step as game_result_from_step


def sample_action_pinned(*args: Any, **kwargs: Any) -> tuple[int, object]:
    """Compatibility wrapper around action_sampling.sample_action_pinned."""
    original_legal_probs_for_cdf = _action_sampling._legal_probs_for_cdf
    _action_sampling._legal_probs_for_cdf = _legal_probs_for_cdf
    try:
        return _action_sampling.sample_action_pinned(*args, **kwargs)
    finally:
        _action_sampling._legal_probs_for_cdf = original_legal_probs_for_cdf


def abort_on_engine_fault_eval(
    *,
    run_dir: Path,
    engine_status: Any,
    decision_id: Any | None = None,
    episode_key: Any | None = None,
    note: str = "engine_status!=0 during evaluation",
) -> None:
    _abort_on_engine_fault_eval(
        run_dir=run_dir,
        engine_status=engine_status,
        decision_id=decision_id,
        episode_key=episode_key,
        note=note,
    )
