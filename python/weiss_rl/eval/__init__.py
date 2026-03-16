"""Evaluation harness entry points."""

from weiss_rl.eval.harness import (
    EvalGameRecord,
    EvalRunResult,
    EvalSamplerAnomalies,
    GameResult,
    MatchupSummary,
    ScheduledGame,
    build_seat_swapped_schedule,
    game_result_from_step,
    record_completed_game,
    run_seat_swapped_matchup,
    sample_action_pinned,
    summarize_game_records,
    summarize_pair_outcomes,
    write_episodes_jsonl,
)
from weiss_rl.eval.payoff_folding import PayoffFoldScheme, fold_game_payoff, paired_seed_mean_score, paired_seed_score
from weiss_rl.eval.rng_pcg32 import NEXT_U64_ORDER, PCG32_XSH_RR_V1, Pcg32XshRrV1

__all__ = [
    "EvalGameRecord",
    "EvalRunResult",
    "EvalSamplerAnomalies",
    "GameResult",
    "MatchupSummary",
    "NEXT_U64_ORDER",
    "PCG32_XSH_RR_V1",
    "PayoffFoldScheme",
    "Pcg32XshRrV1",
    "ScheduledGame",
    "build_seat_swapped_schedule",
    "fold_game_payoff",
    "game_result_from_step",
    "paired_seed_mean_score",
    "paired_seed_score",
    "record_completed_game",
    "run_seat_swapped_matchup",
    "sample_action_pinned",
    "summarize_game_records",
    "summarize_pair_outcomes",
    "write_episodes_jsonl",
]
