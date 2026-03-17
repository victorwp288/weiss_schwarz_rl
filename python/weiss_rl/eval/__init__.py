"""Evaluation harness entry points."""

from weiss_rl.eval.diagnostics import build_seat_advantage_diagnostics, write_matchup_diagnostics_json
from weiss_rl.eval.export import build_matchup_export, load_eval_game_records, write_matchup_summary_csv, write_matchup_summary_json
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
from weiss_rl.eval.payoff_folding import (
    PayoffFoldScheme,
    fold_game_payoff,
    paired_seed_mean_score,
    paired_seed_score,
    paired_seed_scores,
)
from weiss_rl.eval.rng_pcg32 import NEXT_U64_ORDER, PCG32_XSH_RR_V1, Pcg32XshRrV1
from weiss_rl.eval.stage2 import Stage2Decision, Stage2StopReason, summarize_stage2_records
from weiss_rl.eval.uncertainty import (
    EvalUncertaintySummary,
    bayesian_bootstrap_summary,
    paired_seed_uncertainty_summary,
)

__all__ = [
    "EvalGameRecord",
    "EvalRunResult",
    "EvalSamplerAnomalies",
    "EvalUncertaintySummary",
    "GameResult",
    "MatchupSummary",
    "NEXT_U64_ORDER",
    "PCG32_XSH_RR_V1",
    "PayoffFoldScheme",
    "Pcg32XshRrV1",
    "ScheduledGame",
    "Stage2Decision",
    "Stage2StopReason",
    "bayesian_bootstrap_summary",
    "build_matchup_export",
    "build_seat_advantage_diagnostics",
    "build_seat_swapped_schedule",
    "fold_game_payoff",
    "game_result_from_step",
    "load_eval_game_records",
    "paired_seed_mean_score",
    "paired_seed_score",
    "paired_seed_scores",
    "paired_seed_uncertainty_summary",
    "record_completed_game",
    "run_seat_swapped_matchup",
    "sample_action_pinned",
    "summarize_game_records",
    "summarize_pair_outcomes",
    "summarize_stage2_records",
    "write_episodes_jsonl",
    "write_matchup_diagnostics_json",
    "write_matchup_summary_csv",
    "write_matchup_summary_json",
]
