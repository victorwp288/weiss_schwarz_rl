"""Evaluation harness entry points."""

from importlib import import_module

from weiss_rl.eval.diagnostics import (
    HiddenInfoLeakagePair,
    build_hidden_info_leakage_diagnostics,
    build_seat_advantage_diagnostics,
    write_leakage_diagnostics_json,
    write_matchup_diagnostics_json,
)
from weiss_rl.eval.export import (
    build_matchup_export,
    load_eval_game_records,
    write_matchup_summary_csv,
    write_matchup_summary_json,
)
from weiss_rl.eval.final_eval import load_dev_eval_summaries, resolve_final_policy_set, run_final_eval
from weiss_rl.eval.harness import (
    EvalGameRecord,
    EvalRunResult,
    EvalSamplerAnomalies,
    GameResult,
    MatchupSummary,
    ReplaySampleResult,
    ScheduledGame,
    build_seat_swapped_schedule,
    game_result_from_step,
    record_completed_game,
    run_seat_swapped_matchup,
    sample_action_pinned,
    select_action_argmax_pinned,
    summarize_game_records,
    summarize_pair_outcomes,
    write_episodes_jsonl,
)
from weiss_rl.eval.paper_readiness import (
    build_paper_readiness_summary,
    write_paper_readiness_json,
)
from weiss_rl.eval.payoff_folding import (
    PayoffFoldScheme,
    fold_game_payoff,
    paired_seed_mean_score,
    paired_seed_score,
    paired_seed_scores,
)
from weiss_rl.eval.policies.set import (
    DevEvalPolicySummary,
    parse_training_policy_id,
    recommend_focal_policy_id,
    select_final_policy_set_deterministic_v1,
)
from weiss_rl.eval.rng_pcg32 import NEXT_U64_ORDER, PCG32_XSH_RR_V1, Pcg32XshRrV1
from weiss_rl.eval.stage2 import Stage2Decision, Stage2StopReason, summarize_stage2_records
from weiss_rl.eval.uncertainty import (
    EvalUncertaintySummary,
    bayesian_bootstrap_posterior_samples,
    bayesian_bootstrap_summary,
    paired_seed_uncertainty_summary,
    posterior_samples,
)

_LEGACY_EVAL_MODULES = {
    "final_eval_artifacts": "final.artifacts",
    "final_eval_matchup_outputs": "final.matchup_outputs",
    "final_eval_matchup_schedule": "final.matchup_schedule",
    "final_eval_matchups": "final.matchups",
    "final_eval_matrices": "final.matrices",
    "final_eval_payload": "final.payload",
    "final_eval_policy_selection": "final.policy_selection",
    "final_eval_run": "final.run",
    "final_eval_worker": "final.worker",
    "final_eval_worker_runtime": "final.worker_runtime",
    "paper_readiness_check_cli": "readiness.check_cli",
    "paper_readiness_check_entrypoint": "readiness.check_entrypoint",
    "paper_readiness_check_reporting": "readiness.check_reporting",
    "paper_readiness_check_runtime": "readiness.check_runtime",
    "paper_readiness_contracts": "readiness.contracts",
    "paper_readiness_fields": "readiness.fields",
    "paper_readiness_final_eval_summary": "readiness.final_eval_summary",
    "paper_readiness_fixture": "readiness.fixture",
    "paper_readiness_fixture_cli": "readiness.fixture_cli",
    "paper_readiness_fixture_entrypoint": "readiness.fixture_entrypoint",
    "paper_readiness_fixture_writer": "readiness.fixture_writer",
    "paper_readiness_guardrails": "readiness.guardrails",
    "policy_alignment": "policies.alignment",
    "policy_resolution": "policies.resolution",
    "policy_resolution_context": "policies.resolution_context",
    "policy_set": "policies.set",
    "policy_types": "policies.types",
    "targeted_confirm_core": "targeted_confirm.core",
    "targeted_confirm_entrypoint": "targeted_confirm.entrypoint",
    "targeted_confirm_jobs": "targeted_confirm.jobs",
    "targeted_confirm_opponents": "targeted_confirm.opponents",
    "targeted_confirm_plan": "targeted_confirm.plan",
    "targeted_confirm_summary": "targeted_confirm.summary",
}


def __getattr__(name: str):
    if name in _LEGACY_EVAL_MODULES:
        module = import_module(f".{_LEGACY_EVAL_MODULES[name]}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DevEvalPolicySummary",
    "EvalGameRecord",
    "EvalRunResult",
    "EvalSamplerAnomalies",
    "EvalUncertaintySummary",
    "GameResult",
    "HiddenInfoLeakagePair",
    "MatchupSummary",
    "NEXT_U64_ORDER",
    "PCG32_XSH_RR_V1",
    "PayoffFoldScheme",
    "Pcg32XshRrV1",
    "ReplaySampleResult",
    "ScheduledGame",
    "Stage2Decision",
    "Stage2StopReason",
    "bayesian_bootstrap_posterior_samples",
    "bayesian_bootstrap_summary",
    "build_hidden_info_leakage_diagnostics",
    "build_matchup_export",
    "build_paper_readiness_summary",
    "build_seat_advantage_diagnostics",
    "build_seat_swapped_schedule",
    "fold_game_payoff",
    "game_result_from_step",
    "load_eval_game_records",
    "paired_seed_mean_score",
    "paired_seed_score",
    "paired_seed_scores",
    "load_dev_eval_summaries",
    "parse_training_policy_id",
    "paired_seed_uncertainty_summary",
    "posterior_samples",
    "recommend_focal_policy_id",
    "resolve_final_policy_set",
    "run_final_eval",
    "select_final_policy_set_deterministic_v1",
    "record_completed_game",
    "run_seat_swapped_matchup",
    "sample_action_pinned",
    "select_action_argmax_pinned",
    "summarize_game_records",
    "summarize_pair_outcomes",
    "summarize_stage2_records",
    "write_episodes_jsonl",
    "write_leakage_diagnostics_json",
    "write_matchup_diagnostics_json",
    "write_matchup_summary_csv",
    "write_matchup_summary_json",
    "write_paper_readiness_json",
]
