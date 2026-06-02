"""Final-eval orchestration for the deterministic final policy set."""

from __future__ import annotations

from weiss_rl.eval.final import artifacts as _artifacts
from weiss_rl.eval.final import matchups as _matchups
from weiss_rl.eval.final import payload as _payload
from weiss_rl.eval.final import policy_selection as _selection
from weiss_rl.eval.final import run as _run

__all__ = [
    "load_dev_eval_summaries",
    "resolve_final_policy_set",
    "run_final_eval",
]

load_dev_eval_summaries = _selection.load_dev_eval_summaries
resolve_final_policy_set = _selection.resolve_final_policy_set

_bootstrap_seed = _matchups.bootstrap_seed
_build_final_eval_payload = _payload.build_final_eval_payload
_build_matchup_payload = _matchups.build_matchup_payload
_matchup_dir_name = _matchups.matchup_dir_name
_matchup_posterior_samples = _matchups.matchup_posterior_samples
_resolve_policy_ids = _selection.resolve_final_eval_policy_ids
_run_matchup = _matchups.run_final_eval_matchup
_scheduled_game = _matchups.scheduled_game
_slug = _matchups.slug
_validate_seed_budget = _selection.validate_final_eval_seed_budget
_write_final_eval_artifacts = _artifacts.write_final_eval_artifacts
_build_matchup_jobs = _run.build_final_eval_matchup_jobs
_build_run_payload = _run.build_final_eval_run_payload
_resolve_run_policy_ids = _run.resolve_final_eval_run_policy_ids
_run_matchup_jobs = _run.run_final_eval_matchup_jobs
_validate_run_seed_budget = _run.validate_final_eval_run_seed_budget
_write_run_artifacts = _run.write_final_eval_run_artifacts
run_final_eval = _run.run_final_eval
