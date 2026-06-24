"""Policy selection and matchup planning for canonical final evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.config.models import FinalPolicySetSelectionConfig
from weiss_rl.eval.final.matchup_jobs import FinalEvalMatchupJob, build_final_eval_matchup_jobs
from weiss_rl.eval.final.policy_selection import resolve_final_eval_policy_ids, validate_final_eval_seed_budget


@dataclass(frozen=True, slots=True)
class FinalEvalRunPlan:
    policy_ids: list[str]
    selection_payload: Mapping[str, Any]
    matchup_jobs: tuple[FinalEvalMatchupJob, ...]


def build_final_eval_run_plan(
    *,
    policy_ids: Sequence[str] | None,
    snapshot_registry_path: Path | None,
    dev_eval_summaries_path: Path | None,
    selection_config: FinalPolicySetSelectionConfig | None,
    final_policy_set_size: int | None,
    paired_seeds: Sequence[int],
    stage1_paired_seeds: int,
    max_paired_seeds: int,
) -> FinalEvalRunPlan:
    """Resolve the final policy panel and the canonical upper-triangle matchup plan."""

    resolved_policy_ids, selection_payload = resolve_final_eval_policy_ids(
        policy_ids=policy_ids,
        snapshot_registry_path=snapshot_registry_path,
        dev_eval_summaries_path=dev_eval_summaries_path,
        selection_config=selection_config,
        final_policy_set_size=final_policy_set_size,
    )
    validate_final_eval_seed_budget(
        paired_seeds=paired_seeds,
        stage1_paired_seeds=stage1_paired_seeds,
        max_paired_seeds=max_paired_seeds,
    )
    return FinalEvalRunPlan(
        policy_ids=resolved_policy_ids,
        selection_payload=selection_payload,
        matchup_jobs=build_final_eval_matchup_jobs(resolved_policy_ids),
    )


__all__ = ["FinalEvalRunPlan", "build_final_eval_run_plan"]
