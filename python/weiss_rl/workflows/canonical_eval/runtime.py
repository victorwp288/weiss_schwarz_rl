from __future__ import annotations

from pathlib import Path
from typing import Any

from weiss_rl.workflows.canonical_eval.policy_runtime import (
    resolve_canonical_eval_policy_runtime,
    resolve_recommended_focal_policy_id,
)
from weiss_rl.workflows.canonical_eval.seed_budget import resolve_canonical_eval_seed_budget
from weiss_rl.workflows.canonical_eval.state import CanonicalEvalRunState, CanonicalEvalRuntimeState


def resolve_canonical_eval_runtime_state(
    *,
    stack: Any,
    run_dir: Path,
    policy_ids: list[str],
    snapshot_registry_path: Path | None,
    dev_eval_summaries_path: Path | None,
    b1_baseline_run_dir: Path | None,
    paired_seed_limit: int | None,
    stage1_paired_seeds: int | None,
    max_paired_seeds: int | None,
    run_state: CanonicalEvalRunState,
    dependencies: Any,
) -> CanonicalEvalRuntimeState:
    policy_runtime = resolve_canonical_eval_policy_runtime(
        stack=stack,
        run_dir=run_dir,
        policy_ids=policy_ids,
        snapshot_registry_path=snapshot_registry_path,
        dev_eval_summaries_path=dev_eval_summaries_path,
        b1_baseline_run_dir=b1_baseline_run_dir,
        run_state=run_state,
        dependencies=dependencies,
    )
    seed_budget = resolve_canonical_eval_seed_budget(
        stack=stack,
        evaluation=run_state.evaluation,
        paired_seed_limit=paired_seed_limit,
        stage1_paired_seeds=stage1_paired_seeds,
        max_paired_seeds=max_paired_seeds,
        dependencies=dependencies,
    )
    recommended_focal_policy_id = resolve_recommended_focal_policy_id(
        policy_ids=policy_runtime.policy_ids,
        snapshot_registry_path=policy_runtime.snapshot_registry_path,
        dev_eval_summaries_path=policy_runtime.dev_eval_summaries_path,
        dependencies=dependencies,
    )

    return CanonicalEvalRuntimeState(
        policy_ids=policy_runtime.policy_ids,
        selection_details=policy_runtime.selection_details,
        snapshot_registry_path=policy_runtime.snapshot_registry_path,
        dev_eval_summaries_path=policy_runtime.dev_eval_summaries_path,
        runner=policy_runtime.runner,
        paired_seeds=seed_budget.paired_seeds,
        paired_seed_limit=seed_budget.paired_seed_limit,
        stage1_paired_seeds=seed_budget.stage1_paired_seeds,
        max_paired_seeds=seed_budget.max_paired_seeds,
        seed_file_path=seed_budget.seed_file_path,
        recommended_focal_policy_id=recommended_focal_policy_id,
    )


__all__ = ["resolve_canonical_eval_runtime_state"]
