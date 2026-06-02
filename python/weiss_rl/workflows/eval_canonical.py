from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from weiss_rl.workflows.canonical_eval.dependencies import CanonicalEvalDependencies
from weiss_rl.workflows.canonical_eval.outputs import write_canonical_eval_outputs
from weiss_rl.workflows.canonical_eval.runtime import resolve_canonical_eval_runtime_state
from weiss_rl.workflows.canonical_eval.setup import prepare_canonical_eval_run_state
from weiss_rl.workflows.canonical_eval.state import CanonicalEvalRunState, CanonicalEvalRuntimeState

__all__ = [
    "CanonicalEvalDependencies",
    "CanonicalEvalRunState",
    "CanonicalEvalRuntimeState",
    "prepare_canonical_eval_run_state",
    "resolve_canonical_eval_runtime_state",
    "run_canonical_eval_pipeline",
    "write_canonical_eval_outputs",
]


def run_canonical_eval_pipeline(
    *,
    parser: argparse.ArgumentParser,
    stack: Any,
    run_dir: Path,
    final_eval_dir: Path | None,
    policy_ids: list[str],
    snapshot_registry_path: Path | None,
    dev_eval_summaries_path: Path | None,
    b1_baseline_run_dir: Path | None,
    bootstrap_samples: int,
    paired_seed_limit: int | None,
    stage1_paired_seeds: int | None,
    max_paired_seeds: int | None,
    skip_metagame: bool,
    study_config_path: Path | None,
    skip_figures: bool,
    skip_readiness: bool,
    git_commit_override: str,
    dependencies: CanonicalEvalDependencies | None = None,
) -> int:
    if dependencies is None:
        dependencies = CanonicalEvalDependencies()

    run_state = prepare_canonical_eval_run_state(
        parser=parser,
        stack=stack,
        run_dir=run_dir,
        final_eval_dir=final_eval_dir,
        skip_metagame=skip_metagame,
        study_config_path=study_config_path,
        git_commit_override=git_commit_override,
        dependencies=dependencies,
    )
    runtime_state = resolve_canonical_eval_runtime_state(
        stack=stack,
        run_dir=run_dir,
        policy_ids=policy_ids,
        snapshot_registry_path=snapshot_registry_path,
        dev_eval_summaries_path=dev_eval_summaries_path,
        b1_baseline_run_dir=b1_baseline_run_dir,
        paired_seed_limit=paired_seed_limit,
        stage1_paired_seeds=stage1_paired_seeds,
        max_paired_seeds=max_paired_seeds,
        run_state=run_state,
        dependencies=dependencies,
    )

    try:
        return write_canonical_eval_outputs(
            run_dir=run_dir,
            bootstrap_samples=bootstrap_samples,
            skip_metagame=skip_metagame,
            skip_figures=skip_figures,
            skip_readiness=skip_readiness,
            run_state=run_state,
            runtime_state=runtime_state,
            dependencies=dependencies,
        )
    finally:
        run_state.tensorboard_logger.close()
