from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from weiss_rl.workflows.canonical_eval.entrypoint_request import (
    canonical_eval_entrypoint_request,
    run_canonical_entrypoint_request_pipeline,
)


def run_canonical_eval_entrypoint_pipeline(
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
    canonical_dependencies_fn: Any,
    run_canonical_eval_pipeline_fn: Any,
) -> int:
    request = canonical_eval_entrypoint_request(
        parser=parser,
        stack=stack,
        run_dir=run_dir,
        final_eval_dir=final_eval_dir,
        policy_ids=policy_ids,
        snapshot_registry_path=snapshot_registry_path,
        dev_eval_summaries_path=dev_eval_summaries_path,
        b1_baseline_run_dir=b1_baseline_run_dir,
        bootstrap_samples=bootstrap_samples,
        paired_seed_limit=paired_seed_limit,
        stage1_paired_seeds=stage1_paired_seeds,
        max_paired_seeds=max_paired_seeds,
        skip_metagame=skip_metagame,
        study_config_path=study_config_path,
        skip_figures=skip_figures,
        skip_readiness=skip_readiness,
        git_commit_override=git_commit_override,
    )
    return run_canonical_entrypoint_request_pipeline(
        request=request,
        dependencies=canonical_dependencies_fn(),
        run_canonical_eval_pipeline_fn=run_canonical_eval_pipeline_fn,
    )


__all__ = ["run_canonical_eval_entrypoint_pipeline"]
