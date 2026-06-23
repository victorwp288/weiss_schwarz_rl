from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from weiss_rl.workflows.canonical_eval.entrypoint_request import (
    canonical_eval_entrypoint_request,
    run_canonical_entrypoint_request_adapter,
)
from weiss_rl.workflows.eval_entrypoint_support.main import run_eval_entrypoint_main
from weiss_rl.workflows.eval_support.eval_dependencies import build_canonical_eval_dependencies
from weiss_rl.workflows.eval_support.eval_dispatch_dependencies import build_eval_dispatch_dependencies
from weiss_rl.workflows.eval_support.eval_startup_dependencies import build_eval_startup_dependencies


def build_eval_entrypoint_canonical_dependencies(entrypoint_globals: Mapping[str, Any]) -> Any:
    return build_canonical_eval_dependencies(entrypoint_globals)


def build_eval_entrypoint_dispatch_dependencies(entrypoint_globals: Mapping[str, Any]) -> Any:
    return build_eval_dispatch_dependencies(entrypoint_globals)


def build_eval_entrypoint_startup_dependencies(entrypoint_globals: Mapping[str, Any]) -> Any:
    return build_eval_startup_dependencies(entrypoint_globals)


def run_eval_entrypoint_canonical_pipeline(
    *,
    entrypoint_globals: Mapping[str, Any],
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
    return run_canonical_entrypoint_request_adapter(
        request=request,
        canonical_dependencies_fn=lambda: build_eval_entrypoint_canonical_dependencies(entrypoint_globals),
        run_canonical_eval_pipeline_fn=entrypoint_globals["run_canonical_eval_pipeline"],
        run_canonical_eval_entrypoint_pipeline_fn=entrypoint_globals["run_canonical_eval_entrypoint_pipeline"],
    )


def run_eval_entrypoint(*, entrypoint_globals: Mapping[str, Any]) -> None:
    run_eval_entrypoint_main(
        build_eval_parser_fn=entrypoint_globals["build_eval_parser"],
        validate_eval_args_fn=entrypoint_globals["validate_eval_args"],
        prepare_eval_startup_fn=entrypoint_globals["prepare_eval_startup"],
        run_eval_dispatch_fn=entrypoint_globals["run_eval_dispatch"],
        eval_startup_dependencies_fn=lambda: build_eval_entrypoint_startup_dependencies(entrypoint_globals),
        eval_dispatch_dependencies_fn=lambda: build_eval_entrypoint_dispatch_dependencies(entrypoint_globals),
    )


__all__ = [
    "build_eval_entrypoint_canonical_dependencies",
    "build_eval_entrypoint_dispatch_dependencies",
    "build_eval_entrypoint_startup_dependencies",
    "run_eval_entrypoint",
    "run_eval_entrypoint_canonical_pipeline",
]
