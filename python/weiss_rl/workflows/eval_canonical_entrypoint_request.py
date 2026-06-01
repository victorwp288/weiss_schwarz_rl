from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class CanonicalEvalEntrypointRequest:
    parser: argparse.ArgumentParser
    stack: Any
    run_dir: Path
    final_eval_dir: Path | None
    policy_ids: list[str]
    snapshot_registry_path: Path | None
    dev_eval_summaries_path: Path | None
    b1_baseline_run_dir: Path | None
    bootstrap_samples: int
    paired_seed_limit: int | None
    stage1_paired_seeds: int | None
    max_paired_seeds: int | None
    skip_metagame: bool
    study_config_path: Path | None
    skip_figures: bool
    skip_readiness: bool
    git_commit_override: str

    def entrypoint_kwargs(self) -> dict[str, Any]:
        return {
            "parser": self.parser,
            "stack": self.stack,
            "run_dir": self.run_dir,
            "final_eval_dir": self.final_eval_dir,
            "policy_ids": self.policy_ids,
            "snapshot_registry_path": self.snapshot_registry_path,
            "dev_eval_summaries_path": self.dev_eval_summaries_path,
            "b1_baseline_run_dir": self.b1_baseline_run_dir,
            "bootstrap_samples": self.bootstrap_samples,
            "paired_seed_limit": self.paired_seed_limit,
            "stage1_paired_seeds": self.stage1_paired_seeds,
            "max_paired_seeds": self.max_paired_seeds,
            "skip_metagame": self.skip_metagame,
            "study_config_path": self.study_config_path,
            "skip_figures": self.skip_figures,
            "skip_readiness": self.skip_readiness,
            "git_commit_override": self.git_commit_override,
        }

    def pipeline_kwargs(self, *, dependencies: Any) -> dict[str, Any]:
        return {**self.entrypoint_kwargs(), "dependencies": dependencies}


def canonical_eval_entrypoint_request(
    *,
    parser: argparse.ArgumentParser,
    stack: Any,
    run_dir: Path,
    final_eval_dir: Path | None,
    policy_ids: Sequence[str],
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
) -> CanonicalEvalEntrypointRequest:
    return CanonicalEvalEntrypointRequest(
        parser=parser,
        stack=stack,
        run_dir=run_dir,
        final_eval_dir=final_eval_dir,
        policy_ids=list(policy_ids),
        snapshot_registry_path=snapshot_registry_path,
        dev_eval_summaries_path=dev_eval_summaries_path,
        b1_baseline_run_dir=b1_baseline_run_dir,
        bootstrap_samples=int(bootstrap_samples),
        paired_seed_limit=paired_seed_limit,
        stage1_paired_seeds=stage1_paired_seeds,
        max_paired_seeds=max_paired_seeds,
        skip_metagame=bool(skip_metagame),
        study_config_path=study_config_path,
        skip_figures=bool(skip_figures),
        skip_readiness=bool(skip_readiness),
        git_commit_override=str(git_commit_override),
    )


def run_canonical_entrypoint_request_pipeline(
    *,
    request: CanonicalEvalEntrypointRequest,
    dependencies: Any,
    run_canonical_eval_pipeline_fn: Any,
) -> int:
    return int(run_canonical_eval_pipeline_fn(**request.pipeline_kwargs(dependencies=dependencies)))


def run_canonical_entrypoint_request_adapter(
    *,
    request: CanonicalEvalEntrypointRequest,
    canonical_dependencies_fn: Any,
    run_canonical_eval_pipeline_fn: Any,
    run_canonical_eval_entrypoint_pipeline_fn: Any,
) -> int:
    return int(
        run_canonical_eval_entrypoint_pipeline_fn(
            **request.entrypoint_kwargs(),
            canonical_dependencies_fn=canonical_dependencies_fn,
            run_canonical_eval_pipeline_fn=run_canonical_eval_pipeline_fn,
        )
    )


__all__ = [
    "CanonicalEvalEntrypointRequest",
    "canonical_eval_entrypoint_request",
    "run_canonical_entrypoint_request_adapter",
    "run_canonical_entrypoint_request_pipeline",
]
