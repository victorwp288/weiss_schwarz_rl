"""Run-level orchestration phases for canonical final evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from weiss_rl.config.models import FinalPolicySetSelectionConfig, StopRulesConfig
from weiss_rl.eval.final.artifacts import write_final_eval_artifacts
from weiss_rl.eval.final.matchups import run_final_eval_matchup
from weiss_rl.eval.final.payload import build_final_eval_payload
from weiss_rl.eval.final.policy_selection import resolve_final_eval_policy_ids, validate_final_eval_seed_budget
from weiss_rl.eval.harness import EvalGameRunner
from weiss_rl.eval.payoff_folding import PayoffFoldScheme


@dataclass(frozen=True, slots=True)
class FinalEvalMatchupJob:
    focal_index: int
    opponent_index: int
    focal_policy_id: str
    opponent_policy_id: str


class FinalEvalMatchupRunnerFn(Protocol):
    def __call__(
        self,
        *,
        output_dir: Path,
        focal_index: int,
        opponent_index: int,
        focal_policy_id: str,
        opponent_policy_id: str,
        paired_seeds: Sequence[int],
        stage1_paired_seeds: int,
        max_paired_seeds: int,
        stop_rules: StopRulesConfig,
        runner: EvalGameRunner,
        run_id256: str | bytes,
        config_hash256: str,
        spec_hash256: str,
        scheme: PayoffFoldScheme,
        sample_count: int,
    ) -> dict[str, Any]: ...


def resolve_final_eval_run_policy_ids(
    *,
    policy_ids: Sequence[str] | None,
    snapshot_registry_path: Path | None,
    dev_eval_summaries_path: Path | None,
    selection_config: FinalPolicySetSelectionConfig | None,
    final_policy_set_size: int | None,
) -> tuple[list[str], dict[str, Any]]:
    return resolve_final_eval_policy_ids(
        policy_ids=policy_ids,
        snapshot_registry_path=snapshot_registry_path,
        dev_eval_summaries_path=dev_eval_summaries_path,
        selection_config=selection_config,
        final_policy_set_size=final_policy_set_size,
    )


def validate_final_eval_run_seed_budget(
    *,
    paired_seeds: Sequence[int],
    stage1_paired_seeds: int,
    max_paired_seeds: int,
) -> None:
    validate_final_eval_seed_budget(
        paired_seeds=paired_seeds,
        stage1_paired_seeds=stage1_paired_seeds,
        max_paired_seeds=max_paired_seeds,
    )


def build_final_eval_matchup_jobs(policy_ids: Sequence[str]) -> tuple[FinalEvalMatchupJob, ...]:
    jobs: list[FinalEvalMatchupJob] = []
    for focal_index, focal_policy_id in enumerate(policy_ids):
        for opponent_index, opponent_policy_id in enumerate(policy_ids[focal_index:], start=focal_index):
            jobs.append(
                FinalEvalMatchupJob(
                    focal_index=focal_index,
                    opponent_index=opponent_index,
                    focal_policy_id=focal_policy_id,
                    opponent_policy_id=opponent_policy_id,
                )
            )
    return tuple(jobs)


def run_final_eval_matchup_jobs(
    *,
    output_dir: Path,
    jobs: Sequence[FinalEvalMatchupJob],
    runner: EvalGameRunner,
    paired_seeds: Sequence[int],
    stage1_paired_seeds: int,
    max_paired_seeds: int,
    stop_rules: StopRulesConfig,
    run_id256: str | bytes,
    config_hash256: str,
    spec_hash256: str,
    scheme: PayoffFoldScheme,
    sample_count: int,
    run_matchup_fn: FinalEvalMatchupRunnerFn = run_final_eval_matchup,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        run_matchup_fn(
            output_dir=output_dir,
            focal_index=job.focal_index,
            opponent_index=job.opponent_index,
            focal_policy_id=job.focal_policy_id,
            opponent_policy_id=job.opponent_policy_id,
            paired_seeds=paired_seeds,
            stage1_paired_seeds=stage1_paired_seeds,
            max_paired_seeds=max_paired_seeds,
            stop_rules=stop_rules,
            runner=runner,
            run_id256=run_id256,
            config_hash256=config_hash256,
            spec_hash256=spec_hash256,
            scheme=scheme,
            sample_count=sample_count,
        )
        for job in jobs
    ]


def build_final_eval_run_payload(
    *,
    output_dir: Path,
    policy_ids: Sequence[str],
    matchup_results: Sequence[dict[str, Any]],
    paired_seeds: Sequence[int],
    stage1_paired_seeds: int,
    max_paired_seeds: int,
    stop_rules: StopRulesConfig,
    scheme: PayoffFoldScheme,
    sample_count: int,
    selection_payload: Mapping[str, Any],
    metadata: Mapping[str, Any] | None,
    seed_file_path: Path | None,
) -> dict[str, Any]:
    return build_final_eval_payload(
        output_dir=output_dir,
        policy_ids=policy_ids,
        matchup_results=matchup_results,
        stage1_paired_seeds=stage1_paired_seeds,
        max_paired_seeds=max_paired_seeds,
        paired_seeds=paired_seeds,
        stop_rules=stop_rules,
        scheme=scheme,
        sample_count=sample_count,
        selection_payload=selection_payload,
        metadata=metadata,
        seed_file_path=seed_file_path,
    )


def write_final_eval_run_artifacts(
    *,
    output_dir: Path,
    payload: dict[str, Any],
    matchup_results: Sequence[dict[str, Any]],
) -> None:
    write_final_eval_artifacts(output_dir=output_dir, payload=payload, matchup_results=matchup_results)


def run_final_eval(
    *,
    output_dir: Path,
    runner: EvalGameRunner,
    paired_seeds: Sequence[int],
    stage1_paired_seeds: int,
    max_paired_seeds: int,
    stop_rules: StopRulesConfig,
    run_id256: str | bytes,
    config_hash256: str,
    spec_hash256: str,
    scheme: PayoffFoldScheme = "S0",
    sample_count: int = 1000,
    policy_ids: Sequence[str] | None = None,
    snapshot_registry_path: Path | None = None,
    dev_eval_summaries_path: Path | None = None,
    selection_config: FinalPolicySetSelectionConfig | None = None,
    final_policy_set_size: int | None = None,
    metadata: Mapping[str, Any] | None = None,
    seed_file_path: Path | None = None,
) -> dict[str, Any]:
    resolved_policy_ids, selection_payload = resolve_final_eval_run_policy_ids(
        policy_ids=policy_ids,
        snapshot_registry_path=snapshot_registry_path,
        dev_eval_summaries_path=dev_eval_summaries_path,
        selection_config=selection_config,
        final_policy_set_size=final_policy_set_size,
    )
    validate_final_eval_run_seed_budget(
        paired_seeds=paired_seeds,
        stage1_paired_seeds=stage1_paired_seeds,
        max_paired_seeds=max_paired_seeds,
    )
    matchup_results = run_final_eval_matchup_jobs(
        output_dir=output_dir,
        jobs=build_final_eval_matchup_jobs(resolved_policy_ids),
        runner=runner,
        paired_seeds=paired_seeds,
        stage1_paired_seeds=stage1_paired_seeds,
        max_paired_seeds=max_paired_seeds,
        stop_rules=stop_rules,
        run_id256=run_id256,
        config_hash256=config_hash256,
        spec_hash256=spec_hash256,
        scheme=scheme,
        sample_count=sample_count,
    )
    payload = build_final_eval_run_payload(
        output_dir=output_dir,
        policy_ids=resolved_policy_ids,
        matchup_results=matchup_results,
        stage1_paired_seeds=stage1_paired_seeds,
        max_paired_seeds=max_paired_seeds,
        paired_seeds=paired_seeds,
        stop_rules=stop_rules,
        scheme=scheme,
        sample_count=sample_count,
        selection_payload=selection_payload,
        metadata=metadata,
        seed_file_path=seed_file_path,
    )
    write_final_eval_run_artifacts(output_dir=output_dir, payload=payload, matchup_results=matchup_results)
    return payload


__all__ = [
    "FinalEvalMatchupJob",
    "FinalEvalMatchupRunnerFn",
    "build_final_eval_matchup_jobs",
    "build_final_eval_run_payload",
    "resolve_final_eval_run_policy_ids",
    "run_final_eval",
    "run_final_eval_matchup_jobs",
    "validate_final_eval_run_seed_budget",
    "write_final_eval_run_artifacts",
]
