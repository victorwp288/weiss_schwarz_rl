"""Final-eval matchup job planning and execution."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from weiss_rl.config.models import StopRulesConfig
from weiss_rl.eval.analysis.payoff_folding import PayoffFoldScheme
from weiss_rl.eval.final.matchups import run_final_eval_matchup
from weiss_rl.eval.simulator.harness import EvalGameRunner


@dataclass(frozen=True, slots=True)
class FinalEvalMatchupJob:
    """One canonical upper-triangle policy matchup."""

    focal_index: int
    opponent_index: int
    focal_policy_id: str
    opponent_policy_id: str


class FinalEvalMatchupRunnerFn(Protocol):
    """Callable that runs one final-eval matchup job."""

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


def build_final_eval_matchup_jobs(policy_ids: Sequence[str]) -> tuple[FinalEvalMatchupJob, ...]:
    """Build the canonical upper-triangle matchup plan."""
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
    """Run every planned final-eval matchup and return canonical artifacts."""
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


__all__ = [
    "FinalEvalMatchupJob",
    "FinalEvalMatchupRunnerFn",
    "build_final_eval_matchup_jobs",
    "run_final_eval_matchup_jobs",
]
