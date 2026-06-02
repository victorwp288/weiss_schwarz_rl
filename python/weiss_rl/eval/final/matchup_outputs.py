"""Payload and artifact writers for single final-eval matchups."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.config.models import StopRulesConfig
from weiss_rl.eval.diagnostics import build_seat_advantage_diagnostics, write_matchup_diagnostics_json
from weiss_rl.eval.export import build_matchup_export, write_matchup_summary_csv, write_matchup_summary_json
from weiss_rl.eval.harness import EvalGameRecord, EvalGameRunner
from weiss_rl.eval.payoff_folding import PayoffFoldScheme, paired_seed_scores
from weiss_rl.eval.uncertainty import bayesian_bootstrap_posterior_samples


@dataclass(frozen=True, slots=True)
class FinalEvalMatchupArtifactPaths:
    summary_json: Path
    summary_csv: Path
    diagnostics_json: Path
    posterior_samples_json: Path


def matchup_artifact_paths(matchup_dir: Path) -> FinalEvalMatchupArtifactPaths:
    return FinalEvalMatchupArtifactPaths(
        summary_json=matchup_dir / "matchup_summary.json",
        summary_csv=matchup_dir / "matchup_summary.csv",
        diagnostics_json=matchup_dir / "diagnostics.json",
        posterior_samples_json=matchup_dir / "posterior_samples.json",
    )


def build_matchup_payload(
    *,
    records: Sequence[EvalGameRecord],
    stop_rules: StopRulesConfig,
    max_paired_seeds: int,
    scheme: PayoffFoldScheme,
    sample_count: int,
    seed: int,
) -> dict[str, Any]:
    return build_matchup_export(
        tuple(records),
        stop_rules=stop_rules,
        max_paired_seeds=max_paired_seeds,
        scheme=scheme,
        sample_count=sample_count,
        seed=seed,
    )


def matchup_posterior_samples(
    *,
    records: Sequence[EvalGameRecord],
    scheme: PayoffFoldScheme,
    sample_count: int,
    seed: int,
) -> tuple[float, ...]:
    scores = paired_seed_scores(records, scheme=scheme)
    if not scores:
        return ()
    return bayesian_bootstrap_posterior_samples(scores, sample_count=sample_count, seed=seed)


def build_matchup_diagnostics(
    *,
    records: Sequence[EvalGameRecord],
    runner: EvalGameRunner,
) -> dict[str, Any]:
    diagnostics_payload = build_seat_advantage_diagnostics(tuple(records))
    god_search_diagnostics_fn = getattr(runner, "god_search_diagnostics", None)
    if callable(god_search_diagnostics_fn):
        god_search_diagnostics = god_search_diagnostics_fn()
        if god_search_diagnostics is not None:
            diagnostics_payload["god_search"] = god_search_diagnostics
    return diagnostics_payload


def build_posterior_samples_payload(
    *,
    focal_policy_id: str,
    opponent_policy_id: str,
    sample_count: int,
    posterior_samples: Sequence[float],
    summary_payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "focal_policy_id": focal_policy_id,
        "opponent_policy_id": opponent_policy_id,
        "requested_sample_count": sample_count,
        "sample_count": len(posterior_samples),
        "has_payoff_samples": summary_payload["has_payoff_samples"],
        "samples": list(posterior_samples),
    }


def write_matchup_artifacts(
    *,
    matchup_dir: Path,
    focal_policy_id: str,
    opponent_policy_id: str,
    sample_count: int,
    summary_payload: dict[str, Any],
    diagnostics_payload: dict[str, Any],
    posterior_samples: Sequence[float],
) -> FinalEvalMatchupArtifactPaths:
    paths = matchup_artifact_paths(matchup_dir)
    write_matchup_summary_json(paths.summary_json, summary_payload)
    write_matchup_summary_csv(paths.summary_csv, summary_payload)
    write_matchup_diagnostics_json(paths.diagnostics_json, diagnostics_payload)
    paths.posterior_samples_json.write_text(
        json.dumps(
            build_posterior_samples_payload(
                focal_policy_id=focal_policy_id,
                opponent_policy_id=opponent_policy_id,
                sample_count=sample_count,
                posterior_samples=posterior_samples,
                summary_payload=summary_payload,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return paths


__all__ = [
    "FinalEvalMatchupArtifactPaths",
    "build_matchup_diagnostics",
    "build_matchup_payload",
    "build_posterior_samples_payload",
    "matchup_artifact_paths",
    "matchup_posterior_samples",
    "write_matchup_artifacts",
]
