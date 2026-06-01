"""Runtime orchestration for paired-outcome preference surface-cluster diagnostics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.experiments.paired_outcome_preference_surface_clusters import (
    PairedOutcomePreferenceSurfaceClusterConfig,
    build_paired_outcome_preference_surface_cluster_report,
    write_paired_outcome_preference_surface_cluster_report,
)


@dataclass(frozen=True, slots=True)
class PairedOutcomePreferenceSurfaceClusterRunResult:
    output_json: Path
    report: dict[str, Any]


def run_paired_outcome_preference_surface_cluster(
    args: argparse.Namespace,
) -> PairedOutcomePreferenceSurfaceClusterRunResult:
    report = build_paired_outcome_preference_surface_cluster_report(
        paired_outcome_preference_surface_cluster_config_from_args(args)
    )
    write_paired_outcome_preference_surface_cluster_report(args.output_json, report)
    return PairedOutcomePreferenceSurfaceClusterRunResult(output_json=args.output_json, report=report)


def paired_outcome_preference_surface_cluster_config_from_args(
    args: argparse.Namespace,
) -> PairedOutcomePreferenceSurfaceClusterConfig:
    return PairedOutcomePreferenceSurfaceClusterConfig(
        dataset_path=args.dataset,
        spec_bundle_json=args.spec_bundle_json,
        stack_config_path=args.stack_config,
        opponent_context_policy_ids=tuple(str(item) for item in args.opponent_context_policy_id),
        max_examples=int(args.max_examples),
    )


__all__ = [
    "PairedOutcomePreferenceSurfaceClusterRunResult",
    "paired_outcome_preference_surface_cluster_config_from_args",
    "run_paired_outcome_preference_surface_cluster",
]
