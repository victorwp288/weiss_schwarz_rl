"""Runtime orchestration for paired-outcome preference dataset generation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.experiments.paired_outcome_preference_dataset import (
    PairedOutcomePreferenceDatasetConfig,
    build_paired_outcome_preference_dataset,
)
from weiss_rl.experiments.paired_outcome_preference_dataset_cli import parse_opponent_match_aliases
from weiss_rl.replay.trajectory_bc import ReplayTrajectoryDataset


@dataclass(frozen=True, slots=True)
class PairedOutcomePreferenceDatasetRunResult:
    output_dataset: Path
    dataset: ReplayTrajectoryDataset
    summary: dict[str, Any]


def run_paired_outcome_preference_dataset(args: argparse.Namespace) -> PairedOutcomePreferenceDatasetRunResult:
    dataset, summary = build_paired_outcome_preference_dataset(paired_outcome_preference_dataset_config_from_args(args))
    return PairedOutcomePreferenceDatasetRunResult(
        output_dataset=args.output,
        dataset=dataset,
        summary=summary,
    )


def paired_outcome_preference_dataset_config_from_args(
    args: argparse.Namespace,
) -> PairedOutcomePreferenceDatasetConfig:
    return PairedOutcomePreferenceDatasetConfig(
        preferred_dataset=args.preferred_dataset.resolve(),
        rejected_dataset=args.rejected_dataset.resolve(),
        output_dataset=args.output,
        output_summary_json=args.summary_json,
        max_pairs=args.max_pairs,
        preferred_label=str(args.preferred_label),
        rejected_label=str(args.rejected_label),
        opponent_match_aliases=parse_opponent_match_aliases(args.opponent_match_alias),
    )


__all__ = [
    "PairedOutcomePreferenceDatasetRunResult",
    "paired_outcome_preference_dataset_config_from_args",
    "run_paired_outcome_preference_dataset",
]
