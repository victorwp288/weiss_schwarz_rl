"""Runtime orchestration for paired-outcome preference margin reports."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.experiments.paired_outcome_preference_margins import (
    PairedOutcomePreferenceMarginConfig,
    build_paired_outcome_preference_margin_report,
    write_paired_outcome_preference_margin_report,
)


@dataclass(frozen=True, slots=True)
class PairedOutcomePreferenceMarginRunResult:
    output_json: Path
    report: dict[str, Any]


def run_paired_outcome_preference_margins(args: argparse.Namespace) -> PairedOutcomePreferenceMarginRunResult:
    report = build_paired_outcome_preference_margin_report(paired_outcome_preference_margin_config_from_args(args))
    write_paired_outcome_preference_margin_report(args.output_json, report)
    return PairedOutcomePreferenceMarginRunResult(output_json=args.output_json, report=report)


def paired_outcome_preference_margin_config_from_args(
    args: argparse.Namespace,
) -> PairedOutcomePreferenceMarginConfig:
    return PairedOutcomePreferenceMarginConfig(
        dataset_path=args.dataset,
        stack_config_path=args.stack_config,
        run_dir=args.run_dir,
        checkpoint_path=args.checkpoint,
        reference_checkpoint_path=args.reference_checkpoint,
        aggregation=str(args.aggregation),
    )


__all__ = [
    "PairedOutcomePreferenceMarginRunResult",
    "paired_outcome_preference_margin_config_from_args",
    "run_paired_outcome_preference_margins",
]
