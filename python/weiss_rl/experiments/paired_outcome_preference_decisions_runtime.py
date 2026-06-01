"""Runtime orchestration for paired-outcome preference decision reports."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.experiments.paired_outcome_preference_decisions import (
    PairedOutcomePreferenceDecisionConfig,
    build_paired_outcome_preference_decision_report,
    write_paired_outcome_preference_decision_report,
)


@dataclass(frozen=True, slots=True)
class PairedOutcomePreferenceDecisionRunResult:
    output_json: Path
    report: dict[str, Any]


def run_paired_outcome_preference_decisions(
    args: argparse.Namespace,
) -> PairedOutcomePreferenceDecisionRunResult:
    report = build_paired_outcome_preference_decision_report(paired_outcome_preference_decision_config_from_args(args))
    write_paired_outcome_preference_decision_report(args.output_json, report)
    return PairedOutcomePreferenceDecisionRunResult(output_json=args.output_json, report=report)


def paired_outcome_preference_decision_config_from_args(
    args: argparse.Namespace,
) -> PairedOutcomePreferenceDecisionConfig:
    return PairedOutcomePreferenceDecisionConfig(
        dataset_path=args.dataset,
        spec_bundle_json=args.spec_bundle_json,
        max_examples=int(args.max_examples),
        top_action_edges=int(args.top_action_edges),
    )


__all__ = [
    "PairedOutcomePreferenceDecisionRunResult",
    "paired_outcome_preference_decision_config_from_args",
    "run_paired_outcome_preference_decisions",
]
