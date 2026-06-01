"""Runtime orchestration for paired-outcome preference mechanistic gates."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.experiments.paired_outcome_preference_mechanistic_gate import (
    PairedOutcomePreferenceMechanisticGateConfig,
    evaluate_paired_outcome_preference_mechanistic_gate,
    write_paired_outcome_preference_mechanistic_gate,
)


@dataclass(frozen=True, slots=True)
class PairedOutcomePreferenceMechanisticGateRunResult:
    output_json: Path
    report: dict[str, Any]

    @property
    def exit_code(self) -> int:
        return 0 if bool(self.report["passed"]) else 2


def run_paired_outcome_preference_mechanistic_gate(
    args: argparse.Namespace,
) -> PairedOutcomePreferenceMechanisticGateRunResult:
    report = evaluate_paired_outcome_preference_mechanistic_gate(
        paired_outcome_preference_mechanistic_gate_config_from_args(args)
    )
    write_paired_outcome_preference_mechanistic_gate(args.output_json, report)
    return PairedOutcomePreferenceMechanisticGateRunResult(output_json=args.output_json, report=report)


def paired_outcome_preference_mechanistic_gate_config_from_args(
    args: argparse.Namespace,
) -> PairedOutcomePreferenceMechanisticGateConfig:
    return PairedOutcomePreferenceMechanisticGateConfig(
        pre_report_json=args.pre_report_json,
        post_report_json=args.post_report_json,
        min_mean_delta=float(args.min_mean_delta),
        min_min_delta=float(args.min_min_delta),
        min_pair_improved_fraction=float(args.min_pair_improved_fraction),
        max_pair_worsened_fraction=float(args.max_pair_worsened_fraction),
        min_group_mean_delta=float(args.min_group_mean_delta),
        min_required_group_mean_delta=float(args.min_required_group_mean_delta),
        required_groups=tuple(str(item) for item in args.required_group),
        require_context=not bool(args.allow_missing_context),
    )


__all__ = [
    "PairedOutcomePreferenceMechanisticGateRunResult",
    "paired_outcome_preference_mechanistic_gate_config_from_args",
    "run_paired_outcome_preference_mechanistic_gate",
]
