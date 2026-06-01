"""Runtime orchestration for paired-swing mechanistic gates."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.experiments.paired_swing_mechanistic_gate import (
    PairedSwingMechanisticGateConfig,
    evaluate_paired_swing_mechanistic_gate,
    write_paired_swing_mechanistic_gate,
)


@dataclass(frozen=True, slots=True)
class PairedSwingMechanisticGateRunResult:
    output_json: Path
    report: dict[str, Any]

    @property
    def exit_code(self) -> int:
        return 0 if bool(self.report["passed"]) else 2


def run_paired_swing_mechanistic_gate(args: argparse.Namespace) -> PairedSwingMechanisticGateRunResult:
    report = evaluate_paired_swing_mechanistic_gate(paired_swing_mechanistic_gate_config_from_args(args))
    write_paired_swing_mechanistic_gate(args.output_json, report)
    return PairedSwingMechanisticGateRunResult(output_json=args.output_json, report=report)


def paired_swing_mechanistic_gate_config_from_args(args: argparse.Namespace) -> PairedSwingMechanisticGateConfig:
    return PairedSwingMechanisticGateConfig(
        pre_report_json=args.pre_report_json,
        post_report_json=args.post_report_json,
        min_mean_delta=float(args.min_mean_delta),
        min_min_delta=float(args.min_min_delta),
        min_row_improved_fraction=float(args.min_row_improved_fraction),
        max_row_worsened_fraction=float(args.max_row_worsened_fraction),
        min_top_positive_delta=int(args.min_top_positive_delta),
        max_positive_rank_worsened_fraction=float(args.max_positive_rank_worsened_fraction),
        min_protected_label_mean_delta=float(args.min_protected_label_mean_delta),
        protected_label_contains=tuple(str(item) for item in args.protected_label_contains),
        require_context=not bool(args.allow_missing_context),
    )


__all__ = [
    "PairedSwingMechanisticGateRunResult",
    "paired_swing_mechanistic_gate_config_from_args",
    "run_paired_swing_mechanistic_gate",
]
