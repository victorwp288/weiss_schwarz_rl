"""Runtime orchestration for paired-swing repair-target reports."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.experiments.main_league_multiobjective_gate import FIXED_THESIS_OPPONENTS
from weiss_rl.experiments.paired_swing_report import (
    PairedSwingReportConfig,
    build_paired_swing_report,
    write_paired_swing_report,
)


@dataclass(frozen=True, slots=True)
class PairedSwingReportRunResult:
    output_json: Path
    report: dict[str, Any]


def run_paired_swing_report(args: argparse.Namespace) -> PairedSwingReportRunResult:
    report = build_paired_swing_report(paired_swing_report_config_from_args(args))
    write_paired_swing_report(report, args.output_json)
    return PairedSwingReportRunResult(output_json=args.output_json, report=report)


def paired_swing_report_config_from_args(args: argparse.Namespace) -> PairedSwingReportConfig:
    return PairedSwingReportConfig(
        compare_jsons=tuple(path.resolve() for path in args.compare_json),
        opponent_pool_jsonls=tuple(path.resolve() for path in args.opponent_pool_jsonl),
        fixed_opponents=tuple(args.fixed_opponent or FIXED_THESIS_OPPONENTS),
        learned_opponents=tuple(str(item) for item in args.learned_opponent),
        max_examples_per_bucket=int(args.max_examples_per_bucket),
        notes=str(args.notes),
    )


__all__ = [
    "PairedSwingReportRunResult",
    "paired_swing_report_config_from_args",
    "run_paired_swing_report",
]
