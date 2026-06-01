"""Runtime orchestration for paired targeted-outcome comparison reports."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.experiments.main_league_multiobjective_gate import FIXED_THESIS_OPPONENTS
from weiss_rl.experiments.paired_outcome_compare import (
    PairedOutcomeCompareConfig,
    compare_paired_targeted_outcomes,
)


@dataclass(frozen=True, slots=True)
class PairedOutcomeCompareRunResult:
    output_json: Path
    report: dict[str, Any]


def run_paired_outcome_compare(args: argparse.Namespace) -> PairedOutcomeCompareRunResult:
    report = compare_paired_targeted_outcomes(paired_outcome_compare_config_from_args(args))
    write_paired_outcome_compare_report(args.output_json, report)
    return PairedOutcomeCompareRunResult(output_json=args.output_json, report=report)


def paired_outcome_compare_config_from_args(args: argparse.Namespace) -> PairedOutcomeCompareConfig:
    return PairedOutcomeCompareConfig(
        baseline_summary_json=args.baseline_summary_json.resolve(),
        candidate_summary_json=args.candidate_summary_json.resolve(),
        baseline_label=str(args.baseline_label),
        candidate_label=str(args.candidate_label),
        fixed_opponents=tuple(args.fixed_opponent or FIXED_THESIS_OPPONENTS),
        learned_opponents=tuple(str(item) for item in args.learned_opponent),
        max_examples=int(args.max_examples),
        pair_index_split=args.pair_index_split,
    )


def write_paired_outcome_compare_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "PairedOutcomeCompareRunResult",
    "paired_outcome_compare_config_from_args",
    "run_paired_outcome_compare",
    "write_paired_outcome_compare_report",
]
