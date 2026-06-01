"""Runtime orchestration for paired-outcome compare gates."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.experiments.paired_outcome_compare_gate import (
    PairedOutcomeCompareGateConfig,
    evaluate_paired_outcome_compare_gate,
    write_paired_outcome_compare_gate,
)


@dataclass(frozen=True, slots=True)
class PairedOutcomeCompareGateRunResult:
    output_json: Path
    report: dict[str, Any]

    @property
    def exit_code(self) -> int:
        return 0 if bool(self.report["passed"]) else 2


def run_paired_outcome_compare_gate(args: argparse.Namespace) -> PairedOutcomeCompareGateRunResult:
    report = evaluate_paired_outcome_compare_gate(paired_outcome_compare_gate_config_from_args(args))
    write_paired_outcome_compare_gate(args.output_json, report)
    return PairedOutcomeCompareGateRunResult(output_json=args.output_json, report=report)


def paired_outcome_compare_gate_config_from_args(args: argparse.Namespace) -> PairedOutcomeCompareGateConfig:
    return PairedOutcomeCompareGateConfig(
        compare_jsons=tuple(Path(path) for path in args.compare_json),
        min_all_delta_wins=int(args.min_all_delta_wins),
        min_fixed_delta_wins=int(args.min_fixed_delta_wins),
        min_learned_delta_wins=int(args.min_learned_delta_wins),
        max_fixed_row_drop_wins=int(args.max_fixed_row_drop_wins),
        max_learned_row_drop_wins=int(args.max_learned_row_drop_wins),
        required_opponents=tuple(str(item) for item in args.required_opponent),
    )


__all__ = [
    "PairedOutcomeCompareGateRunResult",
    "paired_outcome_compare_gate_config_from_args",
    "run_paired_outcome_compare_gate",
]
