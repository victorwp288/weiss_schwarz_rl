#!/usr/bin/env python3
"""Gate paired-outcome compare screens before larger main-league evals."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from weiss_rl.experiments import paired_outcome_compare_gate as _compare_gate
from weiss_rl.experiments.paired_outcome_compare_gate_cli import (
    build_paired_outcome_compare_gate_parser,
    parse_paired_outcome_compare_gate_args,
)
from weiss_rl.experiments.paired_outcome_compare_gate_reporting import (
    paired_outcome_compare_gate_output_line,
)
from weiss_rl.experiments.paired_outcome_compare_gate_runtime import (
    run_paired_outcome_compare_gate,
)

PairedOutcomeCompareGateConfig = _compare_gate.PairedOutcomeCompareGateConfig
evaluate_paired_outcome_compare_gate = _compare_gate.evaluate_paired_outcome_compare_gate
write_paired_outcome_compare_gate = _compare_gate.write_paired_outcome_compare_gate
_build_parser = build_paired_outcome_compare_gate_parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return parse_paired_outcome_compare_gate_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result = run_paired_outcome_compare_gate(parse_args(argv))
    print(paired_outcome_compare_gate_output_line(output_json=result.output_json, report=result.report))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
