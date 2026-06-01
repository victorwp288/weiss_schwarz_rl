#!/usr/bin/env python3
"""Gate paired-outcome preference margin diagnostics before game eval."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from weiss_rl.experiments import paired_outcome_preference_mechanistic_gate as _gate
from weiss_rl.experiments.paired_outcome_preference_mechanistic_gate_cli import (
    build_paired_outcome_preference_mechanistic_gate_parser,
    parse_paired_outcome_preference_mechanistic_gate_args,
)
from weiss_rl.experiments.paired_outcome_preference_mechanistic_gate_reporting import (
    paired_outcome_preference_mechanistic_gate_output_line,
)
from weiss_rl.experiments.paired_outcome_preference_mechanistic_gate_runtime import (
    run_paired_outcome_preference_mechanistic_gate,
)

PairedOutcomePreferenceMechanisticGateConfig = _gate.PairedOutcomePreferenceMechanisticGateConfig
evaluate_paired_outcome_preference_mechanistic_gate = _gate.evaluate_paired_outcome_preference_mechanistic_gate
write_paired_outcome_preference_mechanistic_gate = _gate.write_paired_outcome_preference_mechanistic_gate
_build_parser = build_paired_outcome_preference_mechanistic_gate_parser


def main(argv: Sequence[str] | None = None) -> int:
    result = run_paired_outcome_preference_mechanistic_gate(parse_args(argv))
    print(paired_outcome_preference_mechanistic_gate_output_line(output_json=result.output_json, report=result.report))
    return result.exit_code


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return parse_paired_outcome_preference_mechanistic_gate_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
