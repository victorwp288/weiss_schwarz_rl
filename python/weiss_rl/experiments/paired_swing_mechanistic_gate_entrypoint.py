#!/usr/bin/env python3
"""Gate paired-swing pre/post margin diagnostics before game eval."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from weiss_rl.experiments import paired_swing_mechanistic_gate as _mechanistic_gate
from weiss_rl.experiments.paired_swing_mechanistic_gate_cli import (
    build_paired_swing_mechanistic_gate_parser,
    parse_paired_swing_mechanistic_gate_args,
)
from weiss_rl.experiments.paired_swing_mechanistic_gate_reporting import (
    paired_swing_mechanistic_gate_output_line,
)
from weiss_rl.experiments.paired_swing_mechanistic_gate_runtime import (
    run_paired_swing_mechanistic_gate,
)

PairedSwingMechanisticGateConfig = _mechanistic_gate.PairedSwingMechanisticGateConfig
evaluate_paired_swing_mechanistic_gate = _mechanistic_gate.evaluate_paired_swing_mechanistic_gate
write_paired_swing_mechanistic_gate = _mechanistic_gate.write_paired_swing_mechanistic_gate
_build_parser = build_paired_swing_mechanistic_gate_parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return parse_paired_swing_mechanistic_gate_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result = run_paired_swing_mechanistic_gate(parse_args(argv))
    print(paired_swing_mechanistic_gate_output_line(output_json=result.output_json, report=result.report))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
