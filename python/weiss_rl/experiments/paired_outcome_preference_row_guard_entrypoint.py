#!/usr/bin/env python3
"""Gate row-level target-action drift on paired outcome preference replay."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from weiss_rl.experiments import paired_outcome_preference_row_guard as _row_guard
from weiss_rl.experiments.paired_outcome_preference_row_guard_cli import (
    build_paired_outcome_preference_row_guard_parser,
    parse_paired_outcome_preference_row_guard_args,
)
from weiss_rl.experiments.paired_outcome_preference_row_guard_reporting import (
    paired_outcome_preference_row_guard_output_line,
)
from weiss_rl.experiments.paired_outcome_preference_row_guard_runtime import (
    run_paired_outcome_preference_row_guard,
)

PairedOutcomePreferenceRowGuardConfig = _row_guard.PairedOutcomePreferenceRowGuardConfig
build_paired_outcome_preference_row_guard = _row_guard.build_paired_outcome_preference_row_guard
write_paired_outcome_preference_row_guard = _row_guard.write_paired_outcome_preference_row_guard
_build_parser = build_paired_outcome_preference_row_guard_parser


def main(argv: Sequence[str] | None = None) -> int:
    result = run_paired_outcome_preference_row_guard(parse_args(argv))
    print(paired_outcome_preference_row_guard_output_line(output_json=result.output_json, report=result.report))
    return result.exit_code


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return parse_paired_outcome_preference_row_guard_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
