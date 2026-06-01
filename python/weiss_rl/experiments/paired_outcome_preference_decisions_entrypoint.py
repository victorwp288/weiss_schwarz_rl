#!/usr/bin/env python3
"""Summarize paired-outcome preference decisions and same-state conflicts."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from weiss_rl.experiments import paired_outcome_preference_decisions as _decisions
from weiss_rl.experiments.paired_outcome_preference_decisions_cli import (
    build_paired_outcome_preference_decisions_parser,
    parse_paired_outcome_preference_decisions_args,
)
from weiss_rl.experiments.paired_outcome_preference_decisions_reporting import (
    paired_outcome_preference_decisions_output_line,
)
from weiss_rl.experiments.paired_outcome_preference_decisions_runtime import (
    run_paired_outcome_preference_decisions,
)

PairedOutcomePreferenceDecisionConfig = _decisions.PairedOutcomePreferenceDecisionConfig
build_paired_outcome_preference_decision_report = _decisions.build_paired_outcome_preference_decision_report
write_paired_outcome_preference_decision_report = _decisions.write_paired_outcome_preference_decision_report
_build_parser = build_paired_outcome_preference_decisions_parser


def main(argv: Sequence[str] | None = None) -> int:
    result = run_paired_outcome_preference_decisions(parse_args(argv))
    print(paired_outcome_preference_decisions_output_line(output_json=result.output_json, report=result.report))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return parse_paired_outcome_preference_decisions_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
