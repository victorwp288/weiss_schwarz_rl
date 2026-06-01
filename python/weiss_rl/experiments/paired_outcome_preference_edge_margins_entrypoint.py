"""Gate row-level paired-outcome preference edge movement before game eval."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from weiss_rl.experiments import paired_outcome_preference_edge_margins as _edge_margins
from weiss_rl.experiments.paired_outcome_preference_edge_margins_cli import (
    build_paired_outcome_preference_edge_margins_parser,
    parse_paired_outcome_preference_edge_margins_args,
)
from weiss_rl.experiments.paired_outcome_preference_edge_margins_reporting import (
    paired_outcome_preference_edge_margins_output_line,
)
from weiss_rl.experiments.paired_outcome_preference_edge_margins_runtime import (
    run_paired_outcome_preference_edge_margins,
)

PairedOutcomePreferenceEdgeMarginConfig = _edge_margins.PairedOutcomePreferenceEdgeMarginConfig
build_paired_outcome_preference_edge_margin_report = _edge_margins.build_paired_outcome_preference_edge_margin_report
write_paired_outcome_preference_edge_margin_report = _edge_margins.write_paired_outcome_preference_edge_margin_report
_build_parser = build_paired_outcome_preference_edge_margins_parser


def main(argv: Sequence[str] | None = None) -> int:
    result = run_paired_outcome_preference_edge_margins(parse_args(argv))
    print(paired_outcome_preference_edge_margins_output_line(output_json=result.output_json, report=result.report))
    return result.exit_code


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return parse_paired_outcome_preference_edge_margins_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
