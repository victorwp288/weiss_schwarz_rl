#!/usr/bin/env python3
"""Report exact prototype-key coverage across paired-outcome preference datasets."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from weiss_rl.experiments import paired_outcome_preference_surface_prototypes as _surface_prototypes
from weiss_rl.experiments.paired_outcome_preference_surface_prototypes_cli import (
    build_paired_outcome_preference_surface_prototype_parser,
    parse_paired_outcome_preference_surface_prototype_args,
)
from weiss_rl.experiments.paired_outcome_preference_surface_prototypes_reporting import (
    paired_outcome_preference_surface_prototype_output_line,
)
from weiss_rl.experiments.paired_outcome_preference_surface_prototypes_runtime import (
    run_paired_outcome_preference_surface_prototype,
)

PairedOutcomePreferenceSurfacePrototypeConfig = _surface_prototypes.PairedOutcomePreferenceSurfacePrototypeConfig
build_paired_outcome_preference_surface_prototype_report = (
    _surface_prototypes.build_paired_outcome_preference_surface_prototype_report
)
write_paired_outcome_preference_surface_prototype_report = (
    _surface_prototypes.write_paired_outcome_preference_surface_prototype_report
)
_build_parser = build_paired_outcome_preference_surface_prototype_parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return parse_paired_outcome_preference_surface_prototype_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result = run_paired_outcome_preference_surface_prototype(parse_args(argv))
    print(paired_outcome_preference_surface_prototype_output_line(output_json=result.output_json, report=result.report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
