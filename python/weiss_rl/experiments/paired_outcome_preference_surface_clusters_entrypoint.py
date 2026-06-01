#!/usr/bin/env python3
"""Classify paired-outcome preference rows by public-surface separability."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from weiss_rl.experiments import paired_outcome_preference_surface_clusters as _surface_clusters
from weiss_rl.experiments.paired_outcome_preference_surface_clusters_cli import (
    build_paired_outcome_preference_surface_cluster_parser,
    parse_paired_outcome_preference_surface_cluster_args,
)
from weiss_rl.experiments.paired_outcome_preference_surface_clusters_reporting import (
    paired_outcome_preference_surface_cluster_output_line,
)
from weiss_rl.experiments.paired_outcome_preference_surface_clusters_runtime import (
    run_paired_outcome_preference_surface_cluster,
)

PairedOutcomePreferenceSurfaceClusterConfig = _surface_clusters.PairedOutcomePreferenceSurfaceClusterConfig
build_paired_outcome_preference_surface_cluster_report = (
    _surface_clusters.build_paired_outcome_preference_surface_cluster_report
)
write_paired_outcome_preference_surface_cluster_report = (
    _surface_clusters.write_paired_outcome_preference_surface_cluster_report
)
_build_parser = build_paired_outcome_preference_surface_cluster_parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return parse_paired_outcome_preference_surface_cluster_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result = run_paired_outcome_preference_surface_cluster(parse_args(argv))
    print(paired_outcome_preference_surface_cluster_output_line(output_json=result.output_json, report=result.report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
