#!/usr/bin/env python3
"""Merge preferred/rejected trajectory datasets into explicit preference replay data."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from weiss_rl.experiments import paired_outcome_preference_dataset as _dataset
from weiss_rl.experiments.paired_outcome_preference_dataset_cli import (
    build_paired_outcome_preference_dataset_parser,
    parse_opponent_match_aliases,
    parse_paired_outcome_preference_dataset_args,
)
from weiss_rl.experiments.paired_outcome_preference_dataset_reporting import (
    paired_outcome_preference_dataset_output_line,
)
from weiss_rl.experiments.paired_outcome_preference_dataset_runtime import (
    run_paired_outcome_preference_dataset,
)

PairedOutcomePreferenceDatasetConfig = _dataset.PairedOutcomePreferenceDatasetConfig
build_paired_outcome_preference_dataset = _dataset.build_paired_outcome_preference_dataset
_build_parser = build_paired_outcome_preference_dataset_parser
_parse_opponent_match_aliases = parse_opponent_match_aliases


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return parse_paired_outcome_preference_dataset_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    result = run_paired_outcome_preference_dataset(parse_args(argv))
    print(
        paired_outcome_preference_dataset_output_line(
            output_dataset=result.output_dataset,
            dataset=result.dataset,
            summary=result.summary,
        )
    )


if __name__ == "__main__":
    main()
