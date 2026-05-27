#!/usr/bin/env python3
"""Filter paired-outcome preference replay episodes by pair metadata."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from weiss_rl.experiments.paired_outcome_preference_filters import (
    PairedOutcomePreferenceFilterConfig,
    filter_paired_outcome_preference_dataset,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--include-preference-pair-id", action="append", default=[], type=int)
    parser.add_argument("--exclude-preference-pair-id", action="append", default=[], type=int)
    parser.add_argument("--include-source-pair-index", action="append", default=[], type=int)
    parser.add_argument("--exclude-source-pair-index", action="append", default=[], type=int)
    parser.add_argument("--include-source-opponent-policy-id", action="append", default=[])
    parser.add_argument("--exclude-source-opponent-policy-id", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _dataset, summary = filter_paired_outcome_preference_dataset(
        PairedOutcomePreferenceFilterConfig(
            dataset_path=args.dataset,
            output_dataset_path=args.output,
            output_summary_json=args.summary_json,
            include_preference_pair_ids=tuple(int(value) for value in args.include_preference_pair_id),
            exclude_preference_pair_ids=tuple(int(value) for value in args.exclude_preference_pair_id),
            include_source_pair_indices=tuple(int(value) for value in args.include_source_pair_index),
            exclude_source_pair_indices=tuple(int(value) for value in args.exclude_source_pair_index),
            include_source_opponent_policy_ids=tuple(str(value) for value in args.include_source_opponent_policy_id),
            exclude_source_opponent_policy_ids=tuple(str(value) for value in args.exclude_source_opponent_policy_id),
        )
    )
    print(
        json.dumps(
            {
                "summary_json": args.summary_json.as_posix(),
                "output": args.output.as_posix(),
                "output_episode_count": summary["output_episode_count"],
                "output_train_rows": summary["output_train_rows"],
                "selected_preference_pair_ids": summary["selected_preference_pair_ids"],
                "selected_source_pair_indices": summary["selected_source_pair_indices"],
                "selected_source_opponent_policy_ids": summary["selected_source_opponent_policy_ids"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
