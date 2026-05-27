#!/usr/bin/env python3
"""Merge preferred/rejected trajectory datasets into explicit preference replay data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from weiss_rl.experiments.paired_outcome_preference_dataset import (
    PairedOutcomePreferenceDatasetConfig,
    build_paired_outcome_preference_dataset,
)


def _parse_opponent_match_aliases(values: list[str]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--opponent-match-alias must be FROM=TO, got: {value!r}")
        source, target = (part.strip() for part in value.split("=", 1))
        if not source or not target:
            raise SystemExit(f"--opponent-match-alias must have non-empty FROM and TO, got: {value!r}")
        aliases[source] = target
    return aliases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preferred-dataset", required=True, type=Path)
    parser.add_argument("--rejected-dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-json", default=None, type=Path)
    parser.add_argument("--max-pairs", default=None, type=int)
    parser.add_argument("--preferred-label", default="preferred")
    parser.add_argument("--rejected-label", default="rejected")
    parser.add_argument(
        "--opponent-match-alias",
        action="append",
        default=[],
        metavar="FROM=TO",
        help=(
            "Canonicalize opponent IDs only for preferred/rejected episode matching. "
            "The original source_opponent_policy_id metadata is preserved for context and audit."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset, summary = build_paired_outcome_preference_dataset(
        PairedOutcomePreferenceDatasetConfig(
            preferred_dataset=args.preferred_dataset.resolve(),
            rejected_dataset=args.rejected_dataset.resolve(),
            output_dataset=args.output,
            output_summary_json=args.summary_json,
            max_pairs=args.max_pairs,
            preferred_label=str(args.preferred_label),
            rejected_label=str(args.rejected_label),
            opponent_match_aliases=_parse_opponent_match_aliases(args.opponent_match_alias),
        )
    )
    print(
        json.dumps(
            {
                "output": args.output.as_posix(),
                "pair_count": summary["pair_count"],
                "episodes": dataset.episode_count,
                "train_rows": dataset.metadata.get("train_rows"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
