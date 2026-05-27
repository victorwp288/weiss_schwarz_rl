#!/usr/bin/env python3
"""Report exact prototype-key coverage across paired-outcome preference datasets."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from weiss_rl.experiments.paired_outcome_preference_surface_prototypes import (
    PairedOutcomePreferenceSurfacePrototypeConfig,
    build_paired_outcome_preference_surface_prototype_report,
    write_paired_outcome_preference_surface_prototype_report,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prototype-dataset", required=True, type=Path)
    parser.add_argument("--probe-dataset", action="append", required=True, type=Path)
    parser.add_argument(
        "--probe-label",
        action="append",
        default=[],
        help="Optional label for each --probe-dataset, in the same order.",
    )
    parser.add_argument(
        "--key-mode",
        choices=("current", "current_history", "current_history_opponent"),
        default="current_history_opponent",
    )
    parser.add_argument(
        "--opponent-key-mode",
        choices=("raw_policy_id", "context_index"),
        default="raw_policy_id",
    )
    parser.add_argument(
        "--stack-config",
        type=Path,
        default=None,
        help="Optional stack YAML used to resolve model opponent context indices.",
    )
    parser.add_argument(
        "--opponent-context-policy-id",
        action="append",
        default=[],
        help="Additional opponent context policy id used when --opponent-key-mode=context_index.",
    )
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = build_paired_outcome_preference_surface_prototype_report(
        PairedOutcomePreferenceSurfacePrototypeConfig(
            prototype_dataset_path=args.prototype_dataset,
            probe_dataset_paths=tuple(args.probe_dataset),
            probe_labels=tuple(args.probe_label),
            stack_config_path=args.stack_config,
            opponent_context_policy_ids=tuple(args.opponent_context_policy_id),
            key_mode=str(args.key_mode),
            opponent_key_mode=str(args.opponent_key_mode),
            max_examples=int(args.max_examples),
        )
    )
    write_paired_outcome_preference_surface_prototype_report(args.output_json, report)
    print(
        json.dumps(
            {
                "output_json": args.output_json.as_posix(),
                "key_mode": report["key_mode"],
                "opponent_key_mode": report["opponent_key_mode"],
                "prototype_train_rows": report["prototype"]["train_rows"],
                "prototype_unique_key_count": report["prototype"]["unique_key_count"],
                "prototype_conflicting_key_count": report["prototype"]["conflicting_key_count"],
                "probes": [
                    {
                        "label": probe["label"],
                        "train_rows": probe["train_rows"],
                        "matched_train_rows": probe["matched_train_rows"],
                        "matched_rate": probe["matched_rate"],
                        "unexpected_matched_rows": probe["unexpected_matched_rows"],
                        "conflicting_matched_key_count": probe["conflicting_matched_key_count"],
                    }
                    for probe in report["probes"]
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
