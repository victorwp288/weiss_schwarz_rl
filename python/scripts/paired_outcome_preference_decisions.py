#!/usr/bin/env python3
"""Summarize paired-outcome preference decisions and same-state conflicts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from weiss_rl.experiments.paired_outcome_preference_decisions import (
    PairedOutcomePreferenceDecisionConfig,
    build_paired_outcome_preference_decision_report,
    write_paired_outcome_preference_decision_report,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--spec-bundle-json",
        type=Path,
        default=None,
        help="Optional spec_bundle.json used to decode action ids into families and slots.",
    )
    parser.add_argument("--max-examples", type=int, default=25)
    parser.add_argument("--top-action-edges", type=int, default=25)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = build_paired_outcome_preference_decision_report(
        PairedOutcomePreferenceDecisionConfig(
            dataset_path=args.dataset,
            spec_bundle_json=args.spec_bundle_json,
            max_examples=int(args.max_examples),
            top_action_edges=int(args.top_action_edges),
        )
    )
    write_paired_outcome_preference_decision_report(args.output_json, report)
    print(
        json.dumps(
            {
                "output_json": args.output_json.as_posix(),
                "preference_pair_count": report["preference_pair_count"],
                "complete_pair_count": report["complete_pair_count"],
                "aligned_different_action_count": report["aligned_different_action_count"],
                "same_current_state_edge_count": report["same_current_state_edge_count"],
                "same_current_state_different_action_edge_count": report[
                    "same_current_state_different_action_edge_count"
                ],
                "current_state_conflict_count": report["current_state_conflict_count"],
                "history_conflict_count": report["history_conflict_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
