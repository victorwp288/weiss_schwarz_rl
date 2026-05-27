#!/usr/bin/env python3
"""Compare paired outcomes between two targeted-confirm summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from weiss_rl.experiments.main_league_multiobjective_gate import FIXED_THESIS_OPPONENTS
from weiss_rl.experiments.paired_outcome_compare import (
    PairedOutcomeCompareConfig,
    compare_paired_targeted_outcomes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-summary-json", required=True, type=Path)
    parser.add_argument("--candidate-summary-json", required=True, type=Path)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--fixed-opponent", action="append", default=None)
    parser.add_argument(
        "--learned-opponent",
        action="append",
        default=[],
        help="Learned/champion/hard-negative opponent to group separately. When omitted, shared non-fixed rows are inferred.",
    )
    parser.add_argument("--max-examples", type=int, default=50)
    parser.add_argument(
        "--pair-index-split",
        type=int,
        default=None,
        help="Optional pair-index boundary for first/extension split summaries, e.g. 64 for confirm128.",
    )
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare_paired_targeted_outcomes(
        PairedOutcomeCompareConfig(
            baseline_summary_json=args.baseline_summary_json.resolve(),
            candidate_summary_json=args.candidate_summary_json.resolve(),
            baseline_label=str(args.baseline_label),
            candidate_label=str(args.candidate_label),
            fixed_opponents=tuple(args.fixed_opponent or FIXED_THESIS_OPPONENTS),
            learned_opponents=tuple(str(item) for item in args.learned_opponent),
            max_examples=int(args.max_examples),
            pair_index_split=args.pair_index_split,
        )
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    groups = report["groups"]
    print(
        json.dumps(
            {
                "output_json": args.output_json.as_posix(),
                "all_delta_wins": groups["all_compared"]["delta_wins"],
                "fixed_delta_wins": groups["fixed_baselines"]["delta_wins"],
                "learned_delta_wins": groups["learned_opponents"]["delta_wins"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
