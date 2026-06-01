#!/usr/bin/env python3
"""Extract exact paired-confirm flip targets for replay audits and repair datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from weiss_rl.experiments.main_league_multiobjective_gate import FIXED_THESIS_OPPONENTS
from weiss_rl.experiments.paired_flip_targets import (
    PairedFlipTargetsConfig,
    build_paired_flip_targets,
    write_paired_flip_targets_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-summary-json", required=True, type=Path)
    parser.add_argument("--candidate-summary-json", required=True, type=Path)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument(
        "--opponent",
        action="append",
        default=None,
        help="Opponent policy id to extract. Repeat for multiple. Defaults to fixed opponents plus learned opponents.",
    )
    parser.add_argument("--fixed-opponent", action="append", default=None)
    parser.add_argument("--learned-opponent", action="append", default=[])
    parser.add_argument("--opponent-pool-jsonl", action="append", type=Path, default=[])
    parser.add_argument(
        "--flip-kind",
        choices=("baseline_win_candidate_nonwin", "baseline_nonwin_candidate_win", "changed_outcome"),
        default="baseline_win_candidate_nonwin",
    )
    parser.add_argument("--pair-index-min", type=int, default=None)
    parser.add_argument("--pair-index-max", type=int, default=None)
    parser.add_argument("--max-targets-per-opponent", type=int, default=None)
    parser.add_argument(
        "--episode-source",
        choices=("baseline", "candidate", "both"),
        default="candidate",
        help="Which source episodes to write when --episode-sets-dir is provided.",
    )
    parser.add_argument(
        "--episode-sets-dir",
        type=Path,
        default=None,
        help="Optional output directory for complete seat-swapped episodes.jsonl subsets.",
    )
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fixed_opponents = tuple(args.fixed_opponent or FIXED_THESIS_OPPONENTS)
    learned_opponents = tuple(str(item) for item in args.learned_opponent)
    opponents = tuple(args.opponent or (*fixed_opponents, *learned_opponents))
    report = build_paired_flip_targets(
        PairedFlipTargetsConfig(
            baseline_summary_json=args.baseline_summary_json.resolve(),
            candidate_summary_json=args.candidate_summary_json.resolve(),
            baseline_label=str(args.baseline_label),
            candidate_label=str(args.candidate_label),
            opponents=opponents,
            fixed_opponents=fixed_opponents,
            learned_opponents=learned_opponents,
            opponent_pool_jsonls=tuple(path.resolve() for path in args.opponent_pool_jsonl),
            flip_kind=args.flip_kind,
            pair_index_min=args.pair_index_min,
            pair_index_max=args.pair_index_max,
            max_targets_per_opponent=args.max_targets_per_opponent,
            episode_source=args.episode_source,
            episode_sets_dir=None if args.episode_sets_dir is None else args.episode_sets_dir.resolve(),
        )
    )
    write_paired_flip_targets_json(args.output_json, report)
    print(
        json.dumps(
            {
                "output_json": args.output_json.as_posix(),
                "target_count": report["target_count"],
                "episode_set_count": len(report["episode_sets"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
