#!/usr/bin/env python3
"""Apply the loose god-search escalation gate to paired outcome comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from weiss_rl.experiments.god_search_scorecard import GodSearchScorecardConfig, build_god_search_scorecard
from weiss_rl.experiments.main_league_multiobjective_gate import FIXED_THESIS_OPPONENTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-json", action="append", required=True, type=Path)
    parser.add_argument("--fixed-opponent", action="append", default=[])
    parser.add_argument("--min-all-delta-wins", type=int, default=1)
    parser.add_argument("--min-fixed-delta-wins", type=int, default=-2)
    parser.add_argument("--min-learned-delta-wins", type=int, default=0)
    parser.add_argument("--max-fixed-row-drop-wins", type=int, default=2)
    parser.add_argument("--max-any-row-drop-wins", type=int, default=4)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scorecard = build_god_search_scorecard(
        GodSearchScorecardConfig(
            compare_jsons=tuple(path.resolve() for path in args.compare_json),
            fixed_opponents=tuple(str(item) for item in args.fixed_opponent) or FIXED_THESIS_OPPONENTS,
            min_all_delta_wins=int(args.min_all_delta_wins),
            min_fixed_delta_wins=int(args.min_fixed_delta_wins),
            min_learned_delta_wins=int(args.min_learned_delta_wins),
            max_fixed_row_drop_wins=int(args.max_fixed_row_drop_wins),
            max_any_row_drop_wins=int(args.max_any_row_drop_wins),
        )
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(scorecard, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_json": args.output_json.as_posix(), "counts": scorecard["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
