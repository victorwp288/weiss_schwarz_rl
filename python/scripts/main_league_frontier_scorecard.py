#!/usr/bin/env python3
"""Build a row-level scorecard for main-league frontier compare reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from weiss_rl.experiments.main_league_frontier_scorecard import (
    MainLeagueFrontierScorecardConfig,
    build_main_league_frontier_scorecard,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-json", action="append", default=[], type=Path)
    parser.add_argument(
        "--compare-glob",
        action="append",
        default=[],
        help="Repo-relative glob for paired_outcome_compare JSONs, e.g. diagnostics/paired_outcome_compare_*confirm64*.json",
    )
    parser.add_argument("--repo-root", default=Path("."), type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--min-sentinel-learned-delta-wins", type=int, default=0)
    parser.add_argument("--max-sentinel-fixed-row-drop-wins", type=int, default=0)
    parser.add_argument("--max-sentinel-learned-row-drop-wins", type=int, default=0)
    parser.add_argument("--min-full-fixed-delta-wins", type=int, default=0)
    parser.add_argument("--min-full-learned-delta-wins", type=int, default=0)
    parser.add_argument("--max-full-fixed-row-drop-wins", type=int, default=0)
    parser.add_argument("--max-confirm128-learned-row-drop-wins", type=int, default=-1)
    parser.add_argument("--max-confirm256-learned-row-drop-wins", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compare_jsons = _resolve_compare_jsons(
        repo_root=args.repo_root.resolve(),
        explicit=tuple(args.compare_json),
        globs=tuple(str(item) for item in args.compare_glob),
    )
    if not compare_jsons:
        raise SystemExit("at least one --compare-json or --compare-glob match is required")
    scorecard = build_main_league_frontier_scorecard(
        MainLeagueFrontierScorecardConfig(
            compare_jsons=compare_jsons,
            min_sentinel_learned_delta_wins=int(args.min_sentinel_learned_delta_wins),
            max_sentinel_fixed_row_drop_wins=int(args.max_sentinel_fixed_row_drop_wins),
            max_sentinel_learned_row_drop_wins=int(args.max_sentinel_learned_row_drop_wins),
            min_full_fixed_delta_wins=int(args.min_full_fixed_delta_wins),
            min_full_learned_delta_wins=int(args.min_full_learned_delta_wins),
            max_full_fixed_row_drop_wins=int(args.max_full_fixed_row_drop_wins),
            max_confirm128_learned_row_drop_wins=int(args.max_confirm128_learned_row_drop_wins),
            max_confirm256_learned_row_drop_wins=int(args.max_confirm256_learned_row_drop_wins),
        )
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(scorecard, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_json": args.output_json.as_posix(),
                "entries": scorecard["counts"]["total"],
                "stop": scorecard["counts"].get("stop", 0),
                "run_confirm128": scorecard["counts"].get("run_confirm128", 0),
                "run_confirm256": scorecard["counts"].get("run_confirm256", 0),
                "publishable_gate_candidate": scorecard["counts"].get("publishable_gate_candidate", 0),
            },
            sort_keys=True,
        )
    )


def _resolve_compare_jsons(*, repo_root: Path, explicit: tuple[Path, ...], globs: tuple[str, ...]) -> tuple[Path, ...]:
    paths: list[Path] = []
    for path in explicit:
        paths.append(path if path.is_absolute() else (repo_root / path))
    for pattern in globs:
        paths.extend(repo_root.glob(pattern))
    resolved = tuple(dict.fromkeys(path.resolve() for path in paths if path.exists()))
    missing = [path for path in paths if not path.exists()]
    if missing:
        missing_text = ", ".join(path.as_posix() for path in missing)
        raise FileNotFoundError(f"compare JSON not found: {missing_text}")
    return resolved


if __name__ == "__main__":
    main()
