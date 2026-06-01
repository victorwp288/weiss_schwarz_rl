#!/usr/bin/env python3
"""Build a compact diagnostic summary from league training scalar logs."""

from __future__ import annotations

import argparse
from pathlib import Path

from weiss_rl.experiments.league_progress_summary import (
    build_league_progress_summary,
    write_league_progress_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize diagnostic league progress from scalars.jsonl")
    parser.add_argument("--scalars-jsonl", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--learned-opponent", action="append", default=[])
    parser.add_argument("--hard-negative-opponent", action="append", default=[])
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_league_progress_summary(
        scalars_jsonl=args.scalars_jsonl,
        learned_opponents=args.learned_opponent,
        hard_negative_opponents=args.hard_negative_opponent,
        notes=args.notes,
    )
    write_league_progress_summary(summary, args.output_json)
    print(args.output_json.as_posix())


if __name__ == "__main__":
    main()
