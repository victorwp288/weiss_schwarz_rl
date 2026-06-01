#!/usr/bin/env python3
"""Merge split targeted-confirm summaries into one row surface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from weiss_rl.experiments.targeted_confirm_merge import (
    merge_targeted_confirm_summaries,
    write_merged_targeted_confirm_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", action="append", required=True, type=Path)
    parser.add_argument("--label", default="merged")
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = merge_targeted_confirm_summaries(
        tuple(path.resolve() for path in args.summary_json),
        label=str(args.label),
    )
    write_merged_targeted_confirm_summary(args.output_json, summary)
    print(
        json.dumps(
            {
                "output_json": args.output_json.as_posix(),
                "rows": len(summary["rows"]),
                "wins": summary["overall"]["wins"],
                "games": summary["overall"]["games"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
