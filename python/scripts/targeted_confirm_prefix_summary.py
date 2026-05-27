#!/usr/bin/env python3
"""Derive a paired-seed prefix summary from a larger targeted-confirm artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from weiss_rl.experiments.targeted_confirm_prefix import (
    TargetedConfirmPrefixConfig,
    derive_targeted_confirm_prefix_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-summary-json", required=True, type=Path)
    parser.add_argument("--paired-seeds", required=True, type=int)
    parser.add_argument("--output-summary-json", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = derive_targeted_confirm_prefix_summary(
        TargetedConfirmPrefixConfig(
            source_summary_json=args.source_summary_json,
            output_summary_json=args.output_summary_json,
            paired_seeds=int(args.paired_seeds),
        )
    )
    print(
        json.dumps(
            {
                "output_summary_json": args.output_summary_json.as_posix(),
                "paired_seeds": summary["paired_seeds"],
                "rows": len(summary["rows"]),
                "overall": summary["overall"],
                "anchor_subset": summary["anchor_subset"],
                "legacy_subset": summary["legacy_subset"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
