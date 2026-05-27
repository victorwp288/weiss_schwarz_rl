#!/usr/bin/env python3
"""Report compact trajectory spans in paired-outcome preference replay datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from weiss_rl.experiments.paired_outcome_preference_span_audit import (
    PairedOutcomePreferenceSpanAuditConfig,
    build_paired_outcome_preference_span_audit,
    write_paired_outcome_preference_span_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--spec-bundle-json", default=None, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--max-gap", type=int, default=1)
    parser.add_argument("--max-compact-span-width", type=int, default=8)
    parser.add_argument("--min-repeated-pair-count", type=int, default=2)
    parser.add_argument("--max-examples", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_paired_outcome_preference_span_audit(
        PairedOutcomePreferenceSpanAuditConfig(
            dataset_path=args.dataset.resolve(),
            spec_bundle_json=None if args.spec_bundle_json is None else args.spec_bundle_json.resolve(),
            max_gap=int(args.max_gap),
            max_compact_span_width=int(args.max_compact_span_width),
            min_repeated_pair_count=int(args.min_repeated_pair_count),
            max_examples=int(args.max_examples),
        )
    )
    write_paired_outcome_preference_span_audit(args.output_json, report)
    print(
        json.dumps(
            {
                "output_json": args.output_json.as_posix(),
                "passed": report["span_gate"]["passed"],
                "complete_pair_count": report["complete_pair_count"],
                "different_action_count": report["different_action_count"],
                "compact_span_count": report["compact_span_count"],
                "passing_opponents": report["span_gate"]["passing_opponents"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
