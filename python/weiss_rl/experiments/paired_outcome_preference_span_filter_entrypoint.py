#!/usr/bin/env python3
"""Filter paired-outcome preference replay rows to audited compact spans."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from weiss_rl.experiments.paired_outcome_preference_filters import (
    PairedOutcomePreferenceSpanFilterConfig,
    filter_paired_outcome_preference_spans,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--span-audit-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument(
        "--include-span-mode",
        action="append",
        default=[],
        choices=[
            "earliest",
            "densest",
            "all_compact",
            "repeated_action_label",
            "repeated_family",
            "repeated_raw_action",
        ],
        help="Span selector to apply. Defaults to repeated_action_label plus repeated_family.",
    )
    parser.add_argument(
        "--allow-failed-audit",
        action="store_true",
        help="Permit filtering from a span audit whose mechanistic gate failed.",
    )
    parser.add_argument(
        "--keep-span-fill-rows",
        action="store_true",
        help="Keep every supervised row in each selected span instead of only differing-action steps.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _dataset, summary = filter_paired_outcome_preference_spans(
        PairedOutcomePreferenceSpanFilterConfig(
            dataset_path=args.dataset,
            span_audit_json=args.span_audit_json,
            output_dataset_path=args.output,
            output_summary_json=args.summary_json,
            include_span_modes=tuple(str(value) for value in args.include_span_mode),
            require_audit_pass=not bool(args.allow_failed_audit),
            keep_span_fill_rows=bool(args.keep_span_fill_rows),
        )
    )
    print(
        json.dumps(
            {
                "summary_json": args.summary_json.as_posix(),
                "output": args.output.as_posix(),
                "input_train_rows": summary["input_train_rows"],
                "output_train_rows": summary["output_train_rows"],
                "selected_span_count": summary["selected_span_count"],
                "selected_preference_pair_ids": summary["selected_preference_pair_ids"],
                "selected_opponents": summary["selected_opponents"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
