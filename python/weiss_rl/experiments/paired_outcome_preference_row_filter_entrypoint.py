#!/usr/bin/env python3
"""Filter paired-outcome preference replay rows to aligned decision spans."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from weiss_rl.experiments.paired_outcome_preference_filters import (
    PairedOutcomePreferenceRowFilterConfig,
    filter_paired_outcome_preference_rows,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument(
        "--keep-aligned-same-action",
        action="store_true",
        help="Keep every aligned supervised step instead of only action-difference steps.",
    )
    parser.add_argument(
        "--exclude-same-family-action-differences",
        action="store_true",
        help="Drop aligned rows where preferred and rejected actions differ only within the same action family.",
    )
    parser.add_argument(
        "--require-same-current-state",
        action="store_true",
        help="Keep only aligned steps where preferred and rejected current-state hashes match.",
    )
    parser.add_argument(
        "--require-same-history",
        action="store_true",
        help="Keep only aligned steps where preferred and rejected trajectory-history hashes match.",
    )
    parser.add_argument(
        "--exclude-current-state-conflicts",
        action="store_true",
        help="Drop same-current-state reverse-label conflict rows from unconditioned preference replay.",
    )
    parser.add_argument(
        "--exclude-history-conflicts",
        action="store_true",
        help="Drop same-history reverse-label conflict rows from unconditioned preference replay.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _dataset, summary = filter_paired_outcome_preference_rows(
        PairedOutcomePreferenceRowFilterConfig(
            dataset_path=args.dataset,
            output_dataset_path=args.output,
            output_summary_json=args.summary_json,
            require_action_difference=not bool(args.keep_aligned_same_action),
            exclude_same_family_action_differences=bool(args.exclude_same_family_action_differences),
            require_same_current_state=bool(args.require_same_current_state),
            require_same_history=bool(args.require_same_history),
            exclude_current_state_conflicts=bool(args.exclude_current_state_conflicts),
            exclude_history_conflicts=bool(args.exclude_history_conflicts),
        )
    )
    print(
        json.dumps(
            {
                "summary_json": args.summary_json.as_posix(),
                "output": args.output.as_posix(),
                "input_train_rows": summary["input_train_rows"],
                "output_train_rows": summary["output_train_rows"],
                "exclude_same_family_action_differences": summary["exclude_same_family_action_differences"],
                "require_same_current_state": summary["require_same_current_state"],
                "require_same_history": summary["require_same_history"],
                "exclude_current_state_conflicts": summary["exclude_current_state_conflicts"],
                "exclude_history_conflicts": summary["exclude_history_conflicts"],
                "pair_count": len(summary["pair_summaries"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
