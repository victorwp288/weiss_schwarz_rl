#!/usr/bin/env python3
"""Emit mechanistic DPO-style margins for paired outcome preference replay data."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from weiss_rl.experiments.paired_outcome_preference_margins import (
    PairedOutcomePreferenceMarginConfig,
    build_paired_outcome_preference_margin_report,
    write_paired_outcome_preference_margin_report,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--stack-config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--reference-checkpoint", required=True, type=Path)
    parser.add_argument("--aggregation", choices=("mean", "sum"), default="mean")
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = build_paired_outcome_preference_margin_report(
        PairedOutcomePreferenceMarginConfig(
            dataset_path=args.dataset,
            stack_config_path=args.stack_config,
            run_dir=args.run_dir,
            checkpoint_path=args.checkpoint,
            reference_checkpoint_path=args.reference_checkpoint,
            aggregation=str(args.aggregation),
        )
    )
    write_paired_outcome_preference_margin_report(args.output_json, report)
    print(
        json.dumps(
            {
                "output_json": args.output_json.as_posix(),
                "pair_count": report["pair_count"],
                "train_rows": report["train_rows"],
                "dpo_margin_mean": report["dpo_margin_mean"],
                "dpo_margin_min": report["dpo_margin_min"],
                "satisfied_fraction": report["satisfied_fraction"],
                "current_context_episode_count": report["current_context_episode_count"],
                "reference_context_episode_count": report["reference_context_episode_count"],
                "current_missing_context_episode_count": report["current_context_coverage"][
                    "missing_context_episode_count"
                ],
                "reference_missing_context_episode_count": report["reference_context_coverage"][
                    "missing_context_episode_count"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
