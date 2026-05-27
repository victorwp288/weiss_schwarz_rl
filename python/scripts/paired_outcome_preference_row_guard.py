#!/usr/bin/env python3
"""Gate row-level target-action drift on paired outcome preference replay."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from weiss_rl.experiments.paired_outcome_preference_row_guard import (
    PairedOutcomePreferenceRowGuardConfig,
    build_paired_outcome_preference_row_guard,
    write_paired_outcome_preference_row_guard,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--stack-config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--reference-checkpoint", required=True, type=Path)
    parser.add_argument("--protected-group", action="append", default=[])
    parser.add_argument("--required-group", action="append", default=[])
    parser.add_argument("--min-required-group-mean-logp-delta", type=float, default=0.0)
    parser.add_argument("--min-protected-mean-logp-delta", type=float, default=0.0)
    parser.add_argument("--max-protected-row-worsened-fraction", type=float, default=0.0)
    parser.add_argument("--max-protected-rank-worsened-fraction", type=float, default=0.0)
    parser.add_argument("--max-protected-top-family-changed-rate", type=float, default=0.0)
    parser.add_argument("--top-action-near-tie-margin", type=float, default=1e-5)
    parser.add_argument("--max-protected-lost-target-non-near-tie-rate", type=float, default=0.0)
    parser.add_argument("--allow-missing-context", action="store_true")
    parser.add_argument("--max-examples", type=int, default=25)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = build_paired_outcome_preference_row_guard(
        PairedOutcomePreferenceRowGuardConfig(
            dataset_path=args.dataset,
            stack_config_path=args.stack_config,
            run_dir=args.run_dir,
            checkpoint_path=args.checkpoint,
            reference_checkpoint_path=args.reference_checkpoint,
            protected_groups=tuple(str(item) for item in args.protected_group),
            required_groups=tuple(str(item) for item in args.required_group),
            min_required_group_mean_logp_delta=float(args.min_required_group_mean_logp_delta),
            min_protected_mean_logp_delta=float(args.min_protected_mean_logp_delta),
            max_protected_row_worsened_fraction=float(args.max_protected_row_worsened_fraction),
            max_protected_rank_worsened_fraction=float(args.max_protected_rank_worsened_fraction),
            max_protected_top_family_changed_rate=float(args.max_protected_top_family_changed_rate),
            top_action_near_tie_margin=float(args.top_action_near_tie_margin),
            max_protected_lost_target_non_near_tie_rate=float(args.max_protected_lost_target_non_near_tie_rate),
            require_context=not bool(args.allow_missing_context),
            max_examples=int(args.max_examples),
        )
    )
    write_paired_outcome_preference_row_guard(args.output_json, report)
    print(
        json.dumps(
            {
                "output_json": args.output_json.as_posix(),
                "passed": report["passed"],
                "failures": report["failures"],
                "row_count": report["row_count"],
                "current_context_episode_count": report["current_context_episode_count"],
                "reference_context_episode_count": report["reference_context_episode_count"],
                "groups": [
                    {
                        "label": group["label"],
                        "protected": group["protected"],
                        "required": group["required"],
                        "row_count": group["row_count"],
                        "mean_target_logp_delta": group["mean_target_logp_delta"],
                        "row_worsened_fraction": group["row_worsened_fraction"],
                        "rank_worsened_fraction": group["rank_worsened_fraction"],
                        "top_family_changed_rate": group["top_family_changed_rate"],
                        "lost_target_non_near_tie_rate": group["lost_target_non_near_tie_rate"],
                    }
                    for group in report["groups"]
                ],
            },
            sort_keys=True,
        )
    )
    return 0 if bool(report["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
