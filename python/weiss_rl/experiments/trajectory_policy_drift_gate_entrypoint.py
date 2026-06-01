#!/usr/bin/env python3
"""Gate trajectory policy drift diagnostics before game eval."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from weiss_rl.experiments.trajectory_policy_drift_gate import (
    TrajectoryPolicyDriftGateConfig,
    evaluate_trajectory_policy_drift_gate,
    write_trajectory_policy_drift_gate,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drift-report-json", required=True, type=Path)
    parser.add_argument("--candidate-label", default=None)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--max-lost-target-top-action-rate", type=float, default=0.0)
    parser.add_argument("--min-gained-target-top-action-rate", type=float, default=0.0)
    parser.add_argument("--min-gain-minus-loss-rate", type=float, default=0.0)
    parser.add_argument("--max-top-family-changed-rate", type=float, default=0.0)
    parser.add_argument("--min-mean-target-probability-delta", type=float, default=0.0)
    parser.add_argument("--max-target-probability-drop", type=float, default=0.0)
    parser.add_argument("--max-top-action-match-drop-rate", type=float, default=0.0)
    parser.add_argument(
        "--top-action-near-tie-margin",
        type=float,
        default=None,
        help="Log-prob margin threshold used to classify top-action flips as near ties.",
    )
    parser.add_argument(
        "--max-lost-target-non-near-tie-rate",
        type=float,
        default=None,
        help="Maximum row rate for lost-target top-action flips whose top-over-target margin exceeds the near-tie threshold.",
    )
    parser.add_argument(
        "--max-top-action-changed-non-near-tie-rate",
        type=float,
        default=None,
        help="Maximum row rate for any top-action flip whose top-over-target margin exceeds the near-tie threshold.",
    )
    parser.add_argument("--allow-missing-context", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = evaluate_trajectory_policy_drift_gate(
        TrajectoryPolicyDriftGateConfig(
            drift_report_json=args.drift_report_json,
            candidate_label=None if args.candidate_label is None else str(args.candidate_label),
            max_lost_target_top_action_rate=float(args.max_lost_target_top_action_rate),
            min_gained_target_top_action_rate=float(args.min_gained_target_top_action_rate),
            min_gain_minus_loss_rate=float(args.min_gain_minus_loss_rate),
            max_top_family_changed_rate=float(args.max_top_family_changed_rate),
            min_mean_target_probability_delta=float(args.min_mean_target_probability_delta),
            max_target_probability_drop=float(args.max_target_probability_drop),
            max_top_action_match_drop_rate=float(args.max_top_action_match_drop_rate),
            top_action_near_tie_margin=None
            if args.top_action_near_tie_margin is None
            else float(args.top_action_near_tie_margin),
            max_lost_target_non_near_tie_rate=None
            if args.max_lost_target_non_near_tie_rate is None
            else float(args.max_lost_target_non_near_tie_rate),
            max_top_action_changed_non_near_tie_rate=None
            if args.max_top_action_changed_non_near_tie_rate is None
            else float(args.max_top_action_changed_non_near_tie_rate),
            require_context=not bool(args.allow_missing_context),
        )
    )
    write_trajectory_policy_drift_gate(args.output_json, report)
    print(
        json.dumps(
            {
                "output_json": args.output_json.as_posix(),
                "passed": bool(report["passed"]),
                "failures": report["failures"],
                "summary": report["summary"],
            },
            sort_keys=True,
        )
    )
    return 0 if bool(report["passed"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
