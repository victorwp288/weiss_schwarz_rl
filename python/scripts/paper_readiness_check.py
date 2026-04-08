from __future__ import annotations

import argparse
import sys
from pathlib import Path

from weiss_rl.eval.paper_readiness import (
    DEFAULT_BASELINE_POLICY_ID,
    DEFAULT_BASELINE_POSTERIOR_MIN,
    DEFAULT_BASELINE_WIN_RATE_THRESHOLD,
    DEFAULT_SEAT_BIAS_MAX_ABS_DELTA,
    DEFAULT_SEAT_BIAS_POSTERIOR_MIN,
    DEFAULT_TRUNCATION_MAX_RATE,
    build_paper_readiness_summary,
    write_paper_readiness_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-readiness guardrails over final_eval artifacts")
    parser.add_argument(
        "--final-eval-dir",
        type=Path,
        required=True,
        help="Path to a final_eval artifact directory containing summary.json and matchup diagnostics",
    )
    parser.add_argument(
        "--readiness-json",
        type=Path,
        default=None,
        help="Output path for the readiness summary JSON (default: <final-eval-dir>/paper_readiness_summary.json)",
    )
    parser.add_argument(
        "--focal-policy-id",
        type=str,
        default="",
        help="Policy to check against the B0 baseline (default: first non-baseline policy in policy_ids)",
    )
    parser.add_argument(
        "--baseline-policy-id",
        type=str,
        default=DEFAULT_BASELINE_POLICY_ID,
        help="Baseline policy ID used for the win-rate guardrail",
    )
    parser.add_argument(
        "--max-truncation-rate",
        type=float,
        default=DEFAULT_TRUNCATION_MAX_RATE,
        help="Maximum allowed aggregate truncation rate across final_eval",
    )
    parser.add_argument(
        "--seat-bias-max-abs-delta",
        type=float,
        default=DEFAULT_SEAT_BIAS_MAX_ABS_DELTA,
        help="Seat-bias alarm margin around 0.5 for the global decisive seat0 win rate",
    )
    parser.add_argument(
        "--seat-bias-posterior-min",
        type=float,
        default=DEFAULT_SEAT_BIAS_POSTERIOR_MIN,
        help="Posterior probability threshold for triggering the seat-bias alarm",
    )
    parser.add_argument(
        "--baseline-win-rate-threshold",
        type=float,
        default=DEFAULT_BASELINE_WIN_RATE_THRESHOLD,
        help="Minimum posterior win-rate threshold for the focal policy versus the baseline",
    )
    parser.add_argument(
        "--baseline-posterior-min",
        type=float,
        default=DEFAULT_BASELINE_POSTERIOR_MIN,
        help="Required posterior probability of exceeding the baseline win-rate threshold",
    )
    args = parser.parse_args()

    readiness_json = args.readiness_json or (args.final_eval_dir / "paper_readiness_summary.json")
    payload = build_paper_readiness_summary(
        final_eval_dir=args.final_eval_dir,
        focal_policy_id=args.focal_policy_id.strip() or None,
        baseline_policy_id=args.baseline_policy_id,
        max_truncation_rate=args.max_truncation_rate,
        seat_bias_max_abs_delta=args.seat_bias_max_abs_delta,
        seat_bias_posterior_min=args.seat_bias_posterior_min,
        baseline_win_rate_threshold=args.baseline_win_rate_threshold,
        baseline_posterior_min=args.baseline_posterior_min,
    )
    write_paper_readiness_json(readiness_json, payload)

    print(f"Paper readiness summary JSON: {readiness_json}")
    if payload["passed"]:
        print("Paper readiness checks passed.")
        return

    alarms = ", ".join(str(alarm) for alarm in payload["alarms"])
    print(f"Paper readiness checks failed: {alarms}", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
