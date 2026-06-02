"""Runtime execution for the paper-readiness check CLI."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.eval.paper_readiness import build_paper_readiness_summary, write_paper_readiness_json
from weiss_rl.eval.readiness.check_cli import default_readiness_json

BuildPaperReadinessSummaryFn = Callable[..., dict[str, Any]]
WritePaperReadinessJsonFn = Callable[[Path, Mapping[str, Any]], None]


@dataclass(frozen=True, slots=True)
class PaperReadinessCheckResult:
    readiness_json: Path
    payload: dict[str, Any]


def run_paper_readiness_check(
    args: argparse.Namespace,
    *,
    build_paper_readiness_summary_fn: BuildPaperReadinessSummaryFn = build_paper_readiness_summary,
    write_paper_readiness_json_fn: WritePaperReadinessJsonFn = write_paper_readiness_json,
) -> PaperReadinessCheckResult:
    readiness_json = args.readiness_json or default_readiness_json(
        run_dir=args.run_dir,
        final_eval_dir=args.final_eval_dir,
    )
    payload = build_paper_readiness_summary_fn(
        run_dir=args.run_dir,
        final_eval_dir=args.final_eval_dir,
        focal_policy_id=args.focal_policy_id.strip() or None,
        baseline_policy_id=args.baseline_policy_id,
        max_truncation_rate=args.max_truncation_rate,
        seat_bias_max_abs_delta=args.seat_bias_max_abs_delta,
        seat_bias_posterior_min=args.seat_bias_posterior_min,
        baseline_win_rate_threshold=args.baseline_win_rate_threshold,
        baseline_posterior_min=args.baseline_posterior_min,
    )
    write_paper_readiness_json_fn(readiness_json, payload)
    return PaperReadinessCheckResult(readiness_json=readiness_json, payload=payload)
