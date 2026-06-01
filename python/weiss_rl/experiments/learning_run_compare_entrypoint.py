from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from weiss_rl.experiments.learning_run_compare_core import (
    DIAGNOSTIC_METRIC_PATHS,
    POLICY_ALIGNMENT_ANCHOR_ALIASES,
    POLICY_ALIGNMENT_FAMILIES,
    build_run_learning_comparison,
    load_dev_eval_records,
    load_learning_progress_metrics,
    load_policy_alignment_metrics,
    write_learning_comparison_json,
)
from weiss_rl.experiments.learning_run_compare_core import (
    best_and_latest as _best_and_latest,
)
from weiss_rl.experiments.learning_run_compare_core import (
    dev_eval_records_from_eval_dirs as _dev_eval_records_from_eval_dirs,
)
from weiss_rl.experiments.learning_run_compare_core import (
    dev_eval_records_from_training_log as _dev_eval_records_from_training_log,
)
from weiss_rl.experiments.learning_run_compare_core import (
    family_alignment_by_name as _family_alignment_by_name,
)
from weiss_rl.experiments.learning_run_compare_core import (
    json_or_none as _json_or_none,
)
from weiss_rl.experiments.learning_run_compare_core import (
    named_score_range as _named_score_range,
)
from weiss_rl.experiments.learning_run_compare_core import (
    numeric_at_path as _numeric_at_path,
)
from weiss_rl.experiments.learning_run_compare_core import (
    policy_alignment_metric_prefix as _policy_alignment_metric_prefix,
)
from weiss_rl.experiments.learning_run_compare_core import (
    score_range as _score_range,
)

__all__ = [
    "DIAGNOSTIC_METRIC_PATHS",
    "POLICY_ALIGNMENT_ANCHOR_ALIASES",
    "POLICY_ALIGNMENT_FAMILIES",
    "_best_and_latest",
    "_dev_eval_records_from_eval_dirs",
    "_dev_eval_records_from_training_log",
    "_family_alignment_by_name",
    "_json_or_none",
    "_named_score_range",
    "_numeric_at_path",
    "_policy_alignment_metric_prefix",
    "_score_range",
    "build_learning_run_compare_parser",
    "build_run_learning_comparison",
    "load_dev_eval_records",
    "load_learning_progress_metrics",
    "load_policy_alignment_metrics",
    "main",
    "run_learning_run_compare_from_args",
    "write_learning_comparison_json",
]


def build_learning_run_compare_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare periodic dev-eval learning trajectories across runs")
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--fragility-threshold", type=float, default=0.25)
    parser.add_argument("--anchor-fragility-threshold", type=float, default=0.25)
    return parser


def run_learning_run_compare_from_args(args: Any) -> dict[str, Any]:
    return build_run_learning_comparison(
        [path.resolve() for path in args.run_dir],
        fragility_threshold=float(args.fragility_threshold),
        anchor_fragility_threshold=float(args.anchor_fragility_threshold),
    )


def main() -> None:
    args = build_learning_run_compare_parser().parse_args()
    summary = run_learning_run_compare_from_args(args)
    if args.output_json is None:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    write_learning_comparison_json(args.output_json, summary)
    print(args.output_json)


if __name__ == "__main__":
    main()
