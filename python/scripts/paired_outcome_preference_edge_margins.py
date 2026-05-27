"""Gate row-level paired-outcome preference edge movement before game eval."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from weiss_rl.experiments.paired_outcome_preference_edge_margins import (
    PairedOutcomePreferenceEdgeMarginConfig,
    build_paired_outcome_preference_edge_margin_report,
    write_paired_outcome_preference_edge_margin_report,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--stack-config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--reference-checkpoint", required=True, type=Path)
    parser.add_argument("--spec-bundle-json", default=None, type=Path)
    parser.add_argument("--include-same-action", action="store_true")
    parser.add_argument("--min-mean-delta", type=float, default=0.0)
    parser.add_argument("--min-min-delta", type=float, default=0.0)
    parser.add_argument("--min-edge-improved-fraction", type=float, default=1.0)
    parser.add_argument("--max-edge-worsened-fraction", type=float, default=0.0)
    parser.add_argument("--min-same-state-mean-delta", type=float, default=0.0)
    parser.add_argument("--min-required-group-mean-delta", type=float, default=0.0)
    parser.add_argument("--required-group", action="append", default=[])
    parser.add_argument("--allow-missing-context", action="store_true")
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = build_paired_outcome_preference_edge_margin_report(
        PairedOutcomePreferenceEdgeMarginConfig(
            dataset_path=args.dataset,
            stack_config_path=args.stack_config,
            run_dir=args.run_dir,
            checkpoint_path=args.checkpoint,
            reference_checkpoint_path=args.reference_checkpoint,
            spec_bundle_json=args.spec_bundle_json,
            include_same_action=bool(args.include_same_action),
            min_mean_delta=float(args.min_mean_delta),
            min_min_delta=float(args.min_min_delta),
            min_edge_improved_fraction=float(args.min_edge_improved_fraction),
            max_edge_worsened_fraction=float(args.max_edge_worsened_fraction),
            min_same_state_mean_delta=float(args.min_same_state_mean_delta),
            min_required_group_mean_delta=float(args.min_required_group_mean_delta),
            required_groups=tuple(str(item) for item in args.required_group),
            require_context=not bool(args.allow_missing_context),
        )
    )
    write_paired_outcome_preference_edge_margin_report(args.output_json, report)
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
