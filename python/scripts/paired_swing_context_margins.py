#!/usr/bin/env python3
"""Emit row-level opponent-context log-prob margins for paired-swing replay rows."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from weiss_rl.experiments.paired_swing_context_margins import (
    PairedSwingContextMarginConfig,
    build_paired_swing_context_margin_report,
    write_paired_swing_context_margin_report,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--stack-config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--positive-action-source", default="actions")
    parser.add_argument("--negative-action-source", default="teacher_action")
    parser.add_argument("--report-action-id", action="append", default=[104, 124], type=int)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = build_paired_swing_context_margin_report(
        PairedSwingContextMarginConfig(
            dataset_path=args.dataset,
            stack_config_path=args.stack_config,
            run_dir=args.run_dir,
            checkpoint_path=args.checkpoint,
            positive_action_source=str(args.positive_action_source),
            negative_action_source=str(args.negative_action_source),
            report_action_ids=tuple(int(action_id) for action_id in args.report_action_id),
        )
    )
    write_paired_swing_context_margin_report(args.output_json, report)
    print(
        json.dumps(
            {
                "output_json": args.output_json.as_posix(),
                "row_count": report["row_count"],
                "context_episode_count": report["context_episode_count"],
                "missing_context_episode_count": report["context_coverage"]["missing_context_episode_count"],
                "positive_margin_min": report["positive_margin_min"],
                "positive_margin_mean": report["positive_margin_mean"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
