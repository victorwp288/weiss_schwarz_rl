"""Runtime orchestration for paired-swing opponent-context margin reports."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.experiments.paired_swing_context_margins import (
    PairedSwingContextMarginConfig,
    build_paired_swing_context_margin_report,
    write_paired_swing_context_margin_report,
)


@dataclass(frozen=True, slots=True)
class PairedSwingContextMarginsRunResult:
    output_json: Path
    report: dict[str, Any]


def run_paired_swing_context_margins(args: argparse.Namespace) -> PairedSwingContextMarginsRunResult:
    report = build_paired_swing_context_margin_report(paired_swing_context_margins_config_from_args(args))
    write_paired_swing_context_margin_report(args.output_json, report)
    return PairedSwingContextMarginsRunResult(output_json=args.output_json, report=report)


def paired_swing_context_margins_config_from_args(args: argparse.Namespace) -> PairedSwingContextMarginConfig:
    return PairedSwingContextMarginConfig(
        dataset_path=args.dataset,
        stack_config_path=args.stack_config,
        run_dir=args.run_dir,
        checkpoint_path=args.checkpoint,
        positive_action_source=str(args.positive_action_source),
        negative_action_source=str(args.negative_action_source),
        report_action_ids=tuple(int(action_id) for action_id in args.report_action_id),
    )


__all__ = [
    "PairedSwingContextMarginsRunResult",
    "paired_swing_context_margins_config_from_args",
    "run_paired_swing_context_margins",
]
