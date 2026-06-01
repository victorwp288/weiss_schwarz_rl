"""Runtime orchestration for paired-outcome preference edge-margin gates."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.experiments.paired_outcome_preference_edge_margins import (
    PairedOutcomePreferenceEdgeMarginConfig,
    build_paired_outcome_preference_edge_margin_report,
    write_paired_outcome_preference_edge_margin_report,
)


@dataclass(frozen=True, slots=True)
class PairedOutcomePreferenceEdgeMarginRunResult:
    output_json: Path
    report: dict[str, Any]

    @property
    def exit_code(self) -> int:
        return 0 if bool(self.report["passed"]) else 2


def run_paired_outcome_preference_edge_margins(
    args: argparse.Namespace,
) -> PairedOutcomePreferenceEdgeMarginRunResult:
    report = build_paired_outcome_preference_edge_margin_report(
        paired_outcome_preference_edge_margin_config_from_args(args)
    )
    write_paired_outcome_preference_edge_margin_report(args.output_json, report)
    return PairedOutcomePreferenceEdgeMarginRunResult(output_json=args.output_json, report=report)


def paired_outcome_preference_edge_margin_config_from_args(
    args: argparse.Namespace,
) -> PairedOutcomePreferenceEdgeMarginConfig:
    return PairedOutcomePreferenceEdgeMarginConfig(
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


__all__ = [
    "PairedOutcomePreferenceEdgeMarginRunResult",
    "paired_outcome_preference_edge_margin_config_from_args",
    "run_paired_outcome_preference_edge_margins",
]
