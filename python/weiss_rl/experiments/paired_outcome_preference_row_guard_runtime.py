"""Runtime orchestration for paired-outcome preference row guards."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.experiments.paired_outcome_preference_row_guard import (
    PairedOutcomePreferenceRowGuardConfig,
    build_paired_outcome_preference_row_guard,
    write_paired_outcome_preference_row_guard,
)


@dataclass(frozen=True, slots=True)
class PairedOutcomePreferenceRowGuardRunResult:
    output_json: Path
    report: dict[str, Any]

    @property
    def exit_code(self) -> int:
        return 0 if bool(self.report["passed"]) else 1


def run_paired_outcome_preference_row_guard(
    args: argparse.Namespace,
) -> PairedOutcomePreferenceRowGuardRunResult:
    report = build_paired_outcome_preference_row_guard(paired_outcome_preference_row_guard_config_from_args(args))
    write_paired_outcome_preference_row_guard(args.output_json, report)
    return PairedOutcomePreferenceRowGuardRunResult(output_json=args.output_json, report=report)


def paired_outcome_preference_row_guard_config_from_args(
    args: argparse.Namespace,
) -> PairedOutcomePreferenceRowGuardConfig:
    return PairedOutcomePreferenceRowGuardConfig(
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


__all__ = [
    "PairedOutcomePreferenceRowGuardRunResult",
    "paired_outcome_preference_row_guard_config_from_args",
    "run_paired_outcome_preference_row_guard",
]
