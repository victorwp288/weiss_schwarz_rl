"""Runtime orchestration for paired-outcome preference surface-prototype diagnostics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.experiments.paired_outcome_preference_surface_prototypes import (
    PairedOutcomePreferenceSurfacePrototypeConfig,
    build_paired_outcome_preference_surface_prototype_report,
    write_paired_outcome_preference_surface_prototype_report,
)


@dataclass(frozen=True, slots=True)
class PairedOutcomePreferenceSurfacePrototypeRunResult:
    output_json: Path
    report: dict[str, Any]


def run_paired_outcome_preference_surface_prototype(
    args: argparse.Namespace,
) -> PairedOutcomePreferenceSurfacePrototypeRunResult:
    report = build_paired_outcome_preference_surface_prototype_report(
        paired_outcome_preference_surface_prototype_config_from_args(args)
    )
    write_paired_outcome_preference_surface_prototype_report(args.output_json, report)
    return PairedOutcomePreferenceSurfacePrototypeRunResult(output_json=args.output_json, report=report)


def paired_outcome_preference_surface_prototype_config_from_args(
    args: argparse.Namespace,
) -> PairedOutcomePreferenceSurfacePrototypeConfig:
    return PairedOutcomePreferenceSurfacePrototypeConfig(
        prototype_dataset_path=args.prototype_dataset,
        probe_dataset_paths=tuple(args.probe_dataset),
        probe_labels=tuple(str(item) for item in args.probe_label),
        stack_config_path=args.stack_config,
        opponent_context_policy_ids=tuple(str(item) for item in args.opponent_context_policy_id),
        key_mode=str(args.key_mode),
        opponent_key_mode=str(args.opponent_key_mode),
        max_examples=int(args.max_examples),
    )


__all__ = [
    "PairedOutcomePreferenceSurfacePrototypeRunResult",
    "paired_outcome_preference_surface_prototype_config_from_args",
    "run_paired_outcome_preference_surface_prototype",
]
