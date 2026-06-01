"""Runtime orchestration for paired-outcome contrastive dataset generation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.experiments.paired_outcome_contrastive_build import (
    PairedOutcomeContrastiveBuildConfig,
    build_paired_outcome_contrastive_dataset,
    write_paired_outcome_contrastive_summary,
)
from weiss_rl.experiments.paired_outcome_contrastive_inspection import (
    PairedOutcomeInspectionConfig,
    inspect_paired_outcome_sources,
)
from weiss_rl.experiments.paired_outcome_contrastive_sources import sources_from_paired_flip_summary
from weiss_rl.replay.trajectory_bc import ReplayTrajectoryDataset


@dataclass(frozen=True, slots=True)
class PairedOutcomeContrastiveRunResult:
    output_dataset: Path
    summary_path: Path
    dataset: ReplayTrajectoryDataset
    summary: dict[str, Any]


def run_paired_outcome_contrastive_dataset(args: argparse.Namespace) -> PairedOutcomeContrastiveRunResult:
    sources = sources_from_paired_flip_summary(
        args.source_summary_json,
        source_role=args.source_role,
        output_dir=args.output_run_dir / "sources",
        include_source_labels=tuple(args.include_source_label),
    )
    contrastive_sources, inspection_summary = inspect_paired_outcome_sources(
        PairedOutcomeInspectionConfig(
            sources=sources,
            stack_config=args.stack_config,
            run_dir=args.run_dir,
            snapshot_registry_json=args.snapshot_registry_json,
            policy_a=str(args.policy_a),
            policy_b=str(args.policy_b),
            top_k=int(args.top_k),
            top_actions=int(args.top_actions),
            accepted_snapshot_config_hashes=tuple(args.accept_snapshot_config_hash or ()),
            max_bundles_per_source=args.max_bundles_per_source,
            resume=not bool(args.no_resume),
        )
    )
    dataset, dataset_summary = build_paired_outcome_contrastive_dataset(
        PairedOutcomeContrastiveBuildConfig(
            sources=contrastive_sources,
            output_dataset=args.output,
            min_total_variation=float(args.min_total_variation),
            max_rows_per_bundle=args.max_rows_per_bundle,
            max_rows=args.max_rows,
            positive_action_source="actions",
            negative_action_source="teacher_action",
        )
    )
    summary = paired_outcome_contrastive_summary(
        args=args,
        inspection_summary=inspection_summary,
        dataset_summary=dataset_summary,
    )
    summary_path = args.summary_json or args.output.with_suffix(".summary.json")
    write_paired_outcome_contrastive_summary(summary_path, summary)
    return PairedOutcomeContrastiveRunResult(
        output_dataset=args.output,
        summary_path=summary_path,
        dataset=dataset,
        summary=summary,
    )


def paired_outcome_contrastive_summary(
    *,
    args: argparse.Namespace,
    inspection_summary: dict[str, Any],
    dataset_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "kind": "paired_outcome_contrastive_dataset_cli_v1",
        "source_summary_json": args.source_summary_json.as_posix(),
        "source_role": str(args.source_role),
        "stack_config": args.stack_config.as_posix(),
        "run_dir": args.run_dir.as_posix(),
        "snapshot_registry_json": None
        if args.snapshot_registry_json is None
        else args.snapshot_registry_json.as_posix(),
        "policy_a": str(args.policy_a),
        "policy_b": str(args.policy_b),
        "output": args.output.as_posix(),
        "inspection_summary": inspection_summary,
        "dataset_summary": dataset_summary,
    }


__all__ = [
    "PairedOutcomeContrastiveRunResult",
    "paired_outcome_contrastive_summary",
    "run_paired_outcome_contrastive_dataset",
]
