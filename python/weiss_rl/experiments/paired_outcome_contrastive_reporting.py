"""Console reporting for paired-outcome contrastive dataset generation."""

from __future__ import annotations

from pathlib import Path

from weiss_rl.replay.trajectory_bc import ReplayTrajectoryDataset


def paired_outcome_contrastive_output_line(
    *,
    output_dataset: Path,
    summary_path: Path,
    dataset: ReplayTrajectoryDataset,
) -> str:
    return (
        "Paired-outcome contrastive dataset written to "
        f"{output_dataset} with {dataset.metadata['train_rows']} train rows, "
        f"{dataset.metadata['bundle_count']} bundles, and "
        f"{dataset.metadata['paired_outcome_contrastive_generation']['distinct_train_rows']} distinct pairs; "
        f"summary written to {summary_path}"
    )


__all__ = ["paired_outcome_contrastive_output_line"]
