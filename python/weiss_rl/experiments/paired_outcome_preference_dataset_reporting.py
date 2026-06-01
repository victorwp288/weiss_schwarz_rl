"""Console reporting for paired-outcome preference dataset generation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from weiss_rl.replay.trajectory_bc import ReplayTrajectoryDataset


def paired_outcome_preference_dataset_output_payload(
    *,
    output_dataset: Path,
    dataset: ReplayTrajectoryDataset,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "output": output_dataset.as_posix(),
        "pair_count": summary["pair_count"],
        "episodes": dataset.episode_count,
        "train_rows": dataset.metadata.get("train_rows"),
    }


def paired_outcome_preference_dataset_output_line(
    *,
    output_dataset: Path,
    dataset: ReplayTrajectoryDataset,
    summary: Mapping[str, Any],
) -> str:
    return json.dumps(
        paired_outcome_preference_dataset_output_payload(
            output_dataset=output_dataset,
            dataset=dataset,
            summary=summary,
        ),
        sort_keys=True,
    )


__all__ = [
    "paired_outcome_preference_dataset_output_line",
    "paired_outcome_preference_dataset_output_payload",
]
