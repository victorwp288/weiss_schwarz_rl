from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CanonicalEvalRunState:
    layout: Any
    tensorboard_logger: Any
    manifest: dict[str, Any]
    run_id256: str
    evaluation: Any
    study_config: Any | None


@dataclass(frozen=True)
class CanonicalEvalRuntimeState:
    policy_ids: list[str]
    selection_details: dict[str, Any]
    snapshot_registry_path: Path | None
    dev_eval_summaries_path: Path | None
    runner: Any
    paired_seeds: list[Any]
    paired_seed_limit: int | None
    stage1_paired_seeds: int
    max_paired_seeds: int
    seed_file_path: Path
    recommended_focal_policy_id: str | None
