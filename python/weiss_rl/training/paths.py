from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.artifacts.manifest import RunArtifacts
from weiss_rl.training.checkpoints import (
    BEST_CHECKPOINT_FILENAME,
    CHECKPOINT_TRACKER_FILENAME,
    LATEST_CHECKPOINT_FILENAME,
)


@dataclass(frozen=True, slots=True)
class TrainingPaths:
    training_dir: Path
    checkpoints_dir: Path
    logs_dir: Path
    snapshots_dir: Path
    tensorboard_dir: Path
    scalars_path: Path
    performance_log_path: Path
    latest_checkpoint_path: Path
    best_checkpoint_path: Path
    checkpoint_tracker_path: Path


def training_paths(run_dir: Path) -> TrainingPaths:
    layout = ArtifactLayout.from_run_dir(run_dir)
    layout.ensure_directories()
    training_dir = layout.training_dir
    checkpoints_dir = layout.training_checkpoints_dir
    logs_dir = layout.training_logs_dir
    snapshots_dir = layout.training_snapshots_dir
    return TrainingPaths(
        training_dir=training_dir,
        checkpoints_dir=checkpoints_dir,
        logs_dir=logs_dir,
        snapshots_dir=snapshots_dir,
        tensorboard_dir=layout.tensorboard_dir,
        scalars_path=logs_dir / "scalars.jsonl",
        performance_log_path=layout.performance_log_path,
        latest_checkpoint_path=checkpoints_dir / LATEST_CHECKPOINT_FILENAME,
        best_checkpoint_path=checkpoints_dir / BEST_CHECKPOINT_FILENAME,
        checkpoint_tracker_path=checkpoints_dir / CHECKPOINT_TRACKER_FILENAME,
    )


def run_artifacts_from_existing_run_dir(run_dir: Path) -> RunArtifacts:
    resolved_run_dir = Path(run_dir).resolve()
    layout = ArtifactLayout.from_run_dir(resolved_run_dir)
    layout.ensure_directories()
    return RunArtifacts(
        run_dir=resolved_run_dir,
        run_dir_name=resolved_run_dir.name,
        layout=layout,
        manifest_path=layout.manifest_path,
        spec_bundle_path=layout.spec_bundle_path,
        spec_hash_path=layout.spec_hash_path,
        config_hash_path=layout.config_hash_path,
        config_json_path=layout.config_json_path,
        environment_path=layout.environment_path,
        run_summary_path=layout.run_summary_path,
        determinism_report_path=layout.determinism_report_path,
        paper_readiness_summary_path=layout.paper_readiness_summary_path,
        performance_log_path=layout.performance_log_path,
    )
