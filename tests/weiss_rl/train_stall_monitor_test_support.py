from __future__ import annotations

from pathlib import Path

from weiss_rl.training.train_entrypoint import TrainingPaths


def make_training_paths(tmp_path: Path) -> TrainingPaths:
    training_dir = tmp_path / "training"
    logs_dir = training_dir / "logs"
    snapshots_dir = training_dir / "snapshots"
    checkpoints_dir = training_dir / "checkpoints"
    tensorboard_dir = tmp_path / "tensorboard"
    for path in (logs_dir, snapshots_dir, checkpoints_dir, tensorboard_dir):
        path.mkdir(parents=True, exist_ok=True)
    return TrainingPaths(
        training_dir=training_dir,
        checkpoints_dir=checkpoints_dir,
        logs_dir=logs_dir,
        snapshots_dir=snapshots_dir,
        tensorboard_dir=tensorboard_dir,
        scalars_path=logs_dir / "training_metrics.jsonl",
        performance_log_path=logs_dir / "performance.jsonl",
        latest_checkpoint_path=checkpoints_dir / "latest.pt",
        best_checkpoint_path=checkpoints_dir / "best.pt",
        checkpoint_tracker_path=checkpoints_dir / "checkpoint_tracker.json",
    )
