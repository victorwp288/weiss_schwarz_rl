from __future__ import annotations

from pathlib import Path

from weiss_rl.actors.actor_worker import ActorWorker

from .actor_worker_test_support import ACTION_SPACE


def test_actor_worker_reports_checkpoint_metadata_lag_in_update_units(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "checkpoint_metadata_100.json").write_text("{}\n", encoding="utf-8")
    (checkpoint_dir / "checkpoint_metadata_250.json").write_text("{}\n", encoding="utf-8")

    worker = ActorWorker(
        actor_id=7,
        unroll_length=1,
        num_envs=1,
        action_space=ACTION_SPACE,
        checkpoint_dir=checkpoint_dir,
        reload_interval_updates=2,
    )

    assert worker.reload_interval_updates == 2

    first = worker.poll_checkpoint_metadata()
    assert first == {"observed_checkpoint_update": 0, "checkpoint_metadata_lag_updates": 250}

    second = worker.poll_checkpoint_metadata()
    assert second == {"observed_checkpoint_update": 250, "checkpoint_metadata_lag_updates": 0}

    (checkpoint_dir / "checkpoint_metadata_400.json").write_text("{}\n", encoding="utf-8")
    third = worker.poll_checkpoint_metadata()
    assert third == {"observed_checkpoint_update": 250, "checkpoint_metadata_lag_updates": 150}

    fourth = worker.poll_checkpoint_metadata()
    assert fourth == {"observed_checkpoint_update": 400, "checkpoint_metadata_lag_updates": 0}


def test_actor_worker_supports_legacy_checkpoint_filenames_for_metadata_tracking(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "checkpoint_latest.pt").write_text("bad\n", encoding="utf-8")
    (checkpoint_dir / "checkpoint_50.pt").write_text("stub\n", encoding="utf-8")

    worker = ActorWorker(
        actor_id=1,
        unroll_length=1,
        num_envs=1,
        action_space=ACTION_SPACE,
        checkpoint_dir=checkpoint_dir,
        reload_interval_updates=1,
    )

    result = worker.poll_checkpoint_metadata()
    assert result == {"observed_checkpoint_update": 50, "checkpoint_metadata_lag_updates": 0}
    assert worker.observed_checkpoint_update == 50
    assert worker.checkpoint_metadata_lag_updates == 0
