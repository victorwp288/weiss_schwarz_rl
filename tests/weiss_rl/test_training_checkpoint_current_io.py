from __future__ import annotations

import json
from pathlib import Path

import pytest
from weiss_rl.training.checkpoints import (
    checkpoint_path_for_update,
    current_focal_policy_id,
    ensure_current_checkpoint,
    write_scalars_record,
)

from .training_checkpoint_test_support import (
    _Learner,
    _TrainingPaths,
)


def test_write_scalars_record_appends_stable_json_line(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("weiss_rl.training.checkpointing.storage.io.time.time", lambda: 105.25)
    scalars_path = tmp_path / "scalars.jsonl"

    record = write_scalars_record(
        scalars_path=scalars_path,
        learner=_Learner(),
        metrics={"loss": 1.5},
        start_time=100.0,
    )

    assert record["update_count"] == 3
    assert record["policy_version"] == 7
    assert record["wall_clock_seconds"] == pytest.approx(5.25)
    assert record["wall_clock_ms"] == 5250
    assert json.loads(scalars_path.read_text(encoding="utf-8")) == record


def test_ensure_current_checkpoint_reuses_existing_file_and_writes_missing_file(tmp_path) -> None:
    paths = _TrainingPaths(tmp_path / "training" / "checkpoints")
    paths.checkpoints_dir.mkdir(parents=True)
    learner = _Learner()
    existing_path = checkpoint_path_for_update(paths.checkpoints_dir, update_count=learner.update_count)
    existing_path.write_bytes(b"existing")
    write_calls: list[Path] = []

    assert current_focal_policy_id(learner=learner) == "train_u3_p7"
    assert (
        ensure_current_checkpoint(
            training_paths=paths,
            learner=learner,
            write_checkpoint=lambda path: write_calls.append(path),
        )
        == existing_path
    )
    assert write_calls == []

    learner.update_count = 4

    def _write_checkpoint(path: Path) -> None:
        write_calls.append(path)
        path.write_bytes(b"new")

    new_path = ensure_current_checkpoint(
        training_paths=paths,
        learner=learner,
        write_checkpoint=_write_checkpoint,
    )

    assert new_path == paths.checkpoints_dir / "checkpoint_4.pt"
    assert new_path.read_bytes() == b"new"
    assert write_calls == [new_path]
