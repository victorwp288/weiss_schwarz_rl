from __future__ import annotations

from pathlib import Path

import pytest

from weiss_rl.training.checkpoint_lifecycle_effects import (
    apply_finalize_to_best_effects,
    apply_rollback_to_best_effects,
)


class _Paths:
    def __init__(self, tmp_path: Path) -> None:
        self.best_checkpoint_path = tmp_path / "training" / "checkpoints" / "best.pt"
        self.snapshots_dir = tmp_path / "training" / "snapshots"


class _Runtime:
    def __init__(self, events: list[tuple[str, object]]) -> None:
        self.events = events

    def maybe_publish_snapshot(self, **kwargs: object) -> dict[str, float]:
        self.events.append(("publish", kwargs))
        return {"snapshot_publish_latency_ms": 1.25, "snapshot_apply_latency_ms": 2.5}

    def reset_outcome_tracker(self) -> None:
        self.events.append(("reset", None))

    def refresh_opponent_pool(self) -> None:
        self.events.append(("refresh", None))


def test_rollback_to_best_effects_restore_demote_publish_then_refresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, object]] = []
    paths = _Paths(tmp_path)
    runtime = _Runtime(events)
    learner_model = object()

    def restore_checkpoint(path: Path, *, restore_counters: bool) -> None:
        events.append(("restore", (path, restore_counters)))

    def demote(_paths: _Paths, *, update_count: int) -> list[str]:
        events.append(("demote", update_count))
        return ["policy_000220"]

    monkeypatch.setattr("weiss_rl.training.checkpoint_lifecycle_effects.demote_registry_champions_newer_than", demote)

    effects = apply_rollback_to_best_effects(
        training_paths=paths,
        runtime=runtime,
        learner_model=learner_model,
        learner_update_count=220,
        best_update_count=160,
        restore_checkpoint=restore_checkpoint,
    )

    assert effects.best_checkpoint_path == paths.best_checkpoint_path
    assert effects.demoted_champions == ["policy_000220"]
    assert effects.publish_metrics == {
        "snapshot_publish_latency_ms": 1.25,
        "snapshot_apply_latency_ms": 2.5,
    }
    assert events == [
        ("restore", (paths.best_checkpoint_path, False)),
        ("demote", 160),
        ("publish", {"learner_model": learner_model, "learner_update_count": 220, "force": True}),
        ("reset", None),
        ("refresh", None),
    ]


def test_finalize_to_best_effects_restore_demote_then_refresh_without_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, object]] = []
    paths = _Paths(tmp_path)
    runtime = _Runtime(events)

    def restore_checkpoint(path: Path, *, restore_counters: bool) -> None:
        events.append(("restore", (path, restore_counters)))

    def demote(_paths: _Paths, *, update_count: int) -> list[str]:
        events.append(("demote", update_count))
        return []

    monkeypatch.setattr("weiss_rl.training.checkpoint_lifecycle_effects.demote_registry_champions_newer_than", demote)

    effects = apply_finalize_to_best_effects(
        training_paths=paths,
        runtime=runtime,
        best_update_count=160,
        restore_checkpoint=restore_checkpoint,
    )

    assert effects.best_checkpoint_path == paths.best_checkpoint_path
    assert effects.demoted_champions == []
    assert effects.publish_metrics == {}
    assert events == [
        ("restore", (paths.best_checkpoint_path, False)),
        ("demote", 160),
        ("reset", None),
        ("refresh", None),
    ]
