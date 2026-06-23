from __future__ import annotations

from pathlib import Path

from weiss_rl.actors.checkpoint_metadata import (
    checkpoint_metadata_path_for_update,
    checkpoint_update_from_path,
    latest_checkpoint_metadata_update,
    observe_new_checkpoint_metadata,
)


def test_checkpoint_update_from_path_accepts_canonical_and_legacy_names() -> None:
    assert checkpoint_update_from_path(Path("checkpoint_metadata_250.json")) == 250
    assert checkpoint_update_from_path(Path("checkpoint_50.pt")) == 50
    assert checkpoint_update_from_path(Path("checkpoint_latest.pt")) is None


def test_latest_checkpoint_metadata_update_ignores_non_numeric_checkpoint_names(tmp_path: Path) -> None:
    (tmp_path / "checkpoint_metadata_100.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "checkpoint_latest.pt").write_text("ignored\n", encoding="utf-8")
    (tmp_path / "checkpoint_75.pt").write_text("legacy\n", encoding="utf-8")

    assert latest_checkpoint_metadata_update(tmp_path) == 100


def test_checkpoint_metadata_path_prefers_metadata_json_over_legacy_checkpoint(tmp_path: Path) -> None:
    metadata_path = tmp_path / "checkpoint_metadata_5.json"
    legacy_path = tmp_path / "checkpoint_5.pt"
    metadata_path.write_text("{}\n", encoding="utf-8")
    legacy_path.write_text("legacy\n", encoding="utf-8")

    assert checkpoint_metadata_path_for_update(tmp_path, 5) == metadata_path


def test_observe_new_checkpoint_metadata_returns_only_newer_marker(tmp_path: Path) -> None:
    marker = tmp_path / "checkpoint_metadata_12.json"
    marker.write_text("{}\n", encoding="utf-8")

    observation = observe_new_checkpoint_metadata(tmp_path, last_observed_update=10)

    assert observation is not None
    assert observation.update_count == 12
    assert observation.path == marker
    assert observe_new_checkpoint_metadata(tmp_path, last_observed_update=12) is None
