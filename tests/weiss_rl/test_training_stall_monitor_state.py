from __future__ import annotations

from pathlib import Path

from weiss_rl.config import load_stack_config
from weiss_rl.training.train_entrypoint import (
    _update_stall_monitor,
)

from tests.weiss_rl._config_paths import repo_root
from tests.weiss_rl.train_stall_monitor_test_support import make_training_paths


def test_update_stall_monitor_marks_run_after_consecutive_truncating_evals(tmp_path: Path) -> None:
    stack = load_stack_config(repo_root() / "configs" / "presets" / "typed_local.yaml")
    training_paths = make_training_paths(tmp_path)
    payload = {
        "anchors": {
            "B0 RandomLegal": {"summary": {"games": 10, "truncations": 4}},
            "B1 NoLeague baseline": {"summary": {"games": 10, "truncations": 3}},
        }
    }

    first = _update_stall_monitor(
        stack=stack,
        training_paths=training_paths,
        update_count=100,
        summary_payload=payload,
    )
    second = _update_stall_monitor(
        stack=stack,
        training_paths=training_paths,
        update_count=200,
        summary_payload=payload,
    )

    assert first is not None
    assert second is not None
    assert first["stall_risk"] is False
    assert second["stall_risk"] is True
    assert second["worst_anchor"] == "B0 RandomLegal"


def test_update_stall_monitor_includes_optional_b2_anchor(tmp_path: Path) -> None:
    stack = load_stack_config(repo_root() / "configs" / "presets" / "typed_local.yaml")
    training_paths = make_training_paths(tmp_path)
    payload = {
        "anchors": {
            "B0 RandomLegal": {"summary": {"games": 10, "truncations": 1}},
            "B1 NoLeague baseline": {"summary": {"games": 10, "truncations": 2}},
            "B2 HeuristicPublic": {"summary": {"games": 10, "truncations": 9}},
        }
    }

    first = _update_stall_monitor(
        stack=stack,
        training_paths=training_paths,
        update_count=100,
        summary_payload=payload,
    )
    second = _update_stall_monitor(
        stack=stack,
        training_paths=training_paths,
        update_count=200,
        summary_payload=payload,
    )

    assert first is not None
    assert second is not None
    assert second["stall_risk"] is True
    assert second["worst_anchor"] == "B2 HeuristicPublic"
