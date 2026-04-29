from __future__ import annotations

import json
from pathlib import Path

import pytest

from weiss_rl.config import (
    apply_stack_overrides,
    canonical_config_dict,
    compute_config_hash256,
    load_stack_config,
    parse_override_tokens,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_parse_override_tokens_reads_grouped_json_values() -> None:
    overrides = parse_override_tokens(
        [
            "training.optimizer.learning_rate=0.0001",
            'model.recurrent_core="none"',
            "league.enabled=false",
        ]
    )

    assert overrides == {
        "training.optimizer.learning_rate": 0.0001,
        "model.recurrent_core": "none",
        "league.enabled": False,
    }


def test_apply_stack_overrides_updates_grouped_fields() -> None:
    stack = load_stack_config(_repo_root() / "configs" / "thesis_locked.yaml")

    updated = apply_stack_overrides(
        stack,
        {
            "training.optimizer.learning_rate": 0.0001,
            "model.recurrent_core": "none",
            "curriculum.stall_monitor.enabled": True,
        },
    )

    assert updated.config.training is not None
    assert updated.config.training.learning_rate == pytest.approx(0.0001)
    assert updated.config.model is not None
    assert updated.config.model.recurrent_core == "none"
    assert updated.config.curriculum is not None
    assert updated.config.curriculum.stall_monitor.enabled is True

    assert stack.config.training is not None
    assert stack.config.training.learning_rate == pytest.approx(0.0002)


def test_apply_stack_overrides_rejects_unknown_fields() -> None:
    stack = load_stack_config(_repo_root() / "configs" / "thesis_locked.yaml")

    with pytest.raises(ValueError, match="unknown field"):
        apply_stack_overrides(stack, {"training.optimizer.mystery_knob": 1})


def test_apply_stack_overrides_updates_preserved_canonical_payload(tmp_path: Path) -> None:
    repo_root = _repo_root()
    stack = load_stack_config(repo_root / "configs" / "local.yaml")
    (tmp_path / "configs").mkdir()
    artifact_path = tmp_path / "runs" / "example_run" / "config_canonical.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(canonical_config_dict(stack)), encoding="utf-8")

    loaded = load_stack_config(artifact_path)
    updated = apply_stack_overrides(loaded, {"rewards.discount.gamma": 0.5})

    assert canonical_config_dict(updated)["config"]["rewards"]["discount"]["gamma"] == pytest.approx(0.5)
    assert compute_config_hash256(updated) != compute_config_hash256(loaded)
