from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from weiss_rl.config import load_stack_config

from ._config_paths import canonical_stack_config_path
from .snapshot_registry_test_support import (
    _load_train_script_module,
    _make_bootstrap_learner,
    _write_b1_baseline_run_fixture,
)


def test_ensure_noleague_baseline_anchor_rejects_unlocked_selected_candidate(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(canonical_stack_config_path())
    selected_run_dir = _write_b1_baseline_run_fixture(
        tmp_path,
        update=15,
        policy_id="selected_candidate",
        experiment_role="guided_league_bootstrap",
    )

    with pytest.raises(FileNotFoundError, match="canonical B1 no-league baseline snapshot"):
        train_script._ensure_noleague_baseline_anchor(
            stack=stack,
            training_paths=train_script._training_paths(tmp_path / "consumer_unlocked_selected_run"),
            run_dir=tmp_path / "consumer_unlocked_selected_run",
            learner=_make_bootstrap_learner(stack),
            device=torch.device("cpu"),
            config_hash256="ab" * 32,
            baseline_run_dir=selected_run_dir,
        )


def test_ensure_noleague_baseline_anchor_rejects_non_b1_imported_run(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(canonical_stack_config_path())
    baseline_run_dir = _write_b1_baseline_run_fixture(tmp_path, experiment_role="main")

    with pytest.raises(RuntimeError, match="must come from a dedicated baseline_noleague run"):
        train_script._ensure_noleague_baseline_anchor(
            stack=stack,
            training_paths=train_script._training_paths(tmp_path / "consumer_run"),
            run_dir=tmp_path / "consumer_run",
            learner=_make_bootstrap_learner(stack),
            device=torch.device("cpu"),
            config_hash256="11" * 32,
            spec_hash256="cd" * 32,
            baseline_run_dir=baseline_run_dir,
        )


def test_ensure_noleague_baseline_anchor_rejects_legacy_non_b1_imported_run(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(canonical_stack_config_path())
    baseline_run_dir = _write_b1_baseline_run_fixture(tmp_path, legacy_training_mode="main")

    with pytest.raises(RuntimeError, match=r"training_family_a\.mode='main'"):
        train_script._ensure_noleague_baseline_anchor(
            stack=stack,
            training_paths=train_script._training_paths(tmp_path / "consumer_run_legacy"),
            run_dir=tmp_path / "consumer_run_legacy",
            learner=_make_bootstrap_learner(stack),
            device=torch.device("cpu"),
            config_hash256="11" * 32,
            spec_hash256="cd" * 32,
            baseline_run_dir=baseline_run_dir,
        )


def test_ensure_noleague_baseline_anchor_rejects_imported_environment_mismatch(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(canonical_stack_config_path())
    baseline_run_dir = _write_b1_baseline_run_fixture(tmp_path)
    manifest_path = baseline_run_dir / "manifest.json"
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_sections = manifest_payload["config_canonical"]["config"]
    config_sections["environment"] = {
        **dict(config_sections["environment"]),
        "best_of": 99,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="config does not match the current run for section='environment'"):
        train_script._ensure_noleague_baseline_anchor(
            stack=stack,
            training_paths=train_script._training_paths(tmp_path / "consumer_run_env_mismatch"),
            run_dir=tmp_path / "consumer_run_env_mismatch",
            learner=_make_bootstrap_learner(stack),
            device=torch.device("cpu"),
            config_hash256="11" * 32,
            spec_hash256="cd" * 32,
            baseline_run_dir=baseline_run_dir,
        )
