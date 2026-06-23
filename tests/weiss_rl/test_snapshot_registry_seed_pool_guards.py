from __future__ import annotations

import json
from pathlib import Path

import pytest
from weiss_rl.config import canonical_config_dict, load_stack_config

from ._config_paths import canonical_stack_config_path
from .snapshot_registry_test_support import (
    _load_train_script_module,
    _make_policy_value_model,
    _write_seed_snapshot_run_fixture,
)


def test_import_seed_snapshot_pool_rejects_environment_mismatch(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(canonical_stack_config_path())
    seed_run_dir = _write_seed_snapshot_run_fixture(tmp_path)
    manifest_path = seed_run_dir / "manifest.json"
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_sections = manifest_payload["config_canonical"]["config"]
    config_sections["environment"] = {
        **dict(config_sections["environment"]),
        "best_of": 99,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="config does not match the current run for section='environment'"):
        train_script._import_seed_snapshot_pool(
            stack=stack,
            training_paths=train_script._training_paths(tmp_path / "consumer_run_seed_env_mismatch"),
            run_dir=tmp_path / "consumer_run_seed_env_mismatch",
            seed_snapshot_run_dir=seed_run_dir,
            expected_model_state_dict=_make_policy_value_model(stack).state_dict(),
            expected_config_canonical=canonical_config_dict(stack),
            expected_spec_hash256="cd" * 32,
        )


def test_import_seed_snapshot_pool_rejects_strict_b1_baseline_role(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(canonical_stack_config_path())
    seed_run_dir = _write_seed_snapshot_run_fixture(tmp_path, experiment_role="baseline_noleague")

    with pytest.raises(RuntimeError, match="Use --b1-baseline-run-dir for the strict B1 baseline"):
        train_script._import_seed_snapshot_pool(
            stack=stack,
            training_paths=train_script._training_paths(tmp_path / "consumer_run_seed_role_mismatch"),
            run_dir=tmp_path / "consumer_run_seed_role_mismatch",
            seed_snapshot_run_dir=seed_run_dir,
            expected_model_state_dict=_make_policy_value_model(stack).state_dict(),
            expected_config_canonical=canonical_config_dict(stack),
            expected_spec_hash256="cd" * 32,
        )
