from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch
from weiss_rl.config import load_stack_config
from weiss_rl.learners.impala import ImpalaLearner

from ._config_paths import canonical_stack_config_path
from .snapshot_registry_test_support import (
    _load_train_script_module,
    _make_policy_value_model,
)


def test_write_checkpoint_payload_shape_is_stable(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(canonical_stack_config_path())
    training_paths = train_script._training_paths(tmp_path / "run")
    learner = ImpalaLearner(
        model=_make_policy_value_model(stack),
        checkpoint_dir=training_paths.checkpoints_dir,
        logs_dir=training_paths.logs_dir,
        pass_action_id=0,
    )
    learner._optimizer_for_step()
    learner.update_count = 9
    learner.policy_version = 4
    learner.total_samples_processed = 288
    checkpoint_path = training_paths.checkpoints_dir / "checkpoint_9.pt"

    payload = train_script._write_checkpoint(
        checkpoint_path=checkpoint_path,
        learner=learner,
        stack=stack,
        device=torch.device("cpu"),
        spec_hash256="ab" * 32,
        algorithm="impala_vtrace_gru",
    )

    expected_keys = {
        "algorithm",
        "config_hash256",
        "device",
        "format",
        "grad_scaler_state_dict",
        "init_schedule_offset_updates",
        "model_state_dict",
        "optimizer_state_dict",
        "policy_anchor_model_state_dict",
        "policy_version",
        "public_heuristic_actor_logit_bias_scale",
        "public_heuristic_logit_bias_scale",
        "recurrent_core",
        "spec_hash256",
        "total_samples_processed",
        "update_count",
    }
    assert set(payload) == expected_keys
    assert payload["format"] == "minimal_train_checkpoint_v1"
    assert payload["update_count"] == 9
    assert payload["policy_version"] == 4
    assert payload["total_samples_processed"] == 288
    assert payload["device"] == "cpu"
    assert payload["spec_hash256"] == "ab" * 32
    assert payload["algorithm"] == "impala_vtrace_gru"
    assert isinstance(payload["model_state_dict"], dict)
    assert isinstance(payload["optimizer_state_dict"], dict)
    assert checkpoint_path.is_file()


def _write_restore_checkpoint_fixture(
    tmp_path: Path,
) -> tuple[ModuleType, Any, Path, ImpalaLearner]:
    train_script = _load_train_script_module()
    stack = load_stack_config(canonical_stack_config_path())
    training_paths = train_script._training_paths(tmp_path / "run")

    learner = ImpalaLearner(
        model=_make_policy_value_model(stack),
        checkpoint_dir=training_paths.checkpoints_dir,
        logs_dir=training_paths.logs_dir,
        pass_action_id=0,
    )
    learner._optimizer_for_step()
    checkpoint_path = training_paths.checkpoints_dir / "checkpoint_bad.pt"
    train_script._write_checkpoint(
        checkpoint_path=checkpoint_path,
        learner=learner,
        stack=stack,
        device=torch.device("cpu"),
        spec_hash256="ab" * 32,
        algorithm="impala_vtrace_gru",
    )
    restore_learner = ImpalaLearner(
        model=_make_policy_value_model(stack),
        checkpoint_dir=training_paths.checkpoints_dir,
        logs_dir=training_paths.logs_dir,
        pass_action_id=0,
    )
    return train_script, stack, checkpoint_path, restore_learner


@pytest.mark.parametrize(
    ("case_name", "match"),
    [
        ("non_dict_payload", "checkpoint payload must be a dict"),
        ("unsupported_format", "unsupported checkpoint format"),
        ("config_hash_mismatch", "checkpoint config hash mismatch"),
        ("spec_hash_mismatch", "checkpoint spec hash mismatch"),
        ("algorithm_mismatch", "checkpoint algorithm mismatch"),
        ("missing_model_state_dict", "checkpoint is missing a model_state_dict"),
    ],
)
def test_restore_checkpoint_rejects_invalid_payload_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case_name: str,
    match: str,
) -> None:
    train_script, stack, checkpoint_path, restore_learner = _write_restore_checkpoint_fixture(tmp_path)
    monkeypatch.delenv("WEISS_RL_ALLOW_RESUME_CONFIG_MISMATCH", raising=False)

    if case_name == "non_dict_payload":
        torch.save(["not", "a", "dict"], checkpoint_path)
    else:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert isinstance(payload, dict)
        if case_name == "unsupported_format":
            payload["format"] = "future_checkpoint_v999"
        elif case_name == "config_hash_mismatch":
            payload["config_hash256"] = "00" * 32
        elif case_name == "spec_hash_mismatch":
            payload["spec_hash256"] = "cd" * 32
        elif case_name == "algorithm_mismatch":
            payload["algorithm"] = "different_algorithm"
        elif case_name == "missing_model_state_dict":
            payload.pop("model_state_dict", None)
        else:
            raise AssertionError(f"unhandled case: {case_name}")
        torch.save(payload, checkpoint_path)

    with pytest.raises(RuntimeError, match=match):
        train_script._restore_learner_from_checkpoint(
            checkpoint_path=checkpoint_path,
            learner=restore_learner,
            stack=stack,
            device=torch.device("cpu"),
            expected_spec_hash256="ab" * 32,
            algorithm="impala_vtrace_gru",
        )


def test_restore_checkpoint_allows_config_hash_mismatch_only_with_escape_hatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_script, stack, checkpoint_path, restore_learner = _write_restore_checkpoint_fixture(tmp_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert isinstance(payload, dict)
    payload["config_hash256"] = "00" * 32
    torch.save(payload, checkpoint_path)

    monkeypatch.setenv("WEISS_RL_ALLOW_RESUME_CONFIG_MISMATCH", "1")
    resume_state = train_script._restore_learner_from_checkpoint(
        checkpoint_path=checkpoint_path,
        learner=restore_learner,
        stack=stack,
        device=torch.device("cpu"),
        expected_spec_hash256="ab" * 32,
        algorithm="impala_vtrace_gru",
    )

    assert resume_state.checkpoint_path == checkpoint_path.resolve()
