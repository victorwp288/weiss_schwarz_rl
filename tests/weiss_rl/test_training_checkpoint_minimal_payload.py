from __future__ import annotations

import pytest
import torch
from weiss_rl.training.checkpoints import (
    initialize_model_from_checkpoint,
    restore_minimal_train_checkpoint,
    write_minimal_train_checkpoint,
)

from .training_checkpoint_test_support import _Learner


def test_write_minimal_train_checkpoint_payload_shape_and_save(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    learner = _Learner()
    learner.init_schedule_offset_updates = 90

    payload = write_minimal_train_checkpoint(
        checkpoint_path=checkpoint_path,
        learner=learner,
        device=torch.device("cpu"),
        config_hash256="abc",
        spec_hash256="def",
        algorithm="impala_vtrace_gru",
        recurrent_core="gru",
        guidance_payload={"public_heuristic_logit_bias_scale": 0.25},
    )

    assert checkpoint_path.is_file()
    assert payload["format"] == "minimal_train_checkpoint_v1"
    assert payload["update_count"] == 3
    assert payload["policy_version"] == 7
    assert payload["device"] == "cpu"
    assert payload["config_hash256"] == "abc"
    assert payload["spec_hash256"] == "def"
    assert payload["algorithm"] == "impala_vtrace_gru"
    assert payload["recurrent_core"] == "gru"
    assert payload["total_samples_processed"] == 42
    assert payload["init_schedule_offset_updates"] == 90
    assert payload["policy_anchor_model_state_dict"] is None
    assert payload["public_heuristic_logit_bias_scale"] == pytest.approx(0.25)
    assert payload["optimizer_state_dict"] == {"lr": 0.01}
    assert payload["grad_scaler_state_dict"] == {"scale": 2.0}
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert loaded["update_count"] == payload["update_count"]
    assert loaded["model_state_dict"]["weight"].tolist() == [1.0]


def test_minimal_train_checkpoint_round_trips_policy_anchor_state(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    writer = _Learner()
    writer.anchor_state = {"anchor_weight": torch.tensor([2.0])}

    payload = write_minimal_train_checkpoint(
        checkpoint_path=checkpoint_path,
        learner=writer,
        device=torch.device("cpu"),
        config_hash256="abc",
        spec_hash256="def",
        algorithm="impala_vtrace_gru",
    )
    restored = _Learner()
    restore_minimal_train_checkpoint(
        checkpoint_path=checkpoint_path,
        learner=restored,
        device=torch.device("cpu"),
        expected_config_hash="abc",
        expected_spec_hash256="def",
        algorithm="impala_vtrace_gru",
        restore_model_guidance=lambda _model, _payload: None,
    )

    assert payload["policy_anchor_model_state_dict"]["anchor_weight"].tolist() == [2.0]
    assert restored.loaded_anchor_state is not None
    assert restored.loaded_anchor_state["anchor_weight"].tolist() == [2.0]


def test_write_minimal_train_checkpoint_requires_model(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="Cannot write a checkpoint without a learner model"):
        write_minimal_train_checkpoint(
            checkpoint_path=tmp_path / "checkpoint.pt",
            learner=_Learner(model=None),
            device=torch.device("cpu"),
            config_hash256="abc",
        )


def test_restore_minimal_train_checkpoint_restores_state_and_can_preserve_counters(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    writer = _Learner()
    writer.init_schedule_offset_updates = 90
    write_minimal_train_checkpoint(
        checkpoint_path=checkpoint_path,
        learner=writer,
        device=torch.device("cpu"),
        config_hash256="abc",
        spec_hash256="def",
        algorithm="impala_vtrace_gru",
        recurrent_core="gru",
        guidance_payload={"public_heuristic_logit_bias_scale": 0.25},
    )
    restored = _Learner()
    restored.update_count = 99
    restored.policy_version = 88
    restored.total_samples_processed = 77
    guidance_calls: list[tuple[object, float]] = []

    resume = restore_minimal_train_checkpoint(
        checkpoint_path=checkpoint_path,
        learner=restored,
        device=torch.device("cpu"),
        expected_config_hash="abc",
        expected_spec_hash256="def",
        algorithm="impala_vtrace_gru",
        restore_model_guidance=lambda model, payload: guidance_calls.append(
            (model, payload["public_heuristic_logit_bias_scale"])
        ),
        restore_counters=False,
    )

    assert resume.update_count == 99
    assert resume.policy_version == 88
    assert resume.total_samples_processed == 77
    assert restored.model.loaded_state is not None
    assert restored.optimizer.loaded_state == {"lr": 0.01}
    assert restored.init_schedule_offset_updates == 90
    assert resume.init_schedule_offset_updates == 90
    assert guidance_calls[0][0] is restored.model
    assert guidance_calls[0][1] == pytest.approx(0.25)


def test_initialize_model_from_checkpoint_loads_weights_without_optimizer_or_counters(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    writer = _Learner()
    writer.init_schedule_offset_updates = 90
    write_minimal_train_checkpoint(
        checkpoint_path=checkpoint_path,
        learner=writer,
        device=torch.device("cpu"),
        config_hash256="source-config",
        spec_hash256="def",
        algorithm="impala_vtrace_gru",
        recurrent_core="gru",
        guidance_payload={"public_heuristic_logit_bias_scale": 0.5},
    )
    initialized = _Learner()
    initialized.update_count = 99
    initialized.policy_version = 88
    initialized.total_samples_processed = 77
    guidance_calls: list[tuple[object, float]] = []

    source = initialize_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        learner=initialized,
        device=torch.device("cpu"),
        expected_spec_hash256="def",
        algorithm="impala_vtrace_gru",
        restore_model_guidance=lambda model, payload: guidance_calls.append(
            (model, payload["public_heuristic_logit_bias_scale"])
        ),
    )

    assert source.update_count == 3
    assert source.policy_version == 7
    assert source.total_samples_processed == 42
    assert source.init_schedule_offset_updates == 90
    assert initialized.update_count == 99
    assert initialized.policy_version == 88
    assert initialized.total_samples_processed == 77
    assert initialized.model.loaded_state is not None
    assert initialized.optimizer.loaded_state is None
    assert initialized._grad_scaler.loaded_state is None
    assert initialized.loaded_anchor_state is None
    assert initialized.reset_anchor_calls == 1
    assert guidance_calls[0][0] is initialized.model
    assert guidance_calls[0][1] == pytest.approx(0.5)
