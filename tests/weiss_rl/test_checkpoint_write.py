from __future__ import annotations

import pytest
import torch
from weiss_rl.training.checkpointing.write import (
    build_minimal_train_checkpoint_payload,
    minimal_train_checkpoint_payload_from_learner,
)


class _Model:
    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"weight": torch.tensor([1.0])}


class _Optimizer:
    def state_dict(self) -> dict[str, float]:
        return {"lr": 0.01}


class _GradScaler:
    def state_dict(self) -> dict[str, float]:
        return {"scale": 2.0}


class _FullLearner:
    update_count = 3
    total_samples_processed = 42
    init_schedule_offset_updates = 9
    model = _Model()
    optimizer = _Optimizer()
    _grad_scaler = _GradScaler()

    def get_policy_version(self) -> int:
        return 7

    def policy_anchor_state_dict(self) -> dict[str, torch.Tensor]:
        return {"anchor_weight": torch.tensor([2.0])}


class _MinimalLearner:
    update_count = 5
    model = _Model()
    optimizer = None

    def get_policy_version(self) -> int:
        return 11


def test_minimal_train_checkpoint_payload_from_learner_collects_serialized_training_state() -> None:
    payload = minimal_train_checkpoint_payload_from_learner(
        learner=_FullLearner(),
        device=torch.device("cpu"),
        config_hash256="config",
        spec_hash256="spec",
        algorithm="impala_vtrace_gru",
        recurrent_core="gru",
        guidance_payload={"public_heuristic_logit_bias_scale": 0.25},
    )

    assert payload["format"] == "minimal_train_checkpoint_v1"
    assert payload["update_count"] == 3
    assert payload["policy_version"] == 7
    assert payload["device"] == "cpu"
    assert payload["config_hash256"] == "config"
    assert payload["spec_hash256"] == "spec"
    assert payload["algorithm"] == "impala_vtrace_gru"
    assert payload["recurrent_core"] == "gru"
    assert payload["total_samples_processed"] == 42
    assert payload["init_schedule_offset_updates"] == 9
    assert payload["model_state_dict"]["weight"].tolist() == [1.0]
    assert payload["policy_anchor_model_state_dict"]["anchor_weight"].tolist() == [2.0]
    assert payload["public_heuristic_logit_bias_scale"] == pytest.approx(0.25)
    assert payload["optimizer_state_dict"] == {"lr": 0.01}
    assert payload["grad_scaler_state_dict"] == {"scale": 2.0}


def test_minimal_train_checkpoint_payload_from_learner_handles_missing_optional_state() -> None:
    payload = minimal_train_checkpoint_payload_from_learner(
        learner=_MinimalLearner(),
        device=torch.device("cpu"),
        config_hash256="config",
    )

    assert payload["update_count"] == 5
    assert payload["policy_version"] == 11
    assert payload["total_samples_processed"] == 0
    assert payload["init_schedule_offset_updates"] == 0
    assert payload["policy_anchor_model_state_dict"] is None
    assert payload["optimizer_state_dict"] is None
    assert payload["grad_scaler_state_dict"] is None


def test_minimal_train_checkpoint_payload_from_learner_requires_model() -> None:
    learner = _MinimalLearner()
    learner.model = None

    with pytest.raises(RuntimeError, match="Cannot write a checkpoint without a learner model"):
        minimal_train_checkpoint_payload_from_learner(
            learner=learner,
            device=torch.device("cpu"),
            config_hash256="config",
        )


def test_build_minimal_train_checkpoint_payload_preserves_guidance_fields() -> None:
    payload = build_minimal_train_checkpoint_payload(
        update_count=1,
        policy_version=2,
        device="cpu",
        config_hash256="config",
        spec_hash256=None,
        algorithm=None,
        recurrent_core=None,
        total_samples_processed=3,
        init_schedule_offset_updates=4,
        model_state_dict={},
        policy_anchor_model_state_dict=None,
        guidance_payload={"public_heuristic_logit_bias_scale": 0.5},
        optimizer_state_dict=None,
        grad_scaler_state_dict=None,
    )

    assert payload["public_heuristic_logit_bias_scale"] == pytest.approx(0.5)
    assert payload["optimizer_state_dict"] is None
    assert payload["grad_scaler_state_dict"] is None
