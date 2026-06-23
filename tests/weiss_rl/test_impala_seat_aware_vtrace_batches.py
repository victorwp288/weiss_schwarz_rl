from __future__ import annotations

from typing import cast

import numpy as np
import pytest
import torch
from weiss_rl.learners.impala import ImpalaLearner
from weiss_rl.model import PolicyValueModel

from .vtrace_test_support import synthetic_vtrace_training_batch, tiny_model_config


def test_impala_learner_forward_time_major_matches_manual_seat_aware_rollout() -> None:
    torch.manual_seed(0)

    model = PolicyValueModel(observation_dim=4, action_dim=4, config=tiny_model_config())
    learner = ImpalaLearner(model=model)
    batch = synthetic_vtrace_training_batch(
        learner,
        seat_field_name="to_play_seat",
        initial_hidden_state=np.linspace(-0.4, 0.4, num=2 * 2 * 8, dtype=np.float32).reshape(2, 2, 8),
    )

    obs = torch.from_numpy(cast(np.ndarray, batch["obs"]))
    to_play_seat = torch.from_numpy(cast(np.ndarray, batch["to_play_seat"]))
    initial_hidden_state = torch.from_numpy(cast(np.ndarray, batch["initial_hidden_state"]))

    with torch.no_grad():
        learner_logits, learner_values = learner._forward_time_major(
            obs,
            initial_hidden_state=initial_hidden_state,
            to_play_seat=to_play_seat,
        )

        manual_hidden_state = initial_hidden_state
        manual_logits_steps: list[torch.Tensor] = []
        manual_value_steps: list[torch.Tensor] = []
        for step_obs, step_seat in zip(obs.unbind(dim=0), to_play_seat.unbind(dim=0), strict=True):
            step_logits, step_value, manual_hidden_state = model.forward_seat_aware(
                step_obs,
                step_seat,
                manual_hidden_state,
            )
            manual_logits_steps.append(step_logits)
            manual_value_steps.append(step_value)

    torch.testing.assert_close(learner_logits, torch.stack(manual_logits_steps, dim=0))
    torch.testing.assert_close(learner_values, torch.stack(manual_value_steps, dim=0))


@pytest.mark.parametrize("seat_field_name", ["to_play_seat", "actor"])
def test_impala_learner_update_reduces_loss_on_seat_aware_batches(seat_field_name: str) -> None:
    torch.manual_seed(0)

    model = PolicyValueModel(observation_dim=4, action_dim=4, config=tiny_model_config())
    learner = ImpalaLearner(
        model=model,
        learning_rate=0.05,
        value_loss_coef=0.5,
        entropy_coef=0.0,
        grad_norm_clip=10.0,
    )
    batch = synthetic_vtrace_training_batch(
        learner,
        seat_field_name=seat_field_name,
        initial_hidden_state=np.linspace(-0.3, 0.3, num=2 * 2 * 8, dtype=np.float32).reshape(2, 2, 8),
    )

    before_loss, before_metrics = learner._loss_and_metrics(batch)
    update_metrics = learner.update(batch)
    after_loss, after_metrics = learner._loss_and_metrics(batch)

    assert update_metrics["loss"] == pytest.approx(float(before_loss.detach()))
    assert update_metrics["policy_loss"] == pytest.approx(before_metrics["policy_loss"])
    assert update_metrics["value_loss"] == pytest.approx(before_metrics["value_loss"])
    assert update_metrics["entropy"] > 0.0
    assert update_metrics["grad_norm"] > 0.0
    assert after_metrics["loss"] < before_metrics["loss"]
    assert float(after_loss.detach()) == pytest.approx(after_metrics["loss"])


@pytest.mark.parametrize(
    ("seat_field_name", "valid_initial_hidden_state", "invalid_initial_hidden_state", "message"),
    [
        (
            None,
            None,
            np.zeros((2, 2, 8), dtype=np.float32),
            r"initial_hidden_state must be 2D \(batch, hidden_size\) when to_play_seat/actor is absent",
        ),
        (
            "to_play_seat",
            np.zeros((2, 2, 8), dtype=np.float32),
            np.zeros((2, 8), dtype=np.float32),
            r"initial_hidden_state must be 3D \(batch, seat, hidden_size\) when to_play_seat/actor is present",
        ),
    ],
)
def test_impala_learner_rejects_hidden_state_shape_mismatch_by_mode(
    seat_field_name: str | None,
    valid_initial_hidden_state: np.ndarray | None,
    invalid_initial_hidden_state: np.ndarray,
    message: str,
) -> None:
    model = PolicyValueModel(observation_dim=4, action_dim=4, config=tiny_model_config())
    learner = ImpalaLearner(model=model)
    batch = synthetic_vtrace_training_batch(
        learner,
        seat_field_name=seat_field_name,
        initial_hidden_state=valid_initial_hidden_state,
    )
    batch["initial_hidden_state"] = invalid_initial_hidden_state

    with pytest.raises(ValueError, match=message):
        learner._loss_and_metrics(batch)


@pytest.mark.parametrize("seat_field_name", ["to_play_seat", "actor"])
def test_impala_learner_rejects_float_valued_seat_fields(seat_field_name: str) -> None:
    model = PolicyValueModel(observation_dim=4, action_dim=4, config=tiny_model_config())
    learner = ImpalaLearner(model=model)
    batch = synthetic_vtrace_training_batch(
        learner,
        seat_field_name=seat_field_name,
        initial_hidden_state=np.zeros((2, 2, 8), dtype=np.float32),
    )
    batch[seat_field_name] = cast(np.ndarray, batch[seat_field_name]).astype(np.float32)

    with pytest.raises(ValueError, match=rf"{seat_field_name} must be integer-valued"):
        learner._loss_and_metrics(batch)


def test_impala_learner_rejects_mismatched_actor_alias() -> None:
    model = PolicyValueModel(observation_dim=4, action_dim=4, config=tiny_model_config())
    learner = ImpalaLearner(model=model)
    batch = synthetic_vtrace_training_batch(
        learner,
        seat_field_name="to_play_seat",
        initial_hidden_state=np.zeros((2, 2, 8), dtype=np.float32),
    )
    batch["actor"] = 1 - cast(np.ndarray, batch["to_play_seat"])

    with pytest.raises(ValueError, match="actor must match to_play_seat"):
        learner._loss_and_metrics(batch)
