from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import pytest
import torch

from weiss_rl.config.models import ModelConfig, ModelDropoutConfig
from weiss_rl.learners.impala_learner import ImpalaLearner, summarize_vtrace_diagnostics
from weiss_rl.learners.vtrace import VTraceTargets, compute_vtrace_targets
from weiss_rl.model import PolicyValueModel

TEST_VECTORS_PATH = Path(__file__).with_name("test_vectors") / "vtrace_v1.json"


class VTraceFixture(TypedDict):
    rho_bar: float
    c_bar: float
    rewards: list[list[float]]
    values: list[list[float]]
    discounts: list[list[float]]
    behavior_logp: list[list[float]]
    target_logp: list[list[float]]
    expected_vs: list[list[float]]
    expected_pg_advantages: list[list[float]]
    expected_rhos: list[list[float]]


def _load_fixture() -> VTraceFixture:
    payload = json.loads(TEST_VECTORS_PATH.read_text(encoding="utf-8"))
    return cast(VTraceFixture, payload)


def _array(rows: list[list[float]]) -> np.ndarray:
    return np.asarray(rows, dtype=np.float32)


def _model_config() -> ModelConfig:
    return ModelConfig(
        gru_hidden_size=8,
        encoder_mlp_width=8,
        encoder_mlp_layers=1,
        layer_norm=False,
        dropout=ModelDropoutConfig(family_a=0.0, ablation=0.0),
    )


def _synthetic_training_batch(
    learner: ImpalaLearner,
    *,
    seat_field_name: str | None = None,
    initial_hidden_state: np.ndarray | None = None,
) -> dict[str, object]:
    obs = np.asarray(
        [
            [[1.0, 0.0, 0.5, -1.0], [0.0, 1.0, -0.5, 0.5]],
            [[0.5, -1.0, 1.0, 0.0], [1.0, 0.5, 0.0, -0.5]],
            [[-0.5, 0.5, 1.5, 0.5], [1.0, -0.5, 0.5, 1.0]],
        ],
        dtype=np.float32,
    )
    actions = np.asarray(
        [
            [1, 2],
            [2, 0],
            [3, 1],
        ],
        dtype=np.int64,
    )
    legal_mask = np.asarray(
        [
            [[1, 1, 0, 0], [1, 0, 1, 0]],
            [[0, 1, 1, 0], [1, 1, 0, 0]],
            [[1, 0, 0, 1], [0, 1, 1, 1]],
        ],
        dtype=np.uint8,
    )
    to_play_seat = np.asarray(
        [
            [0, 1],
            [1, 0],
            [0, 1],
        ],
        dtype=np.int64,
    )

    forward_kwargs: dict[str, object] = {}
    if seat_field_name is not None:
        forward_kwargs[seat_field_name] = to_play_seat
    if initial_hidden_state is not None:
        forward_kwargs["initial_hidden_state"] = initial_hidden_state

    with torch.no_grad():
        _, values = learner._forward_time_major(torch.from_numpy(obs), **forward_kwargs)

    value_targets = values + torch.tensor(
        [[0.50, -0.25], [0.75, 0.50], [-0.50, 0.25]],
        dtype=values.dtype,
        device=values.device,
    )
    advantages = torch.tensor(
        [[1.00, 0.75], [0.50, 1.25], [1.50, 0.50]],
        dtype=values.dtype,
        device=values.device,
    )

    batch: dict[str, object] = {
        "obs": obs,
        "actions": actions,
        "legal_mask": legal_mask,
        "vtrace_result": VTraceTargets(
            vs=value_targets.cpu().numpy().astype(np.float32),
            pg_advantages=advantages.cpu().numpy().astype(np.float32),
            rhos=np.ones((3, 2), dtype=np.float32),
        ),
        "vtrace_rho_bar": 1.0,
        "vtrace_c_bar": 1.0,
    }
    if seat_field_name is not None:
        batch[seat_field_name] = to_play_seat
    if initial_hidden_state is not None:
        batch["initial_hidden_state"] = initial_hidden_state
    return batch


def test_compute_vtrace_targets_matches_golden_fixture() -> None:
    fixture = _load_fixture()

    result = compute_vtrace_targets(
        _array(fixture["rewards"]),
        _array(fixture["values"]),
        _array(fixture["discounts"]),
        _array(fixture["behavior_logp"]),
        _array(fixture["target_logp"]),
        rho_bar=fixture["rho_bar"],
        c_bar=fixture["c_bar"],
    )

    np.testing.assert_allclose(result.vs, _array(fixture["expected_vs"]), rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(result.pg_advantages, _array(fixture["expected_pg_advantages"]), rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(result.rhos, _array(fixture["expected_rhos"]), rtol=0.0, atol=1e-6)


def test_compute_vtrace_targets_uses_rho_bar_for_values_and_policy_advantages() -> None:
    result = compute_vtrace_targets(
        rewards=np.array([[1.0]], dtype=np.float32),
        values=np.array([[0.5], [0.2]], dtype=np.float32),
        discounts=np.array([[0.0]], dtype=np.float32),
        behavior_logp=np.array([[-10.0]], dtype=np.float32),
        target_logp=np.array([[0.0]], dtype=np.float32),
        rho_bar=1.0,
        c_bar=1.0,
    )

    assert result.rhos[0, 0] > 1_000.0
    assert result.vs[0, 0] == pytest.approx(1.0)
    assert result.pg_advantages[0, 0] == pytest.approx(0.5)


def test_compute_vtrace_targets_distinguishes_c_bar_recursion_from_rho_bar_policy_clip() -> None:
    rewards = np.array([[1.0], [2.0]], dtype=np.float32)
    values = np.zeros((3, 1), dtype=np.float32)
    discounts = np.ones((2, 1), dtype=np.float32)
    behavior_logp = np.full((2, 1), -np.log(4.0), dtype=np.float32)
    target_logp = np.zeros((2, 1), dtype=np.float32)

    low_trace_clip = compute_vtrace_targets(
        rewards,
        values,
        discounts,
        behavior_logp,
        target_logp,
        rho_bar=1.0,
        c_bar=0.5,
    )
    high_trace_clip = compute_vtrace_targets(
        rewards,
        values,
        discounts,
        behavior_logp,
        target_logp,
        rho_bar=1.0,
        c_bar=1.0,
    )
    swapped_clips = compute_vtrace_targets(
        rewards,
        values,
        discounts,
        behavior_logp,
        target_logp,
        rho_bar=0.5,
        c_bar=1.0,
    )

    expected_rhos = np.full((2, 1), 4.0, dtype=np.float32)
    np.testing.assert_allclose(low_trace_clip.rhos, expected_rhos, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(low_trace_clip.vs, np.array([[2.0], [2.0]], dtype=np.float32), rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(high_trace_clip.vs, np.array([[3.0], [2.0]], dtype=np.float32), rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(swapped_clips.vs, np.array([[1.5], [1.0]], dtype=np.float32), rtol=0.0, atol=1e-6)

    np.testing.assert_allclose(
        low_trace_clip.pg_advantages,
        np.array([[3.0], [2.0]], dtype=np.float32),
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(high_trace_clip.pg_advantages, low_trace_clip.pg_advantages, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(
        swapped_clips.pg_advantages,
        np.array([[1.0], [1.0]], dtype=np.float32),
        rtol=0.0,
        atol=1e-6,
    )

    assert low_trace_clip.vs[0, 0] == pytest.approx(2.0)
    assert high_trace_clip.vs[0, 0] == pytest.approx(3.0)
    assert low_trace_clip.pg_advantages[0, 0] == pytest.approx(high_trace_clip.pg_advantages[0, 0])
    assert low_trace_clip.pg_advantages[0, 0] == pytest.approx(3.0)
    assert swapped_clips.pg_advantages[0, 0] == pytest.approx(1.0)
    assert not np.allclose(low_trace_clip.vs, swapped_clips.vs)
    assert not np.allclose(low_trace_clip.pg_advantages, swapped_clips.pg_advantages)


def test_compute_vtrace_targets_requires_t_plus_one_values() -> None:
    with pytest.raises(ValueError, match="one extra bootstrap step"):
        compute_vtrace_targets(
            rewards=np.zeros((2, 3), dtype=np.float32),
            values=np.zeros((2, 3), dtype=np.float32),
            discounts=np.zeros((2, 3), dtype=np.float32),
            behavior_logp=np.zeros((2, 3), dtype=np.float32),
            target_logp=np.zeros((2, 3), dtype=np.float32),
        )


def test_summarize_vtrace_diagnostics_reports_percentiles_and_clip_rates() -> None:
    fixture = _load_fixture()
    result = VTraceTargets(
        vs=_array(fixture["expected_vs"]),
        pg_advantages=_array(fixture["expected_pg_advantages"]),
        rhos=_array(fixture["expected_rhos"]),
    )

    metrics = summarize_vtrace_diagnostics(result, rho_bar=fixture["rho_bar"], c_bar=fixture["c_bar"])

    assert metrics == pytest.approx(
        {
            "vtrace_rho_p50": 0.9811104834079742,
            "vtrace_rho_p90": 1.6487212538719178,
            "vtrace_rho_p95": 1.6487212955951691,
            "vtrace_rho_p99": 1.64872132897377,
            "vtrace_rho_clip_rate": 0.5,
            "vtrace_c_clip_rate": 0.5,
        }
    )


def test_impala_learner_update_exposes_vtrace_metrics() -> None:
    fixture = _load_fixture()
    learner = ImpalaLearner()
    result = compute_vtrace_targets(
        _array(fixture["rewards"]),
        _array(fixture["values"]),
        _array(fixture["discounts"]),
        _array(fixture["behavior_logp"]),
        _array(fixture["target_logp"]),
        rho_bar=fixture["rho_bar"],
        c_bar=fixture["c_bar"],
    )

    metrics = learner.update(
        {
            "vtrace_result": result,
            "vtrace_rho_bar": fixture["rho_bar"],
            "vtrace_c_bar": fixture["c_bar"],
        }
    )

    assert metrics["loss"] == pytest.approx(0.0)
    assert metrics["vtrace_rho_p95"] == pytest.approx(1.6487212955951691)
    assert metrics["vtrace_rho_clip_rate"] == pytest.approx(0.5)
    assert metrics["vtrace_c_clip_rate"] == pytest.approx(0.5)


def test_impala_learner_update_reduces_fixed_batch_loss_on_synthetic_targets() -> None:
    torch.manual_seed(0)

    model = PolicyValueModel(observation_dim=4, action_dim=4, config=_model_config())
    learner = ImpalaLearner(
        model=model,
        learning_rate=0.05,
        value_loss_coef=0.5,
        entropy_coef=0.0,
        grad_norm_clip=10.0,
    )
    batch = _synthetic_training_batch(learner)

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


def test_impala_learner_forward_time_major_matches_manual_seat_aware_rollout() -> None:
    torch.manual_seed(0)

    model = PolicyValueModel(observation_dim=4, action_dim=4, config=_model_config())
    learner = ImpalaLearner(model=model)
    batch = _synthetic_training_batch(
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

    model = PolicyValueModel(observation_dim=4, action_dim=4, config=_model_config())
    learner = ImpalaLearner(
        model=model,
        learning_rate=0.05,
        value_loss_coef=0.5,
        entropy_coef=0.0,
        grad_norm_clip=10.0,
    )
    batch = _synthetic_training_batch(
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
    model = PolicyValueModel(observation_dim=4, action_dim=4, config=_model_config())
    learner = ImpalaLearner(model=model)
    batch = _synthetic_training_batch(
        learner,
        seat_field_name=seat_field_name,
        initial_hidden_state=valid_initial_hidden_state,
    )
    batch["initial_hidden_state"] = invalid_initial_hidden_state

    with pytest.raises(ValueError, match=message):
        learner._loss_and_metrics(batch)


@pytest.mark.parametrize("seat_field_name", ["to_play_seat", "actor"])
def test_impala_learner_rejects_float_valued_seat_fields(seat_field_name: str) -> None:
    model = PolicyValueModel(observation_dim=4, action_dim=4, config=_model_config())
    learner = ImpalaLearner(model=model)
    batch = _synthetic_training_batch(
        learner,
        seat_field_name=seat_field_name,
        initial_hidden_state=np.zeros((2, 2, 8), dtype=np.float32),
    )
    batch[seat_field_name] = cast(np.ndarray, batch[seat_field_name]).astype(np.float32)

    with pytest.raises(ValueError, match=rf"{seat_field_name} must be integer-valued"):
        learner._loss_and_metrics(batch)


def test_impala_learner_rejects_mismatched_actor_alias() -> None:
    model = PolicyValueModel(observation_dim=4, action_dim=4, config=_model_config())
    learner = ImpalaLearner(model=model)
    batch = _synthetic_training_batch(
        learner,
        seat_field_name="to_play_seat",
        initial_hidden_state=np.zeros((2, 2, 8), dtype=np.float32),
    )
    batch["actor"] = 1 - cast(np.ndarray, batch["to_play_seat"])

    with pytest.raises(ValueError, match="actor must match to_play_seat"):
        learner._loss_and_metrics(batch)
