from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np
import torch
from weiss_rl.config.models import ModelConfig, ModelDropoutConfig
from weiss_rl.learners.impala import ImpalaLearner
from weiss_rl.learners.vtrace import VTraceTargets

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


def load_vtrace_fixture() -> VTraceFixture:
    payload = json.loads(TEST_VECTORS_PATH.read_text(encoding="utf-8"))
    return cast(VTraceFixture, payload)


def fixture_array(rows: list[list[float]]) -> np.ndarray:
    return np.asarray(rows, dtype=np.float32)


def tiny_model_config() -> ModelConfig:
    return ModelConfig(
        gru_hidden_size=8,
        encoder_mlp_width=8,
        encoder_mlp_layers=1,
        layer_norm=False,
        dropout=ModelDropoutConfig(family_a=0.0, ablation=0.0),
    )


def synthetic_vtrace_training_batch(
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
        _, values = learner._forward_time_major(torch.from_numpy(obs), **cast(Any, forward_kwargs))

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
