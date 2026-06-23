from __future__ import annotations

import numpy as np
import pytest
import torch
from weiss_rl.runtime import QueueRuntime

from .runtime_actor_policy_rows_test_support import (
    ArgmaxRuntimeActorModel,
    packed_policy_rows,
    policy_rows_runtime,
)


def test_policy_rows_ids_argmax_selection_uses_legal_argmax_without_sampler() -> None:
    runtime = policy_rows_runtime()
    model = ArgmaxRuntimeActorModel()
    rows = packed_policy_rows()
    values_out = np.zeros((2,), dtype=np.float32)
    actions_out = np.zeros((2,), dtype=np.int64)
    logp_out = np.full((2,), -99.0, dtype=np.float32)

    QueueRuntime._apply_policy_rows_ids(
        runtime,
        model=model,
        hidden_state=rows.hidden_state,
        row_indices=rows.row_indices,
        obs_step=rows.obs_step,
        actor_step=rows.actor_step,
        legal_ids=rows.legal_ids,
        legal_offsets=rows.legal_offsets,
        legal_action_meta=None,
        logits_out=None,
        values_out=values_out,
        actions_out=actions_out,
        logp_out=logp_out,
        rng=np.random.default_rng(123),
        sample_actions=True,
        action_selection="argmax",
    )

    assert model.sample_calls == 0
    assert actions_out.tolist() == [7, 1]
    assert np.allclose(logp_out, 0.0)
    assert np.allclose(values_out, 0.5)
    assert torch.allclose(rows.hidden_state, torch.ones_like(rows.hidden_state))


def test_policy_rows_ids_argmax_selection_writes_deterministic_logits_for_fused_step() -> None:
    runtime = policy_rows_runtime()
    model = ArgmaxRuntimeActorModel()
    rows = packed_policy_rows()
    values_out = np.zeros((2,), dtype=np.float32)
    logits_out = np.zeros((2, 9), dtype=np.float32)

    QueueRuntime._apply_policy_rows_ids(
        runtime,
        model=model,
        hidden_state=rows.hidden_state,
        row_indices=rows.row_indices,
        obs_step=rows.obs_step,
        actor_step=rows.actor_step,
        legal_ids=rows.legal_ids,
        legal_offsets=rows.legal_offsets,
        legal_action_meta=None,
        logits_out=logits_out,
        values_out=values_out,
        actions_out=None,
        logp_out=None,
        rng=np.random.default_rng(123),
        sample_actions=False,
        action_selection="argmax",
    )

    assert model.sample_calls == 0
    assert logits_out[0, 7] == pytest.approx(0.0)
    assert logits_out[0, 0] == pytest.approx(-100.0)
    assert logits_out[0, 8] == pytest.approx(-100.0)
    assert logits_out[1, 1] == pytest.approx(0.0)
    assert logits_out[1, 7] == pytest.approx(-100.0)
    assert logits_out[1, 8] == pytest.approx(-100.0)
    assert np.all(logits_out[:, [2, 3, 4, 5, 6]] < -1.0e8)
    assert np.allclose(values_out, 0.5)
    assert torch.allclose(rows.hidden_state, torch.ones_like(rows.hidden_state))
