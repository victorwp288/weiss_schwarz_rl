from __future__ import annotations

import numpy as np
import torch
from weiss_rl.runtime import QueueRuntime

from .runtime_actor_policy_rows_test_support import (
    FactorizedRuntimeActorModel,
    packed_policy_rows,
    policy_rows_runtime,
    structured_legal_meta,
)


def test_policy_rows_ids_prefers_factorized_structured_sampler() -> None:
    runtime = policy_rows_runtime()
    model = FactorizedRuntimeActorModel()
    rows = packed_policy_rows()
    values_out = np.zeros((2,), dtype=np.float32)
    actions_out = np.zeros((2,), dtype=np.int64)
    logp_out = np.zeros((2,), dtype=np.float32)

    QueueRuntime._apply_policy_rows_ids(
        runtime,
        model=model,
        hidden_state=rows.hidden_state,
        row_indices=rows.row_indices,
        obs_step=rows.obs_step,
        actor_step=rows.actor_step,
        legal_ids=rows.legal_ids,
        legal_offsets=rows.legal_offsets,
        legal_action_meta=structured_legal_meta(),
        logits_out=None,
        values_out=values_out,
        actions_out=actions_out,
        logp_out=logp_out,
        rng=np.random.default_rng(123),
        sample_actions=True,
    )

    assert model.factorized_calls == 1
    assert model.packed_calls == 0
    assert actions_out.tolist() == [7, 7]
    assert np.allclose(logp_out, -0.25)
    assert np.allclose(values_out, 0.5)
    assert torch.allclose(rows.hidden_state, torch.ones_like(rows.hidden_state))
