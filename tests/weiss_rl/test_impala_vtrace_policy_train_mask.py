from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from weiss_rl.learners.action_logp import masked_action_logp_and_entropy
from weiss_rl.learners.impala.vtrace_targets import resolve_impala_vtrace_targets

from .impala_test_support import ImpalaLearner, TinyPolicyValueModel, _simple_training_batch


def test_impala_learner_raw_vtrace_uses_behavior_logp_on_non_train_rows_dense() -> None:
    torch.manual_seed(0)

    learner = ImpalaLearner(model=TinyPolicyValueModel(action_dim=2), vtrace_rho_bar=10.0, vtrace_c_bar=10.0)
    batch = _simple_training_batch()

    with torch.no_grad():
        logits, _values = learner._forward_time_major(torch.from_numpy(batch["obs"]))
        action_logp, _entropy = masked_action_logp_and_entropy(
            logits,
            torch.from_numpy(batch["legal_mask"]),
            torch.from_numpy(batch["actions"]),
            pass_action_id=None,
        )
    behavior_logp = action_logp.clone()
    behavior_logp[1, 0] = behavior_logp[1, 0] - 3.0

    raw_batch = {
        "obs": batch["obs"],
        "actions": batch["actions"],
        "legal_mask": batch["legal_mask"],
        "rewards": np.zeros((2, 1), dtype=np.float32),
        "discounts": np.ones((2, 1), dtype=np.float32),
        "behavior_logp": behavior_logp.cpu().numpy().astype(np.float32),
        "bootstrap_value": np.zeros((1,), dtype=np.float32),
        "policy_train_mask": np.asarray([[True], [False]], dtype=np.bool_),
    }

    _loss, _metrics, context = learner._loss_and_metrics_with_context(raw_batch)

    torch.testing.assert_close(context["vtrace_rhos"][0, 0], torch.tensor(1.0))
    torch.testing.assert_close(context["vtrace_rhos"][1, 0], torch.tensor(1.0))
    assert context["policy_train_mask"].tolist() == [[1.0], [0.0]]


def test_resolve_impala_vtrace_targets_preserves_off_policy_train_rows_and_masks_non_train_rows() -> None:
    values = torch.zeros((2, 1), dtype=torch.float32)
    action_logp = torch.tensor([[0.0], [-1.0]], dtype=torch.float32)
    behavior_logp = torch.tensor([[-2.0], [-4.0]], dtype=torch.float32)
    loss_mask = torch.tensor([[1.0], [0.0]], dtype=torch.float32)

    def float_target(value: Any, *, expected_shape: torch.Size, like: torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=like.dtype, device=like.device)
        assert tensor.shape == expected_shape
        return tensor

    def resolve_bootstrap(_batch: Any, *, batch_size: int, like: torch.Tensor) -> torch.Tensor:
        return torch.zeros((batch_size,), dtype=like.dtype, device=like.device)

    resolved = resolve_impala_vtrace_targets(
        batch={
            "rewards": torch.zeros((2, 1), dtype=torch.float32),
            "discounts": torch.ones((2, 1), dtype=torch.float32),
            "behavior_logp": behavior_logp,
        },
        vtrace_result=None,
        values=values,
        action_logp=action_logp,
        loss_mask=loss_mask,
        rho_bar=10.0,
        c_bar=10.0,
        float_target=float_target,
        resolve_bootstrap_value=resolve_bootstrap,
        batch_value=lambda batch, key: batch.get(key),
    )

    torch.testing.assert_close(resolved.action_logp, torch.tensor([[0.0], [-4.0]]))
    torch.testing.assert_close(resolved.behavior_logp_for_mask, behavior_logp)
    assert resolved.rhos_for_metrics[0, 0] == pytest.approx(float(np.exp(2.0)))
    assert resolved.rhos_for_metrics[1, 0] == pytest.approx(1.0)
    assert resolved.targets.requires_grad is False
    assert resolved.advantages.requires_grad is False
