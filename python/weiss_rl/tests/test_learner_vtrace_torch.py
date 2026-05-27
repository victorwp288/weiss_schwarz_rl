from __future__ import annotations

import numpy as np
import torch

from weiss_rl.learners.impala_learner import _compute_vtrace_targets_torch
from weiss_rl.learners.vtrace import compute_vtrace_targets
from weiss_rl.learners.vtrace_torch import compute_vtrace_targets_torch


def test_compute_vtrace_targets_torch_matches_numpy_reference() -> None:
    rewards = torch.tensor([[1.0, 0.5], [-0.25, 0.75], [0.0, 1.25]], dtype=torch.float32)
    values = torch.tensor([[0.2, 0.1], [0.3, 0.4], [-0.1, 0.2], [0.0, 0.5]], dtype=torch.float32)
    discounts = torch.tensor([[0.99, 0.9], [0.8, 0.7], [0.0, 0.5]], dtype=torch.float32)
    behavior_logp = torch.tensor([[-0.6, -0.5], [-0.7, -0.2], [-0.3, -0.9]], dtype=torch.float32)
    target_logp = torch.tensor([[-0.4, -0.8], [-0.2, -0.1], [-0.5, -0.3]], dtype=torch.float32)

    vs, advantages, rhos = compute_vtrace_targets_torch(
        rewards,
        values,
        discounts,
        behavior_logp,
        target_logp,
        rho_bar=1.2,
        c_bar=0.8,
    )
    reference = compute_vtrace_targets(
        rewards.numpy(),
        values.numpy(),
        discounts.numpy(),
        behavior_logp.numpy(),
        target_logp.numpy(),
        rho_bar=1.2,
        c_bar=0.8,
    )

    torch.testing.assert_close(vs, torch.as_tensor(reference.vs), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(advantages, torch.as_tensor(reference.pg_advantages), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(rhos, torch.as_tensor(reference.rhos), atol=1e-6, rtol=1e-6)


def test_compute_vtrace_targets_torch_caps_extreme_importance_weights() -> None:
    rewards = torch.zeros((1, 1), dtype=torch.float32)
    values = torch.zeros((2, 1), dtype=torch.float32)
    discounts = torch.ones((1, 1), dtype=torch.float32)
    behavior_logp = torch.zeros((1, 1), dtype=torch.float32)
    target_logp = torch.full((1, 1), 1000.0, dtype=torch.float32)

    _vs, _advantages, rhos = compute_vtrace_targets_torch(
        rewards,
        values,
        discounts,
        behavior_logp,
        target_logp,
        rho_bar=1.0,
        c_bar=1.0,
    )

    assert np.isfinite(float(rhos.item()))
    assert float(rhos.item()) == torch.finfo(torch.float32).max


def test_compute_vtrace_targets_torch_returns_stop_gradient_tensors() -> None:
    rewards = torch.ones((1, 1), dtype=torch.float32, requires_grad=True)
    values = torch.zeros((2, 1), dtype=torch.float32, requires_grad=True)
    discounts = torch.ones((1, 1), dtype=torch.float32, requires_grad=True)
    behavior_logp = torch.zeros((1, 1), dtype=torch.float32, requires_grad=True)
    target_logp = torch.full((1, 1), 0.5, dtype=torch.float32, requires_grad=True)

    vs, advantages, rhos = compute_vtrace_targets_torch(
        rewards,
        values,
        discounts,
        behavior_logp,
        target_logp,
        rho_bar=1.0,
        c_bar=1.0,
    )

    assert not vs.requires_grad
    assert not advantages.requires_grad
    assert not rhos.requires_grad


def test_impala_private_vtrace_wrapper_is_preserved() -> None:
    rewards = torch.tensor([[1.0]], dtype=torch.float32)
    values = torch.tensor([[0.0], [0.5]], dtype=torch.float32)
    discounts = torch.tensor([[0.9]], dtype=torch.float32)
    behavior_logp = torch.tensor([[-0.2]], dtype=torch.float32)
    target_logp = torch.tensor([[-0.1]], dtype=torch.float32)

    public = compute_vtrace_targets_torch(
        rewards,
        values,
        discounts,
        behavior_logp,
        target_logp,
        rho_bar=1.0,
        c_bar=1.0,
    )
    private = _compute_vtrace_targets_torch(
        rewards,
        values,
        discounts,
        behavior_logp,
        target_logp,
        rho_bar=1.0,
        c_bar=1.0,
    )

    for public_tensor, private_tensor in zip(public, private, strict=True):
        torch.testing.assert_close(private_tensor, public_tensor)
