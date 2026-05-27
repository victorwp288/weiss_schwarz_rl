from __future__ import annotations

import pytest
import torch

from weiss_rl.learners.trajectory_retention import trajectory_retention_action_loss


def test_trajectory_retention_loss_is_inactive_when_coef_is_zero() -> None:
    action_logp = torch.tensor([[-0.2, -1.1]], dtype=torch.float32)
    actions = torch.tensor([[1, 2]], dtype=torch.long)
    retention_valid = torch.tensor([[True, True]])

    loss, metrics = trajectory_retention_action_loss(
        action_logp=action_logp,
        actions=actions,
        retention_valid=retention_valid,
        coef=0.0,
    )

    assert loss.item() == pytest.approx(0.0)
    assert metrics == {}


def test_trajectory_retention_loss_tracks_supported_retained_actions() -> None:
    action_logp = torch.tensor([[-0.5, -2.0], [float("-inf"), -0.25]], dtype=torch.float32)
    actions = torch.tensor([[3, 4], [5, 6]], dtype=torch.long)
    retention_valid = torch.tensor([[True, False], [True, True]])
    top_action_ids = torch.tensor([[3, 8], [5, 0]], dtype=torch.long)

    loss, metrics = trajectory_retention_action_loss(
        action_logp=action_logp,
        actions=actions,
        retention_valid=retention_valid,
        coef=0.2,
        top_action_ids=top_action_ids,
    )

    raw_loss = (0.5 + 0.25) / 2.0
    assert loss.item() == pytest.approx(raw_loss * 0.2)
    assert metrics["trajectory_retention_coef_active"] == pytest.approx(0.2)
    assert metrics["trajectory_retention_valid_fraction"] == pytest.approx(3.0 / 4.0)
    assert metrics["trajectory_retention_supported_fraction"] == pytest.approx(2.0 / 3.0)
    assert metrics["trajectory_retention_rows"] == pytest.approx(2.0)
    assert metrics["trajectory_retention_loss"] == pytest.approx(raw_loss)
    assert metrics["trajectory_retention_weighted_loss"] == pytest.approx(raw_loss * 0.2)
    assert metrics["trajectory_retention_logp_mean"] == pytest.approx((-0.5 - 0.25) / 2.0)
    assert metrics["trajectory_retention_top_action_accuracy"] == pytest.approx(0.5)


def test_trajectory_retention_loss_handles_no_supported_rows() -> None:
    action_logp = torch.tensor([[-1.0, -2.0]], dtype=torch.float32)
    actions = torch.tensor([[1, 2]], dtype=torch.long)
    retention_valid = torch.tensor([[False, False]])

    loss, metrics = trajectory_retention_action_loss(
        action_logp=action_logp,
        actions=actions,
        retention_valid=retention_valid,
        coef=0.3,
    )

    assert loss.item() == pytest.approx(0.0)
    assert metrics["trajectory_retention_valid_fraction"] == pytest.approx(0.0)
    assert metrics["trajectory_retention_supported_fraction"] == pytest.approx(0.0)
    assert metrics["trajectory_retention_rows"] == pytest.approx(0.0)
    assert metrics["trajectory_retention_loss"] == pytest.approx(0.0)
    assert metrics["trajectory_retention_weighted_loss"] == pytest.approx(0.0)


def test_trajectory_retention_loss_rejects_shape_mismatches() -> None:
    with pytest.raises(ValueError, match="trajectory_retention_valid must match action_logp shape"):
        trajectory_retention_action_loss(
            action_logp=torch.zeros((2, 2)),
            actions=torch.zeros((2, 2), dtype=torch.long),
            retention_valid=torch.zeros((2, 1), dtype=torch.bool),
            coef=0.1,
        )
