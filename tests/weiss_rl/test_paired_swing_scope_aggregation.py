from __future__ import annotations

import pytest
import torch
from weiss_rl.learners.paired_swing.loss import (
    packed_paired_swing_margin_loss,
)
from weiss_rl.learners.paired_swing.scope import paired_swing_scoped_margin_loss


def test_paired_swing_scoped_margin_loss_preserves_label_mean_contract() -> None:
    loss, margin_mean, satisfied_fraction, metrics = paired_swing_scoped_margin_loss(
        margins=torch.tensor([-1.0, -1.0, 1.0], dtype=torch.float32),
        supported_weight=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
        supported=torch.tensor([True, True, True]),
        positive_actions=torch.tensor([[1, 1, 1]], dtype=torch.long),
        group_ids=torch.tensor([1, 1, 2], dtype=torch.long),
        normalized_scope="label_mean",
        margin=0.25,
    )

    assert loss.item() == pytest.approx(0.625)
    assert margin_mean.item() == pytest.approx(0.0)
    assert satisfied_fraction.item() == pytest.approx(0.5)
    assert metrics["paired_swing_label_count"] == 2.0
    assert metrics["paired_swing_label_rows_mean"] == pytest.approx(1.5)


def test_packed_paired_swing_episode_mean_loss_hinges_once_per_episode() -> None:
    packed_logits = torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.float32)
    legal_ids = torch.tensor([1, 2, 1, 2], dtype=torch.long)
    legal_offsets = torch.tensor([0, 2, 4], dtype=torch.long)
    positive_actions = torch.tensor([[1], [1]], dtype=torch.long)
    negative_actions = torch.tensor([[2], [2]], dtype=torch.long)
    valid = torch.tensor([[True], [True]])
    loss_mask = torch.tensor([[1.0], [1.0]], dtype=torch.float32)

    loss, metrics, _context = packed_paired_swing_margin_loss(
        packed_logits=packed_logits,
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        positive_actions=positive_actions,
        negative_actions=negative_actions,
        negative_valid=valid,
        loss_mask=loss_mask,
        margin=0.25,
        pass_action_id=0,
        loss_scope="episode_mean",
    )

    assert loss.item() == pytest.approx(0.25)
    assert metrics["paired_swing_loss_scope_episode_mean"] == 1.0
    assert metrics["paired_swing_episode_count"] == 1.0
    assert metrics["paired_swing_margin_mean"] == pytest.approx(0.0)


def test_packed_paired_swing_label_mean_loss_balances_source_labels() -> None:
    packed_logits = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0, 0.0], dtype=torch.float32)
    legal_ids = torch.tensor([1, 2, 1, 2, 1, 2], dtype=torch.long)
    legal_offsets = torch.tensor([0, 2, 4, 6], dtype=torch.long)
    positive_actions = torch.tensor([[1, 1, 1]], dtype=torch.long)
    negative_actions = torch.tensor([[2, 2, 2]], dtype=torch.long)
    valid = torch.tensor([[True, True, True]])
    loss_mask = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
    group_ids = torch.tensor([[1, 1, 2]], dtype=torch.long)

    loss, metrics, _context = packed_paired_swing_margin_loss(
        packed_logits=packed_logits,
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        positive_actions=positive_actions,
        negative_actions=negative_actions,
        negative_valid=valid,
        loss_mask=loss_mask,
        margin=0.25,
        pass_action_id=0,
        loss_scope="label_mean",
        group_ids=group_ids,
    )

    assert loss.item() == pytest.approx(0.625)
    assert metrics["paired_swing_loss_scope_label_mean"] == 1.0
    assert metrics["paired_swing_label_count"] == 2.0
    assert metrics["paired_swing_margin_mean"] == pytest.approx(0.0)
