from __future__ import annotations

import pytest
import torch
from weiss_rl.learners.paired_swing.loss import (
    packed_paired_swing_margin_loss,
)


def test_packed_paired_swing_margin_loss_penalizes_positive_below_negative() -> None:
    packed_logits = torch.tensor([0.0, 1.0], dtype=torch.float32)
    legal_ids = torch.tensor([1, 2], dtype=torch.long)
    legal_offsets = torch.tensor([0, 2], dtype=torch.long)
    positive_actions = torch.tensor([[1]], dtype=torch.long)
    negative_actions = torch.tensor([[2]], dtype=torch.long)
    valid = torch.tensor([[True]])
    loss_mask = torch.tensor([[1.0]], dtype=torch.float32)

    loss, metrics, context = packed_paired_swing_margin_loss(
        packed_logits=packed_logits,
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        positive_actions=positive_actions,
        negative_actions=negative_actions,
        negative_valid=valid,
        loss_mask=loss_mask,
        margin=0.25,
        pass_action_id=0,
    )

    assert loss.item() == pytest.approx(1.25)
    assert metrics["paired_swing_rows"] == 1.0
    assert metrics["paired_swing_candidate_rows"] == 1.0
    assert metrics["paired_swing_margin_mean"] == pytest.approx(-1.0)
    assert metrics["paired_swing_satisfied_fraction"] == 0.0
    assert "paired_swing_margins" in context


def test_packed_paired_swing_margin_loss_can_compare_positive_to_top_other() -> None:
    packed_logits = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)
    legal_ids = torch.tensor([1, 2, 3], dtype=torch.long)
    legal_offsets = torch.tensor([0, 3], dtype=torch.long)
    positive_actions = torch.tensor([[1]], dtype=torch.long)
    negative_actions = torch.tensor([[2]], dtype=torch.long)
    valid = torch.tensor([[True]])
    loss_mask = torch.tensor([[1.0]], dtype=torch.float32)

    loss, metrics, context = packed_paired_swing_margin_loss(
        packed_logits=packed_logits,
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        positive_actions=positive_actions,
        negative_actions=negative_actions,
        negative_valid=valid,
        loss_mask=loss_mask,
        margin=0.25,
        pass_action_id=0,
        compare_to="top_other",
    )

    assert loss.item() == pytest.approx(2.25)
    assert metrics["paired_swing_compare_to_top_other"] == 1.0
    assert metrics["paired_swing_margin_mean"] == pytest.approx(-2.0)
    assert context["paired_swing_margins"].tolist() == pytest.approx([-2.0])


def test_packed_paired_swing_margin_loss_ignores_matching_actions() -> None:
    loss, metrics, context = packed_paired_swing_margin_loss(
        packed_logits=torch.tensor([1.0, 0.0], dtype=torch.float32),
        legal_ids=torch.tensor([1, 2], dtype=torch.long),
        legal_offsets=torch.tensor([0, 2], dtype=torch.long),
        positive_actions=torch.tensor([[1]], dtype=torch.long),
        negative_actions=torch.tensor([[1]], dtype=torch.long),
        negative_valid=torch.tensor([[True]]),
        loss_mask=torch.tensor([[1.0]], dtype=torch.float32),
        margin=0.25,
        pass_action_id=0,
    )

    assert loss.item() == 0.0
    assert metrics["paired_swing_candidate_rows"] == 0.0
    assert metrics["paired_swing_rows"] == 0.0
    assert context == {}
