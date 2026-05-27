from __future__ import annotations

import pytest
import torch

from weiss_rl.learners.policy_anchor import packed_candidate_anchor_top_action_loss


def test_packed_candidate_anchor_top_action_loss_tracks_agreement() -> None:
    current_log_probs = torch.log_softmax(
        torch.tensor([0.0, 2.0, 3.0, 0.0], dtype=torch.float32).reshape(2, 2),
        dim=-1,
    ).reshape(-1)
    anchor_log_probs = torch.log_softmax(
        torch.tensor([2.0, 0.0, 0.0, 3.0], dtype=torch.float32).reshape(2, 2),
        dim=-1,
    ).reshape(-1)

    loss, metrics = packed_candidate_anchor_top_action_loss(
        current_log_probs=current_log_probs,
        anchor_log_probs=anchor_log_probs,
        packed_offsets=torch.tensor([0, 2, 4], dtype=torch.long),
        row_shape=(2, 1),
        loss_mask=torch.ones((2, 1), dtype=torch.float32),
    )

    expected = -0.5 * (current_log_probs[0] + current_log_probs[3])
    assert loss.item() == pytest.approx(float(expected.item()))
    assert metrics["policy_anchor_top_action_loss"] == pytest.approx(float(expected.item()))
    assert metrics["policy_anchor_top_action_agreement"] == pytest.approx(0.0)
    assert metrics["policy_anchor_top_action_loss_p95"] > metrics["policy_anchor_top_action_loss"]
