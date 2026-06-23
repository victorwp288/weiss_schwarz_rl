from __future__ import annotations

import pytest
import torch
from weiss_rl.learners.paired_swing.comparison import paired_swing_margin_comparison_rows
from weiss_rl.learners.paired_swing.rows import positive_vs_top_other_margin_by_row


def test_paired_swing_margin_comparison_rows_preserves_negative_action_path() -> None:
    positive_actions = torch.tensor([[1, 2]], dtype=torch.long)
    negative_actions = torch.tensor([[2, 1]], dtype=torch.long)

    comparison = paired_swing_margin_comparison_rows(
        packed_logits=torch.tensor([0.0, 1.0, 2.0, 0.0], dtype=torch.float32),
        legal_ids=torch.tensor([1, 2, 1, 2], dtype=torch.long),
        legal_offsets=torch.tensor([0, 2, 4], dtype=torch.long),
        flat_positive_actions=positive_actions.reshape(-1),
        flat_negative_actions=negative_actions.reshape(-1),
        positive_actions=positive_actions,
        negative_actions=negative_actions,
        active_rows=torch.tensor([True, True]),
        pass_action_id=0,
        compare_to="negative",
    )

    assert comparison.supported.tolist() == [True, True]
    assert comparison.margin_by_row.tolist() == pytest.approx([-1.0, -2.0])
    assert comparison.positive_logp_by_row.tolist() == pytest.approx([-1.3132616, -2.126928])
    assert comparison.negative_logp_by_row.tolist() == pytest.approx([-0.3132616, -0.126928])


def test_positive_vs_top_other_margin_by_row_preserves_packed_row_support_contract() -> None:
    margin_by_row, supported, positive_logp, top_other_logp = positive_vs_top_other_margin_by_row(
        packed_logits=torch.tensor([0.0, 1.0, 2.0, 4.0], dtype=torch.float32),
        legal_ids=torch.tensor([1, 2, 3, 1], dtype=torch.long),
        legal_offsets=torch.tensor([0, 3, 4], dtype=torch.long),
        flat_positive_actions=torch.tensor([1, 1], dtype=torch.long),
        active_rows=torch.tensor([True, True]),
    )

    assert supported.tolist() == [True, False]
    assert margin_by_row[0].item() == pytest.approx(-2.0)
    assert torch.isneginf(margin_by_row[1])
    assert positive_logp[0].item() < top_other_logp[0].item()
