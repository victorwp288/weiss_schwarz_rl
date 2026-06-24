from __future__ import annotations

import torch
from weiss_rl.models.public_heuristic.public_heuristics import (
    apply_public_heuristic_bias,
    combine_public_heuristic_scores,
)


def test_combine_public_heuristic_scores_preserves_weighting() -> None:
    combined = combine_public_heuristic_scores(
        torch.tensor([1.0, 2.0]),
        torch.tensor([4.0, 8.0]),
        torch.tensor([8.0, 4.0]),
        dtype=torch.float32,
    )

    assert torch.equal(combined, torch.tensor([38.0, 73.0]))


def test_apply_public_heuristic_bias_supports_scale_and_family_allow_list() -> None:
    scores = torch.tensor([1.0, 2.0, 3.0])
    raw_scores = torch.tensor([100.0, 50.0, -20.0])

    gated = apply_public_heuristic_bias(
        scores,
        raw_scores,
        scale=10.0,
        family_ids=torch.tensor([2, 3, 4], dtype=torch.long),
        bias_family_ids=torch.tensor([2, 4], dtype=torch.long),
    )
    ungated = apply_public_heuristic_bias(
        scores,
        raw_scores,
        scale=10.0,
        family_ids=None,
        bias_family_ids=torch.empty((0,), dtype=torch.long),
    )
    disabled = apply_public_heuristic_bias(
        scores,
        raw_scores,
        scale=0.0,
        family_ids=torch.tensor([2, 3, 4], dtype=torch.long),
        bias_family_ids=torch.tensor([2, 4], dtype=torch.long),
    )

    assert torch.equal(gated, torch.tensor([11.0, 2.0, 1.0]))
    assert torch.equal(ungated, torch.tensor([11.0, 7.0, 1.0]))
    assert torch.equal(disabled, scores)
