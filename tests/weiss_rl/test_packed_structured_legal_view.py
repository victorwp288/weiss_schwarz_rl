from __future__ import annotations

import pytest
import torch
from weiss_rl.learners.structured_auxiliary import (
    packed_group_log_probs,
    packed_soft_target_cross_entropy,
    packed_structured_legal_view,
)


def test_packed_structured_legal_view_selects_dense_logits_and_normalizes_metadata() -> None:
    unused = torch.iinfo(torch.uint16).max
    logits = torch.tensor([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]])
    packed_ids = torch.tensor([0, 2, 1], dtype=torch.long)
    packed_offsets = torch.tensor([0, 2, 3], dtype=torch.long)
    packed_meta = torch.tensor(
        [
            [0, 10, unused, unused],
            [2, unused, 20, unused],
            [1, 30, 31, 32],
        ],
        dtype=torch.long,
    )

    view = packed_structured_legal_view(
        logits=logits,
        packed_ids=packed_ids,
        packed_offsets=packed_offsets,
        packed_meta=packed_meta,
    )

    assert view is not None
    assert view.row_count == 2
    assert view.row_indices.tolist() == [0, 0, 1]
    assert view.action_ids.tolist() == [0, 2, 1]
    torch.testing.assert_close(view.logits, torch.tensor([1.0, 3.0, 5.0]))
    torch.testing.assert_close(view.row_log_z, torch.tensor([torch.logsumexp(torch.tensor([1.0, 3.0]), dim=0), 5.0]))
    assert view.row_has_candidates.tolist() == [True, True]
    assert view.family_ids.tolist() == [0, 2, 1]
    assert view.arg0.tolist() == [10, -1, 30]
    assert view.arg1.tolist() == [-1, 20, 31]
    assert view.arg2.tolist() == [-1, -1, 32]


def test_packed_structured_legal_view_supports_flat_logits_no_logits_and_empty_rows() -> None:
    packed_ids = torch.tensor([1], dtype=torch.long)
    packed_offsets = torch.tensor([0, 0, 1], dtype=torch.long)
    packed_meta = torch.tensor([[3, 4, 5, 6]], dtype=torch.long)

    flat_view = packed_structured_legal_view(
        logits=torch.tensor([7.0]),
        packed_ids=packed_ids,
        packed_offsets=packed_offsets,
        packed_meta=packed_meta,
    )
    assert flat_view is not None
    assert flat_view.row_indices.tolist() == [1]
    torch.testing.assert_close(flat_view.logits, torch.tensor([7.0]))
    assert flat_view.row_has_candidates.tolist() == [False, True]
    assert flat_view.row_log_z[0].item() == -torch.inf
    torch.testing.assert_close(flat_view.row_log_z[1], torch.tensor(7.0))

    zero_view = packed_structured_legal_view(
        logits=None,
        packed_ids=packed_ids,
        packed_offsets=packed_offsets,
        packed_meta=packed_meta,
    )
    assert zero_view is not None
    torch.testing.assert_close(zero_view.logits, torch.tensor([0.0]))


def test_packed_structured_legal_view_validates_shapes_and_missing_inputs() -> None:
    packed_ids = torch.tensor([0, 1], dtype=torch.long)
    packed_offsets = torch.tensor([0, 2], dtype=torch.long)
    packed_meta = torch.zeros((2, 4), dtype=torch.long)

    assert (
        packed_structured_legal_view(
            logits=torch.zeros((1, 2)),
            packed_ids=None,
            packed_offsets=packed_offsets,
            packed_meta=packed_meta,
        )
        is None
    )
    with pytest.raises(ValueError, match="packed logits must align"):
        packed_structured_legal_view(
            logits=torch.tensor([1.0]),
            packed_ids=packed_ids,
            packed_offsets=packed_offsets,
            packed_meta=packed_meta,
        )
    with pytest.raises(ValueError, match="packed legal offsets must describe 1 rows"):
        packed_structured_legal_view(
            logits=torch.zeros((1, 2)),
            packed_ids=packed_ids,
            packed_offsets=torch.tensor([0, 1, 2], dtype=torch.long),
            packed_meta=packed_meta,
        )
    with pytest.raises(ValueError, match="packed legal metadata must align"):
        packed_structured_legal_view(
            logits=torch.zeros((1, 2)),
            packed_ids=packed_ids,
            packed_offsets=packed_offsets,
            packed_meta=torch.zeros((2, 3), dtype=torch.long),
        )


def test_packed_group_log_probs_handles_groups_masks_and_empty_group_count() -> None:
    view = packed_structured_legal_view(
        logits=torch.tensor([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]]),
        packed_ids=torch.tensor([0, 2, 1], dtype=torch.long),
        packed_offsets=torch.tensor([0, 2, 3], dtype=torch.long),
        packed_meta=torch.zeros((3, 4), dtype=torch.long),
    )
    assert view is not None
    group_ids = torch.tensor([0, 1, 0], dtype=torch.long)

    log_probs = packed_group_log_probs(view, group_ids=group_ids, group_count=2)
    row0_z = torch.logsumexp(torch.tensor([1.0, 3.0]), dim=0)
    torch.testing.assert_close(
        log_probs,
        torch.tensor(
            [
                [1.0 - row0_z, 3.0 - row0_z],
                [0.0, -torch.inf],
            ]
        ),
    )

    masked = packed_group_log_probs(
        view,
        group_ids=group_ids,
        group_count=2,
        candidate_mask=torch.tensor([True, False, True]),
    )
    torch.testing.assert_close(masked, torch.tensor([[0.0, -torch.inf], [0.0, -torch.inf]]))
    assert packed_group_log_probs(view, group_ids=group_ids, group_count=0).shape == (2, 0)


def test_packed_soft_target_cross_entropy_matches_manual_row_calculation() -> None:
    view = packed_structured_legal_view(
        logits=torch.tensor([1.0, 3.0, 5.0]),
        packed_ids=torch.tensor([0, 2, 1], dtype=torch.long),
        packed_offsets=torch.tensor([0, 2, 3], dtype=torch.long),
        packed_meta=torch.zeros((3, 4), dtype=torch.long),
    )
    assert view is not None
    target_logits = torch.tensor([2.0, 0.0, 9.0])

    cross_entropy, top_mass, target_entropy = packed_soft_target_cross_entropy(
        view,
        target_logits=target_logits,
        temperature=1.0,
    )

    row0_target_log_probs = target_logits[:2] - torch.logsumexp(target_logits[:2], dim=0)
    row0_target_probs = torch.exp(row0_target_log_probs)
    row0_student_log_probs = view.logits[:2] - torch.logsumexp(view.logits[:2], dim=0)
    row0_cross_entropy = -(row0_target_probs * row0_student_log_probs).sum()
    row0_entropy = -(row0_target_probs * row0_target_log_probs).sum()

    torch.testing.assert_close(cross_entropy, torch.tensor([row0_cross_entropy, 0.0]))
    torch.testing.assert_close(top_mass, torch.tensor([row0_target_probs[1], 1.0]))
    torch.testing.assert_close(target_entropy, torch.tensor([row0_entropy, 0.0]))

    with pytest.raises(ValueError, match="temperature must be > 0"):
        packed_soft_target_cross_entropy(view, target_logits=target_logits, temperature=0.0)
    with pytest.raises(ValueError, match="target logits must align"):
        packed_soft_target_cross_entropy(view, target_logits=target_logits[:2], temperature=1.0)
