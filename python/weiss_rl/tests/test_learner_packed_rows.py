from __future__ import annotations

import torch

from weiss_rl.learners.packed_rows import (
    packed_candidate_positions_for_rows,
    packed_legal_action_view,
    scatter_packed_candidate_values,
    slice_packed_legal_rows_with_meta,
    subset_observation_context_rows,
)


def _packed() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([10, 11, 20, 21, 22, 30], dtype=torch.long),
        torch.tensor([0, 2, 2, 5, 6], dtype=torch.long),
        torch.tensor(
            [
                [10, 0],
                [11, 1],
                [20, 2],
                [21, 3],
                [22, 4],
                [30, 5],
            ],
            dtype=torch.long,
        ),
    )


def test_packed_legal_action_view_exposes_ids_offsets_and_meta() -> None:
    packed = _packed()

    view = packed_legal_action_view(packed)

    assert view.ids is packed[0]
    assert view.offsets is packed[1]
    assert view.meta is packed[2]


def test_slice_packed_legal_rows_with_meta_preserves_selected_row_order() -> None:
    ids, offsets, meta = slice_packed_legal_rows_with_meta(_packed(), torch.tensor([2, 0], dtype=torch.long))

    assert ids.tolist() == [20, 21, 22, 10, 11]
    assert offsets.tolist() == [0, 3, 5]
    assert meta is not None
    assert meta.tolist() == [[20, 2], [21, 3], [22, 4], [10, 0], [11, 1]]


def test_slice_packed_legal_rows_with_meta_handles_empty_and_zero_width_rows() -> None:
    empty_ids, empty_offsets, empty_meta = slice_packed_legal_rows_with_meta(
        _packed(),
        torch.empty((0,), dtype=torch.long),
    )
    zero_ids, zero_offsets, zero_meta = slice_packed_legal_rows_with_meta(
        _packed(),
        torch.tensor([1], dtype=torch.long),
    )

    assert empty_ids.tolist() == []
    assert empty_offsets.tolist() == [0]
    assert empty_meta is not None and empty_meta.shape == (0, 2)
    assert zero_ids.tolist() == []
    assert zero_offsets.tolist() == [0, 0]
    assert zero_meta is not None and zero_meta.shape == (0, 2)


def test_packed_candidate_positions_for_rows_matches_packed_slice_positions() -> None:
    _ids, offsets, _meta = _packed()

    positions = packed_candidate_positions_for_rows(offsets, torch.tensor([2, 0], dtype=torch.long))
    empty = packed_candidate_positions_for_rows(offsets, torch.empty((0,), dtype=torch.long))
    zero_width = packed_candidate_positions_for_rows(offsets, torch.tensor([1], dtype=torch.long))

    assert positions.tolist() == [2, 3, 4, 0, 1]
    assert empty.tolist() == []
    assert zero_width.tolist() == []


def test_scatter_packed_candidate_values_restores_subset_values_with_fill() -> None:
    values = scatter_packed_candidate_values(
        _packed(),
        torch.tensor([2, 0], dtype=torch.long),
        torch.tensor([0.2, 0.3, 0.4, 0.5, 0.6], dtype=torch.float32),
        fill_value=-1.0,
    )

    assert torch.equal(values, torch.tensor([0.5, 0.6, 0.2, 0.3, 0.4, -1.0]))


def test_subset_observation_context_rows_only_subsets_row_major_tensors() -> None:
    row_context = torch.arange(4 * 2, dtype=torch.float32).reshape(4, 2)
    other_tensor = torch.arange(3, dtype=torch.float32)
    context = {
        "row": row_context,
        "other": other_tensor,
        "scalar": torch.tensor(1.0),
    }

    subset = subset_observation_context_rows(context, torch.tensor([3, 1], dtype=torch.long), row_count=4)

    assert torch.equal(subset["row"], torch.stack([row_context[3], row_context[1]]))
    assert subset["other"] is other_tensor
    assert subset["scalar"] is context["scalar"]
