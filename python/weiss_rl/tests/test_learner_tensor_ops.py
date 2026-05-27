from __future__ import annotations

import numpy as np
import torch

from weiss_rl.learners.tensor_ops import (
    nonfinite_indices,
    segment_group_sum,
    segment_logsumexp,
    segment_max,
    weighted_mean,
)


def test_segment_max_keeps_empty_segments_at_negative_infinity() -> None:
    values = torch.tensor([1.0, 5.0, 2.0, -3.0])
    keys = torch.tensor([0, 2, 0, 2], dtype=torch.long)

    reduced = segment_max(values, keys, num_segments=4)

    torch.testing.assert_close(reduced, torch.tensor([2.0, -torch.inf, 5.0, -torch.inf]))
    torch.testing.assert_close(
        segment_max(values[:0], keys[:0], num_segments=2),
        torch.tensor([-torch.inf, -torch.inf]),
    )


def test_segment_logsumexp_is_stable_and_keeps_empty_segments_at_negative_infinity() -> None:
    values = torch.tensor([1000.0, 1001.0, -2.0])
    keys = torch.tensor([0, 0, 2], dtype=torch.long)

    reduced = segment_logsumexp(values, keys, num_segments=4)

    torch.testing.assert_close(reduced[0], torch.logsumexp(values[:2], dim=0))
    torch.testing.assert_close(reduced[2], values[2])
    assert reduced[1].item() == -torch.inf
    assert reduced[3].item() == -torch.inf
    torch.testing.assert_close(
        segment_logsumexp(values[:0], keys[:0], num_segments=2),
        torch.tensor([-torch.inf, -torch.inf]),
    )


def test_segment_group_sum_ignores_invalid_group_ids_and_handles_empty_shapes() -> None:
    values = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0])
    rows = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)
    groups = torch.tensor([0, 2, 1, -1, 5], dtype=torch.long)

    grouped = segment_group_sum(values, rows, groups, row_count=3, group_count=3)

    torch.testing.assert_close(
        grouped,
        torch.tensor(
            [
                [1.0, 0.0, 2.0],
                [0.0, 4.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ),
    )
    assert segment_group_sum(values, rows, groups, row_count=3, group_count=0).shape == (3, 0)
    torch.testing.assert_close(
        segment_group_sum(values[:0], rows[:0], groups[:0], row_count=2, group_count=2),
        torch.zeros((2, 2)),
    )


def test_weighted_mean_uses_clamped_denominator_for_zero_weight_rows() -> None:
    values = torch.tensor([10.0, 20.0, 30.0])

    torch.testing.assert_close(weighted_mean(values, torch.tensor([1.0, 0.0, 3.0])), torch.tensor(25.0))
    torch.testing.assert_close(weighted_mean(values, torch.zeros(3)), torch.tensor(0.0))


def test_nonfinite_indices_supports_torch_tensors_and_numpy_arrays() -> None:
    tensor_values = torch.tensor([[1.0, float("nan")], [float("inf"), -5.0]])
    array_values = np.array([[1.0, -np.inf], [np.nan, 4.0]])

    assert nonfinite_indices(tensor_values).tolist() == [[0, 1], [1, 0]]
    assert nonfinite_indices(array_values).tolist() == [[0, 1], [1, 0]]
