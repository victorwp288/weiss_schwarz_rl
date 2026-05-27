from __future__ import annotations

import pytest
import torch

from weiss_rl.models.tensor_ops import (
    bucket_card_ids,
    derived_sample_seeds,
    factorized_local_row_indices,
    masked_entropy_from_log_probs,
    masked_log_softmax,
    masked_max_pool,
    masked_mean_pool,
    negative_logits_fill_value,
    optional_embedding,
    packed_local_cdf,
    packed_row_indices,
    packed_row_log_z,
    scatter_factorized_row_values,
    uniform_from_seeds,
)


def test_masked_pooling_ignores_masked_values_and_handles_empty_rows() -> None:
    values = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [100.0, 200.0]],
            [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
        ]
    )
    mask = torch.tensor([[True, True, False], [False, False, False]])

    torch.testing.assert_close(masked_mean_pool(values, mask), torch.tensor([[2.0, 3.0], [0.0, 0.0]]))
    torch.testing.assert_close(masked_max_pool(values, mask), torch.tensor([[3.0, 4.0], [0.0, 0.0]]))
    torch.testing.assert_close(
        masked_max_pool(values[:, :0, :], mask[:, :0]),
        torch.zeros((2, 2)),
    )


def test_bucket_card_ids_maps_nonpositive_ids_to_zero_and_hashes_positive_ids() -> None:
    card_ids = torch.tensor([-3, 0, 1, 5, 6])

    assert bucket_card_ids(card_ids, vocab_size=1).tolist() == [0, 0, 0, 0, 0]
    assert bucket_card_ids(card_ids, vocab_size=5).tolist() == [0, 0, 2, 2, 3]


def test_optional_embedding_offsets_valid_ids_and_maps_negative_ids_to_zero() -> None:
    embedding = torch.nn.Embedding(4, 2)
    with torch.no_grad():
        embedding.weight.copy_(torch.tensor([[0.0, 0.0], [1.0, 1.5], [2.0, 2.5], [3.0, 3.5]]))

    embedded = optional_embedding(embedding, torch.tensor([-1, 0, 2]))

    torch.testing.assert_close(embedded, torch.tensor([[0.0, 0.0], [1.0, 1.5], [3.0, 3.5]]))


def test_negative_logits_fill_value_uses_dtype_min_for_float_masks() -> None:
    assert negative_logits_fill_value(torch.float16) == float(torch.finfo(torch.float16).min)
    assert negative_logits_fill_value(torch.int64) == pytest.approx(-1.0e9)


def test_packed_row_indices_log_z_and_local_cdf_are_row_local() -> None:
    offsets = torch.tensor([0, 2, 2, 5], dtype=torch.long)
    scores = torch.tensor([0.0, 1.0, -1.0, 0.0, 1.0])

    assert packed_row_indices(offsets).tolist() == [0, 0, 2, 2, 2]
    row_log_z = packed_row_log_z(scores, offsets)
    expected_row0 = torch.logsumexp(torch.tensor([0.0, 1.0]), dim=0)
    expected_row2 = torch.logsumexp(torch.tensor([-1.0, 0.0, 1.0]), dim=0)
    torch.testing.assert_close(row_log_z, torch.tensor([expected_row0, -torch.inf, expected_row2]))

    probabilities = torch.exp(scores - row_log_z.index_select(0, packed_row_indices(offsets)))
    torch.testing.assert_close(
        packed_local_cdf(probabilities, offsets),
        torch.tensor(
            [
                probabilities[0],
                probabilities[0] + probabilities[1],
                probabilities[2],
                probabilities[2] + probabilities[3],
                probabilities[2] + probabilities[3] + probabilities[4],
            ]
        ),
    )


def test_uniform_and_derived_sample_seeds_are_deterministic_and_bounded() -> None:
    seeds = torch.tensor([1, 2, 3], dtype=torch.long)

    first = uniform_from_seeds(seeds, dtype=torch.float32)
    second = uniform_from_seeds(seeds, dtype=torch.float32)

    torch.testing.assert_close(first, second)
    assert bool((first > 0.0).all().item())
    assert bool((first < 1.0).all().item())
    assert derived_sample_seeds(seeds, salt=0x55).tolist() == (seeds ^ torch.full_like(seeds, 0x55)).tolist()


def test_uniform_from_seeds_does_not_clamp_negative_hashes_to_epsilon() -> None:
    seeds = torch.arange(10_000, dtype=torch.long)

    values = uniform_from_seeds(seeds, dtype=torch.float32)

    eps = torch.finfo(torch.float32).eps
    assert int((values <= eps * 1.1).sum().item()) == 0
    assert 0.45 <= float(values.mean().item()) <= 0.55
    assert 0.08 <= float((values < 0.1).to(dtype=torch.float32).mean().item()) <= 0.12
    assert 0.08 <= float((values > 0.9).to(dtype=torch.float32).mean().item()) <= 0.12


def test_masked_log_softmax_and_entropy_keep_masked_entries_at_fill_value() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]])
    mask = torch.tensor([[True, False, True], [False, False, False]])

    log_probs = masked_log_softmax(logits, mask)
    fill = negative_logits_fill_value(logits.dtype)

    assert log_probs[0, 1].item() == fill
    assert log_probs[1].tolist() == [fill, fill, fill]
    torch.testing.assert_close(
        torch.exp(log_probs[0, [0, 2]]).sum(),
        torch.tensor(1.0),
    )
    entropy = masked_entropy_from_log_probs(log_probs, mask)
    assert entropy[0].item() > 0.0
    assert entropy[1].item() == 0.0


def test_masked_log_softmax_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="matching shapes"):
        masked_log_softmax(torch.zeros((1, 2)), torch.ones((1, 3), dtype=torch.bool))


def test_factorized_row_lookup_and_scatter_validate_selected_rows() -> None:
    available_rows = torch.tensor([2, 4, 8], dtype=torch.long)
    selected_rows = torch.tensor([4, 8], dtype=torch.long)

    assert factorized_local_row_indices(available_rows, selected_rows).tolist() == [1, 2]
    assert factorized_local_row_indices(available_rows, torch.zeros((0,), dtype=torch.long)).tolist() == []

    with pytest.raises(ValueError, match="at least one available row"):
        factorized_local_row_indices(torch.zeros((0,), dtype=torch.long), torch.tensor([1], dtype=torch.long))
    with pytest.raises(ValueError, match="selected rows to be legal"):
        factorized_local_row_indices(available_rows, torch.tensor([3], dtype=torch.long))

    scattered = scatter_factorized_row_values(
        4,
        torch.tensor([1, 3], dtype=torch.long),
        torch.tensor([[10.0, 11.0], [30.0, 31.0]]),
        fill_value=-5.0,
    )

    torch.testing.assert_close(
        scattered,
        torch.tensor([[-5.0, -5.0], [10.0, 11.0], [-5.0, -5.0], [30.0, 31.0]]),
    )
