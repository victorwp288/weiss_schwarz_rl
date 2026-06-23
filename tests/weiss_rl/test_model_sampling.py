from __future__ import annotations

import pytest
import torch
from weiss_rl.model import _masked_log_softmax, _sample_masked_log_probs, _sample_packed_action_scores
from weiss_rl.models.sampling import sample_masked_log_probs, sample_packed_action_scores


def test_sample_masked_log_probs_is_seeded_and_preserves_empty_row_default() -> None:
    logits = torch.tensor([[0.0, 1.0, 2.0], [5.0, 6.0, 7.0], [1.0, 2.0, 3.0]])
    mask = torch.tensor([[True, True, True], [False, False, False], [False, True, True]])
    log_probs = _masked_log_softmax(logits, mask)

    actions, selected_logp = sample_masked_log_probs(
        log_probs,
        mask,
        sample_seeds=torch.tensor([1, 2, 3]),
        default_index=9,
    )

    assert actions.tolist() == [2, 9, 2]
    torch.testing.assert_close(selected_logp, torch.tensor([-0.4076, 0.0, -0.3133]), atol=1e-4, rtol=1e-4)


def test_sample_packed_action_scores_is_seeded_and_returns_selected_logp() -> None:
    actions, selected_logp = sample_packed_action_scores(
        torch.tensor([0.0, 1.0, 5.0, -1.0, 0.0, 1.0]),
        torch.tensor([10, 11, 20, 30, 31, 32]),
        torch.tensor([0, 2, 3, 6]),
        torch.tensor([1, 2, 3]),
        pass_action_id=99,
    )

    assert actions.tolist() == [11, 20, 32]
    torch.testing.assert_close(selected_logp, torch.tensor([-0.3133, 0.0, -0.4076]), atol=1e-4, rtol=1e-4)


def test_sample_packed_action_scores_temperature_sharpens_behavior_logp() -> None:
    scores = torch.tensor([0.0, 2.0])
    packed_ids = torch.tensor([10, 11])
    offsets = torch.tensor([0, 2])
    seeds = torch.tensor([1])

    actions_hot, logp_hot = sample_packed_action_scores(scores, packed_ids, offsets, seeds, pass_action_id=99)
    actions_cold, logp_cold = sample_packed_action_scores(
        scores,
        packed_ids,
        offsets,
        seeds,
        pass_action_id=99,
        temperature=0.5,
    )

    assert actions_hot.tolist() == [11]
    assert actions_cold.tolist() == [11]
    assert float(logp_cold.item()) > float(logp_hot.item())


def test_sample_masked_log_probs_temperature_sharpens_behavior_logp() -> None:
    logits = torch.tensor([[0.0, 2.0]])
    mask = torch.tensor([[True, True]])
    log_probs = _masked_log_softmax(logits, mask)

    actions_hot, logp_hot = sample_masked_log_probs(log_probs, mask, sample_seeds=torch.tensor([1]))
    actions_cold, logp_cold = sample_masked_log_probs(
        log_probs,
        mask,
        sample_seeds=torch.tensor([1]),
        temperature=0.5,
    )

    assert actions_hot.tolist() == [1]
    assert actions_cold.tolist() == [1]
    assert float(logp_cold.item()) > float(logp_hot.item())


def test_sample_packed_action_scores_tiny_thresholds_choose_first_candidate_not_fallback() -> None:
    row_count = 20_000
    row_length = 11
    scores = torch.zeros(row_count * row_length)
    row_ids = torch.arange(row_length, dtype=torch.long)
    row_ids[-1] = 51
    packed_ids = row_ids.repeat(row_count)
    offsets = torch.arange(row_count + 1, dtype=torch.long) * row_length

    actions, _selected_logp = sample_packed_action_scores(
        scores,
        packed_ids,
        offsets,
        torch.arange(row_count, dtype=torch.long),
        pass_action_id=51,
        uniform_from_seeds_fn=lambda seeds: torch.full_like(seeds, torch.finfo(torch.float32).eps, dtype=torch.float32),
    )

    assert actions.unique().tolist() == [0]


def test_sample_packed_action_scores_uniform_logits_do_not_overselect_last_candidate() -> None:
    row_count = 20_000
    row_length = 11
    scores = torch.zeros(row_count * row_length)
    row_ids = torch.arange(row_length, dtype=torch.long)
    row_ids[-1] = 51
    packed_ids = row_ids.repeat(row_count)
    offsets = torch.arange(row_count + 1, dtype=torch.long) * row_length

    actions, selected_logp = sample_packed_action_scores(
        scores,
        packed_ids,
        offsets,
        torch.arange(row_count, dtype=torch.long),
        pass_action_id=51,
    )

    pass_fraction = float((actions == 51).to(dtype=torch.float32).mean().item())
    assert 0.075 <= pass_fraction <= 0.11
    torch.testing.assert_close(selected_logp, torch.full((row_count,), -torch.log(torch.tensor(float(row_length)))))


def test_sample_packed_action_scores_falls_back_to_last_candidate_when_cdf_undershoots() -> None:
    actions, logp = sample_packed_action_scores(
        torch.tensor([0.0, -0.1, -0.2], dtype=torch.float32),
        torch.tensor([4, 7, 9], dtype=torch.long),
        torch.tensor([0, 3], dtype=torch.long),
        torch.tensor([123], dtype=torch.long),
        pass_action_id=51,
        packed_local_cdf_fn=lambda probabilities, offsets: torch.tensor(
            [0.4, 0.8, 0.9999999],
            dtype=probabilities.dtype,
            device=probabilities.device,
        ),
        uniform_from_seeds_fn=lambda seeds: torch.tensor([0.99999994], dtype=torch.float32, device=seeds.device),
    )

    assert actions.tolist() == [9]
    assert torch.isfinite(logp).all()


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (
            (torch.zeros((1, 2, 1)), torch.ones((1, 2), dtype=torch.bool)),
            "requires 2D tensors",
        ),
        (
            (torch.zeros((1, 2)), torch.ones((1, 2), dtype=torch.bool)),
            r"sample_seeds must have shape \(1,\)",
        ),
    ],
)
def test_sample_masked_log_probs_validates_shapes(args, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        sample_masked_log_probs(*args, sample_seeds=torch.tensor([1, 2]), default_index=0)


def test_sample_packed_action_scores_validates_shapes() -> None:
    with pytest.raises(ValueError, match="packed_scores must be 1D"):
        sample_packed_action_scores(
            torch.zeros((1, 2)),
            torch.zeros((2,), dtype=torch.long),
            torch.tensor([0, 2]),
            torch.tensor([1]),
            pass_action_id=0,
        )
    with pytest.raises(ValueError, match="packed offsets must describe"):
        sample_packed_action_scores(
            torch.zeros((2,)),
            torch.zeros((2,), dtype=torch.long),
            torch.tensor([0, 3]),
            torch.tensor([1]),
            pass_action_id=0,
        )


def test_model_private_sampling_wrappers_are_preserved() -> None:
    assert _sample_masked_log_probs is not sample_masked_log_probs
    assert _sample_packed_action_scores is not sample_packed_action_scores

    logits = torch.tensor([[0.0, 1.0]])
    mask = torch.tensor([[True, True]])
    log_probs = _masked_log_softmax(logits, mask)
    public_actions, public_logp = sample_masked_log_probs(log_probs, mask, sample_seeds=torch.tensor([1]))
    private_actions, private_logp = _sample_masked_log_probs(log_probs, mask, sample_seeds=torch.tensor([1]))

    torch.testing.assert_close(private_actions, public_actions)
    torch.testing.assert_close(private_logp, public_logp)
