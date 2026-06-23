from __future__ import annotations

import torch

from .model_opponent_context_test_support import opponent_context_model


def test_opponent_context_initial_hidden_is_opt_in_and_nonpersistent() -> None:
    model = opponent_context_model(context_scale=0.75)

    plain = model.initial_seat_hidden(2)
    conditioned = model.initial_seat_hidden(
        2,
        opponent_policy_ids=("B2 HeuristicPublic", "seed_x_seed_c3aac2f9dc_policy_000004"),
    )

    assert torch.allclose(plain, torch.zeros_like(plain))
    assert not torch.allclose(conditioned, plain)
    assert torch.allclose(conditioned[:, 0], conditioned[:, 1])
    assert "opponent_context_hidden_offsets" not in "\n".join(model.state_dict().keys())


def test_opponent_context_indices_support_imported_seed_suffixes_and_eval_gate() -> None:
    model = opponent_context_model(context_scale=0.75)

    indices = model.opponent_context_indices_for_policy_ids(
        (
            "seed_run_seed_c3aac2f9dc_policy_000004",
            "B2 HeuristicPublic",
            "unmapped_policy",
        )
    )

    assert indices == [2, 1, 0]
    assert model.should_apply_opponent_context_for_eval_policy("policy_000001")
    assert not model.should_apply_opponent_context_for_eval_policy("policy_000999")


def test_sequence_reset_can_reseed_hidden_from_opponent_context_index() -> None:
    model = opponent_context_model(context_scale=0.75)
    obs = torch.zeros((2, 1, 4), dtype=torch.float32)
    acting_seat = torch.zeros((2, 1), dtype=torch.long)
    reset_before_step = torch.tensor([[False], [True]])

    _logits, _values, contextual_hidden = model.forward_sequence_seat_aware(
        obs,
        acting_seat,
        model.initial_seat_hidden(1),
        reset_before_step=reset_before_step,
        opponent_context_index=torch.tensor([[0], [1]], dtype=torch.long),
    )
    _logits, _values, plain_hidden = model.forward_sequence_seat_aware(
        obs,
        acting_seat,
        model.initial_seat_hidden(1),
        reset_before_step=reset_before_step,
        opponent_context_index=torch.tensor([[0], [0]], dtype=torch.long),
    )

    assert not torch.allclose(contextual_hidden, plain_hidden)
