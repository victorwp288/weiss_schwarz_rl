from __future__ import annotations

import pytest
import torch

from .model_opponent_context_test_support import opponent_context_model


def test_trainable_opponent_context_adapter_is_persistent_and_zero_initialized() -> None:
    model = opponent_context_model(context_scale=0.0, trainable_context_scale=0.5)

    plain = model.initial_seat_hidden(1)
    conditioned = model.initial_seat_hidden(1, opponent_policy_ids=("B2 HeuristicPublic",))

    assert "opponent_context_hidden_adapter" in model.state_dict()
    assert torch.allclose(conditioned, plain)

    with torch.no_grad():
        model.opponent_context_hidden_adapter[1].fill_(2.0)
    conditioned_after_update = model.initial_seat_hidden(1, opponent_policy_ids=("B2 HeuristicPublic",))

    assert torch.allclose(conditioned_after_update[:, 0], torch.full_like(conditioned_after_update[:, 0], 1.0))
    assert torch.allclose(conditioned_after_update[:, 0], conditioned_after_update[:, 1])


def test_trainable_opponent_context_action_bias_is_persistent_and_contextual() -> None:
    model = opponent_context_model(trainable_action_bias_scale=2.0)

    assert "opponent_context_action_bias_adapter" in model.state_dict()

    obs = torch.zeros((2, 4), dtype=torch.float32)
    acting_seat = torch.zeros((2,), dtype=torch.long)
    hidden = model.initial_seat_hidden(2)
    plain_logits, _plain_value, _hidden = model.forward_seat_aware(obs, acting_seat, hidden)

    with torch.no_grad():
        model.opponent_context_action_bias_adapter[1, 3] = 1.5

    contextual_logits, _value, _hidden = model.forward_seat_aware(
        obs,
        acting_seat,
        hidden,
        opponent_context_index=torch.tensor([1, 0], dtype=torch.long),
    )

    assert float(contextual_logits[0, 3].detach()) == pytest.approx(float((plain_logits[0, 3] + 3.0).detach()))
    assert torch.allclose(contextual_logits[1], plain_logits[1])


def test_trainable_opponent_context_recurrent_adapter_affects_current_step() -> None:
    model = opponent_context_model(trainable_recurrent_scale=0.5)

    assert "opponent_context_recurrent_adapter" in model.state_dict()

    obs = torch.zeros((2, 4), dtype=torch.float32)
    acting_seat = torch.zeros((2,), dtype=torch.long)
    hidden = model.initial_seat_hidden(2)
    plain_logits, _plain_value, _hidden = model.forward_seat_aware(obs, acting_seat, hidden)

    with torch.no_grad():
        model.opponent_context_recurrent_adapter[1].fill_(2.0)

    contextual_logits, _value, _hidden = model.forward_seat_aware(
        obs,
        acting_seat,
        hidden,
        opponent_context_index=torch.tensor([1, 0], dtype=torch.long),
    )

    assert not torch.allclose(contextual_logits[0], plain_logits[0])
    assert torch.allclose(contextual_logits[1], plain_logits[1])
