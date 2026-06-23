from __future__ import annotations

from types import SimpleNamespace

import torch
from weiss_rl.core.legal_actions import LegalActionBatch

from .model_opponent_context_test_support import opponent_context_model


def test_trainable_opponent_context_action_bias_applies_to_packed_candidates() -> None:
    model = opponent_context_model(trainable_action_bias_scale=1.0)
    packed_logits = torch.zeros((4,), dtype=torch.float32)
    legal_actions = LegalActionBatch.from_packed(
        [1, 3, 2, 4],
        [0, 2, 4],
        action_space=5,
    )

    with torch.no_grad():
        model.opponent_context_action_bias_adapter[1, 3] = 2.0
        model.opponent_context_action_bias_adapter[2, 4] = -1.0

    biased = model._apply_opponent_context_packed_action_bias(
        packed_logits,
        legal_actions,
        torch.tensor([1, 2], dtype=torch.long),
    )

    assert biased.tolist() == [0.0, 2.0, 0.0, -1.0]


def test_trainable_opponent_context_action_bias_accepts_tensor_packed_offsets() -> None:
    model = opponent_context_model(trainable_action_bias_scale=1.0)
    packed_logits = torch.zeros((4,), dtype=torch.float32)
    legal_actions = SimpleNamespace(
        ids=torch.tensor([1, 3, 2, 4], dtype=torch.long),
        offsets=torch.tensor([0, 2, 4], dtype=torch.long),
        meta=None,
    )

    with torch.no_grad():
        model.opponent_context_action_bias_adapter[1, 3] = 2.0
        model.opponent_context_action_bias_adapter[2, 4] = -1.0

    biased = model._apply_opponent_context_packed_action_bias(
        packed_logits,
        legal_actions,
        torch.tensor([1, 2], dtype=torch.long),
    )

    assert biased.tolist() == [0.0, 2.0, 0.0, -1.0]
