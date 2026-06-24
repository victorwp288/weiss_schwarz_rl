from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn
from weiss_rl.learners.factorized_evaluation import ImpalaFactorizedEvaluationMixin
from weiss_rl.learners.factorized_public_teacher import _attach_initial_hidden_context_gradient


def test_factorized_candidate_helper_uses_contextual_packed_scorer_for_gradients() -> None:
    helper = ImpalaFactorizedEvaluationMixin()
    model = _ContextualFactorizedModel()
    legal_actions = SimpleNamespace(
        ids=torch.tensor([104, 124], dtype=torch.long),
        offsets=torch.tensor([0, 2], dtype=torch.long),
        meta=torch.zeros((2, 3), dtype=torch.long),
    )

    log_probs = helper._factorized_packed_candidate_log_probs(
        model,
        recurrent_flat=torch.zeros((1, 3), dtype=torch.float32),
        obs_rows=torch.zeros((1, 4), dtype=torch.float32),
        legal_actions=legal_actions,
        state_repr=torch.ones((1, 2), dtype=torch.float32),
        observation_context={},
        opponent_context_index=torch.tensor([1], dtype=torch.long),
    )
    loss = log_probs[1] - log_probs[0]
    loss.backward()

    assert model.contextual_called
    assert model.context_weight.grad is not None
    assert float(model.context_weight.grad.detach()) != 0.0


def test_initial_hidden_context_gradient_is_reattached_without_changing_values() -> None:
    model = _HiddenContextModel()
    hidden = torch.zeros((2, 2, 3), dtype=torch.float32)
    context_index = torch.tensor([[1, 0], [1, 0]], dtype=torch.long)

    attached = _attach_initial_hidden_context_gradient(model, hidden, context_index)

    assert torch.allclose(attached.detach(), hidden)
    attached.sum().backward()
    assert model.opponent_context_hidden_adapter.grad is not None
    assert torch.allclose(model.opponent_context_hidden_adapter.grad[0], torch.zeros(3))
    assert torch.allclose(model.opponent_context_hidden_adapter.grad[1], torch.full((3,), 2.0))


class _ContextualFactorizedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.contextual_called = False
        self.context_weight = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
        self.policy_head = SimpleNamespace(factorized_packed_action_log_probs=self._base_log_probs)

    def _base_log_probs(self, *args: object, legal_actions: object, **kwargs: object) -> torch.Tensor:
        del args, kwargs
        return torch.zeros_like(torch.as_tensor(legal_actions.ids), dtype=torch.float32)

    def _factorized_packed_action_log_probs_with_context(
        self,
        recurrent_output: torch.Tensor,
        *,
        obs: torch.Tensor,
        legal_actions: object,
        state_repr: torch.Tensor,
        observation_context: dict[str, torch.Tensor],
        scoring_mode: str = "auto",
        opponent_context_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del recurrent_output, obs, state_repr, observation_context, scoring_mode
        self.contextual_called = True
        ids = torch.as_tensor(legal_actions.ids, dtype=torch.long)
        context_scale = (
            torch.tensor(0.0, dtype=torch.float32)
            if opponent_context_index is None
            else torch.as_tensor(opponent_context_index, dtype=torch.float32).reshape(-1)[0]
        )
        direction = torch.where(ids == 104, torch.tensor(1.0), torch.tensor(-1.0))
        return self._base_log_probs(legal_actions=legal_actions) + self.context_weight * context_scale * direction


class _HiddenContextModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden_size = 3
        self.opponent_context_hidden_adapter = nn.Parameter(torch.zeros((2, 3), dtype=torch.float32))
        self.opponent_context_trainable_hidden_scale = 1.0

    def _opponent_context_hidden(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        opponent_policy_ids: object | None,
        opponent_context_indices: torch.Tensor,
    ) -> torch.Tensor:
        del opponent_policy_ids
        indices = opponent_context_indices.to(device=device, dtype=torch.long).reshape(-1)
        assert int(indices.numel()) == int(batch_size)
        context = self.opponent_context_hidden_adapter.to(device=device, dtype=dtype).index_select(0, indices)
        return context.masked_fill((indices == 0).unsqueeze(1), 0.0)
