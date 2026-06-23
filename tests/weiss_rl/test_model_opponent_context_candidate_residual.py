from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from .model_opponent_context_test_support import (
    candidate_residual_actions,
    opponent_context_model,
)


def _install_candidate_residual_layers(model, *, include_candidate: bool = False) -> None:
    model.opponent_context_candidate_residual_context = nn.Parameter(torch.zeros((3, 2)))
    model.opponent_context_candidate_residual_state = nn.Linear(2, 2, bias=False)
    if include_candidate:
        model.opponent_context_candidate_residual_candidate = nn.Linear(2, 2, bias=False)
    model.opponent_context_candidate_residual_meta = nn.Linear(3, 2, bias=False)
    model.opponent_context_candidate_residual_out = nn.Linear(2, 1, bias=False)


def test_opponent_context_candidate_residual_is_state_and_context_conditioned() -> None:
    model = opponent_context_model()
    model.opponent_context_trainable_candidate_residual_scale = 1.0
    _install_candidate_residual_layers(model)
    packed_logits = torch.zeros((4,), dtype=torch.float32)
    legal_actions = candidate_residual_actions()
    state_repr = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float32)
    with torch.no_grad():
        model.opponent_context_candidate_residual_context[1] = torch.tensor([0.5, 0.0])
        model.opponent_context_candidate_residual_state.weight.copy_(torch.eye(2))
        model.opponent_context_candidate_residual_meta.weight.zero_()
        model.opponent_context_candidate_residual_out.weight.copy_(torch.tensor([[1.0, 0.0]]))

    biased = model._apply_opponent_context_packed_candidate_residual(
        packed_logits,
        legal_actions,
        state_repr,
        torch.tensor([1, 0], dtype=torch.long),
    )

    assert float(biased[0].detach()) == pytest.approx(float(biased[1].detach()))
    assert float(biased[0].detach()) > 0.0
    assert biased[2:].tolist() == [0.0, 0.0]


def test_bilinear_candidate_residual_can_separate_contexts_without_initial_global_bias() -> None:
    model = opponent_context_model(
        trainable_candidate_residual_scale=1.0,
        candidate_residual_mode="bilinear",
    )
    _install_candidate_residual_layers(model)
    packed_logits = torch.zeros((4,), dtype=torch.float32)
    legal_actions = candidate_residual_actions()
    state_repr = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    with torch.no_grad():
        model.opponent_context_candidate_residual_state.weight.zero_()
        model.opponent_context_candidate_residual_meta.weight.copy_(
            torch.tensor(
                [
                    [0.0, 32.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            )
        )
        model.opponent_context_candidate_residual_context[1] = torch.tensor([1.0, 0.0])
        model.opponent_context_candidate_residual_context[2] = torch.tensor([-1.0, 0.0])

    biased = model._apply_opponent_context_packed_candidate_residual(
        packed_logits,
        legal_actions,
        state_repr,
        torch.tensor([1, 2], dtype=torch.long),
    )

    assert float(biased[0].detach()) < float(biased[1].detach())
    assert float(biased[2].detach()) > float(biased[3].detach())


def test_rich_candidate_residual_uses_projected_candidate_representations() -> None:
    model = opponent_context_model(
        trainable_candidate_residual_scale=1.0,
        candidate_residual_mode="rich",
    )
    _install_candidate_residual_layers(model, include_candidate=True)
    packed_logits = torch.zeros((4,), dtype=torch.float32)
    legal_actions = candidate_residual_actions()
    state_repr = torch.zeros((2, 2), dtype=torch.float32)

    def fake_project_candidate_representations(
        _state_repr: torch.Tensor,
        _legal_actions: object,
        _observation_context: object,
        *,
        scoring_mode: str = "auto",
    ) -> torch.Tensor:
        assert scoring_mode == "learner"
        return torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 0.0], [2.0, 0.0]], dtype=torch.float32)

    model.policy_head._project_packed_candidate_representations = fake_project_candidate_representations
    with torch.no_grad():
        model.opponent_context_candidate_residual_context[1].zero_()
        model.opponent_context_candidate_residual_state.weight.zero_()
        model.opponent_context_candidate_residual_candidate.weight.copy_(torch.eye(2))
        model.opponent_context_candidate_residual_meta.weight.zero_()
        model.opponent_context_candidate_residual_out.weight.copy_(torch.tensor([[1.0, 0.0]]))

    biased = model._apply_opponent_context_packed_candidate_residual(
        packed_logits,
        legal_actions,
        state_repr,
        torch.tensor([1, 0], dtype=torch.long),
        observation_context={},
        scoring_mode="learner",
    )

    assert float(biased[1].detach()) > float(biased[0].detach())
    assert biased[2:].tolist() == [0.0, 0.0]


def test_rich_bilinear_candidate_residual_lets_context_weight_candidate_features() -> None:
    model = opponent_context_model(
        trainable_candidate_residual_scale=1.0,
        candidate_residual_mode="rich_bilinear",
    )
    _install_candidate_residual_layers(model, include_candidate=True)
    packed_logits = torch.zeros((4,), dtype=torch.float32)
    legal_actions = candidate_residual_actions(meta=torch.zeros((4, 3), dtype=torch.long))
    state_repr = torch.zeros((2, 2), dtype=torch.float32)

    def fake_project_candidate_representations(
        _state_repr: torch.Tensor,
        _legal_actions: object,
        _observation_context: object,
        *,
        scoring_mode: str = "auto",
    ) -> torch.Tensor:
        assert scoring_mode == "learner"
        return torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

    model.policy_head._project_packed_candidate_representations = fake_project_candidate_representations
    with torch.no_grad():
        model.opponent_context_candidate_residual_context[1] = torch.tensor([1.0, -1.0])
        model.opponent_context_candidate_residual_context[2] = torch.tensor([-1.0, 1.0])
        model.opponent_context_candidate_residual_state.weight.zero_()
        model.opponent_context_candidate_residual_candidate.weight.copy_(torch.eye(2))
        model.opponent_context_candidate_residual_meta.weight.zero_()
        model.opponent_context_candidate_residual_out.weight.zero_()

    biased = model._apply_opponent_context_packed_candidate_residual(
        packed_logits,
        legal_actions,
        state_repr,
        torch.tensor([1, 2], dtype=torch.long),
        observation_context={},
        scoring_mode="learner",
    )

    assert float(biased[0].detach()) > float(biased[1].detach())
    assert float(biased[2].detach()) < float(biased[3].detach())


def test_candidate_residual_action_id_allowlist_masks_non_target_actions() -> None:
    model = opponent_context_model(
        action_dim=130,
        trainable_candidate_residual_scale=1.0,
        candidate_residual_action_ids=(124,),
    )
    _install_candidate_residual_layers(model)
    packed_logits = torch.zeros((4,), dtype=torch.float32)
    legal_actions = SimpleNamespace(
        ids=torch.tensor([104, 124, 108, 124], dtype=torch.long),
        offsets=torch.tensor([0, 2, 4], dtype=torch.long),
        meta=torch.tensor(
            [
                [2, 1, 0],
                [2, 2, 0],
                [2, 3, 0],
                [2, 4, 0],
            ],
            dtype=torch.long,
        ),
    )
    state_repr = torch.ones((2, 2), dtype=torch.float32)
    with torch.no_grad():
        model.opponent_context_candidate_residual_context.fill_(1.0)
        model.opponent_context_candidate_residual_state.weight.zero_()
        model.opponent_context_candidate_residual_meta.weight.zero_()
        model.opponent_context_candidate_residual_out.weight.fill_(1.0)

    biased = model._apply_opponent_context_packed_candidate_residual(
        packed_logits,
        legal_actions,
        state_repr,
        torch.tensor([1, 1], dtype=torch.long),
    )

    assert biased[0].item() == pytest.approx(0.0)
    assert biased[2].item() == pytest.approx(0.0)
    assert biased[1].item() > 0.0
    assert biased[3].item() > 0.0
