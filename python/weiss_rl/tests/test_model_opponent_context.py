from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from weiss_rl.config.models import ModelConfig, ModelDropoutConfig
from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.model import build_policy_value_model


def _config(
    *,
    context_scale: float = 0.0,
    trainable_context_scale: float = 0.0,
    trainable_recurrent_scale: float = 0.0,
    trainable_action_bias_scale: float = 0.0,
    trainable_candidate_residual_scale: float = 0.0,
    candidate_residual_mode: str = "additive",
    candidate_residual_action_ids: tuple[int, ...] = (),
) -> ModelConfig:
    return ModelConfig(
        gru_hidden_size=16,
        encoder_mlp_width=8,
        encoder_mlp_layers=1,
        layer_norm=False,
        dropout=ModelDropoutConfig(family_a=0.0, ablation=0.0),
        opponent_context_policy_ids=("B2 HeuristicPublic", "seed_c3aac2f9dc_policy_000004"),
        opponent_context_hidden_scale=context_scale,
        opponent_context_trainable_hidden_scale=trainable_context_scale,
        opponent_context_trainable_recurrent_scale=trainable_recurrent_scale,
        opponent_context_trainable_action_bias_scale=trainable_action_bias_scale,
        opponent_context_trainable_candidate_residual_scale=trainable_candidate_residual_scale,
        opponent_context_candidate_residual_mode=candidate_residual_mode,
        opponent_context_candidate_residual_action_ids=candidate_residual_action_ids,
        opponent_context_eval_policy_ids=("policy_000001",),
    )


def test_opponent_context_initial_hidden_is_opt_in_and_nonpersistent() -> None:
    model = build_policy_value_model(observation_dim=4, config=_config(context_scale=0.75), action_dim=5)

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
    model = build_policy_value_model(observation_dim=4, config=_config(context_scale=0.75), action_dim=5)

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
    model = build_policy_value_model(observation_dim=4, config=_config(context_scale=0.75), action_dim=5)
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


def test_trainable_opponent_context_adapter_is_persistent_and_zero_initialized() -> None:
    model = build_policy_value_model(
        observation_dim=4,
        config=_config(context_scale=0.0, trainable_context_scale=0.5),
        action_dim=5,
    )

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
    model = build_policy_value_model(
        observation_dim=4,
        config=_config(trainable_action_bias_scale=2.0),
        action_dim=5,
    )

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
    model = build_policy_value_model(
        observation_dim=4,
        config=_config(trainable_recurrent_scale=0.5),
        action_dim=5,
    )

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


def test_trainable_opponent_context_action_bias_applies_to_packed_candidates() -> None:
    model = build_policy_value_model(
        observation_dim=4,
        config=_config(trainable_action_bias_scale=1.0),
        action_dim=5,
    )
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
    model = build_policy_value_model(
        observation_dim=4,
        config=_config(trainable_action_bias_scale=1.0),
        action_dim=5,
    )
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


def test_opponent_context_candidate_residual_is_state_and_context_conditioned() -> None:
    model = build_policy_value_model(observation_dim=4, config=_config(), action_dim=5)
    model.opponent_context_trainable_candidate_residual_scale = 1.0
    model.opponent_context_candidate_residual_context = nn.Parameter(torch.zeros((3, 2)))
    model.opponent_context_candidate_residual_state = nn.Linear(2, 2, bias=False)
    model.opponent_context_candidate_residual_meta = nn.Linear(3, 2, bias=False)
    model.opponent_context_candidate_residual_out = nn.Linear(2, 1, bias=False)
    packed_logits = torch.zeros((4,), dtype=torch.float32)
    legal_actions = SimpleNamespace(
        ids=torch.tensor([104, 124, 104, 124], dtype=torch.long),
        offsets=torch.tensor([0, 2, 4], dtype=torch.long),
        meta=torch.tensor(
            [
                [2, 1, 0],
                [2, 2, 0],
                [2, 1, 0],
                [2, 2, 0],
            ],
            dtype=torch.long,
        ),
    )
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
    model = build_policy_value_model(
        observation_dim=4,
        config=_config(
            trainable_candidate_residual_scale=1.0,
            candidate_residual_mode="bilinear",
        ),
        action_dim=5,
    )
    model.opponent_context_candidate_residual_context = nn.Parameter(torch.zeros((3, 2)))
    model.opponent_context_candidate_residual_state = nn.Linear(2, 2, bias=False)
    model.opponent_context_candidate_residual_meta = nn.Linear(3, 2, bias=False)
    model.opponent_context_candidate_residual_out = nn.Linear(2, 1, bias=False)
    packed_logits = torch.zeros((4,), dtype=torch.float32)
    legal_actions = SimpleNamespace(
        ids=torch.tensor([104, 124, 104, 124], dtype=torch.long),
        offsets=torch.tensor([0, 2, 4], dtype=torch.long),
        meta=torch.tensor(
            [
                [2, 1, 0],
                [2, 2, 0],
                [2, 1, 0],
                [2, 2, 0],
            ],
            dtype=torch.long,
        ),
    )
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
    model = build_policy_value_model(
        observation_dim=4,
        config=_config(
            trainable_candidate_residual_scale=1.0,
            candidate_residual_mode="rich",
        ),
        action_dim=5,
    )
    model.opponent_context_candidate_residual_context = nn.Parameter(torch.zeros((3, 2)))
    model.opponent_context_candidate_residual_state = nn.Linear(2, 2, bias=False)
    model.opponent_context_candidate_residual_candidate = nn.Linear(2, 2, bias=False)
    model.opponent_context_candidate_residual_meta = nn.Linear(3, 2, bias=False)
    model.opponent_context_candidate_residual_out = nn.Linear(2, 1, bias=False)
    packed_logits = torch.zeros((4,), dtype=torch.float32)
    legal_actions = SimpleNamespace(
        ids=torch.tensor([104, 124, 104, 124], dtype=torch.long),
        offsets=torch.tensor([0, 2, 4], dtype=torch.long),
        meta=torch.tensor(
            [
                [2, 1, 0],
                [2, 2, 0],
                [2, 1, 0],
                [2, 2, 0],
            ],
            dtype=torch.long,
        ),
    )
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
    model = build_policy_value_model(
        observation_dim=4,
        config=_config(
            trainable_candidate_residual_scale=1.0,
            candidate_residual_mode="rich_bilinear",
        ),
        action_dim=5,
    )
    model.opponent_context_candidate_residual_context = nn.Parameter(torch.zeros((3, 2)))
    model.opponent_context_candidate_residual_state = nn.Linear(2, 2, bias=False)
    model.opponent_context_candidate_residual_candidate = nn.Linear(2, 2, bias=False)
    model.opponent_context_candidate_residual_meta = nn.Linear(3, 2, bias=False)
    model.opponent_context_candidate_residual_out = nn.Linear(2, 1, bias=False)
    packed_logits = torch.zeros((4,), dtype=torch.float32)
    legal_actions = SimpleNamespace(
        ids=torch.tensor([104, 124, 104, 124], dtype=torch.long),
        offsets=torch.tensor([0, 2, 4], dtype=torch.long),
        meta=torch.zeros((4, 3), dtype=torch.long),
    )
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
    model = build_policy_value_model(
        observation_dim=4,
        config=_config(
            trainable_candidate_residual_scale=1.0,
            candidate_residual_action_ids=(124,),
        ),
        action_dim=130,
    )
    model.opponent_context_candidate_residual_context = nn.Parameter(torch.ones((3, 2)))
    model.opponent_context_candidate_residual_state = nn.Linear(2, 2, bias=False)
    model.opponent_context_candidate_residual_meta = nn.Linear(3, 2, bias=False)
    model.opponent_context_candidate_residual_out = nn.Linear(2, 1, bias=False)
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
