from __future__ import annotations

import torch
from weiss_rl.model import StructuredLegalPolicyValueModel, build_policy_value_model

from tests.weiss_rl.contracts_test_support import (
    _make_structured_joint_scorer_nonuniform,
    _structured_hand_observation_spec,
    _structured_hand_spec_bundle,
    _structured_model_config,
)


def test_structured_legal_policy_value_model_accepts_card_table_features() -> None:
    model = build_policy_value_model(
        observation_dim=8,
        config=_structured_model_config(),
        action_dim=5,
        observation_spec=_structured_hand_observation_spec(),
        spec_bundle=_structured_hand_spec_bundle(),
        card_table={
            "rows": [
                {
                    "card_id": 11,
                    "level": 1,
                    "cost": 0,
                    "power": 4500,
                    "soul": 1,
                    "color": "yellow",
                    "card_type": "character",
                    "traits": ["music"],
                }
            ]
        },
    )

    assert isinstance(model, StructuredLegalPolicyValueModel)
    assert model.policy_head._card_static_features.shape[1] > 0


def test_structured_v2_uses_hand_position_when_scoring_matching_cards() -> None:
    spec_bundle = _structured_hand_spec_bundle()
    model = build_policy_value_model(
        observation_dim=8,
        config=_structured_model_config(),
        action_dim=5,
        observation_spec=_structured_hand_observation_spec(),
        spec_bundle=spec_bundle,
    )
    assert isinstance(model, StructuredLegalPolicyValueModel)
    _make_structured_joint_scorer_nonuniform(model)

    obs_a = torch.zeros((1, 8), dtype=torch.float32)
    obs_a[0, 4] = 11
    obs_a[0, 5] = 22
    obs_b = torch.zeros((1, 8), dtype=torch.float32)
    obs_b[0, 4] = 22
    obs_b[0, 5] = 11

    logits_a, _value_a, _hidden_a = model(obs_a)
    logits_b, _value_b, _hidden_b = model(obs_b)

    assert not torch.isclose(logits_a[0, 0], logits_b[0, 2], atol=1e-6, rtol=0.0)


def test_structured_v2_uses_target_slot_context_when_scoring_play_actions() -> None:
    spec_bundle = _structured_hand_spec_bundle()
    model = build_policy_value_model(
        observation_dim=8,
        config=_structured_model_config(),
        action_dim=5,
        observation_spec=_structured_hand_observation_spec(),
        spec_bundle=spec_bundle,
    )
    assert isinstance(model, StructuredLegalPolicyValueModel)
    _make_structured_joint_scorer_nonuniform(model)

    obs_open = torch.zeros((1, 8), dtype=torch.float32)
    obs_open[0, 4] = 11
    obs_blocked = obs_open.clone()
    obs_blocked[0, 2] = 99

    logits_open, _value_open, _hidden_open = model(obs_open)
    logits_blocked, _value_blocked, _hidden_blocked = model(obs_blocked)

    assert not torch.isclose(logits_open[0, 0], logits_blocked[0, 0], atol=1e-6, rtol=1e-6)
