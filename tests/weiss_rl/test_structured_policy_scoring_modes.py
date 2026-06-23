from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.model import StructuredLegalPolicyValueModel, build_policy_value_model

from tests.weiss_rl.contracts_test_support import (
    _packed_meta_from_ids,
    _structured_hand_observation_spec,
    _structured_hand_spec_bundle,
    _structured_model_config,
)


def test_structured_legal_policy_value_model_can_split_actor_and_learner_public_bias() -> None:
    torch.manual_seed(0)
    model = build_policy_value_model(
        observation_dim=8,
        config=replace(
            _structured_model_config(),
            public_heuristic_logit_bias_scale=1.0,
            public_heuristic_actor_logit_bias_scale=0.0,
        ),
        action_dim=5,
        observation_spec=_structured_hand_observation_spec(),
        spec_bundle=_structured_hand_spec_bundle(),
    )
    assert isinstance(model, StructuredLegalPolicyValueModel)

    obs = torch.zeros((1, 8), dtype=torch.float32)
    obs[0, 4] = 11
    obs[0, 5] = 11
    acting_seat = torch.tensor([0], dtype=torch.long)
    seat_hidden = model.initial_seat_hidden(1)
    packed_ids = np.asarray([0, 1, 2, 3, 4], dtype=np.int32)
    packed_offsets = np.asarray([0, 5], dtype=np.int32)
    packed_meta = _packed_meta_from_ids(ActionCatalog.from_spec_bundle(_structured_hand_spec_bundle()), packed_ids)
    legal_actions = LegalActionBatch.from_packed(
        packed_ids,
        packed_offsets,
        meta=packed_meta,
        action_space=5,
    )

    recurrent_output, state_repr, observation_context, _value, _next_hidden = model.forward_trunk_packed_seat_aware(
        obs,
        acting_seat,
        seat_hidden,
    )

    learner_scores = model.score_packed_legal_candidates(
        recurrent_output,
        obs,
        legal_actions,
        state_repr=state_repr,
        observation_context=observation_context,
        scoring_mode="learner",
    )
    with torch.no_grad():
        actor_scores = model.score_packed_legal_candidates(
            recurrent_output,
            obs,
            legal_actions,
            state_repr=state_repr,
            observation_context=observation_context,
            scoring_mode="actor",
        )

    assert torch.max(torch.abs(learner_scores - actor_scores)).item() > 1e-4


def test_structured_forward_seat_aware_respects_explicit_scoring_mode() -> None:
    torch.manual_seed(0)
    model = build_policy_value_model(
        observation_dim=8,
        config=replace(
            _structured_model_config(),
            public_heuristic_logit_bias_scale=0.0,
            public_heuristic_actor_logit_bias_scale=100.0,
        ),
        action_dim=5,
        observation_spec=_structured_hand_observation_spec(),
        spec_bundle=_structured_hand_spec_bundle(),
    )
    assert isinstance(model, StructuredLegalPolicyValueModel)

    obs = torch.zeros((1, 8), dtype=torch.float32)
    obs[0, 4] = 11
    obs[0, 5] = 11
    acting_seat = torch.tensor([0], dtype=torch.long)
    seat_hidden = model.initial_seat_hidden(1)
    packed_ids = np.asarray([0, 1, 2, 3, 4], dtype=np.int32)
    packed_offsets = np.asarray([0, 5], dtype=np.int32)
    packed_meta = _packed_meta_from_ids(ActionCatalog.from_spec_bundle(_structured_hand_spec_bundle()), packed_ids)
    legal_actions = LegalActionBatch.from_packed(
        packed_ids,
        packed_offsets,
        meta=packed_meta,
        action_space=5,
    )

    with torch.inference_mode():
        learner_logits, _learner_value, _learner_next_hidden = model.forward_seat_aware(
            obs,
            acting_seat,
            seat_hidden,
            legal_actions=legal_actions,
            scoring_mode="learner",
        )
        actor_logits, _actor_value, _actor_next_hidden = model.forward_seat_aware(
            obs,
            acting_seat,
            seat_hidden,
            legal_actions=legal_actions,
            scoring_mode="actor",
        )

    assert torch.max(torch.abs(learner_logits - actor_logits)).item() > 1e-4
