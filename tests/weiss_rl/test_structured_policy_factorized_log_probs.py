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


def test_structured_legal_policy_value_model_factorized_starts_uniform_over_legal_candidates() -> None:
    torch.manual_seed(0)
    model = build_policy_value_model(
        observation_dim=8,
        config=replace(_structured_model_config(), structured_policy_contract="factorized_v1"),
        action_dim=5,
        observation_spec=_structured_hand_observation_spec(),
        spec_bundle=_structured_hand_spec_bundle(),
    )
    assert isinstance(model, StructuredLegalPolicyValueModel)

    obs = torch.zeros((2, 8), dtype=torch.float32)
    obs[0, 4] = 11
    obs[1, 5] = 22
    acting_seat = torch.tensor([0, 1], dtype=torch.long)
    seat_hidden = model.initial_seat_hidden(2)
    packed_ids = np.asarray([0, 1, 4, 2, 3, 4], dtype=np.int32)
    packed_offsets = np.asarray([0, 3, 6], dtype=np.int32)
    packed_meta = _packed_meta_from_ids(ActionCatalog.from_spec_bundle(_structured_hand_spec_bundle()), packed_ids)
    legal_actions = LegalActionBatch.from_packed(
        packed_ids,
        packed_offsets,
        meta=packed_meta,
        action_space=5,
    )

    factorized_eval = model.evaluate_factorized_sequence_packed_seat_aware(
        obs.unsqueeze(0),
        acting_seat.unsqueeze(0),
        seat_hidden,
        legal_actions=legal_actions,
    )
    candidate_log_probs, _values, _next_hidden = model.factorized_packed_action_log_probs_seat_aware(
        obs,
        acting_seat,
        seat_hidden,
        legal_actions=legal_actions,
    )

    expected_play_family_log_prob = torch.full((2,), torch.log(torch.tensor(2.0 / 3.0)))
    expected_pass_family_log_prob = torch.full((2,), torch.log(torch.tensor(1.0 / 3.0)))
    expected_candidate_log_probs = torch.full((6,), -torch.log(torch.tensor(3.0)))
    torch.testing.assert_close(factorized_eval.family_log_probs[0, :, 0], expected_play_family_log_prob)
    torch.testing.assert_close(factorized_eval.family_log_probs[0, :, 1], expected_pass_family_log_prob)
    torch.testing.assert_close(candidate_log_probs, expected_candidate_log_probs)


def test_structured_legal_policy_value_model_factorized_candidate_log_probs_apply_public_bias() -> None:
    torch.manual_seed(0)
    model = build_policy_value_model(
        observation_dim=8,
        config=replace(_structured_model_config(), structured_policy_contract="factorized_v1"),
        action_dim=5,
        observation_spec=_structured_hand_observation_spec(),
        spec_bundle=_structured_hand_spec_bundle(),
    )
    assert isinstance(model, StructuredLegalPolicyValueModel)

    obs = torch.zeros((1, 8), dtype=torch.float32)
    obs[0, 4] = 11
    obs[0, 5] = 22
    acting_seat = torch.tensor([0], dtype=torch.long)
    seat_hidden = model.initial_seat_hidden(1)
    packed_ids = np.asarray([0, 1, 4], dtype=np.int32)
    packed_offsets = np.asarray([0, 3], dtype=np.int32)
    packed_meta = _packed_meta_from_ids(ActionCatalog.from_spec_bundle(_structured_hand_spec_bundle()), packed_ids)
    legal_actions = LegalActionBatch.from_packed(
        packed_ids,
        packed_offsets,
        meta=packed_meta,
        action_space=5,
    )

    unbiased, _values, _next_hidden = model.factorized_packed_action_log_probs_seat_aware(
        obs,
        acting_seat,
        seat_hidden,
        legal_actions=legal_actions,
        scoring_mode="learner",
    )
    model.set_public_heuristic_logit_bias_scale(2.0)
    biased, _values, _next_hidden = model.factorized_packed_action_log_probs_seat_aware(
        obs,
        acting_seat,
        seat_hidden,
        legal_actions=legal_actions,
        scoring_mode="learner",
    )

    assert not torch.allclose(biased, unbiased)
    torch.testing.assert_close(torch.logsumexp(biased, dim=0), torch.tensor(0.0))


def test_structured_legal_policy_value_model_factorized_candidate_log_probs_apply_opponent_context_bias() -> None:
    torch.manual_seed(0)
    model = build_policy_value_model(
        observation_dim=8,
        config=replace(
            _structured_model_config(),
            structured_policy_contract="factorized_v1",
            opponent_context_policy_ids=("B2 HeuristicPublic",),
            opponent_context_trainable_action_bias_scale=1.0,
        ),
        action_dim=5,
        observation_spec=_structured_hand_observation_spec(),
        spec_bundle=_structured_hand_spec_bundle(),
    )
    assert isinstance(model, StructuredLegalPolicyValueModel)

    obs = torch.zeros((1, 8), dtype=torch.float32)
    obs[0, 4] = 11
    obs[0, 5] = 22
    acting_seat = torch.tensor([0], dtype=torch.long)
    seat_hidden = model.initial_seat_hidden(1)
    packed_ids = np.asarray([0, 1, 4], dtype=np.int32)
    packed_offsets = np.asarray([0, 3], dtype=np.int32)
    packed_meta = _packed_meta_from_ids(ActionCatalog.from_spec_bundle(_structured_hand_spec_bundle()), packed_ids)
    legal_actions = LegalActionBatch.from_packed(
        packed_ids,
        packed_offsets,
        meta=packed_meta,
        action_space=5,
    )

    with torch.no_grad():
        model.opponent_context_action_bias_adapter[1, 1] = 2.0

    plain, _values, _next_hidden = model.factorized_packed_action_log_probs_seat_aware(
        obs,
        acting_seat,
        seat_hidden,
        legal_actions=legal_actions,
        opponent_context_index=torch.tensor([0], dtype=torch.long),
    )
    contextual, _values, _next_hidden = model.factorized_packed_action_log_probs_seat_aware(
        obs,
        acting_seat,
        seat_hidden,
        legal_actions=legal_actions,
        opponent_context_index=torch.tensor([1], dtype=torch.long),
    )
    contextual_eval = model.evaluate_factorized_sequence_packed_seat_aware(
        obs.unsqueeze(0),
        acting_seat.unsqueeze(0),
        seat_hidden,
        legal_actions=legal_actions,
        actions=torch.tensor([[1]], dtype=torch.long),
        opponent_context_index=torch.tensor([[1]], dtype=torch.long),
    )

    assert contextual[1] > plain[1]
    assert contextual[0] < plain[0]
    assert contextual[2] < plain[2]
    torch.testing.assert_close(torch.logsumexp(contextual, dim=0), torch.tensor(0.0))
    assert contextual_eval.action_logp is not None
    torch.testing.assert_close(contextual_eval.action_logp.squeeze(), contextual[1])
