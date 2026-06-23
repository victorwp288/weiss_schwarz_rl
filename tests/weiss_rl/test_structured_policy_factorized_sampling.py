from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.model import StructuredLegalPolicyValueModel, build_policy_value_model

from tests.weiss_rl.contracts_test_support import (
    _packed_meta_from_ids,
    _structured_hand_observation_spec,
    _structured_hand_spec_bundle,
    _structured_model_config,
    _structured_spec_bundle,
    _typed_observation_spec,
)


def test_structured_legal_policy_value_model_factorized_packed_sampling_returns_legal_actions() -> None:
    torch.manual_seed(0)
    model = build_policy_value_model(
        observation_dim=8,
        config=replace(_structured_model_config(), structured_policy_contract="factorized_v1"),
        action_dim=5,
        observation_spec=_structured_hand_observation_spec(),
        spec_bundle=_structured_hand_spec_bundle(),
    )
    assert isinstance(model, StructuredLegalPolicyValueModel)
    assert model.supports_factorized_legal_policy is True

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

    sampled_actions, sampled_logp, values, next_hidden = model.sample_factorized_packed_seat_aware(
        obs,
        acting_seat,
        seat_hidden,
        legal_actions=legal_actions,
        sample_seeds=torch.tensor([12345, 67890], dtype=torch.long),
        pass_action_id=4,
    )
    candidate_logp, _candidate_values, candidate_next_hidden = model.factorized_packed_action_log_probs_seat_aware(
        obs,
        acting_seat,
        seat_hidden,
        legal_actions=legal_actions,
    )
    factorized_eval = model.evaluate_factorized_sequence_packed_seat_aware(
        obs.unsqueeze(0),
        acting_seat.unsqueeze(0),
        seat_hidden,
        legal_actions=legal_actions,
        actions=sampled_actions.unsqueeze(0),
    )

    assert sampled_actions.shape == (2,)
    assert sampled_logp.shape == (2,)
    assert values.shape == (2,)
    assert next_hidden.shape == (2, 2, 256)
    assert candidate_logp.shape == (6,)
    assert torch.isfinite(sampled_logp).all()
    assert torch.isfinite(candidate_logp).all()
    assert int(sampled_actions[0].item()) in set(packed_ids[:4].tolist())
    assert int(sampled_actions[1].item()) in set(packed_ids[3:].tolist())
    selected_candidate_positions = torch.tensor(
        [
            int(np.flatnonzero(packed_ids[:3] == int(sampled_actions[0].item()))[0]),
            3 + int(np.flatnonzero(packed_ids[3:] == int(sampled_actions[1].item()))[0]),
        ],
        dtype=torch.long,
    )
    torch.testing.assert_close(candidate_logp.index_select(0, selected_candidate_positions), sampled_logp)
    torch.testing.assert_close(candidate_next_hidden, next_hidden)
    assert factorized_eval.values.shape == (1, 2)
    assert factorized_eval.action_logp is not None
    assert factorized_eval.entropy is not None
    assert factorized_eval.family_log_probs.shape[:2] == (1, 2)
    assert torch.isfinite(factorized_eval.action_logp).all()
    assert torch.isfinite(factorized_eval.entropy).all()
    torch.testing.assert_close(factorized_eval.action_logp.squeeze(0), sampled_logp)


def test_structured_legal_policy_value_model_factorized_sampling_uses_opponent_context_bias() -> None:
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
        model.opponent_context_action_bias_adapter[1, 1] = 50.0

    sampled_actions, sampled_logp, _values, _next_hidden = model.sample_factorized_packed_seat_aware(
        obs,
        acting_seat,
        seat_hidden,
        legal_actions=legal_actions,
        sample_seeds=torch.tensor([12345], dtype=torch.long),
        pass_action_id=4,
        opponent_context_index=torch.tensor([1], dtype=torch.long),
    )

    assert sampled_actions.tolist() == [1]
    assert float(sampled_logp.item()) == pytest.approx(0.0, abs=1e-5)


def test_structured_legal_policy_value_model_factorized_multi_family_sampling_is_legal() -> None:
    torch.manual_seed(0)
    model = build_policy_value_model(
        observation_dim=18,
        config=replace(_structured_model_config(), structured_policy_contract="factorized_v1"),
        action_dim=9,
        observation_spec=_typed_observation_spec(),
        spec_bundle=_structured_spec_bundle(),
    )
    assert isinstance(model, StructuredLegalPolicyValueModel)

    obs = torch.zeros((2, 18), dtype=torch.float32)
    acting_seat = torch.tensor([0, 1], dtype=torch.long)
    seat_hidden = model.initial_seat_hidden(2)
    packed_ids = np.asarray([4, 6, 8, 5, 7, 8], dtype=np.int32)
    packed_offsets = np.asarray([0, 3, 6], dtype=np.int32)
    packed_meta = _packed_meta_from_ids(ActionCatalog.from_spec_bundle(_structured_spec_bundle()), packed_ids)
    legal_actions = LegalActionBatch.from_packed(
        packed_ids,
        packed_offsets,
        meta=packed_meta,
        action_space=9,
    )

    sampled_actions, sampled_logp, values, next_hidden = model.sample_factorized_packed_seat_aware(
        obs,
        acting_seat,
        seat_hidden,
        legal_actions=legal_actions,
        sample_seeds=torch.tensor([13579, 24680], dtype=torch.long),
        pass_action_id=8,
    )
    factorized_eval = model.evaluate_factorized_sequence_packed_seat_aware(
        obs.unsqueeze(0),
        acting_seat.unsqueeze(0),
        seat_hidden,
        legal_actions=legal_actions,
        actions=sampled_actions.unsqueeze(0),
    )

    assert sampled_actions.shape == (2,)
    assert sampled_logp.shape == (2,)
    assert values.shape == (2,)
    assert next_hidden.shape == (2, 2, 256)
    assert torch.isfinite(sampled_logp).all()
    assert int(sampled_actions[0].item()) in set(packed_ids[:3].tolist())
    assert int(sampled_actions[1].item()) in set(packed_ids[3:].tolist())
    assert factorized_eval.action_logp is not None
    assert factorized_eval.entropy is not None
    assert factorized_eval.attack_slot_log_probs is not None
    assert factorized_eval.attack_type_log_probs is not None
    assert torch.isfinite(factorized_eval.action_logp).all()
    assert torch.isfinite(factorized_eval.entropy).all()
    torch.testing.assert_close(factorized_eval.action_logp.squeeze(0), sampled_logp)
