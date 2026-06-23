from __future__ import annotations

from dataclasses import replace
from typing import cast

import numpy as np
import torch
from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.model import StructuredLegalPolicyValueModel, build_policy_value_model

from tests.weiss_rl.contracts_test_support import (
    _packed_meta_from_ids,
    _structured_choice_slot_spec_bundle,
    _structured_model_config,
    _structured_spec_bundle,
    _typed_observation_spec,
)


def test_structured_legal_policy_value_model_scores_legal_candidates() -> None:
    model = build_policy_value_model(
        observation_dim=18,
        config=_structured_model_config(),
        action_dim=9,
        observation_spec=_typed_observation_spec(),
        spec_bundle=_structured_spec_bundle(),
    )
    assert isinstance(model, StructuredLegalPolicyValueModel)

    obs = torch.zeros((2, 18), dtype=torch.float32)
    seat_hidden = model.initial_seat_hidden(2)
    legal_mask = np.zeros((1, 2, 9), dtype=np.bool_)
    legal_mask[0, 0, 0] = True
    legal_mask[0, 1, 4] = True
    logits, values, next_hidden = model.forward_seat_aware(
        obs,
        torch.tensor([0, 1]),
        seat_hidden,
        legal_actions=LegalActionBatch.from_mask(legal_mask),
    )

    assert logits.shape == (2, 9)
    assert values.shape == (2,)
    assert next_hidden.shape == (2, 2, 256)
    assert torch.isfinite(logits[:, [0, 4]]).all()
    assert torch.all(logits[:, [1, 2, 3, 5, 6, 7, 8]] < -1e8)


def test_structured_legal_policy_value_model_applies_candidate_scoring_chunk_config() -> None:
    model = build_policy_value_model(
        observation_dim=18,
        config=replace(
            _structured_model_config(),
            candidate_scoring_chunk_size=131072,
            cuda_learner_candidate_scoring_chunk_size=524288,
        ),
        action_dim=9,
        observation_spec=_typed_observation_spec(),
        spec_bundle=_structured_spec_bundle(),
    )
    assert isinstance(model, StructuredLegalPolicyValueModel)

    assert model.policy_head._candidate_scoring_chunk_size == 131072
    assert model.policy_head._cuda_learner_candidate_scoring_chunk_size == 524288


def test_structured_legal_policy_value_model_distinguishes_index_and_slot_argument_families() -> None:
    torch.manual_seed(0)
    model = build_policy_value_model(
        observation_dim=18,
        config=_structured_model_config(),
        action_dim=5,
        observation_spec=_typed_observation_spec(),
        spec_bundle=_structured_choice_slot_spec_bundle(),
    )
    assert isinstance(model, StructuredLegalPolicyValueModel)
    final_scorer = cast(torch.nn.Linear, model.policy_head.joint_scorer[-1])
    with torch.no_grad():
        torch.nn.init.normal_(final_scorer.weight, mean=0.0, std=0.1)
        torch.nn.init.zeros_(final_scorer.bias)

    obs = torch.zeros((1, 18), dtype=torch.float32)
    obs[0, 4] = 1.0
    obs[0, 7] = 2.0
    seat_hidden = model.initial_seat_hidden(1)
    legal_mask = np.zeros((1, 1, 5), dtype=np.bool_)
    legal_mask[0, 0, [0, 1, 2, 3, 4]] = True
    logits, _values, _next_hidden = model.forward_seat_aware(
        obs,
        torch.tensor([0]),
        seat_hidden,
        legal_actions=LegalActionBatch.from_mask(legal_mask),
    )

    assert not torch.isclose(logits[0, 0], logits[0, 1], atol=1e-6, rtol=0.0)
    assert not torch.isclose(logits[0, 2], logits[0, 3], atol=1e-6, rtol=0.0)


def test_structured_legal_policy_value_model_starts_uniform_over_packed_legal_candidates() -> None:
    torch.manual_seed(0)
    model = build_policy_value_model(
        observation_dim=18,
        config=_structured_model_config(),
        action_dim=9,
        observation_spec=_typed_observation_spec(),
        spec_bundle=_structured_spec_bundle(),
    )
    assert isinstance(model, StructuredLegalPolicyValueModel)

    obs = torch.zeros((2, 18), dtype=torch.float32)
    seat_hidden = model.initial_seat_hidden(2)
    packed_ids = np.asarray([0, 3, 5, 1, 4, 8], dtype=np.int32)
    packed_offsets = np.asarray([0, 3, 6], dtype=np.int32)

    logits, _values, _next_hidden = model.forward_seat_aware(
        obs,
        torch.tensor([0, 1]),
        seat_hidden,
        legal_actions=LegalActionBatch.from_packed(packed_ids, packed_offsets, action_space=9),
    )

    torch.testing.assert_close(logits[0, [0, 3, 5]], torch.zeros((3,), dtype=logits.dtype))
    torch.testing.assert_close(logits[1, [1, 4, 8]], torch.zeros((3,), dtype=logits.dtype))
    assert torch.all(logits[0, [1, 2, 4, 6, 7, 8]] < -1e8)
    assert torch.all(logits[1, [0, 2, 3, 5, 6, 7]] < -1e8)


def test_structured_legal_policy_value_model_packed_legal_matches_mask() -> None:
    torch.manual_seed(0)
    model = build_policy_value_model(
        observation_dim=18,
        config=_structured_model_config(),
        action_dim=9,
        observation_spec=_typed_observation_spec(),
        spec_bundle=_structured_spec_bundle(),
    )
    assert isinstance(model, StructuredLegalPolicyValueModel)

    obs = torch.zeros((2, 18), dtype=torch.float32)
    seat_hidden = model.initial_seat_hidden(2)
    legal_mask = np.zeros((1, 2, 9), dtype=np.bool_)
    legal_mask[0, 0, [0, 3, 5]] = True
    legal_mask[0, 1, [1, 4, 8]] = True
    packed_ids = np.asarray([0, 3, 5, 1, 4, 8], dtype=np.int32)
    packed_offsets = np.asarray([0, 3, 6], dtype=np.int32)

    logits_mask, values_mask, next_hidden_mask = model.forward_seat_aware(
        obs,
        torch.tensor([0, 1]),
        seat_hidden,
        legal_actions=LegalActionBatch.from_mask(legal_mask),
    )
    logits_packed, values_packed, next_hidden_packed = model.forward_seat_aware(
        obs,
        torch.tensor([0, 1]),
        seat_hidden,
        legal_actions=LegalActionBatch.from_packed(packed_ids, packed_offsets, action_space=9),
    )

    torch.testing.assert_close(logits_mask, logits_packed)
    torch.testing.assert_close(values_mask, values_packed)
    torch.testing.assert_close(next_hidden_mask, next_hidden_packed)


def test_structured_legal_policy_value_model_packed_meta_matches_packed_ids() -> None:
    torch.manual_seed(0)
    model = build_policy_value_model(
        observation_dim=18,
        config=_structured_model_config(),
        action_dim=9,
        observation_spec=_typed_observation_spec(),
        spec_bundle=_structured_spec_bundle(),
    )
    assert isinstance(model, StructuredLegalPolicyValueModel)

    obs = torch.zeros((2, 18), dtype=torch.float32)
    seat_hidden = model.initial_seat_hidden(2)
    packed_ids = np.asarray([4, 6, 8, 5, 7, 8], dtype=np.int32)
    packed_offsets = np.asarray([0, 3, 6], dtype=np.int32)
    action_catalog = ActionCatalog.from_spec_bundle(_structured_spec_bundle())
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)

    logits_packed, values_packed, next_hidden_packed = model.forward_seat_aware(
        obs,
        torch.tensor([0, 1]),
        seat_hidden,
        legal_actions=LegalActionBatch.from_packed(packed_ids, packed_offsets, action_space=9),
    )
    logits_meta, values_meta, next_hidden_meta = model.forward_seat_aware(
        obs,
        torch.tensor([0, 1]),
        seat_hidden,
        legal_actions=LegalActionBatch.from_packed(
            packed_ids,
            packed_offsets,
            meta=packed_meta,
            action_space=9,
        ),
    )

    torch.testing.assert_close(logits_packed, logits_meta)
    torch.testing.assert_close(values_packed, values_meta)
    torch.testing.assert_close(next_hidden_packed, next_hidden_meta)
