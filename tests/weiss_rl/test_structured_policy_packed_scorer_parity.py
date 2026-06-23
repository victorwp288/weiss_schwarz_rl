from __future__ import annotations

import numpy as np
import torch
from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.learners.action_logp import packed_scores_action_logp_and_entropy
from weiss_rl.model import StructuredLegalPolicyValueModel, build_policy_value_model

from tests.weiss_rl.contracts_test_support import (
    _packed_meta_from_ids,
    _structured_model_config,
    _structured_spec_bundle,
    _typed_observation_spec,
)


def test_structured_legal_policy_value_model_actor_and_learner_packed_scorers_match() -> None:
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
    acting_seat = torch.tensor([0, 1], dtype=torch.long)
    seat_hidden = model.initial_seat_hidden(2)
    packed_ids = np.asarray([0, 4, 6, 8, 3, 5, 7, 8], dtype=np.int32)
    packed_offsets = np.asarray([0, 4, 8], dtype=np.int32)
    packed_meta = _packed_meta_from_ids(ActionCatalog.from_spec_bundle(_structured_spec_bundle()), packed_ids)
    legal_actions = LegalActionBatch.from_packed(
        packed_ids,
        packed_offsets,
        meta=packed_meta,
        action_space=9,
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

    torch.testing.assert_close(learner_scores, actor_scores)


def test_structured_packed_behavior_logp_matches_learner_recomputed_action_logp() -> None:
    torch.manual_seed(0)
    model = build_policy_value_model(
        observation_dim=18,
        config=_structured_model_config(),
        action_dim=9,
        observation_spec=_typed_observation_spec(),
        spec_bundle=_structured_spec_bundle(),
    )
    assert isinstance(model, StructuredLegalPolicyValueModel)

    obs = torch.randn((4, 18), dtype=torch.float32)
    acting_seat = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    seat_hidden = model.initial_seat_hidden(4)
    packed_ids = np.asarray([0, 4, 8, 1, 5, 8, 2, 6, 8, 3, 7, 8], dtype=np.int32)
    packed_offsets = np.asarray([0, 3, 6, 9, 12], dtype=np.int32)
    packed_meta = _packed_meta_from_ids(ActionCatalog.from_spec_bundle(_structured_spec_bundle()), packed_ids)
    legal_actions = LegalActionBatch.from_packed(
        packed_ids,
        packed_offsets,
        meta=packed_meta,
        action_space=9,
    )
    sample_seeds = torch.tensor([1001, 1002, 1003, 1004], dtype=torch.long)

    with torch.inference_mode():
        sampled_actions, behavior_logp, _value, _next_hidden = model.sample_packed_seat_aware(
            obs,
            acting_seat,
            seat_hidden,
            legal_actions=legal_actions,
            sample_seeds=sample_seeds,
            pass_action_id=8,
        )
        learner_scores, _learner_value, _learner_next_hidden = model.forward_packed_seat_aware(
            obs,
            acting_seat,
            seat_hidden,
            legal_actions=legal_actions,
            scoring_mode="learner",
        )
        recomputed_logp, _entropy = packed_scores_action_logp_and_entropy(
            learner_scores,
            torch.as_tensor(packed_ids, dtype=torch.long),
            torch.as_tensor(packed_offsets, dtype=torch.long),
            sampled_actions,
            pass_action_id=8,
        )

    torch.testing.assert_close(recomputed_logp, behavior_logp)
