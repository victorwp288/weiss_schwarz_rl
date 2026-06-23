from __future__ import annotations

from typing import Any, cast

import numpy as np
import torch
from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.model import StructuredLegalPolicyValueModel, build_policy_value_model

from tests.weiss_rl.contracts_test_support import (
    _packed_meta_from_ids,
    _structured_model_config,
    _structured_spec_bundle,
    _typed_observation_spec,
)


def test_structured_legal_policy_value_model_packed_plan_scorer_matches_dense_scorer() -> None:
    torch.manual_seed(0)
    model = build_policy_value_model(
        observation_dim=18,
        config=_structured_model_config(),
        action_dim=9,
        observation_spec=_typed_observation_spec(),
        spec_bundle=_structured_spec_bundle(),
    )
    assert isinstance(model, StructuredLegalPolicyValueModel)

    obs = torch.zeros((3, 18), dtype=torch.float32)
    obs[0, 4] = 11.0
    obs[0, 5] = 22.0
    obs[1, 2] = 1.0
    obs[1, 6] = 33.0
    obs[2, 14] = 1.0
    acting_seat = torch.tensor([0, 1, 0], dtype=torch.long)
    seat_hidden = model.initial_seat_hidden(3)
    packed_ids = np.asarray([0, 1, 4, 6, 8, 2, 5, 7, 8, 3, 4, 6, 8], dtype=np.int32)
    packed_offsets = np.asarray([0, 5, 9, 13], dtype=np.int32)
    packed_meta = _packed_meta_from_ids(ActionCatalog.from_spec_bundle(_structured_spec_bundle()), packed_ids)

    recurrent_output, state_repr, observation_context, _value, _next_hidden = model.forward_trunk_packed_seat_aware(
        obs,
        acting_seat,
        seat_hidden,
    )
    head = cast(Any, model.policy_head)
    ids = torch.as_tensor(packed_ids, dtype=torch.long)
    offsets = torch.as_tensor(packed_offsets, dtype=torch.long)
    row_indices = torch.repeat_interleave(
        torch.arange(obs.shape[0], dtype=torch.long),
        offsets[1:] - offsets[:-1],
    )
    meta = torch.as_tensor(packed_meta, dtype=torch.long)

    for candidate_meta in (None, meta):
        for scoring_mode in ("actor", "learner"):
            dense_scores = head._score_candidates_chunked(
                state_repr,
                row_indices,
                ids,
                observation_context,
                candidate_meta=candidate_meta,
                scoring_mode=scoring_mode,
            )
            packed_plan = head._build_packed_scoring_plan(
                candidate_ids=ids,
                offsets=offsets,
                candidate_meta=candidate_meta,
            )
            packed_scores = head._score_packed_candidates_chunked(
                state_repr,
                packed_plan,
                observation_context,
                scoring_mode=scoring_mode,
            )

            torch.testing.assert_close(packed_scores, dense_scores)

    legal_actions = LegalActionBatch.from_packed(
        packed_ids,
        packed_offsets,
        meta=packed_meta,
        action_space=9,
    )
    public_scores = model.score_packed_legal_candidates(
        recurrent_output,
        obs,
        legal_actions,
        state_repr=state_repr,
        observation_context=observation_context,
        scoring_mode="actor",
    )
    dense_meta_scores = head._score_candidates_chunked(
        state_repr,
        row_indices,
        ids,
        observation_context,
        candidate_meta=meta,
        scoring_mode="actor",
    )

    torch.testing.assert_close(public_scores, dense_meta_scores)
