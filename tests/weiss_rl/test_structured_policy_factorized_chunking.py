from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

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


def test_structured_legal_policy_value_model_factorized_chunking_matches_unchunked() -> None:
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
    actions = torch.tensor([[1, 3]], dtype=torch.long)

    baseline = model.evaluate_factorized_sequence_packed_seat_aware(
        obs.unsqueeze(0),
        acting_seat.unsqueeze(0),
        seat_hidden,
        legal_actions=legal_actions,
        actions=actions,
    )
    model.policy_head._factorized_row_chunk_size = cast(Any, lambda _row_states: 1)
    chunked = model.evaluate_factorized_sequence_packed_seat_aware(
        obs.unsqueeze(0),
        acting_seat.unsqueeze(0),
        seat_hidden,
        legal_actions=legal_actions,
        actions=actions,
    )

    assert baseline.action_logp is not None
    assert chunked.action_logp is not None
    torch.testing.assert_close(chunked.action_logp, baseline.action_logp)
    torch.testing.assert_close(chunked.entropy, baseline.entropy)
    torch.testing.assert_close(chunked.family_log_probs, baseline.family_log_probs)
