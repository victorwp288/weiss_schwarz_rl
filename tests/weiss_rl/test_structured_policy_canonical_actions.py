from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.model import StructuredLegalPolicyValueModel, build_policy_value_model

from tests.weiss_rl.contracts_test_support import (
    _make_structured_joint_scorer_nonuniform,
    _packed_meta_from_ids,
    _structured_hand_observation_spec,
    _structured_hand_spec_bundle,
    _structured_model_config,
)


def test_structured_legal_policy_value_model_packed_path_uses_hand_position_for_duplicate_cards() -> None:
    torch.manual_seed(0)
    model = build_policy_value_model(
        observation_dim=8,
        config=_structured_model_config(),
        action_dim=5,
        observation_spec=_structured_hand_observation_spec(),
        spec_bundle=_structured_hand_spec_bundle(),
    )
    assert isinstance(model, StructuredLegalPolicyValueModel)
    _make_structured_joint_scorer_nonuniform(model)

    obs = torch.zeros((1, 8), dtype=torch.float32)
    obs[0, 4] = 11.0
    obs[0, 5] = 11.0
    seat_hidden = model.initial_seat_hidden(1)
    packed_ids = np.asarray([0, 2, 4], dtype=np.int32)
    packed_offsets = np.asarray([0, 3], dtype=np.int32)

    logits, _values, _next_hidden = model.forward_seat_aware(
        obs,
        torch.tensor([0]),
        seat_hidden,
        legal_actions=LegalActionBatch.from_packed(packed_ids, packed_offsets, action_space=5),
    )

    assert not torch.isclose(logits[0, 0], logits[0, 2], atol=1e-6, rtol=0.0)


def test_structured_legal_policy_value_model_factorized_path_uses_action_ids_as_canonical_source() -> None:
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
    corrupt_meta = packed_meta.copy()
    corrupt_meta[:, 0] = np.uint16(0)
    corrupt_meta[:, 1:] = np.uint16(np.iinfo(np.uint16).max)

    legal_actions = LegalActionBatch.from_packed(
        packed_ids,
        packed_offsets,
        meta=packed_meta,
        action_space=5,
    )
    corrupt_legal_actions = LegalActionBatch.from_packed(
        packed_ids,
        packed_offsets,
        meta=corrupt_meta,
        action_space=5,
    )
    actions = torch.tensor([[1]], dtype=torch.long)

    reference = model.evaluate_factorized_sequence_packed_seat_aware(
        obs.unsqueeze(0),
        acting_seat.unsqueeze(0),
        seat_hidden,
        legal_actions=legal_actions,
        actions=actions,
    )
    corrupted = model.evaluate_factorized_sequence_packed_seat_aware(
        obs.unsqueeze(0),
        acting_seat.unsqueeze(0),
        seat_hidden,
        legal_actions=corrupt_legal_actions,
        actions=actions,
    )

    assert reference.action_logp is not None
    assert corrupted.action_logp is not None
    torch.testing.assert_close(corrupted.action_logp, reference.action_logp)
    torch.testing.assert_close(corrupted.entropy, reference.entropy)
