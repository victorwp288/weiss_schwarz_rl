from __future__ import annotations

import numpy as np
import torch
from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.model import StructuredLegalPolicyValueModel, build_policy_value_model

from tests.weiss_rl.contracts_test_support import (
    _structured_model_config,
    _structured_spec_bundle,
    _typed_observation_spec,
)


def test_structured_legal_policy_value_model_sequence_forward_matches_stepwise_packed() -> None:
    torch.manual_seed(0)
    model = build_policy_value_model(
        observation_dim=18,
        config=_structured_model_config(),
        action_dim=9,
        observation_spec=_typed_observation_spec(),
        spec_bundle=_structured_spec_bundle(),
    )
    assert isinstance(model, StructuredLegalPolicyValueModel)

    obs = torch.zeros((2, 2, 18), dtype=torch.float32)
    acting_seat = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    seat_hidden = model.initial_seat_hidden(2)
    packed_ids = np.asarray([0, 3, 1, 4, 2, 5, 6, 8], dtype=np.int32)
    packed_offsets = np.asarray([0, 2, 4, 6, 8], dtype=np.int32)
    legal_actions = LegalActionBatch.from_packed(packed_ids, packed_offsets, action_space=9)

    logits_sequence, values_sequence, next_hidden_sequence = model.forward_sequence_seat_aware(
        obs,
        acting_seat,
        seat_hidden,
        legal_actions=legal_actions,
    )

    step_hidden = seat_hidden.clone()
    logits_steps: list[torch.Tensor] = []
    value_steps: list[torch.Tensor] = []
    row_cursor = 0
    for step_index in range(obs.shape[0]):
        step_offsets = packed_offsets[row_cursor : row_cursor + obs.shape[1] + 1]
        step_ids = packed_ids[int(step_offsets[0]) : int(step_offsets[-1])]
        step_legal_actions = LegalActionBatch.from_packed(
            step_ids,
            step_offsets - int(step_offsets[0]),
            action_space=9,
        )
        step_logits, step_values, step_hidden = model.forward_seat_aware(
            obs[step_index],
            acting_seat[step_index],
            step_hidden,
            legal_actions=step_legal_actions,
        )
        logits_steps.append(step_logits)
        value_steps.append(step_values)
        row_cursor += int(obs.shape[1])

    torch.testing.assert_close(logits_sequence, torch.stack(logits_steps, dim=0))
    torch.testing.assert_close(values_sequence, torch.stack(value_steps, dim=0))
    torch.testing.assert_close(next_hidden_sequence, step_hidden)


def test_structured_sequence_forward_resets_hidden_on_episode_boundary() -> None:
    torch.manual_seed(0)
    model = build_policy_value_model(
        observation_dim=18,
        config=_structured_model_config(),
        action_dim=9,
        observation_spec=_typed_observation_spec(),
        spec_bundle=_structured_spec_bundle(),
    )
    assert isinstance(model, StructuredLegalPolicyValueModel)

    obs = torch.randn((2, 2, 18), dtype=torch.float32)
    acting_seat = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    reset_before_step = torch.tensor([[False, False], [True, False]])
    seat_hidden = model.initial_seat_hidden(2)
    packed_ids = np.asarray([0, 3, 1, 4, 2, 5, 6, 8], dtype=np.int32)
    packed_offsets = np.asarray([0, 2, 4, 6, 8], dtype=np.int32)
    legal_actions = LegalActionBatch.from_packed(packed_ids, packed_offsets, action_space=9)

    logits_sequence, values_sequence, next_hidden_sequence = model.forward_sequence_seat_aware(
        obs,
        acting_seat,
        seat_hidden,
        legal_actions=legal_actions,
        reset_before_step=reset_before_step,
    )

    step_hidden = seat_hidden.clone()
    logits_steps: list[torch.Tensor] = []
    value_steps: list[torch.Tensor] = []
    row_cursor = 0
    for step_index in range(obs.shape[0]):
        step_reset = reset_before_step[step_index]
        if bool(step_reset.any().item()):
            step_hidden = step_hidden.clone()
            step_hidden[step_reset] = model.initial_seat_hidden(int(step_reset.sum().item()))
        step_offsets = packed_offsets[row_cursor : row_cursor + obs.shape[1] + 1]
        step_ids = packed_ids[int(step_offsets[0]) : int(step_offsets[-1])]
        step_legal_actions = LegalActionBatch.from_packed(
            step_ids,
            step_offsets - int(step_offsets[0]),
            action_space=9,
        )
        step_logits, step_values, step_hidden = model.forward_seat_aware(
            obs[step_index],
            acting_seat[step_index],
            step_hidden,
            legal_actions=step_legal_actions,
        )
        logits_steps.append(step_logits)
        value_steps.append(step_values)
        row_cursor += int(obs.shape[1])

    torch.testing.assert_close(logits_sequence, torch.stack(logits_steps, dim=0))
    torch.testing.assert_close(values_sequence, torch.stack(value_steps, dim=0))
    torch.testing.assert_close(next_hidden_sequence, step_hidden)
