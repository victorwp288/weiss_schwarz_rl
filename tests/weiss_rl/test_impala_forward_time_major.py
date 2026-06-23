from __future__ import annotations

import numpy as np
import pytest
import torch
from weiss_rl.core.legal_actions import LegalActionBatch

from .impala_test_support import (
    ForwardProxyModel,
    ImpalaLearner,
    SequenceStructuredTeacherModel,
    TinyPolicyValueModel,
    TinyStructuredTeacherModel,
    TrunkStructuredTeacherModel,
    _packed_meta_from_ids,
    _simple_training_batch,
    _teacher_aux_catalog,
)


def test_impala_learner_uses_compiled_forward_model_when_provided() -> None:
    base_model = TinyPolicyValueModel(action_dim=2)
    compiled_proxy = ForwardProxyModel(base_model)
    learner = ImpalaLearner(model=base_model, compiled_model=compiled_proxy)

    loss, _metrics = learner._loss_and_metrics(_simple_training_batch())

    assert float(loss.detach()) != 0.0
    assert compiled_proxy.forward_calls == 2


def test_impala_learner_forward_time_major_matches_manual_legacy_rollout() -> None:
    torch.manual_seed(0)

    model = TinyPolicyValueModel(observation_dim=2, action_dim=3)
    learner = ImpalaLearner(model=model)
    obs = torch.tensor(
        [
            [[0.25, -0.5], [1.0, 0.0]],
            [[-0.75, 0.5], [0.125, 0.25]],
        ],
        dtype=torch.float32,
    )
    initial_hidden = torch.ones((2, 1), dtype=torch.float32)

    with torch.no_grad():
        learner_logits, learner_values = learner._forward_time_major(obs, initial_hidden_state=initial_hidden)

        manual_hidden = initial_hidden
        manual_logits_steps: list[torch.Tensor] = []
        manual_value_steps: list[torch.Tensor] = []
        for step_obs in obs.unbind(dim=0):
            step_logits, step_value, manual_hidden = model(step_obs, manual_hidden)
            manual_logits_steps.append(step_logits)
            manual_value_steps.append(step_value)

    torch.testing.assert_close(learner_logits, torch.stack(manual_logits_steps, dim=0))
    torch.testing.assert_close(learner_values, torch.stack(manual_value_steps, dim=0))


def test_impala_learner_forward_time_major_requires_packed_meta_for_structured_updates() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(model=TinyStructuredTeacherModel(action_catalog))
    obs = torch.zeros((1, 1, 2), dtype=torch.float32)
    legal_actions = LegalActionBatch.from_packed(
        np.asarray([0, 5, 19], dtype=np.uint32),
        np.asarray([0, 3], dtype=np.uint32),
        action_space=action_catalog.action_space_size,
    )

    with pytest.raises(ValueError, match="packed legal_actions metadata"):
        learner._forward_time_major(
            obs,
            to_play_seat=np.asarray([[0]], dtype=np.int64),
            legal_actions=legal_actions,
        )


def test_impala_learner_forward_time_major_uses_sequence_fast_path_and_records_packed_metrics() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=SequenceStructuredTeacherModel(action_catalog),
        profile_timers=True,
    )
    learner._active_timing_metrics = {}
    obs = torch.zeros((2, 1, 2), dtype=torch.float32)
    packed_ids = np.asarray([0, 5, 19, 1, 13, 19], dtype=np.uint32)
    packed_offsets = np.asarray([0, 3, 6], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    legal_actions = LegalActionBatch.from_packed(
        packed_ids,
        packed_offsets,
        meta=packed_meta,
        action_space=action_catalog.action_space_size,
    )

    logits, values = learner._forward_time_major(
        obs,
        to_play_seat=np.asarray([[0], [1]], dtype=np.int64),
        legal_actions=legal_actions,
    )

    model = learner.model
    assert isinstance(model, SequenceStructuredTeacherModel)
    assert logits.shape == (2, 1, action_catalog.action_space_size)
    assert values.shape == (2, 1)
    assert model.sequence_calls == 1
    assert model.step_calls == 0
    assert learner._active_timing_metrics["packed_candidate_count"] == pytest.approx(6.0)
    assert learner._active_timing_metrics["packed_candidate_rows"] == pytest.approx(2.0)
    assert learner._active_timing_metrics["avg_legal_actions_per_row"] == pytest.approx(3.0)
    assert learner._active_timing_metrics["timer_learner_forward_time_major_ms"] >= 0.0


def test_impala_learner_forward_time_major_uses_trunk_sequence_path_and_records_breakdown() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=TrunkStructuredTeacherModel(action_catalog),
        profile_timers=True,
    )
    learner._active_timing_metrics = {}
    obs = torch.zeros((2, 1, 2), dtype=torch.float32)
    packed_ids = np.asarray([0, 5, 19, 1, 13, 19], dtype=np.uint32)
    packed_offsets = np.asarray([0, 3, 6], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    legal_actions = LegalActionBatch.from_packed(
        packed_ids,
        packed_offsets,
        meta=packed_meta,
        action_space=action_catalog.action_space_size,
    )

    packed_logits, values = learner._forward_time_major(
        obs,
        to_play_seat=np.asarray([[0], [1]], dtype=np.int64),
        legal_actions=legal_actions,
    )

    model = learner.model
    assert isinstance(model, TrunkStructuredTeacherModel)
    assert packed_logits.shape == (6,)
    assert values.shape == (2, 1)
    assert model.trunk_calls == 1
    assert model.scorer_calls == 1
    assert model.sequence_calls == 0
    assert learner._active_timing_metrics["timer_learner_trunk_ms"] >= 0.0
    assert learner._active_timing_metrics["timer_learner_packed_scorer_ms"] >= 0.0
