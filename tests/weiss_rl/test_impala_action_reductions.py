from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from weiss_rl.learners.action_logp import (
    packed_scores_action_logp_and_entropy,
    packed_scores_family_entropy,
)
from weiss_rl.learners.impala.losses.action_reductions import resolve_impala_action_reductions
from weiss_rl.learners.impala.losses.loss_inputs import prepare_impala_loss_inputs
from weiss_rl.learners.impala.losses.loss_pipeline import resolve_impala_loss_action_reductions

from .impala_test_support import (
    ImpalaLearner,
    TinyPolicyValueModel,
    _packed_meta_from_ids,
    _simple_training_batch,
    _teacher_aux_catalog,
)


def test_resolve_impala_action_reductions_uses_packed_candidate_family_entropy() -> None:
    action_catalog = _teacher_aux_catalog()
    actions = torch.as_tensor([[5], [12]], dtype=torch.long)
    packed_ids = torch.as_tensor([0, 5, 19, 10, 11, 12, 19], dtype=torch.long)
    packed_offsets = torch.as_tensor([0, 3, 7], dtype=torch.long)
    packed_meta = torch.as_tensor(
        _packed_meta_from_ids(action_catalog, packed_ids.numpy().astype(np.uint32, copy=False)),
        dtype=torch.long,
    )
    packed_logits = torch.as_tensor([0.0, 2.0, 1.0, -1.0, 0.5, 3.0, 0.0], dtype=torch.float32)
    timings: list[tuple[str, float]] = []

    reductions = resolve_impala_action_reductions(
        factorized_result=None,
        logits=None,
        packed_logits=packed_logits,
        legal_mask=None,
        packed_legal=(packed_ids, packed_offsets, packed_meta),
        actions=actions,
        entropy_scope="family",
        pass_action_id=action_catalog.pass_action_id,
        action_catalog=action_catalog,
        record_timing_ms=lambda name, duration: timings.append((name, duration)),
    )
    expected_logp, _candidate_entropy = packed_scores_action_logp_and_entropy(
        packed_logits,
        packed_ids,
        packed_offsets,
        actions,
        pass_action_id=action_catalog.pass_action_id,
    )
    expected_family_entropy = packed_scores_family_entropy(
        packed_logits,
        packed_offsets,
        packed_meta,
        row_shape=actions.shape,
        family_count=len(action_catalog.families),
    )

    torch.testing.assert_close(reductions.action_logp, expected_logp)
    torch.testing.assert_close(reductions.entropy, expected_family_entropy)
    assert reductions.action_logp.shape == actions.shape
    assert reductions.entropy.shape == actions.shape
    assert torch.isfinite(reductions.action_logp).all()
    assert torch.isfinite(reductions.entropy).all()
    assert [name for name, _duration in timings] == ["learner_packed_reductions"]
    assert timings[0][1] >= 0.0


def test_resolve_impala_action_reductions_preserves_family_entropy_requirements() -> None:
    action_catalog = _teacher_aux_catalog()
    actions = torch.as_tensor([[5]], dtype=torch.long)
    packed_ids = torch.as_tensor([0, 5, 19], dtype=torch.long)
    packed_offsets = torch.as_tensor([0, 3], dtype=torch.long)
    packed_logits = torch.as_tensor([0.0, 2.0, 1.0], dtype=torch.float32)

    with pytest.raises(ValueError, match="family entropy requires packed legal-action metadata and action_catalog"):
        resolve_impala_action_reductions(
            factorized_result=None,
            logits=None,
            packed_logits=packed_logits,
            legal_mask=None,
            packed_legal=(packed_ids, packed_offsets, None),
            actions=actions,
            entropy_scope="family",
            pass_action_id=action_catalog.pass_action_id,
            action_catalog=action_catalog,
            record_timing_ms=lambda _name, _duration: None,
        )

    with pytest.raises(ValueError, match="family entropy requires packed candidate logits"):
        resolve_impala_action_reductions(
            factorized_result=None,
            logits=torch.zeros((1, 1, action_catalog.action_space_size), dtype=torch.float32),
            packed_logits=None,
            legal_mask=None,
            packed_legal=(packed_ids, packed_offsets, None),
            actions=actions,
            entropy_scope="family",
            pass_action_id=action_catalog.pass_action_id,
            action_catalog=action_catalog,
            record_timing_ms=lambda _name, _duration: None,
        )


def test_resolve_impala_action_reductions_preserves_factorized_requirement_error() -> None:
    with pytest.raises(ValueError, match="factorized learner path requires action_logp and entropy"):
        resolve_impala_action_reductions(
            factorized_result=SimpleNamespace(action_logp=torch.zeros((1, 1)), entropy=None),
            logits=None,
            packed_logits=None,
            legal_mask=None,
            packed_legal=None,
            actions=torch.zeros((1, 1), dtype=torch.long),
            entropy_scope="candidate",
            pass_action_id=None,
            action_catalog=None,
            record_timing_ms=lambda _name, _duration: None,
        )


def test_resolve_impala_loss_action_reductions_attaches_detached_context_and_checks_finiteness() -> None:
    learner = ImpalaLearner(model=TinyPolicyValueModel(action_dim=2), structured_metrics_mode="off")
    batch = _simple_training_batch()
    inputs = prepare_impala_loss_inputs(learner=learner, batch=batch, batch_value=lambda source, key: source.get(key))
    finite_calls: list[str] = []

    def record_finite(name: str, tensor: torch.Tensor, *, batch: Any, context: dict[str, Any]) -> None:
        del tensor, batch, context
        finite_calls.append(name)

    cast(Any, learner)._ensure_finite_tensor = record_finite

    reductions = resolve_impala_loss_action_reductions(
        learner=learner,
        batch=batch,
        loss_inputs=inputs,
    )

    assert reductions.context is inputs.context
    assert reductions.context["action_logp"].shape == torch.Size((2, 1))
    assert reductions.context["entropy"].shape == torch.Size((2, 1))
    assert reductions.context["action_logp"].requires_grad is False
    assert reductions.context["entropy"].requires_grad is False
    assert finite_calls == ["action_logp", "entropy"]
