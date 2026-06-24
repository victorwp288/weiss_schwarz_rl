from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from weiss_rl.learners.impala.losses.loss_inputs import (
    prepare_impala_loss_inputs,
    resolve_impala_loss_masks,
)
from weiss_rl.learners.impala.losses.loss_legal_mask import resolve_impala_dense_legal_mask
from weiss_rl.learners.impala.losses.loss_masks import (
    resolve_impala_loss_masks as resolve_impala_loss_masks_stage,
)

from .impala_test_support import ImpalaLearner, TinyPolicyValueModel


def test_resolve_impala_loss_masks_converts_reset_and_retention_activity() -> None:
    learner = ImpalaLearner(model=TinyPolicyValueModel(action_dim=2), trajectory_retention_coef=0.4)
    obs = torch.zeros((2, 1, 2), dtype=torch.float32)
    batch = {
        "policy_train_mask": np.asarray([[True], [False]], dtype=np.bool_),
        "reset_before_step": np.asarray([[False], [True]], dtype=np.bool_),
        "trajectory_retention_valid": np.asarray([[False], [True]], dtype=np.bool_),
    }

    masks = resolve_impala_loss_masks(
        learner=learner,
        batch=batch,
        obs=obs,
        batch_value=lambda source, key: source.get(key),
    )

    assert masks.loss_mask.tolist() == [[1.0], [0.0]]
    assert masks.reset_before_step is not None
    assert masks.reset_before_step.dtype == torch.bool
    assert masks.reset_before_step.tolist() == [[False], [True]]
    assert masks.trajectory_retention_valid is not None
    assert masks.trajectory_retention_valid.tolist() == [[0.0], [1.0]]
    assert masks.trajectory_retention_active is not None
    assert masks.trajectory_retention_active.dtype == torch.bool
    assert masks.trajectory_retention_active.tolist() == [[False], [True]]


def test_resolve_impala_loss_masks_defaults_policy_mask_and_disables_retention_activity() -> None:
    obs = torch.zeros((2, 1, 3), dtype=torch.float64)
    batch = {
        "trajectory_retention_valid": np.asarray([[True], [False]], dtype=np.bool_),
    }
    calls: list[tuple[str, Any]] = []
    learner = SimpleNamespace(
        trajectory_retention_coef=0.0,
        _optional_time_major_loss_mask=lambda value, *, expected_shape, like: (
            calls.append(("mask", (value, expected_shape, like.shape, like.dtype)))
            or (torch.as_tensor(value, dtype=torch.float32) if value is not None else None)
        ),
    )

    masks = resolve_impala_loss_masks_stage(
        learner=learner,
        batch=batch,
        obs=obs,
        batch_value=lambda source, key: source.get(key),
    )

    assert masks.loss_mask.dtype == obs.dtype
    assert masks.loss_mask.device == obs.device
    assert masks.loss_mask.tolist() == [[1.0], [1.0]]
    assert masks.reset_before_step is None
    assert masks.trajectory_retention_valid is not None
    assert masks.trajectory_retention_valid.tolist() == [[1.0], [0.0]]
    assert masks.trajectory_retention_active is None
    assert calls[0] == ("mask", (None, torch.Size((2, 1)), torch.Size((2, 1)), torch.float64))
    assert calls[1] == ("mask", (None, torch.Size((2, 1)), torch.Size((2, 1)), torch.float64))
    assert calls[2][0] == "mask"
    assert calls[2][1][0] is batch["trajectory_retention_valid"]
    assert calls[2][1][1:] == (torch.Size((2, 1)), torch.Size((2, 1)), torch.float64)


def test_resolve_impala_dense_legal_mask_returns_none_for_packed_legal_without_resolving() -> None:
    calls: list[str] = []
    learner = SimpleNamespace(_resolve_legal_mask=lambda *args, **kwargs: calls.append("resolve"))
    obs = torch.zeros((2, 1, 3), dtype=torch.float32)
    logits = torch.zeros((2, 1, 5), dtype=torch.float32)
    packed_legal = (
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([0, 1, 2], dtype=torch.long),
        None,
    )

    result = resolve_impala_dense_legal_mask(
        learner=learner,
        batch={},
        obs=obs,
        packed_legal=packed_legal,
        logits=logits,
    )

    assert result is None
    assert calls == []


def test_resolve_impala_dense_legal_mask_resolves_and_validates_dense_shape() -> None:
    obs = torch.zeros((2, 1, 3), dtype=torch.float32)
    logits = torch.zeros((2, 1, 5), dtype=torch.float32)
    legal_mask = torch.ones_like(logits, dtype=torch.bool)
    batch = {"dense": True}
    calls: list[tuple[Any, torch.Size, int]] = []
    learner = SimpleNamespace(
        _resolve_legal_mask=lambda source_batch, *, expected_shape, action_dim: (
            calls.append((source_batch, expected_shape, action_dim)) or legal_mask
        )
    )

    result = resolve_impala_dense_legal_mask(
        learner=learner,
        batch=batch,
        obs=obs,
        packed_legal=None,
        logits=logits,
    )

    assert result is legal_mask
    assert calls == [(batch, torch.Size((2, 1)), 5)]


def test_resolve_impala_dense_legal_mask_rejects_missing_logits_and_shape_mismatch() -> None:
    obs = torch.zeros((2, 1, 3), dtype=torch.float32)
    logits = torch.zeros((2, 1, 5), dtype=torch.float32)
    learner = SimpleNamespace(_resolve_legal_mask=lambda *args, **kwargs: torch.ones((2, 1, 4), dtype=torch.bool))

    with pytest.raises(ValueError, match="dense learner path requires dense logits"):
        resolve_impala_dense_legal_mask(
            learner=learner,
            batch={},
            obs=obs,
            packed_legal=None,
            logits=None,
        )
    with pytest.raises(ValueError, match="legal_mask must match learner logits"):
        resolve_impala_dense_legal_mask(
            learner=learner,
            batch={},
            obs=obs,
            packed_legal=None,
            logits=logits,
        )


def test_prepare_impala_loss_inputs_rejects_dense_legal_mask_shape_mismatch() -> None:
    learner = ImpalaLearner(model=TinyPolicyValueModel(action_dim=2))
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.5, -0.5]]], dtype=np.float32),
        "actions": np.asarray([[0], [1]], dtype=np.int64),
        "legal_mask": np.ones((2, 1, 2), dtype=np.uint8),
    }

    def bad_legal_mask(_batch: Any, *, expected_shape: torch.Size, action_dim: int) -> torch.Tensor:
        del expected_shape, action_dim
        return torch.ones((1, 1, 2), dtype=torch.bool)

    cast(Any, learner)._resolve_legal_mask = bad_legal_mask

    with pytest.raises(ValueError, match="legal_mask must match learner logits"):
        prepare_impala_loss_inputs(learner=learner, batch=batch, batch_value=lambda source, key: source.get(key))
