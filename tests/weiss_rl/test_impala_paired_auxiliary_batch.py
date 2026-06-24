from __future__ import annotations

import numpy as np
import pytest
import torch
from weiss_rl.learners.impala.batching.paired_auxiliary_batch import resolve_paired_auxiliary_batch_inputs

from .impala_test_support import (
    ImpalaLearner,
    TinyPolicyValueModel,
    _packed_meta_from_ids,
    _teacher_aux_catalog,
)


def test_resolve_paired_auxiliary_batch_inputs_preserves_default_loss_mask_contract() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(model=TinyPolicyValueModel(), pass_action_id=action_catalog.pass_action_id)
    packed_ids = np.asarray([0, 5, action_catalog.pass_action_id], dtype=np.uint32)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.5, -0.25]]], dtype=np.float32),
        "legal_ids": np.concatenate([packed_ids, packed_ids]),
        "legal_offsets": np.asarray([0, 3, 6], dtype=np.uint32),
        "legal_action_meta": _packed_meta_from_ids(action_catalog, np.concatenate([packed_ids, packed_ids])),
    }

    inputs = resolve_paired_auxiliary_batch_inputs(
        learner,
        batch,
        packed_legal_error="paired helper requires packed legal actions",
    )

    assert inputs.obs.shape == (2, 1, 2)
    assert inputs.expected_shape == torch.Size([2, 1])
    assert inputs.loss_mask.shape == torch.Size([2, 1])
    assert torch.all(inputs.loss_mask == 1.0)
    assert inputs.packed_legal[0].tolist() == [0, 5, action_catalog.pass_action_id, 0, 5, action_catalog.pass_action_id]


def test_resolve_paired_auxiliary_batch_inputs_preserves_missing_packed_error() -> None:
    learner = ImpalaLearner(model=TinyPolicyValueModel())
    batch = {"obs": np.asarray([[[1.0, 0.0]]], dtype=np.float32)}

    with pytest.raises(ValueError, match="paired helper requires packed legal actions"):
        resolve_paired_auxiliary_batch_inputs(
            learner,
            batch,
            packed_legal_error="paired helper requires packed legal actions",
        )
