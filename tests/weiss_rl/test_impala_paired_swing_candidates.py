from __future__ import annotations

import pytest
import torch
from weiss_rl.learners.impala.auxiliary.paired_swing_candidates import compute_paired_swing_candidate_view
from weiss_rl.learners.impala.batching.paired_auxiliary_batch import resolve_paired_auxiliary_batch_inputs

from tests.weiss_rl.impala_paired_auxiliary_test_support import make_paired_swing_dense_case


def test_compute_paired_swing_candidate_view_preserves_dense_path_outputs() -> None:
    action_catalog, learner, batch = make_paired_swing_dense_case()
    inputs = resolve_paired_auxiliary_batch_inputs(
        learner,
        batch,
        packed_legal_error="paired-swing replay requires packed legal_ids/legal_offsets",
    )

    candidate_view = compute_paired_swing_candidate_view(
        learner,
        batch,
        obs=inputs.obs,
        expected_shape=inputs.expected_shape,
        packed_legal=inputs.packed_legal,
        loss_mask=inputs.loss_mask,
        margin_retention_coef=0.0,
        top_action_retention_coef=0.0,
    )

    assert candidate_view.reference_packed_logits is None
    assert candidate_view.logits is not None
    assert candidate_view.values is not None
    assert candidate_view.zero.item() == pytest.approx(0.0)
    assert candidate_view.packed_view.logits.tolist() == pytest.approx([0.0, 1.0])
    assert candidate_view.logits.shape == torch.Size([1, 1, action_catalog.action_space_size])
    assert candidate_view.values.shape == torch.Size([1, 1])
