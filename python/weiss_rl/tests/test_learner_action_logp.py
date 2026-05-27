from __future__ import annotations

import numpy as np
import pytest
import torch

from weiss_rl.learners.action_logp import (
    learner_logp_from_legal_ids,
    learner_logp_from_mask,
    masked_action_logp_and_entropy,
    packed_action_logp_and_entropy,
    packed_scores_action_logp_and_entropy,
    packed_scores_family_entropy,
    packed_selected_action_logp,
    packed_subset_action_logp_and_top_action,
)
from weiss_rl.learners.impala_learner import (
    _masked_action_logp_and_entropy,
    _packed_selected_action_logp,
)
from weiss_rl.learners.impala_learner import (
    learner_logp_from_legal_ids as impala_learner_logp_from_legal_ids,
)
from weiss_rl.learners.impala_learner import (
    learner_logp_from_mask as impala_learner_logp_from_mask,
)
from weiss_rl.learners.structured_auxiliary import PackedStructuredLegalView


def test_packed_action_logp_matches_dense_mask_with_empty_pass_row() -> None:
    logits = torch.tensor([[[1.0, 2.0, 0.0, -1.0], [0.5, -0.5, 1.5, 2.5], [0.0, 1.0, 2.0, 3.0]]])
    legal_mask = torch.tensor([[[True, True, False, False], [False, False, False, False], [False, False, True, True]]])
    actions = torch.tensor([[1, 0, 3]])
    legal_ids = torch.tensor([0, 1, 2, 3])
    legal_offsets = torch.tensor([0, 2, 2, 4])

    dense_logp, dense_entropy = masked_action_logp_and_entropy(
        logits,
        legal_mask,
        actions,
        pass_action_id=0,
    )
    packed_logp, packed_entropy = packed_action_logp_and_entropy(
        logits,
        legal_ids,
        legal_offsets,
        actions,
        pass_action_id=0,
    )

    assert torch.allclose(packed_logp, dense_logp)
    assert torch.allclose(packed_entropy, dense_entropy)


def test_packed_selected_action_logp_strict_false_marks_unsupported_rows() -> None:
    packed_logits = torch.tensor([1.0, 2.0, 0.5])
    legal_ids = torch.tensor([0, 1, 3])
    legal_offsets = torch.tensor([0, 2, 3, 3])
    actions = torch.tensor([1, 2, 0])

    selected = packed_selected_action_logp(
        packed_logits,
        legal_ids,
        legal_offsets,
        actions,
        pass_action_id=0,
        strict=False,
    )

    expected_row0 = torch.log_softmax(torch.tensor([1.0, 2.0]), dim=0)[1]
    assert selected[0] == pytest.approx(float(expected_row0))
    assert torch.isneginf(selected[1])
    assert selected[2] == pytest.approx(0.0)

    with pytest.raises(ValueError, match="illegal action 2 for row 1"):
        packed_selected_action_logp(
            packed_logits,
            legal_ids,
            legal_offsets,
            actions,
            pass_action_id=0,
            strict=True,
        )


def test_packed_scores_action_logp_and_entropy_matches_dense_packed_scores() -> None:
    logits = torch.tensor([[[1.0, 2.0, 0.0, -1.0], [0.5, -0.5, 1.5, 2.5]]])
    actions = torch.tensor([[1, 3]])
    legal_ids = torch.tensor([0, 1, 2, 3])
    legal_offsets = torch.tensor([0, 2, 4])
    packed_logits = torch.tensor([1.0, 2.0, 1.5, 2.5])

    packed_logp, packed_entropy = packed_action_logp_and_entropy(
        logits,
        legal_ids,
        legal_offsets,
        actions,
        pass_action_id=None,
    )
    score_logp, score_entropy = packed_scores_action_logp_and_entropy(
        packed_logits,
        legal_ids,
        legal_offsets,
        actions,
        pass_action_id=None,
    )

    assert torch.allclose(score_logp, packed_logp)
    assert torch.allclose(score_entropy, packed_entropy)


def test_packed_scores_family_entropy_collapses_candidates_by_family() -> None:
    packed_logits = torch.zeros(5, dtype=torch.float32)
    legal_offsets = torch.tensor([0, 3, 5], dtype=torch.long)
    legal_action_meta = torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
        ],
        dtype=torch.long,
    )

    entropy = packed_scores_family_entropy(
        packed_logits,
        legal_offsets,
        legal_action_meta,
        row_shape=(2,),
        family_count=2,
    )

    expected_row0 = -((2.0 / 3.0) * np.log(2.0 / 3.0) + (1.0 / 3.0) * np.log(1.0 / 3.0))
    assert entropy[0] == pytest.approx(expected_row0)
    assert entropy[1] == pytest.approx(0.0)


def test_packed_subset_action_logp_reports_top_actions_and_unsupported_rows() -> None:
    packed_view = PackedStructuredLegalView(
        row_count=2,
        row_indices=torch.tensor([0, 0, 1]),
        action_ids=torch.tensor([2, 4, 3]),
        logits=torch.tensor([1.0, 2.0, 0.0]),
        row_log_z=torch.zeros(2),
        row_has_candidates=torch.tensor([True, True]),
        family_ids=torch.zeros(3, dtype=torch.long),
        arg0=torch.zeros(3, dtype=torch.long),
        arg1=torch.zeros(3, dtype=torch.long),
        arg2=torch.zeros(3, dtype=torch.long),
    )
    actions = torch.tensor([4, 3])

    selected_logp, top_actions = packed_subset_action_logp_and_top_action(
        packed_view,
        actions,
        candidate_mask=torch.tensor([True, True, True]),
        strict=True,
    )

    expected_row0 = torch.log_softmax(torch.tensor([1.0, 2.0]), dim=0)[1]
    assert selected_logp[0] == pytest.approx(float(expected_row0))
    assert selected_logp[1] == pytest.approx(0.0)
    assert top_actions.tolist() == [4, 3]

    unsupported_logp, unsupported_top = packed_subset_action_logp_and_top_action(
        packed_view,
        actions,
        candidate_mask=torch.tensor([True, False, True]),
        strict=False,
    )

    assert torch.isneginf(unsupported_logp[0])
    assert unsupported_logp[1] == pytest.approx(0.0)
    assert unsupported_top.tolist() == [2, 3]


def test_impala_private_wrappers_delegate_to_action_logp_helpers() -> None:
    logits = torch.tensor([[[0.0, 1.0, -1.0]]])
    mask = torch.tensor([[[True, True, False]]])
    actions = torch.tensor([[1]])

    public_logp, public_entropy = masked_action_logp_and_entropy(logits, mask, actions, pass_action_id=None)
    private_logp, private_entropy = _masked_action_logp_and_entropy(logits, mask, actions, pass_action_id=None)

    assert _masked_action_logp_and_entropy is not masked_action_logp_and_entropy
    assert torch.allclose(private_logp, public_logp)
    assert torch.allclose(private_entropy, public_entropy)

    packed_logits = torch.tensor([0.0, 1.0])
    legal_ids = torch.tensor([0, 1])
    legal_offsets = torch.tensor([0, 2])
    public_selected = packed_selected_action_logp(
        packed_logits,
        legal_ids,
        legal_offsets,
        actions,
        pass_action_id=None,
    )
    private_selected = _packed_selected_action_logp(
        packed_logits,
        legal_ids,
        legal_offsets,
        actions,
        pass_action_id=None,
    )

    assert _packed_selected_action_logp is not packed_selected_action_logp
    assert torch.allclose(private_selected, public_selected)


def test_numpy_learner_logp_facades_delegate_through_impala_compatibility_names() -> None:
    logits = np.asarray([[2.0, -1.0, 0.0, 3.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    legal_mask = np.asarray([[1, 0, 0, 1], [0, 0, 0, 0]], dtype=np.uint8)
    actions = np.asarray([3, 0], dtype=np.int64)
    legal_ids = np.asarray([0, 3], dtype=np.uint32)
    legal_offsets = np.asarray([0, 2, 2], dtype=np.uint32)

    direct_mask_logp = learner_logp_from_mask(logits, legal_mask, actions, pass_action_id=0)
    wrapper_mask_logp = impala_learner_logp_from_mask(logits, legal_mask, actions, pass_action_id=0)
    direct_ids_logp = learner_logp_from_legal_ids(logits, legal_ids, legal_offsets, actions, pass_action_id=0)
    wrapper_ids_logp = impala_learner_logp_from_legal_ids(logits, legal_ids, legal_offsets, actions, pass_action_id=0)

    assert impala_learner_logp_from_mask is not learner_logp_from_mask
    assert impala_learner_logp_from_legal_ids is not learner_logp_from_legal_ids
    assert np.allclose(wrapper_mask_logp, direct_mask_logp)
    assert np.allclose(wrapper_ids_logp, direct_ids_logp)
