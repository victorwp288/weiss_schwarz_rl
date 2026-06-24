from __future__ import annotations

import torch
from weiss_rl.learners.impala.auxiliary.paired_outcome_candidates import compute_paired_outcome_candidate_logps
from weiss_rl.learners.impala.batching.paired_auxiliary_batch import resolve_paired_auxiliary_batch_inputs

from tests.weiss_rl.impala_paired_auxiliary_test_support import make_factorized_paired_outcome_case


def test_compute_paired_outcome_candidate_logps_preserves_current_reference_views() -> None:
    model, learner, batch = make_factorized_paired_outcome_case()
    inputs = resolve_paired_auxiliary_batch_inputs(
        learner,
        batch,
        packed_legal_error="paired outcome preference replay requires packed legal_ids/legal_offsets",
    )
    actions = learner._require_actions(batch["actions"], expected_shape=inputs.expected_shape)

    candidate_logps = compute_paired_outcome_candidate_logps(
        learner,
        batch,
        obs=inputs.obs,
        packed_legal=inputs.packed_legal,
        actions=actions,
        reset_before_step=None,
    )

    assert candidate_logps.current_action_logp.shape == torch.Size([2, 1])
    assert candidate_logps.reference_action_logp.shape == torch.Size([2, 1])
    assert candidate_logps.current_best_non_target_logp.shape == torch.Size([2, 1])
    assert candidate_logps.reference_best_non_target_logp.shape == torch.Size([2, 1])
    assert torch.allclose(candidate_logps.current_action_logp, candidate_logps.reference_action_logp)
    assert torch.allclose(candidate_logps.current_best_non_target_logp, candidate_logps.reference_best_non_target_logp)
    assert candidate_logps.current_action_logp[0, 0] > candidate_logps.current_best_non_target_logp[0, 0]
    assert candidate_logps.current_action_logp[1, 0] < candidate_logps.current_best_non_target_logp[1, 0]
    assert model.factorized_candidate_logp_calls == 1
