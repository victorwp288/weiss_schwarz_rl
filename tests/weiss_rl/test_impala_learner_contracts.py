from __future__ import annotations

import weiss_rl.learners.impala as impala_root
from weiss_rl.learners.impala import ImpalaLearner
from weiss_rl.learners.impala.batch_support import ImpalaBatchSupportMixin
from weiss_rl.learners.impala.fault_support import ImpalaFaultSupportMixin
from weiss_rl.learners.impala.forward_support import ImpalaForwardSupportMixin
from weiss_rl.learners.impala.logging_support import ImpalaLoggingSupportMixin
from weiss_rl.learners.impala.paired_outcome_auxiliary import ImpalaPairedOutcomeAuxiliaryMixin
from weiss_rl.learners.impala.paired_swing_auxiliary import ImpalaPairedSwingAuxiliaryMixin
from weiss_rl.learners.impala.policy_anchor_support import ImpalaPolicyAnchorSupportMixin
from weiss_rl.learners.impala.public_heuristic_support import ImpalaPublicHeuristicSupportMixin
from weiss_rl.learners.impala.structured_teacher_auxiliary import ImpalaStructuredTeacherAuxiliaryMixin


def test_impala_learner_root_star_export_is_learner_only() -> None:
    assert impala_root.__all__ == ("ImpalaLearner",)


def test_impala_learner_root_does_not_reexport_internal_dependencies() -> None:
    retired_names = (
        "ImpalaAuxiliaryLossMixin",
        "ImpalaFactorizedEvaluationMixin",
        "ImpalaPolicyAnchorSupportMixin",
        "ImpalaSupportMixin",
        "ImpalaUpdateLoopMixin",
        "TrainingLogger",
        "compute_impala_loss_and_metrics_with_context",
        "learner_acceleration_state",
        "normalize_public_heuristic_profile_mode",
        "normalize_public_heuristic_profiles",
        "record_timing_ms",
        "should_emit_structured_metrics",
        "teacher_aux_active",
    )

    for name in retired_names:
        assert not hasattr(impala_root, name), name


def test_impala_learner_root_does_not_reexport_structured_teacher_auxiliary_metrics() -> None:
    assert not hasattr(impala_root, "compute_structured_teacher_auxiliary_metrics")


def test_impala_learner_root_does_not_reexport_private_helper_facades() -> None:
    retired_names = (
        "_ForwardTimeMajorResult",
        "_SUPPORTED_PUBLIC_HEURISTIC_PROFILE_MODES",
        "_SUPPORTED_PUBLIC_HEURISTIC_PROFILES",
        "_masked_log_probs_and_entropy",
        "_nonfinite_indices",
        "_normalize_public_heuristic_profile_mode",
        "_normalize_public_heuristic_profiles",
        "_packed_group_log_probs",
        "_packed_scores_action_logp_and_entropy",
        "_packed_scores_family_entropy",
        "_packed_soft_target_cross_entropy",
        "_packed_structured_legal_view",
        "_packed_subset_action_logp_and_top_action",
        "_resolve_public_heuristic_family_ids",
        "_segment_group_sum",
        "_segment_logsumexp",
        "_segment_max",
        "_structured_catalog_metadata",
        "_time_step_legal_actions",
        "_weighted_mean",
    )

    for name in retired_names:
        assert not hasattr(impala_root, name), name


def test_impala_learner_root_does_not_reexport_chosen_action_outcome_metrics() -> None:
    assert not hasattr(impala_root, "_chosen_action_outcome_metrics")
    assert not hasattr(impala_root, "chosen_action_outcome_metrics")


def test_impala_learner_uses_canonical_paired_swing_auxiliary_mixin() -> None:
    assert isinstance(ImpalaLearner(), ImpalaPairedSwingAuxiliaryMixin)


def test_impala_learner_uses_canonical_paired_outcome_auxiliary_mixin() -> None:
    assert isinstance(ImpalaLearner(), ImpalaPairedOutcomeAuxiliaryMixin)


def test_impala_learner_uses_canonical_structured_teacher_auxiliary_mixin() -> None:
    assert isinstance(ImpalaLearner(), ImpalaStructuredTeacherAuxiliaryMixin)


def test_impala_learner_uses_canonical_fault_support_mixin() -> None:
    assert isinstance(ImpalaLearner(), ImpalaFaultSupportMixin)


def test_impala_learner_uses_canonical_public_heuristic_support_mixin() -> None:
    assert isinstance(ImpalaLearner(), ImpalaPublicHeuristicSupportMixin)


def test_impala_learner_uses_canonical_batch_support_mixin() -> None:
    assert isinstance(ImpalaLearner(), ImpalaBatchSupportMixin)


def test_impala_learner_uses_canonical_forward_support_mixin() -> None:
    assert isinstance(ImpalaLearner(), ImpalaForwardSupportMixin)


def test_impala_learner_uses_canonical_logging_support_mixin() -> None:
    assert isinstance(ImpalaLearner(), ImpalaLoggingSupportMixin)


def test_impala_learner_uses_canonical_policy_anchor_support_mixin() -> None:
    assert isinstance(ImpalaLearner(), ImpalaPolicyAnchorSupportMixin)
