from __future__ import annotations

from .test_impala_learner import (
    ImpalaBatchSupportMixin,
    ImpalaFaultSupportMixin,
    ImpalaForwardSupportMixin,
    ImpalaLearner,
    ImpalaLoggingSupportMixin,
    ImpalaPairedOutcomeAuxiliaryMixin,
    ImpalaPairedSwingAuxiliaryMixin,
    ImpalaPolicyAnchorSupportMixin,
    ImpalaPublicHeuristicSupportMixin,
    ImpalaStructuredTeacherAuxiliaryMixin,
    _chosen_action_outcome_metrics,
    chosen_action_outcome_metrics_impl,
    compute_structured_teacher_auxiliary_metrics,
    compute_structured_teacher_auxiliary_metrics_impl,
)


def test_impala_learner_reexports_structured_teacher_auxiliary_metrics() -> None:
    assert compute_structured_teacher_auxiliary_metrics is compute_structured_teacher_auxiliary_metrics_impl


def test_impala_learner_reexports_chosen_action_outcome_metrics() -> None:
    assert _chosen_action_outcome_metrics is chosen_action_outcome_metrics_impl


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
