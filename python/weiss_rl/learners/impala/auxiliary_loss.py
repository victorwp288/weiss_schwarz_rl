"""Auxiliary loss composition mixin for the IMPALA learner."""

from __future__ import annotations

from weiss_rl.learners.impala.paired_outcome_auxiliary import ImpalaPairedOutcomeAuxiliaryMixin
from weiss_rl.learners.impala.paired_swing_auxiliary import ImpalaPairedSwingAuxiliaryMixin
from weiss_rl.learners.impala.structured_teacher_auxiliary import ImpalaStructuredTeacherAuxiliaryMixin


class ImpalaAuxiliaryLossMixin(
    ImpalaStructuredTeacherAuxiliaryMixin,
    ImpalaPairedOutcomeAuxiliaryMixin,
    ImpalaPairedSwingAuxiliaryMixin,
):
    """Compose the behavior-sensitive IMPALA auxiliary loss families."""


__all__ = ["ImpalaAuxiliaryLossMixin"]
