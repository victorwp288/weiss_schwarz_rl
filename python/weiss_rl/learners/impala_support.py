"""Support-method mixin for :class:`weiss_rl.learners.impala_learner.ImpalaLearner`."""

from __future__ import annotations

from weiss_rl.learners.impala_batch_support import ImpalaBatchSupportMixin
from weiss_rl.learners.impala_fault_support import ImpalaFaultSupportMixin
from weiss_rl.learners.impala_forward_support import ImpalaForwardSupportMixin
from weiss_rl.learners.impala_logging_support import ImpalaLoggingSupportMixin
from weiss_rl.learners.impala_optimizer_support import ImpalaOptimizerSupportMixin
from weiss_rl.learners.impala_public_heuristic_support import ImpalaPublicHeuristicSupportMixin


class ImpalaSupportMixin(
    ImpalaPublicHeuristicSupportMixin,
    ImpalaForwardSupportMixin,
    ImpalaLoggingSupportMixin,
    ImpalaBatchSupportMixin,
    ImpalaFaultSupportMixin,
    ImpalaOptimizerSupportMixin,
):
    pass
