"""Support-method mixin for :class:`weiss_rl.learners.impala.ImpalaLearner`."""

from __future__ import annotations

from weiss_rl.learners.impala.batching.batch_support import ImpalaBatchSupportMixin
from weiss_rl.learners.impala.support.fault_support import ImpalaFaultSupportMixin
from weiss_rl.learners.impala.support.forward_support import ImpalaForwardSupportMixin
from weiss_rl.learners.impala.support.logging_support import ImpalaLoggingSupportMixin
from weiss_rl.learners.impala.support.optimizer_support import ImpalaOptimizerSupportMixin
from weiss_rl.learners.impala.support.public_heuristic_support import ImpalaPublicHeuristicSupportMixin


class ImpalaSupportMixin(
    ImpalaPublicHeuristicSupportMixin,
    ImpalaForwardSupportMixin,
    ImpalaLoggingSupportMixin,
    ImpalaBatchSupportMixin,
    ImpalaFaultSupportMixin,
    ImpalaOptimizerSupportMixin,
):
    pass
