"""IMPALA learner logging and checkpoint metadata support."""

from __future__ import annotations

import time
from typing import Any

from weiss_rl.learners.impala.batch_support import _batch_value
from weiss_rl.learners.logging import (
    build_training_metrics,
    custom_log_metrics,
    write_checkpoint_metadata,
)
from weiss_rl.learners.vtrace import VtraceMetrics, compute_vtrace_metrics


class ImpalaLoggingSupportMixin:
    def _write_checkpoint_metadata(self: Any) -> None:
        checkpoint_metadata_path = write_checkpoint_metadata(
            checkpoint_dir=self.checkpoint_dir,
            update_count=self.update_count,
            policy_version=self.policy_version,
        )
        if checkpoint_metadata_path is not None:
            print(f"Saved checkpoint metadata: {checkpoint_metadata_path}")

    def _log_metrics(
        self: Any, update_metrics: dict[str, float], batch: Any, *, context: dict[str, Any] | None = None
    ) -> None:
        if not self.logger:
            return

        rho_bar_value = _batch_value(batch, "vtrace_rho_bar")
        c_bar_value = _batch_value(batch, "vtrace_c_bar")
        vtrace_metrics = compute_vtrace_metrics(
            batch,
            rho_bar=self.vtrace_rho_bar if rho_bar_value is None else float(rho_bar_value),
            c_bar=self.vtrace_c_bar if c_bar_value is None else float(c_bar_value),
            pass_action_id=self.pass_action_id,
        )
        elapsed = time.time() - self.start_time
        metrics = build_training_metrics(
            update_metrics=update_metrics,
            vtrace_metrics=vtrace_metrics,
            update_count=self.update_count,
            policy_version=self.policy_version,
            elapsed_seconds=elapsed,
        )
        self.logger.log(metrics)

    def _custom_log_metrics(
        self: Any,
        update_metrics: dict[str, float],
        vtrace_metrics: VtraceMetrics,
    ) -> dict[str, float]:
        return custom_log_metrics(update_metrics, vtrace_metrics)

    def get_policy_version(self: Any) -> int:
        return self.policy_version


__all__ = ["ImpalaLoggingSupportMixin"]
