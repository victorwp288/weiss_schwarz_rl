"""IMPALA update logging side effects."""

from __future__ import annotations

import time
from typing import Any


def log_impala_update_metrics_if_due(
    *,
    learner: Any,
    batch: Any,
    metrics: dict[str, float],
    now: float | None = None,
) -> bool:
    if not learner.logger:
        return False
    if learner.update_count % learner.logging_interval_updates != 0:
        return False
    learner._log_metrics(metrics, batch)
    learner.last_log_time = time.time() if now is None else float(now)
    learner.last_log_update = learner.update_count
    return True


__all__ = ["log_impala_update_metrics_if_due"]
