from __future__ import annotations

from types import SimpleNamespace

import pytest
from weiss_rl.learners.impala.update_logging import log_impala_update_metrics_if_due


def test_log_impala_update_metrics_if_due_preserves_interval_and_timestamp_contract() -> None:
    calls: list[tuple[dict[str, float], dict[str, bool]]] = []
    metrics = {"loss": 1.0}
    batch = {"batch": True}
    learner = SimpleNamespace(
        logger=object(),
        update_count=6,
        logging_interval_updates=3,
        last_log_time=0.0,
        last_log_update=0,
        _log_metrics=lambda logged_metrics, logged_batch: calls.append((logged_metrics, logged_batch)),
    )

    logged = log_impala_update_metrics_if_due(
        learner=learner,
        batch=batch,
        metrics=metrics,
        now=123.5,
    )

    assert logged is True
    assert calls == [(metrics, batch)]
    assert learner.last_log_time == pytest.approx(123.5)
    assert learner.last_log_update == 6


def test_log_impala_update_metrics_if_due_skips_without_logger_or_interval() -> None:
    calls: list[str] = []
    metrics = {"loss": 1.0}
    batch = {"batch": True}
    no_logger = SimpleNamespace(
        logger=None,
        update_count=6,
        logging_interval_updates=3,
        last_log_time=0.0,
        last_log_update=0,
        _log_metrics=lambda _metrics, _batch: calls.append("no_logger"),
    )
    off_interval = SimpleNamespace(
        logger=object(),
        update_count=5,
        logging_interval_updates=3,
        last_log_time=0.0,
        last_log_update=0,
        _log_metrics=lambda _metrics, _batch: calls.append("off_interval"),
    )

    assert log_impala_update_metrics_if_due(learner=no_logger, batch=batch, metrics=metrics, now=1.0) is False
    assert log_impala_update_metrics_if_due(learner=off_interval, batch=batch, metrics=metrics, now=1.0) is False
    assert calls == []
    assert no_logger.last_log_time == pytest.approx(0.0)
    assert no_logger.last_log_update == 0
    assert off_interval.last_log_time == pytest.approx(0.0)
    assert off_interval.last_log_update == 0
