from __future__ import annotations

import json

from weiss_rl.runtime.components.ipc_shared.logging import PerformanceLogger, process_debug_log


def test_performance_logger_writes_sorted_jsonl(tmp_path) -> None:
    log_path = tmp_path / "logs" / "perf.jsonl"
    logger = PerformanceLogger(log_path)

    logger.log({"z": 2, "a": 1})
    logger.log({"b": 3})

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert lines == ['{"a": 1, "z": 2}', '{"b": 3}']
    assert [json.loads(line) for line in lines] == [{"a": 1, "z": 2}, {"b": 3}]


def test_process_debug_log_respects_env_flag(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("WEISS_RL_PROCESS_DEBUG", raising=False)

    process_debug_log(run_dir=tmp_path, actor_id=3, message="ignored")
    assert not (tmp_path / "training").exists()

    monkeypatch.setenv("WEISS_RL_PROCESS_DEBUG", "yes")
    process_debug_log(run_dir=tmp_path, actor_id=3, message="accepted")

    log_path = tmp_path / "training" / "logs" / "collector_debug_actor03.log"
    line = log_path.read_text(encoding="utf-8").strip()
    timestamp, message = line.split(" ", 1)
    assert float(timestamp) > 0.0
    assert message == "accepted"


def test_process_debug_log_ignores_missing_run_dir(monkeypatch) -> None:
    monkeypatch.setenv("WEISS_RL_PROCESS_DEBUG", "1")

    process_debug_log(run_dir=None, actor_id=0, message="ignored")
