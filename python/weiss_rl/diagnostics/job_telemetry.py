"""Whole-job telemetry sampling and summary helpers for local training runs."""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil


@dataclass(frozen=True, slots=True)
class JobTelemetryConfig:
    sample_interval_seconds: float = 2.0


def sample_process_tree(root_pid: int) -> dict[str, Any]:
    root = psutil.Process(int(root_pid))
    processes = [root, *root.children(recursive=True)]
    live_processes: list[psutil.Process] = []
    cpu_percent_total = 0.0
    rss_bytes_total = 0
    vms_bytes_total = 0
    thread_count_total = 0
    handle_count_total = 0
    per_process: list[dict[str, Any]] = []
    for process in processes:
        try:
            with process.oneshot():
                cpu_percent = float(process.cpu_percent(interval=None))
                memory = process.memory_info()
                thread_count = int(process.num_threads())
                handle_count = int(process.num_handles()) if hasattr(process, "num_handles") else 0
                name = process.name()
            cpu_percent_total += cpu_percent
            rss_bytes_total += int(memory.rss)
            vms_bytes_total += int(memory.vms)
            thread_count_total += thread_count
            handle_count_total += handle_count
            live_processes.append(process)
            per_process.append(
                {
                    "pid": int(process.pid),
                    "name": str(name),
                    "cpu_percent": cpu_percent,
                    "rss_bytes": int(memory.rss),
                    "threads": thread_count,
                    "handles": handle_count,
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    per_process.sort(key=lambda item: (-float(item["cpu_percent"]), -int(item["rss_bytes"]), int(item["pid"])))
    return {
        "root_pid": int(root_pid),
        "process_count": len(live_processes),
        "cpu_percent_total": cpu_percent_total,
        "rss_bytes_total": rss_bytes_total,
        "vms_bytes_total": vms_bytes_total,
        "thread_count_total": thread_count_total,
        "handle_count_total": handle_count_total,
        "top_processes": per_process[:8],
    }


def prime_process_tree_cpu(root_pid: int) -> None:
    try:
        root = psutil.Process(int(root_pid))
    except psutil.Error:
        return
    for process in [root, *root.children(recursive=True)]:
        try:
            process.cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


@dataclass(slots=True)
class ProcessTreeTelemetrySampler:
    _previous_ts: float | None = None
    _previous_cpu_totals: dict[tuple[int, float], float] | None = None

    def sample(self, root_pid: int) -> dict[str, Any]:
        root = psutil.Process(int(root_pid))
        now = time.time()
        previous = {} if self._previous_cpu_totals is None else dict(self._previous_cpu_totals)
        current: dict[tuple[int, float], float] = {}
        per_process: list[dict[str, Any]] = []
        rss_bytes_total = 0
        vms_bytes_total = 0
        thread_count_total = 0
        handle_count_total = 0
        cpu_percent_total = 0.0
        live_count = 0
        delta_wall = None if self._previous_ts is None else max(now - self._previous_ts, 1e-6)
        for process in [root, *root.children(recursive=True)]:
            try:
                with process.oneshot():
                    create_time = float(process.create_time())
                    memory = process.memory_info()
                    thread_count = int(process.num_threads())
                    handle_count = int(process.num_handles()) if hasattr(process, "num_handles") else 0
                    name = process.name()
                    cpu_times = process.cpu_times()
                ident = (int(process.pid), create_time)
                cpu_total = float(cpu_times.user + cpu_times.system)
                previous_total = previous.get(ident)
                process_cpu_percent = 0.0
                if previous_total is not None and delta_wall is not None:
                    process_cpu_percent = (max(cpu_total - previous_total, 0.0) / delta_wall) * 100.0
                current[ident] = cpu_total
                cpu_percent_total += process_cpu_percent
                rss_bytes_total += int(memory.rss)
                vms_bytes_total += int(memory.vms)
                thread_count_total += thread_count
                handle_count_total += handle_count
                live_count += 1
                per_process.append(
                    {
                        "pid": int(process.pid),
                        "name": str(name),
                        "cpu_percent": process_cpu_percent,
                        "rss_bytes": int(memory.rss),
                        "threads": thread_count,
                        "handles": handle_count,
                    }
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        per_process.sort(key=lambda item: (-float(item["cpu_percent"]), -int(item["rss_bytes"]), int(item["pid"])))
        self._previous_ts = now
        self._previous_cpu_totals = current
        return {
            "root_pid": int(root_pid),
            "process_count": live_count,
            "cpu_percent_total": cpu_percent_total,
            "rss_bytes_total": rss_bytes_total,
            "vms_bytes_total": vms_bytes_total,
            "thread_count_total": thread_count_total,
            "handle_count_total": handle_count_total,
            "top_processes": per_process[:8],
        }


def query_gpu_metrics() -> dict[str, Any] | None:
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=5)
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return None
    first = lines[0].split(",")
    if len(first) < 4:
        return None
    try:
        util, mem_used, mem_total, power_draw = [float(part.strip()) for part in first[:4]]
    except ValueError:
        return None
    return {
        "util": util,
        "mem_used_mb": mem_used,
        "mem_total_mb": mem_total,
        "power_draw_w": power_draw,
    }


def write_telemetry_sample(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize_numeric_series(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    total = float(sum(values))
    count = float(len(values))
    return {
        "mean": total / count,
        "max": max(values),
        "min": min(values),
    }


def summarize_job_telemetry(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    process_counts = [float(sample["process"]["process_count"]) for sample in samples if sample.get("process")]
    cpu_percents = [float(sample["process"]["cpu_percent_total"]) for sample in samples if sample.get("process")]
    rss_bytes = [float(sample["process"]["rss_bytes_total"]) for sample in samples if sample.get("process")]
    thread_counts = [float(sample["process"]["thread_count_total"]) for sample in samples if sample.get("process")]
    handle_counts = [float(sample["process"]["handle_count_total"]) for sample in samples if sample.get("process")]
    gpu_util = [float(sample["gpu"]["util"]) for sample in samples if sample.get("gpu")]
    gpu_mem = [float(sample["gpu"]["mem_used_mb"]) for sample in samples if sample.get("gpu")]
    gpu_power = [float(sample["gpu"]["power_draw_w"]) for sample in samples if sample.get("gpu")]
    last_process = None
    for sample in reversed(samples):
        process_payload = sample.get("process")
        if isinstance(process_payload, Mapping):
            last_process = process_payload
            break
    return {
        "format": "job_telemetry_summary_v1",
        "sample_count": int(len(samples)),
        "process_count": summarize_numeric_series(process_counts),
        "cpu_percent_total": summarize_numeric_series(cpu_percents),
        "rss_bytes_total": summarize_numeric_series(rss_bytes),
        "thread_count_total": summarize_numeric_series(thread_counts),
        "handle_count_total": summarize_numeric_series(handle_counts),
        "gpu_util": summarize_numeric_series(gpu_util),
        "gpu_mem_used_mb": summarize_numeric_series(gpu_mem),
        "gpu_power_draw_w": summarize_numeric_series(gpu_power),
        "top_processes_last": list(last_process.get("top_processes", [])) if isinstance(last_process, Mapping) else [],
    }


def summarize_training_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"record_count": 0}
    throughput_samples = [float(record.get("throughput_samples_per_sec", 0.0)) for record in records]
    throughput_updates = [float(record.get("throughput_updates_per_sec", 0.0)) for record in records]
    grad_overflows = [float(record.get("amp_grad_overflow", 0.0)) for record in records]
    return {
        "record_count": int(len(records)),
        "throughput_samples_per_sec": summarize_numeric_series(throughput_samples),
        "throughput_updates_per_sec": summarize_numeric_series(throughput_updates),
        "amp_grad_overflow_count": int(sum(1 for value in grad_overflows if value > 0.5)),
        "last_loss": float(records[-1].get("loss", 0.0)),
    }


def build_benchmark_summary(*, run_dir: Path, telemetry_path: Path) -> dict[str, Any]:
    telemetry_samples = load_jsonl(telemetry_path)
    training_metrics = load_jsonl(run_dir / "training" / "logs" / "training_metrics.jsonl")
    performance_metrics = load_jsonl(run_dir / "training" / "logs" / "performance.jsonl")
    return {
        "format": "profiled_train_summary_v1",
        "run_dir": run_dir.resolve().as_posix(),
        "telemetry": summarize_job_telemetry(telemetry_samples),
        "training_metrics": summarize_training_metrics(training_metrics),
        "performance_record_count": int(len(performance_metrics)),
    }


def sample_job_once(*, root_pid: int) -> dict[str, Any]:
    process_payload = None
    try:
        process_payload = sample_process_tree(root_pid)
    except psutil.Error:
        process_payload = None
    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "root_pid": int(root_pid),
        "process": process_payload,
        "gpu": query_gpu_metrics(),
    }


__all__ = [
    "JobTelemetryConfig",
    "ProcessTreeTelemetrySampler",
    "build_benchmark_summary",
    "load_jsonl",
    "prime_process_tree_cpu",
    "query_gpu_metrics",
    "sample_job_once",
    "sample_process_tree",
    "summarize_job_telemetry",
    "summarize_training_metrics",
    "write_telemetry_sample",
]
