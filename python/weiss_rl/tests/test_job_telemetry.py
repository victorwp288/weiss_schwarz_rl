from __future__ import annotations

from weiss_rl.job_telemetry import summarize_job_telemetry, summarize_training_metrics


def test_summarize_job_telemetry_computes_means_and_peaks() -> None:
    samples = [
        {
            "process": {
                "process_count": 3,
                "cpu_percent_total": 120.0,
                "rss_bytes_total": 2_000_000,
                "thread_count_total": 20,
                "handle_count_total": 100,
                "top_processes": [{"pid": 1, "name": "python"}],
            },
            "gpu": {"util": 25.0, "mem_used_mb": 2048.0, "power_draw_w": 120.0},
        },
        {
            "process": {
                "process_count": 4,
                "cpu_percent_total": 180.0,
                "rss_bytes_total": 3_000_000,
                "thread_count_total": 28,
                "handle_count_total": 130,
                "top_processes": [{"pid": 2, "name": "python"}],
            },
            "gpu": {"util": 40.0, "mem_used_mb": 3072.0, "power_draw_w": 180.0},
        },
    ]

    summary = summarize_job_telemetry(samples)

    assert summary["sample_count"] == 2
    assert summary["cpu_percent_total"]["mean"] == 150.0
    assert summary["cpu_percent_total"]["max"] == 180.0
    assert summary["gpu_util"]["mean"] == 32.5
    assert summary["gpu_mem_used_mb"]["max"] == 3072.0
    assert summary["top_processes_last"][0]["pid"] == 2


def test_summarize_training_metrics_tracks_overflows() -> None:
    records = [
        {"throughput_samples_per_sec": 100.0, "throughput_updates_per_sec": 1.0, "amp_grad_overflow": 0.0, "loss": 0.5},
        {"throughput_samples_per_sec": 140.0, "throughput_updates_per_sec": 2.0, "amp_grad_overflow": 1.0, "loss": 0.2},
    ]

    summary = summarize_training_metrics(records)

    assert summary["record_count"] == 2
    assert summary["throughput_samples_per_sec"]["mean"] == 120.0
    assert summary["throughput_updates_per_sec"]["max"] == 2.0
    assert summary["amp_grad_overflow_count"] == 1
    assert summary["last_loss"] == 0.2
