from __future__ import annotations

import json
from pathlib import Path

from .entrypoints_test_support import (
    _copy_repo_configs,
    _run_entrypoint,
    _write_b1_baseline_run_fixture,
    _write_runtime_weiss_sim,
)


def test_train_entrypoint_profile_timers_does_not_emit_torch_profiler_trace(tmp_path: Path) -> None:
    bundle = _write_runtime_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _copy_repo_configs(tmp_path)
    b1_baseline_run_dir = _write_b1_baseline_run_fixture(tmp_path, stack_config=stack_config)
    run_label = "profile_timers_no_trace"
    result = _run_entrypoint(
        tmp_path,
        module_name="weiss_rl.training.train_entrypoint",
        stack_config=stack_config,
        spec_hash=str(bundle["spec_hash"]),
        run_label=run_label,
        extra_args=[
            "--device",
            "cpu",
            "--num-envs",
            "1",
            "--unroll-length",
            "1",
            "--max-updates",
            "1",
            "--b1-baseline-run-dir",
            str(b1_baseline_run_dir),
            "--profile-timers",
        ],
    )

    assert result.returncode == 0, result.stderr
    run_root = tmp_path / "runs" / run_label
    assert not (run_root / "profiling" / "torch_profiler" / "trace.json").exists()
    run_summary = json.loads((run_root / "run_summary.json").read_text(encoding="utf-8"))
    determinism = json.loads((run_root / "determinism_report.json").read_text(encoding="utf-8"))
    assert run_summary["training_controls"]["profile_timers"] is True
    assert run_summary["training_controls"]["torch_profiler"] is False
    assert determinism["training_controls"]["profile_timers"] is True
    assert determinism["training_controls"]["torch_profiler"] is False


def test_train_entrypoint_emits_torch_profiler_trace(tmp_path: Path) -> None:
    bundle = _write_runtime_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _copy_repo_configs(tmp_path)
    b1_baseline_run_dir = _write_b1_baseline_run_fixture(tmp_path, stack_config=stack_config)
    run_label = "torch_profiler_trace"
    result = _run_entrypoint(
        tmp_path,
        module_name="weiss_rl.training.train_entrypoint",
        stack_config=stack_config,
        spec_hash=str(bundle["spec_hash"]),
        run_label=run_label,
        extra_args=[
            "--device",
            "cpu",
            "--num-envs",
            "1",
            "--unroll-length",
            "1",
            "--max-updates",
            "1",
            "--b1-baseline-run-dir",
            str(b1_baseline_run_dir),
            "--torch-profiler",
        ],
    )

    assert result.returncode == 0, result.stderr
    run_root = tmp_path / "runs" / run_label
    assert (run_root / "profiling" / "torch_profiler" / "trace.json").exists()
    run_summary = json.loads((run_root / "run_summary.json").read_text(encoding="utf-8"))
    determinism = json.loads((run_root / "determinism_report.json").read_text(encoding="utf-8"))
    assert run_summary["training_controls"]["profile_timers"] is False
    assert run_summary["training_controls"]["torch_profiler"] is True
    assert determinism["training_controls"]["profile_timers"] is False
    assert determinism["training_controls"]["torch_profiler"] is True
