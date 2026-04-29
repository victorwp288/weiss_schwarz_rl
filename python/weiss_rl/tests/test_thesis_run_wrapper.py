from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_thesis_run_wrapper_dry_run_writes_plan(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "configs").mkdir(parents=True, exist_ok=True)
    stack_config = repo_root / "configs" / "stack.yaml"
    stack_config.write_text("components: []\nconfig: {}\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "python" / "scripts" / "thesis_run.py"),
            "--repo-root",
            str(repo_root),
            "--stack-config",
            str(stack_config),
            "--run-label",
            "demo_run",
            "--dry-run",
            "--compare-run-dir",
            str(repo_root / "runs" / "baseline_a"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary_path = repo_root / "runs" / "_wrapper_plans" / "demo_run.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["status"] == "planned"
    assert len(payload["steps"]) == 3
    assert payload["steps"][0]["command"][1] == "python/scripts/train.py"
    assert payload["steps"][1]["command"][1] == "python/scripts/eval.py"
    assert payload["steps"][2]["command"][1] == "python/scripts/compare_runs.py"


def test_thesis_run_wrapper_defaults_to_thesis_model_preset_when_stack_config_is_omitted(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "configs").mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "python" / "scripts" / "thesis_run.py"),
            "--repo-root",
            str(repo_root),
            "--run-label",
            "default_preset_run",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary_path = repo_root / "runs" / "_wrapper_plans" / "default_preset_run.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["preset"] == "thesis-model-auto-gpu"
    assert payload["stack_config"].endswith("configs/main_impala_league_server.yaml")
    assert payload["eval_preset"] == "thesis-model-eval-auto-gpu"
    assert payload["eval_stack_config"].endswith("configs/main_eval.yaml")
    assert payload["server_defaults_applied"] is True
    command = payload["steps"][0]["command"]
    assert "--autoscale" in command
    assert command[command.index("--hardware-profile") + 1] == "local"
    assert command[command.index("--runtime-mode") + 1] == "train_async_fast"
    assert command[command.index("--unroll-length") + 1] == "64"
    assert command[command.index("--max-updates") + 1] == "400"
    assert len(payload["steps"]) == 2


def test_thesis_run_wrapper_reuses_custom_stack_config_for_eval_when_no_eval_override_is_supplied(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "configs").mkdir(parents=True, exist_ok=True)
    stack_config = repo_root / "configs" / "stack.yaml"
    stack_config.write_text("components: []\nconfig: {}\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "python" / "scripts" / "thesis_run.py"),
            "--repo-root",
            str(repo_root),
            "--stack-config",
            str(stack_config),
            "--run-label",
            "custom_eval_match",
            "--dry-run",
            "--skip-compare",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary_path = repo_root / "runs" / "_wrapper_plans" / "custom_eval_match.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["stack_config"] == stack_config.resolve().as_posix()
    assert payload["eval_stack_config"] == stack_config.resolve().as_posix()
    assert payload["eval_preset"] == ""


def test_thesis_run_wrapper_resolves_relative_config_paths_against_repo_root(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "configs").mkdir(parents=True, exist_ok=True)
    stack_config = repo_root / "configs" / "train_stack.yaml"
    eval_stack_config = repo_root / "configs" / "eval_stack.yaml"
    stack_config.write_text("components: []\nconfig: {}\n", encoding="utf-8")
    eval_stack_config.write_text("components: []\nconfig: {}\n", encoding="utf-8")

    invocation_cwd = tmp_path / "outside_repo"
    invocation_cwd.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "python" / "scripts" / "thesis_run.py"),
            "--repo-root",
            str(repo_root),
            "--stack-config",
            "configs/train_stack.yaml",
            "--eval-stack-config",
            "configs/eval_stack.yaml",
            "--run-label",
            "relative_paths",
            "--dry-run",
            "--skip-compare",
        ],
        cwd=invocation_cwd,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary_path = repo_root / "runs" / "_wrapper_plans" / "relative_paths.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["stack_config"] == stack_config.resolve().as_posix()
    assert payload["eval_stack_config"] == eval_stack_config.resolve().as_posix()


def test_thesis_run_wrapper_defaults_to_multideck_eval_surface_for_multideck_preset(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "configs" / "ablations").mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "python" / "scripts" / "thesis_run.py"),
            "--repo-root",
            str(repo_root),
            "--preset",
            "thesis-model-multideck",
            "--run-label",
            "multideck_default_eval",
            "--dry-run",
            "--skip-compare",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary_path = repo_root / "runs" / "_wrapper_plans" / "multideck_default_eval.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["preset"] == "thesis-model-multideck"
    assert payload["eval_preset"] == "thesis-model-multideck-eval-auto-gpu"
    assert payload["eval_stack_config"].endswith("configs/ablations/multideck_eval.yaml")


def test_thesis_run_wrapper_defaults_to_gpu_eval_surface_for_thesis_model_auto_gpu_preset(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "configs" / "ablations").mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "python" / "scripts" / "thesis_run.py"),
            "--repo-root",
            str(repo_root),
            "--preset",
            "thesis-model-auto-gpu",
            "--run-label",
            "thesis_model_auto_gpu_default_eval",
            "--dry-run",
            "--skip-compare",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary_path = repo_root / "runs" / "_wrapper_plans" / "thesis_model_auto_gpu_default_eval.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["preset"] == "thesis-model-auto-gpu"
    assert payload["eval_preset"] == "thesis-model-eval-auto-gpu"
    assert payload["eval_stack_config"].endswith("configs/main_eval.yaml")


def test_thesis_run_wrapper_defaults_to_gpu_eval_surface_for_server_train_preset(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "configs" / "ablations").mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "python" / "scripts" / "thesis_run.py"),
            "--repo-root",
            str(repo_root),
            "--preset",
            "thesis-model-server-train",
            "--run-label",
            "thesis_model_server_train_default_eval",
            "--dry-run",
            "--skip-compare",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary_path = repo_root / "runs" / "_wrapper_plans" / "thesis_model_server_train_default_eval.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["preset"] == "thesis-model-server-train"
    assert payload["eval_preset"] == "thesis-model-eval-auto-gpu"
    assert payload["stack_config"].endswith("configs/main_impala_league_server.yaml")
    assert payload["eval_stack_config"].endswith("configs/main_eval.yaml")


def test_thesis_run_wrapper_defaults_b1_anchor_benchmark_to_matching_eval_surface(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "configs" / "baselines").mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "python" / "scripts" / "thesis_run.py"),
            "--repo-root",
            str(repo_root),
            "--preset",
            "b1-anchor-benchmark",
            "--run-label",
            "b1_anchor_benchmark_default_eval",
            "--dry-run",
            "--skip-compare",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary_path = repo_root / "runs" / "_wrapper_plans" / "b1_anchor_benchmark_default_eval.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["preset"] == "b1-anchor-benchmark"
    assert payload["eval_preset"] == "b1-anchor-benchmark-eval-auto-gpu"
    assert payload["stack_config"].endswith("configs/baselines/noleague_benchmark.yaml")
    assert payload["eval_stack_config"].endswith("configs/baselines/noleague_benchmark_eval.yaml")


def test_thesis_run_wrapper_defaults_b1_anchor_phase_presets_to_matching_eval_surface(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "configs" / "baselines").mkdir(parents=True, exist_ok=True)

    for preset, expected_train_suffix in (
        (
            "b1-anchor-benchmark-warmup",
            "configs/baselines/noleague_benchmark_warmup.yaml",
        ),
        (
            "b1-anchor-benchmark-lowlr-continuation",
            "configs/baselines/noleague_benchmark_lowlr_continuation.yaml",
        ),
    ):
        run_label = f"{preset.replace('-', '_')}_default_eval"
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "python" / "scripts" / "thesis_run.py"),
                "--repo-root",
                str(repo_root),
                "--preset",
                preset,
                "--run-label",
                run_label,
                "--dry-run",
                "--skip-compare",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        summary_path = repo_root / "runs" / "_wrapper_plans" / f"{run_label}.json"
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        assert payload["preset"] == preset
        assert payload["eval_preset"] == "b1-anchor-benchmark-eval-auto-gpu"
        assert payload["stack_config"].endswith(expected_train_suffix)
        assert payload["eval_stack_config"].endswith("configs/baselines/noleague_benchmark_eval.yaml")


def test_thesis_run_wrapper_defaults_fullsize_b1_and_b1anchored_league_presets_to_main_eval_surface(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "configs" / "baselines").mkdir(parents=True, exist_ok=True)

    for preset, expected_train_suffix in (
        (
            "b1-anchor-fullsize-warmup",
            "configs/baselines/noleague_fullsize_warmup.yaml",
        ),
        (
            "b1-anchor-fullsize-lowlr-continuation",
            "configs/baselines/noleague_fullsize_lowlr_continuation.yaml",
        ),
        (
            "thesis-model-server-train-b1anchored",
            "configs/main_impala_league_server.yaml",
        ),
        (
            "thesis-model-server-train-b1anchored-refb1strong-lowlr",
            "configs/main_impala_league_server.yaml",
        ),
    ):
        run_label = f"{preset.replace('-', '_')}_default_eval"
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "python" / "scripts" / "thesis_run.py"),
                "--repo-root",
                str(repo_root),
                "--preset",
                preset,
                "--run-label",
                run_label,
                "--dry-run",
                "--skip-compare",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        summary_path = repo_root / "runs" / "_wrapper_plans" / f"{run_label}.json"
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        assert payload["preset"] == preset
        assert payload["eval_preset"] == "thesis-model-eval-auto-gpu"
        assert payload["stack_config"].endswith(expected_train_suffix)
        assert payload["eval_stack_config"].endswith("configs/main_eval.yaml")


def test_thesis_run_wrapper_defaults_b1anchored_benchmark_league_to_benchmark_eval_surface(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "configs" / "baselines").mkdir(parents=True, exist_ok=True)

    for preset, expected_train_suffix in (
        (
            "thesis-model-server-train-b1anchored-benchmark",
            "configs/main_impala_league_server.yaml",
        ),
        (
            "thesis-model-server-train-b1anchored-benchmark-localpromo",
            "configs/main_impala_league_server.yaml",
        ),
        (
            "thesis-model-server-train-b1anchored-benchmark-selfplay-localpromo",
            "configs/main_impala_league_server.yaml",
        ),
        (
            "thesis-model-server-train-b1anchored-benchmark-selfplay-bckl-localpromo",
            "configs/main_impala_league_server.yaml",
        ),
        (
            "thesis-model-server-train-b1anchored-benchmark-selfplay-refb1strong-lowlr-localpromo",
            "configs/main_impala_league_server.yaml",
        ),
        (
            "thesis-model-server-train-b1anchored-benchmark-selfplay-refb1strong-lowlr-evalguard-localpromo",
            "configs/main_impala_league_server.yaml",
        ),
        (
            "thesis-model-server-train-b1anchored-benchmark-modelbridge-localpromo",
            "configs/main_impala_league_server.yaml",
        ),
    ):
        run_label = f"{preset.replace('-', '_')}_default_eval"
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "python" / "scripts" / "thesis_run.py"),
                "--repo-root",
                str(repo_root),
                "--preset",
                preset,
                "--run-label",
                run_label,
                "--dry-run",
                "--skip-compare",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        summary_path = repo_root / "runs" / "_wrapper_plans" / f"{run_label}.json"
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        assert payload["preset"] == preset
        assert payload["eval_preset"] == "b1-anchor-benchmark-eval-auto-gpu"
        assert payload["stack_config"].endswith(expected_train_suffix)
        assert payload["eval_stack_config"].endswith("configs/baselines/noleague_benchmark_eval.yaml")


def test_thesis_run_wrapper_defaults_ablations_to_matching_eval_surfaces(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "configs" / "ablations").mkdir(parents=True, exist_ok=True)

    for preset, expected_eval_preset, expected_suffix in (
        (
            "ablate-teacher-fade",
            "ablate-teacher-fade-eval-auto-gpu",
            "configs/ablations/teacher_fade_eval.yaml",
        ),
        (
            "ablate-no-tactical-bias",
            "ablate-no-tactical-bias-eval-auto-gpu",
            "configs/ablations/no_tactical_bias_eval.yaml",
        ),
        (
            "ablate-teacher-fade-no-tactical-bias",
            "ablate-teacher-fade-no-tactical-bias-eval-auto-gpu",
            "configs/ablations/teacher_fade_no_tactical_bias_eval.yaml",
        ),
        (
            "ablate-no-b1-cutoff",
            "ablate-no-b1-cutoff-eval-auto-gpu",
            "configs/ablations/no_b1_cutoff_eval.yaml",
        ),
        (
            "ablate-reward-shaping",
            "ablate-reward-shaping-eval-auto-gpu",
            "configs/ablations/reward_shaping_eval.yaml",
        ),
    ):
        run_label = f"{preset}_default_eval"
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "python" / "scripts" / "thesis_run.py"),
                "--repo-root",
                str(repo_root),
                "--preset",
                preset,
                "--run-label",
                run_label,
                "--dry-run",
                "--skip-compare",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        payload = json.loads((repo_root / "runs" / "_wrapper_plans" / f"{run_label}.json").read_text(encoding="utf-8"))
        assert payload["eval_preset"] == expected_eval_preset
        assert payload["eval_stack_config"].endswith(expected_suffix)


def test_thesis_run_wrapper_lists_named_presets_without_run_label(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "python" / "scripts" / "thesis_run.py"),
            "--repo-root",
            str(repo_root),
            "--list-presets",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "thesis-model-auto-gpu:" in result.stdout
    assert "thesis-model-server-train:" in result.stdout
    assert "b1-anchor-benchmark:" in result.stdout
    assert "thesis-model-multideck:" in result.stdout
    assert "ablate-teacher-fade:" in result.stdout
    assert "ablate-no-tactical-bias:" in result.stdout
    assert "ablate-teacher-fade-no-tactical-bias:" in result.stdout
    assert "ablate-no-b1-cutoff:" in result.stdout
    assert "ablate-reward-shaping:" in result.stdout


def test_thesis_run_wrapper_passes_b1_baseline_run_dir_to_train_and_eval(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "configs").mkdir(parents=True, exist_ok=True)
    stack_config = repo_root / "configs" / "stack.yaml"
    stack_config.write_text("components: []\nconfig: {}\n", encoding="utf-8")
    baseline_run_dir = repo_root / "runs" / "b1_anchor_seed1"

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "python" / "scripts" / "thesis_run.py"),
            "--repo-root",
            str(repo_root),
            "--stack-config",
            str(stack_config),
            "--run-label",
            "baseline_passthrough",
            "--b1-baseline-run-dir",
            str(baseline_run_dir),
            "--dry-run",
            "--skip-compare",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary_path = repo_root / "runs" / "_wrapper_plans" / "baseline_passthrough.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["b1_baseline_run_dir"] == baseline_run_dir.resolve().as_posix()
    assert "--b1-baseline-run-dir" in payload["steps"][0]["command"]
    assert str(baseline_run_dir) in payload["steps"][0]["command"]
    assert "--b1-baseline-run-dir" in payload["steps"][1]["command"]
    assert str(baseline_run_dir) in payload["steps"][1]["command"]


def test_thesis_run_wrapper_passes_seed_snapshot_run_dir_to_train_only(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "configs").mkdir(parents=True, exist_ok=True)
    stack_config = repo_root / "configs" / "stack.yaml"
    stack_config.write_text("components: []\nconfig: {}\n", encoding="utf-8")
    seed_run_dir = repo_root / "runs" / "league_seed_pool"

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "python" / "scripts" / "thesis_run.py"),
            "--repo-root",
            str(repo_root),
            "--stack-config",
            str(stack_config),
            "--run-label",
            "seedpool_passthrough",
            "--seed-snapshot-run-dir",
            str(seed_run_dir),
            "--dry-run",
            "--skip-compare",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary_path = repo_root / "runs" / "_wrapper_plans" / "seedpool_passthrough.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["seed_snapshot_run_dir"] == seed_run_dir.resolve().as_posix()
    assert "--seed-snapshot-run-dir" in payload["steps"][0]["command"]
    assert str(seed_run_dir) in payload["steps"][0]["command"]
    assert "--seed-snapshot-run-dir" not in payload["steps"][1]["command"]


def test_thesis_run_wrapper_skips_compare_without_explicit_comparison_targets(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "configs").mkdir(parents=True, exist_ok=True)
    stack_config = repo_root / "configs" / "stack.yaml"
    stack_config.write_text("components: []\nconfig: {}\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "python" / "scripts" / "thesis_run.py"),
            "--repo-root",
            str(repo_root),
            "--stack-config",
            str(stack_config),
            "--run-label",
            "no_compare_targets",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary_path = repo_root / "runs" / "_wrapper_plans" / "no_compare_targets.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert len(payload["steps"]) == 2
    assert all(step["command"][1] != "python/scripts/compare_runs.py" for step in payload["steps"])


def test_thesis_run_wrapper_forwards_max_wall_clock_minutes_to_train(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "configs").mkdir(parents=True, exist_ok=True)
    stack_config = repo_root / "configs" / "stack.yaml"
    stack_config.write_text("components: []\nconfig: {}\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "python" / "scripts" / "thesis_run.py"),
            "--repo-root",
            str(repo_root),
            "--stack-config",
            str(stack_config),
            "--run-label",
            "wall_clock_passthrough",
            "--max-wall-clock-minutes",
            "7.5",
            "--dry-run",
            "--skip-compare",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary_path = repo_root / "runs" / "_wrapper_plans" / "wall_clock_passthrough.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["max_wall_clock_minutes"] == 7.5
    assert "--max-wall-clock-minutes" in payload["steps"][0]["command"]
    assert "7.5" in payload["steps"][0]["command"]


def test_thesis_run_wrapper_can_build_torchrun_autoscale_train_command(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "configs").mkdir(parents=True, exist_ok=True)
    stack_config = repo_root / "configs" / "stack.yaml"
    stack_config.write_text("components: []\nconfig: {}\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "python" / "scripts" / "thesis_run.py"),
            "--repo-root",
            str(repo_root),
            "--stack-config",
            str(stack_config),
            "--run-label",
            "server_ddp_plan",
            "--torchrun-nproc",
            "4",
            "--autoscale",
            "--hardware-profile",
            "gpu4",
            "--ddp-backend",
            "nccl",
            "--ddp-timeout-seconds",
            "2400",
            "--dry-run",
            "--skip-compare",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads((repo_root / "runs" / "_wrapper_plans" / "server_ddp_plan.json").read_text(encoding="utf-8"))
    command = payload["steps"][0]["command"]
    assert command[1:7] == [
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        "4",
        "python/scripts/train.py",
    ]
    assert "--autoscale" in command
    assert "--num-envs" not in command
    assert "--ddp" in command
    assert command[command.index("--hardware-profile") + 1] == "gpu4"
    assert command[command.index("--ddp-backend") + 1] == "nccl"
    assert command[command.index("--ddp-timeout-seconds") + 1] == "2400"
    assert payload["torchrun_nproc"] == 4
    assert payload["ddp"] is True
    assert payload["ddp_timeout_seconds"] == 2400


def test_thesis_run_wrapper_autoscale_dry_run_skips_eval_plan(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "configs").mkdir(parents=True, exist_ok=True)
    stack_config = repo_root / "configs" / "stack.yaml"
    stack_config.write_text("components: []\nconfig: {}\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "python" / "scripts" / "thesis_run.py"),
            "--repo-root",
            str(repo_root),
            "--stack-config",
            str(stack_config),
            "--run-label",
            "autoscale_only_plan",
            "--autoscale-dry-run",
            "--hardware-profile",
            "gpu4",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads((repo_root / "runs" / "_wrapper_plans" / "autoscale_only_plan.json").read_text(encoding="utf-8"))
    assert len(payload["steps"]) == 1
    assert "--autoscale-dry-run" in payload["steps"][0]["command"]
    assert payload["autoscale_dry_run"] is True
