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
    (repo_root / "configs" / "presets").mkdir(parents=True, exist_ok=True)

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
    assert payload["stack_config"].endswith("configs/presets/structured_acceptance_thesis_model_auto_gpu.yaml")
    assert payload["eval_preset"] == "thesis-model-eval-auto-gpu"
    assert payload["eval_stack_config"].endswith("configs/presets/structured_acceptance_thesis_model_eval_auto_gpu.yaml")
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
    (repo_root / "configs" / "presets" / "ablations").mkdir(parents=True, exist_ok=True)

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
    assert payload["eval_stack_config"].endswith(
        "configs/presets/structured_acceptance_thesis_model_multideck_eval_auto_gpu.yaml"
    )


def test_thesis_run_wrapper_defaults_to_gpu_eval_surface_for_thesis_model_auto_gpu_preset(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "configs" / "presets" / "ablations").mkdir(parents=True, exist_ok=True)

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
    assert payload["eval_stack_config"].endswith("configs/presets/structured_acceptance_thesis_model_eval_auto_gpu.yaml")


def test_thesis_run_wrapper_defaults_ablations_to_matching_eval_surfaces(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "configs" / "presets" / "ablations").mkdir(parents=True, exist_ok=True)

    for preset, expected_eval_preset, expected_suffix in (
        (
            "ablate-teacher-fade",
            "ablate-teacher-fade-eval-auto-gpu",
            "configs/presets/ablations/structured_acceptance_thesis_model_teacher_fade_eval_auto_gpu.yaml",
        ),
        (
            "ablate-no-tactical-bias",
            "ablate-no-tactical-bias-eval-auto-gpu",
            "configs/presets/ablations/structured_acceptance_thesis_model_no_tactical_bias_eval_auto_gpu.yaml",
        ),
        (
            "ablate-no-b1-cutoff",
            "ablate-no-b1-cutoff-eval-auto-gpu",
            "configs/presets/ablations/structured_acceptance_thesis_model_no_b1_cutoff_eval_auto_gpu.yaml",
        ),
        (
            "ablate-reward-shaping",
            "ablate-reward-shaping-eval-auto-gpu",
            "configs/presets/ablations/structured_acceptance_thesis_model_reward_shaping_eval_auto_gpu.yaml",
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
    assert "thesis-model-multideck:" in result.stdout
    assert "ablate-teacher-fade:" in result.stdout
    assert "ablate-no-tactical-bias:" in result.stdout
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
