from __future__ import annotations

from pathlib import Path

from .thesis_run_wrapper_test_support import (
    read_wrapper_plan,
    run_thesis_wrapper_subprocess,
    write_stack_config,
)


def test_thesis_run_wrapper_dry_run_writes_plan(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    stack_config = write_stack_config(repo_root)

    result = run_thesis_wrapper_subprocess(
        repo_root,
        "--stack-config",
        str(stack_config),
        "--run-label",
        "demo_run",
        "--dry-run",
        "--compare-run-dir",
        str(repo_root / "runs" / "baseline_a"),
    )

    assert result.returncode == 0, result.stderr
    payload = read_wrapper_plan(repo_root, "demo_run")
    assert payload["status"] == "planned"
    assert len(payload["steps"]) == 3
    assert payload["steps"][0]["command"][1:3] == ["-m", "weiss_rl.training.train_entrypoint"]
    assert payload["steps"][1]["command"][1:3] == ["-m", "weiss_rl.workflows.eval_entrypoint"]
    assert payload["steps"][2]["command"][1:3] == ["-m", "weiss_rl.workflows.compare_runs.compare_runs_entrypoint"]


def test_thesis_run_package_module_dry_run_writes_plan(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    stack_config = write_stack_config(repo_root)

    result = run_thesis_wrapper_subprocess(
        repo_root,
        "--stack-config",
        str(stack_config),
        "--run-label",
        "module_demo_run",
        "--dry-run",
        "--skip-compare",
    )

    assert result.returncode == 0, result.stderr
    payload = read_wrapper_plan(repo_root, "module_demo_run")
    assert payload["kind"] == "thesis_run_wrapper_v1"
    assert payload["status"] == "planned"
    assert len(payload["steps"]) == 2
    assert payload["steps"][0]["command"][1:3] == ["-m", "weiss_rl.training.train_entrypoint"]
    assert payload["steps"][1]["command"][1:3] == ["-m", "weiss_rl.workflows.eval_entrypoint"]


def test_thesis_run_wrapper_defaults_to_standard_preset_when_stack_config_is_omitted(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "configs" / "presets").mkdir(parents=True, exist_ok=True)

    result = run_thesis_wrapper_subprocess(
        repo_root,
        "--run-label",
        "default_preset_run",
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    payload = read_wrapper_plan(repo_root, "default_preset_run")
    assert payload["preset"] == "standard"
    assert payload["stack_config"].endswith("configs/presets/structured_acceptance_standard.yaml")
    assert payload["eval_preset"] == "standard-thesis-eval"
    assert payload["eval_stack_config"].endswith("configs/presets/structured_acceptance_standard_thesis_eval.yaml")


def test_thesis_run_wrapper_reuses_custom_stack_config_for_eval_when_no_eval_override_is_supplied(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    stack_config = write_stack_config(repo_root)

    result = run_thesis_wrapper_subprocess(
        repo_root,
        "--stack-config",
        str(stack_config),
        "--run-label",
        "custom_eval_match",
        "--dry-run",
        "--skip-compare",
    )

    assert result.returncode == 0, result.stderr
    payload = read_wrapper_plan(repo_root, "custom_eval_match")
    assert payload["stack_config"] == stack_config.resolve().as_posix()
    assert payload["eval_stack_config"] == stack_config.resolve().as_posix()
    assert payload["eval_preset"] == ""


def test_thesis_run_wrapper_resolves_relative_config_paths_against_repo_root(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    stack_config = write_stack_config(repo_root, name="train_stack.yaml")
    eval_stack_config = write_stack_config(repo_root, name="eval_stack.yaml")

    invocation_cwd = tmp_path / "outside_repo"
    invocation_cwd.mkdir(parents=True, exist_ok=True)
    result = run_thesis_wrapper_subprocess(
        repo_root,
        "--stack-config",
        "configs/train_stack.yaml",
        "--eval-stack-config",
        "configs/eval_stack.yaml",
        "--run-label",
        "relative_paths",
        "--dry-run",
        "--skip-compare",
        cwd=invocation_cwd,
    )

    assert result.returncode == 0, result.stderr
    payload = read_wrapper_plan(repo_root, "relative_paths")
    assert payload["stack_config"] == stack_config.resolve().as_posix()
    assert payload["eval_stack_config"] == eval_stack_config.resolve().as_posix()


def test_thesis_run_wrapper_defaults_to_multideck_eval_surface_for_multideck_preset(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "configs" / "presets" / "ablations").mkdir(parents=True, exist_ok=True)

    result = run_thesis_wrapper_subprocess(
        repo_root,
        "--preset",
        "standard-multideck",
        "--run-label",
        "multideck_default_eval",
        "--dry-run",
        "--skip-compare",
    )

    assert result.returncode == 0, result.stderr
    payload = read_wrapper_plan(repo_root, "multideck_default_eval")
    assert payload["preset"] == "standard-multideck"
    assert payload["eval_preset"] == "standard-multideck"
    assert payload["eval_stack_config"].endswith("configs/presets/structured_acceptance_standard_multideck.yaml")


def test_thesis_run_wrapper_lists_named_presets_without_run_label(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)

    result = run_thesis_wrapper_subprocess(
        repo_root,
        "--list-presets",
    )

    assert result.returncode == 0, result.stderr
    assert "standard:" in result.stdout
    assert "standard-auto-gpu:" in result.stdout
    assert "ablate-ppo-lite:" in result.stdout
    assert "ablate-terminal-only:" in result.stdout


def test_thesis_run_wrapper_passes_b1_baseline_run_dir_to_train_and_eval(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    stack_config = write_stack_config(repo_root)
    baseline_run_dir = repo_root / "runs" / "b1_anchor_seed1"

    result = run_thesis_wrapper_subprocess(
        repo_root,
        "--stack-config",
        str(stack_config),
        "--run-label",
        "baseline_passthrough",
        "--b1-baseline-run-dir",
        str(baseline_run_dir),
        "--dry-run",
        "--skip-compare",
    )

    assert result.returncode == 0, result.stderr
    payload = read_wrapper_plan(repo_root, "baseline_passthrough")
    assert payload["b1_baseline_run_dir"] == baseline_run_dir.resolve().as_posix()
    assert "--b1-baseline-run-dir" in payload["steps"][0]["command"]
    assert str(baseline_run_dir) in payload["steps"][0]["command"]
    assert "--b1-baseline-run-dir" in payload["steps"][1]["command"]
    assert str(baseline_run_dir) in payload["steps"][1]["command"]
