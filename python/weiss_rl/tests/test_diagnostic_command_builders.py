from __future__ import annotations

from pathlib import Path

from weiss_rl.diagnostics.heuristic_sanity_scan_entrypoint import build_heuristic_sanity_command
from weiss_rl.diagnostics.profile_train_job_entrypoint import _repo_root, build_profile_train_command

REPO_ROOT = Path(__file__).resolve().parents[3]


def _option(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_profile_train_job_uses_shared_package_train_command_builder(tmp_path: Path) -> None:
    command = build_profile_train_command(
        repo_root=tmp_path,
        stack_config="configs/presets/typed.yaml",
        run_label="profile_job",
        python_executable="python.exe",
        device="cuda:0",
        profile="fast",
        num_envs=288,
        unroll_length=64,
        max_updates=50,
        runtime_mode="train_async_fast",
        config_overrides=["training.profile_timers=true"],
        train_args=["--checkpoint-interval-updates", "5"],
    )

    assert command[:3] == ["python.exe", "-m", "weiss_rl.training.train_entrypoint"]
    assert _option(command, "--stack-config") == "configs/presets/typed.yaml"
    assert _option(command, "--run-label") == "profile_job"
    assert _option(command, "--device") == "cuda:0"
    assert _option(command, "--profile") == "fast"
    assert _option(command, "--num-envs") == "288"
    assert _option(command, "--runtime-mode") == "train_async_fast"
    assert "training.profile_timers=true" in command
    assert command[-2:] == ["--checkpoint-interval-updates", "5"]


def test_profile_train_job_default_repo_root_is_project_root() -> None:
    assert _repo_root() == REPO_ROOT


def test_heuristic_sanity_scan_preserves_custom_confirm_command_shape() -> None:
    command = build_heuristic_sanity_command(
        focal="B3 HeuristicPublicAggro",
        opponent="B2 HeuristicPublic",
    )

    stack_config = _option(command, "--stack-config")
    run_dir = _option(command, "--run-dir")
    registry = _option(command, "--snapshot-registry-json")
    b1_baseline = _option(command, "--b1-baseline-run-dir")

    assert command[1:3] == ["-m", "weiss_rl.eval.targeted_confirm.entrypoint"]
    assert stack_config == "configs/presets/structured_acceptance_standard_thesis_eval.yaml"
    assert run_dir == "runs/main_champion_hardneg_interp_u10_repair_a015_20260517"
    assert registry == "runs/main_champion_hardneg_interp_u10_repair_a015_20260517/training/snapshots/registry.json"
    assert b1_baseline == run_dir
    assert (REPO_ROOT / stack_config).is_file()
    assert (REPO_ROOT / run_dir).is_dir()
    assert (REPO_ROOT / registry).is_file()
    assert _option(command, "--focal-policy-id") == "B3 HeuristicPublicAggro"
    assert _option(command, "--opponent") == "B2 HeuristicPublic"
    assert _option(command, "--paired-seeds") == "16"
    assert _option(command, "--output-subdir") == "heuristic_sanity16"
    assert "--bootstrap-samples" not in command
