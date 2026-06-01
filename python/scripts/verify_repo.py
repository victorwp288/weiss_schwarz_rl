from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def _run_step(*, label: str, command: list[str], cwd: Path) -> None:
    print()
    print(f"==> {label}")
    subprocess.run(command, cwd=cwd, check=True)


def _write_b1_snapshot_source(run_dir: Path) -> None:
    checkpoint_path = run_dir / "training" / "checkpoints" / "checkpoint_1.pt"
    registry_path = run_dir / "training" / "snapshots" / "registry.json"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"checkpoint")
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "recent_size": 24,
                "champion_size": 4,
                "snapshots": [
                    {
                        "policy_id": "b1_noleague_baseline",
                        "update": 1,
                        "weights_sha256": "a" * 64,
                        "path": "training/snapshots/b1_noleague_baseline/weights.pt",
                    }
                ],
                "champion_snapshots": [],
                "pinned_snapshots": ["b1_noleague_baseline"],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _package_cli_dry_run_commands(*, python_exe: str, workspace: Path) -> list[tuple[str, list[str]]]:
    b1_source_run = workspace / "runs" / "verify_b1_source"
    main_run = workspace / "runs" / "verify_main_smoke"
    _write_b1_snapshot_source(b1_source_run)
    return [
        (
            "Package CLI train-b1 dry-run",
            [
                python_exe,
                "-m",
                "weiss_rl.cli",
                "train-b1",
                "--repo-root",
                str(workspace),
                "--run-label",
                "verify_b1_smoke",
                "--profile",
                "smoke",
                "--dry-run",
            ],
        ),
        (
            "Package CLI train-main dry-run",
            [
                python_exe,
                "-m",
                "weiss_rl.cli",
                "train-main",
                "--repo-root",
                str(workspace),
                "--run-label",
                "verify_main_smoke",
                "--b1-run",
                str(b1_source_run),
                "--profile",
                "smoke",
                "--dry-run",
            ],
        ),
        (
            "Package CLI smoke-eval dry-run",
            [
                python_exe,
                "-m",
                "weiss_rl.cli",
                "smoke-eval",
                "--repo-root",
                str(workspace),
                "--run-dir",
                str(main_run),
                "--b1-run",
                str(b1_source_run),
                "--dry-run",
            ],
        ),
        (
            "Package CLI eval-final dry-run",
            [
                python_exe,
                "-m",
                "weiss_rl.cli",
                "eval-final",
                "--repo-root",
                str(workspace),
                "--run-dir",
                str(main_run),
                "--b1-run",
                str(b1_source_run),
                "--dry-run",
            ],
        ),
        (
            "Package CLI figures dry-run",
            [
                python_exe,
                "-m",
                "weiss_rl.cli",
                "figures",
                "--repo-root",
                str(workspace),
                "--run-dir",
                str(main_run),
                "--format",
                "png",
                "--dry-run",
            ],
        ),
    ]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    python_exe = sys.executable
    commands = [
        ("Core placeholder gate", [python_exe, "python/scripts/check_core_placeholders.py"]),
        ("Ruff check", [python_exe, "-m", "ruff", "check", "python", "tests", "examples", "python/scripts"]),
        (
            "Ruff format check",
            [python_exe, "-m", "ruff", "format", "--check", "python", "tests", "examples", "python/scripts"],
        ),
        (
            "Mypy",
            [
                python_exe,
                "-m",
                "mypy",
                "python/weiss_rl/cli.py",
                "python/weiss_rl/workflows",
                "python/scripts/thesis_run.py",
                "python/scripts/eval.py",
                "python/scripts/play_vs_model.py",
            ],
        ),
        (
            "Vulture",
            [python_exe, "-m", "vulture", "python/weiss_rl", "python/scripts", "examples", "--min-confidence", "80"],
        ),
        ("Pytest", [python_exe, "-m", "pytest", "-q", "python/weiss_rl/tests"]),
    ]
    with tempfile.TemporaryDirectory(prefix="weiss_rl_verify_") as workspace_name:
        workspace = Path(workspace_name)
        commands.extend(_package_cli_dry_run_commands(python_exe=python_exe, workspace=workspace))
        for label, command in commands:
            _run_step(label=label, command=command, cwd=repo_root)
    print()
    print("Local verification completed.")


if __name__ == "__main__":
    main()
