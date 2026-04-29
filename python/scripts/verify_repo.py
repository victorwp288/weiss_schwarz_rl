from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run_step(*, label: str, command: list[str], cwd: Path) -> None:
    print()
    print(f"==> {label}")
    subprocess.run(command, cwd=cwd, check=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the local repository verification ladder.")
    parser.parse_args([] if argv is None else argv)

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
        (
            "Thesis-model wrapper dry-run",
            [
                python_exe,
                "python/scripts/thesis_run.py",
                "--preset",
                "thesis-model-auto-gpu",
                "--run-label",
                "thesis_model_surface_ci",
                "--dry-run",
                "--skip-compare",
            ],
        ),
        (
            "Ablation wrapper dry-run",
            [
                python_exe,
                "python/scripts/thesis_run.py",
                "--preset",
                "ablate-no-tactical-bias",
                "--run-label",
                "ablate_no_tactical_bias_surface_ci",
                "--dry-run",
                "--skip-compare",
            ],
        ),
        (
            "Thesis-model multideck wrapper dry-run",
            [
                python_exe,
                "python/scripts/thesis_run.py",
                "--preset",
                "thesis-model-multideck",
                "--run-label",
                "thesis_model_multideck_surface_ci",
                "--dry-run",
                "--skip-compare",
            ],
        ),
    ]
    for label, command in commands:
        _run_step(label=label, command=command, cwd=repo_root)
    print()
    print("Local verification completed.")


if __name__ == "__main__":
    main(sys.argv[1:])
