from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from weiss_rl.workflows.step_execution import CommandRunner, run_command_steps, run_labeled_command
from weiss_rl.workflows.verification.verify_repo_plan import (
    VerificationRequest,
    VerificationStep,
    build_release_verification_steps,
    build_release_verification_steps_for_request,
    verification_request,
)


def _run_step(
    *,
    label: str,
    command: list[str],
    cwd: Path,
    command_runner: CommandRunner | None = None,
) -> None:
    run_labeled_command(
        label=label,
        command=command,
        cwd=cwd,
        command_runner=subprocess.run if command_runner is None else command_runner,
    )


def run_verification_steps(
    *,
    steps: tuple[VerificationStep, ...],
    repo_root: Path,
    command_runner: CommandRunner | None = None,
) -> None:
    run_command_steps(
        steps=steps,
        repo_root=repo_root,
        command_runner=subprocess.run if command_runner is None else command_runner,
    )


def run_verification_request(
    *,
    request: VerificationRequest,
    command_runner: CommandRunner | None = None,
) -> None:
    run_verification_steps(
        steps=build_release_verification_steps_for_request(request),
        repo_root=request.repo_root,
        command_runner=command_runner,
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    run_verification_request(request=verification_request(repo_root=repo_root, python_exe=sys.executable))
    print()
    print("Local verification completed.")


__all__ = [
    "VerificationRequest",
    "VerificationStep",
    "_run_step",
    "build_release_verification_steps",
    "build_release_verification_steps_for_request",
    "main",
    "run_verification_request",
    "run_verification_steps",
    "verification_request",
]


if __name__ == "__main__":
    main()
