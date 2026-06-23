from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class VerificationRequest:
    repo_root: Path
    python_exe: str


@dataclass(frozen=True, slots=True)
class VerificationStep:
    label: str
    command: tuple[str, ...]


def _module_command(python_exe: str, module: str, *args: str) -> tuple[str, ...]:
    return (python_exe, "-m", module, *args)


def _wrapper_dry_run_command(*, python_exe: str, preset: str, run_label: str) -> tuple[str, ...]:
    return _module_command(
        python_exe,
        "weiss_rl.workflows.thesis_wrapper",
        "--preset",
        preset,
        "--run-label",
        run_label,
        "--dry-run",
        "--skip-compare",
    )


def build_release_verification_steps(*, python_exe: str) -> tuple[VerificationStep, ...]:
    return (
        VerificationStep(
            "Repo hygiene gate",
            _module_command(python_exe, "weiss_rl.diagnostics.repo_hygiene_check_entrypoint"),
        ),
        VerificationStep(
            "Core placeholder gate",
            _module_command(python_exe, "weiss_rl.diagnostics.core_placeholder_check_entrypoint"),
        ),
        VerificationStep(
            "Ruff check",
            _module_command(python_exe, "ruff", "check", "python", "tests", "examples"),
        ),
        VerificationStep(
            "Ruff format check",
            _module_command(
                python_exe,
                "ruff",
                "format",
                "--check",
                "python",
                "tests",
                "examples",
            ),
        ),
        VerificationStep(
            "Mypy",
            _module_command(
                python_exe,
                "mypy",
                "python/weiss_rl/workflows/thesis_wrapper.py",
                "python/weiss_rl/workflows/eval_entrypoint.py",
                "python/weiss_rl/human_play/play_vs_model_entrypoint.py",
            ),
        ),
        VerificationStep(
            "Vulture",
            _module_command(
                python_exe,
                "vulture",
                "python/weiss_rl",
                "examples",
                "--min-confidence",
                "80",
            ),
        ),
        VerificationStep("Pytest", _module_command(python_exe, "pytest", "-q", "tests/weiss_rl")),
        VerificationStep(
            "Standard wrapper dry-run",
            _wrapper_dry_run_command(
                python_exe=python_exe,
                preset="standard",
                run_label="standard_surface_ci",
            ),
        ),
        VerificationStep(
            "Standard auto-gpu wrapper dry-run",
            _wrapper_dry_run_command(
                python_exe=python_exe,
                preset="standard-auto-gpu",
                run_label="standard_auto_gpu_surface_ci",
            ),
        ),
        VerificationStep(
            "Standard multideck wrapper dry-run",
            _wrapper_dry_run_command(
                python_exe=python_exe,
                preset="standard-multideck",
                run_label="standard_multideck_surface_ci",
            ),
        ),
    )


def verification_request(*, repo_root: Path, python_exe: str) -> VerificationRequest:
    return VerificationRequest(repo_root=repo_root, python_exe=python_exe)


def build_release_verification_steps_for_request(request: VerificationRequest) -> tuple[VerificationStep, ...]:
    return build_release_verification_steps(python_exe=request.python_exe)


def render_verification_plan(steps: Sequence[VerificationStep]) -> list[tuple[str, list[str]]]:
    return [(step.label, list(step.command)) for step in steps]


def render_verification_plan_for_request(request: VerificationRequest) -> list[tuple[str, list[str]]]:
    return render_verification_plan(build_release_verification_steps_for_request(request))


__all__ = [
    "VerificationRequest",
    "VerificationStep",
    "build_release_verification_steps",
    "build_release_verification_steps_for_request",
    "render_verification_plan",
    "render_verification_plan_for_request",
    "verification_request",
]
