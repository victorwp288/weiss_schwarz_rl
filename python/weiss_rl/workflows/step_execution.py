from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

CommandRunner = Callable[..., subprocess.CompletedProcess[Any]]
RemoveTree = Callable[..., Any]


class CommandStep(Protocol):
    @property
    def label(self) -> str: ...

    @property
    def command(self) -> tuple[str, ...]: ...


class CleanableCommandStep(Protocol):
    @property
    def label(self) -> str: ...

    @property
    def command(self) -> tuple[str, ...] | None: ...

    @property
    def clean_dir(self) -> Path | None: ...


def resolve_under_repo(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def display_command(command: Sequence[str]) -> str:
    return " ".join(command)


def print_step_label(label: str) -> None:
    print()
    print(f"==> {label}")


def run_labeled_command(
    *,
    label: str,
    command: Sequence[str],
    cwd: Path,
    command_runner: CommandRunner = subprocess.run,
) -> None:
    print_step_label(label)
    command_list = list(command)
    print(display_command(command_list))
    command_runner(command_list, cwd=cwd, check=True)


def run_command_steps(
    *,
    steps: Sequence[CommandStep],
    repo_root: Path,
    command_runner: CommandRunner = subprocess.run,
) -> None:
    for step in steps:
        run_labeled_command(
            label=step.label,
            command=step.command,
            cwd=repo_root,
            command_runner=command_runner,
        )


def run_cleanable_command_steps(
    *,
    steps: Sequence[CleanableCommandStep],
    repo_root: Path,
    dry_run: bool,
    command_runner: CommandRunner = subprocess.run,
    remove_tree: RemoveTree | None = None,
) -> None:
    rmtree = remove_tree
    if rmtree is None:
        import shutil

        rmtree = shutil.rmtree

    for step in steps:
        print_step_label(step.label)
        if step.clean_dir is not None:
            clean_dir = resolve_under_repo(repo_root, step.clean_dir)
            print(f"rm -rf {clean_dir}")
            if not dry_run:
                rmtree(clean_dir, ignore_errors=True)
        if step.command is not None:
            command_list = list(step.command)
            print(display_command(command_list))
            if not dry_run:
                command_runner(command_list, cwd=repo_root, check=True)


__all__ = [
    "CleanableCommandStep",
    "CommandRunner",
    "CommandStep",
    "RemoveTree",
    "display_command",
    "print_step_label",
    "resolve_under_repo",
    "run_cleanable_command_steps",
    "run_command_steps",
    "run_labeled_command",
]
