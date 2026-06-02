from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.workflows.thesis_wrapper_support.inputs import ThesisWrapperInputs


@dataclass(frozen=True, slots=True)
class ThesisWrapperRequest:
    inputs: ThesisWrapperInputs
    repo_root: Path
    python_exe: str
    run_dir: Path
    stack_config: Path
    eval_stack_config: Path
    eval_preset: str

    @property
    def run_label(self) -> str:
        return self.inputs.run_label

    @property
    def dry_run(self) -> bool:
        return self.inputs.dry_run


@dataclass(frozen=True, slots=True)
class ThesisWrapperPlan:
    repo_root: Path
    python_exe: str
    run_label: str
    run_dir: Path
    stack_config: Path
    eval_stack_config: Path
    preset: str
    eval_preset: str
    train_command: list[str]
    eval_command: list[str] | None
    compare_command: list[str] | None
    b1_baseline_run_dir: Path | None
    dry_run: bool


@dataclass(frozen=True, slots=True)
class ThesisWrapperResult:
    plan: ThesisWrapperPlan
    steps: list[dict[str, Any]]
    failed: bool

    @property
    def status(self) -> str:
        if self.failed:
            return "failed"
        return "planned" if self.plan.dry_run else "completed"


__all__ = ["ThesisWrapperPlan", "ThesisWrapperRequest", "ThesisWrapperResult"]
