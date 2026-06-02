from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ArtifactContractRequest:
    repo_root: Path
    python_exe: str
    toy_run_dir: Path = Path("runs/toy_public_demo_ci")
    readiness_run_dir: Path = Path("runs/paper_readiness_fixture_ci")
    dry_run: bool = False
    plan_json: Path | None = None


@dataclass(frozen=True, slots=True)
class ArtifactContractStep:
    label: str
    command: tuple[str, ...] | None = None
    clean_dir: Path | None = None


def _module_command(python_exe: str, module: str, *args: str) -> tuple[str, ...]:
    return (python_exe, "-m", module, *args)


def build_artifact_contract_steps(
    *,
    python_exe: str,
    toy_run_dir: Path = Path("runs/toy_public_demo_ci"),
    readiness_run_dir: Path = Path("runs/paper_readiness_fixture_ci"),
) -> tuple[ArtifactContractStep, ...]:
    return (
        ArtifactContractStep("Clean toy public demo run", clean_dir=toy_run_dir),
        ArtifactContractStep(
            "Train toy public demo",
            _module_command(
                python_exe,
                "weiss_rl.training.train_entrypoint",
                "--stack-config",
                "configs/presets/structured_acceptance_standard.yaml",
                "--public-demo",
                "--run-label",
                toy_run_dir.name,
            ),
        ),
        ArtifactContractStep(
            "Evaluate toy public demo",
            _module_command(
                python_exe,
                "weiss_rl.workflows.eval_entrypoint",
                "--stack-config",
                "configs/presets/structured_acceptance_standard_thesis_eval.yaml",
                "--public-demo",
                "--run-dir",
                toy_run_dir.as_posix(),
            ),
        ),
        ArtifactContractStep(
            "Render toy public demo figures",
            _module_command(
                python_exe,
                "weiss_rl.workflows.figures.figures_entrypoint",
                "--public-demo",
                "--final-eval-dir",
                (toy_run_dir / "eval" / "final_eval").as_posix(),
                "--out-dir",
                (toy_run_dir / "figures").as_posix(),
            ),
        ),
        ArtifactContractStep(
            "Scan toy public demo artifacts",
            _module_command(
                python_exe,
                "weiss_rl.diagnostics.artifact_scan_entrypoint",
                "--artifact-root",
                toy_run_dir.as_posix(),
            ),
        ),
        ArtifactContractStep("Clean paper-readiness fixture", clean_dir=readiness_run_dir),
        ArtifactContractStep(
            "Write paper-readiness fixture",
            _module_command(
                python_exe,
                "weiss_rl.eval.readiness.fixture_entrypoint",
                "--run-dir",
                readiness_run_dir.as_posix(),
            ),
        ),
        ArtifactContractStep(
            "Check paper-readiness fixture",
            _module_command(
                python_exe,
                "weiss_rl.eval.readiness.check_entrypoint",
                "--run-dir",
                readiness_run_dir.as_posix(),
            ),
        ),
    )


def build_artifact_contract_steps_for_request(
    request: ArtifactContractRequest,
) -> tuple[ArtifactContractStep, ...]:
    return build_artifact_contract_steps(
        python_exe=request.python_exe,
        toy_run_dir=request.toy_run_dir,
        readiness_run_dir=request.readiness_run_dir,
    )


def render_artifact_contract_plan(steps: tuple[ArtifactContractStep, ...]) -> list[dict[str, object]]:
    rendered: list[dict[str, object]] = []
    for step in steps:
        payload: dict[str, object] = {"label": step.label}
        if step.clean_dir is not None:
            payload["clean_dir"] = step.clean_dir.as_posix()
        if step.command is not None:
            payload["command"] = list(step.command)
        rendered.append(payload)
    return rendered


def render_artifact_contract_plan_for_request(request: ArtifactContractRequest) -> list[dict[str, object]]:
    return render_artifact_contract_plan(build_artifact_contract_steps_for_request(request))


__all__ = [
    "ArtifactContractRequest",
    "ArtifactContractStep",
    "build_artifact_contract_steps",
    "build_artifact_contract_steps_for_request",
    "render_artifact_contract_plan",
    "render_artifact_contract_plan_for_request",
]
