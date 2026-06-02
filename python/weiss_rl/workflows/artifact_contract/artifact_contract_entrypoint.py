from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from weiss_rl.workflows.artifact_contract.artifact_contract_plan import (
    ArtifactContractRequest,
    ArtifactContractStep,
    build_artifact_contract_steps,
    build_artifact_contract_steps_for_request,
    render_artifact_contract_plan,
)
from weiss_rl.workflows.step_execution import (
    CommandRunner,
    RemoveTree,
    display_command,
    run_cleanable_command_steps,
)
from weiss_rl.workflows.step_execution import (
    resolve_under_repo as _resolve_under_repo,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _display_command(command: tuple[str, ...]) -> str:
    return display_command(command)


def build_artifact_contract_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the artifact hygiene and paper-readiness contract workflow")
    parser.add_argument("--toy-run-dir", type=Path, default=Path("runs/toy_public_demo_ci"))
    parser.add_argument("--readiness-run-dir", type=Path, default=Path("runs/paper_readiness_fixture_ci"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plan-json", type=Path, default=None, help="Optional path to write the rendered step plan")
    parser.add_argument("--repo-root", type=Path, default=None, help=argparse.SUPPRESS)
    return parser


def artifact_contract_request(*, args: argparse.Namespace, repo_root: Path, python_exe: str) -> ArtifactContractRequest:
    return ArtifactContractRequest(
        repo_root=repo_root,
        python_exe=python_exe,
        toy_run_dir=Path(args.toy_run_dir),
        readiness_run_dir=Path(args.readiness_run_dir),
        dry_run=bool(args.dry_run),
        plan_json=args.plan_json,
    )


def write_artifact_contract_plan_json(
    *, request: ArtifactContractRequest, steps: tuple[ArtifactContractStep, ...]
) -> Path:
    if request.plan_json is None:
        raise ValueError("artifact contract plan JSON path is not configured")
    plan_path = _resolve_under_repo(request.repo_root, request.plan_json)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(render_artifact_contract_plan(steps), indent=2) + "\n", encoding="utf-8")
    return plan_path


def run_artifact_contract_steps(
    *,
    steps: tuple[ArtifactContractStep, ...],
    repo_root: Path,
    dry_run: bool,
    command_runner: CommandRunner | None = None,
    remove_tree: RemoveTree | None = None,
) -> None:
    run_cleanable_command_steps(
        steps=steps,
        repo_root=repo_root,
        dry_run=dry_run,
        command_runner=subprocess.run if command_runner is None else command_runner,
        remove_tree=shutil.rmtree if remove_tree is None else remove_tree,
    )


def run_artifact_contract_request(
    *,
    request: ArtifactContractRequest,
    command_runner: CommandRunner | None = None,
    remove_tree: RemoveTree | None = None,
) -> None:
    steps = build_artifact_contract_steps_for_request(request)
    if request.plan_json is not None:
        write_artifact_contract_plan_json(request=request, steps=steps)
    run_artifact_contract_steps(
        steps=steps,
        repo_root=request.repo_root,
        dry_run=request.dry_run,
        command_runner=command_runner,
        remove_tree=remove_tree,
    )


def main() -> None:
    parser = build_artifact_contract_parser()
    args = parser.parse_args()
    repo_root = _repo_root() if args.repo_root is None else Path(args.repo_root).resolve()
    run_artifact_contract_request(
        request=artifact_contract_request(args=args, repo_root=repo_root, python_exe=sys.executable)
    )


__all__ = [
    "ArtifactContractRequest",
    "ArtifactContractStep",
    "_display_command",
    "_repo_root",
    "artifact_contract_request",
    "build_artifact_contract_parser",
    "build_artifact_contract_steps",
    "build_artifact_contract_steps_for_request",
    "main",
    "render_artifact_contract_plan",
    "run_artifact_contract_request",
    "run_artifact_contract_steps",
    "write_artifact_contract_plan_json",
]


if __name__ == "__main__":
    main()
