from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from weiss_rl.workflows.artifact_contract.artifact_contract_entrypoint import (
    _repo_root as artifact_contract_repo_root,
)
from weiss_rl.workflows.artifact_contract.artifact_contract_entrypoint import (
    artifact_contract_request,
    run_artifact_contract_request,
    run_artifact_contract_steps,
)
from weiss_rl.workflows.artifact_contract.artifact_contract_plan import (
    ArtifactContractRequest,
    ArtifactContractStep,
    build_artifact_contract_steps,
    build_artifact_contract_steps_for_request,
    render_artifact_contract_plan,
    render_artifact_contract_plan_for_request,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_artifact_contract_entrypoint_default_repo_root_is_project_root() -> None:
    assert artifact_contract_repo_root() == REPO_ROOT


def test_artifact_contract_plan_preserves_public_demo_and_readiness_contract_steps() -> None:
    request = ArtifactContractRequest(
        repo_root=REPO_ROOT,
        python_exe="python.exe",
        toy_run_dir=Path("runs/toy_public_demo_ci"),
        readiness_run_dir=Path("runs/paper_readiness_fixture_ci"),
        dry_run=True,
    )
    steps = build_artifact_contract_steps_for_request(request)
    legacy_steps = build_artifact_contract_steps(
        python_exe="python.exe",
        toy_run_dir=Path("runs/toy_public_demo_ci"),
        readiness_run_dir=Path("runs/paper_readiness_fixture_ci"),
    )
    rendered = render_artifact_contract_plan(steps)

    assert steps == legacy_steps
    assert render_artifact_contract_plan_for_request(request) == rendered
    assert [step["label"] for step in rendered] == [
        "Clean toy public demo run",
        "Train toy public demo",
        "Evaluate toy public demo",
        "Render toy public demo figures",
        "Scan toy public demo artifacts",
        "Clean paper-readiness fixture",
        "Write paper-readiness fixture",
        "Check paper-readiness fixture",
    ]
    assert rendered[0] == {"label": "Clean toy public demo run", "clean_dir": "runs/toy_public_demo_ci"}
    assert rendered[1]["command"] == [
        "python.exe",
        "-m",
        "weiss_rl.training.train_entrypoint",
        "--stack-config",
        "configs/presets/structured_acceptance_standard.yaml",
        "--public-demo",
        "--run-label",
        "toy_public_demo_ci",
    ]
    assert rendered[4]["command"] == [
        "python.exe",
        "-m",
        "weiss_rl.diagnostics.artifact_scan_entrypoint",
        "--artifact-root",
        "runs/toy_public_demo_ci",
    ]
    assert rendered[6]["command"] == [
        "python.exe",
        "-m",
        "weiss_rl.eval.readiness.fixture_entrypoint",
        "--run-dir",
        "runs/paper_readiness_fixture_ci",
    ]
    assert rendered[7]["command"] == [
        "python.exe",
        "-m",
        "weiss_rl.eval.readiness.check_entrypoint",
        "--run-dir",
        "runs/paper_readiness_fixture_ci",
    ]


def test_artifact_contract_runner_dry_run_does_not_delete_or_execute(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    clean_dir = repo_root / "runs" / "toy_public_demo_ci"
    clean_dir.mkdir(parents=True)
    (clean_dir / "keep.txt").write_text("dry run keeps this\n", encoding="utf-8")
    observed: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[object]:
        observed.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    steps = build_artifact_contract_steps(
        python_exe="python.exe",
        toy_run_dir=Path("runs/toy_public_demo_ci"),
        readiness_run_dir=Path("runs/paper_readiness_fixture_ci"),
    )

    run_artifact_contract_steps(steps=steps, repo_root=repo_root, dry_run=True)

    assert (clean_dir / "keep.txt").is_file()
    assert observed == []


def test_artifact_contract_runner_non_dry_cleans_and_executes_in_order(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    clean_dir = repo_root / "runs" / "old_fixture"
    observed_clean: list[tuple[Path, bool]] = []
    observed_commands: list[tuple[list[str], Path, bool]] = []
    steps = (
        ArtifactContractStep("Clean old fixture", clean_dir=Path("runs/old_fixture")),
        ArtifactContractStep("Run first command", command=("python.exe", "-m", "first.module")),
        ArtifactContractStep("Run second command", command=("python.exe", "-m", "second.module")),
    )

    def fake_remove_tree(path: Path, **kwargs: object) -> None:
        observed_clean.append((path, bool(kwargs.get("ignore_errors"))))

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[object]:
        observed_commands.append((command, Path(kwargs["cwd"]), bool(kwargs["check"])))
        return subprocess.CompletedProcess(command, 0)

    run_artifact_contract_steps(
        steps=steps,
        repo_root=repo_root,
        dry_run=False,
        command_runner=fake_run,
        remove_tree=fake_remove_tree,
    )

    assert observed_clean == [(clean_dir, True)]
    assert observed_commands == [
        (["python.exe", "-m", "first.module"], repo_root, True),
        (["python.exe", "-m", "second.module"], repo_root, True),
    ]


def test_artifact_contract_request_runs_and_writes_plan_json(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True)
    plan_json = Path("plans/artifact_contract.json")
    observed_commands: list[list[str]] = []

    args = SimpleNamespace(
        toy_run_dir=Path("runs/toy_public_demo_ci"),
        readiness_run_dir=Path("runs/paper_readiness_fixture_ci"),
        dry_run=True,
        plan_json=plan_json,
    )

    request = artifact_contract_request(args=args, repo_root=repo_root, python_exe="python.exe")

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[object]:
        observed_commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    run_artifact_contract_request(request=request, command_runner=fake_run)

    payload = json.loads((repo_root / plan_json).read_text(encoding="utf-8"))
    assert request.repo_root == repo_root
    assert request.dry_run is True
    assert observed_commands == []
    assert payload[0]["clean_dir"] == "runs/toy_public_demo_ci"
    assert payload[5]["clean_dir"] == "runs/paper_readiness_fixture_ci"
    assert payload[-1]["command"][2] == "weiss_rl.eval.readiness.check_entrypoint"


def test_artifact_contract_entrypoint_writes_plan_json_without_running_contract(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True)
    plan_json = tmp_path / "artifact_contract_plan.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "weiss_rl.workflows.artifact_contract.artifact_contract_entrypoint",
            "--repo-root",
            str(repo_root),
            "--toy-run-dir",
            "runs/toy_public_demo_ci",
            "--readiness-run-dir",
            "runs/paper_readiness_fixture_ci",
            "--plan-json",
            str(plan_json),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(plan_json.read_text(encoding="utf-8"))
    assert len(payload) == 8
    assert payload[0]["clean_dir"] == "runs/toy_public_demo_ci"
    assert payload[-1]["command"][2] == "weiss_rl.eval.readiness.check_entrypoint"


def test_make_artifact_contract_target_delegates_to_package_workflow() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")

    assert "artifact-contract: sync" in makefile
    assert "$(PYRUN) -m weiss_rl.workflows.artifact_contract.artifact_contract_entrypoint" in makefile
    assert "$(MAKE) artifact-hygiene" not in makefile
