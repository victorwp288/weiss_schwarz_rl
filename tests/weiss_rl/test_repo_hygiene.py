from __future__ import annotations

import subprocess
from importlib.util import find_spec
from pathlib import Path

from weiss_rl.diagnostics.repo_hygiene import RepoHygieneFinding, run_repo_hygiene_check, summary_payload

REPO_ROOT = Path(__file__).resolve().parents[2]


def _init_git_repo(repo_root: Path) -> None:
    subprocess.run(["git", "init"], cwd=repo_root, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.name", "Victor"], cwd=repo_root, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.email", "victor@example.test"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )


def test_current_repo_passes_repo_hygiene_check() -> None:
    summary = run_repo_hygiene_check(repo_root=REPO_ROOT)

    assert summary.passed
    assert summary.legacy_script_count == 0
    assert summary_payload(summary)["findings"] == []


def test_repo_hygiene_flags_generated_top_level_tracked_file(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    (tmp_path / "README.md").write_text("scratch\n", encoding="utf-8")
    (tmp_path / "now.zip").write_text("generated\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md", "now.zip"], cwd=tmp_path, check=True, capture_output=True, text=True)

    summary = run_repo_hygiene_check(repo_root=tmp_path)

    assert not summary.passed
    assert {finding.code for finding in summary.findings} == {
        "unexpected_tracked_top_level",
        "tracked_generated_top_level",
    }


def test_repo_hygiene_flags_legacy_script_entrypoint(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    script_dir = tmp_path / "python" / "scripts"
    script_dir.mkdir(parents=True)
    script_path = script_dir / "legacy.py"
    script_path.write_text("from weiss_rl.cli import main\n", encoding="utf-8")
    subprocess.run(["git", "add", "python/scripts/legacy.py"], cwd=tmp_path, check=True, capture_output=True, text=True)

    summary = run_repo_hygiene_check(repo_root=tmp_path)

    assert not summary.passed
    assert summary.findings == (
        RepoHygieneFinding(
            code="legacy_script_entrypoint",
            path="python/scripts/legacy.py",
            message=(
                "Path-based Python script entrypoints were retired; use package modules under "
                "`python -m weiss_rl...` instead."
            ),
        ),
    )


def test_legacy_cli_parser_facade_is_removed() -> None:
    assert find_spec("weiss_rl.workflows.cli_parser") is None


def test_workflows_package_does_not_recreate_legacy_aliases() -> None:
    import weiss_rl.workflows as workflows

    assert not hasattr(workflows, "__getattr__")
