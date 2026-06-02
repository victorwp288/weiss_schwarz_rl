from __future__ import annotations

import subprocess
from pathlib import Path

from weiss_rl.diagnostics.repo_hygiene import run_repo_hygiene_check, summary_payload
from weiss_rl.workflows import cli_parser, parsers

REPO_ROOT = Path(__file__).resolve().parents[3]


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
    assert summary.script_shim_count >= 30
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


def test_repo_hygiene_flags_oversized_legacy_script(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    script_dir = tmp_path / "python" / "scripts"
    script_dir.mkdir(parents=True)
    script_path = script_dir / "legacy.py"
    script_path.write_text(
        '"""Compatibility shim for a package-owned command."""\n' + "\n".join("x = 1" for _ in range(130)),
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "python/scripts/legacy.py"], cwd=tmp_path, check=True, capture_output=True, text=True)

    summary = run_repo_hygiene_check(repo_root=tmp_path)

    assert not summary.passed
    assert any(finding.code == "oversized_legacy_script" for finding in summary.findings)


def test_legacy_cli_parser_facade_delegates_to_canonical_parser() -> None:
    assert cli_parser.build_workflow_parser is parsers.build_parser
    assert cli_parser.parse_workflow_args is parsers._parse_args
