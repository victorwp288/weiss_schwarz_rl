from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

ALLOWED_TRACKED_TOP_LEVEL = frozenset(
    {
        ".github",
        ".gitignore",
        ".pre-commit-config.yaml",
        "AGENTS.md",
        "CHANGELOG.md",
        "CONTRIBUTING.md",
        "Makefile",
        "README.md",
        "RL_REBUILD_PLAN.md",
        "configs",
        "diagnostics",
        "docs",
        "examples",
        "init.sh",
        "mypy.ini",
        "pyproject.toml",
        "python",
        "run_logs",
        "runs",
        "scripts",
        "tests",
        "thesis_figures_final",
        "uv.lock",
        "vast_artifacts",
        "web",
    }
)

GENERATED_TOP_LEVEL = frozenset(
    {
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".VSCodeCounter",
        "build",
        "dist",
        "now",
        "now.zip",
        "temp",
    }
)

MAX_SCRIPT_SHIM_LINES = 120


@dataclass(frozen=True, slots=True)
class RepoHygieneFinding:
    code: str
    path: str
    message: str


@dataclass(frozen=True, slots=True)
class RepoHygieneSummary:
    passed: bool
    tracked_file_count: int
    script_shim_count: int
    findings: tuple[RepoHygieneFinding, ...]


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[3]


def tracked_files(repo_root: Path) -> tuple[str, ...]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_root,
        capture_output=True,
        check=True,
        text=True,
    )
    return tuple(line for line in result.stdout.splitlines() if line)


def _top_level(path: str) -> str:
    return Path(path).parts[0]


def _tracked_root_findings(paths: tuple[str, ...]) -> list[RepoHygieneFinding]:
    findings: list[RepoHygieneFinding] = []
    for top_level in sorted({_top_level(path) for path in paths}):
        if top_level not in ALLOWED_TRACKED_TOP_LEVEL:
            findings.append(
                RepoHygieneFinding(
                    code="unexpected_tracked_top_level",
                    path=top_level,
                    message="Tracked files should live under the documented repository layout.",
                )
            )
        if top_level in GENERATED_TOP_LEVEL:
            findings.append(
                RepoHygieneFinding(
                    code="tracked_generated_top_level",
                    path=top_level,
                    message="Generated or local-tool output must not be tracked at the repo root.",
                )
            )
    return findings


def _script_shim_findings(repo_root: Path) -> tuple[list[RepoHygieneFinding], int]:
    scripts_dir = repo_root / "python" / "scripts"
    findings: list[RepoHygieneFinding] = []
    script_count = 0
    if not scripts_dir.is_dir():
        return findings, script_count
    for script_path in sorted(scripts_dir.glob("*.py")):
        script_count += 1
        rel_path = script_path.relative_to(repo_root).as_posix()
        text = script_path.read_text(encoding="utf-8")
        line_count = len(text.splitlines())
        if line_count > MAX_SCRIPT_SHIM_LINES:
            findings.append(
                RepoHygieneFinding(
                    code="oversized_legacy_script",
                    path=rel_path,
                    message=(
                        f"Compatibility scripts should stay thin shims; "
                        f"{line_count} lines exceeds {MAX_SCRIPT_SHIM_LINES}."
                    ),
                )
            )
        if "Compatibility shim" not in text[:240]:
            findings.append(
                RepoHygieneFinding(
                    code="legacy_script_missing_shim_marker",
                    path=rel_path,
                    message="Compatibility scripts should state that package modules are canonical.",
                )
            )
    return findings, script_count


def run_repo_hygiene_check(repo_root: Path | None = None) -> RepoHygieneSummary:
    resolved_root = repo_root_from_here() if repo_root is None else repo_root.resolve()
    paths = tracked_files(resolved_root)
    findings = _tracked_root_findings(paths)
    script_findings, script_count = _script_shim_findings(resolved_root)
    findings.extend(script_findings)
    return RepoHygieneSummary(
        passed=not findings,
        tracked_file_count=len(paths),
        script_shim_count=script_count,
        findings=tuple(findings),
    )


def summary_payload(summary: RepoHygieneSummary) -> dict[str, object]:
    return {
        "passed": summary.passed,
        "tracked_file_count": summary.tracked_file_count,
        "script_shim_count": summary.script_shim_count,
        "findings": [asdict(finding) for finding in summary.findings],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check repository layout hygiene")
    parser.add_argument("--repo-root", type=Path, default=None, help="Repository root; defaults to this checkout")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_repo_hygiene_check(repo_root=args.repo_root)
    payload = summary_payload(summary)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif summary.passed:
        print(
            "Repo hygiene checks passed "
            f"({summary.tracked_file_count} tracked files, {summary.script_shim_count} legacy script shims)."
        )
    else:
        print("Repo hygiene checks failed:")
        for finding in summary.findings:
            print(f"- {finding.code}: {finding.path} - {finding.message}")
    return 0 if summary.passed else 1


__all__ = [
    "ALLOWED_TRACKED_TOP_LEVEL",
    "GENERATED_TOP_LEVEL",
    "MAX_SCRIPT_SHIM_LINES",
    "RepoHygieneFinding",
    "RepoHygieneSummary",
    "build_parser",
    "main",
    "repo_root_from_here",
    "run_repo_hygiene_check",
    "summary_payload",
    "tracked_files",
]
