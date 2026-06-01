from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def _repo_root(args_repo_root: Path | None) -> Path:
    return Path(__file__).resolve().parents[3] if args_repo_root is None else args_repo_root.resolve()


def _display(command: list[str]) -> str:
    return " ".join(command)


def _write_plan(*, repo_root: Path, name: str, command: list[str], payload: dict[str, Any]) -> None:
    plan_dir = repo_root / "runs" / "_workflow_plans"
    plan_dir.mkdir(parents=True, exist_ok=True)
    plan_payload = dict(payload)
    plan_payload["command"] = command
    plan_payload["cwd"] = repo_root.as_posix()
    plan_payload["status"] = "planned"
    (plan_dir / f"{name}.json").write_text(json.dumps(plan_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_or_plan(
    *,
    repo_root: Path,
    plan_name: str,
    command: list[str],
    dry_run: bool,
    payload: dict[str, Any],
) -> None:
    print(_display(command))
    if dry_run:
        _write_plan(repo_root=repo_root, name=plan_name, command=command, payload=payload)
        return
    completed = subprocess.run(command, cwd=repo_root, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
