from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def _command_display(command: list[str]) -> str:
    return " ".join(command)


def _summary_path(repo_root: Path, *, run_label: str, dry_run: bool) -> Path:
    if dry_run:
        return repo_root / "runs" / "_wrapper_plans" / f"{run_label}.json"
    return repo_root / "runs" / run_label / "thesis_run_summary.json"


def _run_step(*, command: list[str], cwd: Path, dry_run: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "command": command,
        "cwd": cwd.as_posix(),
        "status": "planned" if dry_run else "running",
    }
    print(_command_display(command))
    if dry_run:
        return payload
    try:
        subprocess.run(command, cwd=cwd, check=True)
    except subprocess.CalledProcessError as exc:
        payload["status"] = "failed"
        payload["returncode"] = int(exc.returncode)
        raise
    payload["status"] = "completed"
    payload["returncode"] = 0
    return payload
