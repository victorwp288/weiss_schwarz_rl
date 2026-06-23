from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def thesis_wrapper_command() -> list[str]:
    return ["-m", "weiss_rl.workflows.thesis_wrapper"]


def write_stack_config(repo_root: Path, name: str = "stack.yaml") -> Path:
    config_dir = repo_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    stack_config = config_dir / name
    stack_config.write_text("components: []\nconfig: {}\n", encoding="utf-8")
    return stack_config


def run_thesis_wrapper_subprocess(
    repo_root: Path,
    *args: str,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            *thesis_wrapper_command(),
            "--repo-root",
            str(repo_root),
            *args,
        ],
        cwd=cwd or REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def read_wrapper_plan(repo_root: Path, run_label: str) -> dict[str, Any]:
    summary_path = repo_root / "runs" / "_wrapper_plans" / f"{run_label}.json"
    return json.loads(summary_path.read_text(encoding="utf-8"))
