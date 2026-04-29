"""Run provenance and manifest utility helpers."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

_GIT_COMMIT_HEX_LENGTH = 40
_U64_MASK = (1 << 64) - 1


def git_output(args: list[str], *, repo_root: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    return result.stdout.strip()


def git_commit(*, repo_root: Path) -> str:
    override = str(os.environ.get("WEISS_RL_GIT_COMMIT", "")).strip().lower()
    if len(override) == _GIT_COMMIT_HEX_LENGTH and all(char in "0123456789abcdef" for char in override):
        return override
    try:
        return git_output(["rev-parse", "HEAD"], repo_root=repo_root)
    except (OSError, subprocess.CalledProcessError):
        return ""


def git_dirty(*, repo_root: Path) -> bool:
    try:
        return bool(git_output(["status", "--short"], repo_root=repo_root))
    except (OSError, subprocess.CalledProcessError):
        return False


def start_nonce() -> int:
    return time.time_ns() & _U64_MASK


def manifest_source_path(path: Path, *, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON must contain an object at the top level")
    return payload
