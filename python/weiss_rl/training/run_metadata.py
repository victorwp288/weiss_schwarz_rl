from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

GIT_COMMIT_HEX_LENGTH = 40
U64_MASK = (1 << 64) - 1


def repo_root(anchor: Path) -> Path:
    return anchor.resolve().parents[2]


def git_output(args: list[str], *, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    return result.stdout.strip()


def git_commit(*, cwd: Path, env: dict[str, str] | None = None) -> str:
    environ = os.environ if env is None else env
    override = str(environ.get("WEISS_RL_GIT_COMMIT", "")).strip().lower()
    if len(override) == GIT_COMMIT_HEX_LENGTH and all(char in "0123456789abcdef" for char in override):
        return override
    try:
        return git_output(["rev-parse", "HEAD"], cwd=cwd)
    except (OSError, subprocess.CalledProcessError):
        return ""


def git_dirty(*, cwd: Path) -> bool:
    try:
        return bool(git_output(["status", "--short"], cwd=cwd))
    except (OSError, subprocess.CalledProcessError):
        return False


def start_nonce() -> int:
    return time.time_ns() & U64_MASK


def hardware_summary(
    learner_device: object = "cpu",
    *,
    actor_device: object = "cpu",
    actor_device_layout: Sequence[str] | None = None,
) -> dict[str, str | int]:
    learner_device_name = str(learner_device)
    payload: dict[str, str | int] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count() or 0,
        "learner_device": learner_device_name,
        "actor_device": str(actor_device),
    }
    if actor_device_layout:
        payload["actor_device_layout"] = ",".join(str(device_name) for device_name in actor_device_layout)
        payload["actor_device_unique_count"] = len(
            dict.fromkeys(str(device_name) for device_name in actor_device_layout)
        )
    return payload


def evaluation_pinning(stack: Any) -> dict[str, str | bool | float]:
    if stack.config.evaluation is None:
        return {}
    evaluation = stack.config.evaluation
    return {
        "eval_device": evaluation.eval_device,
        "eval_sampling_algorithm": evaluation.eval_sampling_algorithm,
        "model_sampling_temperature": float(getattr(evaluation, "model_sampling_temperature", 1.0)),
        "eval_inference_mode": evaluation.eval_inference_mode,
        "seat_swap": evaluation.seat_swap,
        "legal_fingerprint_version": evaluation.legal_fingerprint_checks.version,
        "legal_fingerprint_mismatch_policy": evaluation.legal_fingerprint_checks.mismatch_policy,
    }


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
