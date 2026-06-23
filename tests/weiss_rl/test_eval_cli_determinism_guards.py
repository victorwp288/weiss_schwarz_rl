from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

ENTRYPOINT_MODULES = {
    "parallel_final_eval": "weiss_rl.eval.parallel_final_eval_entrypoint",
    "targeted_confirm_eval": "weiss_rl.eval.targeted_confirm.entrypoint",
}


def _script_env() -> dict[str, str]:
    env = dict(os.environ)
    python_path = str(REPO_ROOT / "python")
    env["PYTHONPATH"] = python_path if not env.get("PYTHONPATH") else python_path + os.pathsep + env["PYTHONPATH"]
    return env


def _script_env_without_hash_seed() -> dict[str, str]:
    env = _script_env()
    env.pop("PYTHONHASHSEED", None)
    return env


def test_targeted_confirm_eval_rejects_parallel_workers_without_escape_hatch() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            ENTRYPOINT_MODULES["targeted_confirm_eval"],
            "--stack-config",
            "missing.yaml",
            "--run-dir",
            "runs/missing",
            "--snapshot-registry-json",
            "missing_registry.json",
            "--b1-baseline-run-dir",
            "runs/missing_b1",
            "--focal-policy-id",
            "policy_000001",
            "--workers",
            "2",
        ],
        cwd=REPO_ROOT,
        env=_script_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "deterministic only with --workers 1" in result.stderr


def test_parallel_final_eval_rejects_parallel_workers_without_escape_hatch() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            ENTRYPOINT_MODULES["parallel_final_eval"],
            "--stack-config",
            "missing.yaml",
            "--run-dir",
            "runs/missing",
            "--policy-id",
            "policy_000001",
            "--workers",
            "2",
        ],
        cwd=REPO_ROOT,
        env=_script_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "deterministic only with --workers 1" in result.stderr


def test_targeted_confirm_eval_requires_fixed_pythonhashseed() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            ENTRYPOINT_MODULES["targeted_confirm_eval"],
            "--stack-config",
            "missing.yaml",
            "--run-dir",
            "runs/missing",
            "--snapshot-registry-json",
            "missing_registry.json",
            "--b1-baseline-run-dir",
            "runs/missing_b1",
            "--focal-policy-id",
            "policy_000001",
            "--workers",
            "1",
        ],
        cwd=REPO_ROOT,
        env=_script_env_without_hash_seed(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "requires a fixed PYTHONHASHSEED" in result.stderr


def test_parallel_final_eval_requires_fixed_pythonhashseed() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            ENTRYPOINT_MODULES["parallel_final_eval"],
            "--stack-config",
            "missing.yaml",
            "--run-dir",
            "runs/missing",
            "--policy-id",
            "policy_000001",
            "--workers",
            "1",
        ],
        cwd=REPO_ROOT,
        env=_script_env_without_hash_seed(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "requires a fixed PYTHONHASHSEED" in result.stderr
