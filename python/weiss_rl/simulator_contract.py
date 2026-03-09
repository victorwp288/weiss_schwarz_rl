"""Helpers for collecting simulator provenance and spec bundles."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.repro import canonical_json_bytes, sha256_hex

_COLLECTION_SCRIPT = """
import json
import weiss_sim

payload = {
    "simulator": {
        "version": getattr(weiss_sim, "__version__", ""),
        "build_info": weiss_sim.build_info(),
        "db_info": weiss_sim.db_info(),
    },
    "spec_bundle": weiss_sim.export_spec_bundle(),
}
print(json.dumps(payload, sort_keys=True))
""".strip()


@dataclass(frozen=True, slots=True)
class SimulatorContract:
    simulator: dict[str, Any]
    spec_bundle: dict[str, Any]
    spec_hash256: str


@dataclass(frozen=True, slots=True)
class _ProbeTarget:
    python: str
    pythonpath: Path | None = None


def _git_common_repo_root(repo_root: Path) -> Path | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--path-format=absolute", "--git-common-dir"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    git_common_dir = Path(result.stdout.strip())
    if git_common_dir.name != ".git":
        return None
    return git_common_dir.parent


def _candidate_pythonpaths(repo_root: Path) -> list[Path]:
    candidates: list[Path] = []
    env_path = os.environ.get("WEISS_SIM_PYTHONPATH", "").strip()
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(repo_root.parent / "weiss-schwarz-simulator" / "python")
    common_repo_root = _git_common_repo_root(repo_root)
    if common_repo_root is not None:
        candidates.append(common_repo_root.parent / "weiss-schwarz-simulator" / "python")

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen or not resolved.is_dir():
            continue
        seen.add(resolved)
        unique_candidates.append(resolved)
    return unique_candidates


def _candidate_pythons() -> list[str]:
    candidates: list[str] = []
    env_python = os.environ.get("WEISS_SIM_PYTHON", "").strip()
    if env_python:
        candidates.append(env_python)
    candidates.extend(
        python
        for python in (
            sys.executable,
            shutil.which("python3.12"),
            shutil.which("python3"),
        )
        if python
    )

    unique_candidates: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)
    return unique_candidates


def _candidate_targets(repo_root: Path) -> list[_ProbeTarget]:
    targets: list[_ProbeTarget] = []
    for python in _candidate_pythons():
        targets.append(_ProbeTarget(python=python))
        targets.extend(
            _ProbeTarget(python=python, pythonpath=pythonpath)
            for pythonpath in _candidate_pythonpaths(repo_root)
        )
    return targets


def _run_probe(target: _ProbeTarget) -> dict[str, Any]:
    env = os.environ.copy()
    if target.pythonpath is not None:
        extra_path = str(target.pythonpath)
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = extra_path if not existing_pythonpath else f"{extra_path}:{existing_pythonpath}"
    result = subprocess.run(
        [target.python, "-c", _COLLECTION_SCRIPT],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise RuntimeError("Simulator probe returned a non-mapping payload")
    return payload


def _target_label(target: _ProbeTarget) -> str:
    if target.pythonpath is None:
        return f"python={target.python}"
    return f"python={target.python} pythonpath={target.pythonpath}"


def load_simulator_contract(repo_root: Path) -> SimulatorContract:
    failures: list[str] = []
    for target in _candidate_targets(repo_root):
        try:
            payload = _run_probe(target)
        except (OSError, subprocess.CalledProcessError, json.JSONDecodeError, RuntimeError) as exc:
            failures.append(f"- {_target_label(target)}: {exc}")
            continue

        simulator = dict(payload.get("simulator", {}))
        spec_bundle = dict(payload.get("spec_bundle", {}))
        if not spec_bundle:
            failures.append(f"- {_target_label(target)}: empty spec_bundle payload")
            continue

        if "spec_hash" in spec_bundle:
            simulator["compatibility_hash"] = str(spec_bundle["spec_hash"])
        return SimulatorContract(
            simulator=simulator,
            spec_bundle=spec_bundle,
            spec_hash256=sha256_hex(canonical_json_bytes(spec_bundle)),
        )

    tried = "\n".join(failures) or "- no simulator candidates found"
    raise RuntimeError(
        "Unable to collect simulator provenance via weiss_sim.export_spec_bundle(). "
        "If weiss_sim is not importable in the active interpreter, set WEISS_SIM_PYTHONPATH and optionally "
        "WEISS_SIM_PYTHON to a working simulator environment.\n"
        f"Tried:\n{tried}"
    )
