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

from weiss_rl.artifacts.reproducibility import canonical_json_bytes, sha256_hex
from weiss_rl.core.spec import assert_spec_bundle_contract, parse_spec_bundle

_COLLECTION_SCRIPT = """
import json
import sys
import weiss_sim

THESIS_DECK_PRESETS = (
    "main_deck_5hy_yotsuba_v1",
    "aggro_deck_5hy_nino_v1",
    "control_deck_jj_s66_v1",
)

def _deck_preset_payload():
    cards = getattr(weiss_sim, "cards", None)
    if cards is None:
        return {"available": [], "profiles": {}, "error": "missing weiss_sim.cards namespace"}
    try:
        available = list(cards.presets())
    except Exception as exc:
        return {"available": [], "profiles": {}, "error": f"cards.presets failed: {exc}"}
    profiles = {}
    for preset in THESIS_DECK_PRESETS:
        try:
            profiles[preset] = str(cards.preset_min_rules_profile(preset))
        except Exception as exc:
            profiles[preset] = f"error: {exc}"
    return {"available": available, "profiles": profiles, "error": ""}

payload = {
    "simulator": {
        "version": getattr(weiss_sim, "__version__", ""),
        "module_file": getattr(weiss_sim, "__file__", ""),
        "build_info": weiss_sim.build_info(),
        "db_info": weiss_sim.db_info(),
        "thesis_deck_presets": _deck_preset_payload(),
    },
    "spec_bundle": weiss_sim.export_spec_bundle(),
}
print(json.dumps(payload, sort_keys=True))
""".strip()

MIN_WEISS_SIM_VERSION = "1.2.0"
_MIN_WEISS_SIM_VERSION_PARTS = (1, 2, 0)
THESIS_DECK_PRESETS = (
    "main_deck_5hy_yotsuba_v1",
    "aggro_deck_5hy_nino_v1",
    "control_deck_jj_s66_v1",
)
_REQUIRED_RUNTIME_ATTRS = (
    "__version__",
    "fast",
    "inspect",
    "make_pool",
    "EnvPoolBuffers",
    "export_spec_bundle",
    "OBS_LEN",
    "ACTION_SPACE_SIZE",
    "SPEC_HASH",
    "PASS_ACTION_ID",
    "rl",
)
_REQUIRED_RL_ATTRS = (
    "reset_rl",
    "step_rl",
    "step_rl_sample_from_logits",
    "step_rl_sample_from_logits_with_logp",
)


@dataclass(frozen=True, slots=True)
class SimulatorContract:
    simulator: dict[str, Any]
    spec_bundle: dict[str, Any]
    spec_hash256: str


@dataclass(frozen=True, slots=True)
class _ProbeTarget:
    python: str
    pythonpath: Path | None = None


def parse_version_parts(version: object) -> tuple[int, int, int] | None:
    if not isinstance(version, str):
        return None
    release = version.strip().split("+", 1)[0].split("-", 1)[0]
    if not release:
        return None
    parts: list[int] = []
    for token in release.split("."):
        if not token.isdigit():
            return None
        parts.append(int(token))
    while len(parts) < 3:
        parts.append(0)
    return (parts[0], parts[1], parts[2])


def validate_simulator_runtime(simulator: dict[str, Any]) -> None:
    version_parts = parse_version_parts(simulator.get("version"))
    if version_parts is None:
        raise RuntimeError("active weiss_sim runtime does not expose a parseable __version__")
    if version_parts < _MIN_WEISS_SIM_VERSION_PARTS:
        raise RuntimeError(
            f"active weiss_sim version {simulator.get('version')} is below required {MIN_WEISS_SIM_VERSION}"
        )

    thesis_decks = simulator.get("thesis_deck_presets")
    if not isinstance(thesis_decks, dict):
        raise RuntimeError("active weiss_sim probe did not report thesis deck presets")
    if str(thesis_decks.get("error", "")).strip():
        raise RuntimeError(str(thesis_decks["error"]))
    available = thesis_decks.get("available", [])
    if not isinstance(available, list):
        raise RuntimeError("active weiss_sim cards.presets() returned an invalid preset list")
    available_names = {str(name) for name in available}
    missing = [preset for preset in THESIS_DECK_PRESETS if preset not in available_names]
    if missing:
        raise RuntimeError(f"active weiss_sim runtime is missing thesis deck presets: {', '.join(missing)}")

    profiles = thesis_decks.get("profiles", {})
    if not isinstance(profiles, dict):
        raise RuntimeError("active weiss_sim probe did not report thesis deck preset profiles")
    bad_profiles = [
        f"{preset}={profiles.get(preset)!r}"
        for preset in THESIS_DECK_PRESETS
        if str(profiles.get(preset, "")).strip().lower() != "approx"
    ]
    if bad_profiles:
        raise RuntimeError(
            "active weiss_sim thesis deck presets are not all marked with the approx min-rules profile: "
            + ", ".join(bad_profiles)
        )


def validate_imported_weiss_sim_runtime(weiss_sim: Any) -> None:
    missing_runtime_attrs = [attr_name for attr_name in _REQUIRED_RUNTIME_ATTRS if not hasattr(weiss_sim, attr_name)]
    if missing_runtime_attrs:
        raise RuntimeError(f"active weiss_sim runtime is missing stepping APIs: {', '.join(missing_runtime_attrs)}")

    version_parts = parse_version_parts(getattr(weiss_sim, "__version__", None))
    if version_parts is None:
        raise RuntimeError("active weiss_sim runtime does not expose a parseable __version__")
    if version_parts < _MIN_WEISS_SIM_VERSION_PARTS:
        raise RuntimeError(
            f"active weiss_sim version {weiss_sim.__version__} is below required {MIN_WEISS_SIM_VERSION}"
        )

    rl_module = weiss_sim.rl
    missing_rl_attrs = [attr_name for attr_name in _REQUIRED_RL_ATTRS if not hasattr(rl_module, attr_name)]
    if missing_rl_attrs:
        raise RuntimeError(f"active weiss_sim.rl is missing runtime methods: {', '.join(missing_rl_attrs)}")

    cards = getattr(weiss_sim, "cards", None)
    if cards is None:
        raise RuntimeError("active weiss_sim runtime is missing cards preset APIs")
    available_names = {str(name) for name in cards.presets()}
    missing = [preset for preset in THESIS_DECK_PRESETS if preset not in available_names]
    if missing:
        raise RuntimeError(f"active weiss_sim runtime is missing thesis deck presets: {', '.join(missing)}")
    bad_profiles = [
        f"{preset}={cards.preset_min_rules_profile(preset)!r}"
        for preset in THESIS_DECK_PRESETS
        if str(cards.preset_min_rules_profile(preset)).strip().lower() != "approx"
    ]
    if bad_profiles:
        raise RuntimeError(
            "active weiss_sim thesis deck presets are not all marked with the approx min-rules profile: "
            + ", ".join(bad_profiles)
        )


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
            _ProbeTarget(python=python, pythonpath=pythonpath) for pythonpath in _candidate_pythonpaths(repo_root)
        )
    return targets


def _run_probe(target: _ProbeTarget) -> dict[str, Any]:
    env = os.environ.copy()
    if target.pythonpath is not None:
        extra_path = str(target.pythonpath)
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = extra_path if not existing_pythonpath else f"{extra_path}{os.pathsep}{existing_pythonpath}"
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
        raw_spec_bundle = payload.get("spec_bundle", {})
        try:
            parsed_spec_bundle = parse_spec_bundle(raw_spec_bundle)
        except ValueError as exc:
            failures.append(f"- {_target_label(target)}: invalid spec_bundle payload: {exc}")
            continue

        spec_bundle = parsed_spec_bundle.to_dict()
        simulator["compatibility_hash"] = parsed_spec_bundle.compatibility_hash
        simulator["probe_python"] = target.python
        simulator["probe_pythonpath"] = None if target.pythonpath is None else target.pythonpath.as_posix()
        simulator["probe_target"] = _target_label(target)
        simulator["probe_source"] = "active_interpreter" if target.pythonpath is None else "pythonpath"
        try:
            validate_simulator_runtime(simulator)
        except RuntimeError as exc:
            failures.append(f"- {_target_label(target)}: {exc}")
            continue
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


def load_verified_simulator_contract(repo_root: Path, *, expected_spec_hash: str) -> SimulatorContract:
    """Load the active simulator contract and optionally assert the caller-supplied hash."""

    contract = load_simulator_contract(repo_root)
    assert_spec_bundle_contract(expected_spec_hash, contract.spec_bundle)
    return contract
