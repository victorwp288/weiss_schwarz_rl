"""Artifact-contract checks used by paper-readiness audits."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from weiss_rl.artifact_contract import REQUIRED_SENSITIVITY_CASE_IDS, resolve_sensitivity_summary_path
from weiss_rl.artifacts import ArtifactLayout


def build_manifest_contract(run_dir: Path) -> dict[str, Any]:
    layout = ArtifactLayout.from_run_dir(run_dir)
    manifest_path = run_dir / "manifest.json"
    try:
        manifest = _load_json_object(manifest_path)
    except Exception as exc:
        return {
            "passed": False,
            "manifest_path": manifest_path.as_posix(),
            "fields": {},
            "consistency_checks": {},
            "missing_fields": [],
            "invalid_fields": [],
            "mismatches": [],
            "reason": exc.__class__.__name__,
            "message": str(exc),
        }

    field_checks = {
        "run_id256": _validate_hex_field(manifest.get("run_id256"), length=64),
        "run_id64": _validate_hex_field(manifest.get("run_id64"), length=16),
        "git_commit": _validate_hex_field(manifest.get("git_commit"), length=40),
        "git_dirty": _validate_bool_field(manifest.get("git_dirty")),
        "spec_hash256": _validate_hex_field(manifest.get("spec_hash256"), length=64),
        "config_hash256": _validate_hex_field(manifest.get("config_hash256"), length=64),
        "simulator": _validate_simulator_manifest(manifest.get("simulator")),
        "spec_bundle": _validate_object_field(manifest.get("spec_bundle"), require_non_empty=True),
        "config_canonical": _validate_object_field(manifest.get("config_canonical"), require_non_empty=True),
        "seed_files": _validate_seed_files_field(manifest.get("seed_files")),
        "hardware": _validate_object_field(manifest.get("hardware"), require_non_empty=True),
        "evaluation_pinning": _validate_object_field(manifest.get("evaluation_pinning"), require_non_empty=True),
        "policy_set_selection": _validate_manifest_policy_set_selection(
            manifest.get("policy_set_selection"),
            details=manifest.get("policy_set_selection_details"),
        ),
    }
    missing_fields = [name for name, result in field_checks.items() if result["reason"] == "missing"]
    invalid_fields = [
        name for name, result in field_checks.items() if not result["passed"] and result["reason"] != "missing"
    ]

    consistency_checks = {
        "spec_bundle_json_matches_manifest": _compare_json_file_to_manifest(
            file_path=run_dir / "spec_bundle.json",
            expected=manifest.get("spec_bundle"),
        ),
        "config_canonical_json_matches_manifest": _compare_json_file_to_manifest(
            file_path=run_dir / "config_canonical.json",
            expected=manifest.get("config_canonical"),
        ),
        "spec_hash_file_matches_manifest": _compare_text_file_to_manifest(
            file_path=run_dir / "spec_hash256.txt",
            expected=manifest.get("spec_hash256"),
        ),
        "config_hash_file_matches_manifest": _compare_text_file_to_manifest(
            file_path=run_dir / "config_hash256.txt",
            expected=manifest.get("config_hash256"),
        ),
        "run_summary_exists": _validate_existing_file(layout.run_summary_path),
        "environment_manifest_exists": _validate_existing_file(layout.environment_path),
        "determinism_report_exists": _validate_existing_file(layout.determinism_report_path),
    }
    mismatches = [name for name, result in consistency_checks.items() if not result["passed"]]
    passed = not missing_fields and not invalid_fields and not mismatches

    return {
        "passed": passed,
        "manifest_path": manifest_path.as_posix(),
        "fields": field_checks,
        "consistency_checks": consistency_checks,
        "missing_fields": missing_fields,
        "invalid_fields": invalid_fields,
        "mismatches": mismatches,
        "message": (
            "manifest satisfies paper-readiness requirements"
            if passed
            else "manifest is missing required fields or has inconsistent companion files"
        ),
    }


def build_final_eval_artifact_contract(final_eval_dir: Path) -> dict[str, Any]:
    summary_path = final_eval_dir / "summary.json"
    try:
        payload = _load_json_object(summary_path)
    except Exception as exc:
        return {
            "passed": False,
            "summary_path": summary_path.as_posix(),
            "policy_ids": [],
            "expected_matchup_count": None,
            "observed_matchup_count": None,
            "missing_matchups": [],
            "duplicate_matchups": [],
            "noncanonical_matchups": [],
            "reference_failures": [],
            "sensitivity_cases": [],
            "reason": exc.__class__.__name__,
            "message": str(exc),
        }

    policy_ids = _policy_ids(payload)
    if len(set(policy_ids)) != len(policy_ids):
        return {
            "passed": False,
            "summary_path": summary_path.as_posix(),
            "policy_ids": list(policy_ids),
            "expected_matchup_count": None,
            "observed_matchup_count": None,
            "missing_matchups": [],
            "duplicate_matchups": [],
            "noncanonical_matchups": [],
            "reference_failures": [],
            "sensitivity_cases": [],
            "reason": "duplicate_policy_ids",
            "message": "final_eval summary policy_ids must be unique",
        }

    expected_keys = {(left, right) for left in range(len(policy_ids)) for right in range(left, len(policy_ids))}
    observed_keys: dict[tuple[int, int], str] = {}
    duplicate_matchups: list[str] = []
    noncanonical_matchups: list[str] = []
    reference_failures: list[str] = []
    policy_set_check = _validate_final_eval_policy_set(final_eval_dir=final_eval_dir, policy_ids=policy_ids)
    sensitivity_check = _validate_sensitivity_summary(final_eval_dir=final_eval_dir, policy_ids=policy_ids)

    try:
        for index, matchup in enumerate(_matchups(payload)):
            focal_index = _matchup_policy_index(
                matchup,
                index_field="focal_policy_index",
                policy_field="focal_policy_id",
                policy_ids=policy_ids,
                context=f"matchups[{index}]",
            )
            opponent_index = _matchup_policy_index(
                matchup,
                index_field="opponent_policy_index",
                policy_field="opponent_policy_id",
                policy_ids=policy_ids,
                context=f"matchups[{index}]",
            )
            pair_label = f"{policy_ids[focal_index]}__vs__{policy_ids[opponent_index]}"
            if focal_index > opponent_index:
                noncanonical_matchups.append(pair_label)
            key = (min(focal_index, opponent_index), max(focal_index, opponent_index))
            if key in observed_keys:
                duplicate_matchups.append(pair_label)
            else:
                observed_keys[key] = pair_label

            for field_name, expected_kind in (
                ("matchup_dir", "directory"),
                ("episodes_path", "file"),
                ("summary_path", "file"),
                ("diagnostics_path", "file"),
                ("posterior_samples_path", "file"),
            ):
                try:
                    artifact_path = _require_relative_artifact_path(
                        final_eval_dir,
                        value=matchup.get(field_name),
                        field_name=f"matchups[{index}].{field_name}",
                    )
                except ValueError as exc:
                    reference_failures.append(str(exc))
                    continue
                exists = artifact_path.is_dir() if expected_kind == "directory" else artifact_path.is_file()
                if not exists:
                    reference_failures.append(
                        "matchups["
                        f"{index}].{field_name} missing {expected_kind}: "
                        f"{artifact_path.relative_to(final_eval_dir).as_posix()}"
                    )
    except ValueError as exc:
        return {
            "passed": False,
            "summary_path": summary_path.as_posix(),
            "policy_ids": list(policy_ids),
            "expected_matchup_count": len(expected_keys),
            "observed_matchup_count": len(observed_keys),
            "missing_matchups": [],
            "duplicate_matchups": duplicate_matchups,
            "noncanonical_matchups": noncanonical_matchups,
            "reference_failures": [str(exc)],
            "policy_set": policy_set_check,
            "sensitivity_summary": sensitivity_check,
            "reason": "invalid_matchup_index",
            "message": str(exc),
        }

    missing_matchups = [
        f"{policy_ids[left]}__vs__{policy_ids[right]}" for left, right in sorted(expected_keys - set(observed_keys))
    ]

    passed = not duplicate_matchups and not noncanonical_matchups and not missing_matchups and not reference_failures
    passed = passed and bool(policy_set_check["passed"]) and bool(sensitivity_check["passed"])

    return {
        "passed": passed,
        "summary_path": summary_path.as_posix(),
        "policy_ids": list(policy_ids),
        "expected_matchup_count": len(expected_keys),
        "observed_matchup_count": len(observed_keys),
        "missing_matchups": missing_matchups,
        "duplicate_matchups": duplicate_matchups,
        "noncanonical_matchups": noncanonical_matchups,
        "reference_failures": reference_failures,
        "policy_set": policy_set_check,
        "sensitivity_summary": sensitivity_check,
        "message": (
            "final_eval artifact contract is complete"
            if passed
            else "final_eval artifact contract is missing required referenced artifacts"
        ),
    }


def _validate_final_eval_policy_set(*, final_eval_dir: Path, policy_ids: Sequence[str]) -> dict[str, Any]:
    policy_set_path = final_eval_dir / "policy_set.json"
    try:
        payload = _load_json_object(policy_set_path)
    except Exception as exc:
        return {
            "passed": False,
            "policy_set_path": policy_set_path.as_posix(),
            "reason": exc.__class__.__name__,
            "message": str(exc),
        }
    recorded_policy_ids = payload.get("policy_ids")
    if not isinstance(recorded_policy_ids, list) or any(not isinstance(item, str) for item in recorded_policy_ids):
        return {
            "passed": False,
            "policy_set_path": policy_set_path.as_posix(),
            "reason": "invalid_policy_ids",
            "message": "final_eval policy_set.json must include string policy_ids",
        }
    return {
        "passed": list(recorded_policy_ids) == list(policy_ids),
        "policy_set_path": policy_set_path.as_posix(),
        "policy_ids": list(recorded_policy_ids),
        "message": (
            "policy_set.json matches summary policy_ids"
            if list(recorded_policy_ids) == list(policy_ids)
            else "policy_set.json policy_ids do not match summary policy_ids"
        ),
    }


def _validate_sensitivity_summary(*, final_eval_dir: Path, policy_ids: Sequence[str]) -> dict[str, Any]:
    summary_path = resolve_sensitivity_summary_path(final_eval_dir)
    try:
        payload = _load_json_object(summary_path)
    except Exception as exc:
        return {
            "passed": False,
            "summary_path": summary_path.as_posix(),
            "cases": [],
            "reason": exc.__class__.__name__,
            "message": str(exc),
        }

    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, dict):
        return {
            "passed": False,
            "summary_path": summary_path.as_posix(),
            "cases": [],
            "reason": "missing_cases",
            "message": "sensitivity summary must include a cases object",
        }
    cases = sorted(str(case_id) for case_id in raw_cases)
    missing_cases = [case_id for case_id in REQUIRED_SENSITIVITY_CASE_IDS if case_id not in raw_cases]
    payload_policy_ids = payload.get("policy_ids")
    policy_ids_match = isinstance(payload_policy_ids, list) and payload_policy_ids == list(policy_ids)
    passed = not missing_cases and policy_ids_match
    return {
        "passed": passed,
        "summary_path": summary_path.as_posix(),
        "cases": cases,
        "missing_cases": missing_cases,
        "policy_ids_match": policy_ids_match,
        "message": (
            "sensitivity summary covers S0-S2 and matches final_eval policy_ids"
            if passed
            else "sensitivity summary is missing required cases or mismatches final_eval policy_ids"
        ),
    }


def _policy_ids(payload: Mapping[str, Any]) -> list[str]:
    raw_policy_ids = payload.get("policy_ids")
    if not isinstance(raw_policy_ids, list) or any(not isinstance(item, str) for item in raw_policy_ids):
        raise ValueError("final_eval summary must include string policy_ids")
    return list(raw_policy_ids)


def _matchups(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw_matchups = payload.get("matchups")
    if not isinstance(raw_matchups, list):
        raise ValueError("final_eval summary must include matchups")
    matchups: list[Mapping[str, Any]] = []
    for index, matchup in enumerate(raw_matchups):
        matchups.append(_mapping(matchup, context=f"matchups[{index}]"))
    return matchups


def _matchup_policy_index(
    matchup: Mapping[str, Any],
    *,
    index_field: str,
    policy_field: str,
    policy_ids: Sequence[str],
    context: str,
) -> int:
    raw_index = matchup.get(index_field)
    if raw_index is not None:
        index = _as_int(raw_index, context=f"{context}.{index_field}")
        if index < 0 or index >= len(policy_ids):
            raise ValueError(
                f"{context}.{index_field}={index} is out of range for policy_ids with length {len(policy_ids)}"
            )
        return index
    policy_id = matchup.get(policy_field)
    if not isinstance(policy_id, str) or not policy_id.strip():
        raise ValueError(f"{context}.{policy_field} must be a non-empty string")
    try:
        return policy_ids.index(policy_id)
    except ValueError as exc:
        raise ValueError(f"{context}.{policy_field}={policy_id!r} is missing from policy_ids") from exc


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return cast(dict[str, Any], payload)


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _as_int(value: Any, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer")
    return int(value)


def _validate_hex_field(value: Any, *, length: int) -> dict[str, Any]:
    if value is None:
        return {"passed": False, "reason": "missing", "message": "field is missing"}
    if not isinstance(value, str):
        return {"passed": False, "reason": "invalid_type", "message": "field must be a string"}
    normalized = value.strip().lower()
    if len(normalized) != length or any(char not in "0123456789abcdef" for char in normalized):
        return {
            "passed": False,
            "reason": "invalid_value",
            "message": f"field must be a {length}-character hex string",
        }
    return {"passed": True, "reason": None, "message": "ok"}


def _validate_bool_field(value: Any) -> dict[str, Any]:
    if value is None:
        return {"passed": False, "reason": "missing", "message": "field is missing"}
    if not isinstance(value, bool):
        return {"passed": False, "reason": "invalid_type", "message": "field must be a boolean"}
    return {"passed": True, "reason": None, "message": "ok"}


def _validate_object_field(value: Any, *, require_non_empty: bool) -> dict[str, Any]:
    if value is None:
        return {"passed": False, "reason": "missing", "message": "field is missing"}
    if not isinstance(value, dict):
        return {"passed": False, "reason": "invalid_type", "message": "field must be an object"}
    if require_non_empty and not value:
        return {"passed": False, "reason": "empty", "message": "field must not be empty"}
    return {"passed": True, "reason": None, "message": "ok"}


def _validate_string_list_field(value: Any, *, require_non_empty: bool) -> dict[str, Any]:
    if value is None:
        return {"passed": False, "reason": "missing", "message": "field is missing"}
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        return {
            "passed": False,
            "reason": "invalid_type",
            "message": "field must be a list of non-empty strings",
        }
    if require_non_empty and not value:
        return {"passed": False, "reason": "empty", "message": "field must not be empty"}
    return {"passed": True, "reason": None, "message": "ok"}


def _validate_manifest_policy_set_selection(value: Any, *, details: Any) -> dict[str, Any]:
    selection_check = _validate_string_list_field(value, require_non_empty=False)
    if not selection_check["passed"]:
        return selection_check
    if value:
        return {"passed": True, "reason": None, "message": "ok"}
    if _documents_unresolved_policy_set_selection(details):
        return {
            "passed": False,
            "reason": "empty",
            "message": "field is documented as unresolved, but paper-grade readiness requires a resolved final policy set",
        }
    return {
        "passed": False,
        "reason": "empty",
        "message": "field must not be empty for a paper-grade readiness pass",
    }


def _documents_unresolved_policy_set_selection(details: Any) -> bool:
    if not isinstance(details, dict):
        return False
    if details.get("status") != "unresolved":
        return False
    reason = details.get("reason")
    if isinstance(reason, str) and reason.strip():
        return True
    missing_inputs = details.get("missing_inputs")
    return isinstance(missing_inputs, list) and any(isinstance(item, str) and item.strip() for item in missing_inputs)


def _validate_seed_files_field(value: Any) -> dict[str, Any]:
    if value is None:
        return {"passed": False, "reason": "missing", "message": "field is missing"}
    if not isinstance(value, dict) or not value:
        return {
            "passed": False,
            "reason": "invalid_type",
            "message": "seed_files must be a non-empty object",
        }
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            return {
                "passed": False,
                "reason": "invalid_key",
                "message": "seed_files keys must be non-empty strings",
            }
        if not isinstance(item, dict):
            return {
                "passed": False,
                "reason": "invalid_value",
                "message": "seed_files entries must be objects",
            }
        path = item.get("path")
        sha256 = item.get("sha256")
        if not isinstance(path, str) or not path.strip():
            return {
                "passed": False,
                "reason": "invalid_value",
                "message": f"seed_files[{key!r}] must include a non-empty path",
            }
        hash_check = _validate_hex_field(sha256, length=64)
        if not hash_check["passed"]:
            return {
                "passed": False,
                "reason": "invalid_value",
                "message": f"seed_files[{key!r}] must include a 64-character hex sha256",
            }
    return {"passed": True, "reason": None, "message": "ok"}


def _validate_existing_file(path: Path) -> dict[str, Any]:
    if path.is_file():
        return {"passed": True, "reason": None, "message": "ok", "path": path.as_posix()}
    return {
        "passed": False,
        "reason": "missing",
        "message": f"required file is missing: {path}",
        "path": path.as_posix(),
    }


def _validate_simulator_manifest(value: Any) -> dict[str, Any]:
    if value is None:
        return {"passed": False, "reason": "missing", "message": "field is missing"}
    if not isinstance(value, dict) or not value:
        return {
            "passed": False,
            "reason": "invalid_type",
            "message": "simulator must be a non-empty object",
        }
    version = value.get("version")
    if not isinstance(version, str) or not version.strip():
        return {
            "passed": False,
            "reason": "invalid_value",
            "message": "simulator must include a non-empty version",
        }
    build_keys = ("compatibility_hash", "build", "build_id", "build_info", "commit", "sha256")
    if not any(isinstance(value.get(key), str) and str(value.get(key)).strip() for key in build_keys):
        return {
            "passed": False,
            "reason": "invalid_value",
            "message": "simulator must include build/version identity information",
        }
    return {"passed": True, "reason": None, "message": "ok"}


def _compare_json_file_to_manifest(*, file_path: Path, expected: Any) -> dict[str, Any]:
    try:
        payload = _load_json_object(file_path)
    except Exception as exc:
        return {
            "passed": False,
            "file_path": file_path.as_posix(),
            "reason": exc.__class__.__name__,
            "message": str(exc),
        }
    passed = payload == expected
    return {
        "passed": passed,
        "file_path": file_path.as_posix(),
        "message": "JSON file matches manifest" if passed else "JSON file does not match manifest",
    }


def _compare_text_file_to_manifest(*, file_path: Path, expected: Any) -> dict[str, Any]:
    try:
        observed = file_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        return {
            "passed": False,
            "file_path": file_path.as_posix(),
            "reason": exc.__class__.__name__,
            "message": str(exc),
        }
    passed = isinstance(expected, str) and observed == expected
    return {
        "passed": passed,
        "file_path": file_path.as_posix(),
        "observed": observed,
        "message": "text file matches manifest" if passed else "text file does not match manifest",
    }


def _require_relative_artifact_path(root: Path, *, value: Any, field_name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty relative path string")
    raw_path = Path(value)
    if raw_path.is_absolute():
        raise ValueError(f"{field_name} must be relative to {root.as_posix()}")
    resolved_root = root.resolve()
    resolved_path = (root / raw_path).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{field_name} resolves outside {root.as_posix()}: {value}") from exc
    return resolved_path
