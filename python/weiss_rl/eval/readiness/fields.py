"""Field-level validators for paper-readiness artifact audits."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast


def load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return cast(dict[str, Any], payload)


def mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object")
    return cast(Mapping[str, Any], value)


def as_int(value: Any, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer")
    return int(value)


def as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("expected numeric matrix cell or null")
    return float(value)


def validate_hex_field(value: Any, *, length: int) -> dict[str, Any]:
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


def validate_bool_field(value: Any) -> dict[str, Any]:
    if value is None:
        return {"passed": False, "reason": "missing", "message": "field is missing"}
    if not isinstance(value, bool):
        return {"passed": False, "reason": "invalid_type", "message": "field must be a boolean"}
    return {"passed": True, "reason": None, "message": "ok"}


def validate_object_field(value: Any, *, require_non_empty: bool) -> dict[str, Any]:
    if value is None:
        return {"passed": False, "reason": "missing", "message": "field is missing"}
    if not isinstance(value, dict):
        return {"passed": False, "reason": "invalid_type", "message": "field must be an object"}
    if require_non_empty and not value:
        return {"passed": False, "reason": "empty", "message": "field must not be empty"}
    return {"passed": True, "reason": None, "message": "ok"}


def validate_string_list_field(value: Any, *, require_non_empty: bool) -> dict[str, Any]:
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


def validate_manifest_policy_set_selection(value: Any, *, details: Any) -> dict[str, Any]:
    selection_check = validate_string_list_field(value, require_non_empty=False)
    if not selection_check["passed"]:
        return selection_check
    if value:
        return {"passed": True, "reason": None, "message": "ok"}
    if documents_unresolved_policy_set_selection(details):
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


def documents_unresolved_policy_set_selection(details: Any) -> bool:
    if not isinstance(details, dict):
        return False
    if details.get("status") != "unresolved":
        return False
    reason = details.get("reason")
    if isinstance(reason, str) and reason.strip():
        return True
    missing_inputs = details.get("missing_inputs")
    return isinstance(missing_inputs, list) and any(isinstance(item, str) and item.strip() for item in missing_inputs)


def validate_seed_files_field(value: Any) -> dict[str, Any]:
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
        hash_check = validate_hex_field(sha256, length=64)
        if not hash_check["passed"]:
            return {
                "passed": False,
                "reason": "invalid_value",
                "message": f"seed_files[{key!r}] must include a 64-character hex sha256",
            }
    return {"passed": True, "reason": None, "message": "ok"}


def validate_existing_file(path: Path) -> dict[str, Any]:
    if path.is_file():
        return {"passed": True, "reason": None, "message": "ok", "path": path.as_posix()}
    return {
        "passed": False,
        "reason": "missing",
        "message": f"required file is missing: {path}",
        "path": path.as_posix(),
    }


def validate_simulator_manifest(value: Any) -> dict[str, Any]:
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


def compare_json_file_to_manifest(*, file_path: Path, expected: Any) -> dict[str, Any]:
    try:
        payload = load_json_object(file_path)
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


def compare_text_file_to_manifest(*, file_path: Path, expected: Any) -> dict[str, Any]:
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


def require_relative_artifact_path(root: Path, *, value: Any, field_name: str) -> Path:
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
