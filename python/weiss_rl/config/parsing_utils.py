"""Shared low-level helpers for strict config document parsing."""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

import yaml

PRESET_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "extends",
        "description",
        "experiment",
        "system",
        "model",
        "training",
        "environment",
        "rewards",
        "curriculum",
        "league",
        "evaluation",
        "reproducibility",
    }
)


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}, got {type(data).__name__}")
    return data


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}, got {type(data).__name__}")
    return data


def resolve_repo_root(stack_file: Path) -> Path:
    for candidate in stack_file.resolve().parents:
        if (candidate / "configs").is_dir():
            return candidate
    raise FileNotFoundError(f"Could not resolve repo root for config path: {stack_file}")


def require_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping, got {type(value).__name__}")
    return dict(value)


def require_int(value: Any, *, field_name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer, got {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}, got {value}")
    return value


def require_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric, got {type(value).__name__}")
    return float(value)


def require_bool(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean, got {type(value).__name__}")
    return value


def require_text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def require_choice(value: Any, *, field_name: str, allowed: Collection[str]) -> str:
    text = require_text(value, field_name=field_name)
    if text not in allowed:
        allowed_values = ", ".join(sorted(allowed))
        raise ValueError(f"{field_name} must be one of: {allowed_values}")
    return text


def require_str_list(value: Any, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    return tuple(require_text(item, field_name=f"{field_name}[]") for item in value)


def require_int_list(value: Any, *, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    return tuple(require_int(item, field_name=f"{field_name}[]", minimum=0) for item in value)


def reject_unknown_keys(body: Mapping[str, Any], *, allowed: Collection[str], context: str) -> None:
    unknown = sorted(key for key in body if key not in allowed)
    if unknown:
        raise ValueError(f"{context} has unsupported keys: {', '.join(unknown)}")


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if key == "extends":
            continue
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(dict(merged[key]), dict(value))
        else:
            merged[key] = value
    return merged


def load_preset_document(path: Path, *, seen: set[Path] | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    active = set() if seen is None else seen
    if resolved in active:
        raise ValueError(f"Config extends cycle detected at {resolved}")
    active.add(resolved)
    doc = load_yaml(resolved)
    reject_unknown_keys(doc, allowed=PRESET_TOP_LEVEL_KEYS, context=str(resolved))
    merged: dict[str, Any] = {}
    parent = doc.get("extends")
    if parent is not None:
        parent_ref = require_text(parent, field_name=f"{resolved}.extends")
        merged = load_preset_document((resolved.parent / parent_ref).resolve(), seen=active)
    active.remove(resolved)
    return deep_merge(merged, doc)


def resolve_repo_path(root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (root / path).resolve()
