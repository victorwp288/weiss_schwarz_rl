"""Small validation primitives shared by config parsers."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any


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


def require_auto_or_int(value: Any, *, field_name: str, minimum: int = 0) -> str:
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "auto":
            return text
        if text.isdecimal():
            parsed = int(text)
        else:
            raise ValueError(f"{field_name} must be 'auto' or an integer >= {minimum}")
    elif isinstance(value, int) and not isinstance(value, bool):
        parsed = int(value)
    else:
        raise ValueError(f"{field_name} must be 'auto' or an integer >= {minimum}")
    if parsed < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}, got {parsed}")
    return str(parsed)


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
