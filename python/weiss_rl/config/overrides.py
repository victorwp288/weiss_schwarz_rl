"""Deterministic stack-config overrides for experiment sweeps."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from typing import Any

from .models import LockedConfig, StackConfig


def parse_override_tokens(tokens: list[str] | None) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for token in tokens or ():
        text = str(token).strip()
        if not text:
            continue
        if "=" not in text:
            raise ValueError(f"config override must be KEY=VALUE, got {token!r}")
        key, raw_value = text.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"config override key must be non-empty, got {token!r}")
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        overrides[key] = value
    return overrides


def apply_stack_overrides(stack: StackConfig, overrides: dict[str, Any]) -> StackConfig:
    if not overrides:
        return stack
    config = stack.config
    for path, value in overrides.items():
        config = _apply_locked_config_override(config, path=path, value=value)
    lock_intent = dict(stack.lock_intent)
    lock_intent.pop("canonical_config_payload", None)
    return replace(stack, config=config, lock_intent=lock_intent)


def _apply_locked_config_override(config: LockedConfig, *, path: str, value: Any) -> LockedConfig:
    segments = [segment.strip() for segment in str(path).split(".") if segment.strip()]
    if len(segments) < 2:
        raise ValueError(f"config override path must include section and field, got {path!r}")
    component_name = segments[0]
    component = getattr(config, component_name, None)
    if component is None:
        raise ValueError(f"config override references unknown section {component_name!r}")
    updated_component = _apply_path(component, segments[1:], value=value)
    return replace(config, **{component_name: updated_component})


def _apply_path(target: Any, path_segments: list[str], *, value: Any) -> Any:
    if not path_segments:
        raise ValueError("override path cannot terminate at an empty segment list")
    if is_dataclass(target):
        return _apply_dataclass_path(target, path_segments, value=value)
    if isinstance(target, Mapping):
        return _apply_mapping_path(target, path_segments, value=value)
    raise ValueError(f"cannot descend into non-config target at {'.'.join(path_segments)!r}")


def _apply_dataclass_path(target: Any, path_segments: list[str], *, value: Any) -> Any:
    field_map = {field.name: field for field in fields(target)}
    field_name = path_segments[0]
    if field_name not in field_map:
        raise ValueError(f"override references unknown field {field_name!r} on {type(target).__name__}")
    current_value = getattr(target, field_name)
    if len(path_segments) == 1:
        return replace(target, **{field_name: value})
    updated_value = _apply_path(current_value, path_segments[1:], value=value)
    return replace(target, **{field_name: updated_value})


def _apply_mapping_path(target: Mapping[str, Any], path_segments: list[str], *, value: Any) -> dict[str, Any]:
    key = path_segments[0]
    if key not in target:
        raise ValueError(f"override references unknown mapping key {key!r}")
    if len(path_segments) == 1:
        updated = dict(target)
        updated[key] = value
        return updated
    updated = dict(target)
    updated[key] = _apply_path(updated[key], path_segments[1:], value=value)
    return updated
