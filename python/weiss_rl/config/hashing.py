"""Canonicalization and hashing helpers for stack configs."""

from __future__ import annotations

from dataclasses import asdict, fields
from typing import Any

from weiss_rl.repro import canonical_json_bytes, sha256_hex

from .models import LockedConfig, StackConfig


def canonical_config_dict(stack: StackConfig) -> dict[str, Any]:
    seed_sets = {key: str(path.relative_to(stack.root).as_posix()) for key, path in sorted(stack.seed_sets.items())}
    config = {
        component_name: asdict(component)
        for component_name in (field.name for field in fields(LockedConfig))
        if (component := getattr(stack.config, component_name)) is not None
    }
    payload: dict[str, Any] = {
        "config": config,
        "seed_sets": seed_sets,
    }
    if stack.schema_version is not None:
        payload["schema_version"] = stack.schema_version
    return payload


def canonical_config_bytes(stack: StackConfig) -> bytes:
    return canonical_json_bytes(canonical_config_dict(stack))


def canonical_config_json(stack: StackConfig) -> str:
    return canonical_config_bytes(stack).decode("utf-8")


def compute_config_hash256(stack: StackConfig) -> str:
    return sha256_hex(canonical_config_bytes(stack))
