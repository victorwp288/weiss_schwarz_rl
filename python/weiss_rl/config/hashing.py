"""Canonicalization and hashing helpers for grouped preset configs."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict
from typing import Any

from weiss_rl.artifacts.reproducibility import canonical_json_bytes, sha256_hex

from .models import StackConfig


def _json_normalize(value: Any) -> Any:
    return json.loads(canonical_json_bytes(value))


def canonical_config_dict(stack: StackConfig) -> dict[str, Any]:
    preserved_payload = stack.lock_intent.get("canonical_config_payload")
    if isinstance(preserved_payload, dict):
        return _json_normalize(deepcopy(preserved_payload))
    seed_sets = {key: str(path.relative_to(stack.root).as_posix()) for key, path in sorted(stack.seed_sets.items())}
    config = {key: value for key, value in asdict(stack.config).items() if value is not None}
    payload: dict[str, Any] = {
        "config": config,
        "seed_sets": seed_sets,
    }
    if stack.schema_version is not None:
        payload["schema_version"] = stack.schema_version
    return _json_normalize(payload)


def canonical_config_bytes(stack: StackConfig) -> bytes:
    return canonical_json_bytes(canonical_config_dict(stack))


def canonical_config_json(stack: StackConfig) -> str:
    return canonical_config_bytes(stack).decode("utf-8")


def compute_config_hash256(stack: StackConfig) -> str:
    return sha256_hex(canonical_config_bytes(stack))
