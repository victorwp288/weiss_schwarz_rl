"""Config loading utilities for the RL stack."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .spec import SpecMismatchPolicy


@dataclass(slots=True)
class StackConfig:
    """Top-level pointer map loaded from `configs/rl_stack_locked.yaml`."""

    root: Path
    components: dict[str, Path]
    seed_sets: dict[str, Path]
    spec_mismatch_policy: SpecMismatchPolicy = SpecMismatchPolicy.HARD_FAIL
    """Policy for handling spec bundle mismatches (eval always HARD_FAIL)."""


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}, got {type(data).__name__}")
    return data


def _parse_spec_mismatch_policy(value: str | None) -> SpecMismatchPolicy:
    """Parse spec mismatch policy from config string."""
    if value is None or value == "":
        return SpecMismatchPolicy.HARD_FAIL
    token = str(value).strip().lower()
    for policy in SpecMismatchPolicy:
        if policy.value == token:
            return policy
    raise ValueError(
        f"Unknown spec_mismatch_policy: {value}. "
        f"Expected one of: {', '.join(p.value for p in SpecMismatchPolicy)}"
    )


def load_stack_config(stack_path: Path | str) -> StackConfig:
    """Load and normalize the consolidated stack config index."""
    stack_file = Path(stack_path).resolve()
    root = stack_file.parents[1]
    doc = _load_yaml(stack_file)
    body = doc.get("rl_stack_locked", doc)
    if not isinstance(body, dict):
        raise ValueError("Missing `rl_stack_locked` mapping in stack config")

    raw_components = body.get("components", {})
    raw_seed_sets = body.get("seed_sets", {})
    if not isinstance(raw_components, dict) or not isinstance(raw_seed_sets, dict):
        raise ValueError("`components` and `seed_sets` must be mappings")

    components = {k: (root / str(v)).resolve() for k, v in raw_components.items()}
    seed_sets = {k: (root / str(v)).resolve() for k, v in raw_seed_sets.items()}
    
    # Load spec mismatch policy if present
    spec_mismatch_policy_str = body.get("spec_mismatch_policy", "hard_fail")
    spec_mismatch_policy = _parse_spec_mismatch_policy(spec_mismatch_policy_str)
    
    return StackConfig(
        root=root,
        components=components,
        seed_sets=seed_sets,
        spec_mismatch_policy=spec_mismatch_policy,
    )
