"""Config loading utilities for the RL stack."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class StackConfig:
    """Top-level pointer map loaded from `configs/rl_stack_locked.yaml`."""

    root: Path
    components: dict[str, Path]
    seed_sets: dict[str, Path]


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}, got {type(data).__name__}")
    return data


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
    return StackConfig(root=root, components=components, seed_sets=seed_sets)
