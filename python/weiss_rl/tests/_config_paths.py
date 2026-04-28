from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def canonical_stack_config_path() -> Path:
    return repo_root() / "configs" / "thesis_locked.yaml"


def baseline_stack_config_path(name: str) -> Path:
    return repo_root() / "configs" / "baselines" / name
