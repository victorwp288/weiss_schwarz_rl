"""Compatibility path helpers for legacy config names and locations."""

from __future__ import annotations

from pathlib import Path


def resolve_repo_root(stack_file: Path) -> Path:
    for candidate in stack_file.resolve().parents:
        if (candidate / "configs").is_dir():
            return candidate
    raise FileNotFoundError(f"Could not resolve repo root for config path: {stack_file}")


def resolve_legacy_config_path(stack_file: Path) -> Path:
    if stack_file.exists():
        return stack_file
    sibling_aliases = {
        "thesis_locked.yaml": "typed_thesis_locked.yaml",
        "local.yaml": "typed_local.yaml",
    }
    sibling = sibling_aliases.get(stack_file.name)
    if sibling is not None:
        candidate = stack_file.with_name(sibling)
        if candidate.exists():
            return candidate
    parts = stack_file.parts
    for index in range(len(parts) - 1):
        if parts[index] == "configs" and parts[index + 1] == "presets":
            candidate = Path(*parts[: index + 1], "archive", *parts[index + 1 :])
            if candidate.exists():
                return candidate
    return stack_file


def resolve_repo_path(root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (root / path).resolve()
