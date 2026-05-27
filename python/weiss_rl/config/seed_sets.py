"""Seed-set resolution helpers for stack configs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .models import EvaluationConfig, LeagueConfig, ReproducibilityConfig
from .parsing_utils import require_text, resolve_repo_path


def resolve_seed_sets(
    *,
    root: Path,
    league: LeagueConfig | None,
    evaluation: EvaluationConfig | None,
    reproducibility: ReproducibilityConfig | None,
) -> dict[str, Path]:
    seed_sets: dict[str, Path] = {}
    if evaluation is not None:
        for key, path in evaluation.seed_files.items():
            seed_sets[key] = resolve_repo_path(root, path)
    if league is not None and league.promotion.seed_file.strip():
        seed_sets.setdefault("promotion_gate", resolve_repo_path(root, league.promotion.seed_file))
    if reproducibility is not None:
        for key, path in reproducibility.seed_files.items():
            seed_sets.setdefault(key, resolve_repo_path(root, path))
    return seed_sets


def parse_seed_sets_override(*, root: Path, seed_sets_doc: Mapping[str, Any]) -> dict[str, Path]:
    return {
        require_text(key, field_name="seed_sets.<key>"): resolve_repo_path(
            root,
            require_text(value, field_name=f"seed_sets.{key}"),
        )
        for key, value in seed_sets_doc.items()
    }
