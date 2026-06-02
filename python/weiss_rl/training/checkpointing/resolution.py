"""Checkpoint alias and resume-path resolution helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

LATEST_CHECKPOINT_FILENAME = "latest.pt"
BEST_CHECKPOINT_FILENAME = "best.pt"
OBSERVED_BEST_CHECKPOINT_FILENAME = "observed_best.pt"

RESUME_CHECKPOINT_ALIAS_FILENAMES: Mapping[str, str] = MappingProxyType(
    {
        "latest": LATEST_CHECKPOINT_FILENAME,
        "best": BEST_CHECKPOINT_FILENAME,
        "observed_best": OBSERVED_BEST_CHECKPOINT_FILENAME,
    }
)


def normalize_resume_checkpoint_reference(resume_from: str) -> str:
    return str(resume_from).strip()


def resume_checkpoint_alias_path(*, alias_name: str, resume_run_dir: Path) -> Path:
    normalized_alias = normalize_resume_checkpoint_reference(alias_name).lower()
    try:
        filename = RESUME_CHECKPOINT_ALIAS_FILENAMES[normalized_alias]
    except KeyError as exc:
        known_aliases = ", ".join(RESUME_CHECKPOINT_ALIAS_FILENAMES)
        raise ValueError(f"unknown resume checkpoint alias {alias_name!r}; expected one of: {known_aliases}") from exc
    return Path(resume_run_dir).resolve() / "training" / "checkpoints" / filename


def resolve_resume_checkpoint_path(
    *,
    resume_from: str,
    resume_run_dir: Path | None,
) -> Path | None:
    normalized = normalize_resume_checkpoint_reference(resume_from)
    if not normalized:
        if resume_run_dir is None:
            return None
        normalized = "latest"

    alias_name = normalized.lower()
    if alias_name in RESUME_CHECKPOINT_ALIAS_FILENAMES:
        if resume_run_dir is None:
            raise ValueError("--resume-from latest|best|observed_best requires --resume-run-dir")
        checkpoint_path = resume_checkpoint_alias_path(alias_name=alias_name, resume_run_dir=resume_run_dir)
    else:
        checkpoint_path = Path(normalized).resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
    return checkpoint_path


__all__ = [
    "BEST_CHECKPOINT_FILENAME",
    "LATEST_CHECKPOINT_FILENAME",
    "OBSERVED_BEST_CHECKPOINT_FILENAME",
    "RESUME_CHECKPOINT_ALIAS_FILENAMES",
    "normalize_resume_checkpoint_reference",
    "resolve_resume_checkpoint_path",
    "resume_checkpoint_alias_path",
]
