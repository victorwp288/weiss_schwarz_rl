from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CanonicalEvalSeedBudget:
    paired_seeds: list[Any]
    paired_seed_limit: int | None
    stage1_paired_seeds: int
    max_paired_seeds: int
    seed_file_path: Path


def resolve_canonical_eval_seed_budget(
    *,
    stack: Any,
    evaluation: Any,
    paired_seed_limit: int | None,
    stage1_paired_seeds: int | None,
    max_paired_seeds: int | None,
    dependencies: Any,
) -> CanonicalEvalSeedBudget:
    seed_file_path = stack.seed_sets["report_eval"]
    all_paired_seeds = dependencies.parse_seed_file_fn(seed_file_path)
    if paired_seed_limit is not None:
        all_paired_seeds = all_paired_seeds[: int(paired_seed_limit)]
    if not all_paired_seeds:
        raise ValueError(f"report_eval seed file produced no usable seeds: {seed_file_path}")

    resolved_stage1 = int(
        stage1_paired_seeds or min(evaluation.final_matrix_stage1_paired_seeds, len(all_paired_seeds))
    )
    resolved_max = int(
        max_paired_seeds or min(evaluation.final_matrix_stage2_adaptive_max_paired_seeds, len(all_paired_seeds))
    )
    if resolved_stage1 > resolved_max:
        raise ValueError(f"stage1 paired seeds ({resolved_stage1}) cannot exceed max paired seeds ({resolved_max})")

    return CanonicalEvalSeedBudget(
        paired_seeds=list(all_paired_seeds),
        paired_seed_limit=paired_seed_limit,
        stage1_paired_seeds=resolved_stage1,
        max_paired_seeds=resolved_max,
        seed_file_path=seed_file_path,
    )


__all__ = ["CanonicalEvalSeedBudget", "resolve_canonical_eval_seed_budget"]
