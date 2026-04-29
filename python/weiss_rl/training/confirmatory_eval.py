"""Confirmatory dev-eval planning helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.config import StackConfig
from weiss_rl.training.dev_eval_metrics import confirmatory_dev_eval_request
from weiss_rl.training.eval_schedule import periodic_dev_eval_schedule
from weiss_rl.training.eval_seeds import expand_periodic_dev_eval_paired_seeds


@dataclass(frozen=True, slots=True)
class ConfirmatoryDevEvalPlan:
    seed_file: Path
    paired_seeds: tuple[int, ...]
    reasons: tuple[str, ...]
    target_pairs: int


def build_confirmatory_dev_eval_plan(
    *,
    stack: StackConfig,
    existing_best_record: Mapping[str, Any] | None,
    dev_eval_summary: Mapping[str, Any],
) -> ConfirmatoryDevEvalPlan | None:
    request = confirmatory_dev_eval_request(
        stack=stack,
        existing_best_record=existing_best_record,
        dev_eval_summary=dev_eval_summary,
    )
    if request is None:
        return None

    seed_file, _validated_sources, base_paired_seeds, seed_file_sha256 = periodic_dev_eval_schedule(stack)
    target_pairs = int(request["target_pairs"])
    paired_seeds = expand_periodic_dev_eval_paired_seeds(
        base_paired_seeds,
        requested_pairs=target_pairs,
        seed_file_sha256=seed_file_sha256,
        update_count=int(dev_eval_summary["update_count"]),
        policy_version=int(dev_eval_summary["policy_version"]),
        scope="periodic_dev_eval_confirmatory",
    )
    return ConfirmatoryDevEvalPlan(
        seed_file=seed_file,
        paired_seeds=tuple(paired_seeds),
        reasons=tuple(str(reason) for reason in request["reasons"]),
        target_pairs=target_pairs,
    )


def format_confirmatory_dev_eval_message(
    *,
    update_count: int,
    paired_seed_count: int,
    aggregate_score: float,
    reasons: tuple[str, ...],
    seed_file: Path,
) -> str:
    return (
        "Confirmatory dev eval: "
        f"update={int(update_count)} paired_seeds={int(paired_seed_count)} "
        f"aggregate={float(aggregate_score):.4f} "
        f"reasons={','.join(reasons)} "
        f"seed_file={seed_file.name}"
    )
