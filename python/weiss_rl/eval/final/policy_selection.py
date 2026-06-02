from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from weiss_rl.config.models import FinalPolicySetSelectionConfig
from weiss_rl.eval.policies.set import (
    DevEvalPolicySummary,
    select_final_policy_set_deterministic_v1,
)


def load_dev_eval_summaries(path: Path) -> dict[str, float | DevEvalPolicySummary]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("dev-eval summaries JSON must contain an object at the top level")

    summaries: dict[str, float | DevEvalPolicySummary] = {}
    for policy_id, raw_summary in payload.items():
        if isinstance(raw_summary, bool):
            raise TypeError(f"dev-eval summary for {policy_id!r} cannot be a boolean")
        if isinstance(raw_summary, (int, float)):
            summaries[policy_id] = float(raw_summary)
            continue
        if not isinstance(raw_summary, dict):
            raise TypeError(
                "dev-eval summary values must be numbers or objects with aggregate_score/anchor_scores, "
                f"got {type(raw_summary).__name__} for {policy_id!r}"
            )
        aggregate_score = raw_summary.get("aggregate_score")
        if isinstance(aggregate_score, bool) or not isinstance(aggregate_score, (int, float)):
            raise TypeError(f"dev-eval summary for {policy_id!r} must include numeric aggregate_score")
        anchor_scores = raw_summary.get("anchor_scores", {})
        if not isinstance(anchor_scores, dict) or any(not isinstance(key, str) for key in anchor_scores):
            raise TypeError(f"dev-eval summary for {policy_id!r} must include object anchor_scores")
        summaries[policy_id] = DevEvalPolicySummary(
            policy_id=policy_id,
            aggregate_score=float(aggregate_score),
            anchor_scores=anchor_scores,
        )
    return summaries


def resolve_final_policy_set(
    *,
    snapshot_registry_path: Path,
    dev_eval_summaries_path: Path,
    config: FinalPolicySetSelectionConfig,
    final_policy_set_size: int,
) -> list[str]:
    from weiss_rl.league.registry import SnapshotRegistry

    registry = SnapshotRegistry.load(snapshot_registry_path)
    summaries = load_dev_eval_summaries(dev_eval_summaries_path)
    return select_final_policy_set_deterministic_v1(
        snapshot_registry=registry,
        dev_eval_summaries=summaries,
        config=config,
        final_policy_set_size=final_policy_set_size,
    )


def resolve_final_eval_policy_ids(
    *,
    policy_ids: Sequence[str] | None,
    snapshot_registry_path: Path | None,
    dev_eval_summaries_path: Path | None,
    selection_config: FinalPolicySetSelectionConfig | None,
    final_policy_set_size: int | None,
) -> tuple[list[str], dict[str, Any]]:
    if policy_ids is not None:
        resolved = [str(policy_id) for policy_id in policy_ids]
        if not resolved:
            raise ValueError("policy_ids must contain at least one policy")
        validate_policy_ids(resolved, context="policy_ids")
        return resolved, {"mode": "explicit", "policy_count": len(resolved)}

    missing: list[str] = []
    if snapshot_registry_path is None:
        missing.append("snapshot_registry_path")
    if dev_eval_summaries_path is None:
        missing.append("dev_eval_summaries_path")
    if selection_config is None:
        missing.append("selection_config")
    if final_policy_set_size is None:
        missing.append("final_policy_set_size")
    if missing:
        raise ValueError(f"run_final_eval requires policy_ids or selection inputs, missing: {', '.join(missing)}")

    assert snapshot_registry_path is not None
    assert dev_eval_summaries_path is not None
    assert selection_config is not None
    assert final_policy_set_size is not None
    resolved = resolve_final_policy_set(
        snapshot_registry_path=snapshot_registry_path,
        dev_eval_summaries_path=dev_eval_summaries_path,
        config=selection_config,
        final_policy_set_size=final_policy_set_size,
    )
    validate_policy_ids(resolved, context="resolved final policy set")
    if len(resolved) < int(final_policy_set_size):
        raise ValueError(
            "resolved final policy set is underfilled: "
            f"expected {int(final_policy_set_size)} policies, found {len(resolved)}"
        )
    return resolved, {
        "mode": "deterministic_v1",
        "policy_count": len(resolved),
        "snapshot_registry_path": snapshot_registry_path.as_posix(),
        "dev_eval_summaries_path": dev_eval_summaries_path.as_posix(),
        "final_policy_set_size": int(final_policy_set_size),
    }


def validate_policy_ids(policy_ids: Sequence[str], *, context: str) -> None:
    duplicates = sorted(policy_id for policy_id, count in Counter(policy_ids).items() if count > 1)
    if duplicates:
        duplicate_list = ", ".join(repr(policy_id) for policy_id in duplicates)
        raise ValueError(f"{context} must be unique, duplicate entries: {duplicate_list}")


def validate_final_eval_seed_budget(
    *,
    paired_seeds: Sequence[int],
    stage1_paired_seeds: int,
    max_paired_seeds: int,
) -> None:
    if stage1_paired_seeds < 1:
        raise ValueError("stage1_paired_seeds must be positive")
    if max_paired_seeds < stage1_paired_seeds:
        raise ValueError("max_paired_seeds must be >= stage1_paired_seeds")
    if len(paired_seeds) < max_paired_seeds:
        raise ValueError(f"final eval requires at least {max_paired_seeds} paired seeds, found {len(paired_seeds)}")


__all__ = [
    "load_dev_eval_summaries",
    "resolve_final_eval_policy_ids",
    "resolve_final_policy_set",
    "validate_final_eval_seed_budget",
    "validate_policy_ids",
]
