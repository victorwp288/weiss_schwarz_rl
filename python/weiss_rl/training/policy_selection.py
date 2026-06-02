from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from weiss_rl.eval import DevEvalPolicySummary
from weiss_rl.eval.policies.set import select_final_policy_set_deterministic_v1
from weiss_rl.league.registry import SnapshotRegistry
from weiss_rl.training.run_metadata import load_json_object, manifest_source_path


def load_snapshot_registry(path: Path) -> SnapshotRegistry:
    if not path.exists():
        raise FileNotFoundError(path)
    return SnapshotRegistry.load(path)


def load_dev_eval_summaries(path: Path) -> dict[str, float | DevEvalPolicySummary]:
    payload = load_json_object(path, label="dev-eval summaries")
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


def selection_requires_snapshot_registry(stack: Any) -> bool:
    evaluation = stack.config.evaluation
    if evaluation is None:
        return False
    selection = evaluation.final_policy_set_selection
    return selection.include_final_champion_snapshot or bool(selection.include_spaced_snapshots_near_percent_updates)


def selection_requires_dev_eval_summaries(stack: Any) -> bool:
    evaluation = stack.config.evaluation
    if evaluation is None:
        return False
    selection = evaluation.final_policy_set_selection
    fixed_slots = int(selection.include_random_legal_baseline_b0) + int(selection.include_no_league_baseline_b1)
    fixed_slots += int(selection.include_final_champion_snapshot)
    fixed_slots += len(selection.include_spaced_snapshots_near_percent_updates)
    if selection.include_heuristic_public_b2_if_exists:
        return True
    return evaluation.final_policy_set_size > fixed_slots


def policy_set_selection(
    stack: Any,
    *,
    snapshot_registry: SnapshotRegistry | None = None,
    dev_eval_summaries: Mapping[str, float | DevEvalPolicySummary] | None = None,
) -> list[str]:
    evaluation = stack.config.evaluation
    if evaluation is None:
        return []
    selection = evaluation.final_policy_set_selection
    if selection.version != "deterministic_v1":
        raise ValueError(f"unsupported final_policy_set_selection.version: {selection.version!r}")
    return select_final_policy_set_deterministic_v1(
        snapshot_registry=snapshot_registry or SnapshotRegistry(),
        dev_eval_summaries=dev_eval_summaries or {},
        config=selection,
        final_policy_set_size=evaluation.final_policy_set_size,
    )


def resolve_policy_set_selection(
    stack: Any,
    *,
    snapshot_registry_path: Path | None = None,
    dev_eval_summaries_path: Path | None = None,
) -> tuple[list[str], dict[str, Any]]:
    evaluation = stack.config.evaluation
    source_paths = {
        "snapshot_registry_json": None
        if snapshot_registry_path is None
        else manifest_source_path(snapshot_registry_path, root=stack.root),
        "dev_eval_summaries_json": None
        if dev_eval_summaries_path is None
        else manifest_source_path(dev_eval_summaries_path, root=stack.root),
    }
    if evaluation is None:
        return [], {"mode": "not_configured", "status": "not_configured", "source_paths": source_paths}

    snapshot_registry = None if snapshot_registry_path is None else load_snapshot_registry(snapshot_registry_path)
    dev_eval_summaries = None if dev_eval_summaries_path is None else load_dev_eval_summaries(dev_eval_summaries_path)

    missing_inputs: list[str] = []
    if selection_requires_snapshot_registry(stack) and snapshot_registry is None:
        missing_inputs.append("snapshot_registry_json")
    if selection_requires_dev_eval_summaries(stack) and dev_eval_summaries is None:
        missing_inputs.append("dev_eval_summaries_json")

    details: dict[str, Any] = {
        "mode": evaluation.final_policy_set_selection.version,
        "status": "resolved",
        "version": evaluation.final_policy_set_selection.version,
        "final_policy_set_size": evaluation.final_policy_set_size,
        "source_paths": source_paths,
        "missing_inputs": missing_inputs,
    }
    if missing_inputs:
        details["mode"] = "unresolved"
        details["status"] = "unresolved"
        details["reason"] = "deterministic final policy set inputs were not provided"
        return [], details

    policy_ids = policy_set_selection(
        stack,
        snapshot_registry=snapshot_registry,
        dev_eval_summaries=dev_eval_summaries,
    )
    details["selected_policy_count"] = len(policy_ids)
    return policy_ids, details
