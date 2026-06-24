"""Deterministic final policy-set selection routines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from weiss_rl.config.models import FinalPolicySetSelectionConfig
from weiss_rl.eval.policies.dev_eval_summaries import (
    DevEvalPolicySummary,
    DevEvalSummaryLike,
    canonicalize_dev_eval_summaries,
    normalize_dev_eval_summaries,
)
from weiss_rl.eval.policies.fixed_panel import (
    AGGRO_DECK_ID,
    CONTROL_DECK_ID,
    FIXED_POLICY_PANEL_ROLES,
    HEURISTIC_PUBLIC_AGGRO_POLICY_ID,
    HEURISTIC_PUBLIC_CONTROL_POLICY_ID,
    HEURISTIC_PUBLIC_POLICY_ID,
    LEGACY_NO_LEAGUE_POLICY_ID,
    MAIN_DECK_ID,
    NO_LEAGUE_POLICY_ID,
    RANDOM_LEGAL_POLICY_ID,
    STARTER_DECK_ID,
    deck_id_for_policy_id,
    fixed_policy_panel_role_payload,
    heuristic_public_policy_ids,
    heuristic_public_profile_name_for_policy_id,
)
from weiss_rl.eval.policies.focal_recommendation import recommend_focal_policy_id
from weiss_rl.eval.policies.registry_view import champion_snapshot_ids, snapshot_training_policies
from weiss_rl.eval.policies.training_policy_ids import (
    TrainingPolicyId,
    parse_training_policy_id,
    training_policy_sort_key,
)


@dataclass(frozen=True, slots=True)
class FinalPolicySetSelectionTraceEntry:
    policy_id: str
    reason: str
    details: Mapping[str, object]

    def as_payload(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "reason": self.reason,
            "details": dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class FinalPolicySetSelectionResult:
    policy_ids: list[str]
    trace: tuple[FinalPolicySetSelectionTraceEntry, ...]

    def trace_payload(self) -> list[dict[str, object]]:
        return [entry.as_payload() for entry in self.trace]


def select_final_policy_set_deterministic_v1(
    snapshot_registry: object,
    dev_eval_summaries: Mapping[str, DevEvalSummaryLike],
    config: FinalPolicySetSelectionConfig,
    final_policy_set_size: int,
) -> list[str]:
    """Select the final policy set deterministically from snapshots and dev-eval summaries."""
    return select_final_policy_set_deterministic_v1_with_trace(
        snapshot_registry=snapshot_registry,
        dev_eval_summaries=dev_eval_summaries,
        config=config,
        final_policy_set_size=final_policy_set_size,
    ).policy_ids


def select_final_policy_set_deterministic_v1_with_trace(
    snapshot_registry: object,
    dev_eval_summaries: Mapping[str, DevEvalSummaryLike],
    config: FinalPolicySetSelectionConfig,
    final_policy_set_size: int,
) -> FinalPolicySetSelectionResult:
    """Select the final policy set and explain why each policy was included."""
    if final_policy_set_size < 1:
        raise ValueError("final_policy_set_size must be at least 1")

    snapshot_policies = snapshot_training_policies(snapshot_registry)
    normalized_summaries = canonicalize_dev_eval_summaries(
        normalize_dev_eval_summaries(dev_eval_summaries),
        snapshot_policies=snapshot_policies,
    )
    snapshot_policies_by_id = {policy.policy_id: policy for policy in snapshot_policies}
    selected: list[str] = []
    trace: list[FinalPolicySetSelectionTraceEntry] = []

    if config.include_random_legal_baseline_b0:
        _append_unique_with_trace(
            selected,
            trace,
            RANDOM_LEGAL_POLICY_ID,
            reason="random_legal_baseline_b0",
            details={"source": "fixed_baseline"},
        )
    if config.include_no_league_baseline_b1:
        _append_unique_with_trace(
            selected,
            trace,
            NO_LEAGUE_POLICY_ID,
            reason="no_league_baseline_b1",
            details={"source": "fixed_baseline"},
        )
    if config.include_heuristic_public_b2_if_exists and HEURISTIC_PUBLIC_POLICY_ID in normalized_summaries:
        _append_unique_with_trace(
            selected,
            trace,
            HEURISTIC_PUBLIC_POLICY_ID,
            reason="heuristic_public_b2_available",
            details={"source": "dev_eval_summaries"},
        )
    if config.include_heuristic_public_anchors_b2_b3_b4:
        for policy_id in heuristic_public_policy_ids(include_base=True):
            _append_unique_with_trace(
                selected,
                trace,
                policy_id,
                reason="heuristic_public_anchor",
                details={"profile": heuristic_public_profile_name_for_policy_id(policy_id)},
            )

    if config.include_final_champion_snapshot:
        latest_champion = _latest_champion_policy(snapshot_registry, snapshot_policies_by_id=snapshot_policies_by_id)
        if latest_champion is not None:
            _append_unique_with_trace(
                selected,
                trace,
                latest_champion.policy_id,
                reason="latest_champion_snapshot",
                details={"update": latest_champion.update, "version": latest_champion.version},
            )

    if snapshot_policies:
        latest_snapshot = max(snapshot_policies, key=training_policy_sort_key)
        for percent in config.include_spaced_snapshots_near_percent_updates:
            target_update = int(latest_snapshot.update * percent / 100)
            closest_snapshot = _find_closest_snapshot(snapshot_policies, target_update)
            _append_unique_with_trace(
                selected,
                trace,
                closest_snapshot.policy_id,
                reason="spaced_snapshot_near_percent_update",
                details={
                    "percent": int(percent),
                    "target_update": target_update,
                    "selected_update": closest_snapshot.update,
                    "version": closest_snapshot.version,
                },
            )

    remaining_slots = final_policy_set_size - len(selected)
    if remaining_slots <= 0:
        truncated = selected[:final_policy_set_size]
        return FinalPolicySetSelectionResult(
            policy_ids=truncated,
            trace=tuple(entry for entry in trace if entry.policy_id in set(truncated)),
        )

    anchor_policy_ids = _configured_anchor_policy_ids(config, normalized_summaries)
    ranked_candidates = _rank_remaining_candidates(
        normalized_summaries,
        anchor_policy_ids=anchor_policy_ids,
        selected_policy_ids=set(selected),
        strategy=config.remaining_slots_strategy,
        tie_break=config.tie_break,
    )
    for policy_id, score in ranked_candidates[:remaining_slots]:
        _append_unique_with_trace(
            selected,
            trace,
            policy_id,
            reason="top_dev_performer_vs_anchor_set",
            details={"mean_anchor_score": score, "anchor_policy_ids": list(anchor_policy_ids)},
        )
    return FinalPolicySetSelectionResult(policy_ids=selected, trace=tuple(trace))


def _append_unique(selected: list[str], policy_id: str) -> None:
    if policy_id not in selected:
        selected.append(policy_id)


def _append_unique_with_trace(
    selected: list[str],
    trace: list[FinalPolicySetSelectionTraceEntry],
    policy_id: str,
    *,
    reason: str,
    details: Mapping[str, object],
) -> None:
    if policy_id in selected:
        return
    selected.append(policy_id)
    trace.append(FinalPolicySetSelectionTraceEntry(policy_id=policy_id, reason=reason, details=details))


def _configured_anchor_policy_ids(
    config: FinalPolicySetSelectionConfig,
    dev_eval_summaries: Mapping[str, DevEvalPolicySummary],
) -> tuple[str, ...]:
    anchor_policy_ids = list(config.fixed_anchor_set_v1.required)
    anchor_policy_ids.extend(
        policy_id for policy_id in config.fixed_anchor_set_v1.optional_if_available if policy_id in dev_eval_summaries
    )
    return tuple(anchor_policy_ids)


def _find_closest_snapshot(
    parsed_snapshots: Sequence[TrainingPolicyId],
    target_update: int,
) -> TrainingPolicyId:
    return min(
        parsed_snapshots,
        key=lambda parsed: (
            abs(parsed.update - target_update),
            parsed.update,
            -parsed.version,
            parsed.policy_id,
        ),
    )


def _latest_champion_policy(
    snapshot_registry: object,
    *,
    snapshot_policies_by_id: Mapping[str, TrainingPolicyId],
) -> TrainingPolicyId | None:
    champion_policies = [
        snapshot_policies_by_id[policy_id]
        for policy_id in champion_snapshot_ids(snapshot_registry)
        if policy_id in snapshot_policies_by_id
    ]
    if not champion_policies:
        return None
    return max(champion_policies, key=training_policy_sort_key)


def _policy_tie_break_key(policy_id: str, *, tie_break: str) -> str:
    if tie_break == "lowest_policy_id":
        return policy_id
    raise ValueError(f"unsupported final-policy-set tie_break: {tie_break!r}")


def _rank_remaining_candidates(
    dev_eval_summaries: Mapping[str, DevEvalPolicySummary],
    *,
    anchor_policy_ids: Sequence[str],
    selected_policy_ids: set[str],
    strategy: str,
    tie_break: str,
) -> list[tuple[str, float]]:
    if strategy != "top_dev_performers_vs_anchor_set_v1":
        raise ValueError(f"unsupported final-policy-set remaining_slots_strategy: {strategy!r}")

    ranked: list[tuple[str, float]] = []
    excluded_policy_ids = selected_policy_ids | set(anchor_policy_ids)
    for policy_id, summary in dev_eval_summaries.items():
        if policy_id in excluded_policy_ids:
            continue
        ranked.append((policy_id, summary.mean_anchor_score(anchor_policy_ids)))

    ranked.sort(key=lambda item: (-item[1], _policy_tie_break_key(item[0], tie_break=tie_break)))
    return ranked


__all__ = [
    "AGGRO_DECK_ID",
    "CONTROL_DECK_ID",
    "DevEvalPolicySummary",
    "FIXED_POLICY_PANEL_ROLES",
    "FinalPolicySetSelectionResult",
    "FinalPolicySetSelectionTraceEntry",
    "HEURISTIC_PUBLIC_AGGRO_POLICY_ID",
    "HEURISTIC_PUBLIC_CONTROL_POLICY_ID",
    "HEURISTIC_PUBLIC_POLICY_ID",
    "LEGACY_NO_LEAGUE_POLICY_ID",
    "MAIN_DECK_ID",
    "NO_LEAGUE_POLICY_ID",
    "RANDOM_LEGAL_POLICY_ID",
    "STARTER_DECK_ID",
    "TrainingPolicyId",
    "deck_id_for_policy_id",
    "fixed_policy_panel_role_payload",
    "heuristic_public_policy_ids",
    "heuristic_public_profile_name_for_policy_id",
    "parse_training_policy_id",
    "recommend_focal_policy_id",
    "select_final_policy_set_deterministic_v1",
    "select_final_policy_set_deterministic_v1_with_trace",
]
