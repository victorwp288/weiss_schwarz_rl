"""Deterministic final policy-set selection routines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import re
from typing import Protocol, cast

from weiss_rl.config.models import FinalPolicySetSelectionConfig

RANDOM_LEGAL_POLICY_ID = "B0 RandomLegal"
NO_LEAGUE_POLICY_ID = "B1 NoLeague baseline"
HEURISTIC_PUBLIC_POLICY_ID = "B2 HeuristicPublic"

_TRAINING_POLICY_ID_RE = re.compile(r"^train_u(?P<update>\d+)_p(?P<version>\d+)$")
_POLICY_VERSION_ID_RE = re.compile(r"^policy_(?P<version>\d+)$")


@dataclass(frozen=True, slots=True)
class TrainingPolicyId:
    policy_id: str
    update: int
    version: int


@dataclass(frozen=True, slots=True)
class DevEvalPolicySummary:
    policy_id: str
    aggregate_score: float
    anchor_scores: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "anchor_scores", {key: float(value) for key, value in self.anchor_scores.items()})

    def mean_anchor_score(self, anchor_policy_ids: Sequence[str]) -> float:
        if not anchor_policy_ids:
            return self.aggregate_score
        missing = [policy_id for policy_id in anchor_policy_ids if policy_id not in self.anchor_scores]
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(
                f"dev-eval summary for {self.policy_id!r} is missing anchor scores for: {missing_text}"
            )
        total = sum(self.anchor_scores[policy_id] for policy_id in anchor_policy_ids)
        return total / len(anchor_policy_ids)


DevEvalSummaryLike = float | DevEvalPolicySummary


class SnapshotEntryLike(Protocol):
    policy_id: str
    update: int


class _SnapshotRegistryAccess(Protocol):
    snapshots: Sequence[SnapshotEntryLike | str]
    champion_snapshots: Sequence[str]


def select_final_policy_set_deterministic_v1(
    snapshot_registry: object,
    dev_eval_summaries: Mapping[str, DevEvalSummaryLike],
    config: FinalPolicySetSelectionConfig,
    final_policy_set_size: int,
) -> list[str]:
    """Select the final policy set deterministically from snapshots and dev-eval summaries."""
    if final_policy_set_size < 1:
        raise ValueError("final_policy_set_size must be at least 1")

    normalized_summaries = _normalize_dev_eval_summaries(dev_eval_summaries)
    snapshot_policies = _snapshot_training_policies(snapshot_registry)
    snapshot_policies_by_id = {policy.policy_id: policy for policy in snapshot_policies}
    selected: list[str] = []

    if config.include_random_legal_baseline_b0:
        _append_unique(selected, RANDOM_LEGAL_POLICY_ID)
    if config.include_no_league_baseline_b1:
        _append_unique(selected, NO_LEAGUE_POLICY_ID)
    if config.include_heuristic_public_b2_if_exists and HEURISTIC_PUBLIC_POLICY_ID in normalized_summaries:
        _append_unique(selected, HEURISTIC_PUBLIC_POLICY_ID)

    if config.include_final_champion_snapshot:
        latest_champion = _latest_champion_policy(snapshot_registry, snapshot_policies_by_id=snapshot_policies_by_id)
        if latest_champion is not None:
            _append_unique(selected, latest_champion.policy_id)

    if snapshot_policies:
        latest_snapshot = max(snapshot_policies, key=_training_policy_sort_key)
        for percent in config.include_spaced_snapshots_near_percent_updates:
            target_update = int(latest_snapshot.update * percent / 100)
            closest_snapshot = _find_closest_snapshot(snapshot_policies, target_update)
            _append_unique(selected, closest_snapshot.policy_id)

    remaining_slots = final_policy_set_size - len(selected)
    if remaining_slots <= 0:
        return selected[:final_policy_set_size]

    anchor_policy_ids = _configured_anchor_policy_ids(config, normalized_summaries)
    ranked_candidates = _rank_remaining_candidates(
        normalized_summaries,
        anchor_policy_ids=anchor_policy_ids,
        selected_policy_ids=set(selected),
        strategy=config.remaining_slots_strategy,
        tie_break=config.tie_break,
    )
    selected.extend(ranked_candidates[:remaining_slots])
    return selected


def parse_training_policy_id(policy_id: str) -> TrainingPolicyId:
    """Parse a legacy training snapshot policy ID like ``train_u50000_p3``."""
    match = _TRAINING_POLICY_ID_RE.fullmatch(policy_id)
    if match is None:
        raise ValueError(
            "training snapshot policy IDs must match 'train_u{update}_p{version}', "
            f"got {policy_id!r}"
        )
    return TrainingPolicyId(
        policy_id=policy_id,
        update=int(match.group("update")),
        version=int(match.group("version")),
    )


def _append_unique(selected: list[str], policy_id: str) -> None:
    if policy_id not in selected:
        selected.append(policy_id)


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
    champion_policies: list[TrainingPolicyId] = []
    for policy_id in _champion_snapshot_ids(snapshot_registry):
        existing = snapshot_policies_by_id.get(policy_id)
        if existing is not None:
            champion_policies.append(existing)
            continue
        parsed = _try_parse_training_policy(policy_id)
        if parsed is not None:
            champion_policies.append(parsed)
    if not champion_policies:
        return None
    return max(champion_policies, key=_training_policy_sort_key)


def _normalize_dev_eval_summaries(
    dev_eval_summaries: Mapping[str, DevEvalSummaryLike],
) -> dict[str, DevEvalPolicySummary]:
    normalized: dict[str, DevEvalPolicySummary] = {}
    for policy_id, summary in dev_eval_summaries.items():
        if isinstance(summary, DevEvalPolicySummary):
            if summary.policy_id != policy_id:
                raise ValueError(
                    f"dev-eval summary key {policy_id!r} does not match embedded policy_id {summary.policy_id!r}"
                )
            normalized[policy_id] = summary
            continue
        if isinstance(summary, bool) or not isinstance(summary, (int, float)):
            raise TypeError(
                "dev_eval_summaries values must be floats or DevEvalPolicySummary instances, "
                f"got {type(summary).__name__} for {policy_id!r}"
            )
        normalized[policy_id] = DevEvalPolicySummary(policy_id=policy_id, aggregate_score=float(summary))
    return normalized


def _snapshot_training_policies(snapshot_registry: object) -> list[TrainingPolicyId]:
    parsed: list[TrainingPolicyId] = []
    for snapshot in _snapshot_entries(snapshot_registry):
        candidate = _parse_registry_snapshot(snapshot)
        if candidate is not None:
            parsed.append(candidate)
    return parsed


def _snapshot_entries(snapshot_registry: object) -> Sequence[SnapshotEntryLike | str]:
    if not hasattr(snapshot_registry, "snapshots"):
        raise TypeError("snapshot_registry must expose a snapshots sequence")
    registry = cast(_SnapshotRegistryAccess, snapshot_registry)
    return registry.snapshots


def _champion_snapshot_ids(snapshot_registry: object) -> Sequence[str]:
    if not hasattr(snapshot_registry, "champion_snapshots"):
        raise TypeError("snapshot_registry must expose champion_snapshots")
    registry = cast(_SnapshotRegistryAccess, snapshot_registry)
    return registry.champion_snapshots


def _parse_registry_snapshot(snapshot: object) -> TrainingPolicyId | None:
    if isinstance(snapshot, str):
        return _try_parse_training_policy(snapshot)
    if hasattr(snapshot, "policy_id") and hasattr(snapshot, "update"):
        snapshot_entry = cast(SnapshotEntryLike, snapshot)
        return _parse_training_policy_like(
            str(snapshot_entry.policy_id),
            update=int(snapshot_entry.update),
        )
    raise TypeError(f"unsupported snapshot entry type: {type(snapshot).__name__}")


def _parse_training_policy_like(policy_id: str, *, update: int | None = None) -> TrainingPolicyId:
    parsed_legacy = _try_parse_training_policy(policy_id)
    if parsed_legacy is not None:
        return parsed_legacy

    match = _POLICY_VERSION_ID_RE.fullmatch(policy_id)
    if match is None:
        raise ValueError(
            "training snapshot policy IDs must either match 'train_u{update}_p{version}' "
            "or the durable registry format 'policy_{version}'"
        )
    if update is None:
        raise ValueError(f"durable snapshot policy ID {policy_id!r} requires registry update metadata")
    return TrainingPolicyId(policy_id=policy_id, update=int(update), version=int(match.group("version")))


def _try_parse_training_policy(policy_id: str) -> TrainingPolicyId | None:
    try:
        return parse_training_policy_id(policy_id)
    except ValueError:
        return None


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
) -> list[str]:
    if strategy != "top_dev_performers_vs_anchor_set_v1":
        raise ValueError(f"unsupported final-policy-set remaining_slots_strategy: {strategy!r}")

    ranked: list[tuple[str, float]] = []
    excluded_policy_ids = selected_policy_ids | set(anchor_policy_ids)
    for policy_id, summary in dev_eval_summaries.items():
        if policy_id in excluded_policy_ids:
            continue
        ranked.append((policy_id, summary.mean_anchor_score(anchor_policy_ids)))

    ranked.sort(key=lambda item: (-item[1], _policy_tie_break_key(item[0], tie_break=tie_break)))
    return [policy_id for policy_id, _score in ranked]


def _training_policy_sort_key(policy: TrainingPolicyId) -> tuple[int, int, str]:
    return (policy.update, policy.version, policy.policy_id)


__all__ = [
    "DevEvalPolicySummary",
    "HEURISTIC_PUBLIC_POLICY_ID",
    "NO_LEAGUE_POLICY_ID",
    "RANDOM_LEGAL_POLICY_ID",
    "TrainingPolicyId",
    "parse_training_policy_id",
    "select_final_policy_set_deterministic_v1",
]
