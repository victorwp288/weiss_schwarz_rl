"""Deterministic final policy-set selection routines."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, cast

from weiss_rl.config.models import FinalPolicySetSelectionConfig

RANDOM_LEGAL_POLICY_ID = "B0 RandomLegal"
NO_LEAGUE_POLICY_ID = "B1 NoLeague baseline"
LEGACY_NO_LEAGUE_POLICY_ID = "b1_noleague_baseline"
HEURISTIC_PUBLIC_POLICY_ID = "B2 HeuristicPublic"
HEURISTIC_PUBLIC_AGGRO_POLICY_ID = "B3 HeuristicPublicAggro"
HEURISTIC_PUBLIC_CONTROL_POLICY_ID = "B4 HeuristicPublicControl"

MAIN_DECK_ID = "preset:main_deck_5hy_yotsuba_v1"
STARTER_DECK_ID = "preset:starter_deck_ws02_v1"
AGGRO_DECK_ID = "preset:aggro_deck_5hy_nino_v1"
CONTROL_DECK_ID = "preset:control_deck_jj_s66_v1"

_HEURISTIC_PUBLIC_PROFILE_BY_POLICY_ID = {
    HEURISTIC_PUBLIC_POLICY_ID: "base",
    HEURISTIC_PUBLIC_AGGRO_POLICY_ID: "aggressive",
    HEURISTIC_PUBLIC_CONTROL_POLICY_ID: "control",
}

_EVAL_DECK_BY_POLICY_ID = {
    RANDOM_LEGAL_POLICY_ID: MAIN_DECK_ID,
    NO_LEAGUE_POLICY_ID: MAIN_DECK_ID,
    LEGACY_NO_LEAGUE_POLICY_ID: MAIN_DECK_ID,
    HEURISTIC_PUBLIC_POLICY_ID: MAIN_DECK_ID,
    HEURISTIC_PUBLIC_AGGRO_POLICY_ID: AGGRO_DECK_ID,
    HEURISTIC_PUBLIC_CONTROL_POLICY_ID: CONTROL_DECK_ID,
}

_TRAINING_POLICY_ID_RE = re.compile(r"^train_u(?P<update>\d+)_p(?P<version>\d+)$")
_POLICY_VERSION_ID_RE = re.compile(r"^policy_(?P<version>\d+)$")


def heuristic_public_profile_name_for_policy_id(policy_id: str) -> str | None:
    return _HEURISTIC_PUBLIC_PROFILE_BY_POLICY_ID.get(str(policy_id))


def heuristic_public_policy_ids(*, include_base: bool = True) -> tuple[str, ...]:
    policy_ids = tuple(_HEURISTIC_PUBLIC_PROFILE_BY_POLICY_ID)
    if include_base:
        return policy_ids
    return tuple(policy_id for policy_id in policy_ids if policy_id != HEURISTIC_PUBLIC_POLICY_ID)


def deck_id_for_policy_id(policy_id: str) -> str:
    return _EVAL_DECK_BY_POLICY_ID.get(str(policy_id), MAIN_DECK_ID)


def recommend_focal_policy_id(
    *,
    snapshot_registry: object,
    dev_eval_summaries: Mapping[str, DevEvalSummaryLike],
    candidate_policy_ids: Sequence[str],
) -> str | None:
    """Recommend a non-baseline focal policy for reporting from a resolved final policy set.

    The recommendation prefers policies with canonicalized dev-eval summaries and falls back to
    the newest training snapshot among the eligible candidates when summary coverage is missing.
    """

    snapshot_policies = _snapshot_training_policies(snapshot_registry)
    normalized_summaries = _canonicalize_dev_eval_summaries(
        _normalize_dev_eval_summaries(dev_eval_summaries),
        snapshot_policies=snapshot_policies,
    )
    eligible_policy_ids = [str(policy_id) for policy_id in candidate_policy_ids if _is_focal_candidate(str(policy_id))]
    if not eligible_policy_ids:
        return None

    summarized_candidates = [
        normalized_summaries[policy_id] for policy_id in eligible_policy_ids if policy_id in normalized_summaries
    ]
    if summarized_candidates:
        return max(
            summarized_candidates,
            key=lambda summary: (
                float(summary.aggregate_score),
                len(summary.anchor_scores),
                *_training_policy_tie_break(summary.policy_id),
                summary.policy_id,
            ),
        ).policy_id

    snapshot_policies_by_id = {policy.policy_id: policy for policy in snapshot_policies}
    parsed_candidates: list[TrainingPolicyId] = []
    seen_policy_ids: set[str] = set()
    for policy_id in eligible_policy_ids:
        parsed_policy = snapshot_policies_by_id.get(policy_id)
        if parsed_policy is None:
            parsed_policy = _try_parse_training_policy(policy_id)
        if parsed_policy is None or parsed_policy.policy_id in seen_policy_ids:
            continue
        parsed_candidates.append(parsed_policy)
        seen_policy_ids.add(parsed_policy.policy_id)
    if parsed_candidates:
        return max(parsed_candidates, key=_training_policy_sort_key).policy_id

    return eligible_policy_ids[0]


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
            raise ValueError(f"dev-eval summary for {self.policy_id!r} is missing anchor scores for: {missing_text}")
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

    snapshot_policies = _snapshot_training_policies(snapshot_registry)
    normalized_summaries = _canonicalize_dev_eval_summaries(
        _normalize_dev_eval_summaries(dev_eval_summaries),
        snapshot_policies=snapshot_policies,
    )
    snapshot_policies_by_id = {policy.policy_id: policy for policy in snapshot_policies}
    selected: list[str] = []

    if config.include_random_legal_baseline_b0:
        _append_unique(selected, RANDOM_LEGAL_POLICY_ID)
    if config.include_no_league_baseline_b1:
        _append_unique(selected, NO_LEAGUE_POLICY_ID)
    if config.include_heuristic_public_b2_if_exists and HEURISTIC_PUBLIC_POLICY_ID in normalized_summaries:
        _append_unique(selected, HEURISTIC_PUBLIC_POLICY_ID)
    if config.include_heuristic_public_anchors_b2_b3_b4:
        for policy_id in (
            HEURISTIC_PUBLIC_POLICY_ID,
            HEURISTIC_PUBLIC_AGGRO_POLICY_ID,
            HEURISTIC_PUBLIC_CONTROL_POLICY_ID,
        ):
            _append_unique(selected, policy_id)

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
        raise ValueError(f"training snapshot policy IDs must match 'train_u{{update}}_p{{version}}', got {policy_id!r}")
    return TrainingPolicyId(
        policy_id=policy_id,
        update=int(match.group("update")),
        version=int(match.group("version")),
    )


def _append_unique(selected: list[str], policy_id: str) -> None:
    if policy_id not in selected:
        selected.append(policy_id)


def _is_focal_candidate(policy_id: str) -> bool:
    if policy_id in {RANDOM_LEGAL_POLICY_ID, NO_LEAGUE_POLICY_ID, LEGACY_NO_LEAGUE_POLICY_ID}:
        return False
    return policy_id not in _HEURISTIC_PUBLIC_PROFILE_BY_POLICY_ID


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


def _training_policy_tie_break(policy_id: str) -> tuple[int, int]:
    parsed = _try_parse_training_policy(policy_id)
    if parsed is None:
        return (-1, -1)
    return (parsed.update, parsed.version)


def _latest_champion_policy(
    snapshot_registry: object,
    *,
    snapshot_policies_by_id: Mapping[str, TrainingPolicyId],
) -> TrainingPolicyId | None:
    champion_policies = [
        snapshot_policies_by_id[policy_id]
        for policy_id in _champion_snapshot_ids(snapshot_registry)
        if policy_id in snapshot_policies_by_id
    ]
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


def _canonicalize_dev_eval_summaries(
    dev_eval_summaries: Mapping[str, DevEvalPolicySummary],
    *,
    snapshot_policies: Sequence[TrainingPolicyId],
) -> dict[str, DevEvalPolicySummary]:
    registry_policy_id_by_key = {(policy.update, policy.version): policy.policy_id for policy in snapshot_policies}
    canonical: dict[str, DevEvalPolicySummary] = {}
    for policy_id, summary in dev_eval_summaries.items():
        canonical_policy_id = policy_id
        parsed = _try_parse_training_policy(policy_id)
        if parsed is not None:
            canonical_policy_id = registry_policy_id_by_key.get((parsed.update, parsed.version), "")
            if not canonical_policy_id:
                continue
        existing = canonical.get(canonical_policy_id)
        candidate = DevEvalPolicySummary(
            policy_id=canonical_policy_id,
            aggregate_score=summary.aggregate_score,
            anchor_scores=summary.anchor_scores,
        )
        if existing is None:
            canonical[canonical_policy_id] = candidate
            continue
        if len(candidate.anchor_scores) > len(existing.anchor_scores):
            canonical[canonical_policy_id] = candidate
            continue
        if (
            len(candidate.anchor_scores) == len(existing.anchor_scores)
            and candidate.aggregate_score > existing.aggregate_score
        ):
            canonical[canonical_policy_id] = candidate
    return canonical


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
        return _try_parse_training_policy_like(
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


def _try_parse_training_policy_like(policy_id: str, *, update: int | None = None) -> TrainingPolicyId | None:
    try:
        return _parse_training_policy_like(policy_id, update=update)
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
    "LEGACY_NO_LEAGUE_POLICY_ID",
    "MAIN_DECK_ID",
    "NO_LEAGUE_POLICY_ID",
    "RANDOM_LEGAL_POLICY_ID",
    "TrainingPolicyId",
    "deck_id_for_policy_id",
    "parse_training_policy_id",
    "select_final_policy_set_deterministic_v1",
]
