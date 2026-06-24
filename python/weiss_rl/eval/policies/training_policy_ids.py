"""Training snapshot policy ID parsing and ordering."""

from __future__ import annotations

import re
from dataclasses import dataclass

_TRAINING_POLICY_ID_RE = re.compile(r"^train_u(?P<update>\d+)_p(?P<version>\d+)$")
_POLICY_VERSION_ID_RE = re.compile(r"^policy_(?P<version>\d+)$")


@dataclass(frozen=True, slots=True)
class TrainingPolicyId:
    policy_id: str
    update: int
    version: int


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


def parse_training_policy_like(policy_id: str, *, update: int | None = None) -> TrainingPolicyId:
    parsed_legacy = try_parse_training_policy(policy_id)
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


def try_parse_training_policy(policy_id: str) -> TrainingPolicyId | None:
    try:
        return parse_training_policy_id(policy_id)
    except ValueError:
        return None


def try_parse_training_policy_like(policy_id: str, *, update: int | None = None) -> TrainingPolicyId | None:
    try:
        return parse_training_policy_like(policy_id, update=update)
    except ValueError:
        return None


def training_policy_tie_break(policy_id: str) -> tuple[int, int]:
    parsed = try_parse_training_policy(policy_id)
    if parsed is None:
        return (-1, -1)
    return (parsed.update, parsed.version)


def training_policy_sort_key(policy: TrainingPolicyId) -> tuple[int, int, str]:
    return (policy.update, policy.version, policy.policy_id)


__all__ = [
    "TrainingPolicyId",
    "parse_training_policy_id",
    "parse_training_policy_like",
    "training_policy_sort_key",
    "training_policy_tie_break",
    "try_parse_training_policy",
    "try_parse_training_policy_like",
]
