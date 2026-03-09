"""Canonical trajectory schema objects (storage contract)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TRAJ_SCHEMA_VERSION: int = 1

LegalRepr = Literal["ids_offsets", "mask", "none"]
StepDefinition = Literal["decision_boundary", "learner_turn_env"]

LEGAL_REPR_FIELDS: dict[LegalRepr, tuple[str, ...]] = {
    "ids_offsets": ("legal_ids", "legal_offsets"),
    "mask": ("legal_mask",),
    "none": (),
}


def legal_storage_fields(legal_repr: LegalRepr) -> tuple[str, ...]:
    """Return the canonical array fields required for the selected legal representation."""
    return LEGAL_REPR_FIELDS[legal_repr]


def requires_k_raw_decisions(step_definition: StepDefinition) -> bool:
    """Learner-turn wrapper steps must record how many raw decisions were folded."""
    return step_definition == "learner_turn_env"


@dataclass(slots=True)
class TrajectoryStep:
    """Required per-step fields for the chosen external step definition."""

    obs: list[int]
    to_play_seat: int
    decision_id: int
    action: int
    reward: float
    terminated: bool
    truncated: bool
    engine_status: int
    episode_seed: int
    episode_key: int | bytes
    behavior_logp: float


@dataclass(slots=True)
class TrajectoryOptional:
    """Optional per-step fields that are useful but not required for correctness."""

    policy_version: int | None = None
    value_pred: float | None = None


@dataclass(slots=True)
class TrajectoryDebug:
    """Optional debug and analysis fields."""

    decision_kind: int | None = None
    legal_fingerprint64: int | None = None
    actor: int | None = None
    k_raw_decisions: int | None = None

    def validate(self, *, step_definition: StepDefinition) -> None:
        if self.k_raw_decisions is not None and self.k_raw_decisions < 1:
            raise ValueError("k_raw_decisions must be >= 1 when recorded")
        if requires_k_raw_decisions(step_definition) and self.k_raw_decisions is None:
            raise ValueError("k_raw_decisions is required for learner_turn_env steps")


@dataclass(slots=True)
class TrajectoryChunkMeta:
    """Chunk-level metadata stored once per unroll."""

    schema_version: int = TRAJ_SCHEMA_VERSION
    obs_dtype: str = "int16"
    legal_repr: LegalRepr = "none"
    visibility_mode: str | None = None
    run_id256: bytes | None = None
    config_hash256: bytes | None = None
    spec_hash256: bytes | None = None
    git_sha: str | None = None
    build_info: str | None = None
    python_version: str | None = None
    torch_version: str | None = None
