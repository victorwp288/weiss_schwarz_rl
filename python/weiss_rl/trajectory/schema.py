from __future__ import annotations

"""Canonical trajectory schema objects (storage contract).

v1 contract:
- Per-step fields correspond to master plan §7.2.
- Reward is stored in actor perspective as provided by the simulator.
- behavior_logp is log π_behavior(a_t | obs_t, legal_t) for the stored action.
- If a wrapper folds multiple raw decisions into one stored step, it must record k_raw_decisions.
"""

from dataclasses import dataclass
from typing import Literal


TRAJ_SCHEMA_VERSION: int = 1

LegalRepr = Literal["ids_offsets", "mask", "none"]


@dataclass(slots=True)
class TrajectoryStep:
    """Required per-step stored fields at the chosen step definition."""

    #Observation (dtype recorded at chunk-level)
    obs: list[int]  # int16 or int32 values, length OBS_LEN

    #Turn identity
    to_play_seat: int  # int8 (canonical values {0,1})

    #Decision identity and action
    decision_id: int  # int32, monotonic per env (simulator-provided)
    action: int  # uint32 action id

    #Reward/termination
    reward: float  #float32, actor perspective
    terminated: bool
    truncated: bool
    engine_status: int  #int16/int32; 0 = OK

    #Episode identity
    episode_seed: int  # uint64
    episode_key: int | bytes  # uint64 or raw bytes

    #Behavior policy
    behavior_logp: float  # float32 log π_behavior(a | obs, legal)


"""Optional per-step fields (not required for training correctness)."""
@dataclass(slots=True)
class TrajectoryOptional:
    policy_version: int | None = None  #int32
    value_pred: float | None = None  #float32 debug


"""Optional debug/analysis per-step fields (not required for training correctness)."""
@dataclass(slots=True)
class TrajectoryDebug:
    decision_kind: int | None = None  #int8/uint8 tag
    legal_fingerprint64: int | None = None  #uint64
    actor: int | None = None  #optional alias of to_play_seat

    #Time-scale disambiguation: number of underlying DecisionBoundaryEnv steps executed (>= 1)
    #Only required when step_definition folds multiple raw decisions into one stored step.
    k_raw_decisions: int | None = None


"""Chunk-level fields stored once per unroll (v1)."""
@dataclass(slots=True)
class TrajectoryChunkMeta:
    schema_version: int = TRAJ_SCHEMA_VERSION

    #Interpretation metadata
    obs_dtype: str = "int16"  #"int16" or "int32"
    legal_repr: LegalRepr = "none"
    visibility_mode: str | None = None  #e.g., "public"

    #Provenance (optional)
    run_id256: bytes | None = None
    config_hash256: bytes | None = None
    spec_hash256: bytes | None = None
    git_sha: str | None = None
    build_info: str | None = None
    python_version: str | None = None
    torch_version: str | None = None
