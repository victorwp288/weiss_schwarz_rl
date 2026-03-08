from __future__ import annotations

"""Canonical trajectory schema objects.

This module defines the v1 stored semantics for trajectory steps and chunk metadata.
Any change to field meaning/layout requires bumping TRAJ_SCHEMA_VERSION and adding a
migration note in the master plan.
"""

from dataclasses import dataclass
from typing import Literal

TRAJ_SCHEMA_VERSION: int = 1

LegalRepr = Literal ["ids_offsets", "mask", "none"]


"""Canonical per-step stored fields (v1) at decision boundary t.
 
    Naming convention:
    - Fields correspond to master plan §7.2 (per-step stored fields).
    - Reward is stored in actor perspective (player-to-act) as provided by the simulator.
    - behavior_logp is log π_behavior(a_t | obs_t, legal_t) corresponding to action.
"""


@dataclass(slots=True)
class TrajectoryStep:
    
    # Observation payload
    # obs is stored as an integer vector (int16 or int32) of length OBS_LEN.
    # The dtype is recorded at chunk-level as obs_dtype.
    obs: list[int]

    # Legal action representation depends on chunk-level legal_repr.
    # If legal_repr == "ids_offsets": legal_ids and legal_offsets are stored at the chunk/unroll
    # array level, not inside this per-step object. If legal_repr == "mask": legal_mask is stored
    # at the chunk/unroll array level. This per-step object keeps no duplicate legal fields.
    # (Writers may still store per-step legal fields in array form in storage.)

    #Turn identity
    to_play_seat: int #int8 (canonical values {0,1})

    #Decision identity and selected action
    decision_id: int #int32, monotonic per env (simulator-provided)
    action: int #uint32 action id
    
    #Rewards and termination
    reward: float #float32 simulator reward, actor perspective
    terminated: bool 
    truncated: bool 
    engine_status: int #int16 or int32; 0 = OK, nonzero indicates fault
    
    episode_seed: int #uint64
    episode_key: int #uint64 or raw bytes (store raw)


@dataclass(slots=True)
class TrajectoryChunkMeta:
    # Behavior policy log-prob
    behavior_logp: float  # float32 log π_behavior(a | obs, legal)

#Optional per-step fields (not required for training correctness).
@dataclass(slots=True)
class TrajectoryOptional:
    policy_version: int | None = None  # int32 (from spec bundle)
    value_pred: float | None = None  # float32 debug value from actor net


#Optional debug/analysis per-step fields (not required for training correctness).
@dataclass(slots=True)
class TrajectoryDebug:
    decision_kind: int | None = None #int8/uint8 tag (may be sentinel unknown)
    legal_fingerprint64: int | None = None #uint64 per §16.6 legal_fingerprint_v1
    actor: int | None = None #optional alias of to_play_seat

"""Chunk-level fields stored once per unroll (v1).
    These describe how to interpret arrays (obs dtype, legal repr) and provide provenance.
"""

@dataclass(slots=True)
class TrajectoryChunkMeta:

    schema_version: int = TRAJ_SCHEMA_VERSION

    # Interpretation metadata
    obs_dtype: str = "int16"  # recorded dtype of obs arrays: "int16" or "int32"
    legal_repr: LegalRepr = "none"
    visibility_mode: str | None = None  # e.g., "public"

    # Provenance (optional but encouraged)
    run_id256: bytes | None = None
    config_hash256: bytes | None = None
    spec_hash256: bytes | None = None
    git_sha: str | None = None
    build_info: str | None = None
    python_version: str | None = None
    torch_version: str | None = None
