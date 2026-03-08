"""Replay bundle serialization scaffold."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

from weiss_rl.repro import (
    derive_replay_key256,
    key256_to_hex,
    key256_to_short64,
    resolve_episode_key256,
)


@dataclass(slots=True)
class ReplayRecord:
    #Always present episode identity
    episode_key: str        # hex of episode_key256 (kept for backward compatibility)
    episode_key64: int      # uint64 short id for filenames
    replay_key256: str      # hex of replay_key256
    replay_key64: int       # uint64 short id for filenames
        
    #Step Payload
    decision_id: int
    action: int
    reward: float
    terminated: bool
    truncated: bool


def make_replay_record(
    *,
    simulator_episode_key: bytes | None,
    run_id256: bytes,
    spec_hash256: bytes,
    actor_id: int,
    env_id: int,
    episode_index: int,
    episode_seed64: int,
    decision_id: int,
    action: int,
    reward: float,
    terminated: bool,
    truncated: bool,
) -> ReplayRecord:
    episode_key256 = resolve_episode_key256(
        simulator_episode_key=simulator_episode_key,
        run_id256=run_id256,
        actor_id=actor_id,
        env_id=env_id,
        episode_index=episode_index,
        episode_seed64=episode_seed64,
    )
    episode_key64 = key256_to_short64(episode_key256)

    replay_key256_b = derive_replay_key256(episode_key256=episode_key256, spec_hash256=spec_hash256)
    replay_key64 = key256_to_short64(replay_key256_b)

    return ReplayRecord(
        episode_key=key256_to_hex(episode_key256),
        episode_key64=episode_key64,
        replay_key256=key256_to_hex(replay_key256_b),
        replay_key64=replay_key64,
        decision_id=decision_id,
        action=action,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
    )


def write_jsonl(records: list[ReplayRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(asdict(record), separators=(",", ":")) + "\n")

