"""Replay bundle serialization scaffold."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from weiss_rl.repro import (
    derive_replay_key256,
    key256_to_hex,
    key256_to_short64,
    resolve_episode_key256,
)


@dataclass(slots=True)
class ReplayRecord:
    episode_key: str
    episode_key64: int
    replay_key256: str
    replay_key64: int
    decision_id: int
    action: int
    reward: float
    terminated: bool
    truncated: bool


def make_replay_record(
    *,
    simulator_episode_key: int | bytes | None,
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
    replay_key256 = derive_replay_key256(episode_key256=episode_key256, spec_hash256=spec_hash256)

    return ReplayRecord(
        episode_key=key256_to_hex(episode_key256),
        episode_key64=key256_to_short64(episode_key256),
        replay_key256=key256_to_hex(replay_key256),
        replay_key64=key256_to_short64(replay_key256),
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
