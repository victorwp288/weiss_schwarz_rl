"""Deterministic capture decisions for simulator eval replay samples."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from weiss_rl.artifacts.reproducibility import canonical_json_bytes, stable_hash64
from weiss_rl.eval.simulator.harness import ScheduledGame

_U64_DENOMINATOR = float(1 << 64)


def should_capture_replay_sample(
    *,
    scheduled_game: ScheduledGame,
    capture_rate: float,
    capture_count: int,
    capture_limit: int,
) -> bool:
    if float(capture_rate) <= 0.0:
        return False
    if int(capture_count) >= int(capture_limit):
        return False
    capture_u64 = stable_hash64(
        canonical_json_bytes(
            {
                "kind": "final_eval_replay_capture_v1",
                "pair_index": int(scheduled_game.pair_index),
                "swap_index": int(scheduled_game.swap_index),
                "episode_index": int(scheduled_game.episode_index),
                "episode_seed": int(scheduled_game.episode_seed),
                "focal_policy_id": scheduled_game.focal_policy_id,
                "opponent_policy_id": scheduled_game.opponent_policy_id,
            }
        )
    )
    return (capture_u64 / _U64_DENOMINATOR) < float(capture_rate)


def replay_capture_visibility_mode(stack: Any) -> str:
    environment_config = stack.config.environment
    if environment_config is None:
        return "full"
    return "public" if str(environment_config.observation_visibility).strip().lower() == "public" else "full"


def replay_sample_dir_name(*, scheduled_game: ScheduledGame) -> str:
    payload = canonical_json_bytes(
        {
            "pair_index": int(scheduled_game.pair_index),
            "swap_index": int(scheduled_game.swap_index),
            "episode_index": int(scheduled_game.episode_index),
            "episode_seed": int(scheduled_game.episode_seed),
            "focal_policy_id": scheduled_game.focal_policy_id,
            "opponent_policy_id": scheduled_game.opponent_policy_id,
        }
    )
    return f"{scheduled_game.pair_index:04d}_{scheduled_game.swap_index:01d}_{stable_hash64(payload):016x}"


def discover_raw_replay_path(*, raw_dir: Path, before_raw_paths: set[Path]) -> Path | None:
    after_paths = set(raw_dir.glob("*.wsr"))
    new_paths = sorted(after_paths - before_raw_paths)
    if len(new_paths) == 1:
        return new_paths[0]
    if len(after_paths) == 1:
        return sorted(after_paths)[0]
    return None


__all__ = [
    "discover_raw_replay_path",
    "replay_capture_visibility_mode",
    "replay_sample_dir_name",
    "should_capture_replay_sample",
]
