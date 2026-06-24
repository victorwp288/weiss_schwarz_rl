"""Rollout bookkeeping helpers for simulator god-search."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from weiss_rl.artifacts.reproducibility import canonical_json_bytes, stable_hash64
from weiss_rl.diagnostics.probes.action_diagnostics import make_action_sequence_state
from weiss_rl.eval.simulator.harness import ScheduledGame


def clone_optional_hidden(hidden: torch.Tensor | None) -> torch.Tensor | None:
    if hidden is None:
        return None
    return hidden.detach().clone()


def clone_seat_hidden_map(
    seat_hidden_by_seat: Mapping[int, torch.Tensor | None],
    *,
    current_seat: int,
    root_next_seat_hidden: torch.Tensor | None,
) -> dict[int, torch.Tensor | None]:
    rollout_hidden = {
        0: clone_optional_hidden(seat_hidden_by_seat.get(0)),
        1: clone_optional_hidden(seat_hidden_by_seat.get(1)),
    }
    rollout_hidden[int(current_seat)] = clone_optional_hidden(root_next_seat_hidden)
    return rollout_hidden


def copy_action_sequence_state(state: Any | None) -> Any:
    copied = make_action_sequence_state(1)
    if state is None:
        return copied
    source = getattr(state, "consecutive_main_moves_by_env", None)
    if source is None:
        return copied
    source_array = np.asarray(source, dtype=np.int32)
    if source_array.shape == copied.consecutive_main_moves_by_env.shape:
        copied.consecutive_main_moves_by_env[...] = source_array
    return copied


def god_search_rollout_sampling_algorithm(*, rollout_policy: str, eval_sampling_algorithm: str) -> str:
    if rollout_policy == "argmax":
        return "model_argmax_pinned_v1"
    if rollout_policy == "sample":
        return "pinned_cdf_pcg_v1"
    return str(eval_sampling_algorithm)


def god_search_rollout_rng_seed(
    *,
    scheduled_game: ScheduledGame,
    seat: int,
    candidate_action: int,
    rollout_index: int,
    decision_id: int,
) -> int:
    return stable_hash64(
        canonical_json_bytes(
            {
                "kind": "god_search_rollout_rng_v1",
                "pair_index": int(scheduled_game.pair_index),
                "swap_index": int(scheduled_game.swap_index),
                "episode_seed": int(scheduled_game.episode_seed),
                "seat": int(seat),
                "candidate_action": int(candidate_action),
                "rollout_index": int(rollout_index),
                "decision_id": int(decision_id),
            }
        )
    )


__all__ = [
    "clone_optional_hidden",
    "clone_seat_hidden_map",
    "copy_action_sequence_state",
    "god_search_rollout_rng_seed",
    "god_search_rollout_sampling_algorithm",
]
