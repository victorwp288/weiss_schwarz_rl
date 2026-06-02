from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import torch

from weiss_rl.runtime.components.opponent_context import initial_seat_hidden_for_opponents


class ActorDoneResetResult(NamedTuple):
    done: np.ndarray
    done_count: int


def reset_actor_hidden_for_done(
    *,
    actor: Any,
    done: np.ndarray,
    device: torch.device,
) -> ActorDoneResetResult:
    done_array = np.asarray(done, dtype=np.bool_)
    done_count = int(np.count_nonzero(done_array))
    if done_count == 0:
        return ActorDoneResetResult(done=done_array, done_count=0)

    done_mask = torch.as_tensor(done_array, dtype=torch.bool, device=device)
    actor.seat_hidden[done_mask] = initial_seat_hidden_for_opponents(
        actor.model,
        done_count,
        device=device,
        opponent_policy_ids=actor.opponent_policy_id_by_env[done_array],
    )
    actor.opponent_hidden[done_mask] = initial_seat_hidden_for_opponents(
        actor.model,
        done_count,
        device=device,
    )
    return ActorDoneResetResult(done=done_array, done_count=done_count)
