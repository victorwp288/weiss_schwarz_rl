from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from weiss_rl.runtime_components.done_resets import reset_actor_hidden_for_done


class _ContextModel:
    def __init__(self) -> None:
        self.calls: list[tuple[int, tuple[str, ...] | None]] = []

    def initial_seat_hidden(
        self,
        batch_size: int,
        *,
        device: torch.device,
        opponent_policy_ids: list[str] | np.ndarray | None = None,
    ) -> torch.Tensor:
        policy_tuple = (
            None if opponent_policy_ids is None else tuple(str(policy_id) for policy_id in opponent_policy_ids)
        )
        self.calls.append((int(batch_size), policy_tuple))
        fill_value = 3.0 if policy_tuple is None else float(len(policy_tuple) + 10)
        return torch.full((int(batch_size), 2), fill_value, dtype=torch.float32, device=device)


def test_reset_actor_hidden_for_done_uses_current_opponent_ids_and_preserves_live_rows() -> None:
    model = _ContextModel()
    actor = SimpleNamespace(
        model=model,
        opponent_policy_id_by_env=np.asarray(["live", "policy_a", "policy_b"], dtype=object),
        seat_hidden=torch.tensor(
            [
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
            ],
            dtype=torch.float32,
        ),
        opponent_hidden=torch.tensor(
            [
                [4.0, 4.0],
                [5.0, 5.0],
                [6.0, 6.0],
            ],
            dtype=torch.float32,
        ),
    )

    result = reset_actor_hidden_for_done(
        actor=actor,
        done=np.asarray([False, True, True], dtype=np.bool_),
        device=torch.device("cpu"),
    )

    assert result.done_count == 2
    assert result.done.tolist() == [False, True, True]
    assert model.calls == [(2, ("policy_a", "policy_b")), (2, None)]
    assert actor.seat_hidden.tolist() == [[1.0, 1.0], [12.0, 12.0], [12.0, 12.0]]
    assert actor.opponent_hidden.tolist() == [[4.0, 4.0], [3.0, 3.0], [3.0, 3.0]]


def test_reset_actor_hidden_for_done_noops_without_done_rows() -> None:
    model = _ContextModel()
    actor = SimpleNamespace(
        model=model,
        opponent_policy_id_by_env=np.asarray(["policy_a"], dtype=object),
        seat_hidden=torch.ones((1, 2), dtype=torch.float32),
        opponent_hidden=torch.full((1, 2), 2.0, dtype=torch.float32),
    )

    result = reset_actor_hidden_for_done(
        actor=actor,
        done=np.asarray([False], dtype=np.bool_),
        device=torch.device("cpu"),
    )

    assert result.done_count == 0
    assert result.done.tolist() == [False]
    assert model.calls == []
    assert actor.seat_hidden.tolist() == [[1.0, 1.0]]
    assert actor.opponent_hidden.tolist() == [[2.0, 2.0]]
