from __future__ import annotations

import numpy as np
import pytest
from weiss_rl.actors.actor_opponents import current_opponent_policy_ids, resample_actor_opponents


class _Sampler:
    def __init__(self, *policy_ids: str) -> None:
        self.policy_ids = tuple(policy_ids)
        self.calls: list[int] = []

    def sample(self, *, count: int, rng: np.random.Generator) -> tuple[str, ...]:
        self.calls.append(count)
        rng.integers(0, 100)
        return self.policy_ids[:count]


def test_resample_actor_opponents_initializes_rng_and_partial_assignments() -> None:
    assignments: list[tuple[np.ndarray, tuple[str, ...]]] = []
    current = np.asarray(["old_a", "old_b", "old_c"], dtype=object)
    opponent_ids = np.asarray(["old_a", "old_b", "old_c"], dtype=object)
    sampler = _Sampler("new_a", "new_c")

    state = resample_actor_opponents(
        opponent_sampler=sampler,
        opponent_rng=None,
        seed=17,
        actor_id=3,
        num_envs=3,
        done=np.asarray([True, False, True], dtype=np.bool_),
        current_opponent_policy_ids=current,
        opponent_id_by_env=opponent_ids,
        opponent_assignment_fn=lambda done, ids: assignments.append((done, ids)),
    )

    assert state.opponent_rng is not None
    assert sampler.calls == [2]
    assert current_opponent_policy_ids(state.current_opponent_policy_ids) == ("new_a", "old_b", "new_c")
    assert state.opponent_id_by_env.tolist() == ["new_a", "old_b", "new_c"]
    assert len(assignments) == 1
    assert np.array_equal(assignments[0][0], np.asarray([True, False, True], dtype=np.bool_))
    assert assignments[0][1] == ("new_a", "old_b", "new_c")


def test_resample_actor_opponents_validates_done_shape() -> None:
    with pytest.raises(ValueError, match=r"done must have shape \(2,\)"):
        resample_actor_opponents(
            opponent_sampler=_Sampler("policy"),
            opponent_rng=None,
            seed=1,
            actor_id=0,
            num_envs=2,
            done=np.asarray([True], dtype=np.bool_),
            current_opponent_policy_ids=None,
            opponent_id_by_env=None,
            opponent_assignment_fn=None,
        )
