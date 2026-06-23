from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import numpy.testing as npt
import torch
from weiss_rl.runtime import QueueRuntime, QueueRuntimeConfig


def test_split_focal_actor_rows_forces_model_policy_on_diverse_model_lane() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._actor_policy_backend = "heuristic_public"
    runtime_any._teacher_policy = object()
    runtime_any._active_actor_heuristic_fraction = lambda: 1.0

    model_rows, heuristic_rows = QueueRuntime._split_focal_actor_rows(
        runtime,
        actor=cast(Any, SimpleNamespace(force_model_policy_lane=True)),
        focal_indices=np.asarray([0, 2, 4], dtype=np.int64),
        rng=np.random.default_rng(7),
    )

    npt.assert_array_equal(model_rows, np.asarray([0, 2, 4], dtype=np.int64))
    assert heuristic_rows.size == 0


def test_policy_train_mask_for_actor_can_exclude_pure_heuristic_lane() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._train_on_heuristic_actor_rows = False
    runtime_any._actor_policy_backend = "heuristic_public"
    runtime_any._active_actor_heuristic_fraction = lambda: 1.0

    mask = QueueRuntime._policy_train_mask_for_actor(
        runtime,
        actor=cast(Any, SimpleNamespace(force_model_policy_lane=False)),
        focal_rows=np.asarray([True, False, True, False], dtype=np.bool_),
    )

    npt.assert_array_equal(mask, np.asarray([False, False, False, False], dtype=np.bool_))


def test_reset_done_rows_fallback_reinitializes_full_actor_state() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any.config = QueueRuntimeConfig(
        mode="train_ordered",
        actor_count=1,
        envs_per_actor=2,
        unroll_length=1,
        batch_unrolls_per_update=2,
        queue_capacity_unrolls=2,
        profile="fast",
        base_seed=7,
        pass_action_id=51,
        actor_reload_interval_updates=1,
    )
    runtime_any._device = torch.device("cpu")

    assign_calls: list[tuple[np.ndarray, bool]] = []

    def fake_assign_episode_roles(actor, done, *, initial: bool = False) -> None:
        assign_calls.append((np.asarray(done, dtype=np.bool_).copy(), initial))
        actor.focal_seat_by_env[:] = np.array([0, 1], dtype=np.int64)
        actor.opponent_policy_id_by_env[:] = np.asarray(["mirror", "policy_000007"], dtype=object)

    runtime_any._assign_episode_roles = fake_assign_episode_roles

    class _FakeEnv:
        def reset_done(self, done) -> None:
            raise RuntimeError("reset_done unsupported")

        def reset(self, *, seed: int):
            return {"seed": seed}

    class _FakeModel:
        def initial_seat_hidden(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
            return torch.zeros((batch_size, 3), device=device)

    actor = cast(
        Any,
        SimpleNamespace(
            env=_FakeEnv(),
            model=_FakeModel(),
            rng=np.random.default_rng(7),
            seat_hidden=torch.ones((2, 3)),
            opponent_hidden=torch.full((2, 3), 2.0),
            focal_seat_by_env=np.array([1, 1], dtype=np.int64),
            opponent_policy_id_by_env=np.asarray(["stale_a", "stale_b"], dtype=object),
        ),
    )

    batch = cast(Any, QueueRuntime._reset_done_rows(runtime, actor, np.array([True, False], dtype=np.bool_)))

    assert isinstance(batch, dict)
    assert isinstance(batch["seed"], int)
    assert torch.count_nonzero(actor.seat_hidden) == 0
    assert torch.count_nonzero(actor.opponent_hidden) == 0
    assert len(assign_calls) == 1
    assert np.array_equal(assign_calls[0][0], np.array([True, True], dtype=np.bool_))
    assert assign_calls[0][1] is True
    assert actor.focal_seat_by_env.tolist() == [0, 1]
    assert actor.opponent_policy_id_by_env.tolist() == ["mirror", "policy_000007"]
