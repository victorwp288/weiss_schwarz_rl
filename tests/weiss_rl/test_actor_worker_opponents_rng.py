from __future__ import annotations

import numpy as np
from weiss_rl.actors.actor_worker import ActorWorker
from weiss_rl.core.masking import sample_actions_from_mask
from weiss_rl.league.opponent_pool import OpponentPoolSampler, sample_opponent_snapshot_ids
from weiss_rl.league.registry import SnapshotRegistry, snapshot_weights_relpath

from .actor_worker_test_support import ACTION_SPACE, PartialDoneResetMaskEnv, StaticMaskEnv, _uniform_policy_logits


def test_actor_worker_samples_opponent_policy_ids_on_episode_boundaries() -> None:
    registry = SnapshotRegistry()
    for update, snapshot_id in enumerate(["s1", "s2", "s3"], start=1):
        registry.add_snapshot(
            policy_id=snapshot_id,
            update=update,
            weights_sha256=(snapshot_id * 64)[:64].ljust(64, "0"),
            path=snapshot_weights_relpath(snapshot_id),
        )
    registry.add_champion("s1")
    sampler = OpponentPoolSampler(
        registry=registry,
        recent_size=2,
        champion_size=1,
        win_rates_by_snapshot_id={"s2": 0.9, "s3": 0.2},
    )
    worker = ActorWorker(
        actor_id=5,
        unroll_length=2,
        num_envs=2,
        action_space=ACTION_SPACE,
        layout_name="mask",
        seed=37,
        opponent_sampler=sampler,
    )
    env = PartialDoneResetMaskEnv()
    assignments: list[tuple[np.ndarray, tuple[str, ...]]] = []

    def record_assignment(done: np.ndarray, opponent_policy_ids: tuple[str, ...]) -> None:
        assignments.append((done.copy(), opponent_policy_ids))

    worker.opponent_assignment_fn = record_assignment
    worker.run_once(env=env, policy_logits_fn=_uniform_policy_logits)

    expected_rng = np.random.default_rng(np.random.SeedSequence([worker.seed, worker.actor_id, 1]))
    pool_ids = sampler.snapshot_ids()
    expected_initial = sample_opponent_snapshot_ids(
        pool_ids,
        count=2,
        rng=expected_rng,
        win_rates_by_snapshot_id=sampler.win_rates_by_snapshot_id,
        power=sampler.power,
        eps_uniform=sampler.eps_uniform,
        neutral_win_rate=sampler.neutral_win_rate,
    )
    expected_reset = sample_opponent_snapshot_ids(
        pool_ids,
        count=1,
        rng=expected_rng,
        win_rates_by_snapshot_id=sampler.win_rates_by_snapshot_id,
        power=sampler.power,
        eps_uniform=sampler.eps_uniform,
        neutral_win_rate=sampler.neutral_win_rate,
    )

    assert len(assignments) == 2
    assert np.array_equal(assignments[0][0], np.array([True, True], dtype=np.bool_))
    assert assignments[0][1] == expected_initial
    assert np.array_equal(assignments[1][0], np.array([True, False], dtype=np.bool_))
    assert assignments[1][1] == (expected_reset[0], expected_initial[1])
    assert worker.current_opponent_policy_ids == (expected_reset[0], expected_initial[1])


def test_actor_worker_preserves_rng_stream_across_run_once_calls() -> None:
    actor_id = 3
    seed = 17
    unroll_length = 8
    num_envs = 4
    worker = ActorWorker(
        actor_id=actor_id,
        unroll_length=unroll_length,
        num_envs=num_envs,
        action_space=ACTION_SPACE,
        layout_name="mask",
        seed=seed,
    )

    first = worker.run_once(env=StaticMaskEnv(num_envs), policy_logits_fn=_uniform_policy_logits)
    second = worker.run_once(env=StaticMaskEnv(num_envs), policy_logits_fn=_uniform_policy_logits)

    rng = np.random.default_rng(seed + actor_id)
    logits = np.zeros((num_envs, ACTION_SPACE), dtype=np.float32)
    legal_mask = np.ones((num_envs, ACTION_SPACE), dtype=np.uint8)
    expected_chunks = []
    for _ in range(2):
        chunk_rows = []
        for _ in range(unroll_length):
            actions, _, _ = sample_actions_from_mask(logits, legal_mask, rng=rng)
            chunk_rows.append(actions.astype(np.uint32, copy=False))
        expected_chunks.append(np.stack(chunk_rows, axis=0))

    assert np.array_equal(first.action, expected_chunks[0])
    assert np.array_equal(second.action, expected_chunks[1])
