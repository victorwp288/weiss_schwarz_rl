from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import numpy.testing as npt
import pytest
from weiss_rl.envs.decision_env import LegalMode, _derive_episode_key, _pack_batch, _validate_actions

from tests.weiss_rl.decision_env_test_support import (
    FakePool,
    done_step,
    ids_step,
    simulator_episode_key,
)


@pytest.mark.parametrize("legality", ["mask", "ids_offsets"])
def test_done_batch_with_actor_minus_one_and_empty_legality_is_pass_only(legality: LegalMode) -> None:
    pass_action_id = 51
    batch = _pack_batch(done_step(legality), legality=legality)

    assert int(batch.actor[0]) == -1
    assert bool(batch.truncated[0])

    _validate_actions(np.array([pass_action_id], dtype=np.uint32), batch, pass_action_id=pass_action_id)
    with pytest.raises(ValueError, match=f"expected pass action {pass_action_id}"):
        _validate_actions(np.array([0], dtype=np.uint32), batch, pass_action_id=pass_action_id)


def test_pack_batch_ids_offsets_trims_raw_capacity_and_derives_episode_identity_from_pool() -> None:
    step = SimpleNamespace(
        obs=np.zeros((2, 4), dtype=np.int16),
        rewards=np.zeros((2,), dtype=np.float32),
        terminated=np.array([False, False]),
        truncated=np.array([False, False]),
        actor=np.array([0, 1], dtype=np.int32),
        decision_kind=np.array([0, 0], dtype=np.int32),
        decision_id=np.array([7, 8], dtype=np.int64),
        engine_status=np.array([0, 0], dtype=np.uint8),
        spec_hash=np.array([11, 12], dtype=np.uint64),
        legal_ids=np.array([1, 2, 3, 4, 99, 98], dtype=np.uint32),
        legal_action_meta=np.array(
            [
                [2, 0, 0, 0],
                [2, 1, 0, 0],
                [3, 0, 1, 0],
                [3, 1, 1, 0],
                [65535, 65535, 65535, 65535],
                [65535, 65535, 65535, 65535],
            ],
            dtype=np.uint16,
        ),
        legal_offsets=np.array([0, 2, 4], dtype=np.uint32),
    )
    pool = FakePool(
        envs_len=2,
        episode_seed_batch=np.array([101, 202], dtype=np.uint64),
        episode_index_batch=np.array([5, 6], dtype=np.uint32),
        env_index_batch=np.array([0, 1], dtype=np.uint32),
    )

    batch = _pack_batch(step, legality="ids_offsets", pool=pool)

    assert batch.ids_offsets is not None
    legal_ids, legal_offsets = batch.ids_offsets
    npt.assert_array_equal(legal_ids, np.array([1, 2, 3, 4], dtype=np.uint32))
    npt.assert_array_equal(legal_offsets, np.array([0, 2, 4], dtype=np.uint32))
    assert batch.legal_action_meta is not None
    npt.assert_array_equal(
        batch.legal_action_meta,
        np.array(
            [
                [2, 0, 0, 0],
                [2, 1, 0, 0],
                [3, 0, 1, 0],
                [3, 1, 1, 0],
            ],
            dtype=np.uint16,
        ),
    )
    npt.assert_array_equal(batch.episode_seed, np.array([101, 202], dtype=np.uint64))
    npt.assert_array_equal(
        batch.episode_key,
        simulator_episode_key(
            np.array([101, 202], dtype=np.uint64),
            np.array([5, 6], dtype=np.uint64),
            np.array([0, 1], dtype=np.uint64),
        ),
    )
    npt.assert_array_equal(batch.decision_kind, np.array([0, 0], dtype=np.int32))
    assert batch.action_space == 52
    assert batch.episode_identity_source == "derived"


def test_pack_batch_prefers_simulator_episode_identity_when_present() -> None:
    step = SimpleNamespace(
        obs=np.zeros((1, 4), dtype=np.int16),
        rewards=np.zeros((1,), dtype=np.float32),
        terminated=np.array([False]),
        truncated=np.array([False]),
        actor=np.array([0], dtype=np.int32),
        decision_kind=np.array([7], dtype=np.int32),
        decision_id=np.array([17], dtype=np.int64),
        engine_status=np.array([0], dtype=np.uint8),
        spec_hash=np.array([11], dtype=np.uint64),
        episode_seed=np.array([333], dtype=np.uint64),
        episode_key=np.array([444], dtype=np.uint64),
        legal_ids=np.array([1, 2], dtype=np.uint32),
        legal_offsets=np.array([0, 2], dtype=np.uint32),
    )
    pool = FakePool(
        envs_len=1,
        episode_seed_batch=np.array([101], dtype=np.uint64),
        episode_index_batch=np.array([5], dtype=np.uint32),
        env_index_batch=np.array([0], dtype=np.uint32),
    )

    batch = _pack_batch(step, legality="ids_offsets", pool=pool)

    npt.assert_array_equal(batch.episode_seed, np.array([333], dtype=np.uint64))
    npt.assert_array_equal(batch.episode_key, np.array([444], dtype=np.uint64))
    npt.assert_array_equal(batch.decision_kind, np.array([7], dtype=np.int32))
    assert batch.episode_identity_source == "simulator"


def test_pack_batch_can_return_views_for_runtime_fast_path() -> None:
    obs = np.arange(8, dtype=np.int16).reshape(2, 4)
    rewards = np.array([1.0, 2.0], dtype=np.float32)
    terminated = np.array([False, True])
    truncated = np.array([False, False])
    actor = np.array([0, 1], dtype=np.int32)
    decision_kind = np.array([3, 4], dtype=np.int32)
    decision_id = np.array([17, 18], dtype=np.int64)
    engine_status = np.array([0, 0], dtype=np.uint8)
    mask = np.array([[True, False, True], [False, True, False]], dtype=np.bool_)
    step = SimpleNamespace(
        obs=obs,
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
        actor=actor,
        decision_kind=decision_kind,
        decision_id=decision_id,
        engine_status=engine_status,
        spec_hash=np.array([11, 12], dtype=np.uint64),
        masks=mask,
        episode_seed=np.array([333, 444], dtype=np.uint64),
        episode_key=np.array([555, 666], dtype=np.uint64),
    )

    batch = _pack_batch(step, legality="mask", copy_arrays=False)

    assert np.shares_memory(batch.obs, obs)
    assert np.shares_memory(batch.reward, rewards)
    assert np.shares_memory(batch.terminated, terminated)
    assert np.shares_memory(batch.truncated, truncated)
    assert np.shares_memory(batch.actor, actor)
    assert np.shares_memory(batch.to_play, actor)
    assert np.shares_memory(batch.decision_kind, decision_kind)
    assert np.shares_memory(batch.decision_id, decision_id)
    assert np.shares_memory(batch.engine_status, engine_status)
    assert batch.mask is not None and np.shares_memory(batch.mask, mask)
    assert np.shares_memory(batch.episode_seed, step.episode_seed)
    assert np.shares_memory(batch.episode_key, step.episode_key)
    assert batch.action_space == 3


def test_derive_episode_key_matches_weiss_sim_runner_helper() -> None:
    episode_seed = np.array([0, 1, 2, 17, 123456789, 2**64 - 1], dtype=np.uint64)
    episode_index = np.array([0, 1, 7, 99, 2**16, 2**32 - 1], dtype=np.uint64)
    env_index = np.array([0, 1, 2, 3, 255, 2**32 - 1], dtype=np.uint64)

    actual = _derive_episode_key(episode_seed, episode_index, env_index)
    expected = simulator_episode_key(episode_seed, episode_index, env_index)

    npt.assert_array_equal(actual, expected)


@pytest.mark.parametrize("legality", ["mask", "ids_offsets"])
def test_pack_batch_returns_snapshot_not_sim_buffer_view(legality: LegalMode) -> None:
    step = (
        done_step(legality) if legality == "mask" else ids_step(engine_status=np.array([0], dtype=np.uint8), reward=0.0)
    )
    batch = _pack_batch(step, legality=legality)

    step.obs[...] = 77
    step.rewards[...] = 12
    step.terminated[...] = True
    step.truncated[...] = False
    step.actor[...] = 5
    step.decision_id[...] = 99
    step.engine_status[...] = 3
    if legality == "mask":
        step.masks[...] = 1
        assert batch.mask is not None
        assert not np.array_equal(batch.mask, step.masks)
    else:
        step.legal_ids[...] = 4
        step.legal_offsets[...] = 1
        assert batch.ids_offsets is not None
        legal_ids, legal_offsets = batch.ids_offsets
        assert not np.array_equal(legal_ids, step.legal_ids)
        assert not np.array_equal(legal_offsets, step.legal_offsets)

    assert not np.array_equal(batch.obs, step.obs)
    assert not np.array_equal(batch.reward, step.rewards)
    assert not np.array_equal(batch.terminated, step.terminated)
    assert not np.array_equal(batch.actor, step.actor)
