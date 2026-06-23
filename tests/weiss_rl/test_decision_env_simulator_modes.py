from __future__ import annotations

import numpy as np
import pytest
from weiss_rl.envs.decision_env import DecisionBoundaryBatch, LegalMode

from tests.weiss_rl.decision_env_test_support import (
    assert_batch_episode_identity_matches_pool,
    first_legal_from_ids,
    first_legal_from_mask,
    illegal_action_from_ids,
    illegal_action_from_mask,
    make_env,
)


def test_mask_mode_reset_and_step_return_typed_batch() -> None:
    env = make_env(legality="mask")
    try:
        batch = env.reset()
        assert isinstance(batch, DecisionBoundaryBatch)
        assert batch.num_envs == env.num_envs
        assert batch.mask is not None
        assert batch.ids_offsets is None
        assert batch.obs.shape[0] == env.num_envs
        assert batch.reward.shape == (env.num_envs,)
        assert batch.terminated.shape == (env.num_envs,)
        assert batch.truncated.shape == (env.num_envs,)
        assert batch.decision_id.shape == (env.num_envs,)
        assert batch.engine_status.shape == (env.num_envs,)
        assert batch.episode_seed.shape == (env.num_envs,)
        assert batch.episode_key.shape == (env.num_envs,)
        assert np.array_equal(batch.to_play, batch.actor)
        assert batch.action_space == env.action_space
        assert_batch_episode_identity_matches_pool(batch, env.pool)

        next_batch = env.step(first_legal_from_mask(batch.mask, pass_action_id=env.pass_action_id))
        assert isinstance(next_batch, DecisionBoundaryBatch)
        assert next_batch.mask is not None
        assert next_batch.ids_offsets is None
        assert next_batch.obs.shape[0] == env.num_envs
        assert next_batch.reward.shape == (env.num_envs,)
        assert next_batch.action_space == env.action_space
        assert_batch_episode_identity_matches_pool(next_batch, env.pool)
    finally:
        env.close()


def test_ids_offsets_mode_reset_and_step_return_typed_batch() -> None:
    env = make_env(legality="ids_offsets")
    try:
        batch = env.reset()
        assert isinstance(batch, DecisionBoundaryBatch)
        assert batch.num_envs == env.num_envs
        assert batch.mask is None
        assert batch.ids_offsets is not None
        legal_ids, legal_offsets = batch.ids_offsets
        assert legal_ids.ndim == 1
        assert legal_offsets.shape == (env.num_envs + 1,)
        assert legal_ids.shape == (int(legal_offsets[-1]),)
        assert batch.episode_seed.shape == (env.num_envs,)
        assert batch.episode_key.shape == (env.num_envs,)
        assert np.array_equal(batch.to_play, batch.actor)
        assert batch.action_space == env.action_space
        assert_batch_episode_identity_matches_pool(batch, env.pool)

        next_batch = env.step(first_legal_from_ids(legal_ids, legal_offsets, pass_action_id=env.pass_action_id))
        assert isinstance(next_batch, DecisionBoundaryBatch)
        assert next_batch.mask is None
        assert next_batch.ids_offsets is not None
        next_legal_ids, next_legal_offsets = next_batch.ids_offsets
        assert next_legal_ids.shape == (int(next_legal_offsets[-1]),)
        assert next_batch.reward.shape == (env.num_envs,)
        assert next_batch.action_space == env.action_space
        assert_batch_episode_identity_matches_pool(next_batch, env.pool)
    finally:
        env.close()


def test_ids_offsets_mode_step_sample_from_logits_with_logp_matches_manual() -> None:
    env = make_env(legality="ids_offsets")
    try:
        batch = env.reset()
        assert batch.ids_offsets is not None
        legal_ids, legal_offsets = batch.ids_offsets
        legal_ids_before = np.asarray(legal_ids, dtype=np.uint32).copy()
        legal_offsets_before = np.asarray(legal_offsets, dtype=np.uint32).copy()
        logits = np.random.default_rng(202).standard_normal((env.num_envs, env.action_space), dtype=np.float32)
        next_batch, actions, action_logp = env.step_sample_from_logits_with_logp(
            logits,
            np.array([11 + idx for idx in range(env.num_envs)], dtype=np.uint64),
        )
        assert isinstance(next_batch, DecisionBoundaryBatch)
        assert actions.shape == (env.num_envs,)
        assert action_logp.shape == (env.num_envs,)
        expected = np.zeros((env.num_envs,), dtype=np.float32)
        for env_index in range(env.num_envs):
            start = int(legal_offsets_before[env_index])
            end = int(legal_offsets_before[env_index + 1])
            ids = np.asarray(legal_ids_before[start:end], dtype=np.int64)
            row = np.asarray(logits[env_index, ids], dtype=np.float64)
            max_logit = float(np.max(row))
            probs = np.exp(row - max_logit)
            total = float(np.sum(probs))
            chosen = int(actions[env_index])
            chosen_index = int(np.flatnonzero(ids == chosen)[0])
            expected[env_index] = float((row[chosen_index] - max_logit) - np.log(total))
        np.testing.assert_allclose(action_logp, expected, rtol=1e-6, atol=1e-6)
    finally:
        env.close()


@pytest.mark.parametrize("legality", ["mask", "ids_offsets"])
def test_step_rejects_illegal_action(legality: LegalMode) -> None:
    env = make_env(legality=legality, num_envs=1)
    try:
        batch = env.reset()
        if batch.mask is not None:
            illegal_action = illegal_action_from_mask(batch.mask)
        else:
            assert batch.ids_offsets is not None
            legal_ids, legal_offsets = batch.ids_offsets
            illegal_action = illegal_action_from_ids(
                legal_ids,
                legal_offsets,
                action_space=env.action_space,
            )

        with pytest.raises(ValueError, match="illegal action"):
            env.step(np.array([illegal_action], dtype=np.uint32))
    finally:
        env.close()
