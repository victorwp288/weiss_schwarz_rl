from __future__ import annotations

import numpy as np
from weiss_rl.envs.decision_env import DecisionBoundaryEnv

from .rl_step_layout_contract_test_support import actions_from_mask, actions_from_packed_legal_ids, sim_module


def test_decision_boundary_env_mask_and_ids_offsets_stay_step_equivalent() -> None:
    sim = sim_module()
    num_envs = 4
    seed = 321

    mask_sim = sim.fast(
        num_envs=num_envs,
        seed=seed,
        max_decisions=200,
        max_ticks=10_000,
        observation_visibility="public",
        legal_repr="mask_u8",
        obs_dtype="i16",
    )
    ids_sim = sim.fast(
        num_envs=num_envs,
        seed=seed,
        max_decisions=200,
        max_ticks=10_000,
        observation_visibility="public",
        legal_repr="ids_u16",
        obs_dtype="i16",
    )

    mask_env = DecisionBoundaryEnv(mask_sim.pool, legality="mask", engine_status_policy="hard_fail")
    ids_env = DecisionBoundaryEnv(ids_sim.pool, legality="ids_offsets", engine_status_policy="hard_fail")

    try:
        mask_batch = mask_env.reset()
        ids_batch = ids_env.reset()

        for _ in range(8):
            assert np.array_equal(mask_batch.obs, ids_batch.obs)
            assert np.array_equal(mask_batch.reward, ids_batch.reward)
            assert np.array_equal(mask_batch.terminated, ids_batch.terminated)
            assert np.array_equal(mask_batch.truncated, ids_batch.truncated)
            assert np.array_equal(mask_batch.actor, ids_batch.actor)
            assert np.array_equal(mask_batch.decision_id, ids_batch.decision_id)
            assert np.array_equal(mask_batch.decision_kind, ids_batch.decision_kind)
            assert np.array_equal(mask_batch.engine_status, ids_batch.engine_status)
            assert np.array_equal(mask_batch.episode_seed, ids_batch.episode_seed)
            assert np.array_equal(mask_batch.episode_key, ids_batch.episode_key)
            assert np.array_equal(mask_batch.main_move_action, ids_batch.main_move_action)
            assert np.array_equal(mask_batch.main_pass_action, ids_batch.main_pass_action)

            mask_actions = actions_from_mask(
                np.asarray(mask_batch.mask),
                action_space=int(sim.ACTION_SPACE_SIZE),
            )
            assert ids_batch.ids_offsets is not None
            ids_actions = actions_from_packed_legal_ids(
                np.asarray(ids_batch.ids_offsets[0]),
                np.asarray(ids_batch.ids_offsets[1]),
                num_envs=num_envs,
                action_space=int(sim.ACTION_SPACE_SIZE),
            )
            assert np.array_equal(mask_actions, ids_actions)

            mask_batch = mask_env.step(mask_actions)
            ids_batch = ids_env.step(ids_actions)
    finally:
        mask_env.close()
        ids_env.close()
