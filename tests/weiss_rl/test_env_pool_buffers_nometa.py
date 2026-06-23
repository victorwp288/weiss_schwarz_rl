from __future__ import annotations

import numpy as np

from .rl_step_layout_contract_test_support import LEGAL_DECK, sim_module


def test_env_pool_buffers_nometa_exposes_context_and_sampled_logp() -> None:
    sim = sim_module()
    pool, buffers = sim.make_pool(
        mode="train",
        num_envs=2,
        db_path=None,
        deck_lists=[LEGAL_DECK, LEGAL_DECK],
        deck_ids=[101, 102],
        max_decisions=200,
        max_ticks=10_000,
        seed=456,
        profile="fast",
        layout="i16_legal_ids_nometa",
    )
    assert buffers.layout == "i16_legal_ids_nometa"

    reset_step = buffers.reset()
    assert reset_step.legal_ids is not None
    assert reset_step.legal_offsets is not None
    assert getattr(reset_step, "legal_action_meta", None) is None

    context, context_offsets = buffers.legal_action_context_v1()
    assert context.dtype == np.int32
    assert context_offsets.shape == reset_step.legal_offsets.shape
    assert int(context_offsets[0]) == 0
    assert int(context_offsets[-1]) == int(reset_step.legal_offsets[-1])
    assert context.shape[0] == int(context_offsets[-1])

    logits = np.zeros((int(pool.envs_len), int(sim.ACTION_SPACE_SIZE)), dtype=np.float32)
    seeds = np.array([11, 12], dtype=np.uint64)
    step, actions, action_logp = buffers.step_sample_from_logits_with_logp(logits, seeds)

    assert step.obs.shape == (int(pool.envs_len), int(sim.OBS_LEN))
    assert actions.shape == (int(pool.envs_len),)
    assert action_logp.shape == (int(pool.envs_len),)
    assert np.all(np.isfinite(action_logp))
    for env_index, action in enumerate(actions):
        start = int(reset_step.legal_offsets[env_index])
        end = int(reset_step.legal_offsets[env_index + 1])
        assert int(action) in set(int(value) for value in reset_step.legal_ids[start:end])
