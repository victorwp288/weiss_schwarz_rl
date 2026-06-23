from __future__ import annotations

import pytest

from .rl_step_layout_contract_test_support import LAYOUTS, Layout, assert_layout_contract, make_pool, sim_module


@pytest.mark.parametrize("layout", LAYOUTS)
def test_rl_step_contract_smoke_covers_supported_layouts(layout: Layout) -> None:
    sim = sim_module()
    pool, buffers = make_pool(layout)
    num_envs = int(pool.envs_len)
    action_space = int(pool.action_space)

    assert num_envs == 2
    assert action_space == int(sim.ACTION_SPACE_SIZE)

    reset_step = sim.rl.reset_rl(pool, layout=layout)
    actions = assert_layout_contract(reset_step, layout=layout, buffers=buffers, action_space=action_space)
    assert actions.shape == (num_envs,)

    step_step = sim.rl.step_rl(pool, actions, layout=layout)
    next_actions = assert_layout_contract(step_step, layout=layout, buffers=buffers, action_space=action_space)
    assert next_actions.shape == (num_envs,)
