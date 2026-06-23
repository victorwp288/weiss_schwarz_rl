from __future__ import annotations

import numpy as np
import pytest
from weiss_rl.runtime.components.central_actor_step import execute_central_actor_step

from .central_actor_step_test_support import (
    StepEnv,
    central_step_actor,
    central_step_inputs,
    central_step_next_batch,
    central_step_packed_batch,
    central_step_runtime_context,
    central_step_state,
)


def test_execute_central_actor_step_requires_structured_packed_action_outputs() -> None:
    state = central_step_state()
    env = StepEnv(central_step_next_batch())
    actor = central_step_actor(env)

    with pytest.raises(ValueError, match="structured central packed execution requires action and logp steps"):
        execute_central_actor_step(
            actor=actor,
            batch=central_step_packed_batch(),
            state=state,
            inputs=central_step_inputs(
                step_index=0,
                structured_action_steps=None,
                structured_logp_steps=[np.asarray([-0.3, -0.4], dtype=np.float32)],
            ),
            runtime=central_step_runtime_context(),
        )
