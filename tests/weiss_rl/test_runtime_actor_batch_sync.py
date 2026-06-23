from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import numpy.testing as npt
from weiss_rl.runtime import QueueRuntime


def test_sync_actor_batch_from_step_out_updates_env_last_batch() -> None:
    runtime = object.__new__(QueueRuntime)
    stale_batch = SimpleNamespace(
        ids_offsets=(
            np.array([51], dtype=np.uint32),
            np.array([0, 0, 1], dtype=np.uint32),
        )
    )
    actor = SimpleNamespace(
        current_batch=stale_batch,
        env=SimpleNamespace(_last_batch=stale_batch),
    )
    step_out = SimpleNamespace(
        obs=np.zeros((2, 3), dtype=np.float32),
        rewards=np.zeros((2,), dtype=np.float32),
        terminated=np.zeros((2,), dtype=np.bool_),
        truncated=np.zeros((2,), dtype=np.bool_),
        actor=np.array([0, 1], dtype=np.int64),
        decision_kind=np.zeros((2,), dtype=np.int32),
        decision_id=np.array([11, 12], dtype=np.uint32),
        engine_status=np.zeros((2,), dtype=np.uint32),
        decision_count=np.zeros((2,), dtype=np.uint32),
        tick_count=np.zeros((2,), dtype=np.uint32),
        no_progress_count=np.zeros((2,), dtype=np.uint32),
        episode_seed=np.array([101, 202], dtype=np.uint64),
        episode_key=np.array([301, 402], dtype=np.uint64),
        legal_ids=np.array([51, 474, 51, 102], dtype=np.uint32),
        legal_offsets=np.array([0, 2, 4], dtype=np.uint32),
    )
    pool = SimpleNamespace(action_space=512)

    batch = runtime._sync_actor_batch_from_step_out(
        actor=cast(Any, actor),
        step_out=step_out,
        pool=pool,
    )

    assert cast(Any, actor.current_batch) is batch
    assert actor.env._last_batch is batch
    assert actor.current_batch is not stale_batch
    npt.assert_array_equal(batch.ids_offsets[0], np.array([51, 474, 51, 102], dtype=np.uint32))
    npt.assert_array_equal(batch.ids_offsets[1], np.array([0, 2, 4], dtype=np.uint32))
