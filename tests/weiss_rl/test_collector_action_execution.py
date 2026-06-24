from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from weiss_rl.diagnostics.probes.action_diagnostics import make_action_sequence_state
from weiss_rl.runtime.components.collection.collector_action_execution import (
    fused_step_mask_from_logits,
    fused_step_packed_from_logits_with_logp,
    record_mask_action_summary,
    record_packed_action_summary,
    sample_and_step_packed_from_logits,
    step_env_with_actions,
)


class _StepEnv:
    def __init__(self) -> None:
        self.actions: np.ndarray | None = None
        self.order: list[str] = []

    def step(self, actions: np.ndarray) -> SimpleNamespace:
        self.order.append("step")
        self.actions = np.array(actions, copy=True)
        return SimpleNamespace(main_move_action=np.array([True, False], dtype=np.bool_))


class _FusedPackedEnv:
    def __init__(self) -> None:
        self.logits: np.ndarray | None = None
        self.seeds: np.ndarray | None = None

    def step_sample_from_logits_with_logp(
        self,
        logits: np.ndarray,
        seeds: np.ndarray,
    ) -> tuple[SimpleNamespace, np.ndarray, np.ndarray]:
        self.logits = np.array(logits, copy=True)
        self.seeds = np.array(seeds, copy=True)
        return (
            SimpleNamespace(main_move_action=np.array([False, True], dtype=np.bool_)),
            np.array([1, 2], dtype=np.uint32),
            np.array([-0.25, -0.5], dtype=np.float64),
        )


class _FusedMaskEnv:
    def __init__(self) -> None:
        self.logits: np.ndarray | None = None
        self.seeds: np.ndarray | None = None

    def step_sample_from_logits(self, logits: np.ndarray, seeds: np.ndarray) -> tuple[SimpleNamespace, np.ndarray]:
        self.logits = np.array(logits, copy=True)
        self.seeds = np.array(seeds, copy=True)
        return SimpleNamespace(main_move_action=np.array([False, False], dtype=np.bool_)), np.array(
            [1, 0],
            dtype=np.uint32,
        )


def _counters() -> dict[str, int]:
    return {"actor_env_step_ms": 0, "actor_action_summary_ms": 0}


def test_step_env_with_actions_casts_uint32_and_runs_hook_inside_timed_step() -> None:
    env = _StepEnv()
    counters = _counters()

    next_batch = step_env_with_actions(
        env=env,
        actions=np.array([1, 2], dtype=np.int64),
        counters=counters,
        before_step=lambda: env.order.append("hook"),
    )

    assert next_batch.main_move_action.tolist() == [True, False]
    assert env.order == ["hook", "step"]
    assert env.actions is not None
    assert env.actions.dtype == np.uint32
    assert env.actions.tolist() == [1, 2]


def test_fused_packed_step_applies_temperature_and_normalizes_outputs() -> None:
    env = _FusedPackedEnv()
    counters = _counters()
    logits = np.array([[2.0, 4.0, 6.0], [1.0, 3.0, 5.0]], dtype=np.float32)

    executed = fused_step_packed_from_logits_with_logp(
        env=env,
        logits=logits,
        rng=np.random.default_rng(7),
        counters=counters,
        temperature=2.0,
    )

    assert env.logits is not None
    np.testing.assert_allclose(env.logits, logits / np.float32(2.0))
    assert env.seeds is not None
    assert env.seeds.dtype == np.int64
    assert env.seeds.shape == (2,)
    assert executed.actions.dtype == np.int64
    assert executed.actions.tolist() == [1, 2]
    assert executed.logp.dtype == np.float32
    np.testing.assert_allclose(executed.logp, np.array([-0.25, -0.5], dtype=np.float32))


def test_fused_mask_step_recomputes_behavior_logp_from_temperature_scaled_logits() -> None:
    env = _FusedMaskEnv()
    counters = _counters()
    logits = np.array([[1.0, 3.0], [5.0, 2.0]], dtype=np.float32)
    legal_mask = np.array([[True, True], [True, False]], dtype=np.bool_)

    executed = fused_step_mask_from_logits(
        env=env,
        logits=logits,
        legal_mask=legal_mask,
        rng=np.random.default_rng(9),
        counters=counters,
        pass_action_id=0,
        temperature=2.0,
    )

    assert env.logits is not None
    np.testing.assert_allclose(env.logits, logits / np.float32(2.0))
    assert executed.actions.tolist() == [1, 0]
    np.testing.assert_allclose(executed.logp, np.array([-0.31326166, 0.0], dtype=np.float32), rtol=1e-6)


def test_sample_and_step_packed_from_logits_steps_sampled_uint32_actions() -> None:
    env = _StepEnv()
    counters = _counters()

    executed = sample_and_step_packed_from_logits(
        env=env,
        logits=np.array([[0.0, 10.0, -1.0], [0.0, -1.0, 10.0]], dtype=np.float32),
        legal_ids=np.array([1, 2], dtype=np.uint32),
        legal_offsets=np.array([0, 1, 2], dtype=np.uint32),
        rng=np.random.default_rng(3),
        counters=counters,
        pass_action_id=0,
        temperature=1.0,
    )

    assert executed.actions.tolist() == [1, 2]
    assert env.actions is not None
    assert env.actions.dtype == np.uint32
    assert env.actions.tolist() == [1, 2]
    np.testing.assert_allclose(executed.logp, np.zeros((2,), dtype=np.float32))


def test_record_action_summary_helpers_preserve_ids_and_mask_semantics() -> None:
    counters = _counters()
    packed_state = make_action_sequence_state(2)
    next_batch = SimpleNamespace(main_move_action=np.array([False, True], dtype=np.bool_))

    record_packed_action_summary(
        counters=counters,
        state=packed_state,
        actions=np.array([51, 402], dtype=np.int64),
        legal_ids=np.array([7, 51, 402], dtype=np.int64),
        legal_offsets=np.array([0, 2, 3], dtype=np.int64),
        pass_action_id=51,
        next_batch=next_batch,
    )

    assert counters["total_actions"] == 2
    assert counters["pass_actions"] == 1
    assert counters["pass_with_nonpass_available"] == 1
    assert counters["main_move_actions"] == 1
    assert counters["max_consecutive_main_moves"] == 1

    mask_counters = _counters()
    mask_state = make_action_sequence_state(2)
    record_mask_action_summary(
        counters=mask_counters,
        state=mask_state,
        actions=np.array([51, 402], dtype=np.int64),
        legal_mask=np.array(
            [
                [True, False, True],
                [False, True, False],
            ],
            dtype=np.bool_,
        ),
        pass_action_id=0,
        next_batch=next_batch,
    )

    assert mask_counters["total_actions"] == 2
    assert mask_counters["main_move_actions"] == 1
    assert mask_counters["max_consecutive_main_moves"] == 1
