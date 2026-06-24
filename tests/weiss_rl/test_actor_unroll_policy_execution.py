from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
from weiss_rl.diagnostics.probes.action_diagnostics import make_action_sequence_state
from weiss_rl.runtime.components.actors.actor_unroll_policy_execution import (
    ActorPolicyExecutionInputs,
    MaskActorPolicyCallbacks,
    MaskActorPolicyStorage,
    PackedActorPolicyCallbacks,
    PackedActorPolicyStorage,
    execute_generic_mask_actor_policy,
    execute_generic_packed_actor_policy,
)
from weiss_rl.runtime.components.batching.counters import collector_counter_template
from weiss_rl.runtime.components.teacher_labels import teacher_label_arrays


class _StepEnv:
    def __init__(self) -> None:
        self.actions: np.ndarray | None = None
        self.order: list[str] = []

    def step(self, actions: np.ndarray) -> SimpleNamespace:
        self.order.append("step")
        self.actions = np.asarray(actions).copy()
        return SimpleNamespace(
            reward=np.asarray([0.5, 1.0], dtype=np.float32),
            terminated=np.asarray([False, False], dtype=np.bool_),
            truncated=np.asarray([False, False], dtype=np.bool_),
            main_move_action=np.asarray([False, True], dtype=np.bool_),
        )


class _FusedMaskEnv:
    def __init__(self) -> None:
        self.logits: np.ndarray | None = None
        self.seeds: np.ndarray | None = None

    def step_sample_from_logits(self, logits: np.ndarray, seeds: np.ndarray) -> tuple[SimpleNamespace, np.ndarray]:
        self.logits = np.asarray(logits).copy()
        self.seeds = np.asarray(seeds).copy()
        return (
            SimpleNamespace(
                reward=np.asarray([0.0, 0.0], dtype=np.float32),
                terminated=np.asarray([False, False], dtype=np.bool_),
                truncated=np.asarray([False, False], dtype=np.bool_),
                main_move_action=np.asarray([False, False], dtype=np.bool_),
            ),
            np.asarray([1, 0], dtype=np.uint32),
        )


def _config() -> SimpleNamespace:
    return SimpleNamespace(pass_action_id=0, actor_sampling_temperature=2.0)


def _inputs(*, env: Any, batch: Any, use_fused: bool, logits: np.ndarray | None = None) -> ActorPolicyExecutionInputs:
    return ActorPolicyExecutionInputs(
        actor=SimpleNamespace(env=env, rng=np.random.default_rng(5)),
        batch=batch,
        obs_step=np.ones((2, 3), dtype=np.float32),
        actor_step=np.asarray([0, 1], dtype=np.int64),
        focal_rows=np.asarray([True, False], dtype=np.bool_),
        value_step=np.zeros((2,), dtype=np.float32),
        action_step=np.zeros((2,), dtype=np.int64),
        logp_step=np.zeros((2,), dtype=np.float32),
        logits_step=np.zeros((2, 3), dtype=np.float32) if logits is None else logits,
        config=_config(),
        counters=collector_counter_template(),
        action_sequence_state=make_action_sequence_state(2),
        use_simulator_fused_logits_step=use_fused,
    )


def test_execute_generic_packed_actor_policy_preserves_nonfused_step_and_debug_validation_order() -> None:
    env = _StepEnv()
    batch = SimpleNamespace(
        ids_offsets=(
            np.asarray([0, 2, 1, 2], dtype=np.uint32),
            np.asarray([0, 2, 4], dtype=np.uint32),
        ),
        legal_action_meta=np.asarray([[0], [2], [1], [2]], dtype=np.uint16),
        decision_kind=np.asarray([1, 1], dtype=np.int32),
    )
    inputs = _inputs(env=env, batch=batch, use_fused=False)
    labels = teacher_label_arrays(2)
    fill_calls: list[dict[str, Any]] = []
    validate_calls: list[dict[str, Any]] = []

    def fill_policy_outputs_ids(**kwargs: Any) -> None:
        fill_calls.append(kwargs)
        kwargs["values_out"][:] = np.asarray([0.25, 0.75], dtype=np.float32)
        kwargs["actions_out"][:] = np.asarray([0, 2], dtype=np.int64)
        kwargs["logp_out"][:] = np.asarray([-0.3, -0.4], dtype=np.float32)

    def validate(**kwargs: Any) -> None:
        env.order.append("validate")
        validate_calls.append(kwargs)

    result = execute_generic_packed_actor_policy(
        inputs=inputs,
        callbacks=PackedActorPolicyCallbacks(
            fill_policy_outputs_ids=fill_policy_outputs_ids,
            maybe_debug_validate_env_step_packed_actions=validate,
            ensure_legal_action_meta=lambda _ids, meta: meta,
            teacher_labels_from_ids=lambda **_: labels,
        ),
        storage=PackedActorPolicyStorage(
            packed_ids=[],
            packed_meta=[],
            packed_offsets=[np.asarray([0], dtype=np.uint32)],
        ),
    )

    assert fill_calls and "sample_actions" not in fill_calls[0]
    assert fill_calls[0]["logits_out"] is None
    assert fill_calls[0]["actions_out"] is inputs.action_step
    assert validate_calls[0]["source_label"] == "collect:packed"
    assert validate_calls[0]["actions"].tolist() == [0, 2]
    assert env.order == ["validate", "step"]
    assert env.actions is not None
    assert env.actions.dtype == np.uint32
    assert result.next_batch.reward.tolist() == [0.5, 1.0]
    assert result.action_step is inputs.action_step
    assert result.logp_step is inputs.logp_step
    assert result.teacher_labels is labels
    assert result.reward_legal_ids is not None
    assert result.reward_legal_ids.tolist() == [0, 2, 1, 2]
    assert result.reward_legal_mask is None
    assert inputs.counters["packed_candidate_count"] == 4
    assert inputs.counters["total_actions"] == 2
    assert inputs.counters["main_move_actions"] == 1


def test_execute_generic_mask_actor_policy_preserves_fused_logits_step_and_reward_mask() -> None:
    env = _FusedMaskEnv()
    batch = SimpleNamespace(
        mask=np.asarray([[True, True, False], [True, False, False]], dtype=np.bool_),
        decision_kind=np.asarray([1, 2], dtype=np.int32),
    )
    logits = np.asarray([[1.0, 3.0, -9.0], [5.0, -9.0, -9.0]], dtype=np.float32)
    inputs = _inputs(env=env, batch=batch, use_fused=True, logits=logits)
    labels = teacher_label_arrays(2)
    fill_calls: list[dict[str, Any]] = []
    mask_steps: list[np.ndarray] = []

    def fill_policy_outputs_mask(**kwargs: Any) -> None:
        fill_calls.append(kwargs)
        kwargs["logits_out"][:] = logits
        kwargs["values_out"][:] = np.asarray([1.25, 1.75], dtype=np.float32)

    result = execute_generic_mask_actor_policy(
        inputs=inputs,
        callbacks=MaskActorPolicyCallbacks(
            fill_policy_outputs_mask=fill_policy_outputs_mask,
            teacher_labels_from_mask=lambda **_: labels,
        ),
        storage=MaskActorPolicyStorage(mask_steps=mask_steps),
    )

    assert fill_calls and fill_calls[0]["sample_actions"] is False
    assert fill_calls[0]["legal_mask"].tolist() == [[True, True, False], [True, False, False]]
    assert env.logits is not None
    np.testing.assert_allclose(env.logits, logits / np.float32(2.0))
    assert result.action_step.tolist() == [1, 0]
    np.testing.assert_allclose(result.logp_step, np.asarray([-0.31326166, 0.0], dtype=np.float32), rtol=1e-6)
    assert result.teacher_labels is labels
    assert result.reward_legal_mask is not None
    assert result.reward_legal_mask.tolist() == [[True, True, False], [True, False, False]]
    assert result.reward_legal_ids is None
    assert mask_steps[0].tolist() == [[True, True, False], [True, False, False]]
    assert inputs.counters["total_actions"] == 2
