from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from weiss_rl.runtime.components.actor_unroll_finalization import (
    ActorUnrollFinalizationCallbacks,
    ActorUnrollFinalizationInputs,
    finalize_generic_actor_unroll,
)
from weiss_rl.runtime.components.collector_state import allocate_collector_unroll_state


class _TimingEnv:
    def __init__(self) -> None:
        self.drained = False

    def drain_timing_counters(self) -> dict[str, int]:
        self.drained = True
        return {"python_step": 11}


class _BootstrapModel:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def value_seat_aware(
        self,
        obs: torch.Tensor,
        actor: torch.Tensor,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        self.calls.append((obs, actor, hidden))
        return torch.as_tensor(self.values[: obs.shape[0]], dtype=torch.float32, device=obs.device)


def _actor(*, layout_name: str = "i16_legal_ids") -> SimpleNamespace:
    return SimpleNamespace(
        actor_id=7,
        next_unroll_seq=3,
        snapshot_version=19,
        layout_name=layout_name,
        seat_hidden=torch.ones((2, 3), dtype=torch.float32),
        current_batch=None,
        env=_TimingEnv(),
    )


def _batch(*, actors: np.ndarray | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        obs=np.asarray([[1.25, 2.25], [3.25, 4.25]], dtype=np.float64),
        actor=np.asarray([0, 1] if actors is None else actors, dtype=np.int32),
    )


def _state(*, layout_name: str = "i16_legal_ids") -> Any:
    state = allocate_collector_unroll_state(
        time_steps=1,
        batch_size=2,
        observation_dim=2,
        obs_dtype=np.float32,
        seat_hidden=torch.zeros((2, 3), dtype=torch.float32),
        trajectory_retention_enabled=True,
    )
    state.obs[0] = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    state.actions[0] = np.asarray([4, 5], dtype=np.uint16)
    state.rewards[0] = np.asarray([0.5, -0.25], dtype=np.float32)
    state.to_play_seat[0] = np.asarray([0, 1], dtype=np.int8)
    state.behavior_logp[0] = np.asarray([-0.1, -0.2], dtype=np.float32)
    state.values[0] = np.asarray([0.3, 0.4], dtype=np.float32)
    state.episode_seed[0] = np.asarray([101, 202], dtype=np.uint64)
    state.policy_train_mask[0] = np.asarray([True, False], dtype=np.bool_)
    state.opponent_context_index[0] = np.asarray([2, 3], dtype=np.int16)
    assert state.trajectory_retention_valid is not None
    state.trajectory_retention_valid[0] = np.asarray([True, False], dtype=np.bool_)
    if layout_name == "i16_legal_ids":
        state.packed_ids.append(np.asarray([4, 5], dtype=np.uint32))
        state.packed_offsets.append(np.asarray([1, 2], dtype=np.uint32))
        state.packed_meta.append(np.asarray([[40, 0], [50, 0]], dtype=np.uint16))
    else:
        state.mask_steps.append(
            np.asarray(
                [
                    [True, False, False],
                    [False, True, False],
                ],
                dtype=np.bool_,
            )
        )
    return state


def test_finalize_generic_actor_unroll_bootstraps_values_and_updates_actor_state() -> None:
    actor = _actor()
    batch = _batch()
    state = _state()
    model = _BootstrapModel(np.asarray([0.25, 0.75], dtype=np.float32))
    resolver_calls: list[Any] = []

    def resolve_model(resolved_actor: Any) -> _BootstrapModel:
        resolver_calls.append(resolved_actor)
        return model

    unroll = finalize_generic_actor_unroll(
        inputs=ActorUnrollFinalizationInputs(
            actor=actor,
            batch=batch,
            state=state,
            action_dim=8,
            started_at=time.perf_counter(),
            actor_behavior_values_required=True,
            actor_amp_enabled=False,
            bootstrap_device=torch.device("cpu"),
        ),
        callbacks=ActorUnrollFinalizationCallbacks(actor_inference_model=resolve_model),
    )

    assert resolver_calls == [actor]
    assert len(model.calls) == 1
    obs_arg, actor_arg, hidden_arg = model.calls[0]
    assert obs_arg.dtype == torch.float32
    assert actor_arg.dtype == torch.int64
    assert hidden_arg.shape == (2, 3)
    assert actor.current_batch is batch
    assert actor.next_unroll_seq == 4
    assert actor.env.drained is True
    assert unroll.counters is state.counters
    assert state.counters["simulator_python_step"] == 11
    assert state.counters["copied_bytes_estimate"] > 0
    assert state.counters["collect_actor_unroll_ms"] >= 0
    assert unroll.actor_id == 7
    assert unroll.unroll_seq == 3
    assert unroll.behavior_policy_version == 19
    assert unroll.bootstrap_obs.dtype == np.float32
    assert unroll.bootstrap_actor.dtype == np.int64
    np.testing.assert_allclose(unroll.bootstrap_value, np.asarray([0.25, 0.75], dtype=np.float32))
    assert unroll.legal_actions.ids is not None
    assert unroll.legal_actions.ids.tolist() == [4, 5]
    assert unroll.legal_actions.meta is not None
    assert unroll.legal_actions.meta[:, 0].tolist() == [40, 50]

    actor.seat_hidden.fill_(99.0)
    np.testing.assert_allclose(unroll.final_hidden_state, np.ones((2, 3), dtype=np.float32))


def test_finalize_generic_actor_unroll_skips_model_and_keeps_dense_mask_layout_when_values_not_required() -> None:
    actor = _actor(layout_name="dense_mask")
    batch = _batch(actors=np.asarray([0, 1], dtype=np.int32))
    state = _state(layout_name="dense_mask")
    resolver_calls: list[Any] = []

    def resolve_model(_: Any) -> Any:
        resolver_calls.append("unexpected")
        raise AssertionError("model resolver should not run when actor behavior values are disabled")

    unroll = finalize_generic_actor_unroll(
        inputs=ActorUnrollFinalizationInputs(
            actor=actor,
            batch=batch,
            state=state,
            action_dim=3,
            started_at=time.perf_counter(),
            actor_behavior_values_required=False,
            actor_amp_enabled=False,
            bootstrap_device=torch.device("cpu"),
        ),
        callbacks=ActorUnrollFinalizationCallbacks(actor_inference_model=resolve_model),
    )

    assert resolver_calls == []
    assert actor.current_batch is batch
    assert actor.next_unroll_seq == 4
    assert actor.env.drained is True
    np.testing.assert_allclose(unroll.bootstrap_value, np.zeros((2,), dtype=np.float32))
    assert unroll.legal_actions.mask is not None
    assert unroll.legal_actions.mask.shape == (1, 2, 3)
    assert unroll.legal_actions.mask.tolist() == [[[True, False, False], [False, True, False]]]
