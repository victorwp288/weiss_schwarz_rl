from __future__ import annotations

import queue
from collections import deque
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import numpy.testing as npt
import pytest
import torch

import weiss_rl.model as model_module
from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.envs.decision_env import DecisionBoundaryBatch
from weiss_rl.eval.policies.set import (
    HEURISTIC_PUBLIC_POLICY_ID,
)
from weiss_rl.runtime import (
    _MIRROR_OPPONENT_POLICY_ID,
    _NOLEAGUE_BASELINE_POLICY_ID,
    QueueRuntime,
    QueueRuntimeConfig,
    RuntimeUnroll,
    _concat_optional_time_major_field,
    _handle_collector_commands,
    _maybe_compile_runtime_actor_model,
    build_runtime_config,
)


def _make_runtime_unroll(
    *,
    actor_id: int,
    unroll_seq: int,
    behavior_policy_version: int,
    counters: dict[str, int] | None = None,
) -> RuntimeUnroll:
    return RuntimeUnroll(
        actor_id=actor_id,
        unroll_seq=unroll_seq,
        behavior_policy_version=behavior_policy_version,
        unroll_hash=f"{actor_id}:{unroll_seq}:{behavior_policy_version}",
        obs=np.zeros((1, 1, 1), dtype=np.float32),
        actions=np.zeros((1, 1), dtype=np.int64),
        rewards=np.zeros((1, 1), dtype=np.float32),
        terminated=np.zeros((1, 1), dtype=np.bool_),
        truncated=np.zeros((1, 1), dtype=np.bool_),
        to_play_seat=np.zeros((1, 1), dtype=np.int64),
        behavior_logp=np.zeros((1, 1), dtype=np.float32),
        values=np.zeros((1, 1), dtype=np.float32),
        legal_actions=LegalActionBatch.from_mask(np.ones((1, 1, 1), dtype=np.bool_)),
        bootstrap_obs=np.zeros((1, 1), dtype=np.float32),
        bootstrap_actor=np.zeros((1,), dtype=np.int64),
        bootstrap_value=np.zeros((1,), dtype=np.float32),
        initial_hidden_state=np.zeros((1, 1), dtype=np.float32),
        final_hidden_state=np.zeros((1, 1), dtype=np.float32),
        episode_seed=np.zeros((1, 1), dtype=np.uint64),
        policy_train_mask=np.ones((1, 1), dtype=np.bool_),
        behavior_logits=None,
        counters=counters,
    )


class _CompileStructuredActorModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(()))
        self.supports_legal_candidate_scoring = True
        self.compile_calls = 0
        self.compile_mode: str | None = None

    def enable_trunk_compile(self, *, mode: str = "reduce-overhead") -> _CompileStructuredActorModel:
        self.compile_calls += 1
        self.compile_mode = mode
        return self


def test_maybe_compile_runtime_actor_model_uses_structured_trunk_compile_hook() -> None:
    model = _CompileStructuredActorModel()

    compiled = _maybe_compile_runtime_actor_model(cast(Any, model), enabled=True)

    assert compiled is model
    assert model.compile_calls == 1
    assert model.compile_mode == "reduce-overhead"


class _FactorizedRuntimeActorModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(()))
        self.supports_legal_candidate_scoring = True
        self.supports_factorized_legal_policy = True
        self.factorized_calls = 0
        self.packed_calls = 0

    def sample_factorized_packed_seat_aware(
        self,
        obs: torch.Tensor,
        acting_seat: torch.Tensor,
        seat_hidden_state: torch.Tensor | None = None,
        *,
        legal_actions: LegalActionBatch,
        sample_seeds: torch.Tensor,
        pass_action_id: int,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del acting_seat, legal_actions, sample_seeds, pass_action_id, temperature
        self.factorized_calls += 1
        batch = int(obs.shape[0])
        next_hidden = (
            torch.zeros((batch, 2, 3), dtype=obs.dtype, device=obs.device)
            if seat_hidden_state is None
            else seat_hidden_state + 1.0
        )
        return (
            torch.full((batch,), 7, dtype=torch.long, device=obs.device),
            torch.full((batch,), -0.25, dtype=obs.dtype, device=obs.device),
            torch.full((batch,), 0.5, dtype=obs.dtype, device=obs.device),
            next_hidden,
        )

    def sample_packed_seat_aware(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del args, kwargs
        self.packed_calls += 1
        raise AssertionError("factorized runtime path should bypass packed sampler")


def test_apply_policy_rows_ids_prefers_factorized_structured_sampler() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._device = torch.device("cpu")
    runtime_any._actor_amp_enabled = False
    runtime_any.config = SimpleNamespace(pass_action_id=8)

    model = _FactorizedRuntimeActorModel()
    hidden_state = torch.zeros((2, 2, 3), dtype=torch.float32)
    row_indices = np.asarray([0, 1], dtype=np.int64)
    obs_step = np.zeros((2, 4), dtype=np.float32)
    actor_step = np.asarray([0, 1], dtype=np.int64)
    legal_ids = np.asarray([0, 7, 8, 1, 7, 8], dtype=np.uint32)
    legal_offsets = np.asarray([0, 3, 6], dtype=np.uint32)
    legal_meta = np.asarray(
        [
            [0, 0, 0, np.iinfo(np.uint16).max],
            [1, 0, 0, np.iinfo(np.uint16).max],
            [2, np.iinfo(np.uint16).max, np.iinfo(np.uint16).max, np.iinfo(np.uint16).max],
            [0, 1, 0, np.iinfo(np.uint16).max],
            [1, 1, 0, np.iinfo(np.uint16).max],
            [2, np.iinfo(np.uint16).max, np.iinfo(np.uint16).max, np.iinfo(np.uint16).max],
        ],
        dtype=np.uint16,
    )
    values_out = np.zeros((2,), dtype=np.float32)
    actions_out = np.zeros((2,), dtype=np.int64)
    logp_out = np.zeros((2,), dtype=np.float32)

    QueueRuntime._apply_policy_rows_ids(
        runtime,
        model=model,
        hidden_state=hidden_state,
        row_indices=row_indices,
        obs_step=obs_step,
        actor_step=actor_step,
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        legal_action_meta=legal_meta,
        logits_out=None,
        values_out=values_out,
        actions_out=actions_out,
        logp_out=logp_out,
        rng=np.random.default_rng(123),
        sample_actions=True,
    )

    assert model.factorized_calls == 1
    assert model.packed_calls == 0
    assert actions_out.tolist() == [7, 7]
    assert np.allclose(logp_out, -0.25)
    assert np.allclose(values_out, 0.5)
    assert torch.allclose(hidden_state, torch.ones_like(hidden_state))


class _ArgmaxRuntimeActorModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(()))
        self.supports_legal_candidate_scoring = True
        self.supports_factorized_legal_policy = True
        self.sample_calls = 0

    def sample_factorized_packed_seat_aware(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, ...]:
        del args, kwargs
        self.sample_calls += 1
        raise AssertionError("argmax opponent rows must not call the sampler")

    def forward_seat_aware(
        self,
        obs: torch.Tensor,
        acting_seat: torch.Tensor,
        seat_hidden_state: torch.Tensor | None = None,
        *,
        legal_actions: LegalActionBatch | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del obs, acting_seat, legal_actions
        logits = torch.tensor(
            [
                [0.0, 1.0, -2.0, 7.0, 3.0, -1.0, 2.0, 9.0, 5.0],
                [0.0, 8.0, -2.0, 7.0, 3.0, -1.0, 2.0, 6.0, 5.0],
            ],
            dtype=torch.float32,
            device=self.bias.device,
        )
        value = torch.full((2,), 0.5, dtype=torch.float32, device=self.bias.device)
        next_hidden = (
            torch.zeros((2, 2, 3), dtype=torch.float32, device=self.bias.device)
            if seat_hidden_state is None
            else seat_hidden_state + 1.0
        )
        return logits, value, next_hidden


def test_apply_policy_rows_ids_argmax_selection_uses_legal_argmax_without_sampler() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._device = torch.device("cpu")
    runtime_any._actor_amp_enabled = False
    runtime_any.config = SimpleNamespace(pass_action_id=8)

    model = _ArgmaxRuntimeActorModel()
    hidden_state = torch.zeros((2, 2, 3), dtype=torch.float32)
    row_indices = np.asarray([0, 1], dtype=np.int64)
    obs_step = np.zeros((2, 4), dtype=np.float32)
    actor_step = np.asarray([0, 1], dtype=np.int64)
    legal_ids = np.asarray([0, 7, 8, 1, 7, 8], dtype=np.uint32)
    legal_offsets = np.asarray([0, 3, 6], dtype=np.uint32)
    values_out = np.zeros((2,), dtype=np.float32)
    actions_out = np.zeros((2,), dtype=np.int64)
    logp_out = np.full((2,), -99.0, dtype=np.float32)

    QueueRuntime._apply_policy_rows_ids(
        runtime,
        model=model,
        hidden_state=hidden_state,
        row_indices=row_indices,
        obs_step=obs_step,
        actor_step=actor_step,
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        legal_action_meta=None,
        logits_out=None,
        values_out=values_out,
        actions_out=actions_out,
        logp_out=logp_out,
        rng=np.random.default_rng(123),
        sample_actions=True,
        action_selection="argmax",
    )

    assert model.sample_calls == 0
    assert actions_out.tolist() == [7, 1]
    assert np.allclose(logp_out, 0.0)
    assert np.allclose(values_out, 0.5)
    assert torch.allclose(hidden_state, torch.ones_like(hidden_state))


def test_apply_policy_rows_ids_argmax_selection_writes_deterministic_logits_for_fused_step() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._device = torch.device("cpu")
    runtime_any._actor_amp_enabled = False
    runtime_any.config = SimpleNamespace(pass_action_id=8)

    model = _ArgmaxRuntimeActorModel()
    hidden_state = torch.zeros((2, 2, 3), dtype=torch.float32)
    row_indices = np.asarray([0, 1], dtype=np.int64)
    obs_step = np.zeros((2, 4), dtype=np.float32)
    actor_step = np.asarray([0, 1], dtype=np.int64)
    legal_ids = np.asarray([0, 7, 8, 1, 7, 8], dtype=np.uint32)
    legal_offsets = np.asarray([0, 3, 6], dtype=np.uint32)
    values_out = np.zeros((2,), dtype=np.float32)
    logits_out = np.zeros((2, 9), dtype=np.float32)

    QueueRuntime._apply_policy_rows_ids(
        runtime,
        model=model,
        hidden_state=hidden_state,
        row_indices=row_indices,
        obs_step=obs_step,
        actor_step=actor_step,
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        legal_action_meta=None,
        logits_out=logits_out,
        values_out=values_out,
        actions_out=None,
        logp_out=None,
        rng=np.random.default_rng(123),
        sample_actions=False,
        action_selection="argmax",
    )

    assert model.sample_calls == 0
    assert logits_out[0, 7] == pytest.approx(0.0)
    assert logits_out[0, 0] == pytest.approx(-100.0)
    assert logits_out[0, 8] == pytest.approx(-100.0)
    assert logits_out[1, 1] == pytest.approx(0.0)
    assert logits_out[1, 7] == pytest.approx(-100.0)
    assert logits_out[1, 8] == pytest.approx(-100.0)
    assert np.all(logits_out[:, [2, 3, 4, 5, 6]] < -1.0e8)
    assert np.allclose(values_out, 0.5)
    assert torch.allclose(hidden_state, torch.ones_like(hidden_state))


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


class _HeuristicRuntimeActorModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(()))
        self.forward_legal_flags: list[bool] = []

    def forward_seat_aware(
        self,
        obs: torch.Tensor,
        acting_seat: torch.Tensor,
        seat_hidden_state: torch.Tensor | None = None,
        *,
        legal_actions: LegalActionBatch | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del acting_seat
        self.forward_legal_flags.append(legal_actions is not None)
        batch = int(obs.shape[0])
        next_hidden = (
            torch.zeros((batch, 2, 3), dtype=obs.dtype, device=obs.device)
            if seat_hidden_state is None
            else seat_hidden_state + 1.0
        )
        return (
            torch.zeros((batch, 8), dtype=obs.dtype, device=obs.device),
            torch.full((batch,), 0.25, dtype=obs.dtype, device=obs.device),
            next_hidden,
        )


class _HeuristicAdvanceOnlyRuntimeActorModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(()))
        self.advance_calls = 0
        self.forward_calls = 0

    def advance_seat_hidden(
        self,
        obs: torch.Tensor,
        acting_seat: torch.Tensor,
        seat_hidden_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del obs, acting_seat
        self.advance_calls += 1
        if seat_hidden_state is None:
            return torch.zeros((0, 2, 3), dtype=self.bias.dtype, device=self.bias.device)
        return seat_hidden_state + 1.0

    def forward_seat_aware(
        self,
        obs: torch.Tensor,
        acting_seat: torch.Tensor,
        seat_hidden_state: torch.Tensor | None = None,
        *,
        legal_actions: LegalActionBatch | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del obs, acting_seat, seat_hidden_state, legal_actions
        self.forward_calls += 1
        raise AssertionError(
            "heuristic actor rows should use advance_seat_hidden when behavior values are not required"
        )


def test_fill_policy_outputs_ids_can_use_heuristic_actor_backend_for_focal_rows() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._device = torch.device("cpu")
    runtime_any._actor_amp_enabled = False
    runtime_any._actor_policy_backend = "heuristic_public"
    runtime_any._actor_heuristic_fraction = 1.0
    runtime_any._teacher_policy = object()
    runtime_any.config = SimpleNamespace(pass_action_id=8)
    runtime_any._heuristic_public_actions_from_ids = lambda **_kwargs: np.array([7, 8], dtype=np.int64)

    model = _HeuristicRuntimeActorModel()
    actor = cast(
        Any,
        SimpleNamespace(
            model=model,
            compiled_model=None,
            seat_hidden=torch.zeros((2, 2, 3), dtype=torch.float32),
            focal_seat_by_env=np.array([0, 1], dtype=np.int64),
            opponent_policy_id_by_env=np.array(["mirror", "mirror"], dtype=object),
            rng=np.random.default_rng(123),
        ),
    )
    obs_step = np.zeros((2, 4), dtype=np.float32)
    actor_step = np.array([0, 1], dtype=np.int64)
    legal_ids = np.array([0, 7, 8, 1, 7, 8], dtype=np.uint32)
    legal_offsets = np.array([0, 3, 6], dtype=np.uint32)
    legal_meta = np.array(
        [
            [0, 0, 0, np.iinfo(np.uint16).max],
            [1, 0, 0, np.iinfo(np.uint16).max],
            [2, np.iinfo(np.uint16).max, np.iinfo(np.uint16).max, np.iinfo(np.uint16).max],
            [0, 1, 0, np.iinfo(np.uint16).max],
            [1, 1, 0, np.iinfo(np.uint16).max],
            [2, np.iinfo(np.uint16).max, np.iinfo(np.uint16).max, np.iinfo(np.uint16).max],
        ],
        dtype=np.uint16,
    )
    values_out = np.zeros((2,), dtype=np.float32)
    actions_out = np.zeros((2,), dtype=np.int64)
    logp_out = np.zeros((2,), dtype=np.float32)

    QueueRuntime._fill_policy_outputs_ids(
        runtime,
        actor=actor,
        obs_step=obs_step,
        actor_step=actor_step,
        focal_rows=np.array([True, True], dtype=np.bool_),
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        legal_action_meta=legal_meta,
        logits_out=None,
        values_out=values_out,
        actions_out=actions_out,
        logp_out=logp_out,
        rng=np.random.default_rng(321),
        sample_actions=True,
    )

    assert model.forward_legal_flags == [False]
    assert np.array_equal(actions_out, np.array([7, 8], dtype=np.int64))
    assert np.allclose(logp_out, 0.0)
    assert np.allclose(values_out, 0.25)
    assert torch.allclose(actor.seat_hidden, torch.ones_like(actor.seat_hidden))


def test_fill_policy_outputs_ids_heuristic_actor_backend_can_skip_behavior_values() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._device = torch.device("cpu")
    runtime_any._actor_amp_enabled = False
    runtime_any._actor_policy_backend = "heuristic_public"
    runtime_any._actor_heuristic_fraction = 1.0
    runtime_any._actor_behavior_values_required = False
    runtime_any._teacher_policy = object()
    runtime_any.config = SimpleNamespace(pass_action_id=8)
    runtime_any._heuristic_public_actions_from_ids = lambda **_kwargs: np.array([7, 8], dtype=np.int64)

    model = _HeuristicAdvanceOnlyRuntimeActorModel()
    actor = cast(
        Any,
        SimpleNamespace(
            model=model,
            compiled_model=None,
            seat_hidden=torch.zeros((2, 2, 3), dtype=torch.float32),
            focal_seat_by_env=np.array([0, 1], dtype=np.int64),
            opponent_policy_id_by_env=np.array(["mirror", "mirror"], dtype=object),
            rng=np.random.default_rng(123),
        ),
    )
    obs_step = np.zeros((2, 4), dtype=np.float32)
    actor_step = np.array([0, 1], dtype=np.int64)
    legal_ids = np.array([0, 7, 8, 1, 7, 8], dtype=np.uint32)
    legal_offsets = np.array([0, 3, 6], dtype=np.uint32)
    legal_meta = np.array(
        [
            [0, 0, 0, np.iinfo(np.uint16).max],
            [1, 0, 0, np.iinfo(np.uint16).max],
            [2, np.iinfo(np.uint16).max, np.iinfo(np.uint16).max, np.iinfo(np.uint16).max],
            [0, 1, 0, np.iinfo(np.uint16).max],
            [1, 1, 0, np.iinfo(np.uint16).max],
            [2, np.iinfo(np.uint16).max, np.iinfo(np.uint16).max, np.iinfo(np.uint16).max],
        ],
        dtype=np.uint16,
    )
    values_out = np.full((2,), 9.0, dtype=np.float32)
    actions_out = np.zeros((2,), dtype=np.int64)
    logp_out = np.zeros((2,), dtype=np.float32)

    QueueRuntime._fill_policy_outputs_ids(
        runtime,
        actor=actor,
        obs_step=obs_step,
        actor_step=actor_step,
        focal_rows=np.array([True, True], dtype=np.bool_),
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        legal_action_meta=legal_meta,
        logits_out=None,
        values_out=values_out,
        actions_out=actions_out,
        logp_out=logp_out,
        rng=np.random.default_rng(321),
        sample_actions=True,
    )

    assert model.advance_calls == 1
    assert model.forward_calls == 0
    assert np.array_equal(actions_out, np.array([7, 8], dtype=np.int64))
    assert np.allclose(logp_out, 0.0)
    assert np.allclose(values_out, 0.0)
    assert torch.allclose(actor.seat_hidden, torch.ones_like(actor.seat_hidden))


def test_fill_policy_outputs_ids_heuristic_actor_backend_can_skip_hidden_tracking() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._device = torch.device("cpu")
    runtime_any._actor_amp_enabled = False
    runtime_any._actor_policy_backend = "heuristic_public"
    runtime_any._actor_heuristic_fraction = 1.0
    runtime_any._actor_behavior_values_required = False
    runtime_any._heuristic_actor_hidden_state_tracking = False
    runtime_any._teacher_policy = object()
    runtime_any.config = SimpleNamespace(pass_action_id=8)
    runtime_any._heuristic_public_actions_from_ids = lambda **_kwargs: np.array([7, 8], dtype=np.int64)

    model = _HeuristicAdvanceOnlyRuntimeActorModel()
    actor = cast(
        Any,
        SimpleNamespace(
            model=model,
            compiled_model=None,
            seat_hidden=torch.zeros((2, 2, 3), dtype=torch.float32),
            focal_seat_by_env=np.array([0, 1], dtype=np.int64),
            opponent_policy_id_by_env=np.array(["mirror", "mirror"], dtype=object),
            rng=np.random.default_rng(123),
        ),
    )
    obs_step = np.zeros((2, 4), dtype=np.float32)
    actor_step = np.array([0, 1], dtype=np.int64)
    legal_ids = np.array([0, 7, 8, 1, 7, 8], dtype=np.uint32)
    legal_offsets = np.array([0, 3, 6], dtype=np.uint32)
    legal_meta = np.array(
        [
            [0, 0, 0, np.iinfo(np.uint16).max],
            [1, 0, 0, np.iinfo(np.uint16).max],
            [2, np.iinfo(np.uint16).max, np.iinfo(np.uint16).max, np.iinfo(np.uint16).max],
            [0, 1, 0, np.iinfo(np.uint16).max],
            [1, 1, 0, np.iinfo(np.uint16).max],
            [2, np.iinfo(np.uint16).max, np.iinfo(np.uint16).max, np.iinfo(np.uint16).max],
        ],
        dtype=np.uint16,
    )
    values_out = np.full((2,), 9.0, dtype=np.float32)
    actions_out = np.zeros((2,), dtype=np.int64)
    logp_out = np.zeros((2,), dtype=np.float32)

    QueueRuntime._fill_policy_outputs_ids(
        runtime,
        actor=actor,
        obs_step=obs_step,
        actor_step=actor_step,
        focal_rows=np.array([True, True], dtype=np.bool_),
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        legal_action_meta=legal_meta,
        logits_out=None,
        values_out=values_out,
        actions_out=actions_out,
        logp_out=logp_out,
        rng=np.random.default_rng(321),
        sample_actions=True,
    )

    assert model.advance_calls == 0
    assert model.forward_calls == 0
    assert np.array_equal(actions_out, np.array([7, 8], dtype=np.int64))
    assert np.allclose(logp_out, 0.0)
    assert np.allclose(values_out, 0.0)
    assert torch.allclose(actor.seat_hidden, torch.zeros_like(actor.seat_hidden))


def test_collect_all_heuristic_ids_fast_requires_all_heuristic_linux_frontier_conditions() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._actor_policy_backend = "heuristic_public"
    runtime_any._actor_heuristic_fraction = 1.0
    runtime_any._fixed_opponent_backend = "simulator_native"
    runtime_any._teacher_policy = object()
    runtime_any._league_config = SimpleNamespace(sampling=SimpleNamespace(heuristic_public_mix_fraction=1.0))
    runtime_any._fixed_opponent_policy_is_active = lambda policy_id: bool(str(policy_id).strip())

    actor = cast(
        Any,
        SimpleNamespace(
            layout_name="i16_legal_ids",
            env=SimpleNamespace(
                pool=SimpleNamespace(choose_heuristic_public_actions_into=lambda *args, **kwargs: None)
            ),
            opponent_policy_id_by_env=np.array(
                [HEURISTIC_PUBLIC_POLICY_ID, HEURISTIC_PUBLIC_POLICY_ID],
                dtype=object,
            ),
            fixed_opponent_policy_id_by_env=np.array(["", HEURISTIC_PUBLIC_POLICY_ID], dtype=object),
        ),
    )

    assert QueueRuntime._can_collect_all_heuristic_ids_fast(runtime, actor) is True


def test_collect_all_heuristic_ids_fast_rejects_nonheuristic_opponent_assignments() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._actor_policy_backend = "heuristic_public"
    runtime_any._actor_heuristic_fraction = 1.0
    runtime_any._fixed_opponent_backend = "simulator_native"
    runtime_any._teacher_policy = object()
    runtime_any._league_config = SimpleNamespace(sampling=SimpleNamespace(heuristic_public_mix_fraction=1.0))
    runtime_any._fixed_opponent_policy_is_active = lambda policy_id: bool(str(policy_id).strip())

    actor = cast(
        Any,
        SimpleNamespace(
            layout_name="i16_legal_ids",
            env=SimpleNamespace(
                pool=SimpleNamespace(choose_heuristic_public_actions_into=lambda *args, **kwargs: None)
            ),
            opponent_policy_id_by_env=np.array(
                [HEURISTIC_PUBLIC_POLICY_ID, "latest_policy_mirror"],
                dtype=object,
            ),
            fixed_opponent_policy_id_by_env=np.array(["", ""], dtype=object),
        ),
    )

    assert QueueRuntime._can_collect_all_heuristic_ids_fast(runtime, actor) is False


def test_collect_all_heuristic_ids_fast_snapshots_reused_step_out_before_step() -> None:
    class ReusingStepOut:
        def __init__(self) -> None:
            self.step_index = 0
            self.obs = np.zeros((2, 4), dtype=np.float32)
            self.actor = np.zeros((2,), dtype=np.int8)
            self.decision_kind = np.zeros((2,), dtype=np.int32)
            self.decision_id = np.zeros((2,), dtype=np.uint32)
            self.legal_ids = np.zeros((4,), dtype=np.uint32)
            self.legal_offsets = np.array([0, 2, 4], dtype=np.uint32)
            self.rewards = np.zeros((2,), dtype=np.float32)
            self.terminated = np.zeros((2,), dtype=np.bool_)
            self.truncated = np.zeros((2,), dtype=np.bool_)
            self.engine_status = np.zeros((2,), dtype=np.int32)
            self.main_move_action = np.zeros((2,), dtype=np.bool_)
            self.main_pass_action = np.zeros((2,), dtype=np.bool_)
            self.fill(0)

        def fill(self, step_index: int) -> None:
            self.step_index = step_index
            self.obs[:] = np.float32(10 + step_index * 7) + np.arange(2, dtype=np.float32)[:, None]
            self.actor[:] = ((np.arange(2, dtype=np.int32) + step_index) % 2).astype(np.int8)
            self.decision_kind[:] = np.int32(20 + step_index)
            self.legal_ids[:] = np.array(
                [
                    (step_index * 4) % 64,
                    (step_index * 4 + 1) % 64,
                    (step_index * 4 + 2) % 64,
                    (step_index * 4 + 3) % 64,
                ],
                dtype=np.uint32,
            )
            self.rewards[:] = np.float32(step_index)
            self.terminated[:] = False
            self.truncated[:] = False
            self.engine_status[:] = 0

    class ReusingPool:
        def __init__(self, step_out: ReusingStepOut) -> None:
            self.step_out = step_out

        def step_into_i16_legal_ids(self, actions: np.ndarray, step_out: ReusingStepOut) -> None:
            assert step_out is self.step_out
            assert actions.shape == (2,)
            step_out.fill(step_out.step_index + 1)

        def reset_done_into_i16_legal_ids(self, done: np.ndarray, step_out: ReusingStepOut) -> None:
            assert not np.any(done)
            assert step_out is self.step_out

        def episode_seed_batch(self) -> np.ndarray:
            return np.array(
                [30_000 + self.step_out.step_index * 10, 30_001 + self.step_out.step_index * 10], dtype=np.uint64
            )

    step_out = ReusingStepOut()
    pool = ReusingPool(step_out)
    env = SimpleNamespace(
        pool=pool,
        _step_out=step_out,
        _record_python_timing=lambda *args, **kwargs: None,
        _handle_engine_status=lambda *args, **kwargs: None,
    )
    initial_batch = SimpleNamespace(
        obs=step_out.obs,
        actor=step_out.actor,
        decision_kind=step_out.decision_kind,
        ids_offsets=(step_out.legal_ids, step_out.legal_offsets),
        legal_action_meta=None,
    )
    actor = SimpleNamespace(
        actor_id=3,
        next_unroll_seq=0,
        snapshot_version=0,
        current_batch=initial_batch,
        env=env,
        focal_seat_by_env=np.array([0, 1], dtype=np.int8),
        seat_hidden=torch.zeros((2, 1, 2), dtype=torch.float32),
        opponent_hidden=torch.zeros((2, 1, 2), dtype=torch.float32),
    )

    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any.config = SimpleNamespace(unroll_length=2, envs_per_actor=2, pass_action_id=63)
    runtime_any.observation_dim = 4
    runtime_any.action_dim = 64
    runtime_any._actor_behavior_values_required = False
    runtime_any._teacher_policy = object()
    runtime_any._teacher_guidance_active_for_collection = lambda: False
    runtime_any._policy_train_mask_for_actor = lambda *, actor, focal_rows, **_kwargs: np.asarray(
        focal_rows, dtype=np.bool_
    )
    runtime_any._ensure_legal_action_meta = lambda legal_ids, meta: None
    runtime_any._should_track_heuristic_actor_hidden_state = lambda: False
    runtime_any._heuristic_public_actions_from_ids = lambda **kwargs: np.array(
        [
            int(kwargs["legal_ids"][int(kwargs["legal_offsets"][row])])
            for row in np.asarray(kwargs["row_indices"], dtype=np.int64)
        ],
        dtype=np.int64,
    )
    runtime_any._maybe_debug_validate_sampled_packed_actions = lambda **kwargs: None
    runtime_any._sync_actor_batch_from_step_out = lambda *, actor, step_out, pool: SimpleNamespace(
        obs=np.array(step_out.obs, copy=True),
        actor=np.array(step_out.actor, copy=True),
    )

    unroll = QueueRuntime._collect_actor_unroll_all_heuristic_ids_fast(runtime, actor)

    npt.assert_array_equal(
        unroll.obs[0],
        np.repeat(np.array([[10.0], [11.0]], dtype=np.float32), 4, axis=1),
    )
    npt.assert_array_equal(
        unroll.obs[1],
        np.repeat(np.array([[17.0], [18.0]], dtype=np.float32), 4, axis=1),
    )
    npt.assert_array_equal(unroll.to_play_seat[0], np.array([0, 1], dtype=np.int8))
    npt.assert_array_equal(unroll.to_play_seat[1], np.array([1, 0], dtype=np.int8))
    npt.assert_array_equal(unroll.episode_seed[0], np.array([30_010, 30_011], dtype=np.uint64))
    npt.assert_array_equal(unroll.episode_seed[1], np.array([30_020, 30_021], dtype=np.uint64))
    assert unroll.legal_actions.ids is not None
    assert unroll.legal_actions.offsets is not None
    npt.assert_array_equal(unroll.legal_actions.ids, np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.uint32))
    npt.assert_array_equal(unroll.legal_actions.offsets, np.array([0, 2, 4, 6, 8], dtype=np.uint32))


def test_collect_all_heuristic_ids_fast_applies_action_surface_guards() -> None:
    class StepOut:
        def __init__(self) -> None:
            self.obs = np.zeros((1, 4), dtype=np.float32)
            self.actor = np.zeros((1,), dtype=np.int8)
            self.decision_kind = np.zeros((1,), dtype=np.int32)
            self.decision_id = np.zeros((1,), dtype=np.uint32)
            self.legal_ids = np.array([51, 10], dtype=np.uint32)
            self.legal_offsets = np.array([0, 2], dtype=np.uint32)
            self.legal_action_meta = np.array([[2, 0, 0, 0], [1, 0, 0, 0]], dtype=np.uint16)
            self.rewards = np.zeros((1,), dtype=np.float32)
            self.terminated = np.zeros((1,), dtype=np.bool_)
            self.truncated = np.zeros((1,), dtype=np.bool_)
            self.engine_status = np.zeros((1,), dtype=np.int32)
            self.main_move_action = np.zeros((1,), dtype=np.bool_)
            self.main_pass_action = np.zeros((1,), dtype=np.bool_)

    captured_actions: list[int] = []

    class Pool:
        def step_into_i16_legal_ids(self, actions: np.ndarray, step_out: StepOut) -> None:
            captured_actions.append(int(actions[0]))
            step_out.obs[:] = 1.0

        def reset_done_into_i16_legal_ids(self, done: np.ndarray, step_out: StepOut) -> None:
            del done, step_out

        def episode_seed_batch(self) -> np.ndarray:
            return np.array([123], dtype=np.uint64)

    step_out = StepOut()
    env = SimpleNamespace(
        pool=Pool(),
        _step_out=step_out,
        _record_python_timing=lambda *args, **kwargs: None,
        _handle_engine_status=lambda *args, **kwargs: None,
    )
    initial_batch = DecisionBoundaryBatch(
        obs=step_out.obs,
        reward=step_out.rewards,
        terminated=step_out.terminated,
        truncated=step_out.truncated,
        to_play=step_out.actor,
        actor=step_out.actor,
        decision_id=np.zeros((1,), dtype=np.uint32),
        engine_status=step_out.engine_status,
        decision_count=np.zeros((1,), dtype=np.uint32),
        tick_count=np.zeros((1,), dtype=np.uint32),
        episode_seed=np.array([123], dtype=np.uint64),
        episode_key=np.array([456], dtype=np.uint64),
        decision_kind=step_out.decision_kind,
        ids_offsets=(step_out.legal_ids, step_out.legal_offsets),
        legal_action_meta=step_out.legal_action_meta,
    )
    actor = SimpleNamespace(
        actor_id=0,
        next_unroll_seq=0,
        snapshot_version=0,
        current_batch=initial_batch,
        env=env,
        focal_seat_by_env=np.array([0], dtype=np.int8),
        seat_hidden=torch.zeros((1, 1, 2), dtype=torch.float32),
        opponent_hidden=torch.zeros((1, 1, 2), dtype=torch.float32),
    )

    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any.config = SimpleNamespace(
        unroll_length=1,
        envs_per_actor=1,
        pass_action_id=51,
        force_attack_over_pass_when_attack_legal=True,
        force_pass_over_main_move_only=False,
        mulligan_force_confirm_after_select=False,
    )
    runtime_any.observation_dim = 4
    runtime_any.action_dim = 64
    runtime_any._actor_behavior_values_required = False
    runtime_any._teacher_policy = object()
    runtime_any._teacher_guidance_active_for_collection = lambda: False
    runtime_any._policy_train_mask_for_actor = lambda *, actor, focal_rows, **_kwargs: np.asarray(
        focal_rows, dtype=np.bool_
    )
    runtime_any._ensure_legal_action_meta = lambda legal_ids, meta: meta
    runtime_any._should_track_heuristic_actor_hidden_state = lambda: False
    runtime_any._action_family_index = {"attack": 1, "pass": 2}
    runtime_any._last_action_arg0_obs_index = -1
    runtime_any._heuristic_public_actions_from_ids = lambda **kwargs: np.array(
        [int(kwargs["legal_ids"][0])],
        dtype=np.int64,
    )
    runtime_any._maybe_debug_validate_sampled_packed_actions = lambda **kwargs: None
    runtime_any._sync_actor_batch_from_step_out = lambda *, actor, step_out, pool: SimpleNamespace(
        obs=np.array(step_out.obs, copy=True),
        actor=np.array(step_out.actor, copy=True),
    )

    unroll = QueueRuntime._collect_actor_unroll_all_heuristic_ids_fast(runtime, actor)

    assert captured_actions == [10]
    assert unroll.actions.tolist() == [[10]]
    assert unroll.legal_actions.ids is not None
    npt.assert_array_equal(unroll.legal_actions.ids, np.array([10], dtype=np.uint32))
    assert unroll.counters is not None
    assert unroll.counters["attack_available_force_attack_actions"] == 1
    assert unroll.counters["focal_row_count"] == 1


def test_collect_all_heuristic_ids_native_rollout_requires_stateless_heuristic_actor() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._actor_policy_backend = "heuristic_public"
    runtime_any._actor_heuristic_fraction = 1.0
    runtime_any._fixed_opponent_backend = "simulator_native"
    runtime_any._teacher_policy = object()
    runtime_any._league_config = SimpleNamespace(sampling=SimpleNamespace(heuristic_public_mix_fraction=1.0))
    runtime_any._fixed_opponent_policy_is_active = lambda policy_id: bool(str(policy_id).strip())
    runtime_any._heuristic_native_rollout_enabled = True
    runtime_any._actor_behavior_values_required = False
    runtime_any._heuristic_actor_hidden_state_tracking = False
    runtime_any.config = SimpleNamespace()

    actor = cast(
        Any,
        SimpleNamespace(
            layout_name="i16_legal_ids",
            env=SimpleNamespace(
                pool=SimpleNamespace(
                    choose_heuristic_public_actions_into=lambda *args, **kwargs: None,
                    rollout_heuristic_public_into_i16_legal_ids=lambda *args, **kwargs: None,
                    reset_done_into_i16_legal_ids=lambda *args, **kwargs: None,
                )
            ),
            opponent_policy_id_by_env=np.array(
                [HEURISTIC_PUBLIC_POLICY_ID, HEURISTIC_PUBLIC_POLICY_ID],
                dtype=object,
            ),
            fixed_opponent_policy_id_by_env=np.array(["", HEURISTIC_PUBLIC_POLICY_ID], dtype=object),
        ),
    )

    assert QueueRuntime._can_collect_all_heuristic_ids_native_rollout(runtime, actor) is True


def test_collect_all_heuristic_ids_native_rollout_rejects_rl_action_surface_guards() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._actor_policy_backend = "heuristic_public"
    runtime_any._actor_heuristic_fraction = 1.0
    runtime_any._fixed_opponent_backend = "simulator_native"
    runtime_any._teacher_policy = object()
    runtime_any._league_config = SimpleNamespace(sampling=SimpleNamespace(heuristic_public_mix_fraction=1.0))
    runtime_any._fixed_opponent_policy_is_active = lambda policy_id: bool(str(policy_id).strip())
    runtime_any._heuristic_native_rollout_enabled = True
    runtime_any._actor_behavior_values_required = False
    runtime_any._heuristic_actor_hidden_state_tracking = False
    runtime_any.config = SimpleNamespace(force_attack_over_pass_when_attack_legal=True)

    actor = cast(
        Any,
        SimpleNamespace(
            layout_name="i16_legal_ids",
            env=SimpleNamespace(
                pool=SimpleNamespace(
                    choose_heuristic_public_actions_into=lambda *args, **kwargs: None,
                    rollout_heuristic_public_into_i16_legal_ids=lambda *args, **kwargs: None,
                    reset_done_into_i16_legal_ids=lambda *args, **kwargs: None,
                )
            ),
            opponent_policy_id_by_env=np.array(
                [HEURISTIC_PUBLIC_POLICY_ID, HEURISTIC_PUBLIC_POLICY_ID],
                dtype=object,
            ),
            fixed_opponent_policy_id_by_env=np.array(["", HEURISTIC_PUBLIC_POLICY_ID], dtype=object),
        ),
    )

    assert QueueRuntime._can_collect_all_heuristic_ids_native_rollout(runtime, actor) is False


def test_collect_all_heuristic_ids_native_rollout_rejects_hidden_tracking() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._actor_policy_backend = "heuristic_public"
    runtime_any._actor_heuristic_fraction = 1.0
    runtime_any._fixed_opponent_backend = "simulator_native"
    runtime_any._teacher_policy = object()
    runtime_any._league_config = SimpleNamespace(sampling=SimpleNamespace(heuristic_public_mix_fraction=1.0))
    runtime_any._fixed_opponent_policy_is_active = lambda policy_id: bool(str(policy_id).strip())
    runtime_any._heuristic_native_rollout_enabled = True
    runtime_any._actor_behavior_values_required = False
    runtime_any._heuristic_actor_hidden_state_tracking = True
    runtime_any.config = SimpleNamespace()

    actor = cast(
        Any,
        SimpleNamespace(
            layout_name="i16_legal_ids",
            env=SimpleNamespace(
                pool=SimpleNamespace(
                    choose_heuristic_public_actions_into=lambda *args, **kwargs: None,
                    rollout_heuristic_public_into_i16_legal_ids=lambda *args, **kwargs: None,
                    reset_done_into_i16_legal_ids=lambda *args, **kwargs: None,
                )
            ),
            opponent_policy_id_by_env=np.array(
                [HEURISTIC_PUBLIC_POLICY_ID, HEURISTIC_PUBLIC_POLICY_ID],
                dtype=object,
            ),
            fixed_opponent_policy_id_by_env=np.array(["", HEURISTIC_PUBLIC_POLICY_ID], dtype=object),
        ),
    )

    assert QueueRuntime._can_collect_all_heuristic_ids_native_rollout(runtime, actor) is False


def _teacher_test_catalog() -> ActionCatalog:
    return ActionCatalog.from_spec_bundle(
        {
            "action": {
                "action_encoding_version": 1,
                "action_space_size": 41,
                "pass_action_id": 40,
                "constants": [["MAX_HAND", 2], ["MAX_STAGE", 5], ["ATTACK_SLOT_COUNT", 3]],
                "families": [
                    {"name": "main_play_character", "base": 0, "count": 10},
                    {"name": "attack", "base": 10, "count": 9},
                    {"name": "main_move", "base": 19, "count": 20},
                    {"name": "climax_play", "base": 39, "count": 1},
                    {"name": "pass", "base": 40, "count": 1},
                ],
                "attack_type_encoding": [["frontal", 0], ["direct", 1], ["side", 2]],
            }
        }
    )


def test_select_pending_unrolls_train_ordered_keeps_same_behavior_version() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any.config = QueueRuntimeConfig(
        mode="train_ordered",
        actor_count=2,
        envs_per_actor=1,
        unroll_length=1,
        batch_unrolls_per_update=3,
        queue_capacity_unrolls=3,
        profile="fast",
        base_seed=7,
        pass_action_id=51,
        actor_reload_interval_updates=1,
    )
    runtime_any._pending_unrolls = deque(
        [
            _make_runtime_unroll(actor_id=1, unroll_seq=0, behavior_policy_version=0),
            _make_runtime_unroll(actor_id=0, unroll_seq=0, behavior_policy_version=0),
            _make_runtime_unroll(actor_id=0, unroll_seq=1, behavior_policy_version=1),
            _make_runtime_unroll(actor_id=1, unroll_seq=1, behavior_policy_version=1),
        ]
    )

    selected = QueueRuntime._select_pending_unrolls(runtime)

    assert [(item.behavior_policy_version, item.unroll_seq, item.actor_id) for item in selected] == [
        (0, 0, 0),
        (0, 0, 1),
    ]


def test_select_pending_unrolls_reserves_diverse_lane_quota_in_async_mode() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any.config = QueueRuntimeConfig(
        mode="train_async_fast",
        actor_count=6,
        envs_per_actor=1,
        unroll_length=1,
        batch_unrolls_per_update=4,
        queue_capacity_unrolls=8,
        profile="fast",
        base_seed=7,
        pass_action_id=51,
        actor_reload_interval_updates=1,
    )
    runtime_any._diverse_opponent_actor_count = 2
    runtime_any._diverse_opponent_batch_fraction = 0.5
    runtime_any._pending_unrolls = deque(
        [
            _make_runtime_unroll(actor_id=4, unroll_seq=0, behavior_policy_version=0),
            _make_runtime_unroll(actor_id=5, unroll_seq=1, behavior_policy_version=0),
            _make_runtime_unroll(actor_id=0, unroll_seq=2, behavior_policy_version=0),
            _make_runtime_unroll(actor_id=1, unroll_seq=3, behavior_policy_version=0),
            _make_runtime_unroll(actor_id=3, unroll_seq=4, behavior_policy_version=0),
        ]
    )

    selected = QueueRuntime._select_pending_unrolls(runtime)

    assert len(selected) == 4
    assert sum(1 for item in selected if item.actor_id in {0, 1}) == 2
    assert [item.actor_id for item in selected[:2]] == [0, 1]


def test_split_focal_actor_rows_forces_model_policy_on_diverse_model_lane() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._actor_policy_backend = "heuristic_public"
    runtime_any._teacher_policy = object()
    runtime_any._active_actor_heuristic_fraction = lambda: 1.0

    model_rows, heuristic_rows = QueueRuntime._split_focal_actor_rows(
        runtime,
        actor=cast(Any, SimpleNamespace(force_model_policy_lane=True)),
        focal_indices=np.asarray([0, 2, 4], dtype=np.int64),
        rng=np.random.default_rng(7),
    )

    npt.assert_array_equal(model_rows, np.asarray([0, 2, 4], dtype=np.int64))
    assert heuristic_rows.size == 0


def test_policy_train_mask_for_actor_can_exclude_pure_heuristic_lane() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._train_on_heuristic_actor_rows = False
    runtime_any._actor_policy_backend = "heuristic_public"
    runtime_any._active_actor_heuristic_fraction = lambda: 1.0

    mask = QueueRuntime._policy_train_mask_for_actor(
        runtime,
        actor=cast(Any, SimpleNamespace(force_model_policy_lane=False)),
        focal_rows=np.asarray([True, False, True, False], dtype=np.bool_),
    )

    npt.assert_array_equal(mask, np.asarray([False, False, False, False], dtype=np.bool_))


def test_reset_done_rows_fallback_reinitializes_full_actor_state() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any.config = QueueRuntimeConfig(
        mode="train_ordered",
        actor_count=1,
        envs_per_actor=2,
        unroll_length=1,
        batch_unrolls_per_update=2,
        queue_capacity_unrolls=2,
        profile="fast",
        base_seed=7,
        pass_action_id=51,
        actor_reload_interval_updates=1,
    )
    runtime_any._device = torch.device("cpu")

    assign_calls: list[tuple[np.ndarray, bool]] = []

    def fake_assign_episode_roles(actor, done, *, initial: bool = False) -> None:
        assign_calls.append((np.asarray(done, dtype=np.bool_).copy(), initial))
        actor.focal_seat_by_env[:] = np.array([0, 1], dtype=np.int64)
        actor.opponent_policy_id_by_env[:] = np.asarray(["mirror", "policy_000007"], dtype=object)

    runtime_any._assign_episode_roles = fake_assign_episode_roles

    class _FakeEnv:
        def reset_done(self, done) -> None:
            raise RuntimeError("reset_done unsupported")

        def reset(self, *, seed: int):
            return {"seed": seed}

    class _FakeModel:
        def initial_seat_hidden(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
            return torch.zeros((batch_size, 3), device=device)

    actor = cast(
        Any,
        SimpleNamespace(
            env=_FakeEnv(),
            model=_FakeModel(),
            rng=np.random.default_rng(7),
            seat_hidden=torch.ones((2, 3)),
            opponent_hidden=torch.full((2, 3), 2.0),
            focal_seat_by_env=np.array([1, 1], dtype=np.int64),
            opponent_policy_id_by_env=np.asarray(["stale_a", "stale_b"], dtype=object),
        ),
    )

    batch = cast(Any, QueueRuntime._reset_done_rows(runtime, actor, np.array([True, False], dtype=np.bool_)))

    assert isinstance(batch, dict)
    assert isinstance(batch["seed"], int)
    assert torch.count_nonzero(actor.seat_hidden) == 0
    assert torch.count_nonzero(actor.opponent_hidden) == 0
    assert len(assign_calls) == 1
    assert np.array_equal(assign_calls[0][0], np.array([True, True], dtype=np.bool_))
    assert assign_calls[0][1] is True
    assert actor.focal_seat_by_env.tolist() == [0, 1]
    assert actor.opponent_policy_id_by_env.tolist() == ["mirror", "policy_000007"]


def test_advance_hidden_only_prefers_hidden_only_model_path() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._device = torch.device("cpu")
    runtime_any._actor_amp_enabled = False

    class _FakeModel:
        def __init__(self) -> None:
            self.advance_calls = 0
            self.forward_calls = 0

        def advance_seat_hidden(self, obs, acting_seat, hidden_state):
            self.advance_calls += 1
            assert tuple(obs.shape) == (2, 3)
            assert tuple(acting_seat.shape) == (2,)
            return hidden_state + 5.0

        def forward_seat_aware(self, obs, acting_seat, hidden_state):
            self.forward_calls += 1
            raise AssertionError("forward_seat_aware should not run when advance_seat_hidden exists")

    hidden = torch.zeros((4, 2), dtype=torch.float32)
    model = _FakeModel()

    QueueRuntime._advance_hidden_only(
        runtime,
        model=model,
        hidden_state=hidden,
        row_indices=np.array([1, 3], dtype=np.int64),
        obs_step=np.zeros((4, 3), dtype=np.float32),
        actor_step=np.array([0, 1, 0, 1], dtype=np.int64),
    )

    assert model.advance_calls == 1
    assert model.forward_calls == 0
    npt.assert_array_equal(hidden.numpy(), np.array([[0.0, 0.0], [5.0, 5.0], [0.0, 0.0], [5.0, 5.0]], dtype=np.float32))


def test_bootstrap_values_prefers_value_only_model_path() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._device = torch.device("cpu")
    runtime_any._actor_amp_enabled = False
    runtime_any._bootstrap_models = None
    runtime_any._actors = {}

    class _FakeModel:
        def __init__(self) -> None:
            self.value_calls = 0
            self.forward_calls = 0

        def value_seat_aware(self, obs, acting_seat, hidden_state):
            self.value_calls += 1
            return torch.full((obs.shape[0],), 7.0, dtype=torch.float32, device=obs.device)

        def forward_seat_aware(self, obs, acting_seat, hidden_state):
            self.forward_calls += 1
            raise AssertionError("forward_seat_aware should not run when value_seat_aware exists")

    model = _FakeModel()
    runtime_any._actors[0] = cast(Any, SimpleNamespace(model=model))
    unroll = _make_runtime_unroll(actor_id=0, unroll_seq=0, behavior_policy_version=0)
    unroll = replace(
        unroll,
        bootstrap_obs=np.zeros((3, 2), dtype=np.float32),
        bootstrap_actor=np.array([0, 2, 1], dtype=np.int64),
        final_hidden_state=np.zeros((3, 4), dtype=np.float32),
    )

    values = QueueRuntime._bootstrap_values(runtime, unroll)

    assert model.value_calls == 1
    assert model.forward_calls == 0
    npt.assert_array_equal(values, np.array([7.0, 0.0, 7.0], dtype=np.float32))


def test_sample_packed_action_scores_falls_back_to_last_candidate_when_cdf_undershoots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packed_scores = torch.tensor([0.0, -0.1, -0.2], dtype=torch.float32)
    packed_ids = torch.tensor([4, 7, 9], dtype=torch.long)
    packed_offsets = torch.tensor([0, 3], dtype=torch.long)
    sample_seeds = torch.tensor([123], dtype=torch.long)

    monkeypatch.setattr(
        model_module,
        "_uniform_from_seeds",
        lambda sample_seeds, *, dtype: torch.tensor([0.99999994], dtype=dtype, device=sample_seeds.device),
    )
    monkeypatch.setattr(
        model_module,
        "_packed_local_cdf",
        lambda probabilities, offsets: torch.tensor(
            [0.4, 0.8, 0.9999999], dtype=probabilities.dtype, device=probabilities.device
        ),
    )

    actions, logp = model_module._sample_packed_action_scores(
        packed_scores,
        packed_ids,
        packed_offsets,
        sample_seeds,
        pass_action_id=51,
    )

    assert actions.tolist() == [9]
    assert torch.isfinite(logp).all()


def test_structured_warmstart_source_mix_balances_sources_and_restores_actor_slots() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any.config = QueueRuntimeConfig(
        mode="train_ordered",
        actor_count=1,
        envs_per_actor=8,
        unroll_length=1,
        batch_unrolls_per_update=2,
        queue_capacity_unrolls=2,
        profile="fast",
        base_seed=7,
        pass_action_id=51,
        actor_reload_interval_updates=1,
    )
    runtime_any._collector_result_queue = None
    runtime_any._teacher_policy = object()
    runtime_any._opponent_heuristic_policies = {}
    runtime_any._opponent_models = {_NOLEAGUE_BASELINE_POLICY_ID: object()}
    runtime_any._device = torch.device("cpu")
    runtime_any._forced_fixed_opponent_policy_ids = ()

    def fake_assign_episode_roles(actor, done, *, initial: bool = False) -> None:
        fixed = actor.fixed_opponent_policy_id_by_env
        assigned = np.full(actor.focal_seat_by_env.shape, _MIRROR_OPPONENT_POLICY_ID, dtype=object)
        if fixed is not None:
            fixed_array = np.asarray(fixed, dtype=object)
            non_empty = np.asarray([bool(str(value).strip()) for value in fixed_array.tolist()], dtype=np.bool_)
            assigned[non_empty] = fixed_array[non_empty]
        actor.opponent_policy_id_by_env[:] = assigned

    runtime_any._assign_episode_roles = fake_assign_episode_roles

    class _FakeEnv:
        def reset_done(self, done):
            return {"reset_done": np.asarray(done, dtype=np.bool_).copy()}

    class _FakeModel:
        def initial_seat_hidden(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
            return torch.zeros((batch_size, 3), device=device)

    actor = cast(
        Any,
        SimpleNamespace(
            env=_FakeEnv(),
            model=_FakeModel(),
            rng=np.random.default_rng(7),
            seat_hidden=torch.ones((8, 3)),
            opponent_hidden=torch.ones((8, 3)),
            focal_seat_by_env=np.zeros((8,), dtype=np.int64),
            opponent_policy_id_by_env=np.full((8,), _MIRROR_OPPONENT_POLICY_ID, dtype=object),
            fixed_opponent_policy_id_by_env=None,
            current_batch=None,
        ),
    )
    runtime_any._actors = [actor]

    with QueueRuntime.structured_warmstart_source_mix(runtime) as metrics:
        assert metrics["structured_warmstart_source_count"] == 3.0
        assert metrics["structured_warmstart_self_play_envs_per_actor"] == 3.0
        assert metrics["structured_warmstart_b1_envs_per_actor"] == 3.0
        assert metrics["structured_warmstart_b2_envs_per_actor"] == 2.0
        assert actor.fixed_opponent_policy_id_by_env is not None
        assert actor.opponent_policy_id_by_env.tolist().count(_NOLEAGUE_BASELINE_POLICY_ID) == 3
        assert actor.opponent_policy_id_by_env.tolist().count(HEURISTIC_PUBLIC_POLICY_ID) == 2
        assert actor.opponent_policy_id_by_env.tolist().count(_MIRROR_OPPONENT_POLICY_ID) == 3
        assert HEURISTIC_PUBLIC_POLICY_ID in runtime_any._opponent_heuristic_policies
        assert runtime_any._forced_fixed_opponent_policy_ids == (
            _NOLEAGUE_BASELINE_POLICY_ID,
            HEURISTIC_PUBLIC_POLICY_ID,
        )

    assert actor.fixed_opponent_policy_id_by_env is None
    assert actor.opponent_policy_id_by_env.tolist() == [_MIRROR_OPPONENT_POLICY_ID] * 8
    assert HEURISTIC_PUBLIC_POLICY_ID not in runtime_any._opponent_heuristic_policies
    assert runtime_any._forced_fixed_opponent_policy_ids == ()


def test_structured_warmstart_source_mix_process_collectors_pushes_fixed_sources() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any.config = QueueRuntimeConfig(
        mode="train_async_fast",
        actor_count=2,
        envs_per_actor=8,
        unroll_length=1,
        batch_unrolls_per_update=2,
        queue_capacity_unrolls=2,
        profile="fast",
        base_seed=11,
        pass_action_id=51,
        actor_reload_interval_updates=1,
    )
    runtime_any._collector_result_queue = object()
    runtime_any._teacher_policy = object()
    runtime_any._opponent_heuristic_policies = {}
    runtime_any._opponent_models = {_NOLEAGUE_BASELINE_POLICY_ID: torch.nn.Linear(1, 1)}
    runtime_any._forced_fixed_opponent_policy_ids = ()
    runtime_any._actors = []

    class _FakeControlQueue:
        def __init__(self) -> None:
            self.commands: list[dict[str, Any]] = []

        def put(self, payload: dict[str, Any]) -> None:
            self.commands.append(payload)

    control_queues = [_FakeControlQueue(), _FakeControlQueue()]
    runtime_any._collector_control_queues = control_queues

    with QueueRuntime.structured_warmstart_source_mix(runtime) as metrics:
        assert metrics["structured_warmstart_source_count"] == 3.0
        assert metrics["structured_warmstart_self_play_envs_per_actor"] == 3.0
        assert metrics["structured_warmstart_b1_envs_per_actor"] == 3.0
        assert metrics["structured_warmstart_b2_envs_per_actor"] == 2.0
        assert runtime_any._forced_fixed_opponent_policy_ids == (
            _NOLEAGUE_BASELINE_POLICY_ID,
            HEURISTIC_PUBLIC_POLICY_ID,
        )

    for control_queue in control_queues:
        assert len(control_queue.commands) == 2
        apply_payload, restore_payload = control_queue.commands
        assert apply_payload["kind"] == "set_fixed_opponents"
        assert apply_payload["restore_defaults"] is False
        assert apply_payload["activate_teacher_heuristic"] is True
        assert tuple(apply_payload["forced_policy_ids"]) == (
            _NOLEAGUE_BASELINE_POLICY_ID,
            HEURISTIC_PUBLIC_POLICY_ID,
        )
        assert apply_payload["fixed_opponent_policy_id_by_env"].count(_NOLEAGUE_BASELINE_POLICY_ID) == 3
        assert apply_payload["fixed_opponent_policy_id_by_env"].count(HEURISTIC_PUBLIC_POLICY_ID) == 2
        assert apply_payload["fixed_opponent_policy_id_by_env"].count("") == 3
        assert isinstance(apply_payload["noleague_baseline_state_dict"], dict)
        assert restore_payload == {"kind": "set_fixed_opponents", "restore_defaults": True}


def test_structured_warmstart_source_mix_restores_actor_slots_after_exception() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any.config = QueueRuntimeConfig(
        mode="train_ordered",
        actor_count=1,
        envs_per_actor=4,
        unroll_length=1,
        batch_unrolls_per_update=2,
        queue_capacity_unrolls=2,
        profile="fast",
        base_seed=17,
        pass_action_id=51,
        actor_reload_interval_updates=1,
    )
    runtime_any._collector_result_queue = None
    teacher_policy = object()
    runtime_any._teacher_policy = teacher_policy
    runtime_any._opponent_heuristic_policies = {}
    runtime_any._opponent_models = {_NOLEAGUE_BASELINE_POLICY_ID: object()}
    runtime_any._device = torch.device("cpu")
    runtime_any._forced_fixed_opponent_policy_ids = ("preexisting",)

    reset_calls: list[np.ndarray | None] = []

    def fake_reset_actor_state_for_fixed_opponents(actor) -> None:
        fixed = actor.fixed_opponent_policy_id_by_env
        reset_calls.append(None if fixed is None else np.asarray(fixed, dtype=object).copy())

    runtime_any._reset_actor_state_for_fixed_opponents = fake_reset_actor_state_for_fixed_opponents
    saved_slots = np.asarray(["existing", ""], dtype=object)
    actor = cast(
        Any,
        SimpleNamespace(
            fixed_opponent_policy_id_by_env=saved_slots.copy(),
        ),
    )
    runtime_any._actors = [actor]

    with pytest.raises(RuntimeError, match="boom"):
        with QueueRuntime.structured_warmstart_source_mix(runtime):
            assert HEURISTIC_PUBLIC_POLICY_ID in runtime_any._opponent_heuristic_policies
            assert actor.fixed_opponent_policy_id_by_env is not None
            raise RuntimeError("boom")

    assert np.array_equal(actor.fixed_opponent_policy_id_by_env, saved_slots)
    assert runtime_any._forced_fixed_opponent_policy_ids == ("preexisting",)
    assert HEURISTIC_PUBLIC_POLICY_ID not in runtime_any._opponent_heuristic_policies
    assert len(reset_calls) == 2
    assert reset_calls[0] is not None
    assert reset_calls[1] is not None
    assert np.array_equal(reset_calls[1], saved_slots)


def test_disable_mirror_policy_fusion_context_restores_previous_state() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._disable_mirror_policy_fusion = False

    with QueueRuntime.disable_mirror_policy_fusion(runtime):
        assert runtime_any._disable_mirror_policy_fusion is True

    assert runtime_any._disable_mirror_policy_fusion is False

    runtime_any._disable_mirror_policy_fusion = True  # type: ignore[unreachable]
    with QueueRuntime.disable_mirror_policy_fusion(runtime):
        assert runtime_any._disable_mirror_policy_fusion is True
    assert runtime_any._disable_mirror_policy_fusion is True


def test_handle_collector_commands_tracks_update_and_refreshes_pool() -> None:
    refresh_calls: list[int] = []

    runtime = cast(
        Any,
        SimpleNamespace(
            _current_learner_update=0,
            _effective_learner_update=0,
            refresh_opponent_pool=lambda: refresh_calls.append(1),
            _teacher_policy=None,
            _opponent_heuristic_policies={},
            _opponent_models={},
            _opponent_model_locks={},
            _forced_fixed_opponent_policy_ids=(),
            _reset_actor_state_for_fixed_opponents=lambda actor: None,
        ),
    )

    class _FakeModel:
        def __init__(self) -> None:
            self.loaded = 0
            self.evaluated = 0

        def load_state_dict(self, state_dict: dict[str, Any]) -> None:
            self.loaded += len(state_dict)

        def eval(self) -> _FakeModel:
            self.evaluated += 1
            return self

    actor = cast(Any, SimpleNamespace(model=_FakeModel(), snapshot_version=0, fixed_opponent_policy_id_by_env=None))

    class _Queue:
        def __init__(self, commands: list[dict[str, Any]]) -> None:
            self._commands = list(commands)

        def get_nowait(self) -> dict[str, Any]:
            if not self._commands:
                raise queue.Empty
            return self._commands.pop(0)

    queue_obj = _Queue(
        [
            {"kind": "set_update", "update": 7, "refresh_opponent_pool": True},
            {"kind": "reload", "model_state_dict": {"w": torch.tensor([1.0])}, "update": 8, "effective_update": 5},
            {"kind": "refresh_opponent_pool", "update": 9},
        ]
    )

    should_stop = _handle_collector_commands(
        runtime=runtime,
        actor=actor,
        control_queue=queue_obj,
        default_fixed_slots=None,
        default_forced_policy_ids=(),
        default_teacher_active=False,
        default_has_noleague_baseline=False,
    )

    assert should_stop is False
    assert actor.snapshot_version == 8
    assert runtime._current_learner_update == 9
    assert runtime._effective_learner_update == 5
    assert actor.model.loaded == 1
    assert actor.model.evaluated == 1
    assert len(refresh_calls) == 2


def test_build_runtime_config_minimal_batch_uses_one_unroll_per_actor() -> None:
    stack = SimpleNamespace(
        config=SimpleNamespace(
            system=SimpleNamespace(
                actor_process_count=12,
                envs_per_actor=8,
                actor_queue_capacity_unrolls=256,
            ),
            training=SimpleNamespace(
                batch_unrolls_per_update=128,
                actor_reload_interval_updates=1000,
            ),
        )
    )

    small = build_runtime_config(
        stack=cast(Any, stack),
        num_envs=1,
        unroll_length=4,
        profile="fast",
        seed=7,
        pass_action_id=51,
        runtime_mode="train_ordered",
        minimal_batch=True,
    )
    assert small.actor_count == 1
    assert small.envs_per_actor == 1
    assert small.batch_unrolls_per_update == 1
    assert small.queue_capacity_unrolls == 1

    full = build_runtime_config(
        stack=cast(Any, stack),
        num_envs=96,
        unroll_length=4,
        profile="fast",
        seed=7,
        pass_action_id=51,
        runtime_mode="train_ordered",
        minimal_batch=True,
    )
    assert full.actor_count == 12
    assert full.envs_per_actor == 8
    assert full.batch_unrolls_per_update == 12
    assert full.queue_capacity_unrolls == 12

    default = build_runtime_config(
        stack=cast(Any, stack),
        num_envs=96,
        unroll_length=4,
        profile="fast",
        seed=7,
        pass_action_id=51,
        runtime_mode="train_ordered",
    )
    assert default.batch_unrolls_per_update == 128
    assert default.queue_capacity_unrolls == 256


def test_runtime_metrics_report_window_and_cumulative_env_step_rates(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._runtime_start = 100.0
    runtime_any._runtime_last_metrics_time = 108.0
    runtime_any._runtime_cumulative_env_steps = 128
    runtime_any._last_published_snapshot_version = 5
    runtime_any._current_learner_update = 5
    runtime_any._effective_learner_update = 3
    runtime_any._league_config = SimpleNamespace(
        sampling=SimpleNamespace(
            heuristic_public_mix_fraction=1.0,
            heuristic_public_mix_end_updates=5,
            heuristic_public_final_mix_fraction=0.25,
        )
    )
    runtime_any._actor_heuristic_fraction = 1.0
    runtime_any._actor_heuristic_end_updates = 5
    runtime_any._actor_heuristic_final_fraction = 0.25
    runtime_any._pfsp_pool_size = 3
    runtime_any._pfsp_quarantined_opponents = 1
    runtime_any._pfsp_champion_pool_size = 1
    runtime_any._pfsp_recent_pool_size = 1
    runtime_any._pfsp_hard_negative_pool_size = 1
    runtime_any._pfsp_last_sampled_envs = 2
    runtime_any._pfsp_last_mirror_envs = 6
    runtime_any._pfsp_last_heuristic_public_envs = 2
    runtime_any._pfsp_last_noleague_baseline_envs = 1
    runtime_any._pfsp_last_champion_envs = 1
    runtime_any._pfsp_last_recent_envs = 0
    runtime_any._pfsp_last_hard_negative_envs = 1
    runtime_any._pfsp_epoch = 3

    monkeypatch.setattr("weiss_rl.runtime.time.time", lambda: 110.0)
    metrics = QueueRuntime._runtime_metrics(
        runtime,
        [
            _make_runtime_unroll(
                actor_id=0,
                unroll_seq=0,
                behavior_policy_version=4,
                counters={
                    "engine_fault_done_rows": 2,
                    "no_progress_timeout_rows": 1,
                    "pass_actions": 3,
                    "main_move_actions": 4,
                    "max_consecutive_main_moves": 2,
                },
            ),
            replace(
                _make_runtime_unroll(actor_id=1, unroll_seq=0, behavior_policy_version=5),
                obs=np.zeros((2, 3, 1), dtype=np.float32),
            ),
        ],
        occupancy_samples=[0.25, 0.75],
    )

    assert metrics["batch_env_steps"] == pytest.approx(7.0)
    assert metrics["actor_env_steps_per_sec"] == pytest.approx(3.5)
    assert metrics["actor_env_steps_per_sec_cumulative"] == pytest.approx(13.5)
    assert metrics["policy_version_lag_p50"] == pytest.approx(0.5)
    assert metrics["learner_actor_update_lag_p50"] == pytest.approx(0.5)
    assert metrics["learner_actor_update_lag_p90"] == pytest.approx(0.9)
    assert metrics["league_effective_update"] == pytest.approx(3.0)
    assert metrics["league_update_lag"] == pytest.approx(2.0)
    assert metrics["actor_heuristic_fraction_active"] == pytest.approx(0.55)
    assert metrics["heuristic_public_mix_fraction_active"] == pytest.approx(0.55)
    assert metrics["pfsp_quarantined_opponents"] == pytest.approx(1.0)
    assert metrics["pfsp_champion_pool_size"] == pytest.approx(1.0)
    assert metrics["pfsp_heuristic_public_envs"] == pytest.approx(2.0)
    assert metrics["pfsp_noleague_baseline_envs"] == pytest.approx(1.0)
    assert metrics["pfsp_hard_negative_envs"] == pytest.approx(1.0)
    assert metrics["pfsp_epoch"] == pytest.approx(3.0)
    assert metrics["queue_occupancy_p50"] == pytest.approx(0.5)
    assert metrics["collector_engine_fault_done_rows"] == pytest.approx(2.0)
    assert metrics["collector_no_progress_timeout_rows"] == pytest.approx(1.0)
    assert metrics["collector_pass_actions"] == pytest.approx(3.0)
    assert metrics["collector_main_move_actions"] == pytest.approx(4.0)
    assert metrics["collector_max_consecutive_main_moves"] == pytest.approx(2.0)
    assert runtime_any._runtime_last_metrics_time == pytest.approx(110.0)
    assert runtime_any._runtime_cumulative_env_steps == 135


def test_teacher_labels_from_actions_group_main_play_and_attack_semantics() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    action_catalog = _teacher_test_catalog()
    main_move_to_slot_2 = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if ((decoded := action_catalog.decode(action_id)).family == "main_move" and decoded.to_slot == 2)
    )
    runtime_any._teacher_guidance_enabled = True
    runtime_any._teacher_action_catalog = action_catalog
    runtime_any._teacher_family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    runtime_any._teacher_attack_type_index = {
        name: index for index, name in enumerate(action_catalog.attack_type_names)
    }

    teacher_family, teacher_slot, teacher_move_source, teacher_attack_type, teacher_action, teacher_valid = (
        QueueRuntime._teacher_labels_from_actions(
            runtime,
            row_indices=np.array([0, 1, 2, 3], dtype=np.int64),
            chosen_actions=np.array([0, 14, main_move_to_slot_2, 40], dtype=np.int64),
            num_rows=4,
        )
    )

    assert teacher_valid.tolist() == [True, True, True, True]
    assert teacher_family.tolist() == [
        runtime_any._teacher_family_index["main_play_character"],
        runtime_any._teacher_family_index["attack"],
        runtime_any._teacher_family_index["main_move"],
        runtime_any._teacher_family_index["pass"],
    ]
    assert teacher_slot.tolist() == [0, 1, 2, -1]
    assert teacher_move_source.tolist() == [-1, -1, 0, -1]
    assert teacher_attack_type.tolist() == [-1, 1, -1, -1]
    assert teacher_action.tolist() == [0, 14, main_move_to_slot_2, 40]


def test_teacher_guidance_active_for_collection_respects_warmstart_only() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._teacher_guidance_enabled = True
    runtime_any._teacher_aux_mode = "warmstart_only"
    runtime_any._teacher_guidance_warmstart_updates = 1
    runtime_any._current_learner_update = 0

    assert QueueRuntime._teacher_guidance_active_for_collection(runtime) is True

    runtime_any._current_learner_update = 1

    assert QueueRuntime._teacher_guidance_active_for_collection(runtime) is False


def test_teacher_labels_from_actions_skip_after_warmstart_only_phase() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    action_catalog = _teacher_test_catalog()
    runtime_any._teacher_guidance_enabled = True
    runtime_any._teacher_aux_mode = "warmstart_only"
    runtime_any._teacher_guidance_warmstart_updates = 1
    runtime_any._current_learner_update = 1
    runtime_any._teacher_action_catalog = action_catalog
    runtime_any._teacher_family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    runtime_any._teacher_attack_type_index = {
        name: index for index, name in enumerate(action_catalog.attack_type_names)
    }

    teacher_family, teacher_slot, teacher_move_source, teacher_attack_type, teacher_action, teacher_valid = (
        QueueRuntime._teacher_labels_from_actions(
            runtime,
            row_indices=np.array([0], dtype=np.int64),
            chosen_actions=np.array([0], dtype=np.int64),
            num_rows=1,
        )
    )

    assert teacher_valid.tolist() == [False]
    assert teacher_family.tolist() == [-1]
    assert teacher_slot.tolist() == [-1]
    assert teacher_move_source.tolist() == [-1]
    assert teacher_attack_type.tolist() == [-1]
    assert teacher_action.tolist() == [-1]


def test_concat_optional_time_major_field_fills_unlabeled_rows_with_sentinels() -> None:
    labeled = SimpleNamespace(
        obs=np.zeros((2, 2, 1), dtype=np.float32),
        teacher_family=np.asarray([[7, 8], [9, 10]], dtype=np.int32),
        teacher_valid=np.asarray([[True, False], [False, True]], dtype=np.bool_),
    )
    unlabeled = SimpleNamespace(
        obs=np.zeros((2, 3, 1), dtype=np.float32),
        teacher_family=None,
        teacher_valid=None,
    )

    teacher_family = _concat_optional_time_major_field(
        [cast(Any, labeled), cast(Any, unlabeled)],
        "teacher_family",
        missing_fill_value=-1,
    )
    teacher_valid = _concat_optional_time_major_field(
        [cast(Any, labeled), cast(Any, unlabeled)],
        "teacher_valid",
        missing_fill_value=False,
    )

    assert teacher_family is not None
    assert teacher_valid is not None
    assert teacher_family.shape == (2, 5)
    assert teacher_valid.shape == (2, 5)
    npt.assert_array_equal(teacher_family[:, :2], labeled.teacher_family)
    npt.assert_array_equal(teacher_family[:, 2:], np.full((2, 3), -1, dtype=np.int32))
    npt.assert_array_equal(teacher_valid[:, :2], labeled.teacher_valid)
    npt.assert_array_equal(teacher_valid[:, 2:], np.zeros((2, 3), dtype=np.bool_))


def test_teacher_labels_from_ids_cover_public_decision_kinds_beyond_tactical_subset() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    action_catalog = _teacher_test_catalog()
    runtime_any._teacher_guidance_enabled = True
    runtime_any._teacher_policy = object()
    runtime_any._teacher_action_catalog = action_catalog
    runtime_any._teacher_family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    runtime_any._teacher_attack_type_index = {
        name: index for index, name in enumerate(action_catalog.attack_type_names)
    }
    runtime_any._heuristic_public_actions_from_ids = lambda **kwargs: np.asarray(
        [
            int(kwargs["legal_ids"][int(kwargs["legal_offsets"][row_index])])
            for row_index in np.asarray(kwargs["row_indices"], dtype=np.int64).tolist()
        ],
        dtype=np.int64,
    )

    legal_ids = np.asarray([0, 40, 14, 40, 39, 40, 40, 40], dtype=np.uint32)
    legal_offsets = np.asarray([0, 2, 4, 6, 8], dtype=np.uint32)
    counters = {"teacher_tactical_row_count": 0}

    teacher_family, teacher_slot, teacher_move_source, teacher_attack_type, teacher_action, teacher_valid = (
        QueueRuntime._teacher_labels_from_ids(
            runtime,
            focal_rows=np.asarray([True, True, True, True], dtype=np.bool_),
            decision_kind=np.asarray([1, 5, 8, 0], dtype=np.int32),
            obs_step=np.zeros((4, 4), dtype=np.float32),
            legal_ids=legal_ids,
            legal_offsets=legal_offsets,
            legal_action_meta=None,
            counters=counters,
        )
    )

    assert teacher_valid.tolist() == [True, True, True, False]
    assert teacher_family.tolist() == [
        runtime_any._teacher_family_index["main_play_character"],
        runtime_any._teacher_family_index["attack"],
        runtime_any._teacher_family_index["climax_play"],
        -1,
    ]
    assert teacher_slot.tolist() == [0, 1, -1, -1]
    assert teacher_move_source.tolist() == [-1, -1, -1, -1]
    assert teacher_attack_type.tolist() == [-1, 1, -1, -1]
    assert teacher_action.tolist() == [0, 14, 39, -1]
    assert counters["teacher_tactical_row_count"] == 3


def test_central_structured_unroll_snapshots_replay_behavior_logp() -> None:
    pytest.importorskip("weiss_sim")
    from pathlib import Path

    from weiss_rl.config import apply_stack_overrides, load_stack_config, parse_override_tokens
    from weiss_rl.core.simulator_contract import load_verified_simulator_contract
    from weiss_rl.learners.action_logp import packed_scores_action_logp_and_entropy
    from weiss_rl.training.environments import spec_dimensions

    repo_root = Path(__file__).resolve().parents[3]
    stack = load_stack_config(repo_root / "configs" / "presets" / "typed_structured_v2.yaml")
    stack = apply_stack_overrides(
        stack,
        parse_override_tokens(
            [
                "system.actor_device=cpu",
                "system.learner_device=cpu",
                "system.collection_backend=central",
                "training.precision.mixed_precision=false",
            ]
        ),
    )
    contract = load_verified_simulator_contract(repo_root, expected_spec_hash="")
    observation_dim, action_dim = spec_dimensions(contract)
    pass_action_id = int(contract.spec_bundle["action"]["pass_action_id"])
    device = torch.device("cpu")
    model = model_module.build_policy_value_model(
        observation_dim=observation_dim,
        config=stack.config.model,
        action_dim=action_dim,
        observation_spec=contract.spec_bundle.get("observation"),
        spec_bundle=contract.spec_bundle,
    ).to(device)
    model.eval()
    runtime_config = build_runtime_config(
        stack=stack,
        num_envs=4,
        unroll_length=4,
        profile="fast",
        seed=20260513,
        pass_action_id=pass_action_id,
        runtime_mode="train_async_fast",
        minimal_batch=True,
    )
    runtime = QueueRuntime(
        stack=stack,
        config=runtime_config,
        model=model,
        observation_dim=observation_dim,
        action_dim=action_dim,
        observation_spec=contract.spec_bundle.get("observation"),
        spec_bundle=contract.spec_bundle,
        learner_device=device,
    )
    try:
        runtime._fill_pending_unrolls(target_count=int(runtime.config.batch_unrolls_per_update), occupancy_samples=[])
        unroll = runtime._select_pending_unrolls()[0]
        replay_model = cast(Any, runtime)._shared_actor_model or model
        replay_model.eval()
        obs = torch.as_tensor(unroll.obs, device=device, dtype=torch.float32)
        acting_seat = torch.as_tensor(unroll.to_play_seat, device=device, dtype=torch.long)
        initial_hidden = torch.as_tensor(unroll.initial_hidden_state, device=device, dtype=torch.float32)
        done = np.logical_or(unroll.terminated, unroll.truncated)
        reset_before_step = np.zeros_like(done, dtype=np.bool_)
        reset_before_step[1:] = done[:-1]
        actions = torch.as_tensor(unroll.actions, device=device, dtype=torch.long)
        behavior_logp = torch.as_tensor(unroll.behavior_logp, device=device, dtype=torch.float32)
        train_mask = torch.as_tensor(unroll.policy_train_mask, device=device, dtype=torch.bool)
        legal_actions = unroll.legal_actions
        assert legal_actions.ids is not None
        assert legal_actions.offsets is not None

        with torch.inference_mode():
            recurrent_flat, state_repr, observation_context, _values, _next_hidden = (
                replay_model.forward_trunk_sequence_seat_aware(
                    obs,
                    acting_seat,
                    initial_hidden,
                    reset_before_step=torch.as_tensor(reset_before_step, device=device, dtype=torch.bool),
                )
            )
            packed_scores = replay_model.score_packed_legal_candidates(
                recurrent_flat,
                obs.reshape(obs.shape[0] * obs.shape[1], obs.shape[2]),
                legal_actions,
                state_repr=state_repr,
                observation_context=observation_context,
                scoring_mode="actor",
            )
            replay_logp, _entropy = packed_scores_action_logp_and_entropy(
                packed_scores,
                torch.as_tensor(legal_actions.ids, device=device, dtype=torch.long),
                torch.as_tensor(legal_actions.offsets, device=device, dtype=torch.long),
                actions,
                pass_action_id=pass_action_id,
            )

        delta = (replay_logp - behavior_logp).abs()
        assert float(delta[train_mask].max().item()) < 1e-5
    finally:
        runtime.close()
