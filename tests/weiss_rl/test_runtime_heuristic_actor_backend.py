from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch
from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.runtime import QueueRuntime


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
    model = _HeuristicRuntimeActorModel()

    actor, values_out, actions_out, logp_out = _fill_heuristic_policy_outputs(model)

    assert model.forward_legal_flags == [False]
    assert np.array_equal(actions_out, np.array([7, 8], dtype=np.int64))
    assert np.allclose(logp_out, 0.0)
    assert np.allclose(values_out, 0.25)
    assert torch.allclose(actor.seat_hidden, torch.ones_like(actor.seat_hidden))


def test_fill_policy_outputs_ids_heuristic_actor_backend_can_skip_behavior_values() -> None:
    model = _HeuristicAdvanceOnlyRuntimeActorModel()

    actor, values_out, actions_out, logp_out = _fill_heuristic_policy_outputs(
        model,
        behavior_values_required=False,
        initial_values=9.0,
    )

    assert model.advance_calls == 1
    assert model.forward_calls == 0
    assert np.array_equal(actions_out, np.array([7, 8], dtype=np.int64))
    assert np.allclose(logp_out, 0.0)
    assert np.allclose(values_out, 0.0)
    assert torch.allclose(actor.seat_hidden, torch.ones_like(actor.seat_hidden))


def test_fill_policy_outputs_ids_heuristic_actor_backend_can_skip_hidden_tracking() -> None:
    model = _HeuristicAdvanceOnlyRuntimeActorModel()

    actor, values_out, actions_out, logp_out = _fill_heuristic_policy_outputs(
        model,
        behavior_values_required=False,
        track_hidden=False,
        initial_values=9.0,
    )

    assert model.advance_calls == 0
    assert model.forward_calls == 0
    assert np.array_equal(actions_out, np.array([7, 8], dtype=np.int64))
    assert np.allclose(logp_out, 0.0)
    assert np.allclose(values_out, 0.0)
    assert torch.allclose(actor.seat_hidden, torch.zeros_like(actor.seat_hidden))


def _fill_heuristic_policy_outputs(
    model: torch.nn.Module,
    *,
    behavior_values_required: bool = True,
    track_hidden: bool = True,
    initial_values: float = 0.0,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    actor = _make_actor(model)
    values_out = np.full((2,), float(initial_values), dtype=np.float32)
    actions_out = np.zeros((2,), dtype=np.int64)
    logp_out = np.zeros((2,), dtype=np.float32)
    QueueRuntime._fill_policy_outputs_ids(
        _make_runtime(behavior_values_required=behavior_values_required, track_hidden=track_hidden),
        actor=actor,
        obs_step=np.zeros((2, 4), dtype=np.float32),
        actor_step=np.array([0, 1], dtype=np.int64),
        focal_rows=np.array([True, True], dtype=np.bool_),
        legal_ids=np.array([0, 7, 8, 1, 7, 8], dtype=np.uint32),
        legal_offsets=np.array([0, 3, 6], dtype=np.uint32),
        legal_action_meta=_legal_action_meta(),
        logits_out=None,
        values_out=values_out,
        actions_out=actions_out,
        logp_out=logp_out,
        rng=np.random.default_rng(321),
        sample_actions=True,
    )
    return actor, values_out, actions_out, logp_out


def _make_runtime(*, behavior_values_required: bool, track_hidden: bool) -> QueueRuntime:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._device = torch.device("cpu")
    runtime_any._actor_amp_enabled = False
    runtime_any._actor_policy_backend = "heuristic_public"
    runtime_any._actor_heuristic_fraction = 1.0
    runtime_any._actor_behavior_values_required = bool(behavior_values_required)
    runtime_any._heuristic_actor_hidden_state_tracking = bool(track_hidden)
    runtime_any._teacher_policy = object()
    runtime_any.config = SimpleNamespace(pass_action_id=8)
    runtime_any._heuristic_public_actions_from_ids = lambda **_kwargs: np.array([7, 8], dtype=np.int64)
    return runtime


def _make_actor(model: torch.nn.Module) -> Any:
    return cast(
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


def _legal_action_meta() -> np.ndarray:
    empty_u16 = np.iinfo(np.uint16).max
    return np.array(
        [
            [0, 0, 0, empty_u16],
            [1, 0, 0, empty_u16],
            [2, empty_u16, empty_u16, empty_u16],
            [0, 1, 0, empty_u16],
            [1, 1, 0, empty_u16],
            [2, empty_u16, empty_u16, empty_u16],
        ],
        dtype=np.uint16,
    )
