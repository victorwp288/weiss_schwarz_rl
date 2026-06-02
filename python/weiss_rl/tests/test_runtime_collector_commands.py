from __future__ import annotations

import queue
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch

from weiss_rl.eval.policies.set import HEURISTIC_PUBLIC_POLICY_ID
from weiss_rl.runtime.components.ipc_shared import collector_commands
from weiss_rl.runtime.components.ipc_shared.collector_commands import handle_collector_commands


class _Queue:
    def __init__(self, commands: list[dict[str, Any]]) -> None:
        self._commands = list(commands)

    def get_nowait(self) -> dict[str, Any]:
        if not self._commands:
            raise queue.Empty
        return self._commands.pop(0)


class _FakeModel:
    def __init__(self) -> None:
        self.loaded = 0
        self.evaluated = 0
        self.guidance_payload: tuple[float, float | None] | None = None

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.loaded += len(state_dict)

    def to(self, _device: torch.device) -> _FakeModel:
        return self

    def eval(self) -> _FakeModel:
        self.evaluated += 1
        return self

    def set_public_heuristic_logit_bias_scale(self, value: float, *, actor_value: float | None = None) -> None:
        self.guidance_payload = (float(value), None if actor_value is None else float(actor_value))


def test_handle_collector_commands_applies_update_reload_refresh_and_stop() -> None:
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
    actor = cast(Any, SimpleNamespace(actor_id=2, model=_FakeModel(), snapshot_version=0))
    queue_obj = _Queue(
        [
            {"kind": "set_update", "update": 7, "refresh_opponent_pool": True},
            {"kind": "reload", "model_state_dict": {"w": torch.tensor([1.0])}, "update": 8, "effective_update": 5},
            {"kind": "refresh_opponent_pool", "update": 9},
            {"kind": "stop"},
        ]
    )

    should_stop = handle_collector_commands(
        runtime=runtime,
        actor=actor,
        control_queue=queue_obj,
        default_fixed_slots=None,
        default_forced_policy_ids=(),
        default_teacher_active=False,
        default_has_noleague_baseline=False,
    )

    assert should_stop is True
    assert actor.snapshot_version == 8
    assert runtime._current_learner_update == 9
    assert runtime._effective_learner_update == 5
    assert actor.model.loaded == 1
    assert actor.model.evaluated == 1
    assert len(refresh_calls) == 2


def test_handle_collector_commands_applies_and_restores_fixed_opponents(monkeypatch) -> None:
    reset_actor_ids: list[int] = []
    teacher_policy = object()
    stale_baseline = object()
    created_models: list[_FakeModel] = []

    def fake_build_policy_value_model(**_kwargs):
        model = _FakeModel()
        created_models.append(model)
        return model

    monkeypatch.setattr(collector_commands, "build_policy_value_model", fake_build_policy_value_model)
    runtime = cast(
        Any,
        SimpleNamespace(
            observation_dim=4,
            action_dim=2,
            stack=SimpleNamespace(config=SimpleNamespace(model=object())),
            _observation_spec=None,
            _spec_bundle=None,
            _device=torch.device("cpu"),
            _teacher_policy=teacher_policy,
            _opponent_heuristic_policies={},
            _opponent_models={"b1_noleague_baseline": stale_baseline},
            _opponent_model_locks={"b1_noleague_baseline": object()},
            _forced_fixed_opponent_policy_ids=("old",),
            _reset_actor_state_for_fixed_opponents=lambda actor: reset_actor_ids.append(int(actor.actor_id)),
        ),
    )
    actor = cast(Any, SimpleNamespace(actor_id=4, fixed_opponent_policy_id_by_env=None))
    default_slots = np.asarray(["default_a", "default_b"], dtype=object)
    queue_obj = _Queue(
        [
            {
                "kind": "set_fixed_opponents",
                "activate_teacher_heuristic": True,
                "forced_policy_ids": ["forced_a", 12],
                "fixed_opponent_policy_id_by_env": ["lane_a", "lane_b"],
                "noleague_baseline_state_dict": {"w": torch.tensor([1.0])},
                "noleague_baseline_guidance_payload": {
                    "public_heuristic_logit_bias_scale": 0.5,
                    "public_heuristic_actor_logit_bias_scale": 0.25,
                },
            },
            {"kind": "set_fixed_opponents", "restore_defaults": True},
        ]
    )

    should_stop = handle_collector_commands(
        runtime=runtime,
        actor=actor,
        control_queue=queue_obj,
        default_fixed_slots=default_slots,
        default_forced_policy_ids=("default_forced",),
        default_teacher_active=False,
        default_has_noleague_baseline=False,
    )

    assert should_stop is False
    assert HEURISTIC_PUBLIC_POLICY_ID not in runtime._opponent_heuristic_policies
    assert runtime._forced_fixed_opponent_policy_ids == ("default_forced",)
    assert "b1_noleague_baseline" not in runtime._opponent_models
    assert "b1_noleague_baseline" not in runtime._opponent_model_locks
    assert len(created_models) == 1
    assert created_models[0].loaded == 1
    assert created_models[0].evaluated == 1
    assert created_models[0].guidance_payload == (0.5, 0.25)
    np.testing.assert_array_equal(actor.fixed_opponent_policy_id_by_env, default_slots)
    assert actor.fixed_opponent_policy_id_by_env is not default_slots
    assert reset_actor_ids == [4, 4]
