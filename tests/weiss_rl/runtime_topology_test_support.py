from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch
from weiss_rl.runtime import QueueRuntime, QueueRuntimeConfig


def async_runtime_config(
    *,
    actor_count: int = 2,
    envs_per_actor: int = 64,
    unroll_length: int = 32,
    batch_unrolls_per_update: int = 96,
    queue_capacity_unrolls: int = 256,
) -> QueueRuntimeConfig:
    return QueueRuntimeConfig(
        mode="train_async_fast",
        actor_count=actor_count,
        envs_per_actor=envs_per_actor,
        unroll_length=unroll_length,
        batch_unrolls_per_update=batch_unrolls_per_update,
        queue_capacity_unrolls=queue_capacity_unrolls,
        profile="fast",
        base_seed=7,
        pass_action_id=51,
        actor_reload_interval_updates=1000,
    )


def runtime_stack(
    *,
    actor_device: str = "cpu",
    learner_device: str | None = None,
    actor_torch_threads: int = 1,
    collection_backend: str | None = None,
    mixed_precision: bool = False,
    compile_learner: bool = False,
    actor_policy_backend: str | None = None,
    structured_warmstart_enabled: bool | None = None,
    role: str = "baseline_noleague",
    league_enabled: bool = False,
    encoder_kind: str = "typed_v1",
    diverse_opponent_actor_count: int | None = None,
    diverse_model_actor_count: int | None = None,
) -> Any:
    system_kwargs: dict[str, Any] = {
        "actor_device": actor_device,
        "actor_torch_threads": actor_torch_threads,
    }
    if learner_device is not None:
        system_kwargs["learner_device"] = learner_device
    if collection_backend is not None:
        system_kwargs["collection_backend"] = collection_backend

    training_kwargs: dict[str, Any] = {
        "mixed_precision": mixed_precision,
        "compile_learner": compile_learner,
    }
    if actor_policy_backend is not None:
        training_kwargs["actor_policy_backend"] = actor_policy_backend
    if structured_warmstart_enabled is not None:
        training_kwargs["structured_warmstart"] = SimpleNamespace(enabled=structured_warmstart_enabled)
    if diverse_opponent_actor_count is not None:
        training_kwargs["diverse_opponent_actor_count"] = diverse_opponent_actor_count
    if diverse_model_actor_count is not None:
        training_kwargs["diverse_model_actor_count"] = diverse_model_actor_count

    return cast(
        Any,
        SimpleNamespace(
            config=SimpleNamespace(
                system=SimpleNamespace(**system_kwargs),
                training=SimpleNamespace(**training_kwargs),
                experiment=SimpleNamespace(role=role),
                league=SimpleNamespace(enabled=True, pfsp_window_episodes=50_000) if league_enabled else None,
                model=SimpleNamespace(encoder_kind=encoder_kind),
            )
        ),
    )


def patch_fake_actor_state_builder(monkeypatch: Any, *, envs_per_actor: int = 64, obs_dim: int = 8) -> None:
    def build_actor_state(
        self: QueueRuntime,
        *,
        model: Any,
        actor_id: int,
        shared_actor_model: Any | None = None,
        shared_compiled_actor_model: Any | None = None,
    ) -> Any:
        del self, shared_actor_model, shared_compiled_actor_model
        return cast(
            Any,
            SimpleNamespace(
                actor_id=actor_id,
                env=SimpleNamespace(close=lambda: None),
                model=model,
                compiled_model=None,
                opponent_policy_id_by_env=np.full((envs_per_actor,), "latest_policy_mirror", dtype=object),
                seat_hidden=torch.zeros((envs_per_actor, 4), device=torch.device("cpu")),
                opponent_hidden=torch.zeros((envs_per_actor, 4), device=torch.device("cpu")),
                current_batch=SimpleNamespace(obs=np.zeros((envs_per_actor, obs_dim), dtype=np.float32)),
                layout_name="i16_legal_ids",
                focal_seat_by_env=np.zeros((envs_per_actor,), dtype=np.int64),
                rng=np.random.default_rng(7),
                snapshot_version=0,
                next_unroll_seq=0,
            ),
        )

    monkeypatch.setattr(QueueRuntime, "_build_actor_state", build_actor_state)


class DummyDeviceModel:
    def to(self, device: torch.device) -> DummyDeviceModel:
        del device
        return self

    def eval(self) -> DummyDeviceModel:
        return self


class DummyProcessModel:
    def to(self, device: torch.device) -> DummyProcessModel:
        del device
        return self
