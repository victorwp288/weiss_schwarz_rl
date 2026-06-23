from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch
from weiss_rl.runtime import QueueRuntime, QueueRuntimeConfig

from .runtime_opponent_pool_test_support import (
    make_opponent_pool_runtime,
    opponent_pool_config,
    write_snapshot_registry,
)


def test_maybe_publish_snapshot_tracks_effective_update_for_reused_weights() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any.config = QueueRuntimeConfig(
        mode="train_ordered",
        actor_count=1,
        envs_per_actor=1,
        unroll_length=1,
        batch_unrolls_per_update=1,
        queue_capacity_unrolls=1,
        profile="fast",
        base_seed=7,
        pass_action_id=51,
        actor_reload_interval_updates=1,
    )
    runtime_any._device = torch.device("cpu")
    runtime_any._collector_result_queue = None
    runtime_any._collector_control_queues = []
    runtime_any._collector_free_queues = []
    runtime_any._shared_actor_model = None
    runtime_any._bootstrap_models = None
    runtime_any._actors = [SimpleNamespace(model=torch.nn.Linear(2, 2), snapshot_version=0)]
    runtime_any._league_enabled = True
    runtime_any._league_config = SimpleNamespace(
        warmup=SimpleNamespace(first_updates=200),
        snapshot_pool_recent_size=8,
        snapshot_pool_champion_size=4,
        promotion_gate_enabled=True,
    )
    runtime_any._opponent_sampler = object()
    runtime_any._opponent_candidate_ids = ("policy_000007",)
    runtime_any._opponent_models = {"policy_000007": object()}
    runtime_any._current_learner_update = 0
    runtime_any._effective_learner_update = 0
    runtime_any._published_snapshot_update_by_fingerprint = {}
    runtime_any._last_published_snapshot_version = 0

    learner_model = torch.nn.Linear(2, 2)
    with torch.no_grad():
        learner_model.weight.fill_(1.0)
        learner_model.bias.fill_(0.5)

    QueueRuntime.maybe_publish_snapshot(
        runtime, learner_model=cast(Any, learner_model), learner_update_count=20, force=True
    )
    assert runtime_any._current_learner_update == 20
    assert runtime_any._effective_learner_update == 20
    assert QueueRuntime._pfsp_sampling_ready(runtime) is False

    restored_model = torch.nn.Linear(2, 2)
    restored_model.load_state_dict(learner_model.state_dict())

    QueueRuntime.maybe_publish_snapshot(
        runtime,
        learner_model=cast(Any, restored_model),
        learner_update_count=220,
        force=True,
    )
    assert runtime_any._current_learner_update == 220
    assert runtime_any._effective_learner_update == 20
    assert QueueRuntime._pfsp_sampling_ready(runtime) is False


def test_refresh_opponent_pool_uses_effective_update_for_champion_age(tmp_path: Path) -> None:
    _run_dir, registry_path = write_snapshot_registry(
        tmp_path,
        [("policy_000120", 120)],
        champions=("policy_000120",),
    )
    runtime = make_opponent_pool_runtime(
        registry_path,
        opponent_pool_config(
            promotion_gate_enabled=True,
            pool=SimpleNamespace(champion_max_age_updates=50),
        ),
        current_update=220,
        effective_update=20,
    )

    QueueRuntime.refresh_opponent_pool(runtime)

    assert runtime._opponent_champion_ids == ("policy_000120",)
    assert runtime._opponent_candidate_ids == ("policy_000120",)
