from __future__ import annotations

import numpy as np
from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.runtime import (
    QueueRuntime,
    RuntimeUnroll,
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


def _make_teacher_runtime_adapter(
    *,
    guidance_enabled: bool = True,
    aux_mode: str = "always",
    warmstart_updates: int = 0,
    learner_update: int = 0,
) -> QueueRuntime:
    runtime = object.__new__(QueueRuntime)
    action_catalog = _teacher_test_catalog()
    runtime._teacher_guidance_enabled = bool(guidance_enabled)
    runtime._teacher_aux_mode = str(aux_mode)
    runtime._teacher_guidance_warmstart_updates = int(warmstart_updates)
    runtime._current_learner_update = int(learner_update)
    runtime._teacher_policy = object()
    runtime._teacher_action_catalog = action_catalog
    runtime._teacher_family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    runtime._teacher_attack_type_index = {name: index for index, name in enumerate(action_catalog.attack_type_names)}
    return runtime
