"""Core stack config section parsers."""

from __future__ import annotations

from typing import Any

from .models import ExperimentConfig, SystemConfig, SystemProfileConfig
from .parsing_utils import reject_unknown_keys, require_choice, require_int, require_mapping, require_text

EXPERIMENT_ROLES = frozenset(
    {
        "main",
        "baseline_noleague",
        "baseline_norecurrence",
        "baseline_ppo_lite",
        "ablation_discount",
        "ablation_reward",
        "ablation_guided",
        "guided_league_seed",
        "guided_league_bootstrap",
    }
)


def parse_experiment_config(body: dict[str, Any]) -> ExperimentConfig:
    reject_unknown_keys(body, allowed={"role"}, context="experiment")
    return ExperimentConfig(
        role=require_choice(body["role"], field_name="experiment.role", allowed=EXPERIMENT_ROLES),
    )


def parse_system_config(body: dict[str, Any]) -> SystemConfig:
    reject_unknown_keys(
        body,
        allowed={
            "profile",
            "mp_start_method",
            "collection_backend",
            "learner_device",
            "actor_device",
            "actor_process_count",
            "envs_per_actor",
            "total_envs",
            "actor_torch_threads",
            "learner_torch_threads",
            "actor_queue_capacity_unrolls",
            "learner_prefetch_batches",
        },
        context="system",
    )
    profile = require_mapping(body["profile"], context="system.profile")
    reject_unknown_keys(
        profile,
        allowed={"training", "local_iteration", "ci_invariant_testing"},
        context="system.profile",
    )
    return SystemConfig(
        profile=SystemProfileConfig(
            training=require_text(profile["training"], field_name="system.profile.training"),
            local_iteration=require_text(profile["local_iteration"], field_name="system.profile.local_iteration"),
            ci_invariant_testing=require_text(
                profile["ci_invariant_testing"],
                field_name="system.profile.ci_invariant_testing",
            ),
        ),
        mp_start_method=require_text(body["mp_start_method"], field_name="system.mp_start_method"),
        collection_backend=require_choice(
            body.get("collection_backend", "auto"),
            field_name="system.collection_backend",
            allowed=("auto", "central", "process"),
        ),
        learner_device=require_text(body["learner_device"], field_name="system.learner_device"),
        actor_device=require_text(body["actor_device"], field_name="system.actor_device"),
        actor_process_count=require_int(
            body["actor_process_count"], field_name="system.actor_process_count", minimum=1
        ),
        envs_per_actor=require_int(body["envs_per_actor"], field_name="system.envs_per_actor", minimum=1),
        total_envs=require_int(body["total_envs"], field_name="system.total_envs", minimum=1),
        actor_torch_threads=require_int(
            body["actor_torch_threads"], field_name="system.actor_torch_threads", minimum=1
        ),
        learner_torch_threads=require_int(
            body["learner_torch_threads"], field_name="system.learner_torch_threads", minimum=1
        ),
        actor_queue_capacity_unrolls=require_int(
            body["actor_queue_capacity_unrolls"],
            field_name="system.actor_queue_capacity_unrolls",
            minimum=1,
        ),
        learner_prefetch_batches=require_int(
            body["learner_prefetch_batches"],
            field_name="system.learner_prefetch_batches",
            minimum=1,
        ),
    )
