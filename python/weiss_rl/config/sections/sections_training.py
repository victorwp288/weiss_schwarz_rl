"""Training stack config section parser."""

from __future__ import annotations

from typing import Any

from weiss_rl.config.loading.parsing_utils import (
    require_bool,
    require_choice,
    require_float,
    require_int,
    require_text,
)
from weiss_rl.config.models import (
    TrainingActionSurfaceConfig,
    TrainingCheckpointingConfig,
    TrainingConfig,
    TrainingExplorationConfig,
    TrainingOptimizerConfig,
    TrainingPpoConfig,
    TrainingPrecisionConfig,
    TrainingRolloutConfig,
    TrainingStructuredMetricsConfig,
    TrainingTeacherAuxConfig,
    TrainingVTraceConfig,
)
from weiss_rl.config.sections.sections_training_schema import (
    TRAINING_ACTOR_POLICY_BACKENDS,
    TRAINING_ALGORITHMS,
    TRAINING_ENTROPY_SCOPES,
    TRAINING_FIXED_MODEL_OPPONENT_ACTION_SELECTIONS,
    TRAINING_FIXED_OPPONENT_BACKENDS,
    TRAINING_STRUCTURED_METRICS_MODES,
    TRAINING_TEACHER_AUX_MODES,
)

from .sections_training_sections import resolve_training_section_mappings
from .sections_training_structured_aux import (
    parse_training_structured_aux_config,
    parse_training_structured_warmstart_config,
)


def parse_training_config(body: dict[str, Any]) -> TrainingConfig:
    sections = resolve_training_section_mappings(body)
    rollout = sections.rollout
    optimizer = sections.optimizer
    exploration = sections.exploration
    precision = sections.precision
    checkpointing = sections.checkpointing
    vtrace = sections.vtrace
    ppo = sections.ppo
    structured_aux = sections.structured_aux
    structured_warmstart = sections.structured_warmstart
    structured_metrics = sections.structured_metrics
    teacher_aux = sections.teacher_aux
    action_surface = sections.action_surface
    actor_sampling_temperature = require_float(
        exploration.get("actor_sampling_temperature", 1.0),
        field_name="training.exploration.actor_sampling_temperature",
    )
    if actor_sampling_temperature <= 0.0:
        raise ValueError("training.exploration.actor_sampling_temperature must be > 0")

    profile_timers = require_bool(body.get("profile_timers", False), field_name="training.profile_timers")
    torch_profiler = require_bool(body.get("torch_profiler", False), field_name="training.torch_profiler")
    fixed_opponent_backend = require_choice(
        body.get("fixed_opponent_backend", "python_scalar"),
        field_name="training.fixed_opponent_backend",
        allowed=TRAINING_FIXED_OPPONENT_BACKENDS,
    )
    fixed_model_opponent_action_selection = require_choice(
        body.get("fixed_model_opponent_action_selection", "sample"),
        field_name="training.fixed_model_opponent_action_selection",
        allowed=TRAINING_FIXED_MODEL_OPPONENT_ACTION_SELECTIONS,
    )
    actor_policy_backend = require_choice(
        body.get("actor_policy_backend", "model"),
        field_name="training.actor_policy_backend",
        allowed=TRAINING_ACTOR_POLICY_BACKENDS,
    )
    actor_heuristic_fraction = require_float(
        body.get("actor_heuristic_fraction", 1.0),
        field_name="training.actor_heuristic_fraction",
    )
    if actor_heuristic_fraction < 0.0 or actor_heuristic_fraction > 1.0:
        raise ValueError(
            f"training.actor_heuristic_fraction must be between 0.0 and 1.0 inclusive, got {actor_heuristic_fraction}"
        )
    actor_heuristic_start_updates = require_int(
        body.get("actor_heuristic_start_updates", 0),
        field_name="training.actor_heuristic_start_updates",
        minimum=0,
    )
    actor_heuristic_end_updates = require_int(
        body.get("actor_heuristic_end_updates", -1),
        field_name="training.actor_heuristic_end_updates",
        minimum=-1,
    )
    if actor_heuristic_end_updates >= 0 and actor_heuristic_end_updates < actor_heuristic_start_updates:
        raise ValueError("training.actor_heuristic_end_updates must be >= training.actor_heuristic_start_updates")
    actor_heuristic_final_fraction = require_float(
        body.get("actor_heuristic_final_fraction", actor_heuristic_fraction),
        field_name="training.actor_heuristic_final_fraction",
    )
    if actor_heuristic_final_fraction < 0.0 or actor_heuristic_final_fraction > 1.0:
        raise ValueError(
            "training.actor_heuristic_final_fraction must be between 0.0 and 1.0 inclusive, "
            f"got {actor_heuristic_final_fraction}"
        )
    heuristic_actor_hidden_state_tracking = require_bool(
        body.get("heuristic_actor_hidden_state_tracking", True),
        field_name="training.heuristic_actor_hidden_state_tracking",
    )
    train_on_heuristic_actor_rows = require_bool(
        body.get("train_on_heuristic_actor_rows", True),
        field_name="training.train_on_heuristic_actor_rows",
    )
    diverse_opponent_actor_count = require_int(
        body.get("diverse_opponent_actor_count", 0),
        field_name="training.diverse_opponent_actor_count",
        minimum=0,
    )
    diverse_model_actor_count = require_int(
        body.get("diverse_model_actor_count", 0),
        field_name="training.diverse_model_actor_count",
        minimum=0,
    )
    diverse_opponent_batch_fraction = require_float(
        body.get("diverse_opponent_batch_fraction", 0.0),
        field_name="training.diverse_opponent_batch_fraction",
    )
    if diverse_opponent_batch_fraction < 0.0 or diverse_opponent_batch_fraction > 1.0:
        raise ValueError(
            "training.diverse_opponent_batch_fraction must be between 0.0 and 1.0 inclusive, "
            f"got {diverse_opponent_batch_fraction}"
        )
    diverse_opponent_batch_wait_ms = require_int(
        body.get("diverse_opponent_batch_wait_ms", 0),
        field_name="training.diverse_opponent_batch_wait_ms",
        minimum=0,
    )
    structured_aux_config = parse_training_structured_aux_config(structured_aux)
    structured_warmstart_config = parse_training_structured_warmstart_config(structured_warmstart)

    return TrainingConfig(
        algorithm=require_choice(body["algorithm"], field_name="training.algorithm", allowed=TRAINING_ALGORITHMS),
        rollout=TrainingRolloutConfig(
            unroll_length=require_int(rollout["unroll_length"], field_name="training.rollout.unroll_length", minimum=1),
            batch_unrolls_per_update=require_int(
                rollout["batch_unrolls_per_update"],
                field_name="training.rollout.batch_unrolls_per_update",
                minimum=1,
            ),
        ),
        optimizer=TrainingOptimizerConfig(
            name=require_text(optimizer["name"], field_name="training.optimizer.name"),
            learning_rate=require_float(optimizer["learning_rate"], field_name="training.optimizer.learning_rate"),
            grad_norm_clip=require_float(optimizer["grad_norm_clip"], field_name="training.optimizer.grad_norm_clip"),
            value_loss_coef=require_float(
                optimizer["value_loss_coef"], field_name="training.optimizer.value_loss_coef"
            ),
        ),
        exploration=TrainingExplorationConfig(
            entropy_coef=require_float(exploration["entropy_coef"], field_name="training.exploration.entropy_coef"),
            entropy_anneal_to=require_float(
                exploration["entropy_anneal_to"], field_name="training.exploration.entropy_anneal_to"
            ),
            entropy_anneal_steps_updates=require_int(
                exploration["entropy_anneal_steps_updates"],
                field_name="training.exploration.entropy_anneal_steps_updates",
                minimum=1,
            ),
            entropy_scope=require_choice(
                exploration.get("entropy_scope", "candidate"),
                field_name="training.exploration.entropy_scope",
                allowed=TRAINING_ENTROPY_SCOPES,
            ),
            actor_sampling_temperature=actor_sampling_temperature,
        ),
        precision=TrainingPrecisionConfig(
            mixed_precision=require_bool(precision["mixed_precision"], field_name="training.precision.mixed_precision"),
            compile_learner=require_bool(precision["compile_learner"], field_name="training.precision.compile_learner"),
            compile_actor_inference=require_bool(
                precision.get("compile_actor_inference", False),
                field_name="training.precision.compile_actor_inference",
            ),
            masking_math_float32=require_bool(
                precision["masking_math_float32"],
                field_name="training.precision.masking_math_float32",
            ),
        ),
        profile_timers=profile_timers,
        torch_profiler=torch_profiler,
        checkpointing=TrainingCheckpointingConfig(
            checkpoint_interval_updates=require_int(
                checkpointing["checkpoint_interval_updates"],
                field_name="training.checkpointing.checkpoint_interval_updates",
                minimum=1,
            ),
            snapshot_interval_updates=require_int(
                checkpointing["snapshot_interval_updates"],
                field_name="training.checkpointing.snapshot_interval_updates",
                minimum=1,
            ),
            actor_reload_interval_updates=require_int(
                checkpointing["actor_reload_interval_updates"],
                field_name="training.checkpointing.actor_reload_interval_updates",
                minimum=1,
            ),
        ),
        vtrace=TrainingVTraceConfig(
            rho_bar=require_float(vtrace["rho_bar"], field_name="training.vtrace.rho_bar"),
            c_bar=require_float(vtrace["c_bar"], field_name="training.vtrace.c_bar"),
        ),
        ppo=TrainingPpoConfig(
            clip_epsilon=require_float(ppo.get("clip_epsilon", 0.2), field_name="training.ppo.clip_epsilon"),
            value_clip_epsilon=require_float(
                ppo.get("value_clip_epsilon", 0.2),
                field_name="training.ppo.value_clip_epsilon",
            ),
            gae_lambda=require_float(ppo.get("gae_lambda", 0.95), field_name="training.ppo.gae_lambda"),
            epochs=require_int(ppo.get("epochs", 4), field_name="training.ppo.epochs", minimum=1),
            target_kl=require_float(ppo.get("target_kl", 0.0), field_name="training.ppo.target_kl"),
            normalize_advantages=require_bool(
                ppo.get("normalize_advantages", True),
                field_name="training.ppo.normalize_advantages",
            ),
        ),
        structured_aux=structured_aux_config,
        structured_warmstart=structured_warmstart_config,
        structured_metrics=TrainingStructuredMetricsConfig(
            mode=require_choice(
                structured_metrics.get("mode", "off"),
                field_name="training.structured_metrics.mode",
                allowed=TRAINING_STRUCTURED_METRICS_MODES,
            ),
        ),
        teacher_aux=TrainingTeacherAuxConfig(
            mode=require_choice(
                teacher_aux.get("mode", "always"),
                field_name="training.teacher_aux.mode",
                allowed=TRAINING_TEACHER_AUX_MODES,
            ),
        ),
        action_surface=TrainingActionSurfaceConfig(
            mulligan_force_confirm_after_select=require_bool(
                action_surface.get("mulligan_force_confirm_after_select", False),
                field_name="training.action_surface.mulligan_force_confirm_after_select",
            ),
            force_pass_over_main_move_only=require_bool(
                action_surface.get("force_pass_over_main_move_only", False),
                field_name="training.action_surface.force_pass_over_main_move_only",
            ),
            main_move_only_max_consecutive=require_int(
                action_surface.get("main_move_only_max_consecutive", 0),
                field_name="training.action_surface.main_move_only_max_consecutive",
                minimum=0,
            ),
            force_attack_over_pass_when_attack_legal=require_bool(
                action_surface.get("force_attack_over_pass_when_attack_legal", False),
                field_name="training.action_surface.force_attack_over_pass_when_attack_legal",
            ),
        ),
        fixed_opponent_backend=fixed_opponent_backend,
        fixed_model_opponent_action_selection=fixed_model_opponent_action_selection,
        actor_policy_backend=actor_policy_backend,
        actor_heuristic_fraction=actor_heuristic_fraction,
        actor_heuristic_start_updates=actor_heuristic_start_updates,
        actor_heuristic_end_updates=actor_heuristic_end_updates,
        actor_heuristic_final_fraction=actor_heuristic_final_fraction,
        train_on_heuristic_actor_rows=train_on_heuristic_actor_rows,
        diverse_opponent_actor_count=diverse_opponent_actor_count,
        diverse_model_actor_count=diverse_model_actor_count,
        diverse_opponent_batch_fraction=diverse_opponent_batch_fraction,
        diverse_opponent_batch_wait_ms=diverse_opponent_batch_wait_ms,
        heuristic_actor_hidden_state_tracking=heuristic_actor_hidden_state_tracking,
    )
