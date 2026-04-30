"""Parser for the large training section of stack configs."""

from __future__ import annotations

from typing import Any

from .models import (
    TrainingCheckpointingConfig,
    TrainingConfig,
    TrainingCounterfactualPositiveConfig,
    TrainingExplorationConfig,
    TrainingMainResidualPolicyConfig,
    TrainingOptimizerConfig,
    TrainingPpoConfig,
    TrainingPrecisionConfig,
    TrainingRawB1DistillConfig,
    TrainingResidualOpponentPolicyConfig,
    TrainingRolloutConfig,
    TrainingScalingConfig,
    TrainingStructuredAuxConfig,
    TrainingStructuredMetricsConfig,
    TrainingStructuredWarmstartConfig,
    TrainingTeacherAuxConfig,
    TrainingVTraceConfig,
)
from .validation import (
    reject_unknown_keys as _reject_unknown_keys,
)
from .validation import (
    require_auto_or_int as _require_auto_or_int,
)
from .validation import (
    require_bool as _require_bool,
)
from .validation import (
    require_choice as _require_choice,
)
from .validation import (
    require_float as _require_float,
)
from .validation import (
    require_int as _require_int,
)
from .validation import (
    require_mapping as _require_mapping,
)
from .validation import (
    require_str_list as _require_str_list,
)
from .validation import (
    require_text as _require_text,
)

_TRAINING_ALGORITHMS = frozenset(
    {"impala_vtrace_gru", "impala_vtrace_ff", "ppo_lite_masked_v1", "structured_v2", "impala_vtrace_structured_v1"}
)
_TRAINING_STRUCTURED_METRICS_MODES = frozenset({"off", "sampled", "full"})
_TRAINING_TEACHER_AUX_MODES = frozenset({"off", "warmstart_only", "always"})
_TRAINING_FIXED_OPPONENT_BACKENDS = frozenset({"python_scalar", "python_batched", "simulator_native"})
_TRAINING_ACTOR_POLICY_BACKENDS = frozenset({"model", "heuristic_public"})
_TRAINING_PUBLIC_HEURISTIC_PROFILES = frozenset({"base", "aggressive", "control"})
_TRAINING_PUBLIC_HEURISTIC_PROFILE_MODES = frozenset({"mixture", "cycle"})
_TRAINING_NATIVE_ROLLOUT_PROFILE_MODES = frozenset({"fixed", "cycle", "random"})
_TRAINING_OPTIMIZER_BACKENDS = frozenset({"auto", "default", "foreach", "fused"})
_TRAINING_LEARNER_PARALLELISM = frozenset({"auto", "single", "ddp", "ddp_cpu_test"})
_TRAINING_ACTOR_TOPOLOGIES = frozenset({"auto", "manual"})


def parse_training_config(body: dict[str, Any]) -> TrainingConfig:
    _reject_unknown_keys(
        body,
        allowed={
            "algorithm",
            "rollout",
            "optimizer",
            "exploration",
            "precision",
            "scaling",
            "profile_timers",
            "torch_profiler",
            "freeze_parameter_prefixes",
            "checkpointing",
            "vtrace",
            "ppo",
            "structured_aux",
            "structured_warmstart",
            "structured_metrics",
            "teacher_aux",
            "raw_b1_distill",
            "counterfactual_positive",
            "fixed_opponent_backend",
            "actor_policy_backend",
            "actor_heuristic_fraction",
            "actor_heuristic_start_updates",
            "actor_heuristic_end_updates",
            "actor_heuristic_final_fraction",
            "train_on_heuristic_actor_rows",
            "policy_loss_coef",
            "behavior_action_bc_coef",
            "reference_policy_top_action_bc_coef",
            "reference_policy_top_action_bc_final_coef",
            "reference_policy_top_action_bc_start_updates",
            "reference_policy_top_action_bc_end_updates",
            "b1_opponent_reference_policy_top_action_bc_coef",
            "b1_second_seat_positive_advantage_policy_coef",
            "b1_second_seat_reference_top_action_avoidance_coef",
            "reference_policy_top_action_family_bc_coef",
            "reference_policy_top_action_family_bc_final_coef",
            "reference_policy_top_action_family_bc_start_updates",
            "reference_policy_top_action_family_bc_end_updates",
            "reference_policy_id",
            "main_residual_policy",
            "diverse_opponent_actor_count",
            "diverse_model_actor_count",
            "diverse_opponent_policy_id",
            "diverse_opponent_policy_ids",
            "residual_opponent_policies",
            "diverse_opponent_batch_fraction",
            "diverse_opponent_batch_wait_ms",
            "collect_batch_prefetch_enabled",
            "heuristic_native_rollout_enabled",
            "heuristic_native_rollout_profile",
            "heuristic_native_rollout_profiles",
            "heuristic_native_rollout_profile_mode",
            "heuristic_actor_hidden_state_tracking",
        },
        context="training",
    )
    rollout = _require_mapping(body["rollout"], context="training.rollout")
    optimizer = _require_mapping(body["optimizer"], context="training.optimizer")
    exploration = _require_mapping(body["exploration"], context="training.exploration")
    precision = _require_mapping(body["precision"], context="training.precision")
    checkpointing = _require_mapping(body["checkpointing"], context="training.checkpointing")
    vtrace = _require_mapping(body["vtrace"], context="training.vtrace")
    ppo = _require_mapping(body.get("ppo", {}), context="training.ppo")
    structured_aux = _require_mapping(body.get("structured_aux", {}), context="training.structured_aux")
    structured_warmstart = _require_mapping(
        body.get("structured_warmstart", {}),
        context="training.structured_warmstart",
    )
    structured_metrics = _require_mapping(
        body.get("structured_metrics", {}),
        context="training.structured_metrics",
    )
    teacher_aux = _require_mapping(body.get("teacher_aux", {}), context="training.teacher_aux")
    raw_b1_distill = _require_mapping(body.get("raw_b1_distill", {}), context="training.raw_b1_distill")
    counterfactual_positive = _require_mapping(
        body.get("counterfactual_positive", {}),
        context="training.counterfactual_positive",
    )
    main_residual_policy = _require_mapping(
        body.get("main_residual_policy", {}),
        context="training.main_residual_policy",
    )
    scaling = _require_mapping(body.get("scaling", {}), context="training.scaling")

    _reject_unknown_keys(rollout, allowed={"unroll_length", "batch_unrolls_per_update"}, context="training.rollout")
    _reject_unknown_keys(
        optimizer,
        allowed={"name", "learning_rate", "grad_norm_clip", "value_loss_coef", "backend"},
        context="training.optimizer",
    )
    _reject_unknown_keys(
        exploration,
        allowed={"entropy_coef", "entropy_anneal_to", "entropy_anneal_steps_updates"},
        context="training.exploration",
    )
    _reject_unknown_keys(
        precision,
        allowed={"mixed_precision", "compile_learner", "compile_actor_inference", "masking_math_float32"},
        context="training.precision",
    )
    _reject_unknown_keys(
        checkpointing,
        allowed={"checkpoint_interval_updates", "snapshot_interval_updates", "actor_reload_interval_updates"},
        context="training.checkpointing",
    )
    _reject_unknown_keys(vtrace, allowed={"rho_bar", "c_bar"}, context="training.vtrace")
    _reject_unknown_keys(
        ppo,
        allowed={"clip_epsilon", "value_clip_epsilon", "gae_lambda", "epochs", "target_kl", "normalize_advantages"},
        context="training.ppo",
    )
    _reject_unknown_keys(
        structured_aux,
        allowed={
            "enabled",
            "teacher_family_coef",
            "teacher_slot_coef",
            "teacher_move_source_coef",
            "teacher_attack_type_coef",
            "teacher_action_coef",
            "teacher_same_family_action_coef",
            "teacher_public_heuristic_coef",
            "teacher_public_main_move_coef",
            "teacher_development_pass_suppression_coef",
            "teacher_public_heuristic_start_updates",
            "teacher_public_heuristic_end_updates",
            "teacher_public_heuristic_final_coef",
            "teacher_public_heuristic_temperature",
            "teacher_public_heuristic_families",
            "teacher_public_heuristic_profiles",
            "teacher_public_heuristic_profile_mode",
            "teacher_public_heuristic_profiles_end_updates",
            "teacher_public_heuristic_label_profile",
        },
        context="training.structured_aux",
    )
    _reject_unknown_keys(
        structured_warmstart,
        allowed={
            "enabled",
            "updates",
            "teacher_family_coef",
            "teacher_slot_coef",
            "teacher_move_source_coef",
            "teacher_attack_type_coef",
            "teacher_action_coef",
            "teacher_same_family_action_coef",
            "teacher_public_heuristic_coef",
            "teacher_public_heuristic_temperature",
            "teacher_public_heuristic_families",
            "teacher_public_heuristic_profiles",
            "teacher_public_heuristic_profile_mode",
            "teacher_public_heuristic_profiles_end_updates",
        },
        context="training.structured_warmstart",
    )
    _reject_unknown_keys(structured_metrics, allowed={"mode"}, context="training.structured_metrics")
    _reject_unknown_keys(teacher_aux, allowed={"mode"}, context="training.teacher_aux")
    _reject_unknown_keys(
        raw_b1_distill,
        allowed={
            "enabled",
            "teacher_policy_id",
            "teacher_surface",
            "student_surface",
            "coef",
            "final_coef",
            "start_updates",
            "end_updates",
            "top_k",
            "temperature",
            "top_action_ce_coef",
            "teacher_public_heuristic_bias_scale",
            "student_public_heuristic_bias_scale",
            "mask_counterfactual_positive_states",
        },
        context="training.raw_b1_distill",
    )
    _reject_unknown_keys(
        counterfactual_positive,
        allowed={
            "enabled",
            "label_dirs",
            "coef",
            "final_coef",
            "start_updates",
            "end_updates",
            "margin_coef",
            "margin",
            "max_labels",
        },
        context="training.counterfactual_positive",
    )
    _reject_unknown_keys(
        scaling,
        allowed={
            "learner_parallelism",
            "learner_gpu_count",
            "actor_topology",
            "target_envs_per_gpu",
            "min_envs_per_actor",
            "max_envs_per_actor",
            "max_actor_process_count",
            "reserve_cpu_cores",
            "learner_cpu_cores_per_gpu",
            "queue_depth_multiplier",
            "ram_queue_fraction",
            "vram_fraction",
        },
        context="training.scaling",
    )

    profile_timers = _require_bool(body.get("profile_timers", False), field_name="training.profile_timers")
    torch_profiler = _require_bool(body.get("torch_profiler", False), field_name="training.torch_profiler")
    freeze_parameter_prefixes = tuple(
        prefix.strip()
        for prefix in _require_str_list(
            body.get("freeze_parameter_prefixes", []),
            field_name="training.freeze_parameter_prefixes",
        )
        if prefix.strip()
    )
    fixed_opponent_backend = _require_choice(
        body.get("fixed_opponent_backend", "python_scalar"),
        field_name="training.fixed_opponent_backend",
        allowed=_TRAINING_FIXED_OPPONENT_BACKENDS,
    )
    actor_policy_backend = _require_choice(
        body.get("actor_policy_backend", "model"),
        field_name="training.actor_policy_backend",
        allowed=_TRAINING_ACTOR_POLICY_BACKENDS,
    )
    actor_heuristic_fraction = _require_float(
        body.get("actor_heuristic_fraction", 1.0),
        field_name="training.actor_heuristic_fraction",
    )
    if actor_heuristic_fraction < 0.0 or actor_heuristic_fraction > 1.0:
        raise ValueError(
            f"training.actor_heuristic_fraction must be between 0.0 and 1.0 inclusive, got {actor_heuristic_fraction}"
        )
    actor_heuristic_start_updates = _require_int(
        body.get("actor_heuristic_start_updates", 0),
        field_name="training.actor_heuristic_start_updates",
        minimum=0,
    )
    actor_heuristic_end_updates = _require_int(
        body.get("actor_heuristic_end_updates", -1),
        field_name="training.actor_heuristic_end_updates",
        minimum=-1,
    )
    if actor_heuristic_end_updates >= 0 and actor_heuristic_end_updates < actor_heuristic_start_updates:
        raise ValueError("training.actor_heuristic_end_updates must be >= training.actor_heuristic_start_updates")
    actor_heuristic_final_fraction = _require_float(
        body.get("actor_heuristic_final_fraction", actor_heuristic_fraction),
        field_name="training.actor_heuristic_final_fraction",
    )
    if actor_heuristic_final_fraction < 0.0 or actor_heuristic_final_fraction > 1.0:
        raise ValueError(
            "training.actor_heuristic_final_fraction must be between 0.0 and 1.0 inclusive, "
            f"got {actor_heuristic_final_fraction}"
        )
    heuristic_actor_hidden_state_tracking = _require_bool(
        body.get("heuristic_actor_hidden_state_tracking", True),
        field_name="training.heuristic_actor_hidden_state_tracking",
    )
    train_on_heuristic_actor_rows = _require_bool(
        body.get("train_on_heuristic_actor_rows", True),
        field_name="training.train_on_heuristic_actor_rows",
    )
    policy_loss_coef = _require_float(body.get("policy_loss_coef", 1.0), field_name="training.policy_loss_coef")
    if policy_loss_coef < 0.0:
        raise ValueError("training.policy_loss_coef must be >= 0.0")
    behavior_action_bc_coef = _require_float(
        body.get("behavior_action_bc_coef", 0.0),
        field_name="training.behavior_action_bc_coef",
    )
    if behavior_action_bc_coef < 0.0:
        raise ValueError("training.behavior_action_bc_coef must be >= 0.0")
    reference_policy_top_action_bc_coef = _require_float(
        body.get("reference_policy_top_action_bc_coef", 0.0),
        field_name="training.reference_policy_top_action_bc_coef",
    )
    if reference_policy_top_action_bc_coef < 0.0:
        raise ValueError("training.reference_policy_top_action_bc_coef must be >= 0.0")
    reference_policy_top_action_bc_final_coef = _require_float(
        body.get("reference_policy_top_action_bc_final_coef", reference_policy_top_action_bc_coef),
        field_name="training.reference_policy_top_action_bc_final_coef",
    )
    if reference_policy_top_action_bc_final_coef < 0.0:
        raise ValueError("training.reference_policy_top_action_bc_final_coef must be >= 0.0")
    reference_policy_top_action_bc_start_updates = _require_int(
        body.get("reference_policy_top_action_bc_start_updates", 0),
        field_name="training.reference_policy_top_action_bc_start_updates",
        minimum=0,
    )
    reference_policy_top_action_bc_end_updates = _require_int(
        body.get("reference_policy_top_action_bc_end_updates", -1),
        field_name="training.reference_policy_top_action_bc_end_updates",
        minimum=-1,
    )
    if (
        reference_policy_top_action_bc_end_updates >= 0
        and reference_policy_top_action_bc_end_updates < reference_policy_top_action_bc_start_updates
    ):
        raise ValueError(
            "training.reference_policy_top_action_bc_end_updates must be >= "
            "training.reference_policy_top_action_bc_start_updates"
        )
    b1_opponent_reference_policy_top_action_bc_coef = _require_float(
        body.get("b1_opponent_reference_policy_top_action_bc_coef", 0.0),
        field_name="training.b1_opponent_reference_policy_top_action_bc_coef",
    )
    if b1_opponent_reference_policy_top_action_bc_coef < 0.0:
        raise ValueError("training.b1_opponent_reference_policy_top_action_bc_coef must be >= 0.0")
    b1_second_seat_positive_advantage_policy_coef = _require_float(
        body.get("b1_second_seat_positive_advantage_policy_coef", 0.0),
        field_name="training.b1_second_seat_positive_advantage_policy_coef",
    )
    if b1_second_seat_positive_advantage_policy_coef < 0.0:
        raise ValueError("training.b1_second_seat_positive_advantage_policy_coef must be >= 0.0")
    b1_second_seat_reference_top_action_avoidance_coef = _require_float(
        body.get("b1_second_seat_reference_top_action_avoidance_coef", 0.0),
        field_name="training.b1_second_seat_reference_top_action_avoidance_coef",
    )
    if b1_second_seat_reference_top_action_avoidance_coef < 0.0:
        raise ValueError("training.b1_second_seat_reference_top_action_avoidance_coef must be >= 0.0")
    reference_policy_top_action_family_bc_coef = _require_float(
        body.get("reference_policy_top_action_family_bc_coef", 0.0),
        field_name="training.reference_policy_top_action_family_bc_coef",
    )
    if reference_policy_top_action_family_bc_coef < 0.0:
        raise ValueError("training.reference_policy_top_action_family_bc_coef must be >= 0.0")
    reference_policy_top_action_family_bc_final_coef = _require_float(
        body.get(
            "reference_policy_top_action_family_bc_final_coef",
            reference_policy_top_action_family_bc_coef,
        ),
        field_name="training.reference_policy_top_action_family_bc_final_coef",
    )
    if reference_policy_top_action_family_bc_final_coef < 0.0:
        raise ValueError("training.reference_policy_top_action_family_bc_final_coef must be >= 0.0")
    reference_policy_top_action_family_bc_start_updates = _require_int(
        body.get("reference_policy_top_action_family_bc_start_updates", 0),
        field_name="training.reference_policy_top_action_family_bc_start_updates",
        minimum=0,
    )
    reference_policy_top_action_family_bc_end_updates = _require_int(
        body.get("reference_policy_top_action_family_bc_end_updates", -1),
        field_name="training.reference_policy_top_action_family_bc_end_updates",
        minimum=-1,
    )
    if (
        reference_policy_top_action_family_bc_end_updates >= 0
        and reference_policy_top_action_family_bc_end_updates < reference_policy_top_action_family_bc_start_updates
    ):
        raise ValueError(
            "training.reference_policy_top_action_family_bc_end_updates must be >= "
            "training.reference_policy_top_action_family_bc_start_updates"
        )
    reference_policy_id = str(body.get("reference_policy_id", "") or "").strip()
    raw_b1_distill_enabled = _require_bool(
        raw_b1_distill.get("enabled", False),
        field_name="training.raw_b1_distill.enabled",
    )
    raw_b1_distill_teacher_policy_id = str(
        raw_b1_distill.get("teacher_policy_id", reference_policy_id or "b1_noleague_baseline") or ""
    ).strip()
    if raw_b1_distill_enabled and not raw_b1_distill_teacher_policy_id:
        raise ValueError("training.raw_b1_distill.teacher_policy_id must be non-empty when enabled")
    raw_b1_distill_teacher_surface = str(raw_b1_distill.get("teacher_surface", "raw_s0") or "raw_s0").strip()
    raw_b1_distill_student_surface = str(raw_b1_distill.get("student_surface", "raw_s0") or "raw_s0").strip()
    raw_b1_distill_coef = _require_float(
        raw_b1_distill.get("coef", 0.0),
        field_name="training.raw_b1_distill.coef",
    )
    if raw_b1_distill_coef < 0.0:
        raise ValueError("training.raw_b1_distill.coef must be >= 0.0")
    raw_b1_distill_final_coef = _require_float(
        raw_b1_distill.get("final_coef", raw_b1_distill_coef),
        field_name="training.raw_b1_distill.final_coef",
    )
    if raw_b1_distill_final_coef < 0.0:
        raise ValueError("training.raw_b1_distill.final_coef must be >= 0.0")
    raw_b1_distill_start_updates = _require_int(
        raw_b1_distill.get("start_updates", 0),
        field_name="training.raw_b1_distill.start_updates",
        minimum=0,
    )
    raw_b1_distill_end_updates = _require_int(
        raw_b1_distill.get("end_updates", -1),
        field_name="training.raw_b1_distill.end_updates",
        minimum=-1,
    )
    if raw_b1_distill_end_updates >= 0 and raw_b1_distill_end_updates < raw_b1_distill_start_updates:
        raise ValueError("training.raw_b1_distill.end_updates must be >= training.raw_b1_distill.start_updates")
    raw_b1_distill_top_k = _require_int(
        raw_b1_distill.get("top_k", 16),
        field_name="training.raw_b1_distill.top_k",
        minimum=1,
    )
    raw_b1_distill_temperature = _require_float(
        raw_b1_distill.get("temperature", 1.5),
        field_name="training.raw_b1_distill.temperature",
    )
    if raw_b1_distill_temperature <= 0.0:
        raise ValueError("training.raw_b1_distill.temperature must be > 0.0")
    raw_b1_distill_top_action_ce_coef = _require_float(
        raw_b1_distill.get("top_action_ce_coef", 0.0),
        field_name="training.raw_b1_distill.top_action_ce_coef",
    )
    if raw_b1_distill_top_action_ce_coef < 0.0:
        raise ValueError("training.raw_b1_distill.top_action_ce_coef must be >= 0.0")
    raw_b1_distill_teacher_bias = _require_float(
        raw_b1_distill.get("teacher_public_heuristic_bias_scale", 0.0),
        field_name="training.raw_b1_distill.teacher_public_heuristic_bias_scale",
    )
    raw_b1_distill_student_bias = _require_float(
        raw_b1_distill.get("student_public_heuristic_bias_scale", 0.0),
        field_name="training.raw_b1_distill.student_public_heuristic_bias_scale",
    )
    if raw_b1_distill_teacher_bias < 0.0 or raw_b1_distill_student_bias < 0.0:
        raise ValueError("training.raw_b1_distill public heuristic bias scales must be >= 0.0")
    if raw_b1_distill_enabled and (
        raw_b1_distill_teacher_surface != "raw_s0" or raw_b1_distill_student_surface != "raw_s0"
    ):
        raise ValueError(
            "training.raw_b1_distill currently supports only teacher_surface=raw_s0 and student_surface=raw_s0"
        )
    raw_b1_distill_config = TrainingRawB1DistillConfig(
        enabled=raw_b1_distill_enabled,
        teacher_policy_id=raw_b1_distill_teacher_policy_id,
        teacher_surface=raw_b1_distill_teacher_surface,
        student_surface=raw_b1_distill_student_surface,
        coef=raw_b1_distill_coef,
        final_coef=raw_b1_distill_final_coef,
        start_updates=raw_b1_distill_start_updates,
        end_updates=raw_b1_distill_end_updates,
        top_k=raw_b1_distill_top_k,
        temperature=raw_b1_distill_temperature,
        top_action_ce_coef=raw_b1_distill_top_action_ce_coef,
        teacher_public_heuristic_bias_scale=raw_b1_distill_teacher_bias,
        student_public_heuristic_bias_scale=raw_b1_distill_student_bias,
    )
    counterfactual_positive_enabled = _require_bool(
        counterfactual_positive.get("enabled", False),
        field_name="training.counterfactual_positive.enabled",
    )
    counterfactual_positive_label_dirs = tuple(
        str(path).strip()
        for path in _require_str_list(
            counterfactual_positive.get("label_dirs", []),
            field_name="training.counterfactual_positive.label_dirs",
        )
        if str(path).strip()
    )
    if counterfactual_positive_enabled and not counterfactual_positive_label_dirs:
        raise ValueError("training.counterfactual_positive.label_dirs must be non-empty when enabled")
    counterfactual_positive_coef = _require_float(
        counterfactual_positive.get("coef", 0.0),
        field_name="training.counterfactual_positive.coef",
    )
    counterfactual_positive_final_coef = _require_float(
        counterfactual_positive.get("final_coef", counterfactual_positive_coef),
        field_name="training.counterfactual_positive.final_coef",
    )
    counterfactual_positive_margin_coef = _require_float(
        counterfactual_positive.get("margin_coef", 0.0),
        field_name="training.counterfactual_positive.margin_coef",
    )
    counterfactual_positive_margin = _require_float(
        counterfactual_positive.get("margin", 1.0),
        field_name="training.counterfactual_positive.margin",
    )
    if (
        counterfactual_positive_coef < 0.0
        or counterfactual_positive_final_coef < 0.0
        or counterfactual_positive_margin_coef < 0.0
    ):
        raise ValueError("training.counterfactual_positive coefficients must be >= 0.0")
    if counterfactual_positive_margin < 0.0:
        raise ValueError("training.counterfactual_positive.margin must be >= 0.0")
    counterfactual_positive_start_updates = _require_int(
        counterfactual_positive.get("start_updates", 0),
        field_name="training.counterfactual_positive.start_updates",
        minimum=0,
    )
    counterfactual_positive_end_updates = _require_int(
        counterfactual_positive.get("end_updates", -1),
        field_name="training.counterfactual_positive.end_updates",
        minimum=-1,
    )
    if (
        counterfactual_positive_end_updates >= 0
        and counterfactual_positive_end_updates < counterfactual_positive_start_updates
    ):
        raise ValueError(
            "training.counterfactual_positive.end_updates must be >= training.counterfactual_positive.start_updates"
        )
    counterfactual_positive_max_labels = _require_int(
        counterfactual_positive.get("max_labels", 0),
        field_name="training.counterfactual_positive.max_labels",
        minimum=0,
    )
    counterfactual_positive_config = TrainingCounterfactualPositiveConfig(
        enabled=counterfactual_positive_enabled,
        label_dirs=counterfactual_positive_label_dirs,
        coef=counterfactual_positive_coef,
        final_coef=counterfactual_positive_final_coef,
        start_updates=counterfactual_positive_start_updates,
        end_updates=counterfactual_positive_end_updates,
        margin_coef=counterfactual_positive_margin_coef,
        margin=counterfactual_positive_margin,
        max_labels=counterfactual_positive_max_labels,
    )
    _reject_unknown_keys(
        main_residual_policy,
        allowed={
            "enabled",
            "base_snapshot_path",
            "initial_residual_state_path",
            "public_heuristic_bias_scale",
            "hidden_dim",
            "alpha",
            "residual_mode",
            "gate_bias",
        },
        context="training.main_residual_policy",
    )
    main_residual_enabled = _require_bool(
        main_residual_policy.get("enabled", False),
        field_name="training.main_residual_policy.enabled",
    )
    main_residual_base_snapshot_path = str(main_residual_policy.get("base_snapshot_path", "") or "").strip()
    main_residual_initial_state_path = str(main_residual_policy.get("initial_residual_state_path", "") or "").strip()
    main_residual_bias_scale = _require_float(
        main_residual_policy.get("public_heuristic_bias_scale", 1.0),
        field_name="training.main_residual_policy.public_heuristic_bias_scale",
    )
    main_residual_hidden_dim = _require_int(
        main_residual_policy.get("hidden_dim", 256),
        field_name="training.main_residual_policy.hidden_dim",
        minimum=1,
    )
    main_residual_alpha = _require_float(
        main_residual_policy.get("alpha", 0.1),
        field_name="training.main_residual_policy.alpha",
    )
    main_residual_mode = str(main_residual_policy.get("residual_mode", "plain") or "plain").strip()
    main_residual_gate_bias = _require_float(
        main_residual_policy.get("gate_bias", 0.0),
        field_name="training.main_residual_policy.gate_bias",
    )
    if main_residual_enabled and not main_residual_base_snapshot_path:
        raise ValueError("training.main_residual_policy.base_snapshot_path must be non-empty when enabled")
    if main_residual_bias_scale < 0.0:
        raise ValueError("training.main_residual_policy.public_heuristic_bias_scale must be >= 0.0")
    if main_residual_alpha <= 0.0:
        raise ValueError("training.main_residual_policy.alpha must be > 0.0")
    if main_residual_mode not in {"plain", "gated", "family_gated"}:
        raise ValueError("training.main_residual_policy.residual_mode must be plain, gated, or family_gated")
    main_residual_policy_config = TrainingMainResidualPolicyConfig(
        enabled=main_residual_enabled,
        base_snapshot_path=main_residual_base_snapshot_path,
        initial_residual_state_path=main_residual_initial_state_path,
        public_heuristic_bias_scale=main_residual_bias_scale,
        hidden_dim=main_residual_hidden_dim,
        alpha=main_residual_alpha,
        residual_mode=main_residual_mode,
        gate_bias=main_residual_gate_bias,
    )
    diverse_opponent_actor_count = _require_int(
        body.get("diverse_opponent_actor_count", 0),
        field_name="training.diverse_opponent_actor_count",
        minimum=-1,
    )
    diverse_model_actor_count = _require_int(
        body.get("diverse_model_actor_count", 0),
        field_name="training.diverse_model_actor_count",
        minimum=0,
    )
    diverse_opponent_policy_id = str(body.get("diverse_opponent_policy_id", "") or "").strip()
    diverse_opponent_policy_ids = tuple(
        str(policy_id).strip()
        for policy_id in _require_str_list(
            body.get("diverse_opponent_policy_ids", []),
            field_name="training.diverse_opponent_policy_ids",
        )
        if str(policy_id).strip()
    )
    residual_opponent_policy_entries = body.get("residual_opponent_policies", [])
    if residual_opponent_policy_entries is None:
        residual_opponent_policy_entries = []
    if not isinstance(residual_opponent_policy_entries, list):
        raise TypeError("training.residual_opponent_policies must be a list of mappings")
    residual_opponent_policies: list[TrainingResidualOpponentPolicyConfig] = []
    seen_residual_policy_ids: set[str] = set()
    for index, entry in enumerate(residual_opponent_policy_entries):
        mapping = _require_mapping(entry, context=f"training.residual_opponent_policies[{index}]")
        _reject_unknown_keys(
            mapping,
            allowed={
                "policy_id",
                "base_snapshot_path",
                "residual_state_path",
                "public_heuristic_bias_scale",
                "role",
            },
            context=f"training.residual_opponent_policies[{index}]",
        )
        policy_id = str(mapping.get("policy_id", "") or "").strip()
        base_snapshot_path = str(mapping.get("base_snapshot_path", "") or "").strip()
        residual_state_path = str(mapping.get("residual_state_path", "") or "").strip()
        role = str(mapping.get("role", "b1_exploiter_candidate") or "").strip()
        if not policy_id:
            raise ValueError(f"training.residual_opponent_policies[{index}].policy_id must be non-empty")
        if policy_id in seen_residual_policy_ids:
            raise ValueError(f"duplicate training.residual_opponent_policies policy_id: {policy_id}")
        seen_residual_policy_ids.add(policy_id)
        if not base_snapshot_path:
            raise ValueError(f"training.residual_opponent_policies[{index}].base_snapshot_path must be non-empty")
        if not residual_state_path:
            raise ValueError(f"training.residual_opponent_policies[{index}].residual_state_path must be non-empty")
        bias_scale = _require_float(
            mapping.get("public_heuristic_bias_scale", 1.0),
            field_name=f"training.residual_opponent_policies[{index}].public_heuristic_bias_scale",
        )
        if bias_scale < 0.0:
            raise ValueError(f"training.residual_opponent_policies[{index}].public_heuristic_bias_scale must be >= 0.0")
        residual_opponent_policies.append(
            TrainingResidualOpponentPolicyConfig(
                policy_id=policy_id,
                base_snapshot_path=base_snapshot_path,
                residual_state_path=residual_state_path,
                public_heuristic_bias_scale=bias_scale,
                role=role or "b1_exploiter_candidate",
            )
        )
    diverse_opponent_batch_fraction = _require_float(
        body.get("diverse_opponent_batch_fraction", 0.0),
        field_name="training.diverse_opponent_batch_fraction",
    )
    if diverse_opponent_batch_fraction < 0.0 or diverse_opponent_batch_fraction > 1.0:
        raise ValueError(
            "training.diverse_opponent_batch_fraction must be between 0.0 and 1.0 inclusive, "
            f"got {diverse_opponent_batch_fraction}"
        )
    diverse_opponent_batch_wait_ms = _require_int(
        body.get("diverse_opponent_batch_wait_ms", 0),
        field_name="training.diverse_opponent_batch_wait_ms",
        minimum=0,
    )
    collect_batch_prefetch_enabled = _require_bool(
        body.get("collect_batch_prefetch_enabled", False),
        field_name="training.collect_batch_prefetch_enabled",
    )
    heuristic_native_rollout_enabled = _require_bool(
        body.get("heuristic_native_rollout_enabled", False),
        field_name="training.heuristic_native_rollout_enabled",
    )
    heuristic_native_rollout_profile = _require_choice(
        body.get("heuristic_native_rollout_profile", "base"),
        field_name="training.heuristic_native_rollout_profile",
        allowed=_TRAINING_PUBLIC_HEURISTIC_PROFILES,
    )
    heuristic_native_rollout_profiles = tuple(
        name.strip().lower()
        for name in _require_str_list(
            body.get("heuristic_native_rollout_profiles", []),
            field_name="training.heuristic_native_rollout_profiles",
        )
        if name.strip()
    )
    invalid_native_rollout_profiles = sorted(
        set(heuristic_native_rollout_profiles) - _TRAINING_PUBLIC_HEURISTIC_PROFILES
    )
    if invalid_native_rollout_profiles:
        raise ValueError(
            "training.heuristic_native_rollout_profiles contains unsupported profiles: "
            + ", ".join(invalid_native_rollout_profiles)
        )
    heuristic_native_rollout_profile_mode = _require_choice(
        body.get("heuristic_native_rollout_profile_mode", "fixed"),
        field_name="training.heuristic_native_rollout_profile_mode",
        allowed=_TRAINING_NATIVE_ROLLOUT_PROFILE_MODES,
    )
    scaling_target_envs_per_gpu = _require_int(
        scaling.get("target_envs_per_gpu", 512),
        field_name="training.scaling.target_envs_per_gpu",
        minimum=1,
    )
    scaling_min_envs_per_actor = _require_int(
        scaling.get("min_envs_per_actor", 32),
        field_name="training.scaling.min_envs_per_actor",
        minimum=1,
    )
    scaling_max_envs_per_actor = _require_int(
        scaling.get("max_envs_per_actor", 64),
        field_name="training.scaling.max_envs_per_actor",
        minimum=1,
    )
    if scaling_max_envs_per_actor < scaling_min_envs_per_actor:
        raise ValueError("training.scaling.max_envs_per_actor must be >= training.scaling.min_envs_per_actor")
    scaling_ram_queue_fraction = _require_float(
        scaling.get("ram_queue_fraction", 0.25),
        field_name="training.scaling.ram_queue_fraction",
    )
    if scaling_ram_queue_fraction <= 0.0 or scaling_ram_queue_fraction > 1.0:
        raise ValueError("training.scaling.ram_queue_fraction must be in (0.0, 1.0]")
    scaling_vram_fraction = _require_float(
        scaling.get("vram_fraction", 0.85),
        field_name="training.scaling.vram_fraction",
    )
    if scaling_vram_fraction <= 0.0 or scaling_vram_fraction > 1.0:
        raise ValueError("training.scaling.vram_fraction must be in (0.0, 1.0]")
    structured_aux_public_temperature = _require_float(
        structured_aux.get("teacher_public_heuristic_temperature", 32.0),
        field_name="training.structured_aux.teacher_public_heuristic_temperature",
    )
    if structured_aux_public_temperature <= 0.0:
        raise ValueError("training.structured_aux.teacher_public_heuristic_temperature must be > 0")
    structured_warmstart_public_temperature = _require_float(
        structured_warmstart.get("teacher_public_heuristic_temperature", 32.0),
        field_name="training.structured_warmstart.teacher_public_heuristic_temperature",
    )
    if structured_warmstart_public_temperature <= 0.0:
        raise ValueError("training.structured_warmstart.teacher_public_heuristic_temperature must be > 0")
    structured_aux_public_profiles = tuple(
        name.strip().lower()
        for name in _require_str_list(
            structured_aux.get("teacher_public_heuristic_profiles", []),
            field_name="training.structured_aux.teacher_public_heuristic_profiles",
        )
        if name.strip()
    )
    invalid_aux_public_profiles = sorted(set(structured_aux_public_profiles) - _TRAINING_PUBLIC_HEURISTIC_PROFILES)
    if invalid_aux_public_profiles:
        raise ValueError(
            "training.structured_aux.teacher_public_heuristic_profiles contains unsupported profiles: "
            + ", ".join(invalid_aux_public_profiles)
        )
    structured_warmstart_public_profiles = tuple(
        name.strip().lower()
        for name in _require_str_list(
            structured_warmstart.get("teacher_public_heuristic_profiles", []),
            field_name="training.structured_warmstart.teacher_public_heuristic_profiles",
        )
        if name.strip()
    )
    invalid_warmstart_public_profiles = sorted(
        set(structured_warmstart_public_profiles) - _TRAINING_PUBLIC_HEURISTIC_PROFILES
    )
    if invalid_warmstart_public_profiles:
        raise ValueError(
            "training.structured_warmstart.teacher_public_heuristic_profiles contains unsupported profiles: "
            + ", ".join(invalid_warmstart_public_profiles)
        )
    structured_aux_public_profile_mode = _require_choice(
        structured_aux.get("teacher_public_heuristic_profile_mode", "mixture"),
        field_name="training.structured_aux.teacher_public_heuristic_profile_mode",
        allowed=_TRAINING_PUBLIC_HEURISTIC_PROFILE_MODES,
    )
    structured_aux_public_start_updates = _require_int(
        structured_aux.get("teacher_public_heuristic_start_updates", 0),
        field_name="training.structured_aux.teacher_public_heuristic_start_updates",
        minimum=0,
    )
    structured_aux_public_end_updates = _require_int(
        structured_aux.get("teacher_public_heuristic_end_updates", -1),
        field_name="training.structured_aux.teacher_public_heuristic_end_updates",
        minimum=-1,
    )
    if (
        structured_aux_public_end_updates >= 0
        and structured_aux_public_end_updates < structured_aux_public_start_updates
    ):
        raise ValueError(
            "training.structured_aux.teacher_public_heuristic_end_updates must be >= "
            "training.structured_aux.teacher_public_heuristic_start_updates"
        )
    structured_aux_public_final_coef = _require_float(
        structured_aux.get(
            "teacher_public_heuristic_final_coef",
            structured_aux.get("teacher_public_heuristic_coef", 0.0),
        ),
        field_name="training.structured_aux.teacher_public_heuristic_final_coef",
    )
    if structured_aux_public_final_coef < 0.0:
        raise ValueError("training.structured_aux.teacher_public_heuristic_final_coef must be >= 0.0")
    structured_aux_public_profiles_end_updates = _require_int(
        structured_aux.get("teacher_public_heuristic_profiles_end_updates", -1),
        field_name="training.structured_aux.teacher_public_heuristic_profiles_end_updates",
        minimum=-1,
    )
    structured_aux_public_label_profile = _require_choice(
        structured_aux.get("teacher_public_heuristic_label_profile", "base"),
        field_name="training.structured_aux.teacher_public_heuristic_label_profile",
        allowed=_TRAINING_PUBLIC_HEURISTIC_PROFILES,
    )
    structured_warmstart_public_profile_mode = _require_choice(
        structured_warmstart.get("teacher_public_heuristic_profile_mode", "mixture"),
        field_name="training.structured_warmstart.teacher_public_heuristic_profile_mode",
        allowed=_TRAINING_PUBLIC_HEURISTIC_PROFILE_MODES,
    )
    structured_warmstart_public_profiles_end_updates = _require_int(
        structured_warmstart.get("teacher_public_heuristic_profiles_end_updates", -1),
        field_name="training.structured_warmstart.teacher_public_heuristic_profiles_end_updates",
        minimum=-1,
    )

    return TrainingConfig(
        algorithm=_require_choice(body["algorithm"], field_name="training.algorithm", allowed=_TRAINING_ALGORITHMS),
        rollout=TrainingRolloutConfig(
            unroll_length=_require_int(
                rollout["unroll_length"], field_name="training.rollout.unroll_length", minimum=1
            ),
            batch_unrolls_per_update=_require_int(
                rollout["batch_unrolls_per_update"],
                field_name="training.rollout.batch_unrolls_per_update",
                minimum=1,
            ),
        ),
        optimizer=TrainingOptimizerConfig(
            name=_require_text(optimizer["name"], field_name="training.optimizer.name"),
            learning_rate=_require_float(optimizer["learning_rate"], field_name="training.optimizer.learning_rate"),
            grad_norm_clip=_require_float(optimizer["grad_norm_clip"], field_name="training.optimizer.grad_norm_clip"),
            value_loss_coef=_require_float(
                optimizer["value_loss_coef"], field_name="training.optimizer.value_loss_coef"
            ),
            backend=_require_choice(
                optimizer.get("backend", "auto"),
                field_name="training.optimizer.backend",
                allowed=_TRAINING_OPTIMIZER_BACKENDS,
            ),
        ),
        exploration=TrainingExplorationConfig(
            entropy_coef=_require_float(exploration["entropy_coef"], field_name="training.exploration.entropy_coef"),
            entropy_anneal_to=_require_float(
                exploration["entropy_anneal_to"], field_name="training.exploration.entropy_anneal_to"
            ),
            entropy_anneal_steps_updates=_require_int(
                exploration["entropy_anneal_steps_updates"],
                field_name="training.exploration.entropy_anneal_steps_updates",
                minimum=1,
            ),
        ),
        precision=TrainingPrecisionConfig(
            mixed_precision=_require_bool(
                precision["mixed_precision"], field_name="training.precision.mixed_precision"
            ),
            compile_learner=_require_bool(
                precision["compile_learner"], field_name="training.precision.compile_learner"
            ),
            compile_actor_inference=_require_bool(
                precision.get("compile_actor_inference", False),
                field_name="training.precision.compile_actor_inference",
            ),
            masking_math_float32=_require_bool(
                precision["masking_math_float32"],
                field_name="training.precision.masking_math_float32",
            ),
        ),
        profile_timers=profile_timers,
        torch_profiler=torch_profiler,
        freeze_parameter_prefixes=freeze_parameter_prefixes,
        checkpointing=TrainingCheckpointingConfig(
            checkpoint_interval_updates=_require_int(
                checkpointing["checkpoint_interval_updates"],
                field_name="training.checkpointing.checkpoint_interval_updates",
                minimum=1,
            ),
            snapshot_interval_updates=_require_int(
                checkpointing["snapshot_interval_updates"],
                field_name="training.checkpointing.snapshot_interval_updates",
                minimum=1,
            ),
            actor_reload_interval_updates=_require_int(
                checkpointing["actor_reload_interval_updates"],
                field_name="training.checkpointing.actor_reload_interval_updates",
                minimum=1,
            ),
        ),
        vtrace=TrainingVTraceConfig(
            rho_bar=_require_float(vtrace["rho_bar"], field_name="training.vtrace.rho_bar"),
            c_bar=_require_float(vtrace["c_bar"], field_name="training.vtrace.c_bar"),
        ),
        ppo=TrainingPpoConfig(
            clip_epsilon=_require_float(ppo.get("clip_epsilon", 0.2), field_name="training.ppo.clip_epsilon"),
            value_clip_epsilon=_require_float(
                ppo.get("value_clip_epsilon", 0.2),
                field_name="training.ppo.value_clip_epsilon",
            ),
            gae_lambda=_require_float(ppo.get("gae_lambda", 0.95), field_name="training.ppo.gae_lambda"),
            epochs=_require_int(ppo.get("epochs", 4), field_name="training.ppo.epochs", minimum=1),
            target_kl=_require_float(ppo.get("target_kl", 0.0), field_name="training.ppo.target_kl"),
            normalize_advantages=_require_bool(
                ppo.get("normalize_advantages", True),
                field_name="training.ppo.normalize_advantages",
            ),
        ),
        structured_aux=TrainingStructuredAuxConfig(
            enabled=_require_bool(
                structured_aux.get("enabled", False),
                field_name="training.structured_aux.enabled",
            ),
            teacher_family_coef=_require_float(
                structured_aux.get("teacher_family_coef", 0.0),
                field_name="training.structured_aux.teacher_family_coef",
            ),
            teacher_slot_coef=_require_float(
                structured_aux.get("teacher_slot_coef", 0.0),
                field_name="training.structured_aux.teacher_slot_coef",
            ),
            teacher_move_source_coef=_require_float(
                structured_aux.get("teacher_move_source_coef", 0.0),
                field_name="training.structured_aux.teacher_move_source_coef",
            ),
            teacher_attack_type_coef=_require_float(
                structured_aux.get("teacher_attack_type_coef", 0.0),
                field_name="training.structured_aux.teacher_attack_type_coef",
            ),
            teacher_action_coef=_require_float(
                structured_aux.get("teacher_action_coef", 0.0),
                field_name="training.structured_aux.teacher_action_coef",
            ),
            teacher_same_family_action_coef=_require_float(
                structured_aux.get("teacher_same_family_action_coef", 0.0),
                field_name="training.structured_aux.teacher_same_family_action_coef",
            ),
            teacher_public_heuristic_coef=_require_float(
                structured_aux.get("teacher_public_heuristic_coef", 0.0),
                field_name="training.structured_aux.teacher_public_heuristic_coef",
            ),
            teacher_public_main_move_coef=_require_float(
                structured_aux.get("teacher_public_main_move_coef", 0.0),
                field_name="training.structured_aux.teacher_public_main_move_coef",
            ),
            teacher_development_pass_suppression_coef=_require_float(
                structured_aux.get("teacher_development_pass_suppression_coef", 0.0),
                field_name="training.structured_aux.teacher_development_pass_suppression_coef",
            ),
            teacher_public_heuristic_start_updates=structured_aux_public_start_updates,
            teacher_public_heuristic_end_updates=structured_aux_public_end_updates,
            teacher_public_heuristic_final_coef=_require_float(
                structured_aux_public_final_coef,
                field_name="training.structured_aux.teacher_public_heuristic_final_coef",
            ),
            teacher_public_heuristic_temperature=structured_aux_public_temperature,
            teacher_public_heuristic_families=_require_str_list(
                structured_aux.get("teacher_public_heuristic_families", []),
                field_name="training.structured_aux.teacher_public_heuristic_families",
            ),
            teacher_public_heuristic_profiles=structured_aux_public_profiles,
            teacher_public_heuristic_profile_mode=structured_aux_public_profile_mode,
            teacher_public_heuristic_profiles_end_updates=structured_aux_public_profiles_end_updates,
            teacher_public_heuristic_label_profile=structured_aux_public_label_profile,
        ),
        structured_warmstart=TrainingStructuredWarmstartConfig(
            enabled=_require_bool(
                structured_warmstart.get("enabled", False),
                field_name="training.structured_warmstart.enabled",
            ),
            updates=_require_int(
                structured_warmstart.get("updates", 0),
                field_name="training.structured_warmstart.updates",
                minimum=0,
            ),
            teacher_family_coef=_require_float(
                structured_warmstart.get("teacher_family_coef", 0.0),
                field_name="training.structured_warmstart.teacher_family_coef",
            ),
            teacher_slot_coef=_require_float(
                structured_warmstart.get("teacher_slot_coef", 0.0),
                field_name="training.structured_warmstart.teacher_slot_coef",
            ),
            teacher_move_source_coef=_require_float(
                structured_warmstart.get("teacher_move_source_coef", 0.0),
                field_name="training.structured_warmstart.teacher_move_source_coef",
            ),
            teacher_attack_type_coef=_require_float(
                structured_warmstart.get("teacher_attack_type_coef", 0.0),
                field_name="training.structured_warmstart.teacher_attack_type_coef",
            ),
            teacher_action_coef=_require_float(
                structured_warmstart.get("teacher_action_coef", 0.0),
                field_name="training.structured_warmstart.teacher_action_coef",
            ),
            teacher_same_family_action_coef=_require_float(
                structured_warmstart.get("teacher_same_family_action_coef", 0.0),
                field_name="training.structured_warmstart.teacher_same_family_action_coef",
            ),
            teacher_public_heuristic_coef=_require_float(
                structured_warmstart.get("teacher_public_heuristic_coef", 0.0),
                field_name="training.structured_warmstart.teacher_public_heuristic_coef",
            ),
            teacher_public_heuristic_temperature=structured_warmstart_public_temperature,
            teacher_public_heuristic_families=_require_str_list(
                structured_warmstart.get("teacher_public_heuristic_families", []),
                field_name="training.structured_warmstart.teacher_public_heuristic_families",
            ),
            teacher_public_heuristic_profiles=structured_warmstart_public_profiles,
            teacher_public_heuristic_profile_mode=structured_warmstart_public_profile_mode,
            teacher_public_heuristic_profiles_end_updates=structured_warmstart_public_profiles_end_updates,
        ),
        structured_metrics=TrainingStructuredMetricsConfig(
            mode=_require_choice(
                structured_metrics.get("mode", "off"),
                field_name="training.structured_metrics.mode",
                allowed=_TRAINING_STRUCTURED_METRICS_MODES,
            ),
        ),
        teacher_aux=TrainingTeacherAuxConfig(
            mode=_require_choice(
                teacher_aux.get("mode", "always"),
                field_name="training.teacher_aux.mode",
                allowed=_TRAINING_TEACHER_AUX_MODES,
            ),
        ),
        raw_b1_distill=raw_b1_distill_config,
        counterfactual_positive=counterfactual_positive_config,
        scaling=TrainingScalingConfig(
            learner_parallelism=_require_choice(
                scaling.get("learner_parallelism", "auto"),
                field_name="training.scaling.learner_parallelism",
                allowed=_TRAINING_LEARNER_PARALLELISM,
            ),
            learner_gpu_count=_require_auto_or_int(
                scaling.get("learner_gpu_count", "auto"),
                field_name="training.scaling.learner_gpu_count",
                minimum=0,
            ),
            actor_topology=_require_choice(
                scaling.get("actor_topology", "auto"),
                field_name="training.scaling.actor_topology",
                allowed=_TRAINING_ACTOR_TOPOLOGIES,
            ),
            target_envs_per_gpu=scaling_target_envs_per_gpu,
            min_envs_per_actor=scaling_min_envs_per_actor,
            max_envs_per_actor=scaling_max_envs_per_actor,
            max_actor_process_count=_require_int(
                scaling.get("max_actor_process_count", 64),
                field_name="training.scaling.max_actor_process_count",
                minimum=1,
            ),
            reserve_cpu_cores=_require_int(
                scaling.get("reserve_cpu_cores", 4),
                field_name="training.scaling.reserve_cpu_cores",
                minimum=0,
            ),
            learner_cpu_cores_per_gpu=_require_int(
                scaling.get("learner_cpu_cores_per_gpu", 2),
                field_name="training.scaling.learner_cpu_cores_per_gpu",
                minimum=1,
            ),
            queue_depth_multiplier=_require_int(
                scaling.get("queue_depth_multiplier", 2),
                field_name="training.scaling.queue_depth_multiplier",
                minimum=1,
            ),
            ram_queue_fraction=scaling_ram_queue_fraction,
            vram_fraction=scaling_vram_fraction,
        ),
        fixed_opponent_backend=fixed_opponent_backend,
        actor_policy_backend=actor_policy_backend,
        actor_heuristic_fraction=actor_heuristic_fraction,
        actor_heuristic_start_updates=actor_heuristic_start_updates,
        actor_heuristic_end_updates=actor_heuristic_end_updates,
        actor_heuristic_final_fraction=actor_heuristic_final_fraction,
        train_on_heuristic_actor_rows=train_on_heuristic_actor_rows,
        policy_loss_coef=policy_loss_coef,
        behavior_action_bc_coef=behavior_action_bc_coef,
        reference_policy_top_action_bc_coef=reference_policy_top_action_bc_coef,
        reference_policy_top_action_bc_final_coef=reference_policy_top_action_bc_final_coef,
        reference_policy_top_action_bc_start_updates=reference_policy_top_action_bc_start_updates,
        reference_policy_top_action_bc_end_updates=reference_policy_top_action_bc_end_updates,
        b1_opponent_reference_policy_top_action_bc_coef=b1_opponent_reference_policy_top_action_bc_coef,
        b1_second_seat_positive_advantage_policy_coef=b1_second_seat_positive_advantage_policy_coef,
        b1_second_seat_reference_top_action_avoidance_coef=(b1_second_seat_reference_top_action_avoidance_coef),
        reference_policy_top_action_family_bc_coef=reference_policy_top_action_family_bc_coef,
        reference_policy_top_action_family_bc_final_coef=reference_policy_top_action_family_bc_final_coef,
        reference_policy_top_action_family_bc_start_updates=reference_policy_top_action_family_bc_start_updates,
        reference_policy_top_action_family_bc_end_updates=reference_policy_top_action_family_bc_end_updates,
        reference_policy_id=reference_policy_id,
        main_residual_policy=main_residual_policy_config,
        diverse_opponent_actor_count=diverse_opponent_actor_count,
        diverse_model_actor_count=diverse_model_actor_count,
        diverse_opponent_policy_id=diverse_opponent_policy_id,
        diverse_opponent_policy_ids=diverse_opponent_policy_ids,
        residual_opponent_policies=tuple(residual_opponent_policies),
        diverse_opponent_batch_fraction=diverse_opponent_batch_fraction,
        diverse_opponent_batch_wait_ms=diverse_opponent_batch_wait_ms,
        collect_batch_prefetch_enabled=collect_batch_prefetch_enabled,
        heuristic_native_rollout_enabled=heuristic_native_rollout_enabled,
        heuristic_native_rollout_profile=heuristic_native_rollout_profile,
        heuristic_native_rollout_profiles=heuristic_native_rollout_profiles,
        heuristic_native_rollout_profile_mode=heuristic_native_rollout_profile_mode,
        heuristic_actor_hidden_state_tracking=heuristic_actor_hidden_state_tracking,
    )
