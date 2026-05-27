"""Model stack config section parser."""

from __future__ import annotations

from typing import Any

from .models import ModelConfig, ModelDropoutConfig
from .parsing_utils import (
    reject_unknown_keys,
    require_bool,
    require_choice,
    require_float,
    require_int,
    require_int_list,
    require_mapping,
    require_str_list,
)

MODEL_ENCODER_KINDS = frozenset({"mlp", "typed_v1", "structured_v2"})
STRUCTURED_POLICY_CONTRACTS = frozenset({"packed_v1", "factorized_v1"})
MODEL_RECURRENT_CORES = frozenset({"gru", "none"})


def parse_model_config(body: dict[str, Any]) -> ModelConfig:
    reject_unknown_keys(
        body,
        allowed={
            "gru_hidden_size",
            "encoder_mlp_width",
            "encoder_mlp_layers",
            "encoder_kind",
            "structured_policy_contract",
            "typed_feature_width",
            "recurrent_core",
            "candidate_scoring_chunk_size",
            "cuda_learner_candidate_scoring_chunk_size",
            "public_heuristic_logit_bias_scale",
            "public_heuristic_actor_logit_bias_scale",
            "public_heuristic_logit_bias_start_updates",
            "public_heuristic_logit_bias_end_updates",
            "public_heuristic_logit_bias_final_scale",
            "public_heuristic_logit_bias_families",
            "opponent_context_policy_ids",
            "opponent_context_hidden_scale",
            "opponent_context_trainable_hidden_scale",
            "opponent_context_trainable_recurrent_scale",
            "opponent_context_trainable_action_bias_scale",
            "opponent_context_trainable_candidate_residual_scale",
            "opponent_context_candidate_residual_width",
            "opponent_context_candidate_residual_mode",
            "opponent_context_candidate_residual_action_ids",
            "opponent_context_adapter_lr_multiplier",
            "opponent_context_adapter_train_only",
            "opponent_context_eval_policy_ids",
            "layer_norm",
            "dropout",
        },
        context="model",
    )
    dropout = require_mapping(body["dropout"], context="model.dropout")
    reject_unknown_keys(dropout, allowed={"family_a", "ablation"}, context="model.dropout")
    public_heuristic_logit_bias_start_updates = require_int(
        body.get("public_heuristic_logit_bias_start_updates", 0),
        field_name="model.public_heuristic_logit_bias_start_updates",
        minimum=0,
    )
    public_heuristic_logit_bias_end_updates = require_int(
        body.get("public_heuristic_logit_bias_end_updates", -1),
        field_name="model.public_heuristic_logit_bias_end_updates",
        minimum=-1,
    )
    if (
        public_heuristic_logit_bias_end_updates >= 0
        and public_heuristic_logit_bias_end_updates < public_heuristic_logit_bias_start_updates
    ):
        raise ValueError(
            "model.public_heuristic_logit_bias_end_updates must be >= model.public_heuristic_logit_bias_start_updates"
        )
    public_heuristic_logit_bias_final_scale = require_float(
        body.get(
            "public_heuristic_logit_bias_final_scale",
            body.get("public_heuristic_logit_bias_scale", 0.0),
        ),
        field_name="model.public_heuristic_logit_bias_final_scale",
    )
    if public_heuristic_logit_bias_final_scale < 0.0:
        raise ValueError("model.public_heuristic_logit_bias_final_scale must be >= 0.0")
    opponent_context_hidden_scale = require_float(
        body.get("opponent_context_hidden_scale", 0.0),
        field_name="model.opponent_context_hidden_scale",
    )
    if opponent_context_hidden_scale < 0.0:
        raise ValueError("model.opponent_context_hidden_scale must be >= 0.0")
    opponent_context_trainable_hidden_scale = require_float(
        body.get("opponent_context_trainable_hidden_scale", 0.0),
        field_name="model.opponent_context_trainable_hidden_scale",
    )
    if opponent_context_trainable_hidden_scale < 0.0:
        raise ValueError("model.opponent_context_trainable_hidden_scale must be >= 0.0")
    opponent_context_trainable_recurrent_scale = require_float(
        body.get("opponent_context_trainable_recurrent_scale", 0.0),
        field_name="model.opponent_context_trainable_recurrent_scale",
    )
    if opponent_context_trainable_recurrent_scale < 0.0:
        raise ValueError("model.opponent_context_trainable_recurrent_scale must be >= 0.0")
    opponent_context_trainable_action_bias_scale = require_float(
        body.get("opponent_context_trainable_action_bias_scale", 0.0),
        field_name="model.opponent_context_trainable_action_bias_scale",
    )
    if opponent_context_trainable_action_bias_scale < 0.0:
        raise ValueError("model.opponent_context_trainable_action_bias_scale must be >= 0.0")
    opponent_context_trainable_candidate_residual_scale = require_float(
        body.get("opponent_context_trainable_candidate_residual_scale", 0.0),
        field_name="model.opponent_context_trainable_candidate_residual_scale",
    )
    if opponent_context_trainable_candidate_residual_scale < 0.0:
        raise ValueError("model.opponent_context_trainable_candidate_residual_scale must be >= 0.0")
    opponent_context_candidate_residual_width = require_int(
        body.get("opponent_context_candidate_residual_width", 32),
        field_name="model.opponent_context_candidate_residual_width",
        minimum=1,
    )
    opponent_context_candidate_residual_mode = require_choice(
        body.get("opponent_context_candidate_residual_mode", "additive"),
        field_name="model.opponent_context_candidate_residual_mode",
        allowed=frozenset({"additive", "bilinear", "rich", "rich_bilinear"}),
    )
    opponent_context_adapter_lr_multiplier = require_float(
        body.get("opponent_context_adapter_lr_multiplier", 1.0),
        field_name="model.opponent_context_adapter_lr_multiplier",
    )
    if opponent_context_adapter_lr_multiplier <= 0.0:
        raise ValueError("model.opponent_context_adapter_lr_multiplier must be > 0.0")
    return ModelConfig(
        gru_hidden_size=require_int(body["gru_hidden_size"], field_name="model.gru_hidden_size", minimum=1),
        encoder_mlp_width=require_int(body["encoder_mlp_width"], field_name="model.encoder_mlp_width", minimum=1),
        encoder_mlp_layers=require_int(body["encoder_mlp_layers"], field_name="model.encoder_mlp_layers", minimum=1),
        encoder_kind=require_choice(
            body.get("encoder_kind", "mlp"),
            field_name="model.encoder_kind",
            allowed=MODEL_ENCODER_KINDS,
        ),
        structured_policy_contract=require_choice(
            body.get("structured_policy_contract", "packed_v1"),
            field_name="model.structured_policy_contract",
            allowed=STRUCTURED_POLICY_CONTRACTS,
        ),
        typed_feature_width=require_int(
            body.get("typed_feature_width", 64),
            field_name="model.typed_feature_width",
            minimum=1,
        ),
        recurrent_core=require_choice(
            body.get("recurrent_core", "gru"),
            field_name="model.recurrent_core",
            allowed=MODEL_RECURRENT_CORES,
        ),
        candidate_scoring_chunk_size=require_int(
            body.get("candidate_scoring_chunk_size", 65536),
            field_name="model.candidate_scoring_chunk_size",
            minimum=1,
        ),
        cuda_learner_candidate_scoring_chunk_size=require_int(
            body.get("cuda_learner_candidate_scoring_chunk_size", 262144),
            field_name="model.cuda_learner_candidate_scoring_chunk_size",
            minimum=1,
        ),
        public_heuristic_logit_bias_scale=require_float(
            body.get("public_heuristic_logit_bias_scale", 0.0),
            field_name="model.public_heuristic_logit_bias_scale",
        ),
        public_heuristic_actor_logit_bias_scale=require_float(
            body.get("public_heuristic_actor_logit_bias_scale", -1.0),
            field_name="model.public_heuristic_actor_logit_bias_scale",
        ),
        public_heuristic_logit_bias_start_updates=public_heuristic_logit_bias_start_updates,
        public_heuristic_logit_bias_end_updates=public_heuristic_logit_bias_end_updates,
        public_heuristic_logit_bias_final_scale=require_float(
            public_heuristic_logit_bias_final_scale,
            field_name="model.public_heuristic_logit_bias_final_scale",
        ),
        public_heuristic_logit_bias_families=require_str_list(
            body.get("public_heuristic_logit_bias_families", []),
            field_name="model.public_heuristic_logit_bias_families",
        ),
        opponent_context_policy_ids=require_str_list(
            body.get("opponent_context_policy_ids", []),
            field_name="model.opponent_context_policy_ids",
        ),
        opponent_context_hidden_scale=opponent_context_hidden_scale,
        opponent_context_trainable_hidden_scale=opponent_context_trainable_hidden_scale,
        opponent_context_trainable_recurrent_scale=opponent_context_trainable_recurrent_scale,
        opponent_context_trainable_action_bias_scale=opponent_context_trainable_action_bias_scale,
        opponent_context_trainable_candidate_residual_scale=opponent_context_trainable_candidate_residual_scale,
        opponent_context_candidate_residual_width=opponent_context_candidate_residual_width,
        opponent_context_candidate_residual_mode=opponent_context_candidate_residual_mode,
        opponent_context_candidate_residual_action_ids=require_int_list(
            body.get("opponent_context_candidate_residual_action_ids", []),
            field_name="model.opponent_context_candidate_residual_action_ids",
        ),
        opponent_context_adapter_lr_multiplier=opponent_context_adapter_lr_multiplier,
        opponent_context_adapter_train_only=require_bool(
            body.get("opponent_context_adapter_train_only", False),
            field_name="model.opponent_context_adapter_train_only",
        ),
        opponent_context_eval_policy_ids=require_str_list(
            body.get("opponent_context_eval_policy_ids", []),
            field_name="model.opponent_context_eval_policy_ids",
        ),
        layer_norm=require_bool(body["layer_norm"], field_name="model.layer_norm"),
        dropout=ModelDropoutConfig(
            family_a=require_float(dropout["family_a"], field_name="model.dropout.family_a"),
            ablation=require_float(dropout["ablation"], field_name="model.dropout.ablation"),
        ),
    )
