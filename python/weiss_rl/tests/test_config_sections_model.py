from __future__ import annotations

import pytest

from weiss_rl.config.sections_model import parse_model_config


def _model_body() -> dict[str, object]:
    return {
        "gru_hidden_size": 128,
        "encoder_mlp_width": 256,
        "encoder_mlp_layers": 2,
        "layer_norm": True,
        "dropout": {"family_a": 0.1, "ablation": 0.0},
    }


def test_parse_model_config_applies_existing_defaults() -> None:
    config = parse_model_config(_model_body())

    assert config.encoder_kind == "mlp"
    assert config.structured_policy_contract == "packed_v1"
    assert config.typed_feature_width == 64
    assert config.recurrent_core == "gru"
    assert config.candidate_scoring_chunk_size == 65536
    assert config.cuda_learner_candidate_scoring_chunk_size == 262144
    assert config.public_heuristic_logit_bias_scale == pytest.approx(0.0)
    assert config.public_heuristic_actor_logit_bias_scale == pytest.approx(-1.0)
    assert config.public_heuristic_logit_bias_start_updates == 0
    assert config.public_heuristic_logit_bias_end_updates == -1
    assert config.public_heuristic_logit_bias_final_scale == pytest.approx(0.0)
    assert config.public_heuristic_logit_bias_families == ()
    assert config.opponent_context_policy_ids == ()
    assert config.opponent_context_hidden_scale == pytest.approx(0.0)
    assert config.opponent_context_trainable_hidden_scale == pytest.approx(0.0)
    assert config.opponent_context_trainable_recurrent_scale == pytest.approx(0.0)
    assert config.opponent_context_trainable_action_bias_scale == pytest.approx(0.0)
    assert config.opponent_context_trainable_candidate_residual_scale == pytest.approx(0.0)
    assert config.opponent_context_candidate_residual_width == 32
    assert config.opponent_context_candidate_residual_mode == "additive"
    assert config.opponent_context_candidate_residual_action_ids == ()
    assert config.opponent_context_adapter_lr_multiplier == pytest.approx(1.0)
    assert config.opponent_context_adapter_train_only is False
    assert config.opponent_context_eval_policy_ids == ()
    assert config.dropout.family_a == pytest.approx(0.1)


def test_parse_model_config_accepts_structured_overrides() -> None:
    body = _model_body()
    body.update(
        {
            "encoder_kind": "structured_v2",
            "structured_policy_contract": "factorized_v1",
            "typed_feature_width": 96,
            "recurrent_core": "none",
            "candidate_scoring_chunk_size": 1024,
            "cuda_learner_candidate_scoring_chunk_size": 2048,
            "public_heuristic_logit_bias_scale": 0.25,
            "public_heuristic_actor_logit_bias_scale": 0.5,
            "public_heuristic_logit_bias_start_updates": 10,
            "public_heuristic_logit_bias_end_updates": 20,
            "public_heuristic_logit_bias_final_scale": 0.75,
            "public_heuristic_logit_bias_families": ["main", "backup"],
            "opponent_context_policy_ids": ["B2 HeuristicPublic", "seed_policy"],
            "opponent_context_hidden_scale": 0.5,
            "opponent_context_trainable_hidden_scale": 0.25,
            "opponent_context_trainable_recurrent_scale": 0.60,
            "opponent_context_trainable_action_bias_scale": 0.75,
            "opponent_context_trainable_candidate_residual_scale": 0.125,
            "opponent_context_candidate_residual_width": 12,
            "opponent_context_candidate_residual_mode": "bilinear",
            "opponent_context_candidate_residual_action_ids": [104, 124],
            "opponent_context_adapter_lr_multiplier": 16.0,
            "opponent_context_adapter_train_only": True,
            "opponent_context_eval_policy_ids": ["policy_000001"],
        }
    )

    config = parse_model_config(body)

    assert config.encoder_kind == "structured_v2"
    assert config.structured_policy_contract == "factorized_v1"
    assert config.recurrent_core == "none"
    assert config.public_heuristic_logit_bias_final_scale == pytest.approx(0.75)
    assert config.public_heuristic_logit_bias_families == ("main", "backup")
    assert config.opponent_context_policy_ids == ("B2 HeuristicPublic", "seed_policy")
    assert config.opponent_context_hidden_scale == pytest.approx(0.5)
    assert config.opponent_context_trainable_hidden_scale == pytest.approx(0.25)
    assert config.opponent_context_trainable_recurrent_scale == pytest.approx(0.60)
    assert config.opponent_context_trainable_action_bias_scale == pytest.approx(0.75)
    assert config.opponent_context_trainable_candidate_residual_scale == pytest.approx(0.125)
    assert config.opponent_context_candidate_residual_width == 12
    assert config.opponent_context_candidate_residual_mode == "bilinear"
    assert config.opponent_context_candidate_residual_action_ids == (104, 124)
    assert config.opponent_context_adapter_lr_multiplier == pytest.approx(16.0)
    assert config.opponent_context_adapter_train_only is True
    assert config.opponent_context_eval_policy_ids == ("policy_000001",)


def test_parse_model_config_accepts_rich_candidate_residual_mode() -> None:
    body = _model_body()
    body["opponent_context_candidate_residual_mode"] = "rich"

    config = parse_model_config(body)

    assert config.opponent_context_candidate_residual_mode == "rich"


def test_parse_model_config_accepts_rich_bilinear_candidate_residual_mode() -> None:
    body = _model_body()
    body["opponent_context_candidate_residual_mode"] = "rich_bilinear"

    config = parse_model_config(body)

    assert config.opponent_context_candidate_residual_mode == "rich_bilinear"


def test_parse_model_config_reuses_bias_scale_as_final_scale_default() -> None:
    body = _model_body()
    body["public_heuristic_logit_bias_scale"] = 0.4

    config = parse_model_config(body)

    assert config.public_heuristic_logit_bias_final_scale == pytest.approx(0.4)


def test_parse_model_config_preserves_choice_and_schedule_errors() -> None:
    bad_encoder = _model_body()
    bad_encoder["encoder_kind"] = "transformer"
    with pytest.raises(ValueError, match="model.encoder_kind must be one of: mlp, structured_v2, typed_v1"):
        parse_model_config(bad_encoder)

    bad_contract = _model_body()
    bad_contract["structured_policy_contract"] = "packed_v2"
    with pytest.raises(
        ValueError,
        match="model.structured_policy_contract must be one of: factorized_v1, packed_v1",
    ):
        parse_model_config(bad_contract)

    bad_schedule = _model_body()
    bad_schedule["public_heuristic_logit_bias_start_updates"] = 10
    bad_schedule["public_heuristic_logit_bias_end_updates"] = 5
    with pytest.raises(
        ValueError,
        match="model.public_heuristic_logit_bias_end_updates must be >= "
        "model.public_heuristic_logit_bias_start_updates",
    ):
        parse_model_config(bad_schedule)

    bad_final_scale = _model_body()
    bad_final_scale["public_heuristic_logit_bias_final_scale"] = -0.1
    with pytest.raises(ValueError, match="model.public_heuristic_logit_bias_final_scale must be >= 0.0"):
        parse_model_config(bad_final_scale)

    bad_context_scale = _model_body()
    bad_context_scale["opponent_context_hidden_scale"] = -0.1
    with pytest.raises(ValueError, match="model.opponent_context_hidden_scale must be >= 0.0"):
        parse_model_config(bad_context_scale)

    bad_trainable_context_scale = _model_body()
    bad_trainable_context_scale["opponent_context_trainable_hidden_scale"] = -0.1
    with pytest.raises(ValueError, match="model.opponent_context_trainable_hidden_scale must be >= 0.0"):
        parse_model_config(bad_trainable_context_scale)

    bad_trainable_recurrent_scale = _model_body()
    bad_trainable_recurrent_scale["opponent_context_trainable_recurrent_scale"] = -0.1
    with pytest.raises(ValueError, match="model.opponent_context_trainable_recurrent_scale must be >= 0.0"):
        parse_model_config(bad_trainable_recurrent_scale)

    bad_trainable_action_bias_scale = _model_body()
    bad_trainable_action_bias_scale["opponent_context_trainable_action_bias_scale"] = -0.1
    with pytest.raises(ValueError, match="model.opponent_context_trainable_action_bias_scale must be >= 0.0"):
        parse_model_config(bad_trainable_action_bias_scale)

    bad_trainable_candidate_residual_scale = _model_body()
    bad_trainable_candidate_residual_scale["opponent_context_trainable_candidate_residual_scale"] = -0.1
    with pytest.raises(
        ValueError,
        match="model.opponent_context_trainable_candidate_residual_scale must be >= 0.0",
    ):
        parse_model_config(bad_trainable_candidate_residual_scale)

    bad_candidate_residual_width = _model_body()
    bad_candidate_residual_width["opponent_context_candidate_residual_width"] = 0
    with pytest.raises(ValueError, match="model.opponent_context_candidate_residual_width must be >= 1, got 0"):
        parse_model_config(bad_candidate_residual_width)

    bad_candidate_residual_mode = _model_body()
    bad_candidate_residual_mode["opponent_context_candidate_residual_mode"] = "attention"
    with pytest.raises(
        ValueError,
        match="model.opponent_context_candidate_residual_mode must be one of: additive, bilinear, rich, rich_bilinear",
    ):
        parse_model_config(bad_candidate_residual_mode)

    bad_context_adapter_lr = _model_body()
    bad_context_adapter_lr["opponent_context_adapter_lr_multiplier"] = 0.0
    with pytest.raises(ValueError, match="model.opponent_context_adapter_lr_multiplier must be > 0.0"):
        parse_model_config(bad_context_adapter_lr)


def test_parse_model_config_preserves_unknown_key_and_nested_validation() -> None:
    unknown = _model_body()
    unknown["unknown"] = True
    with pytest.raises(ValueError, match="model has unsupported keys: unknown"):
        parse_model_config(unknown)

    bad_dropout = _model_body()
    bad_dropout["dropout"] = {"family_a": 0.1, "ablation": 0.0, "extra": 1}
    with pytest.raises(ValueError, match="model.dropout has unsupported keys: extra"):
        parse_model_config(bad_dropout)

    bad_width = _model_body()
    bad_width["typed_feature_width"] = 0
    with pytest.raises(ValueError, match="model.typed_feature_width must be >= 1, got 0"):
        parse_model_config(bad_width)
