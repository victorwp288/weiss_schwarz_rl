from __future__ import annotations

import pytest
import torch

from weiss_rl.config.models import ModelConfig, ModelDropoutConfig
from weiss_rl.model import GLOBAL_ACTION_SPACE_SIZE, PolicyValueModel
from weiss_rl.spec import (
    HARD_FAIL_SPEC_MISMATCH_POLICY,
    assert_spec_bundle_contract,
    assert_spec_compatibility,
    canonical_json_bytes,
    normalize_bool_flag,
    normalize_spec_mismatch_policy,
    require_fail_on_spec_mismatch,
    sha256_hex,
    spec_bundle_hash,
)

_VALID_BUNDLE = {
    "encoding_versions": {"obs": 1},
    "action_space_size": 9,
    "pass_id": 8,
    "observation_dtype": "float32",
    "observation_length": 512,
    "spec_hash": 123,
}


def test_spec_compatibility_accepts_match() -> None:
    assert_spec_compatibility(expected_spec_hash=123, observed_bundle=_VALID_BUNDLE)


def test_spec_compatibility_rejects_mismatch() -> None:
    with pytest.raises(RuntimeError, match="expected 124"):
        assert_spec_compatibility(expected_spec_hash=124, observed_bundle=_VALID_BUNDLE)


def test_spec_bundle_contract_accepts_bundle_sha256() -> None:
    expected_hash = sha256_hex(canonical_json_bytes(_VALID_BUNDLE))

    assert_spec_bundle_contract(expected_hash, _VALID_BUNDLE)
    assert spec_bundle_hash(_VALID_BUNDLE) == expected_hash


def test_spec_bundle_contract_rejects_bundle_sha256_mismatch() -> None:
    with pytest.raises(RuntimeError, match="Spec bundle hash mismatch"):
        assert_spec_bundle_contract("0" * 64, _VALID_BUNDLE)


def test_normalize_spec_mismatch_policy_rejects_non_fail_fast_modes() -> None:
    with pytest.raises(ValueError, match="must be 'hard_fail'"):
        normalize_spec_mismatch_policy("warn", source="test.policy")


@pytest.mark.parametrize("value", [False, 0, 1, [], {}])
def test_normalize_spec_mismatch_policy_rejects_non_string_values(value: object) -> None:
    with pytest.raises(ValueError, match="must be a string policy"):
        normalize_spec_mismatch_policy(value, source="test.policy")


def test_require_fail_on_spec_mismatch_rejects_false() -> None:
    with pytest.raises(ValueError, match="must stay true"):
        require_fail_on_spec_mismatch(False, source="test.flag")


def test_normalize_bool_flag_rejects_string_boolean_flags() -> None:
    with pytest.raises(ValueError, match="must be a boolean"):
        normalize_bool_flag("false", source="test.flag", default=True)


def test_fail_fast_helpers_default_to_hard_fail() -> None:
    assert normalize_spec_mismatch_policy(None, source="test.policy") == HARD_FAIL_SPEC_MISMATCH_POLICY
    assert require_fail_on_spec_mismatch(None, source="test.flag") == HARD_FAIL_SPEC_MISMATCH_POLICY


def _model_config() -> ModelConfig:
    return ModelConfig(
        gru_hidden_size=256,
        encoder_mlp_width=256,
        encoder_mlp_layers=2,
        layer_norm=True,
        dropout=ModelDropoutConfig(family_a=0.0, ablation=0.1),
    )


def test_policy_value_model_forward_shapes() -> None:
    model = PolicyValueModel(observation_dim=512, config=_model_config())
    obs = torch.randn(4, 512)

    logits, value, next_hidden = model(obs)

    assert logits.shape == (4, GLOBAL_ACTION_SPACE_SIZE)
    assert value.shape == (4,)
    assert next_hidden.shape == (4, 256)


def test_policy_value_model_accepts_explicit_hidden_state() -> None:
    model = PolicyValueModel(observation_dim=512, config=_model_config())
    obs = torch.randn(3, 512)
    hidden_state = model.initial_hidden(batch_size=3)

    logits, value, next_hidden = model(obs, hidden_state)

    assert logits.shape == (3, GLOBAL_ACTION_SPACE_SIZE)
    assert value.shape == (3,)
    assert next_hidden.shape == hidden_state.shape


@pytest.mark.parametrize(
    ("obs_shape", "hidden_shape", "message"),
    [
        ((2, 511), None, "obs feature dimension mismatch"),
        ((2, 512), (3, 256), "hidden_state batch mismatch"),
        ((2, 512), (2, 255), "hidden_state feature mismatch"),
    ],
)
def test_policy_value_model_shape_checks(
    obs_shape: tuple[int, int],
    hidden_shape: tuple[int, int] | None,
    message: str,
) -> None:
    model = PolicyValueModel(observation_dim=512, config=_model_config())
    obs = torch.randn(*obs_shape)
    hidden_state = None if hidden_shape is None else torch.randn(*hidden_shape)

    with pytest.raises(ValueError, match=message):
        model(obs, hidden_state)
