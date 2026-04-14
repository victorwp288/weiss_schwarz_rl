from __future__ import annotations

import pytest
import torch

from weiss_rl.config.models import ModelConfig, ModelDropoutConfig
from weiss_rl.model import GLOBAL_ACTION_SPACE_SIZE, SEAT_COUNT, PolicyValueModel
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


def _typed_model_config() -> ModelConfig:
    return ModelConfig(
        gru_hidden_size=256,
        encoder_mlp_width=256,
        encoder_mlp_layers=2,
        layer_norm=True,
        dropout=ModelDropoutConfig(family_a=0.0, ablation=0.1),
        encoder_kind="typed_v1",
        typed_feature_width=64,
    )


def _feedforward_model_config() -> ModelConfig:
    return ModelConfig(
        gru_hidden_size=256,
        encoder_mlp_width=256,
        encoder_mlp_layers=2,
        layer_norm=True,
        dropout=ModelDropoutConfig(family_a=0.0, ablation=0.1),
        recurrent_core="none",
    )


def _typed_observation_spec() -> dict[str, object]:
    return {
        "obs_encoding_version": 2,
        "dtype": "f32",
        "obs_len": 18,
        "self_first": True,
        "header_fields": [
            {"name": "phase", "index": 0},
            {"name": "choice_total", "index": 1},
        ],
        "player_blocks": [
            {
                "name": "self",
                "base": 2,
                "len": 8,
                "slices": [
                    {"name": "level_count", "start": 0, "len": 1},
                    {"name": "clock_count", "start": 1, "len": 1},
                    {"name": "stage", "start": 2, "len": 6},
                ],
            },
            {
                "name": "opponent",
                "base": 10,
                "len": 6,
                "slices": [
                    {"name": "level_count", "start": 0, "len": 1},
                    {"name": "clock_count", "start": 1, "len": 1},
                    {"name": "stage", "start": 2, "len": 4},
                ],
            },
        ],
        "tail_slices": [
            {"name": "choice_page", "start": 16, "len": 2},
        ],
    }


def test_policy_value_model_forward_shapes() -> None:
    model = PolicyValueModel(observation_dim=512, config=_model_config())
    obs = torch.randn(4, 512)

    logits, value, next_hidden = model(obs)

    assert logits.shape == (4, GLOBAL_ACTION_SPACE_SIZE)
    assert value.shape == (4,)
    assert next_hidden.shape == (4, 256)


def test_policy_value_model_typed_encoder_forward_shapes() -> None:
    model = PolicyValueModel(
        observation_dim=18,
        config=_typed_model_config(),
        observation_spec=_typed_observation_spec(),
    )
    obs = torch.randn(4, 18)

    logits, value, next_hidden = model(obs)

    assert logits.shape == (4, GLOBAL_ACTION_SPACE_SIZE)
    assert value.shape == (4,)
    assert next_hidden.shape == (4, 256)


def test_policy_value_model_typed_encoder_requires_observation_spec() -> None:
    with pytest.raises(ValueError, match="requires observation_spec"):
        PolicyValueModel(observation_dim=18, config=_typed_model_config())


def test_policy_value_model_accepts_explicit_hidden_state() -> None:
    model = PolicyValueModel(observation_dim=512, config=_model_config())
    obs = torch.randn(3, 512)
    hidden_state = model.initial_hidden(batch_size=3)

    logits, value, next_hidden = model(obs, hidden_state)

    assert logits.shape == (3, GLOBAL_ACTION_SPACE_SIZE)
    assert value.shape == (3,)
    assert next_hidden.shape == hidden_state.shape


def test_policy_value_model_feedforward_core_keeps_hidden_shape_and_outputs() -> None:
    model = PolicyValueModel(observation_dim=512, config=_feedforward_model_config())
    obs = torch.randn(3, 512)
    hidden_state = model.initial_hidden(batch_size=3)

    logits, value, next_hidden = model(obs, hidden_state)

    assert logits.shape == (3, GLOBAL_ACTION_SPACE_SIZE)
    assert value.shape == (3,)
    assert next_hidden.shape == hidden_state.shape
    torch.testing.assert_close(next_hidden, hidden_state)


def test_policy_value_model_feedforward_seat_core_leaves_hidden_unchanged() -> None:
    model = PolicyValueModel(observation_dim=512, config=_feedforward_model_config())
    obs = torch.randn(4, 512)
    acting_seat = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    seat_hidden_state = model.initial_seat_hidden(batch_size=4)
    seat_hidden_state[:, 0, :] = -1.0
    seat_hidden_state[:, 1, :] = 2.0

    logits, value, next_hidden = model.forward_seat_aware(obs, acting_seat, seat_hidden_state)

    assert logits.shape == (4, GLOBAL_ACTION_SPACE_SIZE)
    assert value.shape == (4,)
    torch.testing.assert_close(next_hidden, seat_hidden_state)


def test_policy_value_model_initial_seat_hidden_shape() -> None:
    model = PolicyValueModel(observation_dim=512, config=_model_config())

    seat_hidden_state = model.initial_seat_hidden(batch_size=5)

    assert seat_hidden_state.shape == (5, SEAT_COUNT, 256)
    torch.testing.assert_close(seat_hidden_state, torch.zeros_like(seat_hidden_state))


def test_policy_value_model_seat_aware_forward_updates_only_acting_seat() -> None:
    torch.manual_seed(0)
    model = PolicyValueModel(observation_dim=512, config=_model_config())
    obs = torch.randn(4, 512)
    acting_seat = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    seat_hidden_state = model.initial_seat_hidden(batch_size=4)
    seat_hidden_state[:, 0, :] = -1.0
    seat_hidden_state[:, 1, :] = 2.0

    logits, value, next_seat_hidden = model.forward_seat_aware(obs, acting_seat, seat_hidden_state)

    assert logits.shape == (4, GLOBAL_ACTION_SPACE_SIZE)
    assert value.shape == (4,)
    assert next_seat_hidden.shape == seat_hidden_state.shape

    batch_index = torch.arange(obs.shape[0])
    non_acting_seat = 1 - acting_seat
    torch.testing.assert_close(
        next_seat_hidden[batch_index, non_acting_seat],
        seat_hidden_state[batch_index, non_acting_seat],
    )
    assert not torch.allclose(
        next_seat_hidden[batch_index, acting_seat],
        seat_hidden_state[batch_index, acting_seat],
    )


def test_policy_value_model_seat_aware_inplace_matches_regular_forward() -> None:
    torch.manual_seed(7)
    model = PolicyValueModel(observation_dim=512, config=_model_config())
    obs = torch.randn(4, 512)
    acting_seat = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    seat_hidden_state = model.initial_seat_hidden(batch_size=4)
    seat_hidden_state[:, 0, :] = -1.0
    seat_hidden_state[:, 1, :] = 2.0

    logits_ref, value_ref, next_hidden_ref = model.forward_seat_aware(
        obs,
        acting_seat,
        seat_hidden_state.clone(),
    )
    inplace_hidden = seat_hidden_state.clone()
    logits_inplace, value_inplace, next_hidden_inplace = model.forward_seat_aware_inplace(
        obs,
        acting_seat,
        inplace_hidden,
    )

    torch.testing.assert_close(logits_inplace, logits_ref)
    torch.testing.assert_close(value_inplace, value_ref)
    torch.testing.assert_close(next_hidden_inplace, next_hidden_ref)
    torch.testing.assert_close(inplace_hidden, next_hidden_ref)



def test_policy_value_model_write_acting_hidden_preserves_state_dtype() -> None:
    model = PolicyValueModel(observation_dim=512, config=_model_config())
    seat_hidden_state = model.initial_seat_hidden(batch_size=3)
    acting_seat = torch.tensor([0, 1, 0], dtype=torch.long)
    next_acting_hidden = torch.randn(3, model.hidden_size, dtype=torch.float16)

    next_seat_hidden = model._write_acting_hidden(seat_hidden_state, acting_seat, next_acting_hidden)

    assert next_seat_hidden.dtype == seat_hidden_state.dtype
    assert next_seat_hidden.shape == seat_hidden_state.shape


def test_policy_value_model_seat_aware_poisoned_other_seat_does_not_change_outputs() -> None:
    torch.manual_seed(1)
    model = PolicyValueModel(observation_dim=512, config=_model_config())
    obs = torch.randn(4, 512)
    acting_seat = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    clean_hidden_state = model.initial_seat_hidden(batch_size=4)
    poisoned_hidden_state = clean_hidden_state.clone()
    batch_index = torch.arange(obs.shape[0])
    non_acting_seat = 1 - acting_seat
    poisoned_hidden_state[batch_index, non_acting_seat] = 10_000.0

    clean_logits, clean_value, clean_next_hidden = model.forward_seat_aware(obs, acting_seat, clean_hidden_state)
    poisoned_logits, poisoned_value, poisoned_next_hidden = model.forward_seat_aware(
        obs,
        acting_seat,
        poisoned_hidden_state,
    )

    torch.testing.assert_close(poisoned_logits, clean_logits)
    torch.testing.assert_close(poisoned_value, clean_value)
    torch.testing.assert_close(
        poisoned_next_hidden[batch_index, acting_seat],
        clean_next_hidden[batch_index, acting_seat],
    )
    torch.testing.assert_close(
        poisoned_next_hidden[batch_index, non_acting_seat],
        poisoned_hidden_state[batch_index, non_acting_seat],
    )


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


@pytest.mark.parametrize(
    ("seat_hidden_shape", "acting_seat", "message"),
    [
        ((2, 256), torch.tensor([0, 1]), "seat_hidden_state must be 3D"),
        ((2, 3, 256), torch.tensor([0, 1]), "seat_hidden_state seat mismatch"),
        ((2, 2, 255), torch.tensor([0, 1]), "seat_hidden_state feature mismatch"),
        ((2, 2, 256), torch.tensor([0]), "acting_seat batch mismatch"),
        ((2, 2, 256), torch.tensor([0, 2]), "acting_seat values must be 0 or 1"),
    ],
)
def test_policy_value_model_seat_aware_shape_checks(
    seat_hidden_shape: tuple[int, ...],
    acting_seat: torch.Tensor,
    message: str,
) -> None:
    model = PolicyValueModel(observation_dim=512, config=_model_config())
    obs = torch.randn(2, 512)
    seat_hidden_state = torch.randn(*seat_hidden_shape)

    with pytest.raises(ValueError, match=message):
        model.forward_seat_aware(obs, acting_seat, seat_hidden_state)
