from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from weiss_rl.action_catalog import ActionCatalog
from weiss_rl.config.models import ModelConfig, ModelDropoutConfig
from weiss_rl.legal_actions import LegalActionBatch
from weiss_rl.model import (
    GLOBAL_ACTION_SPACE_SIZE,
    SEAT_COUNT,
    PolicyValueModel,
    StructuredLegalPolicyValueModel,
    _negative_logits_fill_value,
    build_policy_value_model,
)
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


def test_build_policy_value_model_rejects_public_heuristic_config_on_non_structured_model() -> None:
    with pytest.raises(ValueError, match="public_heuristic_\\* model settings require encoder_kind='structured_v2'"):
        build_policy_value_model(
            observation_dim=32,
            action_dim=9,
            config=replace(_model_config(), public_heuristic_logit_bias_scale=0.5),
        )


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


def _structured_model_config() -> ModelConfig:
    return ModelConfig(
        gru_hidden_size=256,
        encoder_mlp_width=128,
        encoder_mlp_layers=2,
        layer_norm=True,
        dropout=ModelDropoutConfig(family_a=0.0, ablation=0.1),
        encoder_kind="structured_v2",
        typed_feature_width=64,
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


def _structured_spec_bundle() -> dict[str, object]:
    observation = _typed_observation_spec()
    action = {
        "action_encoding_version": 1,
        "action_space_size": 9,
        "pass_action_id": 8,
        "constants": [["MAX_HAND", 2], ["MAX_STAGE", 2], ["ATTACK_SLOT_COUNT", 1]],
        "families": [
            {"name": "main_play_character", "base": 0, "count": 4},
            {"name": "main_move", "base": 4, "count": 2},
            {"name": "attack", "base": 6, "count": 2},
            {"name": "pass", "base": 8, "count": 1},
        ],
        "attack_type_encoding": [["frontal", 0]],
    }
    return {
        "action": action,
        "observation": observation,
        "compatibility_hash": "structured_v2_test_hash",
    }


def _structured_choice_slot_spec_bundle() -> dict[str, object]:
    observation = _typed_observation_spec()
    action = {
        "action_encoding_version": 1,
        "action_space_size": 5,
        "pass_action_id": 4,
        "constants": [["MAX_HAND", 1], ["MAX_STAGE", 2], ["ATTACK_SLOT_COUNT", 1]],
        "families": [
            {"name": "choice_select", "base": 0, "count": 2},
            {"name": "encore_pay", "base": 2, "count": 2},
            {"name": "pass", "base": 4, "count": 1},
        ],
        "attack_type_encoding": [["frontal", 0]],
    }
    return {
        "action": action,
        "observation": observation,
        "compatibility_hash": "structured_v2_choice_slot_test_hash",
    }


def _structured_hand_observation_spec() -> dict[str, object]:
    return {
        "obs_encoding_version": 2,
        "dtype": "f32",
        "obs_len": 8,
        "self_first": True,
        "sentinel_hidden": -1,
        "sentinel_empty_card": 0,
        "header_fields": [
            {"name": "phase", "index": 0},
            {"name": "choice_total", "index": 1},
        ],
        "player_blocks": [
            {
                "name": "self",
                "base": 2,
                "len": 4,
                "slices": [
                    {"name": "stage", "start": 0, "len": 2},
                    {"name": "hand", "start": 2, "len": 2},
                ],
            },
            {
                "name": "opponent",
                "base": 6,
                "len": 2,
                "slices": [
                    {"name": "stage", "start": 0, "len": 2},
                ],
            },
        ],
        "tail_slices": [],
    }


def _structured_hand_spec_bundle() -> dict[str, object]:
    return {
        "observation": _structured_hand_observation_spec(),
        "action": {
            "action_encoding_version": 1,
            "action_space_size": 5,
            "pass_action_id": 4,
            "constants": [["MAX_HAND", 2], ["MAX_STAGE", 2], ["ATTACK_SLOT_COUNT", 1]],
            "families": [
                {"name": "main_play_character", "base": 0, "count": 4},
                {"name": "pass", "base": 4, "count": 1},
            ],
            "attack_type_encoding": [["frontal", 0]],
        },
        "compatibility_hash": "structured_v2_hand_test_hash",
    }


def _packed_meta_from_ids(action_catalog: ActionCatalog, packed_ids: np.ndarray) -> np.ndarray:
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    attack_type_index = {name: index for index, name in enumerate(action_catalog.attack_type_names)}
    unused = np.iinfo(np.uint16).max
    rows = np.full((int(packed_ids.shape[0]), 4), unused, dtype=np.uint16)
    for row_index, action_id in enumerate(np.asarray(packed_ids, dtype=np.int64).tolist()):
        decoded = action_catalog.decode(int(action_id))
        rows[row_index, 0] = np.uint16(family_index[decoded.family])
        if decoded.hand_index is not None:
            rows[row_index, 1] = np.uint16(decoded.hand_index)
        if decoded.stage_slot is not None:
            rows[row_index, 2] = np.uint16(decoded.stage_slot)
        if decoded.from_slot is not None:
            rows[row_index, 1] = np.uint16(decoded.from_slot)
        if decoded.to_slot is not None:
            rows[row_index, 2] = np.uint16(decoded.to_slot)
        if decoded.slot is not None:
            rows[row_index, 1] = np.uint16(decoded.slot)
        if decoded.attack_type is not None:
            rows[row_index, 2] = np.uint16(attack_type_index[decoded.attack_type])
        if decoded.index is not None:
            rows[row_index, 1] = np.uint16(decoded.index)
    return rows


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


def test_structured_legal_policy_value_model_scores_legal_candidates() -> None:
    model = build_policy_value_model(
        observation_dim=18,
        config=_structured_model_config(),
        action_dim=9,
        observation_spec=_typed_observation_spec(),
        spec_bundle=_structured_spec_bundle(),
    )
    assert isinstance(model, StructuredLegalPolicyValueModel)

    obs = torch.zeros((2, 18), dtype=torch.float32)
    seat_hidden = model.initial_seat_hidden(2)
    legal_mask = np.zeros((1, 2, 9), dtype=np.bool_)
    legal_mask[0, 0, 0] = True
    legal_mask[0, 1, 4] = True
    logits, values, next_hidden = model.forward_seat_aware(
        obs,
        torch.tensor([0, 1]),
        seat_hidden,
        legal_actions=LegalActionBatch.from_mask(legal_mask),
    )

    assert logits.shape == (2, 9)
    assert values.shape == (2,)
    assert next_hidden.shape == (2, 2, 256)
    assert torch.isfinite(logits[:, [0, 4]]).all()
    assert torch.all(logits[:, [1, 2, 3, 5, 6, 7, 8]] < -1e8)


def test_structured_legal_policy_value_model_applies_candidate_scoring_chunk_config() -> None:
    model = build_policy_value_model(
        observation_dim=18,
        config=replace(
            _structured_model_config(),
            candidate_scoring_chunk_size=131072,
            cuda_learner_candidate_scoring_chunk_size=524288,
        ),
        action_dim=9,
        observation_spec=_typed_observation_spec(),
        spec_bundle=_structured_spec_bundle(),
    )
    assert isinstance(model, StructuredLegalPolicyValueModel)

    assert model.policy_head._candidate_scoring_chunk_size == 131072
    assert model.policy_head._cuda_learner_candidate_scoring_chunk_size == 524288


def test_structured_legal_policy_value_model_distinguishes_index_and_slot_argument_families() -> None:
    torch.manual_seed(0)
    model = build_policy_value_model(
        observation_dim=18,
        config=_structured_model_config(),
        action_dim=5,
        observation_spec=_typed_observation_spec(),
        spec_bundle=_structured_choice_slot_spec_bundle(),
    )
    assert isinstance(model, StructuredLegalPolicyValueModel)

    obs = torch.zeros((1, 18), dtype=torch.float32)
    obs[0, 4] = 1.0
    obs[0, 7] = 2.0
    seat_hidden = model.initial_seat_hidden(1)
    legal_mask = np.zeros((1, 1, 5), dtype=np.bool_)
    legal_mask[0, 0, [0, 1, 2, 3, 4]] = True
    logits, _values, _next_hidden = model.forward_seat_aware(
        obs,
        torch.tensor([0]),
        seat_hidden,
        legal_actions=LegalActionBatch.from_mask(legal_mask),
    )

    assert not torch.isclose(logits[0, 0], logits[0, 1], atol=1e-6, rtol=0.0)
    assert not torch.isclose(logits[0, 2], logits[0, 3], atol=1e-6, rtol=0.0)


def test_structured_legal_policy_value_model_packed_legal_matches_mask() -> None:
    torch.manual_seed(0)
    model = build_policy_value_model(
        observation_dim=18,
        config=_structured_model_config(),
        action_dim=9,
        observation_spec=_typed_observation_spec(),
        spec_bundle=_structured_spec_bundle(),
    )
    assert isinstance(model, StructuredLegalPolicyValueModel)

    obs = torch.zeros((2, 18), dtype=torch.float32)
    seat_hidden = model.initial_seat_hidden(2)
    legal_mask = np.zeros((1, 2, 9), dtype=np.bool_)
    legal_mask[0, 0, [0, 3, 5]] = True
    legal_mask[0, 1, [1, 4, 8]] = True
    packed_ids = np.asarray([0, 3, 5, 1, 4, 8], dtype=np.int32)
    packed_offsets = np.asarray([0, 3, 6], dtype=np.int32)

    logits_mask, values_mask, next_hidden_mask = model.forward_seat_aware(
        obs,
        torch.tensor([0, 1]),
        seat_hidden,
        legal_actions=LegalActionBatch.from_mask(legal_mask),
    )
    logits_packed, values_packed, next_hidden_packed = model.forward_seat_aware(
        obs,
        torch.tensor([0, 1]),
        seat_hidden,
        legal_actions=LegalActionBatch.from_packed(packed_ids, packed_offsets, action_space=9),
    )

    torch.testing.assert_close(logits_mask, logits_packed)
    torch.testing.assert_close(values_mask, values_packed)
    torch.testing.assert_close(next_hidden_mask, next_hidden_packed)


def test_structured_legal_policy_value_model_packed_meta_matches_packed_ids() -> None:
    torch.manual_seed(0)
    model = build_policy_value_model(
        observation_dim=18,
        config=_structured_model_config(),
        action_dim=9,
        observation_spec=_typed_observation_spec(),
        spec_bundle=_structured_spec_bundle(),
    )
    assert isinstance(model, StructuredLegalPolicyValueModel)

    obs = torch.zeros((2, 18), dtype=torch.float32)
    seat_hidden = model.initial_seat_hidden(2)
    packed_ids = np.asarray([4, 6, 8, 5, 7, 8], dtype=np.int32)
    packed_offsets = np.asarray([0, 3, 6], dtype=np.int32)
    action_catalog = ActionCatalog.from_spec_bundle(_structured_spec_bundle())
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)

    logits_packed, values_packed, next_hidden_packed = model.forward_seat_aware(
        obs,
        torch.tensor([0, 1]),
        seat_hidden,
        legal_actions=LegalActionBatch.from_packed(packed_ids, packed_offsets, action_space=9),
    )
    logits_meta, values_meta, next_hidden_meta = model.forward_seat_aware(
        obs,
        torch.tensor([0, 1]),
        seat_hidden,
        legal_actions=LegalActionBatch.from_packed(
            packed_ids,
            packed_offsets,
            meta=packed_meta,
            action_space=9,
        ),
    )

    torch.testing.assert_close(logits_packed, logits_meta)
    torch.testing.assert_close(values_packed, values_meta)
    torch.testing.assert_close(next_hidden_packed, next_hidden_meta)


def test_structured_legal_policy_value_model_factorized_packed_sampling_returns_legal_actions() -> None:
    torch.manual_seed(0)
    model = build_policy_value_model(
        observation_dim=8,
        config=replace(_structured_model_config(), structured_policy_contract="factorized_v1"),
        action_dim=5,
        observation_spec=_structured_hand_observation_spec(),
        spec_bundle=_structured_hand_spec_bundle(),
    )
    assert isinstance(model, StructuredLegalPolicyValueModel)
    assert model.supports_factorized_legal_policy is True

    obs = torch.zeros((2, 8), dtype=torch.float32)
    obs[0, 4] = 11
    obs[1, 5] = 22
    acting_seat = torch.tensor([0, 1], dtype=torch.long)
    seat_hidden = model.initial_seat_hidden(2)
    packed_ids = np.asarray([0, 1, 4, 2, 3, 4], dtype=np.int32)
    packed_offsets = np.asarray([0, 3, 6], dtype=np.int32)
    packed_meta = _packed_meta_from_ids(ActionCatalog.from_spec_bundle(_structured_hand_spec_bundle()), packed_ids)
    legal_actions = LegalActionBatch.from_packed(
        packed_ids,
        packed_offsets,
        meta=packed_meta,
        action_space=5,
    )

    sampled_actions, sampled_logp, values, next_hidden = model.sample_factorized_packed_seat_aware(
        obs,
        acting_seat,
        seat_hidden,
        legal_actions=legal_actions,
        sample_seeds=torch.tensor([12345, 67890], dtype=torch.long),
        pass_action_id=4,
    )
    factorized_eval = model.evaluate_factorized_sequence_packed_seat_aware(
        obs.unsqueeze(0),
        acting_seat.unsqueeze(0),
        seat_hidden,
        legal_actions=legal_actions,
        actions=sampled_actions.unsqueeze(0),
    )

    assert sampled_actions.shape == (2,)
    assert sampled_logp.shape == (2,)
    assert values.shape == (2,)
    assert next_hidden.shape == (2, 2, 256)
    assert torch.isfinite(sampled_logp).all()
    assert int(sampled_actions[0].item()) in set(packed_ids[:4].tolist())
    assert int(sampled_actions[1].item()) in set(packed_ids[3:].tolist())
    assert factorized_eval.values.shape == (1, 2)
    assert factorized_eval.action_logp is not None
    assert factorized_eval.entropy is not None
    assert factorized_eval.family_log_probs.shape[:2] == (1, 2)
    assert torch.isfinite(factorized_eval.action_logp).all()
    assert torch.isfinite(factorized_eval.entropy).all()


def test_structured_legal_policy_value_model_packed_path_uses_hand_position_for_duplicate_cards() -> None:
    torch.manual_seed(0)
    model = build_policy_value_model(
        observation_dim=8,
        config=_structured_model_config(),
        action_dim=5,
        observation_spec=_structured_hand_observation_spec(),
        spec_bundle=_structured_hand_spec_bundle(),
    )
    assert isinstance(model, StructuredLegalPolicyValueModel)

    obs = torch.zeros((1, 8), dtype=torch.float32)
    obs[0, 4] = 11.0
    obs[0, 5] = 11.0
    seat_hidden = model.initial_seat_hidden(1)
    packed_ids = np.asarray([0, 2, 4], dtype=np.int32)
    packed_offsets = np.asarray([0, 3], dtype=np.int32)

    logits, _values, _next_hidden = model.forward_seat_aware(
        obs,
        torch.tensor([0]),
        seat_hidden,
        legal_actions=LegalActionBatch.from_packed(packed_ids, packed_offsets, action_space=5),
    )

    assert not torch.isclose(logits[0, 0], logits[0, 2], atol=1e-6, rtol=0.0)


def test_structured_legal_policy_value_model_factorized_path_uses_action_ids_as_canonical_source() -> None:
    torch.manual_seed(0)
    model = build_policy_value_model(
        observation_dim=8,
        config=replace(_structured_model_config(), structured_policy_contract="factorized_v1"),
        action_dim=5,
        observation_spec=_structured_hand_observation_spec(),
        spec_bundle=_structured_hand_spec_bundle(),
    )
    assert isinstance(model, StructuredLegalPolicyValueModel)

    obs = torch.zeros((1, 8), dtype=torch.float32)
    obs[0, 4] = 11
    obs[0, 5] = 22
    acting_seat = torch.tensor([0], dtype=torch.long)
    seat_hidden = model.initial_seat_hidden(1)
    packed_ids = np.asarray([0, 1, 4], dtype=np.int32)
    packed_offsets = np.asarray([0, 3], dtype=np.int32)
    packed_meta = _packed_meta_from_ids(ActionCatalog.from_spec_bundle(_structured_hand_spec_bundle()), packed_ids)
    corrupt_meta = packed_meta.copy()
    corrupt_meta[:, 0] = np.uint16(0)
    corrupt_meta[:, 1:] = np.uint16(np.iinfo(np.uint16).max)

    legal_actions = LegalActionBatch.from_packed(
        packed_ids,
        packed_offsets,
        meta=packed_meta,
        action_space=5,
    )
    corrupt_legal_actions = LegalActionBatch.from_packed(
        packed_ids,
        packed_offsets,
        meta=corrupt_meta,
        action_space=5,
    )
    actions = torch.tensor([[1]], dtype=torch.long)

    reference = model.evaluate_factorized_sequence_packed_seat_aware(
        obs.unsqueeze(0),
        acting_seat.unsqueeze(0),
        seat_hidden,
        legal_actions=legal_actions,
        actions=actions,
    )
    corrupted = model.evaluate_factorized_sequence_packed_seat_aware(
        obs.unsqueeze(0),
        acting_seat.unsqueeze(0),
        seat_hidden,
        legal_actions=corrupt_legal_actions,
        actions=actions,
    )

    assert reference.action_logp is not None
    assert corrupted.action_logp is not None
    torch.testing.assert_close(corrupted.action_logp, reference.action_logp)
    torch.testing.assert_close(corrupted.entropy, reference.entropy)


def test_structured_legal_policy_value_model_factorized_chunking_matches_unchunked() -> None:
    torch.manual_seed(0)
    model = build_policy_value_model(
        observation_dim=8,
        config=replace(_structured_model_config(), structured_policy_contract="factorized_v1"),
        action_dim=5,
        observation_spec=_structured_hand_observation_spec(),
        spec_bundle=_structured_hand_spec_bundle(),
    )
    assert isinstance(model, StructuredLegalPolicyValueModel)

    obs = torch.zeros((2, 8), dtype=torch.float32)
    obs[0, 4] = 11
    obs[1, 5] = 22
    acting_seat = torch.tensor([0, 1], dtype=torch.long)
    seat_hidden = model.initial_seat_hidden(2)
    packed_ids = np.asarray([0, 1, 4, 2, 3, 4], dtype=np.int32)
    packed_offsets = np.asarray([0, 3, 6], dtype=np.int32)
    packed_meta = _packed_meta_from_ids(ActionCatalog.from_spec_bundle(_structured_hand_spec_bundle()), packed_ids)
    legal_actions = LegalActionBatch.from_packed(
        packed_ids,
        packed_offsets,
        meta=packed_meta,
        action_space=5,
    )
    actions = torch.tensor([[1, 3]], dtype=torch.long)

    baseline = model.evaluate_factorized_sequence_packed_seat_aware(
        obs.unsqueeze(0),
        acting_seat.unsqueeze(0),
        seat_hidden,
        legal_actions=legal_actions,
        actions=actions,
    )
    model.policy_head._factorized_row_chunk_size = lambda _row_states: 1  # type: ignore[method-assign]
    chunked = model.evaluate_factorized_sequence_packed_seat_aware(
        obs.unsqueeze(0),
        acting_seat.unsqueeze(0),
        seat_hidden,
        legal_actions=legal_actions,
        actions=actions,
    )

    assert baseline.action_logp is not None
    assert chunked.action_logp is not None
    torch.testing.assert_close(chunked.action_logp, baseline.action_logp)
    torch.testing.assert_close(chunked.entropy, baseline.entropy)
    torch.testing.assert_close(chunked.family_log_probs, baseline.family_log_probs)


def test_structured_legal_policy_value_model_factorized_multi_family_sampling_is_legal() -> None:
    torch.manual_seed(0)
    model = build_policy_value_model(
        observation_dim=18,
        config=replace(_structured_model_config(), structured_policy_contract="factorized_v1"),
        action_dim=9,
        observation_spec=_typed_observation_spec(),
        spec_bundle=_structured_spec_bundle(),
    )
    assert isinstance(model, StructuredLegalPolicyValueModel)

    obs = torch.zeros((2, 18), dtype=torch.float32)
    acting_seat = torch.tensor([0, 1], dtype=torch.long)
    seat_hidden = model.initial_seat_hidden(2)
    packed_ids = np.asarray([4, 6, 8, 5, 7, 8], dtype=np.int32)
    packed_offsets = np.asarray([0, 3, 6], dtype=np.int32)
    packed_meta = _packed_meta_from_ids(ActionCatalog.from_spec_bundle(_structured_spec_bundle()), packed_ids)
    legal_actions = LegalActionBatch.from_packed(
        packed_ids,
        packed_offsets,
        meta=packed_meta,
        action_space=9,
    )

    sampled_actions, sampled_logp, values, next_hidden = model.sample_factorized_packed_seat_aware(
        obs,
        acting_seat,
        seat_hidden,
        legal_actions=legal_actions,
        sample_seeds=torch.tensor([13579, 24680], dtype=torch.long),
        pass_action_id=8,
    )
    factorized_eval = model.evaluate_factorized_sequence_packed_seat_aware(
        obs.unsqueeze(0),
        acting_seat.unsqueeze(0),
        seat_hidden,
        legal_actions=legal_actions,
        actions=sampled_actions.unsqueeze(0),
    )

    assert sampled_actions.shape == (2,)
    assert sampled_logp.shape == (2,)
    assert values.shape == (2,)
    assert next_hidden.shape == (2, 2, 256)
    assert torch.isfinite(sampled_logp).all()
    assert int(sampled_actions[0].item()) in set(packed_ids[:3].tolist())
    assert int(sampled_actions[1].item()) in set(packed_ids[3:].tolist())
    assert factorized_eval.action_logp is not None
    assert factorized_eval.entropy is not None
    assert factorized_eval.attack_slot_log_probs is not None
    assert factorized_eval.attack_type_log_probs is not None
    assert torch.isfinite(factorized_eval.action_logp).all()
    assert torch.isfinite(factorized_eval.entropy).all()


def test_structured_legal_policy_value_model_actor_and_learner_packed_scorers_match() -> None:
    torch.manual_seed(0)
    model = build_policy_value_model(
        observation_dim=18,
        config=_structured_model_config(),
        action_dim=9,
        observation_spec=_typed_observation_spec(),
        spec_bundle=_structured_spec_bundle(),
    )
    assert isinstance(model, StructuredLegalPolicyValueModel)

    obs = torch.zeros((2, 18), dtype=torch.float32)
    acting_seat = torch.tensor([0, 1], dtype=torch.long)
    seat_hidden = model.initial_seat_hidden(2)
    packed_ids = np.asarray([0, 4, 6, 8, 3, 5, 7, 8], dtype=np.int32)
    packed_offsets = np.asarray([0, 4, 8], dtype=np.int32)
    packed_meta = _packed_meta_from_ids(ActionCatalog.from_spec_bundle(_structured_spec_bundle()), packed_ids)
    legal_actions = LegalActionBatch.from_packed(
        packed_ids,
        packed_offsets,
        meta=packed_meta,
        action_space=9,
    )

    recurrent_output, state_repr, observation_context, _value, _next_hidden = model.forward_trunk_packed_seat_aware(
        obs,
        acting_seat,
        seat_hidden,
    )

    learner_scores = model.score_packed_legal_candidates(
        recurrent_output,
        obs,
        legal_actions,
        state_repr=state_repr,
        observation_context=observation_context,
        scoring_mode="learner",
    )
    with torch.no_grad():
        actor_scores = model.score_packed_legal_candidates(
            recurrent_output,
            obs,
            legal_actions,
            state_repr=state_repr,
            observation_context=observation_context,
            scoring_mode="actor",
        )

    torch.testing.assert_close(learner_scores, actor_scores)


def test_structured_legal_policy_value_model_packed_plan_scorer_matches_legacy_chunked_scorer() -> None:
    torch.manual_seed(0)
    model = build_policy_value_model(
        observation_dim=18,
        config=_structured_model_config(),
        action_dim=9,
        observation_spec=_typed_observation_spec(),
        spec_bundle=_structured_spec_bundle(),
    )
    assert isinstance(model, StructuredLegalPolicyValueModel)

    obs = torch.zeros((2, 18), dtype=torch.float32)
    acting_seat = torch.tensor([0, 1], dtype=torch.long)
    seat_hidden = model.initial_seat_hidden(2)
    packed_ids = np.asarray([0, 4, 6, 8, 3, 5, 7, 8], dtype=np.int32)
    packed_offsets = np.asarray([0, 4, 8], dtype=np.int32)
    action_catalog = ActionCatalog.from_spec_bundle(_structured_spec_bundle())
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    legal_actions = LegalActionBatch.from_packed(
        packed_ids,
        packed_offsets,
        meta=packed_meta,
        action_space=9,
    )

    recurrent_output, state_repr, observation_context, _value, _next_hidden = model.forward_trunk_packed_seat_aware(
        obs,
        acting_seat,
        seat_hidden,
    )
    ids = torch.as_tensor(legal_actions.ids, dtype=torch.long)
    offsets = torch.as_tensor(legal_actions.offsets, dtype=torch.long)
    meta = torch.as_tensor(legal_actions.meta, dtype=torch.long)
    row_indices = torch.repeat_interleave(torch.arange(obs.shape[0], dtype=torch.long), offsets[1:] - offsets[:-1])

    legacy_scores = model.policy_head._score_candidates_chunked(  # type: ignore[attr-defined]
        state_repr,
        row_indices,
        ids,
        observation_context,
        candidate_meta=meta,
        scoring_mode="learner",
    )
    plan_scores = model.score_packed_legal_candidates(
        recurrent_output,
        obs,
        legal_actions,
        state_repr=state_repr,
        observation_context=observation_context,
        scoring_mode="learner",
    )

    torch.testing.assert_close(plan_scores, legacy_scores)


def test_structured_legal_policy_value_model_can_split_actor_and_learner_public_bias() -> None:
    torch.manual_seed(0)
    model = build_policy_value_model(
        observation_dim=8,
        config=replace(
            _structured_model_config(),
            public_heuristic_logit_bias_scale=1.0,
            public_heuristic_actor_logit_bias_scale=0.0,
        ),
        action_dim=5,
        observation_spec=_structured_hand_observation_spec(),
        spec_bundle=_structured_hand_spec_bundle(),
    )
    assert isinstance(model, StructuredLegalPolicyValueModel)

    obs = torch.zeros((1, 8), dtype=torch.float32)
    obs[0, 4] = 11
    obs[0, 5] = 11
    acting_seat = torch.tensor([0], dtype=torch.long)
    seat_hidden = model.initial_seat_hidden(1)
    packed_ids = np.asarray([0, 1, 2, 3, 4], dtype=np.int32)
    packed_offsets = np.asarray([0, 5], dtype=np.int32)
    packed_meta = _packed_meta_from_ids(ActionCatalog.from_spec_bundle(_structured_hand_spec_bundle()), packed_ids)
    legal_actions = LegalActionBatch.from_packed(
        packed_ids,
        packed_offsets,
        meta=packed_meta,
        action_space=5,
    )

    recurrent_output, state_repr, observation_context, _value, _next_hidden = model.forward_trunk_packed_seat_aware(
        obs,
        acting_seat,
        seat_hidden,
    )

    learner_scores = model.score_packed_legal_candidates(
        recurrent_output,
        obs,
        legal_actions,
        state_repr=state_repr,
        observation_context=observation_context,
        scoring_mode="learner",
    )
    with torch.no_grad():
        actor_scores = model.score_packed_legal_candidates(
            recurrent_output,
            obs,
            legal_actions,
            state_repr=state_repr,
            observation_context=observation_context,
            scoring_mode="actor",
        )

    assert torch.max(torch.abs(learner_scores - actor_scores)).item() > 1e-4


def test_structured_forward_seat_aware_respects_explicit_scoring_mode() -> None:
    torch.manual_seed(0)
    model = build_policy_value_model(
        observation_dim=8,
        config=replace(
            _structured_model_config(),
            public_heuristic_logit_bias_scale=0.0,
            public_heuristic_actor_logit_bias_scale=100.0,
        ),
        action_dim=5,
        observation_spec=_structured_hand_observation_spec(),
        spec_bundle=_structured_hand_spec_bundle(),
    )
    assert isinstance(model, StructuredLegalPolicyValueModel)

    obs = torch.zeros((1, 8), dtype=torch.float32)
    obs[0, 4] = 11
    obs[0, 5] = 11
    acting_seat = torch.tensor([0], dtype=torch.long)
    seat_hidden = model.initial_seat_hidden(1)
    packed_ids = np.asarray([0, 1, 2, 3, 4], dtype=np.int32)
    packed_offsets = np.asarray([0, 5], dtype=np.int32)
    packed_meta = _packed_meta_from_ids(ActionCatalog.from_spec_bundle(_structured_hand_spec_bundle()), packed_ids)
    legal_actions = LegalActionBatch.from_packed(
        packed_ids,
        packed_offsets,
        meta=packed_meta,
        action_space=5,
    )

    with torch.inference_mode():
        learner_logits, _learner_value, _learner_next_hidden = model.forward_seat_aware(
            obs,
            acting_seat,
            seat_hidden,
            legal_actions=legal_actions,
            scoring_mode="learner",
        )
        actor_logits, _actor_value, _actor_next_hidden = model.forward_seat_aware(
            obs,
            acting_seat,
            seat_hidden,
            legal_actions=legal_actions,
            scoring_mode="actor",
        )

    assert torch.max(torch.abs(learner_logits - actor_logits)).item() > 1e-4


def test_structured_legal_policy_value_model_sequence_forward_matches_stepwise_packed() -> None:
    torch.manual_seed(0)
    model = build_policy_value_model(
        observation_dim=18,
        config=_structured_model_config(),
        action_dim=9,
        observation_spec=_typed_observation_spec(),
        spec_bundle=_structured_spec_bundle(),
    )
    assert isinstance(model, StructuredLegalPolicyValueModel)

    obs = torch.zeros((2, 2, 18), dtype=torch.float32)
    acting_seat = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    seat_hidden = model.initial_seat_hidden(2)
    packed_ids = np.asarray([0, 3, 1, 4, 2, 5, 6, 8], dtype=np.int32)
    packed_offsets = np.asarray([0, 2, 4, 6, 8], dtype=np.int32)
    legal_actions = LegalActionBatch.from_packed(packed_ids, packed_offsets, action_space=9)

    logits_sequence, values_sequence, next_hidden_sequence = model.forward_sequence_seat_aware(
        obs,
        acting_seat,
        seat_hidden,
        legal_actions=legal_actions,
    )

    step_hidden = seat_hidden.clone()
    logits_steps: list[torch.Tensor] = []
    value_steps: list[torch.Tensor] = []
    row_cursor = 0
    for step_index in range(obs.shape[0]):
        step_offsets = packed_offsets[row_cursor : row_cursor + obs.shape[1] + 1]
        step_ids = packed_ids[int(step_offsets[0]) : int(step_offsets[-1])]
        step_legal_actions = LegalActionBatch.from_packed(
            step_ids,
            step_offsets - int(step_offsets[0]),
            action_space=9,
        )
        step_logits, step_values, step_hidden = model.forward_seat_aware(
            obs[step_index],
            acting_seat[step_index],
            step_hidden,
            legal_actions=step_legal_actions,
        )
        logits_steps.append(step_logits)
        value_steps.append(step_values)
        row_cursor += int(obs.shape[1])

    torch.testing.assert_close(logits_sequence, torch.stack(logits_steps, dim=0))
    torch.testing.assert_close(values_sequence, torch.stack(value_steps, dim=0))
    torch.testing.assert_close(next_hidden_sequence, step_hidden)


def test_structured_legal_policy_value_model_accepts_card_table_features() -> None:
    model = build_policy_value_model(
        observation_dim=8,
        config=_structured_model_config(),
        action_dim=5,
        observation_spec=_structured_hand_observation_spec(),
        spec_bundle=_structured_hand_spec_bundle(),
        card_table={
            "rows": [
                {
                    "card_id": 11,
                    "level": 1,
                    "cost": 0,
                    "power": 4500,
                    "soul": 1,
                    "color": "yellow",
                    "card_type": "character",
                    "traits": ["music"],
                }
            ]
        },
    )

    assert isinstance(model, StructuredLegalPolicyValueModel)
    assert model.policy_head._card_static_features.shape[1] > 0


def test_structured_v2_uses_hand_position_when_scoring_matching_cards() -> None:
    spec_bundle = _structured_hand_spec_bundle()
    model = build_policy_value_model(
        observation_dim=8,
        config=_structured_model_config(),
        action_dim=5,
        observation_spec=_structured_hand_observation_spec(),
        spec_bundle=spec_bundle,
    )

    obs_a = torch.zeros((1, 8), dtype=torch.float32)
    obs_a[0, 4] = 11
    obs_a[0, 5] = 22
    obs_b = torch.zeros((1, 8), dtype=torch.float32)
    obs_b[0, 4] = 22
    obs_b[0, 5] = 11

    logits_a, _value_a, _hidden_a = model(obs_a)
    logits_b, _value_b, _hidden_b = model(obs_b)

    assert not torch.isclose(logits_a[0, 0], logits_b[0, 2], atol=1e-6, rtol=0.0)


def test_structured_v2_uses_target_slot_context_when_scoring_play_actions() -> None:
    spec_bundle = _structured_hand_spec_bundle()
    model = build_policy_value_model(
        observation_dim=8,
        config=_structured_model_config(),
        action_dim=5,
        observation_spec=_structured_hand_observation_spec(),
        spec_bundle=spec_bundle,
    )

    obs_open = torch.zeros((1, 8), dtype=torch.float32)
    obs_open[0, 4] = 11
    obs_blocked = obs_open.clone()
    obs_blocked[0, 2] = 99

    logits_open, _value_open, _hidden_open = model(obs_open)
    logits_blocked, _value_blocked, _hidden_blocked = model(obs_blocked)

    assert not torch.isclose(logits_open[0, 0], logits_blocked[0, 0], atol=1e-6, rtol=1e-6)


def test_negative_logits_fill_value_is_safe_for_float16_masks() -> None:
    fill_value = _negative_logits_fill_value(torch.float16)

    assert np.isfinite(fill_value)
    assert fill_value <= -60000.0
