from __future__ import annotations

import pytest
import torch
import weiss_rl.model as model_facade
from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.model import GLOBAL_ACTION_SPACE_SIZE, SEAT_COUNT, PolicyValueModel
from weiss_rl.models.architecture_map import (
    MODEL_ARCHITECTURE_COMPONENTS,
    model_architecture_component_payload,
)
from weiss_rl.models.backbone.trunk_contract import structured_trunk_output_contract_payload
from weiss_rl.models.heads.structured_head import _StructuredLegalActionHead
from weiss_rl.models.heads.structured_head_blueprint import build_structured_head_blueprint
from weiss_rl.models.heads.structured_head_scoring_surfaces import structured_head_scoring_surface_payload
from weiss_rl.models.policy.policy_value_factory import (
    build_policy_value_model as owner_build_policy_value_model,
)
from weiss_rl.models.policy.policy_value_factory import (
    policy_value_factory_route_payload,
)
from weiss_rl.models.policy.policy_value_model import PolicyValueModel as OwnerPolicyValueModel
from weiss_rl.models.policy.structured_policy_value_model import (
    StructuredLegalPolicyValueModel as OwnerStructuredLegalPolicyValueModel,
)

from tests.weiss_rl.contracts_test_support import (
    _feedforward_model_config,
    _model_config,
    _structured_spec_bundle,
    _typed_model_config,
    _typed_observation_spec,
)


def test_model_facade_preserves_public_imports() -> None:
    assert model_facade.PolicyValueModel is OwnerPolicyValueModel
    assert model_facade.StructuredLegalPolicyValueModel is OwnerStructuredLegalPolicyValueModel
    assert model_facade.build_policy_value_model is owner_build_policy_value_model
    assert model_facade.PolicyValueModel.__module__ == "weiss_rl.model"
    assert model_facade.StructuredLegalPolicyValueModel.__module__ == "weiss_rl.model"


def test_model_architecture_map_names_component_owners_and_evidence() -> None:
    assert [component.key for component in MODEL_ARCHITECTURE_COMPONENTS] == [
        "factory",
        "dense_trunk",
        "opponent_context",
        "structured_head",
        "public_heuristic_bias",
        "diagnostics",
    ]
    payload = model_architecture_component_payload()
    assert payload[0] == {
        "key": "factory",
        "role": "Chooses dense fallback versus structured legal-action model construction.",
        "owner_modules": ["weiss_rl.models.policy.policy_value_factory"],
        "evidence": ["model config", "simulator spec bundle", "action catalog"],
    }
    assert "weiss_rl.models.heads.structured_head_build_plan" in payload[3]["owner_modules"]


def test_policy_value_factory_routes_name_model_selection_inputs() -> None:
    payload = policy_value_factory_route_payload()

    assert [route["route_id"] for route in payload] == ["structured_v2", "dense_fallback"]
    assert payload[0]["model_class"] == "StructuredLegalPolicyValueModel"
    assert "spec_bundle" in payload[0]["required_inputs"]
    assert payload[1]["model_class"] == "PolicyValueModel"
    assert payload[1]["condition"] == "all other encoder kinds"


def test_structured_head_blueprint_resolves_catalog_tables_and_dimensions() -> None:
    action_catalog = ActionCatalog.from_spec_bundle(_structured_spec_bundle())

    blueprint = build_structured_head_blueprint(
        action_catalog=action_catalog,
        action_dim=action_catalog.action_space_size,
        action_feature_width=64,
        public_heuristic_logit_bias_families=("pass",),
    )

    assert blueprint.family_count == 4
    assert blueprint.catalog_view.family_index["pass"] == 3
    assert blueprint.action_tables.family_ids.tolist() == [0, 0, 0, 0, 1, 1, 2, 2, 3]
    assert blueprint.factorized_tables.family_noarg_action_ids[3] == action_catalog.pass_action_id
    assert blueprint.offsets.numeric < blueprint.dimensions.candidate_input_dim


def test_structured_head_build_plan_names_install_order() -> None:
    assert [step.step_id for step in _StructuredLegalActionHead.build_plan] == [
        "validate_inputs",
        "resolve_blueprint",
        "install_catalog_view",
        "install_representation_modules",
        "install_action_tables",
        "install_factorized_modules",
        "install_public_heuristic_buffers",
        "set_runtime_chunking",
    ]
    assert _StructuredLegalActionHead.build_plan[1].title == "Resolve catalog blueprint"


def test_structured_head_scoring_surfaces_name_runtime_and_diagnostic_paths() -> None:
    payload = structured_head_scoring_surface_payload()

    assert [surface["name"] for surface in payload] == [
        "dense_legal_logits",
        "packed_candidate_logits",
        "factorized_policy",
        "public_heuristic_bias",
    ]
    assert _StructuredLegalActionHead.scoring_surfaces[1].entrypoints == (
        "score_packed_candidates",
        "forward_packed_seat_aware",
        "sample_packed_seat_aware",
    )
    assert "weiss_rl.models.heads.structured_head_scoring_surfaces" in (MODEL_ARCHITECTURE_COMPONENTS[3].owner_modules)


def test_structured_trunk_output_contract_names_tuple_positions() -> None:
    payload = structured_trunk_output_contract_payload()

    assert [field["name"] for field in payload] == [
        "recurrent_output",
        "state_repr",
        "observation_context",
        "value",
        "next_seat_hidden",
    ]
    assert OwnerStructuredLegalPolicyValueModel.trunk_output_contract[2].consumer == (
        "candidate features and public-heuristic bias"
    )


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
