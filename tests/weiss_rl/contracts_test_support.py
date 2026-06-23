from __future__ import annotations

from typing import cast

import numpy as np
import torch
from weiss_rl.config.models import ModelConfig, ModelDropoutConfig
from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.model import StructuredLegalPolicyValueModel


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


def _make_structured_joint_scorer_nonuniform(model: StructuredLegalPolicyValueModel) -> None:
    final_scorer = cast(torch.nn.Linear, model.policy_head.joint_scorer[-1])
    with torch.no_grad():
        torch.nn.init.normal_(final_scorer.weight, mean=0.0, std=0.1)
        torch.nn.init.zeros_(final_scorer.bias)


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
