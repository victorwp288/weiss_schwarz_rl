from __future__ import annotations

from typing import Any, cast

import numpy as np

from weiss_rl.config.models import ModelConfig, ModelDropoutConfig
from weiss_rl.core.action_catalog import ActionCatalog


def structured_profile_model_config() -> ModelConfig:
    return ModelConfig(
        gru_hidden_size=256,
        encoder_mlp_width=128,
        encoder_mlp_layers=2,
        layer_norm=True,
        dropout=ModelDropoutConfig(family_a=0.0, ablation=0.1),
        encoder_kind="structured_v2",
        typed_feature_width=64,
    )


def typed_profile_observation_spec() -> dict[str, object]:
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


def structured_profile_spec_bundle() -> dict[str, object]:
    observation = typed_profile_observation_spec()
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
        "compatibility_hash": "structured_v2_profile_hash",
    }


def heuristic_profile_spec_bundle() -> dict[str, object]:
    return {
        "policy_version": 2,
        "spec_hash": 123,
        "observation": {
            "obs_encoding_version": 2,
            "obs_len": 512,
            "dtype": "i32",
            "self_first": True,
            "header_fields": [
                {"name": "active_player", "index": 0},
                {"name": "phase", "index": 1},
                {"name": "decision_kind", "index": 2},
                {"name": "decision_player", "index": 3},
                {"name": "terminal", "index": 4},
                {"name": "last_action_kind", "index": 5},
                {"name": "last_action_arg0", "index": 6},
                {"name": "last_action_arg1", "index": 7},
                {"name": "attack_slot", "index": 8},
                {"name": "defender_slot", "index": 9},
                {"name": "attack_type", "index": 10},
                {"name": "attack_damage", "index": 11},
                {"name": "attack_counter_power", "index": 12},
                {"name": "focus_slot", "index": 13},
                {"name": "choice_page_start", "index": 14},
                {"name": "choice_total", "index": 15},
            ],
            "player_blocks": [
                {
                    "player_index": 0,
                    "base": 16,
                    "len": 42,
                    "slices": [
                        {"name": "level_count", "start": 0, "len": 1, "visibility": "public"},
                        {"name": "clock_count", "start": 1, "len": 1, "visibility": "public"},
                        {"name": "hand_count", "start": 2, "len": 1, "visibility": "private"},
                        {"name": "stage", "start": 3, "len": 35, "visibility": "public"},
                        {"name": "hand", "start": 38, "len": 4, "visibility": "private"},
                    ],
                },
                {
                    "player_index": 1,
                    "base": 58,
                    "len": 42,
                    "slices": [
                        {"name": "level_count", "start": 0, "len": 1, "visibility": "public"},
                        {"name": "clock_count", "start": 1, "len": 1, "visibility": "public"},
                        {"name": "hand_count", "start": 2, "len": 1, "visibility": "private"},
                        {"name": "stage", "start": 3, "len": 35, "visibility": "public"},
                        {"name": "hand", "start": 38, "len": 4, "visibility": "private"},
                    ],
                },
            ],
        },
        "action": {
            "action_encoding_version": 1,
            "action_space_size": 527,
            "pass_action_id": 51,
            "attack_type_encoding": [["frontal", 0], ["side", 1], ["direct", 2]],
            "constants": [["MAX_HAND", 50], ["MAX_STAGE", 5], ["ATTACK_SLOT_COUNT", 3]],
            "families": [
                {"name": "mulligan_confirm", "base": 0, "count": 1},
                {"name": "mulligan_select", "base": 1, "count": 50},
                {"name": "pass", "base": 51, "count": 1},
                {"name": "clock_from_hand", "base": 52, "count": 50},
                {"name": "main_play_character", "base": 102, "count": 250},
                {"name": "main_play_event", "base": 352, "count": 50},
                {"name": "main_move", "base": 402, "count": 20},
                {"name": "climax_play", "base": 422, "count": 50},
                {"name": "attack", "base": 472, "count": 9},
                {"name": "level_up", "base": 481, "count": 7},
                {"name": "encore_pay", "base": 488, "count": 5},
                {"name": "encore_decline", "base": 493, "count": 5},
                {"name": "trigger_order", "base": 498, "count": 10},
                {"name": "choice_select", "base": 508, "count": 16},
                {"name": "choice_prev_page", "base": 524, "count": 1},
                {"name": "choice_next_page", "base": 525, "count": 1},
                {"name": "concede", "base": 526, "count": 1},
            ],
        },
    }


def empty_profile_observation() -> np.ndarray:
    observation = cast(dict[str, Any], heuristic_profile_spec_bundle()["observation"])
    return np.zeros((int(observation["obs_len"]),), dtype=np.int32)


def set_profile_stage(
    obs: np.ndarray,
    *,
    player_index: int,
    slot: int,
    occupied: bool,
    attacked: bool = False,
    power: int = 0,
    effective_soul: int = 0,
    side_attack_allowed: bool = True,
) -> None:
    player_base = 16 if player_index == 0 else 58
    stage_base = player_base + 3 + slot * 7
    obs[stage_base] = 100 + slot if occupied else 0
    obs[stage_base + 2] = int(attacked)
    obs[stage_base + 3] = int(power)
    obs[stage_base + 5] = int(effective_soul)
    obs[stage_base + 6] = int(side_attack_allowed)


def packed_profile_meta(action_ids: np.ndarray) -> np.ndarray:
    catalog = ActionCatalog.from_spec_bundle(heuristic_profile_spec_bundle())
    family_index = {family.name: index for index, family in enumerate(catalog.families)}
    attack_type_index = {name: index for index, name in enumerate(catalog.attack_type_names)}
    unused = np.iinfo(np.uint16).max
    rows = np.full((int(action_ids.shape[0]), 4), unused, dtype=np.uint16)
    for row_index, action_id in enumerate(np.asarray(action_ids, dtype=np.int64).tolist()):
        decoded = catalog.decode(int(action_id))
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
