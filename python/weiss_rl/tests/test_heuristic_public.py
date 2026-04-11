from __future__ import annotations

from pathlib import Path

import numpy as np

from weiss_rl.config import load_stack_config
from weiss_rl.eval.heuristic_public import HeuristicPublicPolicy
from weiss_rl.eval.policy_set import HEURISTIC_PUBLIC_POLICY_ID, RANDOM_LEGAL_POLICY_ID
from weiss_rl.eval.simulator_runner import resolve_eval_policies


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _heuristic_spec_bundle() -> dict[str, object]:
    return {
        "policy_version": 2,
        "spec_hash": 123,
        "observation": {
            "obs_encoding_version": 2,
            "obs_len": 100,
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


def _empty_obs() -> np.ndarray:
    return np.zeros((_heuristic_spec_bundle()["observation"]["obs_len"],), dtype=np.int32)  # type: ignore[index]


def _set_stage(
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


def test_heuristic_public_prefers_center_front_character_play() -> None:
    policy = HeuristicPublicPolicy.from_spec_bundle(_heuristic_spec_bundle())
    obs = _empty_obs()

    legal_ids = np.array([102, 103, 104, 105, 106], dtype=np.uint32)

    assert policy.choose_action(obs, legal_ids) == 103


def test_heuristic_public_prefers_direct_attack_when_available() -> None:
    policy = HeuristicPublicPolicy.from_spec_bundle(_heuristic_spec_bundle())
    obs = _empty_obs()
    _set_stage(obs, player_index=0, slot=0, occupied=True, power=5000, effective_soul=1)

    legal_ids = np.array([472, 473, 474], dtype=np.uint32)

    assert policy.choose_action(obs, legal_ids) == 474


def test_heuristic_public_ignores_private_observation_changes() -> None:
    policy = HeuristicPublicPolicy.from_spec_bundle(_heuristic_spec_bundle())
    obs_a = _empty_obs()
    obs_b = _empty_obs()
    obs_a[18] = 1
    obs_b[18] = 4
    obs_a[54:58] = np.array([1, 2, 3, 4], dtype=np.int32)
    obs_b[54:58] = np.array([9, 8, 7, 6], dtype=np.int32)

    legal_ids = np.array([52, 53, 54], dtype=np.uint32)

    assert policy.choose_action(obs_a, legal_ids) == policy.choose_action(obs_b, legal_ids) == 52


def test_heuristic_public_prefers_pass_over_main_move_loops() -> None:
    policy = HeuristicPublicPolicy.from_spec_bundle(_heuristic_spec_bundle())
    obs = _empty_obs()

    legal_ids = np.array([51, 402], dtype=np.uint32)

    assert policy.choose_action(obs, legal_ids) == 51


def test_resolve_eval_policies_supports_b2_without_snapshot_weights(tmp_path: Path) -> None:
    stack = load_stack_config(_repo_root() / "configs/rl_stack_locked.yaml")

    resolved = resolve_eval_policies(
        stack=stack,
        policy_ids=[RANDOM_LEGAL_POLICY_ID, HEURISTIC_PUBLIC_POLICY_ID],
        run_dir=tmp_path,
        observation_dim=100,
        action_dim=527,
        spec_bundle=_heuristic_spec_bundle(),
    )

    assert resolved[RANDOM_LEGAL_POLICY_ID].kind == "random_legal"
    assert resolved[HEURISTIC_PUBLIC_POLICY_ID].kind == "heuristic_public"
    assert resolved[HEURISTIC_PUBLIC_POLICY_ID].heuristic_policy is not None
