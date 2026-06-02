from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.testing as npt
import pytest
import torch

from weiss_rl.config import StackConfig, canonical_config_dict, load_stack_config
from weiss_rl.config.models import ModelConfig, ModelDropoutConfig
from weiss_rl.core.action_catalog import ActionCatalog as SharedActionCatalog
from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.envs.decision_env import DecisionBoundaryEnv
from weiss_rl.eval import policy_resolution_context as policy_resolution_context_module
from weiss_rl.eval.b1_policy_resolution import resolve_b1_policy as b1_policy_resolution_resolve_b1_policy
from weiss_rl.eval.heuristic_public import HeuristicPublicPolicy
from weiss_rl.eval.policies.resolution import (
    _is_recursive_registry_search_root as policy_resolution_is_recursive_registry_search_root,
)
from weiss_rl.eval.policies.resolution import _resolve_b1_policy as policy_resolution_resolve_b1_policy
from weiss_rl.eval.policies.resolution import (
    _resolve_snapshot_registry_policy as policy_resolution_resolve_snapshot_registry_policy,
)
from weiss_rl.eval.policies.resolution import (
    _resolve_static_eval_policy as policy_resolution_resolve_static_eval_policy,
)
from weiss_rl.eval.policies.resolution import (
    _should_include_common_search_root as policy_resolution_should_include_common_search_root,
)
from weiss_rl.eval.policies.resolution import resolve_eval_policies as policy_resolution_resolve_eval_policies
from weiss_rl.eval.policies.resolution_context import EvalPolicyResolutionContext
from weiss_rl.eval.policies.set import (
    HEURISTIC_PUBLIC_AGGRO_POLICY_ID,
    HEURISTIC_PUBLIC_CONTROL_POLICY_ID,
    HEURISTIC_PUBLIC_POLICY_ID,
    NO_LEAGUE_POLICY_ID,
    RANDOM_LEGAL_POLICY_ID,
)
from weiss_rl.eval.simulator_runner import (
    _is_recursive_registry_search_root,
    _resolve_b1_policy,
    _resolve_snapshot_registry_policy,
    _resolve_static_eval_policy,
    _should_include_common_search_root,
    resolve_eval_policies,
)
from weiss_rl.eval.snapshot_policy_resolution import (
    resolve_snapshot_registry_policy as snapshot_policy_resolution_resolve_snapshot_registry_policy,
)
from weiss_rl.eval.snapshot_registry_resolution import (
    is_recursive_registry_search_root as snapshot_registry_is_recursive_registry_search_root,
)
from weiss_rl.eval.snapshot_registry_resolution import (
    should_include_common_search_root as snapshot_registry_should_include_common_search_root,
)
from weiss_rl.eval.static_policy_resolution import (
    resolve_static_eval_policy as static_policy_resolve_static_eval_policy,
)
from weiss_rl.league.registry import SnapshotRegistry
from weiss_rl.model import (
    PolicyValueModel,
    StructuredLegalPolicyValueModel,
    build_policy_value_model,
)
from weiss_rl.tests._config_paths import canonical_stack_config_path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_simulator_runner_reexports_policy_resolution_helpers() -> None:
    assert resolve_eval_policies is policy_resolution_resolve_eval_policies
    assert _is_recursive_registry_search_root is policy_resolution_is_recursive_registry_search_root
    assert _should_include_common_search_root is policy_resolution_should_include_common_search_root
    assert _resolve_b1_policy is policy_resolution_resolve_b1_policy
    assert _resolve_snapshot_registry_policy is policy_resolution_resolve_snapshot_registry_policy
    assert _resolve_static_eval_policy is policy_resolution_resolve_static_eval_policy
    assert policy_resolution_is_recursive_registry_search_root is snapshot_registry_is_recursive_registry_search_root
    assert policy_resolution_should_include_common_search_root is snapshot_registry_should_include_common_search_root
    assert policy_resolution_resolve_b1_policy is b1_policy_resolution_resolve_b1_policy
    assert (
        policy_resolution_resolve_snapshot_registry_policy
        is snapshot_policy_resolution_resolve_snapshot_registry_policy
    )
    assert policy_resolution_resolve_static_eval_policy is static_policy_resolve_static_eval_policy
    assert policy_resolution_resolve_eval_policies.__module__ == "weiss_rl.eval.policies.resolution"
    assert policy_resolution_resolve_b1_policy.__module__ == "weiss_rl.eval.b1_policy_resolution"
    assert policy_resolution_resolve_snapshot_registry_policy.__module__ == "weiss_rl.eval.snapshot_policy_resolution"
    assert policy_resolution_resolve_static_eval_policy.__module__ == "weiss_rl.eval.static_policy_resolution"
    assert (
        policy_resolution_is_recursive_registry_search_root.__module__ == "weiss_rl.eval.snapshot_registry_resolution"
    )


def test_eval_policy_resolution_context_owns_registry_policy_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_run_dir = tmp_path / "runs" / "source_run"
    registry_path = registry_run_dir / "training" / "snapshots" / "registry.json"
    weights_path = registry_run_dir / "training" / "snapshots" / "policy_000100" / "weights.pt"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.write_bytes(b"weights")
    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="policy_000100",
        update=100,
        weights_sha256=hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        path="training/snapshots/policy_000100/weights.pt",
    )
    registry.save(registry_path)
    consumer_run_dir = tmp_path / "runs" / "consumer"
    consumer_run_dir.mkdir(parents=True)
    context = EvalPolicyResolutionContext.load(
        run_dir=consumer_run_dir,
        policy_ids=["policy_000100"],
        snapshot_registry_path=registry_path,
    )
    observed: dict[str, object] = {}

    def _fake_resolve_snapshot_registry_policy(**kwargs: object):
        observed.update(kwargs)
        return kwargs["policy_id"]

    monkeypatch.setattr(
        policy_resolution_context_module,
        "resolve_snapshot_registry_policy",
        _fake_resolve_snapshot_registry_policy,
    )

    resolved = context.resolve_registry_policy(
        policy_id="policy_000100",
        stack=cast(StackConfig, object()),
        observation_dim=512,
        action_dim=9,
        spec_bundle={"observation": {}},
    )

    assert resolved == "policy_000100"
    assert observed["registry_source"] is context.registry_source
    assert context.registry_path == registry_path
    assert context.registry is context.registry_source.registry
    assert context.snapshots_by_policy_id is context.registry_source.snapshots_by_policy_id
    assert observed["snapshot_run_dir"] == registry_run_dir.resolve()
    assert observed["observation_dim"] == 512
    assert observed["action_dim"] == 9
    assert context.registry_run_dir_if_resolved == registry_run_dir.resolve()


def _heuristic_spec_bundle() -> dict[str, object]:
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


def _empty_obs() -> np.ndarray:
    return np.zeros((_heuristic_spec_bundle()["observation"]["obs_len"],), dtype=np.int32)  # type: ignore[index]


_LEGAL_DECK = (list(range(1, 14)) * 4)[:50]


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


def _packed_meta(action_ids: np.ndarray) -> np.ndarray:
    catalog = SharedActionCatalog.from_spec_bundle(_heuristic_spec_bundle())
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


def test_heuristic_public_profiles_choose_different_attack_lines() -> None:
    aggressive = HeuristicPublicPolicy.from_spec_bundle(_heuristic_spec_bundle(), scoring_profile="aggressive")
    control = HeuristicPublicPolicy.from_spec_bundle(_heuristic_spec_bundle(), scoring_profile="control")
    obs = _empty_obs()
    _set_stage(obs, player_index=0, slot=0, occupied=True, power=6000, effective_soul=3)
    _set_stage(obs, player_index=1, slot=0, occupied=True, power=2000)

    legal_ids = np.array([472, 474], dtype=np.uint32)

    assert aggressive.choose_action(obs, legal_ids) == 474
    assert control.choose_action(obs, legal_ids) == 472


def test_heuristic_public_meta_fast_path_matches_scalar_policy() -> None:
    policy = HeuristicPublicPolicy.from_spec_bundle(_heuristic_spec_bundle())
    obs = _empty_obs()
    _set_stage(obs, player_index=0, slot=0, occupied=True, power=5000, effective_soul=1)

    legal_ids = np.array([472, 473, 474, 51, 402], dtype=np.uint32)

    assert policy.choose_action_from_meta(obs, legal_ids, _packed_meta(legal_ids)) == policy.choose_action(
        obs, legal_ids
    )


def test_heuristic_public_batch_meta_fast_path_matches_scalar_policy() -> None:
    policy = HeuristicPublicPolicy.from_spec_bundle(_heuristic_spec_bundle())
    obs_attack = _empty_obs()
    _set_stage(obs_attack, player_index=0, slot=0, occupied=True, power=5000, effective_soul=1)
    obs_play = _empty_obs()
    _set_stage(obs_play, player_index=0, slot=1, occupied=True, power=2500)
    obs_clock = _empty_obs()
    obs_clock[16] = 0
    obs_clock[17] = 3
    obs_clock[14] = 16
    obs_clock[15] = 40

    row_legal_ids = [
        np.array([472, 473, 474, 51, 402], dtype=np.uint32),
        np.array([102, 103, 104, 105, 106], dtype=np.uint32),
        np.array([52, 53, 524, 525, 51], dtype=np.uint32),
    ]
    obs_rows = np.stack([obs_attack, obs_play, obs_clock], axis=0)
    legal_ids = np.concatenate(row_legal_ids, axis=0)
    offsets = np.asarray(
        [0, row_legal_ids[0].size, row_legal_ids[0].size + row_legal_ids[1].size, legal_ids.size], dtype=np.uint32
    )
    meta = _packed_meta(legal_ids)

    batch_actions = policy.choose_actions_from_meta_batch(obs_rows, legal_ids, offsets, meta)
    scalar_actions = np.asarray(
        [
            policy.choose_action_from_meta(
                obs_rows[row_index], row_legal_ids[row_index], _packed_meta(row_legal_ids[row_index])
            )
            for row_index in range(len(row_legal_ids))
        ],
        dtype=np.int64,
    )

    npt.assert_array_equal(batch_actions, scalar_actions)


def test_structured_model_public_heuristic_scores_match_b2_batch_meta_choices() -> None:
    spec_bundle = _heuristic_spec_bundle()
    model = build_policy_value_model(
        observation_dim=int(spec_bundle["observation"]["obs_len"]),  # type: ignore[index]
        config=ModelConfig(
            gru_hidden_size=64,
            encoder_mlp_width=64,
            encoder_mlp_layers=1,
            layer_norm=True,
            dropout=ModelDropoutConfig(family_a=0.0, ablation=0.0),
            encoder_kind="structured_v2",
            typed_feature_width=32,
        ),
        action_dim=int(spec_bundle["action"]["action_space_size"]),  # type: ignore[index]
        observation_spec=cast(dict[str, Any], spec_bundle["observation"]),
        spec_bundle=spec_bundle,
    )
    assert isinstance(model, StructuredLegalPolicyValueModel)

    policy = HeuristicPublicPolicy.from_spec_bundle(spec_bundle)
    obs_attack = _empty_obs()
    _set_stage(obs_attack, player_index=0, slot=0, occupied=True, power=5000, effective_soul=1)
    obs_play = _empty_obs()
    _set_stage(obs_play, player_index=0, slot=1, occupied=True, power=2500)
    obs_clock = _empty_obs()
    obs_clock[16] = 0
    obs_clock[17] = 3
    obs_clock[14] = 16
    obs_clock[15] = 40

    row_legal_ids = [
        np.array([472, 473, 474, 51, 402], dtype=np.uint32),
        np.array([102, 103, 104, 105, 106], dtype=np.uint32),
        np.array([52, 53, 524, 525, 51], dtype=np.uint32),
    ]
    obs_rows = np.stack([obs_attack, obs_play, obs_clock], axis=0)
    legal_ids = np.concatenate(row_legal_ids, axis=0)
    offsets = np.asarray(
        [0, row_legal_ids[0].size, row_legal_ids[0].size + row_legal_ids[1].size, legal_ids.size], dtype=np.uint32
    )
    meta = _packed_meta(legal_ids)

    scores = (
        model.score_packed_public_heuristic_candidates(
            torch.as_tensor(obs_rows, dtype=torch.float32),
            LegalActionBatch.from_packed(
                legal_ids,
                offsets,
                meta=meta,
                action_space=int(spec_bundle["action"]["action_space_size"]),  # type: ignore[index]
            ),
        )
        .detach()
        .cpu()
        .numpy()
    )
    chosen = np.full((obs_rows.shape[0],), policy.pass_action_id, dtype=np.int64)
    for row_index in range(obs_rows.shape[0]):
        start = int(offsets[row_index])
        stop = int(offsets[row_index + 1])
        row_scores = scores[start:stop]
        row_best = np.flatnonzero(row_scores == row_scores.max())
        chosen[row_index] = int(legal_ids[start:stop][int(row_best[0])])

    expected = policy.choose_actions_from_meta_batch(obs_rows, legal_ids, offsets, meta)
    npt.assert_array_equal(chosen, expected)


def test_structured_model_public_bias_guides_live_packed_scores_toward_b2_choices() -> None:
    spec_bundle = _heuristic_spec_bundle()
    torch.manual_seed(0)
    model = build_policy_value_model(
        observation_dim=int(spec_bundle["observation"]["obs_len"]),  # type: ignore[index]
        config=ModelConfig(
            gru_hidden_size=64,
            encoder_mlp_width=64,
            encoder_mlp_layers=1,
            layer_norm=True,
            dropout=ModelDropoutConfig(family_a=0.0, ablation=0.0),
            encoder_kind="structured_v2",
            typed_feature_width=32,
            public_heuristic_logit_bias_scale=0.0,
            public_heuristic_actor_logit_bias_scale=100.0,
        ),
        action_dim=int(spec_bundle["action"]["action_space_size"]),  # type: ignore[index]
        observation_spec=cast(dict[str, Any], spec_bundle["observation"]),
        spec_bundle=spec_bundle,
    )
    assert isinstance(model, StructuredLegalPolicyValueModel)

    policy = HeuristicPublicPolicy.from_spec_bundle(spec_bundle)
    obs_attack = _empty_obs()
    _set_stage(obs_attack, player_index=0, slot=0, occupied=True, power=5000, effective_soul=1)
    obs_play = _empty_obs()
    _set_stage(obs_play, player_index=0, slot=1, occupied=True, power=2500)
    obs_clock = _empty_obs()
    obs_clock[16] = 0
    obs_clock[17] = 3
    obs_clock[14] = 16
    obs_clock[15] = 40

    row_legal_ids = [
        np.array([472, 473, 474, 51, 402], dtype=np.uint32),
        np.array([102, 103, 104, 105, 106], dtype=np.uint32),
        np.array([52, 53, 524, 525, 51], dtype=np.uint32),
    ]
    obs_rows = np.stack([obs_attack, obs_play, obs_clock], axis=0)
    legal_ids = np.concatenate(row_legal_ids, axis=0)
    offsets = np.asarray(
        [0, row_legal_ids[0].size, row_legal_ids[0].size + row_legal_ids[1].size, legal_ids.size], dtype=np.uint32
    )
    meta = _packed_meta(legal_ids)
    legal_batch = LegalActionBatch.from_packed(
        legal_ids,
        offsets,
        meta=meta,
        action_space=int(spec_bundle["action"]["action_space_size"]),  # type: ignore[index]
    )
    obs_tensor = torch.as_tensor(obs_rows, dtype=torch.float32)

    with torch.no_grad():
        encoded = model.encode(obs_tensor)
        recurrent_output, _ = model.recurrent_step_seat_aware(encoded, 0, None)
        scores = (
            model.score_packed_legal_candidates(
                recurrent_output,
                obs_tensor,
                legal_batch,
                scoring_mode="actor",
            )
            .detach()
            .cpu()
            .numpy()
        )

    chosen = np.full((obs_rows.shape[0],), policy.pass_action_id, dtype=np.int64)
    for row_index in range(obs_rows.shape[0]):
        start = int(offsets[row_index])
        stop = int(offsets[row_index + 1])
        row_scores = scores[start:stop]
        row_best = np.flatnonzero(row_scores == row_scores.max())
        chosen[row_index] = int(legal_ids[start:stop][int(row_best[0])])

    expected = policy.choose_actions_from_meta_batch(obs_rows, legal_ids, offsets, meta)
    npt.assert_array_equal(chosen, expected)


def test_structured_model_public_bias_family_gate_only_affects_selected_families() -> None:
    spec_bundle = _heuristic_spec_bundle()
    torch.manual_seed(0)
    baseline_model = build_policy_value_model(
        observation_dim=int(spec_bundle["observation"]["obs_len"]),  # type: ignore[index]
        config=ModelConfig(
            gru_hidden_size=64,
            encoder_mlp_width=64,
            encoder_mlp_layers=1,
            layer_norm=True,
            dropout=ModelDropoutConfig(family_a=0.0, ablation=0.0),
            encoder_kind="structured_v2",
            typed_feature_width=32,
            public_heuristic_logit_bias_scale=0.0,
            public_heuristic_actor_logit_bias_scale=0.0,
        ),
        action_dim=int(spec_bundle["action"]["action_space_size"]),  # type: ignore[index]
        observation_spec=cast(dict[str, Any], spec_bundle["observation"]),
        spec_bundle=spec_bundle,
    )
    torch.manual_seed(0)
    gated_model = build_policy_value_model(
        observation_dim=int(spec_bundle["observation"]["obs_len"]),  # type: ignore[index]
        config=ModelConfig(
            gru_hidden_size=64,
            encoder_mlp_width=64,
            encoder_mlp_layers=1,
            layer_norm=True,
            dropout=ModelDropoutConfig(family_a=0.0, ablation=0.0),
            encoder_kind="structured_v2",
            typed_feature_width=32,
            public_heuristic_logit_bias_scale=0.0,
            public_heuristic_actor_logit_bias_scale=100.0,
            public_heuristic_logit_bias_families=("attack",),
        ),
        action_dim=int(spec_bundle["action"]["action_space_size"]),  # type: ignore[index]
        observation_spec=cast(dict[str, Any], spec_bundle["observation"]),
        spec_bundle=spec_bundle,
    )
    assert isinstance(baseline_model, StructuredLegalPolicyValueModel)
    assert isinstance(gated_model, StructuredLegalPolicyValueModel)

    obs_attack = _empty_obs()
    _set_stage(obs_attack, player_index=0, slot=0, occupied=True, power=5000, effective_soul=1)
    obs_play = _empty_obs()
    _set_stage(obs_play, player_index=0, slot=1, occupied=True, power=2500)
    obs_clock = _empty_obs()
    obs_clock[16] = 0
    obs_clock[17] = 3
    obs_clock[14] = 16
    obs_clock[15] = 40

    row_legal_ids = [
        np.array([472, 473, 474, 51, 402], dtype=np.uint32),
        np.array([102, 103, 104, 105, 106], dtype=np.uint32),
        np.array([52, 53, 524, 525, 51], dtype=np.uint32),
    ]
    obs_rows = np.stack([obs_attack, obs_play, obs_clock], axis=0)
    legal_ids = np.concatenate(row_legal_ids, axis=0)
    offsets = np.asarray(
        [0, row_legal_ids[0].size, row_legal_ids[0].size + row_legal_ids[1].size, legal_ids.size], dtype=np.uint32
    )
    meta = _packed_meta(legal_ids)
    legal_batch = LegalActionBatch.from_packed(
        legal_ids,
        offsets,
        meta=meta,
        action_space=int(spec_bundle["action"]["action_space_size"]),  # type: ignore[index]
    )
    obs_tensor = torch.as_tensor(obs_rows, dtype=torch.float32)

    def _scores(model: StructuredLegalPolicyValueModel) -> np.ndarray:
        with torch.no_grad():
            encoded = model.encode(obs_tensor)
            recurrent_output, _ = model.recurrent_step_seat_aware(encoded, 0, None)
            return (
                model.score_packed_legal_candidates(
                    recurrent_output,
                    obs_tensor,
                    legal_batch,
                    scoring_mode="actor",
                )
                .detach()
                .cpu()
                .numpy()
            )

    baseline_scores = _scores(baseline_model)
    gated_scores = _scores(gated_model)

    npt.assert_allclose(
        gated_scores[int(offsets[1]) : int(offsets[2])],
        baseline_scores[int(offsets[1]) : int(offsets[2])],
    )
    npt.assert_allclose(
        gated_scores[int(offsets[2]) : int(offsets[3])],
        baseline_scores[int(offsets[2]) : int(offsets[3])],
    )
    assert not np.allclose(
        gated_scores[int(offsets[0]) : int(offsets[1])],
        baseline_scores[int(offsets[0]) : int(offsets[1])],
    )


def test_heuristic_public_batch_meta_fast_path_falls_back_on_invalid_meta() -> None:
    policy = HeuristicPublicPolicy.from_spec_bundle(_heuristic_spec_bundle())
    obs_a = _empty_obs()
    obs_b = _empty_obs()
    _set_stage(obs_a, player_index=0, slot=0, occupied=True, power=4000, effective_soul=1)
    legal_ids = np.array([472, 473, 474, 51], dtype=np.uint32)
    obs_rows = np.stack([obs_a, obs_b], axis=0)
    offsets = np.asarray([0, 4, 8], dtype=np.uint32)
    repeated_ids = np.concatenate([legal_ids, legal_ids], axis=0)
    malformed_meta = np.zeros((int(repeated_ids.shape[0]), 2), dtype=np.uint16)

    batch_actions = policy.choose_actions_from_meta_batch(obs_rows, repeated_ids, offsets, malformed_meta)

    assert batch_actions.tolist() == [
        policy.choose_action(obs_a, legal_ids),
        policy.choose_action(obs_b, legal_ids),
    ]


def test_simulator_native_heuristic_pool_matches_python_oracle_across_live_steps() -> None:
    weiss_sim = pytest.importorskip(
        "weiss_sim",
        reason="native heuristic pool parity test requires the optional simulator package",
    )

    env = DecisionBoundaryEnv.create(
        legality="ids_offsets",
        mode="train",
        num_envs=4,
        db_path=None,
        deck_lists=[_LEGAL_DECK, _LEGAL_DECK],
        deck_ids=[101, 102],
        max_decisions=200,
        max_ticks=10_000,
        seed=321,
    )
    try:
        batch = env.reset()
        for _ in range(24):
            assert batch.ids_offsets is not None
            assert batch.legal_action_meta is not None
            legal_ids, legal_offsets = batch.ids_offsets
            policy = HeuristicPublicPolicy.from_spec_bundle(weiss_sim.spec_bundle())
            native_actions = np.zeros((env.num_envs,), dtype=np.uint16)
            env.pool.choose_heuristic_public_actions_into(
                np.arange(env.num_envs, dtype=np.uint32),
                native_actions,
            )
            oracle_actions = policy.choose_actions_from_meta_batch(
                np.asarray(batch.obs, dtype=np.int32),
                np.asarray(legal_ids, dtype=np.uint32),
                np.asarray(legal_offsets, dtype=np.uint32),
                np.asarray(batch.legal_action_meta, dtype=np.uint16),
            )
            npt.assert_array_equal(native_actions.astype(np.int64), oracle_actions)
            batch = env.step(native_actions.astype(np.int64))
    finally:
        env.close()


def test_action_catalog_is_shared_with_legacy_heuristic_export() -> None:
    from weiss_rl.eval.heuristic_public import ActionCatalog as LegacyActionCatalog

    assert LegacyActionCatalog is SharedActionCatalog
    catalog = SharedActionCatalog.from_spec_bundle(_heuristic_spec_bundle())

    decoded = catalog.decode(474)

    assert decoded.family == "attack"
    assert decoded.attack_type == "direct"


def test_resolve_eval_policies_supports_b2_without_snapshot_weights(tmp_path: Path) -> None:
    stack = load_stack_config(canonical_stack_config_path())

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


def test_resolve_eval_policies_supports_heuristic_public_variants(tmp_path: Path) -> None:
    stack = load_stack_config(canonical_stack_config_path())

    resolved = resolve_eval_policies(
        stack=stack,
        policy_ids=[HEURISTIC_PUBLIC_AGGRO_POLICY_ID, HEURISTIC_PUBLIC_CONTROL_POLICY_ID],
        run_dir=tmp_path,
        observation_dim=100,
        action_dim=527,
        spec_bundle=_heuristic_spec_bundle(),
    )

    assert resolved[HEURISTIC_PUBLIC_AGGRO_POLICY_ID].kind == "heuristic_public"
    assert resolved[HEURISTIC_PUBLIC_AGGRO_POLICY_ID].heuristic_policy is not None
    assert resolved[HEURISTIC_PUBLIC_CONTROL_POLICY_ID].kind == "heuristic_public"
    assert resolved[HEURISTIC_PUBLIC_CONTROL_POLICY_ID].heuristic_policy is not None


def test_resolve_eval_policies_loads_snapshots_from_registry_run_root(tmp_path: Path) -> None:
    stack = load_stack_config(canonical_stack_config_path())
    assert stack.config.model is not None

    source_run_dir = tmp_path / "source_run"
    registry_path = source_run_dir / "training" / "snapshots" / "registry.json"
    weights_path = source_run_dir / "training" / "snapshots" / "policy_000100" / "weights.pt"
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    model = PolicyValueModel(
        observation_dim=512,
        config=stack.config.model,
        action_dim=9,
        observation_spec=_heuristic_spec_bundle()["observation"],  # type: ignore[arg-type]
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "structured_policy_contract": stack.config.model.structured_policy_contract,
        },
        weights_path,
    )

    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="policy_000100",
        update=100,
        weights_sha256="1" * 64,
        path="training/snapshots/policy_000100/weights.pt",
    )
    registry.save(registry_path)

    consumer_run_dir = tmp_path / "consumer_run"
    manifest_path = consumer_run_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"run_id256": "ab" * 32}), encoding="utf-8")

    resolved = resolve_eval_policies(
        stack=stack,
        policy_ids=["policy_000100"],
        run_dir=consumer_run_dir,
        observation_dim=512,
        action_dim=9,
        spec_bundle=_heuristic_spec_bundle(),
        snapshot_registry_path=registry_path,
    )

    assert resolved["policy_000100"].source_run_dir == source_run_dir.resolve().as_posix()
    assert resolved["policy_000100"].snapshot_path == "training/snapshots/policy_000100/weights.pt"
    assert resolved["policy_000100"].model is not None


def _write_eval_snapshot(
    *,
    stack: StackConfig,
    run_dir: Path,
    policy_id: str,
    update: int,
) -> Path:
    model_config = stack.config.model
    assert model_config is not None
    weights_path = run_dir / "training" / "snapshots" / policy_id / "weights.pt"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    model = PolicyValueModel(
        observation_dim=512,
        config=model_config,
        action_dim=9,
        observation_spec=_heuristic_spec_bundle()["observation"],  # type: ignore[arg-type]
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "policy_id": policy_id,
            "structured_policy_contract": model_config.structured_policy_contract,
            "update": int(update),
        },
        weights_path,
    )
    return weights_path


def _write_snapshot_registry(
    *,
    registry_path: Path,
    snapshots: list[tuple[str, int, Path]],
) -> None:
    registry = SnapshotRegistry()
    for policy_id, update, weights_path in snapshots:
        registry.add_snapshot(
            policy_id=policy_id,
            update=update,
            weights_sha256=hashlib.sha256(weights_path.read_bytes()).hexdigest(),
            path=f"training/snapshots/{policy_id}/weights.pt",
        )
    registry.save(registry_path)


def test_resolve_eval_policies_loads_snapshots_from_copied_registry_json(tmp_path: Path) -> None:
    stack = load_stack_config(canonical_stack_config_path())

    source_run_dir = tmp_path / "external_runs" / "source_run"
    registry_path = source_run_dir / "training" / "snapshots" / "registry.json"
    snapshot_weights = _write_eval_snapshot(
        stack=stack,
        run_dir=source_run_dir,
        policy_id="policy_000100",
        update=100,
    )
    _write_snapshot_registry(
        registry_path=registry_path,
        snapshots=[("policy_000100", 100, snapshot_weights)],
    )

    copied_registry_path = tmp_path / "cache" / "policy_set_snapshot_registry.json"
    copied_registry_path.parent.mkdir(parents=True, exist_ok=True)
    copied_registry_path.write_text(registry_path.read_text(encoding="utf-8"), encoding="utf-8")
    consumer_run_dir = tmp_path / "runs" / "consumer_run"
    (consumer_run_dir / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (consumer_run_dir / "manifest.json").write_text(json.dumps({"run_id256": "ab" * 32}), encoding="utf-8")

    resolved = resolve_eval_policies(
        stack=stack,
        policy_ids=["policy_000100"],
        run_dir=consumer_run_dir,
        observation_dim=512,
        action_dim=9,
        spec_bundle=_heuristic_spec_bundle(),
        snapshot_registry_path=copied_registry_path,
    )

    assert resolved["policy_000100"].source_run_dir == source_run_dir.resolve().as_posix()
    assert resolved["policy_000100"].snapshot_path == "training/snapshots/policy_000100/weights.pt"
    assert resolved["policy_000100"].model is not None


def test_resolve_eval_policies_accepts_unique_imported_seed_suffix(tmp_path: Path) -> None:
    stack = load_stack_config(canonical_stack_config_path())

    run_dir = tmp_path / "runs" / "seeded_consumer"
    registry_path = run_dir / "training" / "snapshots" / "registry.json"
    actual_policy_id = "seed_newrun_seed_oldrun_policy_000005"
    requested_policy_id = "seed_oldrun_policy_000005"
    snapshot_weights = _write_eval_snapshot(
        stack=stack,
        run_dir=run_dir,
        policy_id=actual_policy_id,
        update=0,
    )
    _write_snapshot_registry(
        registry_path=registry_path,
        snapshots=[(actual_policy_id, 0, snapshot_weights)],
    )

    resolved = resolve_eval_policies(
        stack=stack,
        policy_ids=[requested_policy_id],
        run_dir=run_dir,
        observation_dim=512,
        action_dim=9,
        spec_bundle=_heuristic_spec_bundle(),
        snapshot_registry_path=registry_path,
    )

    assert resolved[requested_policy_id].policy_id == requested_policy_id
    assert resolved[requested_policy_id].source_run_dir == run_dir.resolve().as_posix()
    assert resolved[requested_policy_id].snapshot_path == f"training/snapshots/{actual_policy_id}/weights.pt"
    assert resolved[requested_policy_id].model is not None


def test_resolve_eval_policies_prefers_explicit_run_dir_for_ambiguous_copied_registry(tmp_path: Path) -> None:
    stack = load_stack_config(canonical_stack_config_path())

    source_run_dir = tmp_path / "runs" / "source_run"
    registry_path = source_run_dir / "training" / "snapshots" / "registry.json"
    snapshot_weights = _write_eval_snapshot(
        stack=stack,
        run_dir=source_run_dir,
        policy_id="policy_000100",
        update=100,
    )
    _write_snapshot_registry(
        registry_path=registry_path,
        snapshots=[("policy_000100", 100, snapshot_weights)],
    )

    copied_run_dir = tmp_path / "runs" / "copied_run"
    copied_weights_path = copied_run_dir / "training" / "snapshots" / "policy_000100" / "weights.pt"
    copied_weights_path.parent.mkdir(parents=True, exist_ok=True)
    copied_weights_path.write_bytes(snapshot_weights.read_bytes())

    copied_registry_path = tmp_path / "cache" / "policy_set_snapshot_registry.json"
    copied_registry_path.parent.mkdir(parents=True, exist_ok=True)
    copied_registry_path.write_text(registry_path.read_text(encoding="utf-8"), encoding="utf-8")

    resolved = resolve_eval_policies(
        stack=stack,
        policy_ids=["policy_000100"],
        run_dir=copied_run_dir,
        observation_dim=512,
        action_dim=9,
        spec_bundle=_heuristic_spec_bundle(),
        snapshot_registry_path=copied_registry_path,
    )

    assert resolved["policy_000100"].source_run_dir == copied_run_dir.resolve().as_posix()
    assert resolved["policy_000100"].snapshot_path == "training/snapshots/policy_000100/weights.pt"
    assert resolved["policy_000100"].model is not None


def test_resolve_eval_policies_ignores_canonical_looking_copied_registry_without_weights(tmp_path: Path) -> None:
    stack = load_stack_config(canonical_stack_config_path())

    source_run_dir = tmp_path / "external_runs" / "source_run"
    registry_path = source_run_dir / "training" / "snapshots" / "registry.json"
    snapshot_weights = _write_eval_snapshot(
        stack=stack,
        run_dir=source_run_dir,
        policy_id="policy_000100",
        update=100,
    )
    _write_snapshot_registry(
        registry_path=registry_path,
        snapshots=[("policy_000100", 100, snapshot_weights)],
    )

    copied_registry_path = tmp_path / "cache" / "training" / "snapshots" / "registry.json"
    copied_registry_path.parent.mkdir(parents=True, exist_ok=True)
    copied_registry_path.write_text(registry_path.read_text(encoding="utf-8"), encoding="utf-8")
    consumer_run_dir = tmp_path / "runs" / "consumer_run_canonical_copy"
    (consumer_run_dir / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (consumer_run_dir / "manifest.json").write_text(json.dumps({"run_id256": "ab" * 32}), encoding="utf-8")

    resolved = resolve_eval_policies(
        stack=stack,
        policy_ids=["policy_000100"],
        run_dir=consumer_run_dir,
        observation_dim=512,
        action_dim=9,
        spec_bundle=_heuristic_spec_bundle(),
        snapshot_registry_path=copied_registry_path,
    )

    assert resolved["policy_000100"].source_run_dir == source_run_dir.resolve().as_posix()
    assert resolved["policy_000100"].snapshot_path == "training/snapshots/policy_000100/weights.pt"
    assert resolved["policy_000100"].model is not None


def test_resolve_eval_policies_requires_full_requested_snapshot_set_for_copied_registry(tmp_path: Path) -> None:
    stack = load_stack_config(canonical_stack_config_path())

    source_run_dir = tmp_path / "external_runs" / "source_run"
    registry_path = source_run_dir / "training" / "snapshots" / "registry.json"
    snapshot_specs = [
        ("policy_000100", 100),
        ("policy_000200", 200),
        ("policy_000300", 300),
        ("policy_000400", 400),
        ("policy_000500", 500),
    ]
    snapshots: list[tuple[str, int, Path]] = []
    for policy_id, update in snapshot_specs:
        snapshots.append(
            (
                policy_id,
                update,
                _write_eval_snapshot(
                    stack=stack,
                    run_dir=source_run_dir,
                    policy_id=policy_id,
                    update=update,
                ),
            )
        )
    _write_snapshot_registry(
        registry_path=registry_path,
        snapshots=snapshots,
    )

    copied_registry_path = tmp_path / "cache" / "policy_set_snapshot_registry.json"
    copied_registry_path.parent.mkdir(parents=True, exist_ok=True)
    copied_registry_path.write_text(registry_path.read_text(encoding="utf-8"), encoding="utf-8")
    for policy_id in ("policy_000100", "policy_000300", "policy_000500"):
        source_weights = next(
            weights_path for snapshot_id, _update, weights_path in snapshots if snapshot_id == policy_id
        )
        copied_weights = tmp_path / "cache" / "training" / "snapshots" / policy_id / "weights.pt"
        copied_weights.parent.mkdir(parents=True, exist_ok=True)
        copied_weights.write_bytes(source_weights.read_bytes())
    consumer_run_dir = tmp_path / "runs" / "consumer_run_full_request"
    (consumer_run_dir / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (consumer_run_dir / "manifest.json").write_text(json.dumps({"run_id256": "ab" * 32}), encoding="utf-8")

    resolved = resolve_eval_policies(
        stack=stack,
        policy_ids=[policy_id for policy_id, _update in snapshot_specs],
        run_dir=consumer_run_dir,
        observation_dim=512,
        action_dim=9,
        spec_bundle=_heuristic_spec_bundle(),
        snapshot_registry_path=copied_registry_path,
    )

    assert {resolved[policy_id].source_run_dir for policy_id, _update in snapshot_specs} == {
        source_run_dir.resolve().as_posix()
    }


def test_resolve_eval_policies_loads_b1_from_registry_source_run(tmp_path: Path) -> None:
    stack = load_stack_config(canonical_stack_config_path())

    source_run_dir = tmp_path / "external_runs" / "source_run"
    registry_path = source_run_dir / "training" / "snapshots" / "registry.json"
    policy_weights = _write_eval_snapshot(
        stack=stack,
        run_dir=source_run_dir,
        policy_id="policy_000100",
        update=100,
    )
    b1_weights = _write_eval_snapshot(
        stack=stack,
        run_dir=source_run_dir,
        policy_id="b1_noleague_baseline",
        update=5,
    )
    _write_snapshot_registry(
        registry_path=registry_path,
        snapshots=[
            ("b1_noleague_baseline", 5, b1_weights),
            ("policy_000100", 100, policy_weights),
        ],
    )

    copied_registry_path = tmp_path / "cache" / "policy_set_snapshot_registry.json"
    copied_registry_path.parent.mkdir(parents=True, exist_ok=True)
    copied_registry_path.write_text(registry_path.read_text(encoding="utf-8"), encoding="utf-8")
    consumer_run_dir = tmp_path / "runs" / "consumer_run"
    (consumer_run_dir / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (consumer_run_dir / "manifest.json").write_text(json.dumps({"run_id256": "ab" * 32}), encoding="utf-8")

    resolved = resolve_eval_policies(
        stack=stack,
        policy_ids=[NO_LEAGUE_POLICY_ID, "policy_000100"],
        run_dir=consumer_run_dir,
        observation_dim=512,
        action_dim=9,
        spec_bundle=_heuristic_spec_bundle(),
        snapshot_registry_path=copied_registry_path,
    )

    assert resolved[NO_LEAGUE_POLICY_ID].source_run_dir == source_run_dir.resolve().as_posix()
    assert resolved[NO_LEAGUE_POLICY_ID].snapshot_path == "training/snapshots/b1_noleague_baseline/weights.pt"
    assert resolved[NO_LEAGUE_POLICY_ID].model is not None


def test_resolve_eval_policies_preserves_b1_display_id_preference_when_both_aliases_exist(tmp_path: Path) -> None:
    stack = load_stack_config(canonical_stack_config_path())

    b1_run_dir = tmp_path / "baselines" / "b1_run_with_aliases"
    display_weights = _write_eval_snapshot(
        stack=stack,
        run_dir=b1_run_dir,
        policy_id=NO_LEAGUE_POLICY_ID,
        update=1,
    )
    canonical_weights = _write_eval_snapshot(
        stack=stack,
        run_dir=b1_run_dir,
        policy_id="b1_noleague_baseline",
        update=5,
    )
    _write_snapshot_registry(
        registry_path=b1_run_dir / "training" / "snapshots" / "registry.json",
        snapshots=[
            (NO_LEAGUE_POLICY_ID, 1, display_weights),
            ("b1_noleague_baseline", 5, canonical_weights),
        ],
    )

    consumer_run_dir = tmp_path / "runs" / "consumer_run_b1_alias_preference"
    (consumer_run_dir / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (consumer_run_dir / "manifest.json").write_text(json.dumps({"run_id256": "ab" * 32}), encoding="utf-8")

    resolved = resolve_eval_policies(
        stack=stack,
        policy_ids=[NO_LEAGUE_POLICY_ID],
        run_dir=consumer_run_dir,
        observation_dim=512,
        action_dim=9,
        spec_bundle=_heuristic_spec_bundle(),
        b1_baseline_run_dir=b1_run_dir,
    )

    assert resolved[NO_LEAGUE_POLICY_ID].snapshot_path == "training/snapshots/B1 NoLeague baseline/weights.pt"


def test_resolve_eval_policies_requires_b1_snapshot_for_mixed_copied_registry_requests(tmp_path: Path) -> None:
    stack = load_stack_config(canonical_stack_config_path())

    source_run_dir = tmp_path / "external_runs" / "source_run"
    registry_path = source_run_dir / "training" / "snapshots" / "registry.json"
    policy_weights = _write_eval_snapshot(
        stack=stack,
        run_dir=source_run_dir,
        policy_id="policy_000100",
        update=100,
    )
    b1_weights = _write_eval_snapshot(
        stack=stack,
        run_dir=source_run_dir,
        policy_id="b1_noleague_baseline",
        update=5,
    )
    _write_snapshot_registry(
        registry_path=registry_path,
        snapshots=[
            ("b1_noleague_baseline", 5, b1_weights),
            ("policy_000100", 100, policy_weights),
        ],
    )

    copied_registry_path = tmp_path / "cache" / "policy_set_snapshot_registry.json"
    copied_registry_path.parent.mkdir(parents=True, exist_ok=True)
    copied_registry_path.write_text(registry_path.read_text(encoding="utf-8"), encoding="utf-8")
    copied_policy_weights = tmp_path / "cache" / "training" / "snapshots" / "policy_000100" / "weights.pt"
    copied_policy_weights.parent.mkdir(parents=True, exist_ok=True)
    copied_policy_weights.write_bytes(policy_weights.read_bytes())
    consumer_run_dir = tmp_path / "runs" / "consumer_run_mixed_b1"
    (consumer_run_dir / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (consumer_run_dir / "manifest.json").write_text(json.dumps({"run_id256": "ab" * 32}), encoding="utf-8")

    resolved = resolve_eval_policies(
        stack=stack,
        policy_ids=[NO_LEAGUE_POLICY_ID, "policy_000100"],
        run_dir=consumer_run_dir,
        observation_dim=512,
        action_dim=9,
        spec_bundle=_heuristic_spec_bundle(),
        snapshot_registry_path=copied_registry_path,
    )

    assert resolved[NO_LEAGUE_POLICY_ID].source_run_dir == source_run_dir.resolve().as_posix()
    assert resolved[NO_LEAGUE_POLICY_ID].snapshot_path == "training/snapshots/b1_noleague_baseline/weights.pt"
    assert resolved["policy_000100"].source_run_dir == source_run_dir.resolve().as_posix()


def test_resolve_eval_policies_skips_registry_resolution_for_explicit_b1_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = load_stack_config(canonical_stack_config_path())

    source_run_dir = tmp_path / "external_runs" / "source_run"
    registry_path = source_run_dir / "training" / "snapshots" / "registry.json"
    policy_weights = _write_eval_snapshot(
        stack=stack,
        run_dir=source_run_dir,
        policy_id="policy_000100",
        update=100,
    )
    _write_snapshot_registry(
        registry_path=registry_path,
        snapshots=[("policy_000100", 100, policy_weights)],
    )

    copied_registry_path = tmp_path / "cache" / "policy_set_snapshot_registry.json"
    copied_registry_path.parent.mkdir(parents=True, exist_ok=True)
    copied_registry_path.write_text(registry_path.read_text(encoding="utf-8"), encoding="utf-8")

    b1_run_dir = tmp_path / "baselines" / "b1_run"
    b1_weights = _write_eval_snapshot(
        stack=stack,
        run_dir=b1_run_dir,
        policy_id="b1_noleague_baseline",
        update=5,
    )
    _write_snapshot_registry(
        registry_path=b1_run_dir / "training" / "snapshots" / "registry.json",
        snapshots=[("b1_noleague_baseline", 5, b1_weights)],
    )

    consumer_run_dir = tmp_path / "runs" / "consumer_run"
    (consumer_run_dir / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (consumer_run_dir / "manifest.json").write_text(json.dumps({"run_id256": "ab" * 32}), encoding="utf-8")

    def _unexpected_registry_resolution(**kwargs):
        raise AssertionError("registry source resolution should not be attempted")

    monkeypatch.setattr(
        policy_resolution_context_module.SnapshotRegistrySource,
        "resolve_run_dir",
        _unexpected_registry_resolution,
    )

    resolved = resolve_eval_policies(
        stack=stack,
        policy_ids=[RANDOM_LEGAL_POLICY_ID, NO_LEAGUE_POLICY_ID],
        run_dir=consumer_run_dir,
        observation_dim=512,
        action_dim=9,
        spec_bundle=_heuristic_spec_bundle(),
        snapshot_registry_path=copied_registry_path,
        b1_baseline_run_dir=b1_run_dir,
    )

    assert resolved[RANDOM_LEGAL_POLICY_ID].kind == "random_legal"
    assert resolved[NO_LEAGUE_POLICY_ID].kind == "baseline_noleague"
    assert resolved[NO_LEAGUE_POLICY_ID].source_run_dir == b1_run_dir.resolve().as_posix()


def test_resolve_eval_policies_refuses_nested_manifest_latest_only_b1_snapshot(tmp_path: Path) -> None:
    stack = load_stack_config(canonical_stack_config_path())

    b1_run_dir = tmp_path / "baselines" / "b1_run"
    latest_weights = _write_eval_snapshot(
        stack=stack,
        run_dir=b1_run_dir,
        policy_id="policy_000005",
        update=5,
    )
    _write_snapshot_registry(
        registry_path=b1_run_dir / "training" / "snapshots" / "registry.json",
        snapshots=[("policy_000005", 5, latest_weights)],
    )
    config_canonical = canonical_config_dict(stack)
    config_canonical["config"]["experiment"] = {
        **dict(config_canonical["config"].get("experiment", {})),
        "role": "baseline_noleague",
    }
    (b1_run_dir / "manifest.json").write_text(
        json.dumps({"config_canonical": config_canonical}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    consumer_run_dir = tmp_path / "runs" / "consumer_run"
    (consumer_run_dir / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (consumer_run_dir / "manifest.json").write_text(json.dumps({"run_id256": "ab" * 32}), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="mandatory B1 NoLeague baseline"):
        resolve_eval_policies(
            stack=stack,
            policy_ids=[NO_LEAGUE_POLICY_ID],
            run_dir=consumer_run_dir,
            observation_dim=512,
            action_dim=9,
            spec_bundle=_heuristic_spec_bundle(),
            b1_baseline_run_dir=b1_run_dir,
        )


def test_resolve_eval_policies_refuses_legacy_manifest_latest_only_b1_snapshot(tmp_path: Path) -> None:
    stack = load_stack_config(canonical_stack_config_path())

    b1_run_dir = tmp_path / "baselines" / "b1_run_legacy"
    latest_weights = _write_eval_snapshot(
        stack=stack,
        run_dir=b1_run_dir,
        policy_id="policy_000005",
        update=5,
    )
    _write_snapshot_registry(
        registry_path=b1_run_dir / "training" / "snapshots" / "registry.json",
        snapshots=[("policy_000005", 5, latest_weights)],
    )
    config_sections = dict(cast(dict[str, Any], canonical_config_dict(stack).get("config", {})))
    config_sections.pop("experiment", None)
    config_sections["training_family_a"] = {"mode": "b1_no_league"}
    (b1_run_dir / "manifest.json").write_text(
        json.dumps({"config_canonical": config_sections}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    consumer_run_dir = tmp_path / "runs" / "consumer_run_legacy"
    (consumer_run_dir / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (consumer_run_dir / "manifest.json").write_text(json.dumps({"run_id256": "ab" * 32}), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="mandatory B1 NoLeague baseline"):
        resolve_eval_policies(
            stack=stack,
            policy_ids=[NO_LEAGUE_POLICY_ID],
            run_dir=consumer_run_dir,
            observation_dim=512,
            action_dim=9,
            spec_bundle=_heuristic_spec_bundle(),
            b1_baseline_run_dir=b1_run_dir,
        )


def test_recursive_registry_search_root_rejects_filesystem_anchor() -> None:
    anchor_root = Path(Path.cwd().anchor)

    assert _is_recursive_registry_search_root(anchor_root) is False
    assert _is_recursive_registry_search_root(anchor_root / "workspace") is True


def test_common_search_root_is_only_used_for_sibling_search_trees(tmp_path: Path) -> None:
    sibling_common_root = tmp_path / "staging"
    sibling_search_roots = [
        sibling_common_root / "runs",
        sibling_common_root / "cache",
    ]
    broad_common_root = tmp_path / "home"
    broad_search_roots = [
        broad_common_root / "Desktop" / "repo" / "runs",
        broad_common_root / "Downloads",
    ]

    assert (
        _should_include_common_search_root(
            search_roots=sibling_search_roots,
            common_search_root=sibling_common_root,
        )
        is True
    )
    assert (
        _should_include_common_search_root(
            search_roots=broad_search_roots,
            common_search_root=broad_common_root,
        )
        is False
    )
