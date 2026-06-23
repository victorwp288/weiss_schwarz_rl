from __future__ import annotations

import numpy as np
import numpy.testing as npt
from weiss_rl.core.action_catalog import ActionCatalog as SharedActionCatalog
from weiss_rl.eval.heuristic_public import HeuristicPublicPolicy

from .heuristic_public_test_support import _empty_obs, _heuristic_spec_bundle, _packed_meta, _set_stage


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


def test_action_catalog_stays_in_core_module_not_heuristic_policy_exports() -> None:
    from weiss_rl.eval import heuristic_public as heuristic_public_module

    assert "ActionCatalog" not in heuristic_public_module.__all__
    assert not hasattr(heuristic_public_module, "ActionCatalog")
    catalog = SharedActionCatalog.from_spec_bundle(_heuristic_spec_bundle())

    decoded = catalog.decode(474)

    assert decoded.family == "attack"
    assert decoded.attack_type == "direct"
