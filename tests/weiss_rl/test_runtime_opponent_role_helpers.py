from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from weiss_rl.eval.policies.set import HEURISTIC_PUBLIC_POLICY_ID
from weiss_rl.experiments.baselines import NOLEAGUE_BASELINE_POLICY_ID
from weiss_rl.runtime.components.counters import collector_counter_template
from weiss_rl.runtime.components.opponents.episode_roles import (
    accumulate_last_pfsp_exposure_counters,
    nondiverse_opponent_role_assignment,
    resolve_fixed_opponent_role_assignment,
)
from weiss_rl.runtime.components.policy_ids import MIRROR_OPPONENT_POLICY_ID


def test_resolve_fixed_opponent_role_assignment_counts_active_fixed_anchor_lanes() -> None:
    fixed = np.asarray(
        [HEURISTIC_PUBLIC_POLICY_ID, NOLEAGUE_BASELINE_POLICY_ID, "inactive", ""],
        dtype=object,
    )

    assignment = resolve_fixed_opponent_role_assignment(
        done=np.asarray([True, True, True, False], dtype=np.bool_),
        fixed_policy_ids=fixed,
        policy_is_active=lambda policy_id: policy_id != "inactive",
    )

    assert assignment.assign_mask.tolist() == [True, True, False, False]
    assert assignment.remaining_mask.tolist() == [False, False, True, False]
    assert assignment.heuristic_public_envs == 1
    assert assignment.noleague_baseline_envs == 1


def test_nondiverse_opponent_role_assignment_exposes_mirror_and_heuristic_anchor_state() -> None:
    mirror_assignment = nondiverse_opponent_role_assignment(
        remaining_count=3,
        league_enabled=False,
        heuristic_anchor_active=True,
    )
    heuristic_assignment = nondiverse_opponent_role_assignment(
        remaining_count=2,
        league_enabled=True,
        heuristic_anchor_active=True,
    )

    assert mirror_assignment.policy_id == MIRROR_OPPONENT_POLICY_ID
    assert mirror_assignment.sampled_envs == 0
    assert mirror_assignment.mirror_envs == 3
    assert mirror_assignment.heuristic_public_policy_envs == {}
    assert heuristic_assignment.policy_id == HEURISTIC_PUBLIC_POLICY_ID
    assert heuristic_assignment.sampled_envs == 2
    assert heuristic_assignment.mirror_envs == 0
    assert heuristic_assignment.heuristic_public_envs == 2
    assert heuristic_assignment.sampled_policy_envs == {HEURISTIC_PUBLIC_POLICY_ID: 2}


def test_accumulate_last_pfsp_exposure_counters_preserves_fixed_anchor_policy_counters() -> None:
    runtime = SimpleNamespace(
        _pfsp_last_sampled_envs=4,
        _pfsp_last_mirror_envs=1,
        _pfsp_last_heuristic_public_envs=1,
        _pfsp_last_heuristic_public_variant_envs=0,
        _pfsp_last_noleague_baseline_envs=1,
        _pfsp_last_champion_envs=0,
        _pfsp_last_recent_envs=2,
        _pfsp_last_hard_negative_envs=0,
        _pfsp_last_warmup_snapshot_envs=0,
        _pfsp_last_sampled_policy_envs={"recent/a": 2},
        _pfsp_last_heuristic_public_policy_envs={},
        _pfsp_last_heuristic_public_variant_policy_envs={},
        _pfsp_last_noleague_baseline_policy_envs={},
        _pfsp_last_champion_policy_envs={},
        _pfsp_last_recent_policy_envs={"recent/a": 2},
        _pfsp_last_hard_negative_policy_envs={},
        _pfsp_last_warmup_snapshot_policy_envs={},
    )
    counters = collector_counter_template()

    accumulate_last_pfsp_exposure_counters(
        counters,
        runtime,
        fixed_heuristic_public_envs=1,
        fixed_noleague_baseline_envs=1,
    )

    assert counters["pfsp_sampled_envs"] == 4
    assert counters["pfsp_mirror_envs"] == 1
    assert counters["pfsp_recent_envs"] == 2
    assert counters["pfsp_sampled_policy_envs__recent_a"] == 2
    assert counters["pfsp_recent_policy_envs__recent_a"] == 2
    assert counters["pfsp_sampled_policy_envs__b2_heuristicpublic"] == 1
    assert counters["pfsp_heuristic_public_policy_envs__b2_heuristicpublic"] == 1
    assert counters["pfsp_sampled_policy_envs__b1_noleague_baseline"] == 1
    assert counters["pfsp_noleague_baseline_policy_envs__b1_noleague_baseline"] == 1
