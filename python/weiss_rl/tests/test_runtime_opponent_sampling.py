from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from weiss_rl.eval.policies.set import (
    HEURISTIC_PUBLIC_AGGRO_POLICY_ID,
    HEURISTIC_PUBLIC_CONTROL_POLICY_ID,
    HEURISTIC_PUBLIC_POLICY_ID,
)
from weiss_rl.league.outcomes import OnlineOutcomeTracker
from weiss_rl.runtime import (
    _MIRROR_OPPONENT_POLICY_ID,
    _NOLEAGUE_BASELINE_POLICY_ID,
    QueueRuntime,
)
from weiss_rl.runtime.components.counters import collector_counter_template
from weiss_rl.runtime.components.opponents.episode_roles import (
    accumulate_last_pfsp_exposure_counters,
    nondiverse_opponent_role_assignment,
    resolve_fixed_opponent_role_assignment,
)


def test_sample_opponent_policy_ids_can_force_hard_negative_bucket() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._league_enabled = True
    runtime_any._pfsp_last_sampled_envs = 0
    runtime_any._pfsp_last_mirror_envs = 0
    runtime_any._pfsp_last_heuristic_public_envs = 0
    runtime_any._pfsp_last_noleague_baseline_envs = 0
    runtime_any._pfsp_last_champion_envs = 0
    runtime_any._pfsp_last_recent_envs = 0
    runtime_any._pfsp_last_hard_negative_envs = 0
    runtime_any._opponent_candidate_ids = ("policy_hard", "policy_recent")
    runtime_any._opponent_hard_negative_ids = ("policy_hard",)
    runtime_any._opponent_champion_ids = ()
    runtime_any._opponent_recent_ids = ("policy_recent",)
    runtime_any._opponent_heuristic_policies = {}
    runtime_any._league_config = SimpleNamespace(
        pfsp_power=2.0,
        pfsp_epsilon_uniform=0.0,
        sampling=SimpleNamespace(
            heuristic_public_start_updates=0,
            heuristic_public_mix_fraction=0.0,
            champion_mix_fraction=0.0,
            hard_negative_mix_fraction=1.0,
        ),
    )
    runtime_any._outcomes = OnlineOutcomeTracker(window_size=128)
    runtime_any._opponent_models = {"policy_hard": object(), "policy_recent": object()}
    runtime_any._pfsp_sampling_ready = lambda: True

    sampled = QueueRuntime._sample_opponent_policy_ids(
        runtime,
        count=4,
        rng=np.random.default_rng(7),
    )

    assert sampled == ("policy_hard", "policy_hard", "policy_hard", "policy_hard")
    assert runtime_any._pfsp_last_hard_negative_envs == 4
    assert runtime_any._pfsp_last_recent_envs == 0
    assert runtime_any._pfsp_last_heuristic_public_envs == 0


def test_resolve_fixed_opponent_role_assignment_counts_active_fixed_anchor_lanes() -> None:
    fixed = np.asarray(
        [HEURISTIC_PUBLIC_POLICY_ID, _NOLEAGUE_BASELINE_POLICY_ID, "inactive", ""],
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

    assert mirror_assignment.policy_id == _MIRROR_OPPONENT_POLICY_ID
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


def test_assign_episode_roles_records_per_policy_exposure_counters() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)

    def fake_sample(*, count: int, rng: np.random.Generator) -> tuple[str, ...]:
        assert count == 3
        assert rng is actor.rng
        runtime_any._pfsp_last_sampled_envs = 3
        runtime_any._pfsp_last_mirror_envs = 0
        runtime_any._pfsp_last_heuristic_public_envs = 0
        runtime_any._pfsp_last_heuristic_public_variant_envs = 0
        runtime_any._pfsp_last_noleague_baseline_envs = 0
        runtime_any._pfsp_last_champion_envs = 2
        runtime_any._pfsp_last_recent_envs = 0
        runtime_any._pfsp_last_hard_negative_envs = 1
        runtime_any._pfsp_last_warmup_snapshot_envs = 0
        runtime_any._pfsp_last_sampled_policy_envs = {"champ a": 2, "hard/b": 1}
        runtime_any._pfsp_last_heuristic_public_policy_envs = {}
        runtime_any._pfsp_last_heuristic_public_variant_policy_envs = {}
        runtime_any._pfsp_last_noleague_baseline_policy_envs = {}
        runtime_any._pfsp_last_champion_policy_envs = {"champ a": 2}
        runtime_any._pfsp_last_recent_policy_envs = {}
        runtime_any._pfsp_last_hard_negative_policy_envs = {"hard/b": 1}
        runtime_any._pfsp_last_warmup_snapshot_policy_envs = {}
        return ("champ a", "hard/b", "champ a")

    actor = SimpleNamespace(
        actor_id=0,
        focal_seat_by_env=np.zeros((3,), dtype=np.int64),
        fixed_opponent_policy_id_by_env=None,
        opponent_policy_id_by_env=np.full((3,), "unknown", dtype=object),
        diverse_opponent_lane=True,
        rng=np.random.default_rng(3),
    )
    runtime_any._sample_opponent_policy_ids = fake_sample
    counters = collector_counter_template()

    QueueRuntime._assign_episode_roles(
        runtime,
        actor,
        np.ones((3,), dtype=np.bool_),
        initial=True,
        counters=counters,
    )

    assert counters["pfsp_champion_envs"] == 2
    assert counters["pfsp_hard_negative_envs"] == 1
    assert counters["pfsp_sampled_policy_envs__champ_a"] == 2
    assert counters["pfsp_sampled_policy_envs__hard_b"] == 1
    assert counters["pfsp_champion_policy_envs__champ_a"] == 2
    assert counters["pfsp_hard_negative_policy_envs__hard_b"] == 1


def test_sample_opponent_policy_ids_can_force_heuristic_public_bucket_before_pfsp_ready() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._league_enabled = True
    runtime_any._pfsp_last_sampled_envs = 0
    runtime_any._pfsp_last_mirror_envs = 0
    runtime_any._pfsp_last_heuristic_public_envs = 0
    runtime_any._pfsp_last_noleague_baseline_envs = 0
    runtime_any._pfsp_last_champion_envs = 0
    runtime_any._pfsp_last_recent_envs = 0
    runtime_any._pfsp_last_hard_negative_envs = 0
    runtime_any._opponent_candidate_ids = ()
    runtime_any._opponent_hard_negative_ids = ()
    runtime_any._opponent_champion_ids = ()
    runtime_any._opponent_recent_ids = ()
    runtime_any._opponent_heuristic_policies = {"B2 HeuristicPublic": object()}
    runtime_any._league_config = SimpleNamespace(
        pfsp_power=2.0,
        pfsp_epsilon_uniform=0.0,
        sampling=SimpleNamespace(
            heuristic_public_start_updates=0,
            heuristic_public_mix_fraction=1.0,
            champion_mix_fraction=0.0,
            hard_negative_mix_fraction=0.0,
        ),
    )
    runtime_any._outcomes = OnlineOutcomeTracker(window_size=128)
    runtime_any._opponent_models = {}
    runtime_any._pfsp_sampling_ready = lambda: False
    runtime_any._league_reference_update = lambda: 0

    sampled = QueueRuntime._sample_opponent_policy_ids(
        runtime,
        count=4,
        rng=np.random.default_rng(7),
    )

    assert sampled == (
        "B2 HeuristicPublic",
        "B2 HeuristicPublic",
        "B2 HeuristicPublic",
        "B2 HeuristicPublic",
    )
    assert runtime_any._pfsp_last_sampled_envs == 4
    assert runtime_any._pfsp_last_mirror_envs == 0
    assert runtime_any._pfsp_last_heuristic_public_envs == 4


def test_sample_opponent_policy_ids_can_force_noleague_baseline_bucket_before_pfsp_ready() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._league_enabled = True
    runtime_any._pfsp_last_sampled_envs = 0
    runtime_any._pfsp_last_mirror_envs = 0
    runtime_any._pfsp_last_heuristic_public_envs = 0
    runtime_any._pfsp_last_noleague_baseline_envs = 0
    runtime_any._pfsp_last_champion_envs = 0
    runtime_any._pfsp_last_recent_envs = 0
    runtime_any._pfsp_last_hard_negative_envs = 0
    runtime_any._opponent_candidate_ids = ()
    runtime_any._opponent_hard_negative_ids = ()
    runtime_any._opponent_champion_ids = ()
    runtime_any._opponent_recent_ids = ()
    runtime_any._opponent_heuristic_policies = {}
    runtime_any._league_config = SimpleNamespace(
        pfsp_power=2.0,
        pfsp_epsilon_uniform=0.0,
        sampling=SimpleNamespace(
            heuristic_public_start_updates=0,
            heuristic_public_mix_fraction=0.0,
            noleague_baseline_mix_fraction=1.0,
            noleague_baseline_mix_end_updates=-1,
            champion_mix_fraction=0.0,
            hard_negative_mix_fraction=0.0,
        ),
    )
    runtime_any._outcomes = OnlineOutcomeTracker(window_size=128)
    runtime_any._opponent_models = {_NOLEAGUE_BASELINE_POLICY_ID: object()}
    runtime_any._pfsp_sampling_ready = lambda: False
    runtime_any._league_reference_update = lambda: 0

    sampled = QueueRuntime._sample_opponent_policy_ids(
        runtime,
        count=4,
        rng=np.random.default_rng(7),
    )

    assert sampled == (
        _NOLEAGUE_BASELINE_POLICY_ID,
        _NOLEAGUE_BASELINE_POLICY_ID,
        _NOLEAGUE_BASELINE_POLICY_ID,
        _NOLEAGUE_BASELINE_POLICY_ID,
    )
    assert runtime_any._pfsp_last_sampled_envs == 4
    assert runtime_any._pfsp_last_mirror_envs == 0
    assert runtime_any._pfsp_last_noleague_baseline_envs == 4


def test_sample_opponent_policy_ids_disables_noleague_mix_after_end_update() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._league_enabled = True
    runtime_any._pfsp_last_sampled_envs = 0
    runtime_any._pfsp_last_mirror_envs = 0
    runtime_any._pfsp_last_heuristic_public_envs = 0
    runtime_any._pfsp_last_noleague_baseline_envs = 0
    runtime_any._pfsp_last_champion_envs = 0
    runtime_any._pfsp_last_recent_envs = 0
    runtime_any._pfsp_last_hard_negative_envs = 0
    runtime_any._opponent_candidate_ids = ()
    runtime_any._opponent_hard_negative_ids = ()
    runtime_any._opponent_champion_ids = ()
    runtime_any._opponent_recent_ids = ()
    runtime_any._opponent_heuristic_policies = {}
    runtime_any._league_config = SimpleNamespace(
        pfsp_power=2.0,
        pfsp_epsilon_uniform=0.0,
        sampling=SimpleNamespace(
            heuristic_public_start_updates=0,
            heuristic_public_mix_fraction=0.0,
            noleague_baseline_mix_fraction=1.0,
            noleague_baseline_mix_end_updates=1,
            champion_mix_fraction=0.0,
            hard_negative_mix_fraction=0.0,
        ),
    )
    runtime_any._outcomes = OnlineOutcomeTracker(window_size=128)
    runtime_any._opponent_models = {_NOLEAGUE_BASELINE_POLICY_ID: object()}
    runtime_any._pfsp_sampling_ready = lambda: False
    runtime_any._league_reference_update = lambda: 1

    sampled = QueueRuntime._sample_opponent_policy_ids(
        runtime,
        count=4,
        rng=np.random.default_rng(7),
    )

    assert sampled == (
        _MIRROR_OPPONENT_POLICY_ID,
        _MIRROR_OPPONENT_POLICY_ID,
        _MIRROR_OPPONENT_POLICY_ID,
        _MIRROR_OPPONENT_POLICY_ID,
    )
    assert runtime_any._pfsp_last_sampled_envs == 0
    assert runtime_any._pfsp_last_mirror_envs == 4
    assert runtime_any._pfsp_last_noleague_baseline_envs == 0


def test_sample_opponent_policy_ids_can_force_warmup_snapshot_bucket_before_pfsp_ready() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._league_enabled = True
    runtime_any._pfsp_last_sampled_envs = 0
    runtime_any._pfsp_last_mirror_envs = 0
    runtime_any._pfsp_last_heuristic_public_envs = 0
    runtime_any._pfsp_last_noleague_baseline_envs = 0
    runtime_any._pfsp_last_champion_envs = 0
    runtime_any._pfsp_last_recent_envs = 0
    runtime_any._pfsp_last_hard_negative_envs = 0
    runtime_any._pfsp_last_warmup_snapshot_envs = 0
    runtime_any._opponent_candidate_ids = ("seed_recent_a", "seed_recent_b")
    runtime_any._opponent_hard_negative_ids = ()
    runtime_any._opponent_champion_ids = ()
    runtime_any._opponent_recent_ids = ()
    runtime_any._opponent_heuristic_policies = {}
    runtime_any._league_config = SimpleNamespace(
        pfsp_power=2.0,
        pfsp_epsilon_uniform=0.0,
        warmup=SimpleNamespace(first_updates=200000),
        sampling=SimpleNamespace(
            heuristic_public_start_updates=0,
            heuristic_public_mix_fraction=0.0,
            noleague_baseline_mix_fraction=0.0,
            noleague_baseline_mix_end_updates=-1,
            warmup_snapshot_mix_fraction=1.0,
            champion_mix_fraction=0.0,
            hard_negative_mix_fraction=0.0,
        ),
    )
    runtime_any._outcomes = OnlineOutcomeTracker(window_size=128)
    runtime_any._opponent_models = {"seed_recent_a": object(), "seed_recent_b": object()}
    runtime_any._pfsp_sampling_ready = lambda: False
    runtime_any._league_reference_update = lambda: 0

    sampled = QueueRuntime._sample_opponent_policy_ids(
        runtime,
        count=4,
        rng=np.random.default_rng(7),
    )

    assert set(sampled) <= {"seed_recent_a", "seed_recent_b"}
    assert len(sampled) == 4
    assert runtime_any._pfsp_last_sampled_envs == 4
    assert runtime_any._pfsp_last_mirror_envs == 0
    assert runtime_any._pfsp_last_recent_envs == 0
    assert runtime_any._pfsp_last_warmup_snapshot_envs == 4


def test_sample_opponent_policy_ids_respects_fractional_warmup_snapshot_mix_before_pfsp_ready() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._league_enabled = True
    runtime_any._pfsp_last_sampled_envs = 0
    runtime_any._pfsp_last_mirror_envs = 0
    runtime_any._pfsp_last_heuristic_public_envs = 0
    runtime_any._pfsp_last_noleague_baseline_envs = 0
    runtime_any._pfsp_last_champion_envs = 0
    runtime_any._pfsp_last_recent_envs = 0
    runtime_any._pfsp_last_hard_negative_envs = 0
    runtime_any._pfsp_last_warmup_snapshot_envs = 0
    runtime_any._opponent_candidate_ids = ("seed_recent_a", "seed_recent_b")
    runtime_any._opponent_hard_negative_ids = ()
    runtime_any._opponent_champion_ids = ()
    runtime_any._opponent_recent_ids = ()
    runtime_any._opponent_heuristic_policies = {HEURISTIC_PUBLIC_POLICY_ID: object()}
    runtime_any._league_config = SimpleNamespace(
        pfsp_power=2.0,
        pfsp_epsilon_uniform=0.0,
        warmup=SimpleNamespace(first_updates=200000),
        sampling=SimpleNamespace(
            heuristic_public_start_updates=0,
            heuristic_public_mix_fraction=0.25,
            noleague_baseline_mix_fraction=0.0,
            noleague_baseline_mix_end_updates=-1,
            warmup_snapshot_mix_fraction=0.5,
            champion_mix_fraction=0.0,
            hard_negative_mix_fraction=0.0,
        ),
    )
    runtime_any._outcomes = OnlineOutcomeTracker(window_size=128)
    runtime_any._opponent_models = {"seed_recent_a": object(), "seed_recent_b": object()}
    runtime_any._pfsp_sampling_ready = lambda: False
    runtime_any._league_reference_update = lambda: 0

    sampled = QueueRuntime._sample_opponent_policy_ids(
        runtime,
        count=200,
        rng=np.random.default_rng(7),
    )

    assert len(sampled) == 200
    assert runtime_any._pfsp_last_heuristic_public_envs > 0
    assert runtime_any._pfsp_last_warmup_snapshot_envs > 0
    assert runtime_any._pfsp_last_mirror_envs > 0
    assert runtime_any._pfsp_last_sampled_envs == (
        runtime_any._pfsp_last_heuristic_public_envs + runtime_any._pfsp_last_warmup_snapshot_envs
    )


def test_active_heuristic_public_mix_fraction_linearly_anneals_with_update() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._league_config = SimpleNamespace(
        sampling=SimpleNamespace(
            heuristic_public_mix_fraction=1.0,
            heuristic_public_mix_end_updates=5,
            heuristic_public_final_mix_fraction=0.25,
        )
    )
    runtime_any._current_learner_update = 0
    runtime_any._effective_learner_update = 0

    assert QueueRuntime._active_heuristic_public_mix_fraction(runtime) == pytest.approx(1.0)

    runtime_any._effective_learner_update = 3

    assert QueueRuntime._active_heuristic_public_mix_fraction(runtime) == pytest.approx(0.55)

    runtime_any._effective_learner_update = 5

    assert QueueRuntime._active_heuristic_public_mix_fraction(runtime) == pytest.approx(0.25)


def test_active_heuristic_public_variant_mix_fraction_linearly_anneals_with_update() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._league_config = SimpleNamespace(
        sampling=SimpleNamespace(
            heuristic_public_variant_mix_fraction=0.4,
            heuristic_public_variant_mix_end_updates=4,
            heuristic_public_variant_final_mix_fraction=0.1,
        )
    )
    runtime_any._current_learner_update = 0
    runtime_any._effective_learner_update = 0
    runtime_any._league_reference_update = lambda: runtime_any._effective_learner_update

    assert QueueRuntime._active_heuristic_public_variant_mix_fraction(runtime) == pytest.approx(0.4)

    runtime_any._effective_learner_update = 2

    assert QueueRuntime._active_heuristic_public_variant_mix_fraction(runtime) == pytest.approx(0.25)

    runtime_any._effective_learner_update = 4

    assert QueueRuntime._active_heuristic_public_variant_mix_fraction(runtime) == pytest.approx(0.1)


def test_heuristic_opponent_policy_falls_back_to_teacher_for_b2() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    teacher_policy = object()
    runtime_any._teacher_policy = teacher_policy
    runtime_any._opponent_heuristic_policies = {
        HEURISTIC_PUBLIC_AGGRO_POLICY_ID: object(),
    }

    assert QueueRuntime._heuristic_opponent_policy(runtime, HEURISTIC_PUBLIC_POLICY_ID) is teacher_policy
    assert (
        QueueRuntime._heuristic_opponent_policy(runtime, HEURISTIC_PUBLIC_AGGRO_POLICY_ID)
        is runtime_any._opponent_heuristic_policies[HEURISTIC_PUBLIC_AGGRO_POLICY_ID]
    )


def test_active_warmup_snapshot_mix_fraction_turns_off_after_league_warmup() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._league_config = SimpleNamespace(
        warmup=SimpleNamespace(first_updates=5),
        sampling=SimpleNamespace(warmup_snapshot_mix_fraction=0.4),
    )
    runtime_any._opponent_candidate_ids = ("seed_a",)
    runtime_any._opponent_models = {"seed_a": object()}
    runtime_any._current_learner_update = 0
    runtime_any._effective_learner_update = 0

    assert QueueRuntime._active_warmup_snapshot_mix_fraction(runtime) == pytest.approx(0.4)

    runtime_any._effective_learner_update = 5

    assert QueueRuntime._active_warmup_snapshot_mix_fraction(runtime) == pytest.approx(0.0)


def test_assign_episode_roles_uses_weighted_sampler_on_diverse_warmup_lane() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._pfsp_last_sampled_envs = 0
    runtime_any._pfsp_last_mirror_envs = 0
    runtime_any._pfsp_last_heuristic_public_envs = 0
    runtime_any._pfsp_last_noleague_baseline_envs = 0
    runtime_any._pfsp_last_champion_envs = 0
    runtime_any._pfsp_last_recent_envs = 0
    runtime_any._pfsp_last_hard_negative_envs = 0
    runtime_any._pfsp_last_warmup_snapshot_envs = 0
    runtime_any._league_enabled = True
    runtime_any._league_config = SimpleNamespace()
    runtime_any._pfsp_sampling_ready = lambda: False
    runtime_any._active_warmup_snapshot_mix_fraction = lambda: 0.5
    runtime_any._opponent_candidate_ids = ("seed_recent_a", "seed_recent_b")
    warmup_calls: list[int] = []
    weighted_calls: list[int] = []

    def sample_warmup_only(*, count, rng):
        warmup_calls.append(int(count))
        return ("seed_recent_a",) * count

    def sample_weighted(*, count, rng):
        weighted_calls.append(int(count))
        return ("seed_recent_a", "B2 HeuristicPublic")[:count]

    runtime_any._sample_warmup_snapshot_policy_ids = sample_warmup_only
    runtime_any._sample_opponent_policy_ids = sample_weighted
    runtime_any._fixed_opponent_policy_is_active = lambda policy_id: False

    actor = cast(
        Any,
        SimpleNamespace(
            actor_id=0,
            rng=np.random.default_rng(7),
            focal_seat_by_env=np.asarray([0, 1], dtype=np.int64),
            opponent_policy_id_by_env=np.asarray(
                [_MIRROR_OPPONENT_POLICY_ID, _MIRROR_OPPONENT_POLICY_ID], dtype=object
            ),
            fixed_opponent_policy_id_by_env=None,
            diverse_opponent_lane=True,
        ),
    )

    QueueRuntime._assign_episode_roles(
        runtime,
        actor,
        np.asarray([True, True], dtype=np.bool_),
        initial=False,
    )

    assert actor.opponent_policy_id_by_env.tolist() == ["seed_recent_a", "B2 HeuristicPublic"]
    assert weighted_calls == [2]
    assert warmup_calls == []


def test_assign_episode_roles_uses_mirror_for_nondiverse_lane_when_league_disabled() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._pfsp_last_sampled_envs = 0
    runtime_any._pfsp_last_mirror_envs = 0
    runtime_any._pfsp_last_heuristic_public_envs = 0
    runtime_any._pfsp_last_heuristic_public_variant_envs = 0
    runtime_any._pfsp_last_noleague_baseline_envs = 0
    runtime_any._pfsp_last_champion_envs = 0
    runtime_any._pfsp_last_recent_envs = 0
    runtime_any._pfsp_last_hard_negative_envs = 0
    runtime_any._pfsp_last_warmup_snapshot_envs = 0
    runtime_any._league_enabled = False
    runtime_any._fixed_opponent_policy_is_active = lambda policy_id: True

    actor = cast(
        Any,
        SimpleNamespace(
            actor_id=0,
            rng=np.random.default_rng(7),
            focal_seat_by_env=np.asarray([0, 1], dtype=np.int64),
            opponent_policy_id_by_env=np.asarray(["old0", "old1"], dtype=object),
            fixed_opponent_policy_id_by_env=None,
            diverse_opponent_lane=False,
        ),
    )
    counters = collector_counter_template()

    QueueRuntime._assign_episode_roles(
        runtime,
        actor,
        np.asarray([True, True], dtype=np.bool_),
        initial=False,
        counters=cast(Any, counters),
    )

    assert actor.opponent_policy_id_by_env.tolist() == [_MIRROR_OPPONENT_POLICY_ID, _MIRROR_OPPONENT_POLICY_ID]
    assert runtime_any._pfsp_last_sampled_envs == 0
    assert runtime_any._pfsp_last_mirror_envs == 2
    assert runtime_any._pfsp_last_heuristic_public_envs == 0
    assert counters["pfsp_sampled_envs"] == 0
    assert counters["pfsp_mirror_envs"] == 2
    assert counters["pfsp_heuristic_public_envs"] == 0


def test_sample_opponent_policy_ids_respects_heuristic_public_mix_anneal_end_update() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._league_enabled = True
    runtime_any._pfsp_last_sampled_envs = 0
    runtime_any._pfsp_last_mirror_envs = 0
    runtime_any._pfsp_last_heuristic_public_envs = 0
    runtime_any._pfsp_last_noleague_baseline_envs = 0
    runtime_any._pfsp_last_champion_envs = 0
    runtime_any._pfsp_last_recent_envs = 0
    runtime_any._pfsp_last_hard_negative_envs = 0
    runtime_any._opponent_candidate_ids = ()
    runtime_any._opponent_hard_negative_ids = ()
    runtime_any._opponent_champion_ids = ()
    runtime_any._opponent_recent_ids = ()
    runtime_any._opponent_heuristic_policies = {"B2 HeuristicPublic": object()}
    runtime_any._league_config = SimpleNamespace(
        pfsp_power=2.0,
        pfsp_epsilon_uniform=0.0,
        sampling=SimpleNamespace(
            heuristic_public_start_updates=0,
            heuristic_public_mix_fraction=1.0,
            heuristic_public_mix_end_updates=1,
            heuristic_public_final_mix_fraction=0.0,
            champion_mix_fraction=0.0,
            hard_negative_mix_fraction=0.0,
        ),
    )
    runtime_any._outcomes = OnlineOutcomeTracker(window_size=128)
    runtime_any._opponent_models = {}
    runtime_any._pfsp_sampling_ready = lambda: False
    runtime_any._league_reference_update = lambda: 1

    sampled = QueueRuntime._sample_opponent_policy_ids(
        runtime,
        count=4,
        rng=np.random.default_rng(7),
    )

    assert sampled == (
        _MIRROR_OPPONENT_POLICY_ID,
        _MIRROR_OPPONENT_POLICY_ID,
        _MIRROR_OPPONENT_POLICY_ID,
        _MIRROR_OPPONENT_POLICY_ID,
    )
    assert runtime_any._pfsp_last_sampled_envs == 0
    assert runtime_any._pfsp_last_mirror_envs == 4
    assert runtime_any._pfsp_last_heuristic_public_envs == 0


def test_sample_opponent_policy_ids_can_force_heuristic_public_variant_bucket_before_pfsp_ready() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._pfsp_last_sampled_envs = 0
    runtime_any._pfsp_last_mirror_envs = 0
    runtime_any._pfsp_last_heuristic_public_envs = 0
    runtime_any._pfsp_last_heuristic_public_variant_envs = 0
    runtime_any._pfsp_last_noleague_baseline_envs = 0
    runtime_any._pfsp_last_champion_envs = 0
    runtime_any._pfsp_last_recent_envs = 0
    runtime_any._pfsp_last_hard_negative_envs = 0
    runtime_any._pfsp_last_warmup_snapshot_envs = 0
    runtime_any._league_enabled = True
    runtime_any._league_config = SimpleNamespace(
        sampling=SimpleNamespace(
            heuristic_public_start_updates=0,
            heuristic_public_mix_fraction=0.0,
            heuristic_public_variant_mix_fraction=1.0,
            champion_mix_fraction=0.0,
            hard_negative_mix_fraction=0.0,
        ),
        pfsp_power=1.5,
        pfsp_epsilon_uniform=0.3,
    )
    runtime_any._opponent_heuristic_policies = {
        HEURISTIC_PUBLIC_AGGRO_POLICY_ID: object(),
        HEURISTIC_PUBLIC_CONTROL_POLICY_ID: object(),
    }
    runtime_any._opponent_models = {}
    runtime_any._opponent_candidate_ids = ()
    runtime_any._outcomes = OnlineOutcomeTracker(window_size=128)
    runtime_any._pfsp_sampling_ready = lambda: False
    runtime_any._league_reference_update = lambda: 0

    sampled = QueueRuntime._sample_opponent_policy_ids(
        runtime,
        count=8,
        rng=np.random.default_rng(7),
    )

    assert set(sampled).issubset({HEURISTIC_PUBLIC_AGGRO_POLICY_ID, HEURISTIC_PUBLIC_CONTROL_POLICY_ID})
    assert runtime_any._pfsp_last_heuristic_public_envs == 0
    assert runtime_any._pfsp_last_heuristic_public_variant_envs == 8


def test_active_actor_heuristic_fraction_linearly_anneals_with_update() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._actor_heuristic_fraction = 1.0
    runtime_any._actor_heuristic_start_updates = 0
    runtime_any._actor_heuristic_end_updates = 5
    runtime_any._actor_heuristic_final_fraction = 0.25
    runtime_any._current_learner_update = 0
    runtime_any._effective_learner_update = 0

    assert QueueRuntime._active_actor_heuristic_fraction(runtime) == pytest.approx(1.0)

    runtime_any._effective_learner_update = 3

    assert QueueRuntime._active_actor_heuristic_fraction(runtime) == pytest.approx(0.55)

    runtime_any._effective_learner_update = 5

    assert QueueRuntime._active_actor_heuristic_fraction(runtime) == pytest.approx(0.25)


def test_active_actor_heuristic_fraction_respects_delayed_anneal_start() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._actor_heuristic_fraction = 1.0
    runtime_any._actor_heuristic_start_updates = 4
    runtime_any._actor_heuristic_end_updates = 8
    runtime_any._actor_heuristic_final_fraction = 0.25
    runtime_any._current_learner_update = 0
    runtime_any._effective_learner_update = 0

    assert QueueRuntime._active_actor_heuristic_fraction(runtime) == pytest.approx(1.0)

    runtime_any._effective_learner_update = 6

    assert QueueRuntime._active_actor_heuristic_fraction(runtime) == pytest.approx(0.625)

    runtime_any._effective_learner_update = 8

    assert QueueRuntime._active_actor_heuristic_fraction(runtime) == pytest.approx(0.25)


def test_split_focal_actor_rows_respects_actor_heuristic_anneal_endpoint() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._actor_policy_backend = "heuristic_public"
    runtime_any._teacher_policy = object()
    runtime_any._actor_heuristic_fraction = 1.0
    runtime_any._actor_heuristic_end_updates = 5
    runtime_any._actor_heuristic_final_fraction = 0.0
    runtime_any._current_learner_update = 5
    runtime_any._effective_learner_update = 5

    model_rows, heuristic_rows = QueueRuntime._split_focal_actor_rows(
        runtime,
        actor=cast(Any, SimpleNamespace(force_model_policy_lane=False)),
        focal_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
        rng=np.random.default_rng(7),
    )

    assert model_rows.tolist() == [0, 1, 2, 3]
    assert heuristic_rows.tolist() == []


def test_assign_episode_roles_prioritizes_fixed_anchor_lanes() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._pfsp_last_sampled_envs = 0
    runtime_any._pfsp_last_mirror_envs = 0
    runtime_any._pfsp_last_heuristic_public_envs = 0
    runtime_any._pfsp_last_noleague_baseline_envs = 0
    runtime_any._pfsp_last_champion_envs = 0
    runtime_any._pfsp_last_recent_envs = 0
    runtime_any._pfsp_last_hard_negative_envs = 0
    runtime_any._fixed_opponent_policy_is_active = lambda policy_id: True

    def fake_sample(*, count: int, rng) -> tuple[str, ...]:
        runtime_any._pfsp_last_sampled_envs = count
        runtime_any._pfsp_last_mirror_envs = 0
        runtime_any._pfsp_last_heuristic_public_envs = 0
        runtime_any._pfsp_last_noleague_baseline_envs = 0
        runtime_any._pfsp_last_champion_envs = 0
        runtime_any._pfsp_last_recent_envs = count
        runtime_any._pfsp_last_hard_negative_envs = 0
        return tuple(f"recent_{index}" for index in range(count))

    runtime_any._sample_opponent_policy_ids = fake_sample

    actor = cast(
        Any,
        SimpleNamespace(
            actor_id=0,
            rng=np.random.default_rng(7),
            focal_seat_by_env=np.asarray([0, 1, 0, 1], dtype=np.int64),
            opponent_policy_id_by_env=np.asarray(["old0", "old1", "old2", "old3"], dtype=object),
            fixed_opponent_policy_id_by_env=np.asarray(
                ["B2 HeuristicPublic", "b1_noleague_baseline", "", ""],
                dtype=object,
            ),
        ),
    )

    QueueRuntime._assign_episode_roles(
        runtime, actor, np.asarray([True, True, True, True], dtype=np.bool_), initial=False
    )

    assert actor.opponent_policy_id_by_env.tolist() == [
        "B2 HeuristicPublic",
        "b1_noleague_baseline",
        "recent_0",
        "recent_1",
    ]
    assert runtime_any._pfsp_last_sampled_envs == 4
    assert runtime_any._pfsp_last_heuristic_public_envs == 1
    assert runtime_any._pfsp_last_noleague_baseline_envs == 1
    assert runtime_any._pfsp_last_recent_envs == 2


def test_overwrite_central_outputs_with_opponents_only_touches_non_mirror_rows() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._device = torch.device("cpu")
    runtime_any._actor_amp_enabled = False
    runtime_any._opponent_model_locks = {"policy_000007": threading.Lock()}
    runtime_any._opponent_heuristic_policies = {}

    class _FakeModel:
        def __init__(self) -> None:
            self.calls: list[tuple[np.ndarray, np.ndarray]] = []

        def forward_seat_aware(self, obs_tensor, actor_tensor, hidden_tensor):
            obs_np = obs_tensor.detach().cpu().numpy()
            actor_np = actor_tensor.detach().cpu().numpy()
            self.calls.append((obs_np.copy(), actor_np.copy()))
            logits = torch.full((obs_tensor.shape[0], 5), -7.0, dtype=torch.float32)
            values = torch.full((obs_tensor.shape[0],), 42.0, dtype=torch.float32)
            return logits, values, hidden_tensor + 1.0

    model = _FakeModel()
    runtime_any._opponent_models = {"policy_000007": model}

    actor = cast(
        Any,
        SimpleNamespace(
            layout_name="i16_legal_ids",
            focal_seat_by_env=np.asarray([0, 0, 0], dtype=np.int64),
            opponent_policy_id_by_env=np.asarray(
                [_MIRROR_OPPONENT_POLICY_ID, "policy_000007", _MIRROR_OPPONENT_POLICY_ID],
                dtype=object,
            ),
            opponent_hidden=torch.zeros((3, 4)),
            rng=np.random.default_rng(7),
        ),
    )
    batch = cast(
        Any,
        SimpleNamespace(
            obs=np.zeros((3, 8), dtype=np.float32),
            actor=np.asarray([1, 1, 1], dtype=np.int64),
            ids_offsets=(
                np.asarray([0, 1, 2], dtype=np.uint32),
                np.asarray([0, 1, 2, 3], dtype=np.uint32),
            ),
            mask=None,
        ),
    )
    logits = np.zeros((3, 5), dtype=np.float32)
    values = np.zeros((3,), dtype=np.float32)

    QueueRuntime._overwrite_central_outputs_with_opponents(
        runtime,
        actor=actor,
        batch=batch,
        obs_step=np.asarray(batch.obs, dtype=np.float32),
        actor_step=np.asarray(batch.actor, dtype=np.int64),
        logits_out=logits,
        values_out=values,
    )

    assert len(model.calls) == 1
    assert np.array_equal(model.calls[0][0], np.zeros((1, 8), dtype=np.float32))
    assert np.array_equal(model.calls[0][1], np.asarray([1], dtype=np.int64))
    assert values.tolist() == [0.0, 42.0, 0.0]
    assert np.all(logits[0] == 0.0)
    assert np.all(logits[1] == -7.0)
    assert np.all(logits[2] == 0.0)
    assert actor.opponent_hidden[1].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert actor.opponent_hidden[0].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_overwrite_central_outputs_with_batched_opponents_groups_rows_across_actors() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._device = torch.device("cpu")
    runtime_any._actor_amp_enabled = False
    runtime_any._opponent_model_locks = {"policy_000007": threading.Lock()}

    class _FakeModel:
        def __init__(self) -> None:
            self.calls: list[tuple[np.ndarray, np.ndarray]] = []

        def forward_seat_aware(self, obs_tensor, actor_tensor, hidden_tensor):
            obs_np = obs_tensor.detach().cpu().numpy()
            actor_np = actor_tensor.detach().cpu().numpy()
            self.calls.append((obs_np.copy(), actor_np.copy()))
            logits = torch.arange(obs_tensor.shape[0] * 5, dtype=torch.float32).reshape(obs_tensor.shape[0], 5)
            values = torch.arange(obs_tensor.shape[0], dtype=torch.float32) + 10.0
            return logits, values, hidden_tensor + 2.0

    model = _FakeModel()
    runtime_any._opponent_models = {"policy_000007": model}

    actor_a = cast(
        Any,
        SimpleNamespace(
            focal_seat_by_env=np.asarray([0, 0], dtype=np.int64),
            opponent_policy_id_by_env=np.asarray(["policy_000007", _MIRROR_OPPONENT_POLICY_ID], dtype=object),
            opponent_hidden=torch.zeros((2, 3)),
        ),
    )
    actor_b = cast(
        Any,
        SimpleNamespace(
            focal_seat_by_env=np.asarray([1, 1], dtype=np.int64),
            opponent_policy_id_by_env=np.asarray([_MIRROR_OPPONENT_POLICY_ID, "policy_000007"], dtype=object),
            opponent_hidden=torch.zeros((2, 3)),
        ),
    )
    logits_a = np.zeros((2, 5), dtype=np.float32)
    logits_b = np.zeros((2, 5), dtype=np.float32)
    values_a = np.zeros((2,), dtype=np.float32)
    values_b = np.zeros((2,), dtype=np.float32)
    obs_a = np.asarray([[1.0, 0.0], [9.0, 9.0]], dtype=np.float32)
    obs_b = np.asarray([[8.0, 8.0], [2.0, 0.0]], dtype=np.float32)
    actor_step_a = np.asarray([1, 0], dtype=np.int64)
    actor_step_b = np.asarray([1, 0], dtype=np.int64)

    QueueRuntime._overwrite_central_outputs_with_batched_opponents(
        runtime,
        actors=[actor_a, actor_b],
        batches=[cast(Any, SimpleNamespace()), cast(Any, SimpleNamespace())],
        obs_steps=[obs_a, obs_b],
        actor_steps=[actor_step_a, actor_step_b],
        logits_outs=[logits_a, logits_b],
        values_outs=[values_a, values_b],
    )

    assert len(model.calls) == 1
    assert np.array_equal(model.calls[0][0], np.asarray([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32))
    assert np.array_equal(model.calls[0][1], np.asarray([1, 0], dtype=np.int64))
    assert np.all(logits_a[0] == np.asarray([0, 1, 2, 3, 4], dtype=np.float32))
    assert np.all(logits_b[1] == np.asarray([5, 6, 7, 8, 9], dtype=np.float32))
    assert values_a.tolist() == [10.0, 0.0]
    assert values_b.tolist() == [0.0, 11.0]
    assert actor_a.opponent_hidden[0].tolist() == [2.0, 2.0, 2.0]
    assert actor_b.opponent_hidden[1].tolist() == [2.0, 2.0, 2.0]


def test_overwrite_central_outputs_with_batched_opponents_batches_heuristic_public_rows_across_actors() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._device = torch.device("cpu")
    runtime_any._actor_amp_enabled = False
    runtime_any._profile_timers = False
    runtime_any.action_dim = 32
    runtime_any._opponent_models = {}
    runtime_any._opponent_model_locks = {}

    class _AdvanceOnlyModel:
        def advance_seat_hidden(self, obs_tensor, actor_tensor, hidden_tensor):
            return hidden_tensor + 1.0

    class _FakeHeuristicPolicy:
        def __init__(self) -> None:
            self.calls: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]] = []

        def choose_actions_from_meta_batch(self, obs_rows, legal_ids, legal_offsets, legal_action_meta):
            obs_array = np.asarray(obs_rows, dtype=np.int32)
            ids_array = np.asarray(legal_ids, dtype=np.uint32)
            offsets_array = np.asarray(legal_offsets, dtype=np.uint32)
            meta_array = None if legal_action_meta is None else np.asarray(legal_action_meta, dtype=np.uint16)
            self.calls.append(
                (
                    obs_array.copy(),
                    ids_array.copy(),
                    offsets_array.copy(),
                    None if meta_array is None else meta_array.copy(),
                )
            )
            return np.asarray(
                [int(ids_array[int(offsets_array[row_index])]) for row_index in range(offsets_array.shape[0] - 1)],
                dtype=np.int64,
            )

    heuristic_policy = _FakeHeuristicPolicy()
    runtime_any._opponent_heuristic_policies = {HEURISTIC_PUBLIC_POLICY_ID: heuristic_policy}

    shared_model = _AdvanceOnlyModel()
    actor_a = cast(
        Any,
        SimpleNamespace(
            model=shared_model,
            compiled_model=None,
            focal_seat_by_env=np.asarray([0, 0], dtype=np.int64),
            opponent_policy_id_by_env=np.asarray(
                [HEURISTIC_PUBLIC_POLICY_ID, _MIRROR_OPPONENT_POLICY_ID], dtype=object
            ),
            seat_hidden=torch.zeros((2, 3)),
            opponent_hidden=torch.zeros((2, 3)),
        ),
    )
    actor_b = cast(
        Any,
        SimpleNamespace(
            model=shared_model,
            compiled_model=None,
            focal_seat_by_env=np.asarray([1, 1], dtype=np.int64),
            opponent_policy_id_by_env=np.asarray(
                [_MIRROR_OPPONENT_POLICY_ID, HEURISTIC_PUBLIC_POLICY_ID], dtype=object
            ),
            seat_hidden=torch.zeros((2, 3)),
            opponent_hidden=torch.zeros((2, 3)),
        ),
    )
    batch_a = cast(
        Any,
        SimpleNamespace(
            ids_offsets=(
                np.asarray([10, 11, 12], dtype=np.uint32),
                np.asarray([0, 2, 3], dtype=np.uint32),
            ),
            legal_action_meta=np.asarray(
                [
                    [0, 0, 0, 0],
                    [1, 1, 1, 0],
                    [2, 2, 2, 0],
                ],
                dtype=np.uint16,
            ),
            mask=None,
        ),
    )
    batch_b = cast(
        Any,
        SimpleNamespace(
            ids_offsets=(
                np.asarray([20, 21, 22], dtype=np.uint32),
                np.asarray([0, 1, 3], dtype=np.uint32),
            ),
            legal_action_meta=np.asarray(
                [
                    [3, 0, 0, 0],
                    [4, 1, 0, 0],
                    [5, 2, 0, 0],
                ],
                dtype=np.uint16,
            ),
            mask=None,
        ),
    )
    obs_a = np.asarray([[1, 0, 0], [9, 9, 9]], dtype=np.float32)
    obs_b = np.asarray([[8, 8, 8], [2, 0, 0]], dtype=np.float32)
    actor_step_a = np.asarray([1, 0], dtype=np.int64)
    actor_step_b = np.asarray([1, 0], dtype=np.int64)
    logits_a = np.full((2, 32), -5.0, dtype=np.float32)
    logits_b = np.full((2, 32), -5.0, dtype=np.float32)
    values_a = np.ones((2,), dtype=np.float32)
    values_b = np.ones((2,), dtype=np.float32)

    QueueRuntime._overwrite_central_outputs_with_batched_opponents(
        runtime,
        actors=[actor_a, actor_b],
        batches=[batch_a, batch_b],
        obs_steps=[obs_a, obs_b],
        actor_steps=[actor_step_a, actor_step_b],
        logits_outs=[logits_a, logits_b],
        values_outs=[values_a, values_b],
    )

    assert len(heuristic_policy.calls) == 1
    call_obs, call_ids, call_offsets, call_meta = heuristic_policy.calls[0]
    assert np.array_equal(call_obs, np.asarray([[1, 0, 0], [2, 0, 0]], dtype=np.int32))
    assert np.array_equal(call_ids, np.asarray([10, 11, 21, 22], dtype=np.uint32))
    assert np.array_equal(call_offsets, np.asarray([0, 2, 4], dtype=np.uint32))
    assert call_meta is not None
    assert np.array_equal(
        call_meta,
        np.asarray(
            [
                [0, 0, 0, 0],
                [1, 1, 1, 0],
                [4, 1, 0, 0],
                [5, 2, 0, 0],
            ],
            dtype=np.uint16,
        ),
    )
    assert actor_a.seat_hidden[0].tolist() == [1.0, 1.0, 1.0]
    assert actor_b.seat_hidden[1].tolist() == [1.0, 1.0, 1.0]
    assert values_a.tolist() == [0.0, 1.0]
    assert values_b.tolist() == [1.0, 0.0]
    assert logits_a[0, 10] == pytest.approx(0.0)
    assert logits_a[0, 11] < 0.0
    assert logits_b[1, 21] == pytest.approx(0.0)
    assert logits_b[1, 22] < 0.0


def test_apply_opponent_rows_ids_uses_simulator_native_backend_for_heuristic_public() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._device = torch.device("cpu")
    runtime_any._actor_amp_enabled = False
    runtime_any._fixed_opponent_backend = "simulator_native"
    runtime_any.action_dim = 32
    runtime_any._opponent_models = {}
    runtime_any._opponent_model_locks = {}

    class _AdvanceOnlyModel:
        def advance_seat_hidden(self, obs_tensor, actor_tensor, hidden_tensor):
            return hidden_tensor + 1.0

    class _FailHeuristicPolicy:
        def choose_actions_from_meta_batch(self, obs_rows, legal_ids, legal_offsets, legal_action_meta):
            raise AssertionError("python heuristic batch path should not be used when simulator_native is enabled")

    class _FakePool:
        def __init__(self) -> None:
            self.calls: list[np.ndarray] = []

        def choose_heuristic_public_actions_into(self, env_indices: np.ndarray, actions_out: np.ndarray) -> None:
            indices = np.asarray(env_indices, dtype=np.uint32)
            self.calls.append(indices.copy())
            actions_out[...] = np.asarray([11, 20], dtype=np.uint16)

    fake_pool = _FakePool()
    runtime_any._opponent_heuristic_policies = {HEURISTIC_PUBLIC_POLICY_ID: _FailHeuristicPolicy()}

    actor = cast(
        Any,
        SimpleNamespace(
            model=_AdvanceOnlyModel(),
            compiled_model=None,
            env=SimpleNamespace(pool=fake_pool),
            opponent_policy_id_by_env=np.asarray(
                [HEURISTIC_PUBLIC_POLICY_ID, HEURISTIC_PUBLIC_POLICY_ID, _MIRROR_OPPONENT_POLICY_ID],
                dtype=object,
            ),
            seat_hidden=torch.zeros((3, 2)),
            opponent_hidden=torch.zeros((3, 2)),
        ),
    )
    row_indices = np.asarray([0, 1], dtype=np.int64)
    obs_step = np.asarray([[1, 0], [2, 0], [3, 0]], dtype=np.float32)
    actor_step = np.asarray([1, 1, 0], dtype=np.int64)
    legal_ids = np.asarray([10, 11, 12, 20, 21], dtype=np.uint32)
    legal_offsets = np.asarray([0, 3, 5, 5], dtype=np.uint32)
    legal_action_meta = np.asarray(
        [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [2, 0, 0, 0],
            [3, 0, 0, 0],
            [4, 0, 0, 0],
        ],
        dtype=np.uint16,
    )
    values_out = np.ones((3,), dtype=np.float32)
    actions_out = np.full((3,), 99, dtype=np.int64)
    logp_out = np.full((3,), -1.0, dtype=np.float32)

    QueueRuntime._apply_opponent_rows_ids(
        runtime,
        actor=actor,
        row_indices=row_indices,
        obs_step=obs_step,
        actor_step=actor_step,
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        legal_action_meta=legal_action_meta,
        logits_out=None,
        values_out=values_out,
        actions_out=actions_out,
        logp_out=logp_out,
        rng=np.random.default_rng(7),
        sample_actions=True,
    )

    assert len(fake_pool.calls) == 1
    assert np.array_equal(fake_pool.calls[0], np.asarray([0, 1], dtype=np.uint32))
    assert actor.seat_hidden[0].tolist() == [1.0, 1.0]
    assert actor.seat_hidden[1].tolist() == [1.0, 1.0]
    assert values_out.tolist() == [0.0, 0.0, 1.0]
    assert actions_out.tolist() == [11, 20, 99]
    assert logp_out.tolist() == [0.0, 0.0, -1.0]


def test_apply_opponent_rows_ids_uses_python_profile_oracle_for_b3_b4_native_backend() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._device = torch.device("cpu")
    runtime_any._actor_amp_enabled = False
    runtime_any._fixed_opponent_backend = "simulator_native"
    runtime_any.action_dim = 32
    runtime_any._opponent_models = {}
    runtime_any._opponent_model_locks = {}

    class _AdvanceOnlyModel:
        def advance_seat_hidden(self, obs_tensor, actor_tensor, hidden_tensor):
            return hidden_tensor + 1.0

    class _RecordingHeuristicPolicy:
        def __init__(self) -> None:
            self.calls = 0

        def choose_actions_from_meta_batch(self, obs_rows, legal_ids, legal_offsets, legal_action_meta):
            self.calls += 1
            return np.asarray([11, 20], dtype=np.int64)

    class _FakePool:
        def __init__(self) -> None:
            self.profile_calls: list[tuple[np.ndarray, str]] = []

        def choose_heuristic_public_actions_into(self, env_indices: np.ndarray, actions_out: np.ndarray) -> None:
            raise AssertionError("base native heuristic must not be used for B3/B4 profiles")

        def choose_heuristic_public_profile_actions_into(
            self,
            env_indices: np.ndarray,
            actions_out: np.ndarray,
            profile_name: str,
        ) -> None:
            raise AssertionError("profile-native heuristic is not used until simulator/RL profile parity is proven")

    fake_pool = _FakePool()
    heuristic_policy = _RecordingHeuristicPolicy()
    runtime_any._opponent_heuristic_policies = {HEURISTIC_PUBLIC_AGGRO_POLICY_ID: heuristic_policy}
    actor = cast(
        Any,
        SimpleNamespace(
            model=_AdvanceOnlyModel(),
            compiled_model=None,
            env=SimpleNamespace(pool=fake_pool),
            opponent_policy_id_by_env=np.asarray(
                [HEURISTIC_PUBLIC_AGGRO_POLICY_ID, HEURISTIC_PUBLIC_AGGRO_POLICY_ID, _MIRROR_OPPONENT_POLICY_ID],
                dtype=object,
            ),
            seat_hidden=torch.zeros((3, 2)),
            opponent_hidden=torch.zeros((3, 2)),
        ),
    )
    values_out = np.ones((3,), dtype=np.float32)
    actions_out = np.full((3,), 99, dtype=np.int64)
    logp_out = np.full((3,), -1.0, dtype=np.float32)

    QueueRuntime._apply_opponent_rows_ids(
        runtime,
        actor=actor,
        row_indices=np.asarray([0, 1], dtype=np.int64),
        obs_step=np.asarray([[1, 0], [2, 0], [3, 0]], dtype=np.float32),
        actor_step=np.asarray([1, 1, 0], dtype=np.int64),
        legal_ids=np.asarray([10, 11, 12, 20, 21], dtype=np.uint32),
        legal_offsets=np.asarray([0, 3, 5, 5], dtype=np.uint32),
        legal_action_meta=np.zeros((5, 4), dtype=np.uint16),
        logits_out=None,
        values_out=values_out,
        actions_out=actions_out,
        logp_out=logp_out,
        rng=np.random.default_rng(7),
        sample_actions=True,
    )

    assert fake_pool.profile_calls == []
    assert heuristic_policy.calls == 1
    assert values_out.tolist() == [0.0, 0.0, 1.0]
    assert actions_out.tolist() == [11, 20, 99]
    assert logp_out.tolist() == [0.0, 0.0, -1.0]


def test_apply_opponent_rows_ids_falls_back_for_profile_when_profile_native_hook_is_unavailable() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._device = torch.device("cpu")
    runtime_any._actor_amp_enabled = False
    runtime_any._fixed_opponent_backend = "simulator_native"
    runtime_any.action_dim = 32
    runtime_any._opponent_models = {}
    runtime_any._opponent_model_locks = {}

    class _AdvanceOnlyModel:
        def advance_seat_hidden(self, obs_tensor, actor_tensor, hidden_tensor):
            return hidden_tensor + 1.0

    class _RecordingHeuristicPolicy:
        def __init__(self) -> None:
            self.calls = 0

        def choose_actions_from_meta_batch(self, obs_rows, legal_ids, legal_offsets, legal_action_meta):
            self.calls += 1
            return np.asarray([10, 20], dtype=np.int64)

    class _FakePool:
        def choose_heuristic_public_actions_into(self, env_indices: np.ndarray, actions_out: np.ndarray) -> None:
            raise AssertionError("base native heuristic must not be used as a B4/control fallback")

    heuristic_policy = _RecordingHeuristicPolicy()
    runtime_any._opponent_heuristic_policies = {HEURISTIC_PUBLIC_CONTROL_POLICY_ID: heuristic_policy}
    actor = cast(
        Any,
        SimpleNamespace(
            model=_AdvanceOnlyModel(),
            compiled_model=None,
            env=SimpleNamespace(pool=_FakePool()),
            opponent_policy_id_by_env=np.asarray(
                [HEURISTIC_PUBLIC_CONTROL_POLICY_ID, HEURISTIC_PUBLIC_CONTROL_POLICY_ID, _MIRROR_OPPONENT_POLICY_ID],
                dtype=object,
            ),
            seat_hidden=torch.zeros((3, 2)),
            opponent_hidden=torch.zeros((3, 2)),
        ),
    )
    values_out = np.ones((3,), dtype=np.float32)
    actions_out = np.full((3,), 99, dtype=np.int64)
    logp_out = np.full((3,), -1.0, dtype=np.float32)

    QueueRuntime._apply_opponent_rows_ids(
        runtime,
        actor=actor,
        row_indices=np.asarray([0, 1], dtype=np.int64),
        obs_step=np.asarray([[1, 0], [2, 0], [3, 0]], dtype=np.float32),
        actor_step=np.asarray([1, 1, 0], dtype=np.int64),
        legal_ids=np.asarray([10, 11, 12, 20, 21], dtype=np.uint32),
        legal_offsets=np.asarray([0, 3, 5, 5], dtype=np.uint32),
        legal_action_meta=np.zeros((5, 4), dtype=np.uint16),
        logits_out=None,
        values_out=values_out,
        actions_out=actions_out,
        logp_out=logp_out,
        rng=np.random.default_rng(7),
        sample_actions=True,
    )

    assert heuristic_policy.calls == 1
    assert values_out.tolist() == [0.0, 0.0, 1.0]
    assert actions_out.tolist() == [10, 20, 99]
    assert logp_out.tolist() == [0.0, 0.0, -1.0]


def test_apply_opponent_rows_ids_falls_back_when_simulator_native_pool_hook_is_unavailable() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._device = torch.device("cpu")
    runtime_any._actor_amp_enabled = False
    runtime_any._fixed_opponent_backend = "simulator_native"
    runtime_any.action_dim = 32
    runtime_any._opponent_models = {}
    runtime_any._opponent_model_locks = {}

    class _AdvanceOnlyModel:
        def advance_seat_hidden(self, obs_tensor, actor_tensor, hidden_tensor):
            return hidden_tensor + 1.0

    class _RecordingHeuristicPolicy:
        def __init__(self) -> None:
            self.calls: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]] = []

        def choose_actions_from_meta_batch(self, obs_rows, legal_ids, legal_offsets, legal_action_meta):
            obs_array = np.asarray(obs_rows, dtype=np.int32)
            ids_array = np.asarray(legal_ids, dtype=np.uint32)
            offsets_array = np.asarray(legal_offsets, dtype=np.uint32)
            meta_array = None if legal_action_meta is None else np.asarray(legal_action_meta, dtype=np.uint16)
            self.calls.append(
                (
                    obs_array.copy(),
                    ids_array.copy(),
                    offsets_array.copy(),
                    None if meta_array is None else meta_array.copy(),
                )
            )
            return np.asarray(
                [int(ids_array[int(offsets_array[row_index])]) for row_index in range(offsets_array.shape[0] - 1)],
                dtype=np.int64,
            )

    heuristic_policy = _RecordingHeuristicPolicy()
    runtime_any._opponent_heuristic_policies = {HEURISTIC_PUBLIC_POLICY_ID: heuristic_policy}

    actor = cast(
        Any,
        SimpleNamespace(
            model=_AdvanceOnlyModel(),
            compiled_model=None,
            env=SimpleNamespace(pool=SimpleNamespace()),
            opponent_policy_id_by_env=np.asarray(
                [HEURISTIC_PUBLIC_POLICY_ID, HEURISTIC_PUBLIC_POLICY_ID, _MIRROR_OPPONENT_POLICY_ID],
                dtype=object,
            ),
            seat_hidden=torch.zeros((3, 2)),
            opponent_hidden=torch.zeros((3, 2)),
        ),
    )
    row_indices = np.asarray([0, 1], dtype=np.int64)
    obs_step = np.asarray([[1, 0], [2, 0], [3, 0]], dtype=np.float32)
    actor_step = np.asarray([1, 1, 0], dtype=np.int64)
    legal_ids = np.asarray([10, 11, 12, 20, 21], dtype=np.uint32)
    legal_offsets = np.asarray([0, 3, 5, 5], dtype=np.uint32)
    legal_action_meta = np.asarray(
        [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [2, 0, 0, 0],
            [3, 0, 0, 0],
            [4, 0, 0, 0],
        ],
        dtype=np.uint16,
    )
    values_out = np.ones((3,), dtype=np.float32)
    actions_out = np.full((3,), 99, dtype=np.int64)
    logp_out = np.full((3,), -1.0, dtype=np.float32)

    QueueRuntime._apply_opponent_rows_ids(
        runtime,
        actor=actor,
        row_indices=row_indices,
        obs_step=obs_step,
        actor_step=actor_step,
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        legal_action_meta=legal_action_meta,
        logits_out=None,
        values_out=values_out,
        actions_out=actions_out,
        logp_out=logp_out,
        rng=np.random.default_rng(7),
        sample_actions=True,
    )

    assert len(heuristic_policy.calls) == 1
    call_obs, call_ids, call_offsets, call_meta = heuristic_policy.calls[0]
    assert np.array_equal(call_obs, np.asarray([[1, 0], [2, 0]], dtype=np.int32))
    assert np.array_equal(call_ids, np.asarray([10, 11, 12, 20, 21], dtype=np.uint32))
    assert np.array_equal(call_offsets, np.asarray([0, 3, 5], dtype=np.uint32))
    assert call_meta is not None
    assert np.array_equal(call_meta, legal_action_meta)
    assert actor.seat_hidden[0].tolist() == [1.0, 1.0]
    assert actor.seat_hidden[1].tolist() == [1.0, 1.0]
    assert values_out.tolist() == [0.0, 0.0, 1.0]
    assert actions_out.tolist() == [10, 20, 99]
    assert logp_out.tolist() == [0.0, 0.0, -1.0]
