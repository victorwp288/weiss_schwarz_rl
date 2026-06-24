from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from weiss_rl.runtime import QueueRuntime
from weiss_rl.runtime.components.actors.actor_routing import policy_train_mask_for_actor, split_focal_actor_rows


def test_split_focal_actor_rows_requires_teacher_policy() -> None:
    with pytest.raises(RuntimeError, match="heuristic actor policy backend requires an initialized teacher policy"):
        split_focal_actor_rows(
            focal_indices=np.array([0], dtype=np.int64),
            rng=np.random.default_rng(1),
            teacher_policy_available=False,
            force_model_policy_lane=False,
            heuristic_fraction=1.0,
        )


def test_split_focal_actor_rows_force_model_lane_ignores_heuristic_fraction() -> None:
    model_rows, heuristic_rows = split_focal_actor_rows(
        focal_indices=np.array([0, 2, 4], dtype=np.int64),
        rng=np.random.default_rng(1),
        teacher_policy_available=True,
        force_model_policy_lane=True,
        heuristic_fraction=1.0,
    )

    assert model_rows.tolist() == [0, 2, 4]
    assert heuristic_rows.tolist() == []


def test_split_focal_actor_rows_handles_fraction_endpoints() -> None:
    focal = np.array([0, 2, 4], dtype=np.int64)

    model_rows, heuristic_rows = split_focal_actor_rows(
        focal_indices=focal,
        rng=np.random.default_rng(1),
        teacher_policy_available=True,
        force_model_policy_lane=False,
        heuristic_fraction=0.0,
    )
    assert model_rows.tolist() == [0, 2, 4]
    assert heuristic_rows.tolist() == []

    model_rows, heuristic_rows = split_focal_actor_rows(
        focal_indices=focal,
        rng=np.random.default_rng(1),
        teacher_policy_available=True,
        force_model_policy_lane=False,
        heuristic_fraction=1.0,
    )
    assert model_rows.tolist() == []
    assert heuristic_rows.tolist() == [0, 2, 4]


def test_policy_train_mask_for_actor_can_exclude_pure_heuristic_lane() -> None:
    mask = policy_train_mask_for_actor(
        focal_rows=np.array([True, False, True], dtype=np.bool_),
        train_on_heuristic_actor_rows=False,
        actor_policy_backend="heuristic_public",
        force_model_policy_lane=False,
        heuristic_fraction=1.0,
    )

    assert mask.tolist() == [False, False, False]


def test_policy_train_mask_for_actor_excludes_mixed_heuristic_lane_until_rows_are_attributed() -> None:
    mask = policy_train_mask_for_actor(
        focal_rows=np.array([True, False, True], dtype=np.bool_),
        train_on_heuristic_actor_rows=False,
        actor_policy_backend="heuristic_public",
        force_model_policy_lane=False,
        heuristic_fraction=0.25,
    )

    assert mask.tolist() == [False, False, False]


def test_policy_train_mask_for_actor_preserves_focal_rows_for_non_heuristic_cases() -> None:
    focal_rows = np.array([True, False, True], dtype=np.bool_)

    assert policy_train_mask_for_actor(
        focal_rows=focal_rows,
        train_on_heuristic_actor_rows=True,
        actor_policy_backend="heuristic_public",
        force_model_policy_lane=False,
        heuristic_fraction=1.0,
    ).tolist() == [True, False, True]
    assert policy_train_mask_for_actor(
        focal_rows=focal_rows,
        train_on_heuristic_actor_rows=False,
        actor_policy_backend="model",
        force_model_policy_lane=False,
        heuristic_fraction=1.0,
    ).tolist() == [True, False, True]
    assert policy_train_mask_for_actor(
        focal_rows=focal_rows,
        train_on_heuristic_actor_rows=False,
        actor_policy_backend="heuristic_public",
        force_model_policy_lane=True,
        heuristic_fraction=1.0,
    ).tolist() == [True, False, True]


def test_policy_train_mask_for_actor_includes_model_sampled_mirror_opponent_rows() -> None:
    mask = policy_train_mask_for_actor(
        focal_rows=np.array([True, False, False, True], dtype=np.bool_),
        train_on_heuristic_actor_rows=False,
        actor_policy_backend="model",
        force_model_policy_lane=False,
        heuristic_fraction=0.0,
        opponent_policy_id_by_env=np.array(
            ["latest_policy_mirror", "latest_policy_mirror", "policy_000001", "latest_policy_mirror"],
            dtype=object,
        ),
    )

    assert mask.tolist() == [True, True, False, True]


def test_policy_train_mask_for_actor_keeps_mirror_rows_when_focal_heuristic_rows_are_dropped() -> None:
    mask = policy_train_mask_for_actor(
        focal_rows=np.array([True, False, True, False], dtype=np.bool_),
        train_on_heuristic_actor_rows=False,
        actor_policy_backend="heuristic_public",
        force_model_policy_lane=False,
        heuristic_fraction=0.25,
        opponent_policy_id_by_env=np.array(
            ["latest_policy_mirror", "latest_policy_mirror", "latest_policy_mirror", "policy_000001"],
            dtype=object,
        ),
    )

    assert mask.tolist() == [False, True, False, False]


def test_trajectory_retention_mask_keeps_only_retained_model_opponents() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._trajectory_retention_enabled = True
    runtime_any._trajectory_retention_policy_ids = ("seed", "missing_model")
    runtime_any._trajectory_retention_sources = ()
    runtime_any._opponent_models = {"seed": object(), "other": object()}
    actor = SimpleNamespace(
        opponent_policy_id_by_env=np.array(["seed", "seed", "other", "missing_model"], dtype=object),
    )

    mask = QueueRuntime._trajectory_retention_mask_for_actor(
        runtime,
        actor=actor,
        focal_rows=np.array([True, False, False, False], dtype=np.bool_),
    )

    assert mask is not None
    assert mask.tolist() == [False, True, False, False]


def test_trajectory_retention_mask_can_source_champions_and_warmup_snapshots() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._trajectory_retention_enabled = True
    runtime_any._trajectory_retention_policy_ids = ()
    runtime_any._trajectory_retention_sources = ("champions", "warmup_snapshots")
    runtime_any._opponent_models = {"champion": object(), "candidate": object(), "heuristic": object()}
    runtime_any._opponent_champion_ids = ("champion",)
    runtime_any._opponent_candidate_ids = ("candidate", "heuristic_missing_model")
    runtime_any._active_warmup_snapshot_mix_fraction = lambda: 0.25
    actor = SimpleNamespace(
        opponent_policy_id_by_env=np.array(["champion", "candidate", "heuristic", "candidate"], dtype=object),
    )

    mask = QueueRuntime._trajectory_retention_mask_for_actor(
        runtime,
        actor=actor,
        focal_rows=np.array([False, False, False, True], dtype=np.bool_),
    )

    assert mask is not None
    assert mask.tolist() == [True, True, False, False]
