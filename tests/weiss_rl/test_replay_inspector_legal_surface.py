from __future__ import annotations

import numpy as np
import pytest
from weiss_rl.config import load_stack_config
from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.eval.policies.set import HEURISTIC_PUBLIC_POLICY_ID
from weiss_rl.model import GLOBAL_ACTION_SPACE_SIZE
from weiss_rl.replay.bundles import ReplayStep
from weiss_rl.replay.inspection_policy_execution import (
    policy_action_surface_batch_and_ids,
)
from weiss_rl.replay.inspection_policy_loading import LoadedReplayPolicy
from weiss_rl.replay.inspection_step_diffs import build_step_diff
from weiss_rl.replay.inspection_summaries import summarize_step_diffs
from weiss_rl.runtime.components.actions.legal_meta import action_catalog_indices

from .replay_inspector_test_support import REPO_ROOT, _heuristic_spec_bundle, _ids_batch


def test_replay_policy_surface_guard_filters_model_only_main_move_rows() -> None:
    stack = load_stack_config(REPO_ROOT / "configs" / "thesis" / "b1_noleague.yaml")

    class FakeModel:
        action_catalog = ActionCatalog.from_spec_bundle(_heuristic_spec_bundle())

    batch = _ids_batch(
        decision_id=10,
        actor=0,
        reward=0.0,
        terminated=False,
        truncated=False,
        engine_status=0,
        legal_ids=np.array([51, 402, 403], dtype=np.uint16),
        legal_action_meta=np.array([[2], [6], [6]], dtype=np.uint16),
        episode_seed=44,
        episode_key=555,
    )
    model_policy = LoadedReplayPolicy(
        spec="policy_a",
        label="policy_a",
        kind="model",
        weights_path=None,
        model=FakeModel(),  # type: ignore[arg-type]
    )
    heuristic_policy = LoadedReplayPolicy(
        spec=HEURISTIC_PUBLIC_POLICY_ID,
        label=HEURISTIC_PUBLIC_POLICY_ID,
        kind="heuristic_public",
        weights_path=None,
        heuristic_policy=object(),  # type: ignore[arg-type]
    )

    model_batch, model_legal_ids = policy_action_surface_batch_and_ids(
        policy=model_policy,
        stack=stack,
        batch=batch,
        legal_ids=np.array([51, 402, 403], dtype=np.uint32),
        pass_action_id=51,
    )
    heuristic_batch, heuristic_legal_ids = policy_action_surface_batch_and_ids(
        policy=heuristic_policy,
        stack=stack,
        batch=batch,
        legal_ids=np.array([51, 402, 403], dtype=np.uint32),
        pass_action_id=51,
    )

    assert model_batch is not batch
    assert model_legal_ids.tolist() == [51]
    assert heuristic_batch is batch
    assert heuristic_legal_ids.tolist() == [51, 402, 403]


def test_replay_policy_surface_guard_filters_model_only_pass_when_attack_is_available() -> None:
    stack = load_stack_config(REPO_ROOT / "configs" / "thesis" / "b1_noleague.yaml")

    class FakeModel:
        action_catalog = ActionCatalog.from_spec_bundle(_heuristic_spec_bundle())

    family_index, _attack_type_index = action_catalog_indices(FakeModel.action_catalog)
    pass_family = int(family_index["pass"])
    attack_family = int(family_index["attack"])
    batch = _ids_batch(
        decision_id=10,
        actor=0,
        reward=0.0,
        terminated=False,
        truncated=False,
        engine_status=0,
        legal_ids=np.array([51, 300, 301], dtype=np.uint16),
        legal_action_meta=np.array([[pass_family], [attack_family], [attack_family]], dtype=np.uint16),
        episode_seed=44,
        episode_key=555,
    )
    model_policy = LoadedReplayPolicy(
        spec="policy_a",
        label="policy_a",
        kind="model",
        weights_path=None,
        model=FakeModel(),  # type: ignore[arg-type]
    )
    heuristic_policy = LoadedReplayPolicy(
        spec=HEURISTIC_PUBLIC_POLICY_ID,
        label=HEURISTIC_PUBLIC_POLICY_ID,
        kind="heuristic_public",
        weights_path=None,
        heuristic_policy=object(),  # type: ignore[arg-type]
    )

    model_batch, model_legal_ids = policy_action_surface_batch_and_ids(
        policy=model_policy,
        stack=stack,
        batch=batch,
        legal_ids=np.array([51, 300, 301], dtype=np.uint32),
        pass_action_id=51,
    )
    heuristic_batch, heuristic_legal_ids = policy_action_surface_batch_and_ids(
        policy=heuristic_policy,
        stack=stack,
        batch=batch,
        legal_ids=np.array([51, 300, 301], dtype=np.uint32),
        pass_action_id=51,
    )

    assert model_batch is not batch
    assert model_legal_ids.tolist() == [300, 301]
    assert heuristic_batch is batch
    assert heuristic_legal_ids.tolist() == [51, 300, 301]


def test_step_diff_supports_per_policy_legal_surfaces() -> None:
    action_catalog = ActionCatalog.from_spec_bundle(_heuristic_spec_bundle())
    logits_a = np.zeros((GLOBAL_ACTION_SPACE_SIZE,), dtype=np.float32)
    logits_b = np.zeros((GLOBAL_ACTION_SPACE_SIZE,), dtype=np.float32)
    logits_b[402] = 5.0
    diff = build_step_diff(
        step_index=0,
        expected_step=ReplayStep(
            t=0,
            decision_id=10,
            actor=0,
            action=51,
            reward=0.0,
            terminated=False,
            truncated=False,
            engine_status=0,
            legal_fingerprint64=0,
        ),
        raw_legal_ids=np.array([51, 402], dtype=np.uint32),
        legal_ids_a=np.array([51], dtype=np.uint32),
        legal_ids_b=np.array([51, 402], dtype=np.uint32),
        logits_a=logits_a,
        logits_b=logits_b,
        top_actions=2,
        action_catalog=action_catalog,
    )

    assert diff["raw_legal_action_count"] == 2
    assert diff["policy_a_legal_action_count"] == 1
    assert diff["policy_b_legal_action_count"] == 2
    assert diff["policy_a_legal_surface_is_filtered"] is True
    assert diff["policy_b_legal_surface_is_filtered"] is False
    assert diff["policy_a_legal_surface_removed_action_count"] == 1
    assert diff["policy_b_legal_surface_removed_action_count"] == 0
    assert diff["policy_b_top_action_legal_for_policy_a"] is False
    assert diff["policy_a_top_action_legal_for_policy_b"] is True
    assert diff["policy_a_top_action"]["family"] == "pass"
    assert diff["policy_b_top_action"]["family"] == "main_move"
    assert diff["policy_a_probability_on_policy_b_top_action"] == 0.0
    assert diff["policy_a_rank_of_policy_b_top_action"] == 2

    summarized = summarize_step_diffs([diff], top_k=1)

    assert summarized["policy_a_legal_surface_filter_rate"] == pytest.approx(1.0)
    assert summarized["policy_b_legal_surface_filter_rate"] == pytest.approx(0.0)
    assert summarized["policy_a_mean_raw_minus_policy_a_legal_action_count"] == pytest.approx(1.0)
    assert summarized["policy_b_top_action_illegal_for_policy_a_rate"] == pytest.approx(1.0)
    assert summarized["raw_legal_action_count_percentiles"]["p50"] == pytest.approx(2.0)
    assert summarized["policy_a_legal_action_count_percentiles"]["p50"] == pytest.approx(1.0)
