from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from weiss_rl.config import load_stack_config
from weiss_rl.league.registry import SnapshotRegistry
from weiss_rl.replay.bundles import ReplayRerunContract, ReplayStep
from weiss_rl.replay.inspector import inspect_replay_bundle
from weiss_rl.replay.inspector_report import format_replay_inspection_report

from ._config_paths import canonical_stack_config_path
from .replay_inspector_test_support import (
    FakeReplayEnv,
    _fingerprint,
    _ids_batch,
    _return_fake_env,
    _typed_observation_spec,
    _write_bundle,
    _write_policy_weights,
)


def test_inspect_replay_bundle_compares_policy_distributions_and_ranks_top_diffs(tmp_path: Path) -> None:
    stack = load_stack_config(canonical_stack_config_path())
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "spec_bundle.json").write_text(
        json.dumps({"observation": _typed_observation_spec(obs_len=4)}, indent=2) + "\n",
        encoding="utf-8",
    )
    registry_path = run_dir / "training" / "snapshots" / "registry.json"

    policy_a_path = _write_policy_weights(
        run_dir=run_dir,
        stack=stack,
        policy_id="policy_a",
        observation_dim=4,
        logits={4: 3.0, 9: 0.0, 5: 0.5, 2: 0.0},
    )
    policy_b_path = _write_policy_weights(
        run_dir=run_dir,
        stack=stack,
        policy_id="policy_b",
        observation_dim=4,
        logits={4: 0.0, 9: 3.0, 5: 0.5, 2: 0.0},
    )
    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="policy_a",
        update=1,
        weights_sha256="sha-a",
        path=policy_a_path.relative_to(run_dir).as_posix(),
    )
    registry.add_snapshot(
        policy_id="policy_b",
        update=2,
        weights_sha256="sha-b",
        path=policy_b_path.relative_to(run_dir).as_posix(),
    )
    registry.save(registry_path)

    contract = ReplayRerunContract(version=2, observation_visibility="public", max_decisions=200, max_ticks=10_000)
    bundle_path = _write_bundle(
        tmp_path,
        contract=contract,
        steps=[
            ReplayStep(
                t=0,
                decision_id=10,
                actor=0,
                action=4,
                reward=0.1,
                terminated=False,
                truncated=False,
                engine_status=0,
                legal_fingerprint64=_fingerprint(decision_id=10, legal_ids=np.array([4, 9], dtype=np.uint16)),
            ),
            ReplayStep(
                t=1,
                decision_id=11,
                actor=1,
                action=5,
                reward=1.0,
                terminated=True,
                truncated=False,
                engine_status=0,
                legal_fingerprint64=_fingerprint(decision_id=11, legal_ids=np.array([2, 5], dtype=np.uint16)),
            ),
        ],
    )
    env = FakeReplayEnv(
        _ids_batch(
            decision_id=10,
            actor=0,
            reward=0.0,
            terminated=False,
            truncated=False,
            engine_status=0,
            legal_ids=np.array([4, 9], dtype=np.uint16),
            episode_seed=44,
            episode_key=555,
        ),
        transitions=[
            (
                4,
                _ids_batch(
                    decision_id=11,
                    actor=1,
                    reward=0.1,
                    terminated=False,
                    truncated=False,
                    engine_status=0,
                    legal_ids=np.array([2, 5], dtype=np.uint16),
                    episode_seed=44,
                    episode_key=555,
                ),
            ),
            (
                5,
                _ids_batch(
                    decision_id=11,
                    actor=1,
                    reward=1.0,
                    terminated=True,
                    truncated=False,
                    engine_status=0,
                    legal_ids=np.array([], dtype=np.uint16),
                    episode_seed=44,
                    episode_key=555,
                ),
            ),
        ],
    )

    report = inspect_replay_bundle(
        bundle_path=bundle_path,
        stack=stack,
        run_dir=run_dir,
        snapshot_registry_path=registry_path,
        policy_a="policy_a",
        policy_b="policy_b",
        top_k=1,
        top_actions=2,
        env_factory=lambda observed_contract: _return_fake_env(observed_contract, contract, env),
    )

    assert report["compared_steps"] == 2
    assert report["summary"]["compared_steps"] == 2
    actor_summaries = report["summary"]["actor_summaries"]
    assert [item["actor"] for item in actor_summaries] == [0, 1]
    assert [item["compared_steps"] for item in actor_summaries] == [1, 1]
    assert actor_summaries[0]["mean_total_variation"] == pytest.approx(0.90514825, rel=1e-6)
    assert actor_summaries[0]["policy_a_matches_policy_b_top_action_rate"] == 0.0
    assert actor_summaries[0]["policy_a_mean_probability_on_policy_b_top_action"] == pytest.approx(
        0.047425873, rel=1e-6
    )
    assert actor_summaries[1]["mean_total_variation"] == 0.0
    assert actor_summaries[1]["policy_a_matches_policy_b_top_action_rate"] == 1.0
    assert actor_summaries[1]["policy_a_mean_probability_on_policy_b_top_action"] == pytest.approx(0.62245935, rel=1e-6)
    assert actor_summaries[0]["top_action_family_confusions"][0]["count"] == 1
    assert actor_summaries[1]["policy_b_top_family_summaries"][0]["count"] == 1
    assert report["top_differences"][0]["step_index"] == 0
    assert report["top_differences"][0]["policy_a_top_action"]["action"] == 4
    assert report["top_differences"][0]["policy_b_top_action"]["action"] == 9
    assert report["top_differences"][0]["total_variation"] == pytest.approx(0.90514825, rel=1e-6)
    assert report["top_differences"][0]["top_action_deltas"][0]["action"] == 9
    assert report["policy_a"]["weights_path"].endswith("training/snapshots/policy_a/weights.pt")
    assert env.actions == [4, 5]
    assert env.closed is True

    text_report = format_replay_inspection_report(report)
    assert "Replay inspector" in text_report
    assert "step=0 decision_id=10 actor=0" in text_report
    assert "actor_summaries: actor=0 steps=1" in text_report
    assert "policy_a" in text_report
    assert "policy_b" in text_report
