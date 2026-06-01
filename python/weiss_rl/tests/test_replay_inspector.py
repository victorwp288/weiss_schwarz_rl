from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch

from weiss_rl.config import StackConfig, compute_config_hash256, load_stack_config
from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.envs.decision_env import DecisionBoundaryBatch
from weiss_rl.eval.policy_set import (
    HEURISTIC_PUBLIC_AGGRO_POLICY_ID,
    HEURISTIC_PUBLIC_CONTROL_POLICY_ID,
    HEURISTIC_PUBLIC_POLICY_ID,
)
from weiss_rl.league.registry import SnapshotRegistry
from weiss_rl.model import GLOBAL_ACTION_SPACE_SIZE, PolicyValueModel
from weiss_rl.replay.bundles import (
    ReplayRerunContract,
    ReplayStep,
    compute_legal_fingerprint64,
    make_replay_bundle_meta,
    write_replay_bundle,
)
from weiss_rl.replay.inspector import (
    LoadedReplayPolicy,
    _build_step_diff,
    _forward_policy,
    _opponent_context_index_for_policy,
    _policy_action_surface_batch_and_ids,
    _resolve_policy_weights_path,
    _summarize_step_diffs,
    format_replay_inspection_report,
    inspect_replay_bundle,
)
from weiss_rl.replay.inspector import (
    write_replay_inspection_report as facade_write_replay_inspection_report,
)
from weiss_rl.replay.inspector_report import (
    format_replay_inspection_report as report_format_replay_inspection_report,
)
from weiss_rl.replay.inspector_report import (
    write_replay_inspection_report,
)
from weiss_rl.runtime_components.legal_meta import action_catalog_indices
from weiss_rl.tests._config_paths import canonical_stack_config_path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_replay_inspector_report_helpers_are_package_owned() -> None:
    assert format_replay_inspection_report is report_format_replay_inspection_report
    assert facade_write_replay_inspection_report is write_replay_inspection_report
    assert report_format_replay_inspection_report.__module__ == "weiss_rl.replay.inspector_report"


class FakeReplayEnv:
    def __init__(
        self,
        initial_batch: DecisionBoundaryBatch,
        transitions: list[tuple[int, DecisionBoundaryBatch]],
    ) -> None:
        self._initial_batch = initial_batch
        self._transitions = list(transitions)
        self.actions: list[int] = []
        self.closed = False

    def reset(self, seed: int | None = None) -> DecisionBoundaryBatch:
        return self._initial_batch

    def step(self, actions: np.ndarray) -> DecisionBoundaryBatch:
        action = int(np.asarray(actions, dtype=np.int64)[0])
        self.actions.append(action)
        expected_action, next_batch = self._transitions.pop(0)
        assert action == expected_action
        return next_batch

    def close(self) -> None:
        self.closed = True


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


def test_inspect_replay_bundle_accepts_run_manifest_snapshot_config_hash(tmp_path: Path) -> None:
    stack = load_stack_config(canonical_stack_config_path())
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    manifest_config_hash = "cd" * 32
    (run_dir / "manifest.json").write_text(
        json.dumps({"config_hash256": manifest_config_hash}, indent=2) + "\n",
        encoding="utf-8",
    )
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
        logits={4: 1.0, 9: 0.0},
        config_hash256=manifest_config_hash,
    )
    policy_b_path = _write_policy_weights(
        run_dir=run_dir,
        stack=stack,
        policy_id="policy_b",
        observation_dim=4,
        logits={4: 0.0, 9: 1.0},
        config_hash256=manifest_config_hash,
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
                reward=1.0,
                terminated=True,
                truncated=False,
                engine_status=0,
                legal_fingerprint64=_fingerprint(decision_id=10, legal_ids=np.array([4, 9], dtype=np.uint16)),
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
                    decision_id=10,
                    actor=0,
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

    assert report["compared_steps"] == 1


def test_inspect_replay_bundle_rejects_unmatched_snapshot_config_hash(tmp_path: Path) -> None:
    stack = load_stack_config(canonical_stack_config_path())
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"config_hash256": "cd" * 32}, indent=2) + "\n",
        encoding="utf-8",
    )
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
        logits={4: 1.0, 9: 0.0},
        config_hash256="ef" * 32,
    )
    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="policy_a",
        update=1,
        weights_sha256="sha-a",
        path=policy_a_path.relative_to(run_dir).as_posix(),
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
                reward=1.0,
                terminated=True,
                truncated=False,
                engine_status=0,
                legal_fingerprint64=_fingerprint(decision_id=10, legal_ids=np.array([4, 9], dtype=np.uint16)),
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
        transitions=[],
    )

    with pytest.raises(RuntimeError, match="Snapshot config hash mismatch"):
        inspect_replay_bundle(
            bundle_path=bundle_path,
            stack=stack,
            run_dir=run_dir,
            snapshot_registry_path=registry_path,
            policy_a="policy_a",
            policy_b="policy_a",
            top_k=1,
            top_actions=2,
            env_factory=lambda observed_contract: _return_fake_env(observed_contract, contract, env),
        )


def test_inspect_replay_bundle_accepts_explicit_extra_snapshot_config_hash(tmp_path: Path) -> None:
    stack = load_stack_config(canonical_stack_config_path())
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"config_hash256": "cd" * 32}, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "spec_bundle.json").write_text(
        json.dumps({"observation": _typed_observation_spec(obs_len=4)}, indent=2) + "\n",
        encoding="utf-8",
    )
    registry_path = run_dir / "training" / "snapshots" / "registry.json"
    imported_config_hash = "ef" * 32

    policy_a_path = _write_policy_weights(
        run_dir=run_dir,
        stack=stack,
        policy_id="imported_seed",
        observation_dim=4,
        logits={4: 1.0, 9: 0.0},
        config_hash256=imported_config_hash,
    )
    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="imported_seed",
        update=0,
        weights_sha256="sha-imported",
        path=policy_a_path.relative_to(run_dir).as_posix(),
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
                reward=1.0,
                terminated=True,
                truncated=False,
                engine_status=0,
                legal_fingerprint64=_fingerprint(decision_id=10, legal_ids=np.array([4, 9], dtype=np.uint16)),
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
                    decision_id=10,
                    actor=0,
                    reward=1.0,
                    terminated=True,
                    truncated=False,
                    engine_status=0,
                    legal_ids=np.array([], dtype=np.uint16),
                    episode_seed=44,
                    episode_key=555,
                ),
            )
        ],
    )

    report = inspect_replay_bundle(
        bundle_path=bundle_path,
        stack=stack,
        run_dir=run_dir,
        snapshot_registry_path=registry_path,
        policy_a="imported_seed",
        policy_b="imported_seed",
        top_k=1,
        top_actions=2,
        env_factory=lambda observed_contract: _return_fake_env(observed_contract, contract, env),
        accepted_snapshot_config_hashes=[imported_config_hash],
    )

    assert report["compared_steps"] == 1


def test_inspect_replay_bundle_supports_heuristic_public_and_action_family_labels(tmp_path: Path) -> None:
    stack = load_stack_config(canonical_stack_config_path())
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "spec_bundle.json").write_text(
        json.dumps(_heuristic_spec_bundle(), indent=2) + "\n",
        encoding="utf-8",
    )
    registry_path = run_dir / "training" / "snapshots" / "registry.json"

    policy_a_path = _write_policy_weights(
        run_dir=run_dir,
        stack=stack,
        policy_id="policy_a",
        observation_dim=512,
        logits={51: 3.0, 472: 0.5, 473: 0.2, 474: 0.0},
        observation_spec=_heuristic_spec_bundle()["observation"],  # type: ignore[arg-type]
    )
    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="policy_a",
        update=1,
        weights_sha256="sha-a",
        path=policy_a_path.relative_to(run_dir).as_posix(),
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
                action=474,
                reward=1.0,
                terminated=True,
                truncated=False,
                engine_status=0,
                legal_fingerprint64=_fingerprint(
                    decision_id=10, legal_ids=np.array([51, 472, 473, 474], dtype=np.uint16)
                ),
            ),
        ],
    )

    obs = _heuristic_obs()
    obs[16] = 1
    obs[17] = 6
    obs[18] = 7
    obs[58] = 0
    obs[59] = 4
    obs[60] = 6
    _set_stage(obs, player_index=0, slot=0, occupied=True, power=5000, effective_soul=1)
    env = FakeReplayEnv(
        _ids_batch(
            decision_id=10,
            actor=0,
            reward=0.0,
            terminated=False,
            truncated=False,
            engine_status=0,
            legal_ids=np.array([51, 472, 473, 474], dtype=np.uint16),
            episode_seed=44,
            episode_key=555,
            obs=obs,
        ),
        transitions=[
            (
                474,
                _ids_batch(
                    decision_id=10,
                    actor=0,
                    reward=1.0,
                    terminated=True,
                    truncated=False,
                    engine_status=0,
                    legal_ids=np.array([], dtype=np.uint16),
                    episode_seed=44,
                    episode_key=555,
                    obs=obs,
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
        policy_b=HEURISTIC_PUBLIC_POLICY_ID,
        top_k=1,
        top_actions=3,
        env_factory=lambda observed_contract: _return_fake_env(observed_contract, contract, env),
    )

    assert report["policy_b"]["kind"] == "heuristic_public"
    assert report["top_differences"][0]["policy_a_top_action"]["family"] == "pass"
    assert report["top_differences"][0]["policy_b_top_action"]["family"] == "attack"
    assert report["top_differences"][0]["policy_b_top_action"]["attack_type"] == "direct"
    assert report["top_differences"][0]["policy_a_probability_on_policy_b_top_action"] == pytest.approx(
        0.0417437858,
        rel=1e-6,
    )
    assert report["top_differences"][0]["policy_a_probability_on_policy_b_top_action_family"] == pytest.approx(
        0.1615536310,
        rel=1e-6,
    )
    assert report["summary"]["policy_a_mean_probability_on_policy_b_top_action_family"] == pytest.approx(
        0.1615536310,
        rel=1e-6,
    )
    assert report["summary"]["policy_a_mean_family_probability_masses"][0]["family"] == "pass"
    assert report["top_differences"][0]["policy_a_rank_of_policy_b_top_action"] == 4
    assert report["summary"]["policy_a_matches_policy_b_top_action_rate"] == 0.0
    assert report["summary"]["policy_a_matches_policy_b_top_action_family_rate"] == 0.0
    assert report["summary"]["policy_a_top_action_mismatch_count"] == 1
    assert report["summary"]["policy_a_top_action_family_mismatch_count"] == 1
    assert report["summary"]["top_action_family_confusions"][0] == {
        "policy_b_family": "attack",
        "policy_a_family": "pass",
        "count": 1,
    }
    trajectory_summary = report["trajectory_summary"]
    assert trajectory_summary["recorded_family_counts"][0] == {"family": "attack", "count": 1}
    assert trajectory_summary["decision_kind_counts"] == [{"decision_kind": "0", "count": 1}]
    assert trajectory_summary["legal_family_presence_rates"][-2:] == [
        {"family": "attack", "rate": 1.0},
        {"family": "pass", "rate": 1.0},
    ]
    assert trajectory_summary["numeric_summaries"]["self_clock_count"]["mean"] == pytest.approx(6.0)
    assert trajectory_summary["numeric_summaries"]["self_hand_count"]["mean"] == pytest.approx(7.0)
    assert trajectory_summary["numeric_summaries"]["opponent_clock_count"]["mean"] == pytest.approx(4.0)
    assert trajectory_summary["numeric_summaries"]["self_stage_occupied_count"]["mean"] == pytest.approx(1.0)
    assert trajectory_summary["actor_summaries"][0]["actor"] == 0
    assert trajectory_summary["actor_summaries"][0]["recorded_family_counts"][0] == {"family": "attack", "count": 1}

    text_report = format_replay_inspection_report(report)
    assert "trajectory:" in text_report
    assert "attack->pass x1" in text_report
    assert "a474[attack, slot=0, attack_type=direct]" in text_report
    assert "family_match=False" in text_report


def test_inspect_replay_bundle_supports_all_heuristic_public_policy_ids(tmp_path: Path) -> None:
    stack = load_stack_config(canonical_stack_config_path())
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "spec_bundle.json").write_text(
        json.dumps(_heuristic_spec_bundle(), indent=2) + "\n",
        encoding="utf-8",
    )
    registry_path = run_dir / "training" / "snapshots" / "registry.json"

    policy_a_path = _write_policy_weights(
        run_dir=run_dir,
        stack=stack,
        policy_id="policy_a",
        observation_dim=512,
        logits={51: 1.0, 472: 0.5, 473: 0.2, 474: 0.0},
        observation_spec=_heuristic_spec_bundle()["observation"],  # type: ignore[arg-type]
    )
    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="policy_a",
        update=1,
        weights_sha256="sha-a",
        path=policy_a_path.relative_to(run_dir).as_posix(),
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
                action=474,
                reward=1.0,
                terminated=True,
                truncated=False,
                engine_status=0,
                legal_fingerprint64=_fingerprint(
                    decision_id=10, legal_ids=np.array([51, 472, 473, 474], dtype=np.uint16)
                ),
            ),
        ],
    )

    obs = _heuristic_obs()
    _set_stage(obs, player_index=0, slot=0, occupied=True, power=5000, effective_soul=1)
    heuristic_policy_ids = (
        HEURISTIC_PUBLIC_POLICY_ID,
        HEURISTIC_PUBLIC_AGGRO_POLICY_ID,
        HEURISTIC_PUBLIC_CONTROL_POLICY_ID,
    )

    for policy_id in heuristic_policy_ids:
        env = FakeReplayEnv(
            _ids_batch(
                decision_id=10,
                actor=0,
                reward=0.0,
                terminated=False,
                truncated=False,
                engine_status=0,
                legal_ids=np.array([51, 472, 473, 474], dtype=np.uint16),
                episode_seed=44,
                episode_key=555,
                obs=obs,
            ),
            transitions=[
                (
                    474,
                    _ids_batch(
                        decision_id=10,
                        actor=0,
                        reward=1.0,
                        terminated=True,
                        truncated=False,
                        engine_status=0,
                        legal_ids=np.array([], dtype=np.uint16),
                        episode_seed=44,
                        episode_key=555,
                        obs=obs,
                    ),
                ),
            ],
        )

        def env_factory(
            observed_contract: ReplayRerunContract,
            replay_env: FakeReplayEnv = env,
        ) -> FakeReplayEnv:
            return _return_fake_env(observed_contract, contract, replay_env)

        report = inspect_replay_bundle(
            bundle_path=bundle_path,
            stack=stack,
            run_dir=run_dir,
            snapshot_registry_path=registry_path,
            policy_a="policy_a",
            policy_b=policy_id,
            top_k=1,
            top_actions=2,
            env_factory=env_factory,
        )

        assert report["policy_b"]["kind"] == "heuristic_public"
        assert report["policy_b"]["spec"] == policy_id


def test_replay_policy_surface_guard_filters_model_only_main_move_rows() -> None:
    stack = load_stack_config(
        REPO_ROOT / "configs" / "thesis" / "ablations" / "final_eval_mainmoveguard_mulliganguard.yaml"
    )

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

    model_batch, model_legal_ids = _policy_action_surface_batch_and_ids(
        policy=model_policy,
        stack=stack,
        batch=batch,
        legal_ids=np.array([51, 402, 403], dtype=np.uint32),
        pass_action_id=51,
    )
    heuristic_batch, heuristic_legal_ids = _policy_action_surface_batch_and_ids(
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
    stack = load_stack_config(
        REPO_ROOT / "configs" / "thesis" / "ablations" / "final_eval_attackguard_mainmoveguard_mulliganguard.yaml"
    )

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

    model_batch, model_legal_ids = _policy_action_surface_batch_and_ids(
        policy=model_policy,
        stack=stack,
        batch=batch,
        legal_ids=np.array([51, 300, 301], dtype=np.uint32),
        pass_action_id=51,
    )
    heuristic_batch, heuristic_legal_ids = _policy_action_surface_batch_and_ids(
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


def test_replay_forward_policy_uses_packed_candidate_scoring_for_structured_models() -> None:
    class PackedOnlyModel:
        supports_legal_candidate_scoring = True
        supports_factorized_legal_policy = False

        def forward_packed_seat_aware(
            self,
            _obs,
            _seat,
            hidden,
            *,
            legal_actions,
            scoring_mode,
            opponent_context_index=None,
        ):
            assert scoring_mode == "learner"
            assert opponent_context_index is None
            assert np.asarray(legal_actions.ids).tolist() == [2, 5]
            return torch.tensor([4.0, 9.0]), torch.tensor([0.0]), hidden + 1.0

        def forward_seat_aware(self, *_args, **_kwargs):
            raise AssertionError("replay inspector must not use dense scoring for structured models")

    policy = LoadedReplayPolicy(
        spec="policy_a",
        label="policy_a",
        kind="model",
        weights_path=None,
        model=PackedOnlyModel(),  # type: ignore[arg-type]
    )
    batch = _ids_batch(
        decision_id=10,
        actor=0,
        reward=0.0,
        terminated=False,
        truncated=False,
        engine_status=0,
        legal_ids=np.array([2, 5], dtype=np.uint16),
        episode_seed=44,
        episode_key=555,
    )
    logits, next_hidden = _forward_policy(
        policy=policy,
        batch=batch,
        seat_hidden=torch.zeros((1,)),
        legal_ids=np.array([2, 5], dtype=np.uint32),
    )

    assert logits[2] == pytest.approx(4.0)
    assert logits[5] == pytest.approx(9.0)
    assert next_hidden is not None
    assert float(next_hidden.item()) == pytest.approx(1.0)


def test_replay_forward_policy_passes_opponent_context_index_to_packed_scoring() -> None:
    class PackedOnlyModel:
        supports_legal_candidate_scoring = True
        supports_factorized_legal_policy = False

        def forward_packed_seat_aware(
            self,
            _obs,
            _seat,
            hidden,
            *,
            legal_actions,
            scoring_mode,
            opponent_context_index=None,
        ):
            assert scoring_mode == "learner"
            assert opponent_context_index is not None
            assert opponent_context_index.detach().cpu().tolist() == [3]
            assert np.asarray(legal_actions.ids).tolist() == [2, 5]
            return torch.tensor([4.0, 9.0]), torch.tensor([0.0]), hidden + 1.0

        def forward_seat_aware(self, *_args, **_kwargs):
            raise AssertionError("replay inspector must not use dense scoring for structured models")

    policy = LoadedReplayPolicy(
        spec="policy_a",
        label="policy_a",
        kind="model",
        weights_path=None,
        model=PackedOnlyModel(),  # type: ignore[arg-type]
    )
    batch = _ids_batch(
        decision_id=10,
        actor=0,
        reward=0.0,
        terminated=False,
        truncated=False,
        engine_status=0,
        legal_ids=np.array([2, 5], dtype=np.uint16),
        episode_seed=44,
        episode_key=555,
    )
    logits, next_hidden = _forward_policy(
        policy=policy,
        batch=batch,
        seat_hidden=torch.zeros((1,)),
        legal_ids=np.array([2, 5], dtype=np.uint32),
        opponent_context_index=3,
    )

    assert logits[2] == pytest.approx(4.0)
    assert logits[5] == pytest.approx(9.0)
    assert next_hidden is not None
    assert float(next_hidden.item()) == pytest.approx(1.0)


def test_opponent_context_index_for_policy_can_require_nonzero() -> None:
    class FakeModel:
        def opponent_context_indices_for_policy_ids(self, policy_ids):
            return [7 if str(policy_ids[0]) == "known_policy" else 0]

    policy = LoadedReplayPolicy(
        spec="policy_a",
        label="policy_a",
        kind="model",
        weights_path=None,
        model=FakeModel(),  # type: ignore[arg-type]
    )

    assert (
        _opponent_context_index_for_policy(
            policy=policy,
            opponent_context_policy_id="known_policy",
            require_nonzero=True,
        )
        == 7
    )
    with pytest.raises(RuntimeError, match="has no opponent-context index"):
        _opponent_context_index_for_policy(
            policy=policy,
            opponent_context_policy_id="missing_policy",
            require_nonzero=True,
        )


def test_step_diff_supports_per_policy_legal_surfaces() -> None:
    action_catalog = ActionCatalog.from_spec_bundle(_heuristic_spec_bundle())
    logits_a = np.zeros((GLOBAL_ACTION_SPACE_SIZE,), dtype=np.float32)
    logits_b = np.zeros((GLOBAL_ACTION_SPACE_SIZE,), dtype=np.float32)
    logits_b[402] = 5.0
    diff = _build_step_diff(
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

    summarized = _summarize_step_diffs([diff], top_k=1)

    assert summarized["policy_a_legal_surface_filter_rate"] == pytest.approx(1.0)
    assert summarized["policy_b_legal_surface_filter_rate"] == pytest.approx(0.0)
    assert summarized["policy_a_mean_raw_minus_policy_a_legal_action_count"] == pytest.approx(1.0)
    assert summarized["policy_b_top_action_illegal_for_policy_a_rate"] == pytest.approx(1.0)
    assert summarized["raw_legal_action_count_percentiles"]["p50"] == pytest.approx(2.0)
    assert summarized["policy_a_legal_action_count_percentiles"]["p50"] == pytest.approx(1.0)


def test_step_diff_reports_top_logit_and_probability_margins() -> None:
    logits_a = np.zeros((GLOBAL_ACTION_SPACE_SIZE,), dtype=np.float32)
    logits_b = np.zeros((GLOBAL_ACTION_SPACE_SIZE,), dtype=np.float32)
    logits_a[1] = 3.0
    logits_a[2] = 2.25
    logits_a[3] = 2.0
    logits_b[2] = 5.0

    diff = _build_step_diff(
        step_index=0,
        expected_step=ReplayStep(
            t=0,
            decision_id=10,
            actor=0,
            action=1,
            reward=0.0,
            terminated=False,
            truncated=False,
            engine_status=0,
            legal_fingerprint64=0,
        ),
        raw_legal_ids=np.array([1, 2, 3], dtype=np.uint32),
        legal_ids_a=np.array([1, 2, 3], dtype=np.uint32),
        legal_ids_b=np.array([1, 2, 3], dtype=np.uint32),
        logits_a=logits_a,
        logits_b=logits_b,
        top_actions=2,
        action_catalog=None,
    )
    summarized = _summarize_step_diffs([diff], top_k=1)

    assert diff["policy_a_top_logit_margin"] == pytest.approx(0.75)
    assert diff["policy_a_gap_from_top_logit_to_policy_b_top_action"] == pytest.approx(0.75)
    assert diff["policy_a_top_probability_margin"] > 0.0
    assert summarized["policy_a_top_logit_margin_percentiles"]["p50"] == pytest.approx(0.75)
    assert summarized["policy_a_gap_from_top_logit_to_policy_b_top_action_percentiles"]["p50"] == pytest.approx(0.75)
    assert summarized["policy_a_probability_on_policy_b_top_action_percentiles"]["count"] == 1


def test_step_diff_reports_policy_b_top_family_margin_summaries() -> None:
    action_catalog = ActionCatalog.from_spec_bundle(_heuristic_spec_bundle())
    logits_a = np.zeros((GLOBAL_ACTION_SPACE_SIZE,), dtype=np.float32)
    logits_b = np.zeros((GLOBAL_ACTION_SPACE_SIZE,), dtype=np.float32)
    logits_a[51] = 3.0
    logits_a[472] = 1.0
    logits_a[473] = 2.0
    logits_a[474] = 1.5
    logits_b[473] = 5.0

    diff = _build_step_diff(
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
        raw_legal_ids=np.array([51, 472, 473, 474], dtype=np.uint32),
        legal_ids_a=np.array([51, 472, 473, 474], dtype=np.uint32),
        legal_ids_b=np.array([51, 472, 473, 474], dtype=np.uint32),
        logits_a=logits_a,
        logits_b=logits_b,
        top_actions=2,
        action_catalog=action_catalog,
    )
    summarized = _summarize_step_diffs([diff], top_k=1)

    assert diff["policy_a_top_action"]["family"] == "pass"
    assert diff["policy_b_top_action"]["family"] == "attack"
    assert diff["policy_a_policy_b_top_action_same_family_logit_margin"] == pytest.approx(0.5)
    assert summarized["policy_a_policy_b_top_action_same_family_logit_margin_percentiles"]["p50"] == pytest.approx(0.5)
    assert summarized["policy_b_top_family_summaries"][0]["family"] == "attack"
    assert summarized["policy_b_top_family_summaries"][0]["count"] == 1
    assert summarized["policy_b_top_family_summaries"][0][
        "policy_a_policy_b_top_action_same_family_logit_margin_percentiles"
    ]["p50"] == pytest.approx(0.5)
    assert summarized["policy_b_top_family_summaries"][0]["policy_a_matches_policy_b_top_action_family_rate"] == 0.0
    assert summarized["policy_b_top_family_summaries"][0]["policy_b_top_action_legal_for_policy_a_rate"] == 1.0
    assert summarized["policy_b_top_family_summaries"][0]["policy_a_legal_surface_filter_rate"] == 0.0


def test_resolve_policy_weights_path_prefers_run_dir_for_relative_specs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    cwd_dir = tmp_path / "cwd"
    run_dir.mkdir()
    cwd_dir.mkdir()

    relative_spec = Path("training/snapshots/policy_a/weights.pt")
    cwd_weights_path = cwd_dir / relative_spec
    cwd_weights_path.parent.mkdir(parents=True)
    cwd_weights_path.write_bytes(b"cwd")

    run_dir_weights_path = run_dir / relative_spec
    run_dir_weights_path.parent.mkdir(parents=True)
    run_dir_weights_path.write_bytes(b"run-dir")

    monkeypatch.chdir(cwd_dir)

    resolved_path, label = _resolve_policy_weights_path(
        spec=relative_spec.as_posix(),
        run_dir=run_dir,
        registry=None,
    )

    assert label == relative_spec.as_posix()
    assert resolved_path == run_dir_weights_path.resolve()


def test_resolve_policy_weights_path_accepts_imported_seed_wrapped_suffix_ids(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    actual_policy_id = "seed_outer_seed_inner_policy_000002"
    requested_policy_id = "seed_inner_policy_000002"

    weights_path = run_dir / "training" / "snapshots" / actual_policy_id / "weights.pt"
    weights_path.parent.mkdir(parents=True)
    weights_path.write_bytes(b"weights")

    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id=actual_policy_id,
        update=2,
        weights_sha256="sha",
        path=weights_path.relative_to(run_dir).as_posix(),
    )

    resolved_path, label = _resolve_policy_weights_path(
        spec=requested_policy_id,
        run_dir=run_dir,
        registry=registry,
    )

    assert label == actual_policy_id
    assert resolved_path == weights_path.resolve()


def test_replay_inspector_cli_main_supports_json_stdout_and_report_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module_path = REPO_ROOT / "python" / "scripts" / "replay_inspector.py"
    spec = importlib.util.spec_from_file_location("replay_inspector_script", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    report_path = tmp_path / "report.json"
    canned_report = {
        "bundle_path": "bundle.zip",
        "policy_a": {"label": "A", "weights_path": "a.pt"},
        "policy_b": {"label": "B", "weights_path": "b.pt"},
        "summary": {
            "compared_steps": 1,
            "top_k": 1,
            "max_total_variation": 0.5,
            "mean_total_variation": 0.5,
            "median_total_variation": 0.5,
            "max_abs_probability_delta": 0.5,
        },
        "top_differences": [],
        "compared_steps": 1,
    }
    captured_kwargs: dict[str, object] = {}

    def fake_inspect_replay_bundle(**_: object) -> dict[str, object]:
        captured_kwargs.update(_)
        return canned_report

    monkeypatch.setattr(module._impl, "inspect_replay_bundle", fake_inspect_replay_bundle)

    exit_code = module.main(
        [
            "--bundle",
            str(tmp_path / "bundle.zip"),
            "--stack-config",
            str(canonical_stack_config_path()),
            "--policy-a",
            "policy_a",
            "--policy-b",
            "policy_b",
            "--json",
            "--report-json",
            str(report_path),
            "--opponent-context-policy-id",
            "B2 HeuristicPublic",
            "--require-opponent-context-index",
        ]
    )

    assert exit_code == 0
    assert captured_kwargs["opponent_context_policy_id"] == "B2 HeuristicPublic"
    assert captured_kwargs["require_opponent_context_index"] is True
    stdout = capsys.readouterr().out
    assert json.loads(stdout) == canned_report
    assert json.loads(report_path.read_text(encoding="utf-8")) == canned_report


def _write_policy_weights(
    *,
    run_dir: Path,
    stack: StackConfig,
    policy_id: str,
    observation_dim: int,
    logits: dict[int, float],
    observation_spec: dict[str, object] | None = None,
    config_hash256: str | None = None,
) -> Path:
    weights_dir = run_dir / "training" / "snapshots" / policy_id
    weights_dir.mkdir(parents=True, exist_ok=True)
    model_config = stack.config.model
    assert model_config is not None
    model = PolicyValueModel(
        observation_dim=observation_dim,
        config=model_config,
        action_dim=GLOBAL_ACTION_SPACE_SIZE,
        observation_spec=_typed_observation_spec(obs_len=observation_dim)
        if observation_spec is None
        else observation_spec,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        for action_id, value in logits.items():
            cast(Any, model.policy_head).bias[action_id] = value
    weights_path = weights_dir / "weights.pt"
    torch.save(
        {
            "format": "minimal_train_snapshot_weights_v1",
            "policy_id": policy_id,
            "update": 1,
            "config_hash256": compute_config_hash256(stack) if config_hash256 is None else config_hash256,
            "model_state_dict": model.state_dict(),
        },
        weights_path,
    )
    return weights_path


def _typed_observation_spec(*, obs_len: int) -> dict[str, object]:
    return {
        "obs_encoding_version": 2,
        "dtype": "f32",
        "obs_len": obs_len,
        "self_first": True,
        "header_fields": [{"name": f"feature_{index}", "index": index} for index in range(obs_len)],
        "player_blocks": [],
        "tail_slices": [],
    }


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


def _heuristic_obs() -> np.ndarray:
    return np.zeros((512,), dtype=np.int32)


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


def _write_bundle(tmp_path: Path, *, contract: ReplayRerunContract, steps: list[ReplayStep]) -> Path:
    meta = make_replay_bundle_meta(
        simulator_episode_key=555,
        run_id256=b"r" * 32,
        spec_hash256=bytes.fromhex("ab" * 32),
        actor_id=1,
        env_id=2,
        episode_index=3,
        episode_seed64=44,
        rerun_contract=contract,
    )
    return write_replay_bundle(out_dir=tmp_path, meta=meta, steps=steps)


def _return_fake_env(
    observed_contract: ReplayRerunContract,
    expected_contract: ReplayRerunContract,
    env: FakeReplayEnv,
) -> FakeReplayEnv:
    assert observed_contract == expected_contract
    return env


def _fingerprint(*, decision_id: int, legal_ids: np.ndarray) -> int:
    return compute_legal_fingerprint64(
        spec_hash256=bytes.fromhex("ab" * 32),
        decision_id=decision_id,
        legal_ids=legal_ids,
    )


def _ids_batch(
    *,
    decision_id: int,
    actor: int,
    reward: float,
    terminated: bool,
    truncated: bool,
    engine_status: int,
    legal_ids: np.ndarray,
    episode_seed: int,
    episode_key: int,
    legal_action_meta: np.ndarray | None = None,
    obs: np.ndarray | None = None,
) -> DecisionBoundaryBatch:
    ids = np.asarray(legal_ids, dtype=np.uint32)
    return DecisionBoundaryBatch(
        obs=np.asarray(np.zeros((4,), dtype=np.int16) if obs is None else obs).reshape(1, -1),
        reward=np.array([reward], dtype=np.float32),
        terminated=np.array([terminated], dtype=np.bool_),
        truncated=np.array([truncated], dtype=np.bool_),
        to_play=np.array([actor], dtype=np.int32),
        actor=np.array([actor], dtype=np.int32),
        decision_id=np.array([decision_id], dtype=np.int64),
        engine_status=np.array([engine_status], dtype=np.uint8),
        decision_count=np.array([0], dtype=np.uint32),
        tick_count=np.array([0], dtype=np.uint32),
        episode_seed=np.array([episode_seed], dtype=np.uint64),
        episode_key=np.array([episode_key], dtype=np.uint64),
        ids_offsets=(ids, np.array([0, int(ids.size)], dtype=np.int32)),
        legal_action_meta=None if legal_action_meta is None else np.asarray(legal_action_meta, dtype=np.uint16),
    )
