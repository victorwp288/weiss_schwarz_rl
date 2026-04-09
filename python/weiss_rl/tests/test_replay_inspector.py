from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from weiss_rl.config import StackConfig, compute_config_hash256, load_stack_config
from weiss_rl.envs.decision_env import DecisionBoundaryBatch
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
    _resolve_policy_weights_path,
    format_replay_inspection_report,
    inspect_replay_bundle,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


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
    stack = load_stack_config(REPO_ROOT / "configs/rl_stack_locked.yaml")
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
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

    contract = ReplayRerunContract(version=1, observation_visibility="public", max_decisions=200, max_ticks=10_000)
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
    assert "policy_a" in text_report
    assert "policy_b" in text_report


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

    def fake_inspect_replay_bundle(**_: object) -> dict[str, object]:
        return canned_report

    monkeypatch.setattr(module, "inspect_replay_bundle", fake_inspect_replay_bundle)

    exit_code = module.main(
        [
            "--bundle",
            str(tmp_path / "bundle.zip"),
            "--stack-config",
            str(REPO_ROOT / "configs" / "rl_stack_locked.yaml"),
            "--policy-a",
            "policy_a",
            "--policy-b",
            "policy_b",
            "--json",
            "--report-json",
            str(report_path),
        ]
    )

    assert exit_code == 0
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
) -> Path:
    weights_dir = run_dir / "training" / "snapshots" / policy_id
    weights_dir.mkdir(parents=True, exist_ok=True)
    model_config = stack.config.model
    assert model_config is not None
    model = PolicyValueModel(observation_dim=observation_dim, config=model_config, action_dim=GLOBAL_ACTION_SPACE_SIZE)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        for action_id, value in logits.items():
            model.policy_head.bias[action_id] = value
    weights_path = weights_dir / "weights.pt"
    torch.save(
        {
            "format": "minimal_train_snapshot_weights_v1",
            "policy_id": policy_id,
            "update": 1,
            "config_hash256": compute_config_hash256(stack),
            "model_state_dict": model.state_dict(),
        },
        weights_path,
    )
    return weights_path


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
) -> DecisionBoundaryBatch:
    ids = np.asarray(legal_ids, dtype=np.uint32)
    return DecisionBoundaryBatch(
        obs=np.zeros((1, 4), dtype=np.int16),
        reward=np.array([reward], dtype=np.float32),
        terminated=np.array([terminated], dtype=np.bool_),
        truncated=np.array([truncated], dtype=np.bool_),
        to_play=np.array([actor], dtype=np.int32),
        actor=np.array([actor], dtype=np.int32),
        decision_id=np.array([decision_id], dtype=np.int64),
        engine_status=np.array([engine_status], dtype=np.uint8),
        episode_seed=np.array([episode_seed], dtype=np.uint64),
        episode_key=np.array([episode_key], dtype=np.uint64),
        ids_offsets=(ids, np.array([0, int(ids.size)], dtype=np.int32)),
    )
