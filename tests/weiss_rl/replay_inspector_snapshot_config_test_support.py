"""Shared fixtures for replay-inspector snapshot config-hash tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from weiss_rl.config import StackConfig
from weiss_rl.league.registry import SnapshotRegistry
from weiss_rl.replay.bundles import ReplayRerunContract, ReplayStep

from tests.weiss_rl.replay_inspector_test_support import (
    FakeReplayEnv,
    _fingerprint,
    _ids_batch,
    _typed_observation_spec,
    _write_bundle,
    _write_policy_weights,
)


def write_replay_run_manifest_and_spec(*, run_dir: Path, config_hash256: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"config_hash256": config_hash256}, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "spec_bundle.json").write_text(
        json.dumps({"observation": _typed_observation_spec(obs_len=4)}, indent=2) + "\n",
        encoding="utf-8",
    )


def write_registry_policy(
    *,
    registry: SnapshotRegistry,
    run_dir: Path,
    stack: StackConfig,
    policy_id: str,
    update: int,
    logits: dict[int, float],
    config_hash256: str,
) -> Path:
    weights_path = _write_policy_weights(
        run_dir=run_dir,
        stack=stack,
        policy_id=policy_id,
        observation_dim=4,
        logits=logits,
        config_hash256=config_hash256,
    )
    registry.add_snapshot(
        policy_id=policy_id,
        update=update,
        weights_sha256=f"sha-{policy_id}",
        path=weights_path.relative_to(run_dir).as_posix(),
    )
    return weights_path


def write_single_step_bundle(tmp_path: Path) -> tuple[ReplayRerunContract, Path]:
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
    return contract, bundle_path


def single_step_fake_env(*, include_terminal_transition: bool) -> FakeReplayEnv:
    terminal_batch = _ids_batch(
        decision_id=10,
        actor=0,
        reward=1.0,
        terminated=True,
        truncated=False,
        engine_status=0,
        legal_ids=np.array([], dtype=np.uint16),
        episode_seed=44,
        episode_key=555,
    )
    transitions: list[tuple[int, Any]] = [(4, terminal_batch)] if include_terminal_transition else []
    return FakeReplayEnv(
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
        transitions=transitions,
    )
