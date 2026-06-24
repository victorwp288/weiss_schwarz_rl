"""Replay capture for simulator-backed evaluation games."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.config import StackConfig
from weiss_rl.envs.decision_env import DecisionBoundaryBatch, DecisionBoundaryEnv
from weiss_rl.envs.env_config import build_env_config_from_stack
from weiss_rl.eval.replay.simulator_replay_capture import (
    discover_raw_replay_path,
    replay_capture_visibility_mode,
    replay_sample_dir_name,
    should_capture_replay_sample,
)
from weiss_rl.eval.simulator.harness import ReplaySampleResult, ScheduledGame
from weiss_rl.replay.bundles import (
    ReplayRerunContract,
    ReplayStep,
    compute_legal_fingerprint64,
    make_replay_bundle_meta,
    write_replay_bundle,
)
from weiss_rl.replay.runner import verify_replay_bundle


@dataclass(slots=True)
class ReplayCaptureState:
    raw_dir: Path
    before_raw_paths: set[Path]
    simulator_episode_key: int | bytes | None = None
    steps: list[ReplayStep] | None = None


class SimulatorReplayRecorder:
    """Owns deterministic replay capture and bundle verification for final eval."""

    def __init__(
        self,
        *,
        stack: StackConfig,
        artifact_layout: ArtifactLayout,
        run_id256_bytes: bytes,
        spec_hash256_bytes: bytes,
        capture_rate: float,
        capture_limit: int,
    ) -> None:
        self.stack = stack
        self.artifact_layout = artifact_layout
        self.run_id256_bytes = run_id256_bytes
        self.spec_hash256_bytes = spec_hash256_bytes
        self.capture_rate = float(capture_rate)
        self.capture_limit = int(capture_limit)
        self._capture_count = 0

    def maybe_start(
        self,
        *,
        env: DecisionBoundaryEnv,
        scheduled_game: ScheduledGame,
    ) -> ReplayCaptureState | None:
        if not self._should_capture(scheduled_game=scheduled_game):
            return None
        raw_dir = self.artifact_layout.replays_raw_dir / replay_sample_dir_name(scheduled_game=scheduled_game)
        raw_dir.mkdir(parents=True, exist_ok=True)
        enable_replay_sampling = getattr(env.pool, "enable_replay_sampling", None)
        before_paths = set(raw_dir.glob("*.wsr"))
        if callable(enable_replay_sampling):
            enable_replay_sampling(
                sample_rate=1.0,
                out_dir=raw_dir.as_posix(),
                compress=False,
                visibility_mode=replay_capture_visibility_mode(self.stack),
                store_actions=True,
            )
        self._capture_count += 1
        return ReplayCaptureState(raw_dir=raw_dir, before_raw_paths=before_paths, steps=[])

    def record_initial_batch(self, capture: ReplayCaptureState, *, batch: DecisionBoundaryBatch) -> None:
        capture.simulator_episode_key = int(np.asarray(batch.episode_key, dtype=np.uint64)[0])

    def record_step(
        self,
        capture: ReplayCaptureState,
        *,
        decision_id: int,
        actor: int,
        action: int,
        next_batch: DecisionBoundaryBatch,
        legal_ids: np.ndarray,
    ) -> None:
        capture.steps = capture.steps or []
        capture.steps.append(
            ReplayStep(
                t=len(capture.steps),
                decision_id=int(decision_id),
                actor=int(actor),
                action=int(action),
                reward=float(np.asarray(next_batch.reward, dtype=np.float32)[0]),
                terminated=bool(np.asarray(next_batch.terminated, dtype=np.bool_)[0]),
                truncated=bool(np.asarray(next_batch.truncated, dtype=np.bool_)[0]),
                engine_status=int(np.asarray(next_batch.engine_status, dtype=np.int64)[0]),
                legal_fingerprint64=compute_legal_fingerprint64(
                    spec_hash256=self.spec_hash256_bytes,
                    decision_id=int(decision_id),
                    legal_ids=legal_ids,
                ),
            )
        )

    def finish(
        self,
        *,
        scheduled_game: ScheduledGame,
        capture: ReplayCaptureState,
    ) -> ReplaySampleResult:
        raw_replay_path = discover_raw_replay_path(
            raw_dir=capture.raw_dir,
            before_raw_paths=capture.before_raw_paths,
        )
        rerun_contract = self._rerun_contract(scheduled_game=scheduled_game)
        meta = make_replay_bundle_meta(
            simulator_episode_key=capture.simulator_episode_key,
            run_id256=self.run_id256_bytes,
            spec_hash256=self.spec_hash256_bytes,
            actor_id=0,
            env_id=0,
            episode_index=int(scheduled_game.episode_index),
            episode_seed64=int(scheduled_game.episode_seed),
            rerun_contract=rerun_contract,
        )
        bundle_path = write_replay_bundle(
            out_dir=self.artifact_layout.replays_bundles_dir,
            meta=meta,
            steps=list(capture.steps or ()),
        )
        report_path = self.artifact_layout.replays_verification_dir / f"replay_{meta.replay_key64:016x}.json"
        error: str | None = None
        matched = False
        verification_status = "pending"
        try:
            report = verify_replay_bundle(bundle_path=bundle_path, report_path=report_path)
            matched = bool(report.get("matched", False))
            verification_status = str(report.get("status", "unknown"))
            error = None if report.get("error") is None else str(report.get("error"))
        except Exception as exc:
            error = str(exc)
            verification_status = "failed"
            matched = False
        return ReplaySampleResult(
            pair_index=int(scheduled_game.pair_index),
            swap_index=int(scheduled_game.swap_index),
            episode_index=int(scheduled_game.episode_index),
            focal_policy_id=scheduled_game.focal_policy_id,
            opponent_policy_id=scheduled_game.opponent_policy_id,
            raw_replay_path=None if raw_replay_path is None else self.artifact_layout.relative(raw_replay_path),
            bundle_path=self.artifact_layout.relative(bundle_path),
            verification_report_path=self.artifact_layout.relative(report_path),
            verification_status=verification_status,
            replay_key64=f"{meta.replay_key64:016x}",
            matched=matched,
            error=error,
        )

    def _should_capture(self, *, scheduled_game: ScheduledGame) -> bool:
        return should_capture_replay_sample(
            scheduled_game=scheduled_game,
            capture_rate=self.capture_rate,
            capture_count=self._capture_count,
            capture_limit=self.capture_limit,
        )

    def _rerun_contract(self, *, scheduled_game: ScheduledGame | None = None) -> ReplayRerunContract:
        if self.stack.config.environment is None:
            raise RuntimeError("stack config is missing environment config")
        env_config = build_env_config_from_stack(
            self.stack,
            seed=0,
            deck=None if scheduled_game is None else scheduled_game.seat0_deck,
            opponent_deck=None if scheduled_game is None else scheduled_game.seat1_deck,
        )
        return ReplayRerunContract(
            version=2,
            observation_visibility=str(env_config["observation_visibility"]),
            max_decisions=int(env_config["max_decisions"]),
            max_ticks=int(env_config["max_ticks"]),
            reward_json=None if "reward_json" not in env_config else str(env_config["reward_json"]),
            curriculum_json=None if "curriculum_json" not in env_config else str(env_config["curriculum_json"]),
            deck=None if "deck" not in env_config else str(env_config["deck"]),
            opponent_deck=None if "opponent_deck" not in env_config else str(env_config["opponent_deck"]),
        )
