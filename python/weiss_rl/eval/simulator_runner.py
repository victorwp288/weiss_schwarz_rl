"""Simulator-backed deterministic evaluation runner."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from weiss_rl.action_diagnostics import (
    ActionSummaryCounters,
    make_action_sequence_state,
    summarize_eval_action_counters,
    update_eval_action_counters,
)
from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.config import StackConfig
from weiss_rl.envs.decision_env import DecisionBoundaryBatch, DecisionBoundaryEnv
from weiss_rl.envs.pool_factory import build_env_config_from_stack, make_env_pool_from_config
from weiss_rl.eval.harness import (
    EvalGameRunner,
    GameResult,
    ReplaySampleResult,
    ScheduledGame,
    abort_on_engine_fault_eval,
    game_result_from_step,
    sample_action_pinned,
)
from weiss_rl.eval.heuristic_public import HeuristicPublicPolicy
from weiss_rl.eval.policy_set import HEURISTIC_PUBLIC_POLICY_ID, NO_LEAGUE_POLICY_ID, RANDOM_LEGAL_POLICY_ID
from weiss_rl.eval.rng_pcg32 import Pcg32XshRrV1
from weiss_rl.league.registry import SnapshotMeta, SnapshotRegistry
from weiss_rl.masking import assert_strictly_increasing_legal_ids
from weiss_rl.model import PolicyValueModel, build_policy_value_model
from weiss_rl.replay.bundles import (
    ReplayRerunContract,
    ReplayStep,
    compute_legal_fingerprint64,
    make_replay_bundle_meta,
    write_replay_bundle,
)
from weiss_rl.replay.runner import verify_replay_bundle
from weiss_rl.repro import canonical_json_bytes, stable_hash64

_LEGACY_B1_POLICY_ID = "b1_noleague_baseline"
_U64_DENOMINATOR = float(1 << 64)


@dataclass(frozen=True, slots=True)
class ResolvedEvalPolicy:
    policy_id: str
    kind: str
    source_run_dir: str | None = None
    snapshot_path: str | None = None
    model: PolicyValueModel | None = None
    heuristic_policy: HeuristicPublicPolicy | None = None

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "kind": self.kind,
            "source_run_dir": self.source_run_dir,
            "snapshot_path": self.snapshot_path,
        }


@dataclass(slots=True)
class _ReplayCaptureState:
    raw_dir: Path
    before_raw_paths: set[Path]
    simulator_episode_key: int | bytes | None = None
    steps: list[ReplayStep] | None = None


def resolve_eval_policies(
    *,
    stack: StackConfig,
    policy_ids: list[str],
    run_dir: Path,
    observation_dim: int,
    action_dim: int,
    spec_bundle: Mapping[str, object] | None = None,
    snapshot_registry_path: Path | None = None,
    b1_baseline_run_dir: Path | None = None,
) -> dict[str, ResolvedEvalPolicy]:
    registry_path = snapshot_registry_path or (
        ArtifactLayout.from_run_dir(run_dir).training_snapshots_dir / "registry.json"
    )
    registry_run_dir = Path(registry_path).resolve().parent.parent.parent
    registry = SnapshotRegistry.load(registry_path)
    snapshots_by_policy_id = {snapshot.policy_id: snapshot for snapshot in registry.snapshots}
    resolved: dict[str, ResolvedEvalPolicy] = {}

    for policy_id in policy_ids:
        if policy_id == RANDOM_LEGAL_POLICY_ID:
            resolved[policy_id] = ResolvedEvalPolicy(policy_id=policy_id, kind="random_legal")
            continue
        if policy_id == HEURISTIC_PUBLIC_POLICY_ID:
            if spec_bundle is None:
                raise RuntimeError("Resolving B2 HeuristicPublic requires the loaded simulator spec bundle")
            resolved[policy_id] = ResolvedEvalPolicy(
                policy_id=policy_id,
                kind="heuristic_public",
                heuristic_policy=HeuristicPublicPolicy.from_spec_bundle(spec_bundle),
            )
            continue
        if policy_id == NO_LEAGUE_POLICY_ID:
            resolved[policy_id] = _resolve_b1_policy(
                run_dir=run_dir,
                b1_baseline_run_dir=b1_baseline_run_dir,
                stack=stack,
                observation_dim=observation_dim,
                action_dim=action_dim,
                spec_bundle=spec_bundle,
            )
            continue

        snapshot = snapshots_by_policy_id.get(policy_id)
        if snapshot is None:
            raise FileNotFoundError(f"Could not resolve eval policy {policy_id!r} in snapshot registry {registry_path}")
        model = _load_snapshot_eval_model(
            run_dir=registry_run_dir,
            snapshot_path=snapshot.path,
            stack=stack,
            observation_dim=observation_dim,
            action_dim=action_dim,
            observation_spec=_observation_spec_from_bundle(spec_bundle),
            spec_bundle=spec_bundle,
        )
        resolved[policy_id] = ResolvedEvalPolicy(
            policy_id=policy_id,
            kind="snapshot_registry",
            source_run_dir=registry_run_dir.as_posix(),
            snapshot_path=snapshot.path,
            model=model,
        )

    return resolved


class SimulatorEvalRunner(EvalGameRunner):
    def __init__(
        self,
        *,
        stack: StackConfig,
        policies: Mapping[str, ResolvedEvalPolicy],
        artifact_layout: ArtifactLayout,
        run_id256: str,
        spec_hash256: str,
        action_dim: int,
        pass_action_id: int,
        require_sorted_legal_ids: bool,
        replay_capture_rate: float,
        regression_capture_count: int,
    ) -> None:
        self.stack = stack
        self.policies = dict(policies)
        self.artifact_layout = artifact_layout
        self.run_id256_bytes = bytes.fromhex(run_id256)
        self.spec_hash256_bytes = bytes.fromhex(spec_hash256)
        self.pass_action_id = int(pass_action_id)
        self.require_sorted_legal_ids = bool(require_sorted_legal_ids)
        self.replay_capture_rate = float(replay_capture_rate)
        self.regression_capture_count = int(regression_capture_count)
        self._capture_count = 0
        self._device = torch.device("cpu")
        self._baseline_logits = np.zeros((int(action_dim),), dtype=np.float32)

    def run_game(self, scheduled_game: ScheduledGame) -> GameResult:
        env = self._build_ids_eval_env(seed=scheduled_game.episode_seed)
        replay_capture = self._maybe_enable_replay_capture(env=env, scheduled_game=scheduled_game)
        seat_hidden = {
            seat: self._initial_hidden(scheduled_game.seat0_policy_id if seat == 0 else scheduled_game.seat1_policy_id)
            for seat in (0, 1)
        }
        seat_rngs = {seat: Pcg32XshRrV1(self._rng_seed(scheduled_game=scheduled_game, seat=seat)) for seat in (0, 1)}
        action_counters = ActionSummaryCounters()
        action_sequence_state = make_action_sequence_state(1)
        last_acting_seat: int | None = None

        try:
            batch = env.reset(seed=scheduled_game.episode_seed)
            self._abort_on_fault(batch=batch, scheduled_game=scheduled_game)
            if replay_capture is not None:
                replay_capture.simulator_episode_key = self._simulator_episode_key(batch)
            while True:
                if bool(batch.terminated[0]) or bool(batch.truncated[0]):
                    result = game_result_from_step(
                        batch,
                        env_index=0,
                        acting_seat=last_acting_seat,
                        episode_seed=scheduled_game.episode_seed,
                        max_decisions=getattr(env, "max_decisions", None),
                        max_ticks=getattr(env, "max_ticks", None),
                        max_no_progress_decisions=getattr(env, "max_no_progress_decisions", None),
                    )
                    result_payload = {
                        "episode_seed": result.episode_seed,
                        "terminated": result.terminated,
                        "truncated": result.truncated,
                        "winner_seat": result.winner_seat,
                        "engine_status": result.engine_status,
                        "decision_count": result.decision_count,
                        "tick_count": result.tick_count,
                        "no_progress_count": result.no_progress_count,
                        "termination_reason": result.termination_reason,
                        "simulator_episode_key": result.simulator_episode_key,
                        **summarize_eval_action_counters(action_counters),
                    }
                    if replay_capture is None:
                        return GameResult(**result_payload)
                    replay_sample = self._finalize_replay_capture(
                        scheduled_game=scheduled_game,
                        replay_capture=replay_capture,
                    )
                    result_payload["replay_sample"] = replay_sample
                    return GameResult(**result_payload)

                current_seat = int(batch.actor[0])
                current_policy_id = (
                    scheduled_game.seat0_policy_id if current_seat == 0 else scheduled_game.seat1_policy_id
                )
                legal_ids = self._legal_ids_for_env_row(batch=batch)
                action, next_hidden = self._select_action(
                    batch=batch,
                    current_seat=current_seat,
                    current_policy_id=current_policy_id,
                    seat_hidden=seat_hidden[current_seat],
                    rng=seat_rngs[current_seat],
                    legal_ids=legal_ids,
                )
                update_eval_action_counters(
                    counters=action_counters,
                    state=action_sequence_state,
                    action=int(action),
                    legal_ids=legal_ids,
                    pass_action_id=self.pass_action_id,
                )
                decision_id = int(np.asarray(batch.decision_id, dtype=np.int64)[0])
                last_acting_seat = current_seat
                next_batch = env.step(np.asarray([action], dtype=np.uint32))
                self._abort_on_fault(batch=next_batch, scheduled_game=scheduled_game)
                if replay_capture is not None:
                    replay_capture.steps = replay_capture.steps or []
                    replay_capture.steps.append(
                        ReplayStep(
                            t=len(replay_capture.steps),
                            decision_id=decision_id,
                            actor=current_seat,
                            action=int(action),
                            reward=float(np.asarray(next_batch.reward, dtype=np.float32)[0]),
                            terminated=bool(np.asarray(next_batch.terminated, dtype=np.bool_)[0]),
                            truncated=bool(np.asarray(next_batch.truncated, dtype=np.bool_)[0]),
                            engine_status=int(np.asarray(next_batch.engine_status, dtype=np.int64)[0]),
                            legal_fingerprint64=compute_legal_fingerprint64(
                                spec_hash256=self.spec_hash256_bytes,
                                decision_id=decision_id,
                                legal_ids=legal_ids,
                            ),
                        )
                    )
                seat_hidden[current_seat] = next_hidden
                batch = next_batch
        finally:
            env.close()

    def _build_ids_eval_env(self, *, seed: int) -> DecisionBoundaryEnv:
        env_config = build_env_config_from_stack(self.stack, seed=int(seed))
        pool, layout_name = make_env_pool_from_config(
            env_config,
            profile="fast",
            num_envs=1,
        )
        if layout_name != "i16_legal_ids":
            raise RuntimeError(
                f"Pinned evaluation requires ids-based legality for deterministic CPU sampling, got {layout_name!r}."
            )
        max_no_progress_decisions = None
        curriculum = self.stack.config.curriculum
        if curriculum is not None:
            raw_limit = curriculum.simulator.get("max_no_progress_decisions")
            if raw_limit is not None:
                max_no_progress_decisions = int(raw_limit)
        return DecisionBoundaryEnv(
            pool,
            legality="ids_offsets",
            pass_action_id=self.pass_action_id,
            engine_status_policy="hard_fail",
            max_decisions=int(env_config["max_decisions"]),
            max_ticks=int(env_config["max_ticks"]),
            max_no_progress_decisions=max_no_progress_decisions,
        )

    def _select_action(
        self,
        *,
        batch: DecisionBoundaryBatch,
        current_seat: int,
        current_policy_id: str,
        seat_hidden: torch.Tensor | None,
        rng: Pcg32XshRrV1,
        legal_ids: np.ndarray,
    ) -> tuple[int, torch.Tensor | None]:
        policy = self.policies.get(current_policy_id)
        if policy is None:
            raise RuntimeError(f"Missing resolved eval policy for {current_policy_id!r}")
        if policy.heuristic_policy is not None:
            action = policy.heuristic_policy.choose_action(
                np.asarray(batch.obs[0], dtype=np.float32),
                legal_ids,
            )
            return int(action), seat_hidden
        if policy.model is None:
            action, _logp = sample_action_pinned(
                self._baseline_logits,
                legal_ids,
                rng=rng,
            )
            return action, seat_hidden
        if seat_hidden is None:
            raise RuntimeError(f"Missing hidden state for eval policy {current_policy_id!r}")
        with torch.inference_mode():
            logits_tensor, _value_tensor, next_seat_hidden = policy.model.forward_seat_aware(
                torch.as_tensor(np.asarray(batch.obs, dtype=np.float32), device=self._device),
                torch.as_tensor([current_seat], device=self._device, dtype=torch.long),
                seat_hidden,
            )
        logits = logits_tensor[0].detach().cpu().numpy().astype(np.float32, copy=False)
        action, _logp = sample_action_pinned(
            logits,
            legal_ids,
            rng=rng,
        )
        return action, next_seat_hidden

    def _initial_hidden(self, policy_id: str) -> torch.Tensor | None:
        policy = self.policies.get(policy_id)
        if policy is None or policy.model is None:
            return None
        return policy.model.initial_seat_hidden(1, device=self._device)

    def _abort_on_fault(self, *, batch: DecisionBoundaryBatch, scheduled_game: ScheduledGame) -> None:
        matchup_dir = (
            self.artifact_layout.final_eval_matchups_dir
            / f"{scheduled_game.pair_index:04d}_{scheduled_game.swap_index:01d}_{scheduled_game.episode_seed:016x}"
        )
        abort_on_engine_fault_eval(
            run_dir=matchup_dir,
            engine_status=batch.engine_status,
            decision_id=batch.decision_id,
            episode_key=batch.episode_key,
            note="engine_status!=0 during canonical final eval",
        )

    def _legal_ids_for_env_row(self, *, batch: DecisionBoundaryBatch) -> np.ndarray:
        if batch.ids_offsets is None:
            raise RuntimeError("Pinned evaluation requires ids_offsets legality")
        legal_ids, legal_offsets = batch.ids_offsets
        row = np.asarray(legal_ids[int(legal_offsets[0]) : int(legal_offsets[1])], dtype=np.uint32)
        if self.require_sorted_legal_ids:
            assert_strictly_increasing_legal_ids(row)
        return row

    def _maybe_enable_replay_capture(
        self,
        *,
        env: DecisionBoundaryEnv,
        scheduled_game: ScheduledGame,
    ) -> _ReplayCaptureState | None:
        if not self._should_capture_replay(scheduled_game=scheduled_game):
            return None
        raw_dir = self.artifact_layout.replays_raw_dir / self._replay_sample_dir_name(scheduled_game=scheduled_game)
        raw_dir.mkdir(parents=True, exist_ok=True)
        enable_replay_sampling = getattr(env.pool, "enable_replay_sampling", None)
        before_paths = set(raw_dir.glob("*.wsr"))
        if callable(enable_replay_sampling):
            enable_replay_sampling(
                sample_rate=1.0,
                out_dir=raw_dir.as_posix(),
                compress=False,
                visibility_mode=self._replay_visibility_mode(),
                store_actions=True,
            )
        self._capture_count += 1
        return _ReplayCaptureState(raw_dir=raw_dir, before_raw_paths=before_paths, steps=[])

    def _finalize_replay_capture(
        self,
        *,
        scheduled_game: ScheduledGame,
        replay_capture: _ReplayCaptureState,
    ) -> ReplaySampleResult:
        raw_replay_path = self._discover_raw_replay_path(replay_capture=replay_capture)
        rerun_contract = self._replay_rerun_contract()
        meta = make_replay_bundle_meta(
            simulator_episode_key=replay_capture.simulator_episode_key,
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
            steps=list(replay_capture.steps or ()),
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

    def _discover_raw_replay_path(self, *, replay_capture: _ReplayCaptureState) -> Path | None:
        after_paths = set(replay_capture.raw_dir.glob("*.wsr"))
        new_paths = sorted(after_paths - replay_capture.before_raw_paths)
        if len(new_paths) == 1:
            return new_paths[0]
        if len(after_paths) == 1:
            return sorted(after_paths)[0]
        return None

    def _replay_rerun_contract(self) -> ReplayRerunContract:
        if self.stack.config.environment is None:
            raise RuntimeError("stack config is missing environment config")
        env_config = build_env_config_from_stack(self.stack, seed=0)
        return ReplayRerunContract(
            version=2,
            observation_visibility=str(env_config["observation_visibility"]),
            max_decisions=int(env_config["max_decisions"]),
            max_ticks=int(env_config["max_ticks"]),
            reward_json=None if "reward_json" not in env_config else str(env_config["reward_json"]),
            curriculum_json=None if "curriculum_json" not in env_config else str(env_config["curriculum_json"]),
        )

    def _rng_seed(self, *, scheduled_game: ScheduledGame, seat: int) -> int:
        payload = canonical_json_bytes(
            {
                "kind": "simulator_eval_rng_v1",
                "pair_index": int(scheduled_game.pair_index),
                "swap_index": int(scheduled_game.swap_index),
                "episode_seed": int(scheduled_game.episode_seed),
                "seat": int(seat),
                "seat_policy_id": (scheduled_game.seat0_policy_id if seat == 0 else scheduled_game.seat1_policy_id),
            }
        )
        return stable_hash64(payload)

    def _should_capture_replay(self, *, scheduled_game: ScheduledGame) -> bool:
        if self.replay_capture_rate <= 0.0:
            return False
        if self._capture_count >= self.regression_capture_count:
            return False
        capture_u64 = stable_hash64(
            canonical_json_bytes(
                {
                    "kind": "final_eval_replay_capture_v1",
                    "pair_index": int(scheduled_game.pair_index),
                    "swap_index": int(scheduled_game.swap_index),
                    "episode_index": int(scheduled_game.episode_index),
                    "episode_seed": int(scheduled_game.episode_seed),
                    "focal_policy_id": scheduled_game.focal_policy_id,
                    "opponent_policy_id": scheduled_game.opponent_policy_id,
                }
            )
        )
        return (capture_u64 / _U64_DENOMINATOR) < self.replay_capture_rate

    def _replay_visibility_mode(self) -> str:
        environment_config = self.stack.config.environment
        if environment_config is None:
            return "full"
        return "public" if str(environment_config.observation_visibility).strip().lower() == "public" else "full"

    def _replay_sample_dir_name(self, *, scheduled_game: ScheduledGame) -> str:
        payload = canonical_json_bytes(
            {
                "pair_index": int(scheduled_game.pair_index),
                "swap_index": int(scheduled_game.swap_index),
                "episode_index": int(scheduled_game.episode_index),
                "episode_seed": int(scheduled_game.episode_seed),
                "focal_policy_id": scheduled_game.focal_policy_id,
                "opponent_policy_id": scheduled_game.opponent_policy_id,
            }
        )
        return f"{scheduled_game.pair_index:04d}_{scheduled_game.swap_index:01d}_{stable_hash64(payload):016x}"

    def _simulator_episode_key(self, batch: DecisionBoundaryBatch) -> int | None:
        return int(np.asarray(batch.episode_key, dtype=np.uint64)[0])


def _resolve_b1_policy(
    *,
    run_dir: Path,
    b1_baseline_run_dir: Path | None,
    stack: StackConfig,
    observation_dim: int,
    action_dim: int,
    spec_bundle: Mapping[str, object] | None,
) -> ResolvedEvalPolicy:
    for candidate_run_dir in _candidate_b1_run_dirs(run_dir=run_dir, b1_baseline_run_dir=b1_baseline_run_dir):
        snapshot = _find_b1_snapshot(candidate_run_dir)
        if snapshot is None:
            continue
        model = _load_snapshot_eval_model(
            run_dir=candidate_run_dir,
            snapshot_path=snapshot.path,
            stack=stack,
            observation_dim=observation_dim,
            action_dim=action_dim,
            observation_spec=_observation_spec_from_bundle(spec_bundle, run_dir=candidate_run_dir),
            spec_bundle=spec_bundle,
        )
        return ResolvedEvalPolicy(
            policy_id=NO_LEAGUE_POLICY_ID,
            kind="baseline_noleague",
            source_run_dir=candidate_run_dir.as_posix(),
            snapshot_path=snapshot.path,
            model=model,
        )
    raise FileNotFoundError(
        "Could not resolve the mandatory B1 NoLeague baseline. "
        "Pass --b1-baseline-run-dir or evaluate from a baseline_noleague run that persisted the canonical baseline snapshot."
    )


def _candidate_b1_run_dirs(*, run_dir: Path, b1_baseline_run_dir: Path | None) -> list[Path]:
    candidates: list[Path] = []
    if b1_baseline_run_dir is not None:
        candidates.append(Path(b1_baseline_run_dir))
    candidates.append(Path(run_dir))
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        key = resolved.as_posix()
        if key in seen:
            continue
        seen.add(key)
        unique.append(resolved)
    return unique


def _find_b1_snapshot(run_dir: Path) -> SnapshotMeta | None:
    layout = ArtifactLayout.from_run_dir(run_dir)
    registry_path = layout.training_snapshots_dir / "registry.json"
    if not registry_path.is_file():
        return None
    registry = SnapshotRegistry.load(registry_path)
    snapshots_by_id = {snapshot.policy_id: snapshot for snapshot in registry.snapshots}
    for policy_id in (NO_LEAGUE_POLICY_ID, _LEGACY_B1_POLICY_ID):
        snapshot = snapshots_by_id.get(policy_id)
        if snapshot is not None:
            return snapshot

    manifest_path = layout.manifest_path
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config_canonical = manifest.get("config_canonical", {})
        if isinstance(config_canonical, dict):
            experiment = config_canonical.get("experiment", {})
            role = ""
            if isinstance(experiment, Mapping):
                role = str(experiment.get("role", "")).strip()
            if role == "baseline_noleague":
                return max(
                    registry.snapshots,
                    key=lambda snapshot: (int(snapshot.update), str(snapshot.policy_id)),
                    default=None,
                )
    return None


def _load_snapshot_eval_model(
    *,
    run_dir: Path,
    snapshot_path: str,
    stack: StackConfig,
    observation_dim: int,
    action_dim: int,
    observation_spec: Mapping[str, object] | None = None,
    spec_bundle: Mapping[str, object] | None = None,
) -> PolicyValueModel:
    payload = torch.load(run_dir / snapshot_path, map_location="cpu", weights_only=True)
    model_state_dict = payload.get("model_state_dict")
    if not isinstance(model_state_dict, dict):
        raise RuntimeError(f"Snapshot weights payload missing model_state_dict: {snapshot_path}")
    model_config = stack.config.model
    if model_config is None:
        raise RuntimeError("The locked stack is missing the model config block")
    eval_model = build_policy_value_model(
        observation_dim=observation_dim,
        config=model_config,
        action_dim=action_dim,
        observation_spec=observation_spec,
        spec_bundle=spec_bundle,
    ).to(torch.device("cpu"))
    eval_model.load_state_dict(model_state_dict)
    eval_model.eval()
    return eval_model


def _observation_spec_from_bundle(
    spec_bundle: Mapping[str, object] | None,
    *,
    run_dir: Path | None = None,
) -> Mapping[str, object] | None:
    if spec_bundle is not None:
        observation = spec_bundle.get("observation")
        if isinstance(observation, Mapping):
            return observation
    if run_dir is None:
        return None
    layout = ArtifactLayout.from_run_dir(run_dir)
    if not layout.spec_bundle_path.is_file():
        return None
    payload = json.loads(layout.spec_bundle_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"spec_bundle.json must contain an object: {layout.spec_bundle_path}")
    observation = payload.get("observation")
    if observation is None:
        return None
    if not isinstance(observation, Mapping):
        raise RuntimeError(f"spec_bundle.json observation payload must be an object: {layout.spec_bundle_path}")
    return observation
