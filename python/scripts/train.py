from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager, nullcontext, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.cli_banner import print_startup_banner
from weiss_rl.config import (
    StackConfig,
    apply_stack_overrides,
    canonical_config_dict,
    compute_config_hash256,
    load_stack_config,
    parse_override_tokens,
)
from weiss_rl.envs.decision_env import DecisionBoundaryBatch, DecisionBoundaryEnv
from weiss_rl.envs.pool_factory import build_env_config_from_stack, make_env_pool_from_config
from weiss_rl.eval import (
    DevEvalPolicySummary,
    PayoffFoldScheme,
    Pcg32XshRrV1,
    build_matchup_export,
    build_seat_advantage_diagnostics,
    game_result_from_step,
    run_seat_swapped_matchup,
    sample_action_pinned,
    write_matchup_diagnostics_json,
    write_matchup_summary_csv,
    write_matchup_summary_json,
)
from weiss_rl.eval.harness import ScheduledGame, abort_on_engine_fault_eval
from weiss_rl.eval.heuristic_public import HeuristicPublicPolicy
from weiss_rl.eval.policy_set import (
    heuristic_public_profile_name_for_policy_id,
    select_final_policy_set_deterministic_v1,
)
from weiss_rl.league import run_promotion_gate
from weiss_rl.league.registry import (
    REGISTRY_FILENAME,
    SNAPSHOT_METADATA_FILENAME,
    SNAPSHOT_WEIGHTS_FILENAME,
    SnapshotMeta,
    SnapshotRegistry,
    snapshot_weights_relpath,
)
from weiss_rl.learners.impala_learner import ImpalaLearner
from weiss_rl.learners.ppo_lite_learner import PpoLiteLearner
from weiss_rl.learners.vtrace import VTraceTargets, compute_vtrace_targets
from weiss_rl.manifest import (
    RunArtifacts,
    RunManifest,
    build_seed_file_manifest,
    default_run_dir_name,
    write_run_artifacts,
)
from weiss_rl.masking import assert_strictly_increasing_legal_ids, masked_logp_from_mask
from weiss_rl.model import PolicyValueModel, build_policy_value_model
from weiss_rl.repro import (
    canonical_json_bytes,
    compute_run_id64,
    compute_run_id256,
    hash_seed_file,
    parse_seed_file,
    stable_hash64,
)
from weiss_rl.runtime import QueueRuntime, QueueRuntimeMode, build_runtime_config, resolve_actor_device_layout
from weiss_rl.simulator_contract import SimulatorContract, load_verified_simulator_contract
from weiss_rl.spec import assert_spec_bundle_contract
from weiss_rl.tensorboard_logger import TensorBoardLogger, tensorboard_unavailable_reason
from weiss_rl.toy_public_demo import (
    PUBLIC_DEMO_MODE,
    public_demo_simulator_info,
    public_demo_spec_bundle,
    public_demo_spec_hash256,
    stage_public_demo_run,
)

_SHA256_HEX_LENGTH = 64
_U64_MASK = (1 << 64) - 1
_PROMOTION_GATE_RANDOMLEGAL_NAME = "B0 RandomLegal"
_PROMOTION_GATE_RANDOMLEGAL_POLICY_ID = "b0_randomlegal"
_PROMOTION_GATE_NOLEAGUE_BASELINE_NAME = "B1 NoLeague baseline"
_PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID = "b1_noleague_baseline"
_PROMOTION_GATE_NOLEAGUE_BASELINE_CHECKPOINT = "baseline_checkpoint.pt"
_LATEST_CHECKPOINT_FILENAME = "latest.pt"
_BEST_CHECKPOINT_FILENAME = "best.pt"
_CHECKPOINT_TRACKER_FILENAME = "checkpoint_tracker.json"
_IMPALA_ALGORITHMS = frozenset(
    {"impala_vtrace_gru", "impala_vtrace_ff", "structured_v2", "impala_vtrace_structured_v1"}
)
_PPO_ALGORITHMS = frozenset({"ppo_lite_masked_v1"})
_CONFIRMATORY_DEV_EVAL_MAX_PROB_SHORTFALL = 0.1
_CONFIRMATORY_DEV_EVAL_MAX_CI_EXCESS = 0.05
_GIT_COMMIT_HEX_LENGTH = 40


@dataclass(frozen=True, slots=True)
class MinimalRollout:
    obs: np.ndarray
    legal_mask: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    to_play_seat: np.ndarray
    behavior_logp: np.ndarray
    logits: np.ndarray
    values: np.ndarray
    bootstrap_obs: np.ndarray
    bootstrap_actor: np.ndarray


@dataclass(frozen=True, slots=True)
class TrainingPaths:
    training_dir: Path
    checkpoints_dir: Path
    logs_dir: Path
    snapshots_dir: Path
    tensorboard_dir: Path
    scalars_path: Path
    performance_log_path: Path
    latest_checkpoint_path: Path
    best_checkpoint_path: Path
    checkpoint_tracker_path: Path


@dataclass(frozen=True, slots=True)
class ResumeCheckpoint:
    checkpoint_path: Path
    update_count: int
    policy_version: int
    total_samples_processed: int


class _PeriodicDevEvalRunner:
    def __init__(
        self,
        *,
        stack: StackConfig,
        model: PolicyValueModel,
        opponent_policy_id: str,
        observation_dim: int,
        action_dim: int,
        pass_action_id: int,
        artifact_dir: Path,
        focal_policy_id: str,
        require_sorted_legal_ids: bool,
        opponent_model: PolicyValueModel | None = None,
        heuristic_policy: HeuristicPublicPolicy | None = None,
    ) -> None:
        self.stack = stack
        self.model = model
        self.opponent_policy_id = opponent_policy_id
        self.opponent_model = opponent_model
        self.heuristic_policy = heuristic_policy
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.pass_action_id = pass_action_id
        self.artifact_dir = artifact_dir
        self.focal_policy_id = focal_policy_id
        self.require_sorted_legal_ids = require_sorted_legal_ids
        self._baseline_logits = np.zeros((action_dim,), dtype=np.float32)
        self._device = torch.device("cpu")

    def run_game(self, scheduled_game: ScheduledGame):
        env = _build_ids_eval_env(
            self.stack,
            seed=scheduled_game.episode_seed,
            pass_action_id=self.pass_action_id,
        )
        focal_hidden = self.model.initial_seat_hidden(1, device=self._device)
        opponent_hidden = (
            None if self.opponent_model is None else self.opponent_model.initial_seat_hidden(1, device=self._device)
        )
        seat_rngs = {
            seat: Pcg32XshRrV1(_periodic_dev_eval_rng_seed(scheduled_game=scheduled_game, seat=seat)) for seat in (0, 1)
        }
        last_acting_seat: int | None = None

        try:
            batch = env.reset(seed=scheduled_game.episode_seed)
            self._abort_on_fault(batch)
            while True:
                if bool(batch.terminated[0]) or bool(batch.truncated[0]):
                    return game_result_from_step(
                        batch,
                        env_index=0,
                        acting_seat=last_acting_seat,
                        episode_seed=scheduled_game.episode_seed,
                        max_decisions=getattr(env, "max_decisions", None),
                        max_ticks=getattr(env, "max_ticks", None),
                        max_no_progress_decisions=getattr(env, "max_no_progress_decisions", None),
                    )

                current_seat = int(batch.actor[0])
                action, focal_hidden, opponent_hidden = self._select_action(
                    batch=batch,
                    scheduled_game=scheduled_game,
                    current_seat=current_seat,
                    focal_hidden=focal_hidden,
                    opponent_hidden=opponent_hidden,
                    rng=seat_rngs[current_seat],
                )
                last_acting_seat = current_seat
                batch = env.step(np.asarray([action], dtype=np.uint32))
                self._abort_on_fault(batch)
        finally:
            env.close()

    def _select_action(
        self,
        *,
        batch: DecisionBoundaryBatch,
        scheduled_game: ScheduledGame,
        current_seat: int,
        focal_hidden: torch.Tensor,
        opponent_hidden: torch.Tensor | None,
        rng: Pcg32XshRrV1,
    ) -> tuple[int, torch.Tensor, torch.Tensor | None]:
        legal_ids = _legal_ids_for_env_row(
            batch=batch,
            env_index=0,
            require_sorted=self.require_sorted_legal_ids,
        )
        current_policy_id = scheduled_game.seat0_policy_id if current_seat == 0 else scheduled_game.seat1_policy_id
        if current_policy_id == self.focal_policy_id:
            action, focal_hidden = self._sample_model_action(
                model=self.model,
                seat_hidden=focal_hidden,
                batch=batch,
                current_seat=current_seat,
                legal_ids=legal_ids,
                rng=rng,
            )
            return action, focal_hidden, opponent_hidden
        if self.opponent_model is not None and current_policy_id == self.opponent_policy_id:
            assert opponent_hidden is not None
            action, opponent_hidden = self._sample_model_action(
                model=self.opponent_model,
                seat_hidden=opponent_hidden,
                batch=batch,
                current_seat=current_seat,
                legal_ids=legal_ids,
                rng=rng,
            )
            return action, focal_hidden, opponent_hidden
        if self.heuristic_policy is not None and current_policy_id == self.opponent_policy_id:
            action = self.heuristic_policy.choose_action(
                np.asarray(batch.obs[0], dtype=np.float32),
                legal_ids,
            )
            return int(action), focal_hidden, opponent_hidden
        action, _ = sample_action_pinned(
            self._baseline_logits,
            legal_ids,
            rng=rng,
            pass_action_id=self.pass_action_id,
        )
        return action, focal_hidden, opponent_hidden

    def _sample_model_action(
        self,
        *,
        model: PolicyValueModel,
        seat_hidden: torch.Tensor,
        batch: DecisionBoundaryBatch,
        current_seat: int,
        legal_ids: np.ndarray,
        rng: Pcg32XshRrV1,
    ) -> tuple[int, torch.Tensor]:
        with torch.inference_mode():
            logits_tensor, _value_tensor, next_seat_hidden = model.forward_seat_aware(
                torch.as_tensor(np.asarray(batch.obs, dtype=np.float32), device=self._device),
                torch.as_tensor([current_seat], device=self._device, dtype=torch.long),
                seat_hidden,
                scoring_mode="learner",
            )
        logits = logits_tensor[0].detach().cpu().numpy().astype(np.float32, copy=False)
        action, _ = sample_action_pinned(
            logits,
            legal_ids,
            rng=rng,
            pass_action_id=self.pass_action_id,
        )
        return action, next_seat_hidden

    def _abort_on_fault(self, batch: DecisionBoundaryBatch) -> None:
        abort_on_engine_fault_eval(
            run_dir=self.artifact_dir,
            engine_status=batch.engine_status,
            decision_id=batch.decision_id,
            episode_key=batch.episode_key,
            note="engine_status!=0 during periodic dev eval",
        )


def _normalize_sha256(value: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != _SHA256_HEX_LENGTH:
        return ""
    if any(char not in "0123456789abcdef" for char in normalized):
        return ""
    return normalized


def _expected_sha256(value: str, *, flag_name: str) -> str:
    if not value.strip():
        return ""
    normalized = _normalize_sha256(value)
    if not normalized:
        raise ValueError(f"{flag_name} must be a 64-character lowercase or uppercase SHA-256 hex string")
    return normalized


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json_file(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_snapshot_artifact(
    *,
    snapshots_dir: Path,
    run_dir: Path,
    checkpoint_path: Path,
    policy_id: str,
    update: int,
    config_hash256: str,
    device: torch.device,
    model_state_dict: dict[str, Any],
) -> tuple[Path, str]:
    snapshot_dir = snapshots_dir / policy_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    weights_path = snapshot_dir / "weights.pt"
    weights_payload = {
        "format": "minimal_train_snapshot_weights_v1",
        "policy_id": policy_id,
        "update": int(update),
        "device": str(device),
        "config_hash256": config_hash256,
        "model_state_dict": model_state_dict,
    }
    torch.save(weights_payload, weights_path)
    weights_sha256 = _sha256_file(weights_path)

    _write_json_file(
        snapshot_dir / SNAPSHOT_METADATA_FILENAME,
        {
            "format": "minimal_train_snapshot_metadata_v1",
            "policy_id": policy_id,
            "update": int(update),
            "weights_path": snapshot_weights_relpath(policy_id),
            "weights_sha256": weights_sha256,
            "source_checkpoint_path": checkpoint_path.relative_to(run_dir).as_posix(),
        },
    )
    return weights_path, weights_sha256


def _sync_snapshot_registry_retention(stack: StackConfig, registry: SnapshotRegistry) -> None:
    league = stack.config.league
    if league is None:
        return
    registry.recent_size = int(league.snapshot_pool_recent_size)
    registry.champion_size = int(league.snapshot_pool_champion_size)


def _snapshot_artifact_dir_for_prune(
    *,
    training_paths: TrainingPaths,
    run_dir: Path,
    snapshot: SnapshotMeta,
) -> Path:
    snapshots_root = training_paths.snapshots_dir.resolve()
    weights_path = (run_dir / snapshot.path).resolve()
    try:
        weights_path.relative_to(snapshots_root)
    except ValueError as exc:
        raise RuntimeError(f"refusing to delete snapshot artifact outside {snapshots_root}: {snapshot.path}") from exc
    if weights_path.name != SNAPSHOT_WEIGHTS_FILENAME:
        raise RuntimeError(f"refusing to delete unexpected snapshot artifact path: {snapshot.path}")

    snapshot_dir = weights_path.parent
    try:
        snapshot_dir.relative_to(snapshots_root)
    except ValueError as exc:
        raise RuntimeError(f"refusing to delete snapshot directory outside {snapshots_root}: {snapshot_dir}") from exc
    if snapshot_dir == snapshots_root or snapshot_dir.name != snapshot.policy_id:
        raise RuntimeError(f"refusing to delete unexpected snapshot directory: {snapshot_dir}")
    return snapshot_dir


def _delete_pruned_snapshot_artifacts(
    *,
    training_paths: TrainingPaths,
    run_dir: Path,
    pruned_snapshots: list[SnapshotMeta],
) -> None:
    for snapshot in pruned_snapshots:
        snapshot_dir = _snapshot_artifact_dir_for_prune(
            training_paths=training_paths,
            run_dir=run_dir,
            snapshot=snapshot,
        )
        if snapshot_dir.exists():
            shutil.rmtree(snapshot_dir)


def _save_snapshot_registry_with_retention(
    *,
    stack: StackConfig,
    training_paths: TrainingPaths,
    run_dir: Path,
    registry: SnapshotRegistry,
) -> None:
    registry_path = training_paths.snapshots_dir / REGISTRY_FILENAME
    _sync_snapshot_registry_retention(stack, registry)
    pruned_snapshots = registry.prune()
    registry.save(registry_path)
    _delete_pruned_snapshot_artifacts(
        training_paths=training_paths,
        run_dir=run_dir,
        pruned_snapshots=pruned_snapshots,
    )


def _persist_snapshot_registry_entry(
    *,
    stack: StackConfig,
    training_paths: TrainingPaths,
    run_dir: Path,
    checkpoint_path: Path,
    model_state_dict: dict[str, Any],
    config_hash256: str,
    device: torch.device,
    update: int,
    policy_version: int,
) -> str:
    policy_id = f"policy_{int(policy_version):06d}"
    weights_path, weights_sha256 = _write_snapshot_artifact(
        snapshots_dir=training_paths.snapshots_dir,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        policy_id=policy_id,
        update=update,
        config_hash256=config_hash256,
        device=device,
        model_state_dict=model_state_dict,
    )

    registry_path = training_paths.snapshots_dir / REGISTRY_FILENAME
    reg = SnapshotRegistry.load(registry_path)
    _sync_snapshot_registry_retention(stack, reg)
    reg.add_snapshot(
        policy_id=policy_id,
        update=int(update),
        weights_sha256=weights_sha256,
        path=weights_path.relative_to(run_dir).as_posix(),
    )
    _save_snapshot_registry_with_retention(
        stack=stack,
        training_paths=training_paths,
        run_dir=run_dir,
        registry=reg,
    )
    return policy_id


def _require_matching_hash(*, flag_name: str, expected: str, actual: str) -> None:
    if expected and expected != actual:
        raise RuntimeError(f"{flag_name} mismatch: expected {expected}, observed {actual}")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_output(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
        cwd=_repo_root(),
    )
    return result.stdout.strip()


def _git_commit() -> str:
    override = str(os.environ.get("WEISS_RL_GIT_COMMIT", "")).strip().lower()
    if len(override) == _GIT_COMMIT_HEX_LENGTH and all(char in "0123456789abcdef" for char in override):
        return override
    try:
        return _git_output(["rev-parse", "HEAD"])
    except (OSError, subprocess.CalledProcessError):
        return ""


def _git_dirty() -> bool:
    try:
        return bool(_git_output(["status", "--short"]))
    except (OSError, subprocess.CalledProcessError):
        return False


def _start_nonce() -> int:
    return time.time_ns() & _U64_MASK


def _hardware_summary(
    learner_device: torch.device | str = "cpu",
    *,
    actor_device: torch.device | str = "cpu",
    actor_device_layout: Sequence[str] | None = None,
) -> dict[str, str | int]:
    learner_device_name = str(learner_device)
    payload: dict[str, str | int] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count() or 0,
        "learner_device": learner_device_name,
        "actor_device": str(actor_device),
    }
    if actor_device_layout:
        payload["actor_device_layout"] = ",".join(str(device_name) for device_name in actor_device_layout)
        payload["actor_device_unique_count"] = len(
            dict.fromkeys(str(device_name) for device_name in actor_device_layout)
        )
    return payload


def _manifest_actor_device_layout(
    *,
    stack: StackConfig,
    num_envs: int,
    unroll_length: int,
    profile: str,
    seed: int,
    pass_action_id: int,
    runtime_mode: QueueRuntimeMode,
    learner_device: torch.device,
) -> tuple[str, ...] | None:
    if stack.config.system is None or stack.config.training is None:
        return None
    runtime_config = build_runtime_config(
        stack=stack,
        num_envs=num_envs,
        unroll_length=unroll_length,
        profile=profile,
        seed=seed,
        pass_action_id=pass_action_id,
        runtime_mode=runtime_mode,
    )
    return tuple(
        str(device_name)
        for device_name in resolve_actor_device_layout(
            stack,
            actor_count=int(runtime_config.actor_count),
            learner_device=learner_device,
            prefer_process_collectors=True,
        )
    )


def _evaluation_pinning(stack: StackConfig) -> dict[str, str | bool]:
    if stack.config.evaluation is None:
        return {}
    evaluation = stack.config.evaluation
    return {
        "eval_device": evaluation.eval_device,
        "eval_sampling_algorithm": evaluation.eval_sampling_algorithm,
        "eval_inference_mode": evaluation.eval_inference_mode,
        "seat_swap": evaluation.seat_swap,
        "legal_fingerprint_version": evaluation.legal_fingerprint_checks.version,
        "legal_fingerprint_mismatch_policy": evaluation.legal_fingerprint_checks.mismatch_policy,
    }


def _manifest_source_path(path: Path, *, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON must contain an object at the top level")
    return payload


def _apply_training_flag_overrides(
    stack: StackConfig,
    *,
    enable_profile_timers: bool,
    enable_torch_profiler: bool,
) -> StackConfig:
    training_config = stack.config.training
    if training_config is None:
        return stack
    overrides: dict[str, Any] = {}
    if enable_profile_timers and not bool(training_config.profile_timers):
        overrides["training.profile_timers"] = True
    if enable_torch_profiler and not bool(training_config.torch_profiler):
        overrides["training.torch_profiler"] = True
    return apply_stack_overrides(stack, overrides)


def _experiment_role(stack: StackConfig) -> str:
    experiment = stack.config.experiment
    return "" if experiment is None else str(experiment.role).strip()


def _is_noleague_baseline_role(role: str) -> bool:
    return str(role).strip() == "baseline_noleague"


def _canonical_config_sections(config_canonical: Mapping[str, Any]) -> Mapping[str, Any]:
    config = config_canonical.get("config")
    return config if isinstance(config, Mapping) else config_canonical


def _role_from_config_canonical(config_canonical: Mapping[str, Any]) -> str:
    experiment = _canonical_config_sections(config_canonical).get("experiment", {})
    if isinstance(experiment, Mapping):
        role = str(experiment.get("role", "")).strip()
        if role:
            return role
    return ""


def _legacy_noleague_baseline_mode(config_canonical: Mapping[str, Any]) -> str:
    training_family = _canonical_config_sections(config_canonical).get("training_family_a", {})
    if isinstance(training_family, Mapping):
        return str(training_family.get("mode", "")).strip()
    return ""


def _config_marks_noleague_baseline(config_canonical: Mapping[str, Any]) -> bool:
    role = _role_from_config_canonical(config_canonical)
    if role:
        return _is_noleague_baseline_role(role)
    legacy_mode = _legacy_noleague_baseline_mode(config_canonical)
    if legacy_mode:
        return legacy_mode == "b1_no_league"
    return False


def _assert_noleague_baseline_config(config_canonical: Mapping[str, Any]) -> None:
    role = _role_from_config_canonical(config_canonical)
    if role:
        if not _is_noleague_baseline_role(role):
            raise RuntimeError(
                f"Imported B1 baseline must come from a dedicated baseline_noleague run, got experiment.role={role!r}"
            )
        return
    legacy_mode = _legacy_noleague_baseline_mode(config_canonical)
    if legacy_mode and legacy_mode != "b1_no_league":
        raise RuntimeError(
            "Imported B1 baseline must come from a dedicated baseline_noleague run, "
            f"got training_family_a.mode={legacy_mode!r}"
        )


def _read_optional_hash_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def _validate_imported_snapshot_contract(
    *,
    source_run_dir: Path,
    payload: dict[str, Any],
    expected_model_state_dict: dict[str, Any],
    expected_config_canonical: dict[str, Any] | None,
    expected_spec_hash256: str | None,
) -> None:
    source_layout = ArtifactLayout.from_run_dir(source_run_dir)
    manifest_path = source_layout.manifest_path
    source_manifest = (
        _load_json_object(manifest_path, label="imported B1 manifest") if manifest_path.is_file() else None
    )
    source_config_canonical = source_manifest.get("config_canonical") if isinstance(source_manifest, dict) else None
    if isinstance(source_config_canonical, dict):
        source_config_sections = _canonical_config_sections(source_config_canonical)
        _assert_noleague_baseline_config(source_config_canonical)
        if isinstance(expected_config_canonical, dict):
            expected_config_sections = _canonical_config_sections(expected_config_canonical)
            for section_name in ("model", "environment"):
                source_section = source_config_sections.get(section_name)
                expected_section = expected_config_sections.get(section_name)
                if source_section is None or expected_section is None:
                    continue
                if source_section != expected_section:
                    raise RuntimeError(
                        f"Imported B1 baseline config does not match the current run for section={section_name!r}"
                    )

    if expected_spec_hash256 is not None:
        source_spec_hash = _read_optional_hash_file(source_layout.spec_hash_path)
        if source_spec_hash is not None and source_spec_hash != expected_spec_hash256:
            raise RuntimeError(
                "Imported B1 baseline spec hash does not match the current run: "
                f"source={source_spec_hash} expected={expected_spec_hash256}"
            )

    source_model_state_dict = payload.get("model_state_dict")
    if not isinstance(source_model_state_dict, dict):
        raise RuntimeError(f"Imported B1 baseline weights payload is missing model_state_dict: {source_run_dir}")
    source_keys = set(source_model_state_dict)
    expected_keys = set(expected_model_state_dict)
    if source_keys != expected_keys:
        missing = sorted(expected_keys - source_keys)
        extra = sorted(source_keys - expected_keys)
        raise RuntimeError(
            "Imported B1 baseline model contract does not match the current run: "
            f"missing_keys={missing} extra_keys={extra}"
        )
    for key in sorted(expected_keys):
        source_value = source_model_state_dict[key]
        expected_value = expected_model_state_dict[key]
        if not isinstance(source_value, torch.Tensor) or not isinstance(expected_value, torch.Tensor):
            continue
        if tuple(source_value.shape) != tuple(expected_value.shape) or source_value.dtype != expected_value.dtype:
            raise RuntimeError(
                "Imported B1 baseline tensor contract does not match the current run: "
                f"key={key} source_shape={tuple(source_value.shape)} "
                f"expected_shape={tuple(expected_value.shape)} "
                f"source_dtype={source_value.dtype} expected_dtype={expected_value.dtype}"
            )


def _load_snapshot_registry(path: Path) -> SnapshotRegistry:
    if not path.exists():
        raise FileNotFoundError(path)
    return SnapshotRegistry.load(path)


def _load_dev_eval_summaries(path: Path) -> dict[str, float | DevEvalPolicySummary]:
    payload = _load_json_object(path, label="dev-eval summaries")
    summaries: dict[str, float | DevEvalPolicySummary] = {}
    for policy_id, raw_summary in payload.items():
        if isinstance(raw_summary, bool):
            raise TypeError(f"dev-eval summary for {policy_id!r} cannot be a boolean")
        if isinstance(raw_summary, (int, float)):
            summaries[policy_id] = float(raw_summary)
            continue
        if not isinstance(raw_summary, dict):
            raise TypeError(
                "dev-eval summary values must be numbers or objects with aggregate_score/anchor_scores, "
                f"got {type(raw_summary).__name__} for {policy_id!r}"
            )
        aggregate_score = raw_summary.get("aggregate_score")
        if isinstance(aggregate_score, bool) or not isinstance(aggregate_score, (int, float)):
            raise TypeError(f"dev-eval summary for {policy_id!r} must include numeric aggregate_score")
        anchor_scores = raw_summary.get("anchor_scores", {})
        if not isinstance(anchor_scores, dict) or any(not isinstance(key, str) for key in anchor_scores):
            raise TypeError(f"dev-eval summary for {policy_id!r} must include object anchor_scores")
        summaries[policy_id] = DevEvalPolicySummary(
            policy_id=policy_id,
            aggregate_score=float(aggregate_score),
            anchor_scores=anchor_scores,
        )
    return summaries


def _selection_requires_snapshot_registry(stack: StackConfig) -> bool:
    evaluation = stack.config.evaluation
    if evaluation is None:
        return False
    selection = evaluation.final_policy_set_selection
    return selection.include_final_champion_snapshot or bool(selection.include_spaced_snapshots_near_percent_updates)


def _selection_requires_dev_eval_summaries(stack: StackConfig) -> bool:
    evaluation = stack.config.evaluation
    if evaluation is None:
        return False
    selection = evaluation.final_policy_set_selection
    fixed_slots = int(selection.include_random_legal_baseline_b0) + int(selection.include_no_league_baseline_b1)
    fixed_slots += int(selection.include_final_champion_snapshot)
    fixed_slots += len(selection.include_spaced_snapshots_near_percent_updates)
    if selection.include_heuristic_public_b2_if_exists:
        return True
    return evaluation.final_policy_set_size > fixed_slots


def _policy_set_selection(
    stack: StackConfig,
    *,
    snapshot_registry: SnapshotRegistry | None = None,
    dev_eval_summaries: Mapping[str, float | DevEvalPolicySummary] | None = None,
) -> list[str]:
    evaluation = stack.config.evaluation
    if evaluation is None:
        return []
    selection = evaluation.final_policy_set_selection
    if selection.version != "deterministic_v1":
        raise ValueError(f"unsupported final_policy_set_selection.version: {selection.version!r}")
    return select_final_policy_set_deterministic_v1(
        snapshot_registry=snapshot_registry or SnapshotRegistry(),
        dev_eval_summaries=dev_eval_summaries or {},
        config=selection,
        final_policy_set_size=evaluation.final_policy_set_size,
    )


def _resolve_policy_set_selection(
    stack: StackConfig,
    *,
    snapshot_registry_path: Path | None = None,
    dev_eval_summaries_path: Path | None = None,
) -> tuple[list[str], dict[str, Any]]:
    evaluation = stack.config.evaluation
    source_paths = {
        "snapshot_registry_json": None
        if snapshot_registry_path is None
        else _manifest_source_path(snapshot_registry_path, root=stack.root),
        "dev_eval_summaries_json": None
        if dev_eval_summaries_path is None
        else _manifest_source_path(dev_eval_summaries_path, root=stack.root),
    }
    if evaluation is None:
        return [], {"mode": "not_configured", "status": "not_configured", "source_paths": source_paths}

    snapshot_registry = None if snapshot_registry_path is None else _load_snapshot_registry(snapshot_registry_path)
    dev_eval_summaries = None if dev_eval_summaries_path is None else _load_dev_eval_summaries(dev_eval_summaries_path)

    missing_inputs: list[str] = []
    if _selection_requires_snapshot_registry(stack) and snapshot_registry is None:
        missing_inputs.append("snapshot_registry_json")
    if _selection_requires_dev_eval_summaries(stack) and dev_eval_summaries is None:
        missing_inputs.append("dev_eval_summaries_json")

    details: dict[str, Any] = {
        "mode": evaluation.final_policy_set_selection.version,
        "status": "resolved",
        "version": evaluation.final_policy_set_selection.version,
        "final_policy_set_size": evaluation.final_policy_set_size,
        "source_paths": source_paths,
        "missing_inputs": missing_inputs,
    }
    if missing_inputs:
        details["mode"] = "unresolved"
        details["status"] = "unresolved"
        details["reason"] = "deterministic final policy set inputs were not provided"
        return [], details

    policy_ids = _policy_set_selection(
        stack,
        snapshot_registry=snapshot_registry,
        dev_eval_summaries=dev_eval_summaries,
    )
    details["selected_policy_count"] = len(policy_ids)
    return policy_ids, details


def _spec_mismatch_policy(stack: StackConfig) -> str:
    return "hard_fail"


def _resolve_run_label(parser: argparse.ArgumentParser, run_label: str, run_id_alias: str) -> str:
    normalized_label = run_label.strip()
    normalized_alias = run_id_alias.strip()
    if normalized_label and normalized_alias and normalized_label != normalized_alias:
        parser.error("--run-label and deprecated --run-id must match when both are provided")
    if normalized_alias:
        print("Warning: --run-id is deprecated; use --run-label instead.", file=sys.stderr)
    return normalized_label or normalized_alias


def _require_positive_int(name: str, value: int) -> int:
    number = int(value)
    if number < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return number


def _resolve_runtime_profile(stack: StackConfig, profile_override: str) -> str:
    if profile_override.strip():
        return profile_override.strip()
    system_config = stack.config.system
    if system_config is None:
        return "fast"
    return system_config.profile.local_iteration


def _resolve_device(stack: StackConfig, device_override: str) -> torch.device:
    requested = device_override.strip()
    if not requested:
        system_config = stack.config.system
        requested = "cpu" if system_config is None else getattr(system_config, "learner_device", "cpu")
    normalized = str(requested).strip().lower()
    if normalized in {"auto", "cuda:auto"}:
        requested = "cuda:0" if torch.cuda.is_available() and int(torch.cuda.device_count()) > 0 else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(
            "Requested CUDA device is unavailable; falling back to cpu for the canonical single-node run.",
            file=sys.stderr,
        )
        requested = "cpu"
    return torch.device(requested)


def _resolve_seed(stack: StackConfig, seed_override: int | None) -> int:
    if seed_override is not None:
        return int(seed_override)
    reproducibility = stack.config.reproducibility
    if reproducibility is None:
        return 7
    return int(reproducibility.seed_derivation.base_seed64)


def _manifest_scaffold_only_reason(stack: StackConfig) -> str | None:
    missing_blocks: list[str] = []
    if stack.config.environment is None:
        missing_blocks.append("environment")
    if stack.config.training is None:
        missing_blocks.append("training")
    if stack.config.model is None:
        missing_blocks.append("model")
    if missing_blocks:
        return f"missing config blocks: {', '.join(missing_blocks)}"
    return None


def _runtime_training_prerequisite_failure(stack: StackConfig) -> str | None:
    if _manifest_scaffold_only_reason(stack) is not None:
        return None

    try:
        weiss_sim = importlib.import_module("weiss_sim")
    except ModuleNotFoundError:
        return "weiss_sim is not importable in the active interpreter"

    missing_runtime_attrs = [
        attr_name for attr_name in ("fast", "inspect", "rl", "PASS_ACTION_ID") if not hasattr(weiss_sim, attr_name)
    ]
    if missing_runtime_attrs:
        return f"active weiss_sim runtime is missing stepping APIs: {', '.join(missing_runtime_attrs)}"

    rl_module = weiss_sim.rl
    missing_rl_attrs = [attr_name for attr_name in ("reset_rl", "step_rl") if not hasattr(rl_module, attr_name)]
    if missing_rl_attrs:
        return f"active weiss_sim.rl is missing runtime methods: {', '.join(missing_rl_attrs)}"

    return None


def _print_manifest_only_message(reason: str) -> None:
    print("Manifest scaffold only: no learner training or rollout collection was executed.")
    print(f"Reason: {reason}.")


def _raise_runtime_prerequisite_failure(reason: str) -> None:
    raise RuntimeError(
        "Canonical simulator-backed training requires a weiss_sim runtime with stepping support. "
        f"Startup failed because {reason}."
    )


def _training_paths(run_dir: Path) -> TrainingPaths:
    layout = ArtifactLayout.from_run_dir(run_dir)
    layout.ensure_directories()
    training_dir = layout.training_dir
    checkpoints_dir = layout.training_checkpoints_dir
    logs_dir = layout.training_logs_dir
    snapshots_dir = layout.training_snapshots_dir
    return TrainingPaths(
        training_dir=training_dir,
        checkpoints_dir=checkpoints_dir,
        logs_dir=logs_dir,
        snapshots_dir=snapshots_dir,
        tensorboard_dir=layout.tensorboard_dir,
        scalars_path=logs_dir / "scalars.jsonl",
        performance_log_path=layout.performance_log_path,
        latest_checkpoint_path=checkpoints_dir / _LATEST_CHECKPOINT_FILENAME,
        best_checkpoint_path=checkpoints_dir / _BEST_CHECKPOINT_FILENAME,
        checkpoint_tracker_path=checkpoints_dir / _CHECKPOINT_TRACKER_FILENAME,
    )


def _run_artifacts_from_existing_run_dir(run_dir: Path) -> RunArtifacts:
    resolved_run_dir = Path(run_dir).resolve()
    layout = ArtifactLayout.from_run_dir(resolved_run_dir)
    layout.ensure_directories()
    return RunArtifacts(
        run_dir=resolved_run_dir,
        run_dir_name=resolved_run_dir.name,
        layout=layout,
        manifest_path=layout.manifest_path,
        spec_bundle_path=layout.spec_bundle_path,
        spec_hash_path=layout.spec_hash_path,
        config_hash_path=layout.config_hash_path,
        config_json_path=layout.config_json_path,
        environment_path=layout.environment_path,
        run_summary_path=layout.run_summary_path,
        determinism_report_path=layout.determinism_report_path,
        paper_readiness_summary_path=layout.paper_readiness_summary_path,
        performance_log_path=layout.performance_log_path,
    )


def _configure_torch_threads(stack: StackConfig) -> None:
    system_config = stack.config.system
    if system_config is None:
        return
    torch.set_num_threads(int(system_config.learner_torch_threads))
    with suppress(RuntimeError):
        torch.set_num_interop_threads(1)


@contextmanager
def _torch_num_threads_scope(num_threads: int | None):
    if num_threads is None:
        yield
        return
    target = int(num_threads)
    if target < 1:
        raise ValueError("num_threads must be >= 1")
    previous = int(torch.get_num_threads())
    if previous == target:
        yield
        return
    torch.set_num_threads(target)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _central_runtime_actor_torch_threads(stack: StackConfig, runtime: QueueRuntime) -> int | None:
    system_config = stack.config.system
    if system_config is None:
        return None
    if str(system_config.actor_device).strip().lower() != "cpu":
        return None
    if bool(getattr(runtime, "_use_process_collectors", False)):
        return None
    if not bool(getattr(runtime, "_use_central_batched_collection", False)):
        return None
    return int(system_config.actor_torch_threads)


def _spec_dimensions(contract: SimulatorContract) -> tuple[int, int]:
    observation_dim = int(contract.spec_bundle["observation"]["obs_len"])
    action_dim = int(contract.spec_bundle["action"]["action_space_size"])
    return observation_dim, action_dim


def _env_pool_config(stack: StackConfig, *, seed: int) -> dict[str, Any]:
    return build_env_config_from_stack(stack, seed=int(seed))


def _build_env(
    stack: StackConfig,
    *,
    profile: str,
    num_envs: int,
    seed: int,
) -> DecisionBoundaryEnv:
    env_config = _env_pool_config(stack, seed=seed)
    pool, layout_name = make_env_pool_from_config(
        env_config,
        profile=profile,  # type: ignore[arg-type]
        num_envs=num_envs,
    )
    if layout_name != "mask":
        raise RuntimeError(
            "The compatibility training path expects mask legality because ImpalaLearner consumes legal_mask. "
            f"Profile {profile!r} resolved to layout {layout_name!r}."
        )
    max_no_progress_decisions = None
    curriculum = stack.config.curriculum
    if curriculum is not None:
        raw_limit = curriculum.simulator.get("max_no_progress_decisions")
        if raw_limit is not None:
            max_no_progress_decisions = int(raw_limit)
    return DecisionBoundaryEnv(
        pool,
        legality="mask",
        engine_status_policy="hard_fail",
        max_decisions=int(env_config["max_decisions"]),
        max_ticks=int(env_config["max_ticks"]),
        max_no_progress_decisions=max_no_progress_decisions,
    )


def _build_ids_eval_env(
    stack: StackConfig,
    *,
    seed: int,
    pass_action_id: int,
) -> DecisionBoundaryEnv:
    env_config = _env_pool_config(stack, seed=seed)
    pool, layout_name = make_env_pool_from_config(
        env_config,
        profile="fast",
        num_envs=1,
    )
    if layout_name != "i16_legal_ids":
        raise RuntimeError(
            "Periodic dev eval requires ids-based legality for the pinned eval protocol. "
            f"Profile 'fast' resolved to layout {layout_name!r}."
        )
    max_no_progress_decisions = None
    curriculum = stack.config.curriculum
    if curriculum is not None:
        raw_limit = curriculum.simulator.get("max_no_progress_decisions")
        if raw_limit is not None:
            max_no_progress_decisions = int(raw_limit)
    return DecisionBoundaryEnv(
        pool,
        legality="ids_offsets",
        pass_action_id=pass_action_id,
        engine_status_policy="hard_fail",
        max_decisions=int(env_config["max_decisions"]),
        max_ticks=int(env_config["max_ticks"]),
        max_no_progress_decisions=max_no_progress_decisions,
    )


def _bootstrap_values(
    model: PolicyValueModel,
    rollout: MinimalRollout,
    final_seat_hidden: torch.Tensor,
    *,
    device: torch.device,
) -> np.ndarray:
    bootstrap_value = np.zeros((rollout.bootstrap_obs.shape[0],), dtype=np.float32)
    valid_rows = (rollout.bootstrap_actor == 0) | (rollout.bootstrap_actor == 1)
    if not np.any(valid_rows):
        return bootstrap_value

    with torch.inference_mode():
        _, value_tensor, _ = model.forward_seat_aware(
            torch.as_tensor(rollout.bootstrap_obs[valid_rows], device=device),
            torch.as_tensor(rollout.bootstrap_actor[valid_rows], device=device, dtype=torch.long),
            final_seat_hidden[valid_rows],
        )
    bootstrap_value[valid_rows] = value_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    return bootstrap_value


def _build_learner_batch(
    stack: StackConfig,
    rollout: MinimalRollout,
    bootstrap_value: np.ndarray,
    *,
    action_dim: int,
    initial_hidden_state: torch.Tensor,
    pass_action_id: int,
) -> dict[str, Any]:
    training_config = stack.config.training
    rewards_config = stack.config.rewards
    if training_config is None or rewards_config is None:
        raise RuntimeError("The canonical single-node path requires training and rewards config blocks")

    target_logp = masked_logp_from_mask(
        rollout.logits.reshape(-1, action_dim),
        rollout.legal_mask.reshape(-1, action_dim),
        rollout.actions.reshape(-1),
        pass_action_id=pass_action_id,
    ).reshape(rollout.actions.shape)

    rewards = np.asarray(rollout.rewards, dtype=np.float32)

    discounts = np.logical_not(rollout.terminated).astype(np.float32) * float(rewards_config.gamma)
    if not bool(rewards_config.truncation.bootstrap_value):
        discounts *= np.logical_not(rollout.truncated).astype(np.float32)

    values = np.concatenate([rollout.values, bootstrap_value[np.newaxis, :]], axis=0)
    vtrace_result: VTraceTargets = compute_vtrace_targets(
        rewards,
        values,
        discounts,
        rollout.behavior_logp,
        target_logp,
        rho_bar=training_config.vtrace_rho_bar,
        c_bar=training_config.vtrace_c_bar,
    )

    return {
        "obs": rollout.obs,
        "actions": rollout.actions,
        "legal_mask": rollout.legal_mask,
        "to_play_seat": rollout.to_play_seat,
        "actor": rollout.to_play_seat,
        "initial_hidden_state": initial_hidden_state.detach().cpu().numpy(),
        "rewards": rewards,
        "discounts": discounts,
        "behavior_logp": rollout.behavior_logp,
        "behavior_logits": rollout.logits,
        "logits": rollout.logits,
        "vtrace_result": vtrace_result,
        "vtrace_rho_bar": float(training_config.vtrace_rho_bar),
        "vtrace_c_bar": float(training_config.vtrace_c_bar),
    }


def _write_scalars_record(
    *,
    scalars_path: Path,
    learner: ImpalaLearner,
    metrics: dict[str, float],
    start_time: float,
) -> None:
    wall_clock_seconds = time.time() - start_time
    record = {
        "update_count": int(learner.update_count),
        "policy_version": int(learner.get_policy_version()),
        "wall_clock_seconds": wall_clock_seconds,
        "wall_clock_ms": int(wall_clock_seconds * 1000),
        **metrics,
    }
    with scalars_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _write_checkpoint(
    *,
    checkpoint_path: Path,
    learner: ImpalaLearner,
    stack: StackConfig,
    device: torch.device,
    spec_hash256: str | None = None,
    algorithm: str | None = None,
) -> dict[str, Any]:
    if learner.model is None:
        raise RuntimeError("Cannot write a checkpoint without a learner model")

    payload = {
        "format": "minimal_train_checkpoint_v1",
        "update_count": int(learner.update_count),
        "policy_version": int(learner.get_policy_version()),
        "device": str(device),
        "config_hash256": compute_config_hash256(stack),
        "spec_hash256": spec_hash256,
        "algorithm": algorithm,
        "recurrent_core": getattr(stack.config.model, "recurrent_core", None),
        "total_samples_processed": int(getattr(learner, "total_samples_processed", 0)),
        "model_state_dict": learner.model.state_dict(),
        "optimizer_state_dict": None if learner.optimizer is None else learner.optimizer.state_dict(),
        "grad_scaler_state_dict": (
            None if getattr(learner, "_grad_scaler", None) is None else learner._grad_scaler.state_dict()
        ),
    }
    torch.save(payload, checkpoint_path)
    return payload


def _relative_path_text(path: Path, *, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _load_checkpoint_tracker(training_paths: TrainingPaths) -> dict[str, Any]:
    if not training_paths.checkpoint_tracker_path.is_file():
        return {"format": "checkpoint_tracker_v1", "latest": None, "best": None}
    payload = json.loads(training_paths.checkpoint_tracker_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"checkpoint tracker must be a JSON object: {training_paths.checkpoint_tracker_path}")
    payload.setdefault("format", "checkpoint_tracker_v1")
    payload.setdefault("latest", None)
    payload.setdefault("best", None)
    return payload


def _write_checkpoint_tracker(training_paths: TrainingPaths, payload: dict[str, Any]) -> None:
    training_paths.checkpoint_tracker_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _checkpoint_guard_log_path(training_paths: TrainingPaths) -> Path:
    return training_paths.logs_dir / "checkpoint_guard.jsonl"


def _build_checkpoint_record(
    *,
    alias_name: str,
    alias_path: Path,
    source_checkpoint_path: Path,
    artifacts: RunArtifacts,
    learner: ImpalaLearner,
    metric_kind: str | None = None,
    metric_value: float | None = None,
) -> dict[str, Any]:
    return {
        "alias": alias_name,
        "alias_path": _relative_path_text(alias_path, root=artifacts.run_dir),
        "source_checkpoint_path": _relative_path_text(source_checkpoint_path, root=artifacts.run_dir),
        "update_count": int(learner.update_count),
        "policy_version": int(learner.get_policy_version()),
        "metric_kind": metric_kind,
        "metric_value": metric_value,
    }


def _dev_eval_aggregate_score(dev_eval_summary: Mapping[str, Any] | None) -> float | None:
    if dev_eval_summary is None:
        return None
    aggregate_score = dev_eval_summary.get("aggregate_score")
    if isinstance(aggregate_score, (int, float)) and np.isfinite(float(aggregate_score)):
        return float(aggregate_score)
    uncertainty = dev_eval_summary.get("uncertainty")
    if isinstance(uncertainty, Mapping):
        mean_value = uncertainty.get("mean")
        if isinstance(mean_value, (int, float)) and np.isfinite(float(mean_value)):
            return float(mean_value)
    return None


def _dev_eval_worst_truncation_rate(dev_eval_summary: Mapping[str, Any] | None) -> float | None:
    if dev_eval_summary is None:
        return None
    stall_monitor = dev_eval_summary.get("stall_monitor")
    if isinstance(stall_monitor, Mapping):
        worst_rate = stall_monitor.get("worst_truncation_rate")
        if isinstance(worst_rate, (int, float)) and np.isfinite(float(worst_rate)):
            return float(worst_rate)
    anchors = dev_eval_summary.get("anchors")
    if not isinstance(anchors, Mapping):
        return None
    worst_rate: float | None = None
    for anchor_payload in anchors.values():
        if not isinstance(anchor_payload, Mapping):
            continue
        summary = anchor_payload.get("summary")
        if not isinstance(summary, Mapping):
            continue
        games = summary.get("games")
        truncations = summary.get("truncations")
        if not isinstance(games, (int, float)) or not isinstance(truncations, (int, float)):
            continue
        if float(games) <= 0:
            continue
        rate = float(truncations) / float(games)
        worst_rate = rate if worst_rate is None else max(worst_rate, rate)
    return worst_rate


def _summary_rate(matchup_summary: Mapping[str, Any], key: str) -> float | None:
    games = matchup_summary.get("games")
    count = matchup_summary.get(key)
    if not isinstance(games, (int, float)) or not isinstance(count, (int, float)):
        return None
    if float(games) <= 0.0:
        return None
    return float(count) / float(games)


def _dev_eval_worst_reason_rate(
    dev_eval_summary: Mapping[str, Any] | None,
    *,
    summary_key: str,
    stall_monitor_key: str,
) -> float | None:
    if dev_eval_summary is None:
        return None
    stall_monitor = dev_eval_summary.get("stall_monitor")
    if isinstance(stall_monitor, Mapping):
        worst_rate = stall_monitor.get(stall_monitor_key)
        if isinstance(worst_rate, (int, float)) and np.isfinite(float(worst_rate)):
            return float(worst_rate)
    anchors = dev_eval_summary.get("anchors")
    if not isinstance(anchors, Mapping):
        return None
    worst_rate: float | None = None
    for anchor_payload in anchors.values():
        if not isinstance(anchor_payload, Mapping):
            continue
        summary = anchor_payload.get("summary")
        if not isinstance(summary, Mapping):
            continue
        rate = _summary_rate(summary, summary_key)
        if rate is None:
            continue
        worst_rate = rate if worst_rate is None else max(worst_rate, rate)
    return worst_rate


def _dev_eval_worst_no_progress_timeout_rate(dev_eval_summary: Mapping[str, Any] | None) -> float | None:
    return _dev_eval_worst_reason_rate(
        dev_eval_summary,
        summary_key="no_progress_timeouts",
        stall_monitor_key="worst_no_progress_timeout_rate",
    )


def _dev_eval_worst_natural_timeout_rate(dev_eval_summary: Mapping[str, Any] | None) -> float | None:
    return _dev_eval_worst_reason_rate(
        dev_eval_summary,
        summary_key="natural_timeouts",
        stall_monitor_key="worst_natural_timeout_rate",
    )


def _dev_eval_worst_stall_rate(dev_eval_summary: Mapping[str, Any] | None) -> float | None:
    no_progress_rate = _dev_eval_worst_no_progress_timeout_rate(dev_eval_summary)
    if no_progress_rate is not None:
        return no_progress_rate
    return _dev_eval_worst_truncation_rate(dev_eval_summary)


def _dev_eval_confidence_stats(dev_eval_summary: Mapping[str, Any] | None) -> dict[str, float | None]:
    stats = {
        "min_prob_gt_half": None,
        "max_prob_lt_half": None,
        "max_ci_half_width": None,
    }
    if dev_eval_summary is None:
        return stats
    anchors = dev_eval_summary.get("anchors")
    if not isinstance(anchors, Mapping):
        return stats
    min_prob_gt_half: float | None = None
    max_prob_lt_half: float | None = None
    max_ci_half_width: float | None = None
    for anchor_payload in anchors.values():
        if not isinstance(anchor_payload, Mapping):
            continue
        uncertainty = anchor_payload.get("uncertainty")
        if not isinstance(uncertainty, Mapping):
            continue
        prob_gt_half = uncertainty.get("prob_gt_half")
        prob_lt_half = uncertainty.get("prob_lt_half")
        ci_half_width = uncertainty.get("ci_half_width")
        if isinstance(prob_gt_half, (int, float)) and np.isfinite(float(prob_gt_half)):
            min_prob_gt_half = (
                float(prob_gt_half) if min_prob_gt_half is None else min(min_prob_gt_half, float(prob_gt_half))
            )
        if isinstance(prob_lt_half, (int, float)) and np.isfinite(float(prob_lt_half)):
            max_prob_lt_half = (
                float(prob_lt_half) if max_prob_lt_half is None else max(max_prob_lt_half, float(prob_lt_half))
            )
        if isinstance(ci_half_width, (int, float)) and np.isfinite(float(ci_half_width)):
            max_ci_half_width = (
                float(ci_half_width) if max_ci_half_width is None else max(max_ci_half_width, float(ci_half_width))
            )
    stats["min_prob_gt_half"] = min_prob_gt_half
    stats["max_prob_lt_half"] = max_prob_lt_half
    stats["max_ci_half_width"] = max_ci_half_width
    return stats


def _dev_eval_ineligibility_reasons(
    stack: StackConfig,
    *,
    dev_eval_summary: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    if dev_eval_summary is None:
        return ("missing",)
    current_score = _dev_eval_aggregate_score(dev_eval_summary)
    if current_score is None:
        return ("missing_score",)
    curriculum = stack.config.curriculum
    if curriculum is None or not curriculum.stall_monitor.enabled:
        return ()
    reasons: list[str] = []
    worst_rate = _dev_eval_worst_stall_rate(dev_eval_summary)
    if worst_rate is not None and worst_rate >= float(curriculum.stall_monitor.truncation_rate_threshold):
        reasons.append("truncation")
    checkpoint_guard = curriculum.checkpoint_guard
    confidence = _dev_eval_confidence_stats(dev_eval_summary)
    min_prob_gt_half = confidence["min_prob_gt_half"]
    max_ci_half_width = confidence["max_ci_half_width"]
    if min_prob_gt_half is not None and (float(min_prob_gt_half) < float(checkpoint_guard.promote_min_prob_gt_half)):
        reasons.append("confidence_prob")
    if max_ci_half_width is not None and (float(max_ci_half_width) > float(checkpoint_guard.promote_max_ci_half_width)):
        reasons.append("confidence_ci")
    return tuple(reasons)


def _dev_eval_metric_eligible(stack: StackConfig, *, dev_eval_summary: Mapping[str, Any] | None) -> bool:
    return not _dev_eval_ineligibility_reasons(stack, dev_eval_summary=dev_eval_summary)


def _confirmatory_dev_eval_target_pairs(stack: StackConfig) -> int:
    evaluation = _evaluation_config_or_raise(stack)
    base_pairs = int(evaluation.periodic_dev_eval_paired_seeds)
    max_pairs = int(evaluation.final_matrix_stage2_adaptive_max_paired_seeds)
    return max(base_pairs, min(max_pairs, max(32, base_pairs * 4)))


def _expand_periodic_dev_eval_paired_seeds(
    base_paired_seeds: Sequence[int],
    *,
    requested_pairs: int,
    seed_file_sha256: str,
    update_count: int,
    policy_version: int,
    scope: str,
) -> list[int]:
    requested_pairs_i = int(requested_pairs)
    paired_seeds = [int(seed) for seed in base_paired_seeds[:requested_pairs_i]]
    seen = set(paired_seeds)
    extra_index = 0
    while len(paired_seeds) < requested_pairs_i:
        derived_seed = (
            stable_hash64(
                canonical_json_bytes(
                    {
                        "kind": "periodic_dev_eval_confirmatory_seed_v1",
                        "scope": str(scope),
                        "seed_file_sha256": str(seed_file_sha256),
                        "update_count": int(update_count),
                        "policy_version": int(policy_version),
                        "extra_index": int(extra_index),
                    }
                )
            )
            & _U64_MASK
        )
        extra_index += 1
        if derived_seed in seen:
            continue
        paired_seeds.append(int(derived_seed))
        seen.add(int(derived_seed))
    return paired_seeds


def _confirmatory_dev_eval_request(
    *,
    stack: StackConfig,
    existing_best_record: Mapping[str, Any] | None,
    dev_eval_summary: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    reasons = _dev_eval_ineligibility_reasons(stack, dev_eval_summary=dev_eval_summary)
    if any(reason not in {"confidence_prob", "confidence_ci"} for reason in reasons):
        return None
    current_score = _dev_eval_aggregate_score(dev_eval_summary)
    if current_score is None:
        return None
    curriculum = stack.config.curriculum
    if curriculum is None:
        return None
    checkpoint_guard = curriculum.checkpoint_guard
    if float(current_score) < float(checkpoint_guard.min_best_score):
        return None

    existing_metric_kind = ""
    existing_metric_value: float | None = None
    score_shortfall = 0.0
    if existing_best_record is not None:
        existing_metric_kind = str(existing_best_record.get("metric_kind", "")).strip()
        raw_existing_metric_value = existing_best_record.get("metric_value")
        if isinstance(raw_existing_metric_value, (int, float)) and np.isfinite(float(raw_existing_metric_value)):
            existing_metric_value = float(raw_existing_metric_value)
            score_shortfall = max(0.0, existing_metric_value - float(current_score))
    if (
        existing_metric_kind == "dev_eval_mean"
        and existing_metric_value is not None
        and score_shortfall > 0.0
        and score_shortfall > 2.0 * float(checkpoint_guard.rollback_score_margin)
    ):
        return None

    confidence = _dev_eval_confidence_stats(dev_eval_summary)
    confirmatory_reasons: list[str] = []
    prob_shortfall = 0.0
    if "confidence_prob" in reasons:
        min_prob_gt_half = confidence["min_prob_gt_half"]
        if min_prob_gt_half is None:
            return None
        prob_shortfall = max(0.0, float(checkpoint_guard.promote_min_prob_gt_half) - float(min_prob_gt_half))
        if prob_shortfall <= _CONFIRMATORY_DEV_EVAL_MAX_PROB_SHORTFALL:
            confirmatory_reasons.append("confidence_prob")
    ci_excess = 0.0
    if "confidence_ci" in reasons:
        max_ci_half_width = confidence["max_ci_half_width"]
        if max_ci_half_width is None:
            return None
        ci_excess = max(0.0, float(max_ci_half_width) - float(checkpoint_guard.promote_max_ci_half_width))
        if ci_excess <= _CONFIRMATORY_DEV_EVAL_MAX_CI_EXCESS:
            confirmatory_reasons.append("confidence_ci")
    if (
        existing_metric_kind == "dev_eval_mean"
        and existing_metric_value is not None
        and score_shortfall > 0.0
        and score_shortfall <= 2.0 * float(checkpoint_guard.rollback_score_margin)
    ):
        confirmatory_reasons.append("score_drop")
    if not confirmatory_reasons:
        return None
    if prob_shortfall > _CONFIRMATORY_DEV_EVAL_MAX_PROB_SHORTFALL:
        return None
    if ci_excess > _CONFIRMATORY_DEV_EVAL_MAX_CI_EXCESS:
        return None

    return {
        "reasons": confirmatory_reasons,
        "current_score": float(current_score),
        "existing_best_score": existing_metric_value,
        "prob_shortfall": prob_shortfall,
        "ci_excess": ci_excess,
        "target_pairs": _confirmatory_dev_eval_target_pairs(stack),
    }


def _checkpoint_candidate_metric(
    *,
    stack: StackConfig,
    latest_metrics: Mapping[str, float] | None,
    dev_eval_summary: Mapping[str, Any] | None,
) -> tuple[str | None, float | None]:
    if _dev_eval_metric_eligible(stack, dev_eval_summary=dev_eval_summary):
        aggregate_score = _dev_eval_aggregate_score(dev_eval_summary)
        if aggregate_score is not None:
            return "dev_eval_mean", aggregate_score
    evaluation = stack.config.evaluation
    if (
        evaluation is not None
        and int(evaluation.periodic_dev_eval_interval_updates) > 0
        and dev_eval_summary is not None
    ):
        return None, None
    if latest_metrics is not None:
        loss_value = latest_metrics.get("loss")
        if isinstance(loss_value, (int, float)) and np.isfinite(float(loss_value)):
            return "training_loss", float(loss_value)
    return None, None


def _should_promote_best_checkpoint(
    *,
    existing_record: Mapping[str, Any] | None,
    candidate_kind: str | None,
    candidate_value: float | None,
) -> bool:
    if candidate_kind is None:
        return False
    if existing_record is None:
        return True
    existing_kind = existing_record.get("metric_kind")
    existing_value = existing_record.get("metric_value")
    if candidate_kind == "dev_eval_mean":
        if existing_kind != "dev_eval_mean":
            return True
        if not isinstance(existing_value, (int, float)):
            return True
        return float(candidate_value) > float(existing_value)
    if candidate_kind == "training_loss":
        if existing_kind == "dev_eval_mean":
            return False
        if not isinstance(existing_value, (int, float)):
            return True
        return float(candidate_value) < float(existing_value)
    return False


def _publish_checkpoint_aliases(
    *,
    stack: StackConfig,
    training_paths: TrainingPaths,
    artifacts: RunArtifacts,
    checkpoint_path: Path,
    learner: ImpalaLearner,
    latest_metrics: Mapping[str, float] | None,
    dev_eval_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    tracker = _load_checkpoint_tracker(training_paths)

    shutil.copy2(checkpoint_path, training_paths.latest_checkpoint_path)
    latest_kind, latest_value = _checkpoint_candidate_metric(
        stack=stack,
        latest_metrics=latest_metrics,
        dev_eval_summary=dev_eval_summary,
    )
    latest_record = _build_checkpoint_record(
        alias_name="latest",
        alias_path=training_paths.latest_checkpoint_path,
        source_checkpoint_path=checkpoint_path,
        artifacts=artifacts,
        learner=learner,
        metric_kind=latest_kind,
        metric_value=latest_value,
    )
    tracker["latest"] = latest_record

    best_record = tracker.get("best")
    if not isinstance(best_record, Mapping):
        best_record = None
    should_update_best = best_record is None or _should_promote_best_checkpoint(
        existing_record=cast(Mapping[str, Any], best_record),
        candidate_kind=latest_kind,
        candidate_value=latest_value,
    )
    if should_update_best:
        shutil.copy2(checkpoint_path, training_paths.best_checkpoint_path)
        tracker["best"] = _build_checkpoint_record(
            alias_name="best",
            alias_path=training_paths.best_checkpoint_path,
            source_checkpoint_path=checkpoint_path,
            artifacts=artifacts,
            learner=learner,
            metric_kind=latest_kind,
            metric_value=latest_value,
        )

    _write_checkpoint_tracker(training_paths, tracker)
    return tracker


def _append_checkpoint_guard_event(training_paths: TrainingPaths, payload: Mapping[str, Any]) -> None:
    path = _checkpoint_guard_log_path(training_paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")


def _maybe_log_structured_mainmove_guard(
    *,
    training_paths: TrainingPaths,
    learner: ImpalaLearner,
    latest_metrics: Mapping[str, float] | None,
    dev_eval_summary: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if latest_metrics is None:
        return None
    top1_rate = latest_metrics.get("structured_main_move_0_2_top1_rate")
    move_share = latest_metrics.get("structured_main_move_share_when_play_available")
    if top1_rate is None or move_share is None:
        return None
    if not np.isfinite(float(top1_rate)) or not np.isfinite(float(move_share)):
        return None
    if float(top1_rate) < 0.15 and float(move_share) < 0.35:
        return None

    aggregate_score = _dev_eval_aggregate_score(dev_eval_summary) if dev_eval_summary is not None else None
    b2_score = _extract_structured_guard_b2_anchor_score(dev_eval_summary)
    if b2_score is not None and float(b2_score) > 0.10:
        return None
    if b2_score is None and aggregate_score is not None and float(aggregate_score) > 0.40:
        return None

    payload = {
        "format": "checkpoint_guard_event_v1",
        "event_kind": "structured_mainmove_warning_v1",
        "update_count": int(learner.update_count),
        "policy_version": int(learner.get_policy_version()),
        "structured_main_move_0_2_top1_rate": float(top1_rate),
        "structured_main_move_share_when_play_available": float(move_share),
        "dev_eval_aggregate_score": None if aggregate_score is None else float(aggregate_score),
        "b2_anchor_score": None if b2_score is None else float(b2_score),
    }
    _append_checkpoint_guard_event(training_paths, payload)
    return payload


def _extract_structured_guard_b2_anchor_score(dev_eval_summary: Mapping[str, Any] | None) -> float | None:
    if dev_eval_summary is None:
        return None
    anchor_scores = dev_eval_summary.get("anchor_scores")
    if not isinstance(anchor_scores, Mapping):
        return None
    for key, value in anchor_scores.items():
        key_text = str(key).strip().lower()
        if "b2" not in key_text:
            continue
        if isinstance(value, (int, float)) and np.isfinite(float(value)):
            return float(value)
    return None


def _demote_registry_champions_newer_than(training_paths: TrainingPaths, *, update_count: int) -> list[str]:
    registry_path = training_paths.snapshots_dir / REGISTRY_FILENAME
    if not registry_path.is_file():
        return []
    registry = SnapshotRegistry.load(registry_path)
    removed = registry.demote_champions_newer_than(int(update_count))
    if removed:
        registry.save(registry_path)
    return removed


def _best_checkpoint_record(training_paths: TrainingPaths) -> Mapping[str, Any] | None:
    tracker = _load_checkpoint_tracker(training_paths)
    best_record = tracker.get("best")
    return best_record if isinstance(best_record, Mapping) else None


def _resolve_resume_checkpoint_path(
    *,
    resume_from: str,
    resume_run_dir: Path | None,
) -> Path | None:
    normalized = str(resume_from).strip()
    if not normalized:
        if resume_run_dir is None:
            return None
        normalized = "latest"
    alias_name = normalized.lower()
    if alias_name in {"latest", "best"}:
        if resume_run_dir is None:
            raise ValueError("--resume-from latest|best requires --resume-run-dir")
        filename = _LATEST_CHECKPOINT_FILENAME if alias_name == "latest" else _BEST_CHECKPOINT_FILENAME
        checkpoint_path = Path(resume_run_dir).resolve() / "training" / "checkpoints" / filename
    else:
        checkpoint_path = Path(normalized).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def _restore_learner_from_checkpoint(
    *,
    checkpoint_path: Path,
    learner: ImpalaLearner,
    stack: StackConfig,
    device: torch.device,
    expected_spec_hash256: str,
    algorithm: str,
    restore_counters: bool = True,
) -> ResumeCheckpoint:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"checkpoint payload must be a dict: {checkpoint_path}")
    if str(payload.get("format", "")).strip() != "minimal_train_checkpoint_v1":
        raise RuntimeError(f"unsupported checkpoint format in {checkpoint_path}")
    payload_config_hash = str(payload.get("config_hash256", "")).strip().lower()
    expected_config_hash = compute_config_hash256(stack)
    if payload_config_hash != expected_config_hash:
        raise RuntimeError(
            f"checkpoint config hash mismatch for {checkpoint_path}: expected {expected_config_hash}, got {payload_config_hash}"
        )
    payload_spec_hash = payload.get("spec_hash256")
    if payload_spec_hash is not None and str(payload_spec_hash).strip().lower() != expected_spec_hash256:
        raise RuntimeError(
            f"checkpoint spec hash mismatch for {checkpoint_path}: expected {expected_spec_hash256}, got {payload_spec_hash}"
        )
    payload_algorithm = payload.get("algorithm")
    if payload_algorithm is not None and str(payload_algorithm).strip() and str(payload_algorithm).strip() != algorithm:
        raise RuntimeError(
            f"checkpoint algorithm mismatch for {checkpoint_path}: expected {algorithm}, got {payload_algorithm}"
        )
    model_state_dict = payload.get("model_state_dict")
    if learner.model is None or not isinstance(model_state_dict, dict):
        raise RuntimeError(f"checkpoint is missing a model_state_dict: {checkpoint_path}")
    learner.model.load_state_dict(model_state_dict)
    optimizer_state_dict = payload.get("optimizer_state_dict")
    if optimizer_state_dict is not None:
        optimizer = learner._optimizer_for_step()
        optimizer.load_state_dict(optimizer_state_dict)
    grad_scaler_state_dict = payload.get("grad_scaler_state_dict")
    if grad_scaler_state_dict is not None and getattr(learner, "_grad_scaler", None) is not None:
        learner._grad_scaler.load_state_dict(grad_scaler_state_dict)
    if restore_counters:
        learner.update_count = int(payload.get("update_count", 0))
        learner.policy_version = int(payload.get("policy_version", 0))
        learner.total_samples_processed = int(payload.get("total_samples_processed", 0))
        learner.start_time = time.time()
    return ResumeCheckpoint(
        checkpoint_path=checkpoint_path.resolve(),
        update_count=learner.update_count,
        policy_version=learner.policy_version,
        total_samples_processed=learner.total_samples_processed,
    )


def _build_training_learner(
    *,
    algorithm: str,
    model: PolicyValueModel,
    compiled_model: nn.Module | None,
    training_config: Any,
    training_paths: TrainingPaths,
    pass_action_id: int,
    checkpoint_interval_updates: int,
) -> ImpalaLearner | PpoLiteLearner:
    common_kwargs = {
        "model": model,
        "compiled_model": compiled_model,
        "learning_rate": training_config.learning_rate,
        "value_loss_coef": training_config.value_loss_coef,
        "entropy_coef": training_config.entropy_coef,
        "grad_norm_clip": training_config.grad_norm_clip,
        "mixed_precision": bool(training_config.mixed_precision),
        "checkpoint_dir": training_paths.checkpoints_dir,
        "checkpoint_interval_updates": int(checkpoint_interval_updates),
        "logs_dir": training_paths.logs_dir,
        "logging_interval_updates": 1,
        "pass_action_id": pass_action_id,
        "teacher_family_coef": training_config.teacher_family_coef,
        "teacher_slot_coef": training_config.teacher_slot_coef,
        "teacher_move_source_coef": training_config.teacher_move_source_coef,
        "teacher_attack_type_coef": training_config.teacher_attack_type_coef,
        "teacher_action_coef": training_config.teacher_action_coef,
        "teacher_same_family_action_coef": training_config.teacher_same_family_action_coef,
        "teacher_public_heuristic_coef": training_config.teacher_public_heuristic_coef,
        "teacher_public_heuristic_temperature": training_config.teacher_public_heuristic_temperature,
        "teacher_public_heuristic_families": training_config.teacher_public_heuristic_families,
        "teacher_public_heuristic_profiles": training_config.teacher_public_heuristic_profiles,
        "teacher_public_heuristic_profile_mode": training_config.teacher_public_heuristic_profile_mode,
        "teacher_public_heuristic_profiles_end_updates": training_config.teacher_public_heuristic_profiles_end_updates,
        "profile_timers": bool(getattr(training_config, "profile_timers", False)),
        "structured_metrics_mode": str(getattr(training_config, "structured_metrics_mode", "full")),
        "teacher_aux_mode": str(getattr(training_config, "teacher_aux_mode", "always")),
    }
    if algorithm in _IMPALA_ALGORITHMS:
        return ImpalaLearner(
            **common_kwargs,
            vtrace_rho_bar=training_config.vtrace_rho_bar,
            vtrace_c_bar=training_config.vtrace_c_bar,
        )
    if algorithm in _PPO_ALGORITHMS:
        return PpoLiteLearner(
            **common_kwargs,
            ppo_clip_epsilon=training_config.ppo_clip_epsilon,
            value_clip_epsilon=training_config.ppo_value_clip_epsilon,
            ppo_epochs=int(training_config.ppo_epochs),
            target_kl=training_config.ppo_target_kl,
            normalize_advantages=bool(training_config.ppo_normalize_advantages),
        )
    raise RuntimeError(f"Unsupported training.algorithm: {algorithm}")


def _entropy_coef_for_next_update(training_config: Any, *, update_count: int) -> float:
    start = float(training_config.entropy_coef)
    target = float(training_config.entropy_anneal_to)
    steps = max(1, int(training_config.entropy_anneal_steps_updates))
    progress = min(max(int(update_count), 0), steps) / float(steps)
    return float(start + (target - start) * progress)


def _maybe_compile_learner_model(
    *,
    model: PolicyValueModel,
    training_config: Any,
    device: torch.device,
) -> nn.Module | None:
    if not bool(getattr(training_config, "compile_learner", False)):
        return None
    if device.type != "cuda":
        print(
            "Learner compile note: compile_learner is enabled but the learner device is not CUDA; skipping torch.compile."
        )
        return None
    if bool(getattr(model, "supports_legal_candidate_scoring", False)):
        enable_trunk_compile = getattr(model, "enable_trunk_compile", None)
        if callable(enable_trunk_compile):
            try:
                enable_trunk_compile(mode="reduce-overhead")
            except Exception as exc:
                print(f"Learner compile note: structured trunk compile failed; skipping torch.compile ({exc!r}).")
                return None
            print("Enabled torch.compile for the structured learner trunk (mode=reduce-overhead).")
            return model
        print(
            "Learner compile note: structured legal scoring is enabled but no trunk compile hook exists; skipping torch.compile."
        )
        return None
    compiled = torch.compile(model, mode="reduce-overhead")
    print("Enabled torch.compile for the learner forward path (mode=reduce-overhead).")
    return compiled


@contextmanager
def _profile_block(enabled: bool, name: str):
    if not enabled:
        yield
        return
    with torch.autograd.profiler.record_function(name):
        yield


def _build_training_profiler(
    *,
    enabled: bool,
    run_dir: Path,
    device: torch.device,
) -> tuple[torch.profiler.profile | None, Any, Path | None]:
    if not enabled:
        return None, nullcontext(), None

    profile_dir = run_dir / "profiling" / "torch_profiler"
    profile_dir.mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    profiler = torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    )
    return profiler, profiler, profile_dir


def _collect_training_batch(
    *,
    runtime: QueueRuntime,
    algorithm: str,
    training_config: Any,
    rewards_config: Any,
) -> Any:
    if algorithm in _IMPALA_ALGORITHMS:
        return runtime.collect_update_batch(
            gamma=float(rewards_config.gamma),
            truncation_reward=float(rewards_config.truncation.reward),
            truncation_bootstrap_value=bool(rewards_config.truncation.bootstrap_value),
            vtrace_rho_bar=float(training_config.vtrace_rho_bar),
            vtrace_c_bar=float(training_config.vtrace_c_bar),
        )
    if algorithm in _PPO_ALGORITHMS:
        return runtime.collect_policy_batch(
            gamma=float(rewards_config.gamma),
            gae_lambda=float(training_config.ppo_gae_lambda),
            truncation_reward=float(rewards_config.truncation.reward),
            truncation_bootstrap_value=bool(rewards_config.truncation.bootstrap_value),
        )
    raise RuntimeError(f"Unsupported training.algorithm: {algorithm}")


def _run_structured_warmstart(
    *,
    learner: ImpalaLearner,
    runtime: QueueRuntime,
    algorithm: str,
    training_config: Any,
    rewards_config: Any,
    training_paths: TrainingPaths,
    tensorboard_logger: TensorBoardLogger | None,
    start_time: float,
    profile_timers: bool = False,
    actor_torch_threads: int | None = None,
    learner_torch_threads: int | None = None,
) -> dict[str, float]:
    if not bool(getattr(training_config, "structured_warmstart_enabled", False)):
        return {}
    if algorithm not in _IMPALA_ALGORITHMS:
        raise RuntimeError("structured warmstart currently supports only IMPALA learners")
    warmstart_cfg = training_config.structured_warmstart
    updates = int(warmstart_cfg.updates)
    if updates <= 0:
        return {}

    previous_family = float(training_config.teacher_family_coef)
    previous_slot = float(training_config.teacher_slot_coef)
    previous_move_source = float(training_config.teacher_move_source_coef)
    previous_attack_type = float(training_config.teacher_attack_type_coef)
    previous_action = float(training_config.teacher_action_coef)
    previous_same_family_action = float(training_config.teacher_same_family_action_coef)
    previous_public_heuristic = float(training_config.teacher_public_heuristic_coef)
    previous_public_heuristic_temperature = float(training_config.teacher_public_heuristic_temperature)
    previous_public_heuristic_families = tuple(training_config.teacher_public_heuristic_families)
    previous_public_heuristic_profiles = tuple(training_config.teacher_public_heuristic_profiles)
    previous_public_heuristic_profile_mode = str(training_config.teacher_public_heuristic_profile_mode)
    previous_public_heuristic_profiles_end_updates = int(training_config.teacher_public_heuristic_profiles_end_updates)
    learner.set_teacher_aux_coefs(
        family=float(warmstart_cfg.teacher_family_coef),
        slot=float(warmstart_cfg.teacher_slot_coef),
        move_source=float(warmstart_cfg.teacher_move_source_coef),
        attack_type=float(warmstart_cfg.teacher_attack_type_coef),
        action=float(warmstart_cfg.teacher_action_coef),
        same_family_action=float(warmstart_cfg.teacher_same_family_action_coef),
        public_heuristic=float(warmstart_cfg.teacher_public_heuristic_coef),
        public_heuristic_temperature=float(warmstart_cfg.teacher_public_heuristic_temperature),
        public_heuristic_families=tuple(warmstart_cfg.teacher_public_heuristic_families),
        public_heuristic_profiles=tuple(warmstart_cfg.teacher_public_heuristic_profiles),
        public_heuristic_profile_mode=str(warmstart_cfg.teacher_public_heuristic_profile_mode),
        public_heuristic_profiles_end_updates=int(warmstart_cfg.teacher_public_heuristic_profiles_end_updates),
    )
    latest_metrics: dict[str, float] = {}
    try:
        with (
            runtime.structured_warmstart_source_mix() as warmstart_source_metrics,
            runtime.disable_mirror_policy_fusion(),
        ):
            for warmstart_step in range(updates):
                with (
                    _profile_block(profile_timers, "collect_training_batch"),
                    _torch_num_threads_scope(actor_torch_threads),
                ):
                    runtime_batch = _collect_training_batch(
                        runtime=runtime,
                        algorithm=algorithm,
                        training_config=training_config,
                        rewards_config=rewards_config,
                    )
                with (
                    _profile_block(profile_timers, "learner_auxiliary_update"),
                    _torch_num_threads_scope(learner_torch_threads),
                ):
                    latest_metrics = learner.auxiliary_update(runtime_batch.learner_batch)
                latest_metrics.update(runtime_batch.runtime_metrics)
                latest_metrics.update(warmstart_source_metrics)
                latest_metrics["warmstart_phase"] = 1.0
                latest_metrics["warmstart_step"] = float(warmstart_step + 1)
                _write_scalars_record(
                    scalars_path=training_paths.scalars_path,
                    learner=learner,
                    metrics=latest_metrics,
                    start_time=start_time,
                )
                if tensorboard_logger is not None:
                    tensorboard_logger.log_training_step(
                        update_count=int(learner.update_count),
                        policy_version=int(learner.get_policy_version()),
                        wall_clock_seconds=time.time() - start_time,
                        metrics=latest_metrics,
                    )
    finally:
        learner.set_teacher_aux_coefs(
            family=previous_family,
            slot=previous_slot,
            move_source=previous_move_source,
            attack_type=previous_attack_type,
            action=previous_action,
            same_family_action=previous_same_family_action,
            public_heuristic=previous_public_heuristic,
            public_heuristic_temperature=previous_public_heuristic_temperature,
            public_heuristic_families=previous_public_heuristic_families,
            public_heuristic_profiles=previous_public_heuristic_profiles,
            public_heuristic_profile_mode=previous_public_heuristic_profile_mode,
            public_heuristic_profiles_end_updates=previous_public_heuristic_profiles_end_updates,
        )
    return latest_metrics


def _validate_algorithm_model_contract(*, algorithm: str, recurrent_core: str, encoder_kind: str) -> None:
    normalized_core = str(recurrent_core).strip().lower()
    normalized_encoder = str(encoder_kind).strip().lower()
    if algorithm == "impala_vtrace_gru" and normalized_core != "gru":
        raise RuntimeError("impala_vtrace_gru requires model.recurrent_core=gru")
    if algorithm == "impala_vtrace_ff" and normalized_core != "none":
        raise RuntimeError("impala_vtrace_ff requires model.recurrent_core=none")
    if algorithm in {"structured_v2", "impala_vtrace_structured_v1"} and normalized_core not in {"gru", "none"}:
        raise RuntimeError(f"{algorithm} requires a supported model.recurrent_core value")
    if algorithm in {"structured_v2", "impala_vtrace_structured_v1"} and normalized_encoder != "structured_v2":
        raise RuntimeError(f"{algorithm} requires model.encoder_kind=structured_v2")


def _json_relative_path(path: Path, *, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _slug_policy_id(value: str) -> str:
    parts = [
        "".join(char.lower() for char in chunk if char.isalnum())
        for chunk in str(value).replace("-", " ").replace("_", " ").split()
    ]
    return "_".join(part for part in parts if part)


def _promotion_anchor_policy_id_candidates(anchor_name: str) -> tuple[str, ...]:
    if anchor_name == _PROMOTION_GATE_RANDOMLEGAL_NAME:
        return (_PROMOTION_GATE_RANDOMLEGAL_POLICY_ID,)
    if anchor_name == _PROMOTION_GATE_NOLEAGUE_BASELINE_NAME:
        return (_PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID, anchor_name)
    if heuristic_public_profile_name_for_policy_id(anchor_name) is not None:
        return (anchor_name,)
    normalized = _slug_policy_id(anchor_name)
    if not normalized:
        return ()
    return tuple(dict.fromkeys((normalized, anchor_name)))


def _resolve_symbolic_promotion_anchor_policy_id(
    anchor_name: str,
    *,
    registry: SnapshotRegistry,
) -> str | None:
    if anchor_name == "Latest champion snapshot":
        champion_ids = registry.latest_champions(1)
        return None if not champion_ids else str(champion_ids[-1])
    if anchor_name == "Previous champion snapshot":
        champion_ids = registry.latest_champions(2)
        return None if len(champion_ids) < 2 else str(champion_ids[-2])
    if anchor_name == "Latest recent snapshot":
        recent_ids = registry.latest_ids(1)
        return None if not recent_ids else str(recent_ids[-1])
    if anchor_name == "Previous recent snapshot":
        recent_ids = registry.latest_ids(2)
        return None if len(recent_ids) < 2 else str(recent_ids[-2])
    return None


def _build_heuristic_public_policy(
    spec_bundle: Mapping[str, object],
    *,
    scoring_profile: str,
) -> HeuristicPublicPolicy:
    factory = HeuristicPublicPolicy.from_spec_bundle
    supports_scoring_profile = False
    try:
        supports_scoring_profile = "scoring_profile" in inspect.signature(factory).parameters
    except (TypeError, ValueError):
        supports_scoring_profile = False
    if supports_scoring_profile:
        return factory(spec_bundle, scoring_profile=scoring_profile)
    return factory(spec_bundle)


def _find_noleague_baseline_snapshot(run_dir: Path) -> SnapshotMeta | None:
    layout = ArtifactLayout.from_run_dir(run_dir)
    registry_path = layout.training_snapshots_dir / REGISTRY_FILENAME
    if not registry_path.is_file():
        return None
    registry = SnapshotRegistry.load(registry_path)
    snapshots_by_id = {snapshot.policy_id: snapshot for snapshot in registry.snapshots}
    for policy_id in _promotion_anchor_policy_id_candidates(_PROMOTION_GATE_NOLEAGUE_BASELINE_NAME):
        snapshot = snapshots_by_id.get(policy_id)
        if snapshot is not None:
            return snapshot

    manifest_path = layout.manifest_path
    if not manifest_path.is_file():
        return None
    manifest = _load_json_object(manifest_path, label="run manifest")
    config_canonical = manifest.get("config_canonical", {})
    if not isinstance(config_canonical, dict):
        return None
    if not _config_marks_noleague_baseline(config_canonical):
        return None
    if not registry.snapshots:
        return None
    return max(registry.snapshots, key=lambda snapshot: snapshot.sort_key())


def _import_noleague_baseline_anchor(
    *,
    training_paths: TrainingPaths,
    run_dir: Path,
    baseline_run_dir: Path,
    expected_model_state_dict: dict[str, Any],
    expected_config_canonical: dict[str, Any] | None,
    expected_spec_hash256: str | None,
) -> tuple[Path, str, int]:
    source_run_dir = Path(baseline_run_dir).resolve()
    source_snapshot = _find_noleague_baseline_snapshot(source_run_dir)
    if source_snapshot is None:
        raise FileNotFoundError(
            "Could not resolve the canonical B1 no-league baseline snapshot in "
            f"{source_run_dir}. Run a dedicated baseline_noleague training job first."
        )

    source_weights_path = source_run_dir / source_snapshot.path
    if not source_weights_path.is_file():
        raise FileNotFoundError(f"Resolved B1 baseline snapshot is missing its weights artifact: {source_weights_path}")

    snapshot_dir = training_paths.snapshots_dir / _PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    weights_path = snapshot_dir / SNAPSHOT_WEIGHTS_FILENAME
    payload = torch.load(source_weights_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Imported B1 baseline weights payload must be a dict: {source_weights_path}")
    _validate_imported_snapshot_contract(
        source_run_dir=source_run_dir,
        payload=payload,
        expected_model_state_dict=expected_model_state_dict,
        expected_config_canonical=expected_config_canonical,
        expected_spec_hash256=expected_spec_hash256,
    )
    imported_payload = dict(payload)
    imported_payload["policy_id"] = _PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID
    imported_payload["imported_from_run_dir"] = source_run_dir.as_posix()
    imported_payload["imported_from_policy_id"] = source_snapshot.policy_id
    imported_payload["imported_from_snapshot_path"] = source_snapshot.path
    torch.save(imported_payload, weights_path)
    weights_sha256 = _sha256_file(weights_path)

    _write_json_file(
        snapshot_dir / SNAPSHOT_METADATA_FILENAME,
        {
            "format": "imported_train_snapshot_metadata_v1",
            "policy_id": _PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID,
            "update": int(source_snapshot.update),
            "weights_path": snapshot_weights_relpath(_PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID),
            "weights_sha256": weights_sha256,
            "imported_from_run_dir": source_run_dir.as_posix(),
            "imported_from_policy_id": source_snapshot.policy_id,
            "imported_from_snapshot_path": source_snapshot.path,
        },
    )
    return weights_path, weights_sha256, int(source_snapshot.update)


def _validate_seed_snapshot_import_contract(
    *,
    source_run_dir: Path,
    payload: dict[str, Any],
    expected_model_state_dict: dict[str, Any],
    expected_config_canonical: dict[str, Any] | None,
    expected_spec_hash256: str | None,
) -> None:
    source_layout = ArtifactLayout.from_run_dir(source_run_dir)
    manifest_path = source_layout.manifest_path
    source_manifest = (
        _load_json_object(manifest_path, label="seed snapshot manifest") if manifest_path.is_file() else None
    )
    source_config_canonical = source_manifest.get("config_canonical") if isinstance(source_manifest, dict) else None
    if isinstance(source_config_canonical, dict) and isinstance(expected_config_canonical, dict):
        source_config_sections = _canonical_config_sections(source_config_canonical)
        expected_config_sections = _canonical_config_sections(expected_config_canonical)
        for section_name in ("model", "environment"):
            source_section = source_config_sections.get(section_name)
            expected_section = expected_config_sections.get(section_name)
            if source_section is None or expected_section is None:
                continue
            if source_section != expected_section:
                raise RuntimeError(
                    f"Imported seed snapshot config does not match the current run for section={section_name!r}"
                )

    if expected_spec_hash256 is not None:
        source_spec_hash = _read_optional_hash_file(source_layout.spec_hash_path)
        if source_spec_hash is not None and source_spec_hash != expected_spec_hash256:
            raise RuntimeError(
                "Imported seed snapshot spec hash does not match the current run: "
                f"source={source_spec_hash} expected={expected_spec_hash256}"
            )

    source_model_state_dict = payload.get("model_state_dict")
    if not isinstance(source_model_state_dict, dict):
        raise RuntimeError(f"Imported seed snapshot weights payload is missing model_state_dict: {source_run_dir}")
    source_keys = set(source_model_state_dict)
    expected_keys = set(expected_model_state_dict)
    if source_keys != expected_keys:
        missing = sorted(expected_keys - source_keys)
        extra = sorted(source_keys - expected_keys)
        raise RuntimeError(
            "Imported seed snapshot model contract does not match the current run: "
            f"missing_keys={missing} extra_keys={extra}"
        )
    for key in sorted(expected_keys):
        source_value = source_model_state_dict[key]
        expected_value = expected_model_state_dict[key]
        if not isinstance(source_value, torch.Tensor) or not isinstance(expected_value, torch.Tensor):
            continue
        if tuple(source_value.shape) != tuple(expected_value.shape) or source_value.dtype != expected_value.dtype:
            raise RuntimeError(
                "Imported seed snapshot tensor contract does not match the current run: "
                f"key={key} source_shape={tuple(source_value.shape)} "
                f"expected_shape={tuple(expected_value.shape)} "
                f"source_dtype={source_value.dtype} expected_dtype={expected_value.dtype}"
            )


def _seed_snapshot_policy_id(*, source_run_dir: Path, source_policy_id: str) -> str:
    source_hash = hashlib.sha1(source_run_dir.as_posix().encode("utf-8")).hexdigest()[:10]
    safe_policy_id = str(source_policy_id).replace("/", "_").replace("\\", "_").strip()
    return f"seed_{source_hash}_{safe_policy_id}"


def _import_seed_snapshot_pool(
    *,
    stack: StackConfig,
    training_paths: TrainingPaths,
    run_dir: Path,
    seed_snapshot_run_dir: Path,
    expected_model_state_dict: dict[str, Any],
    expected_config_canonical: dict[str, Any] | None,
    expected_spec_hash256: str | None,
) -> list[str]:
    source_run_dir = Path(seed_snapshot_run_dir).resolve()
    source_layout = ArtifactLayout.from_run_dir(source_run_dir)
    source_registry_path = source_layout.training_snapshots_dir / REGISTRY_FILENAME
    if not source_registry_path.is_file():
        raise FileNotFoundError(
            f"Could not resolve a snapshot registry in the seed snapshot run: {source_registry_path}"
        )
    source_registry = SnapshotRegistry.load(source_registry_path)
    source_snapshots = [
        snapshot
        for snapshot in source_registry.snapshots
        if snapshot.policy_id not in _promotion_anchor_policy_id_candidates(_PROMOTION_GATE_NOLEAGUE_BASELINE_NAME)
    ]
    if not source_snapshots:
        return []

    registry_path = training_paths.snapshots_dir / REGISTRY_FILENAME
    registry = SnapshotRegistry.load(registry_path)
    _sync_snapshot_registry_retention(stack, registry)
    existing_policy_ids = {snapshot.policy_id for snapshot in registry.snapshots}
    source_champions = set(source_registry.champion_snapshots)
    imported_policy_ids: list[str] = []
    for source_snapshot in source_snapshots:
        imported_policy_id = _seed_snapshot_policy_id(
            source_run_dir=source_run_dir,
            source_policy_id=source_snapshot.policy_id,
        )
        if imported_policy_id in existing_policy_ids:
            imported_policy_ids.append(imported_policy_id)
            continue
        source_weights_path = source_run_dir / source_snapshot.path
        if not source_weights_path.is_file():
            raise FileNotFoundError(f"Resolved seed snapshot is missing its weights artifact: {source_weights_path}")
        payload = torch.load(source_weights_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise RuntimeError(f"Imported seed snapshot weights payload must be a dict: {source_weights_path}")
        _validate_seed_snapshot_import_contract(
            source_run_dir=source_run_dir,
            payload=payload,
            expected_model_state_dict=expected_model_state_dict,
            expected_config_canonical=expected_config_canonical,
            expected_spec_hash256=expected_spec_hash256,
        )
        snapshot_dir = training_paths.snapshots_dir / imported_policy_id
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        weights_path = snapshot_dir / SNAPSHOT_WEIGHTS_FILENAME
        imported_payload = dict(payload)
        imported_payload["policy_id"] = imported_policy_id
        imported_payload["imported_from_run_dir"] = source_run_dir.as_posix()
        imported_payload["imported_from_policy_id"] = source_snapshot.policy_id
        imported_payload["imported_from_snapshot_path"] = source_snapshot.path
        imported_payload["seeded_from_external_registry"] = True
        torch.save(imported_payload, weights_path)
        weights_sha256 = _sha256_file(weights_path)
        _write_json_file(
            snapshot_dir / SNAPSHOT_METADATA_FILENAME,
            {
                "format": "seeded_train_snapshot_metadata_v1",
                "policy_id": imported_policy_id,
                "update": int(source_snapshot.update),
                "weights_path": snapshot_weights_relpath(imported_policy_id),
                "weights_sha256": weights_sha256,
                "imported_from_run_dir": source_run_dir.as_posix(),
                "imported_from_policy_id": source_snapshot.policy_id,
                "imported_from_snapshot_path": source_snapshot.path,
            },
        )
        registry.add_snapshot(
            policy_id=imported_policy_id,
            update=int(source_snapshot.update),
            weights_sha256=weights_sha256,
            path=weights_path.relative_to(run_dir).as_posix(),
        )
        if source_snapshot.policy_id in source_champions:
            registry.add_champion(imported_policy_id)
        existing_policy_ids.add(imported_policy_id)
        imported_policy_ids.append(imported_policy_id)

    if imported_policy_ids:
        _save_snapshot_registry_with_retention(
            stack=stack,
            training_paths=training_paths,
            run_dir=run_dir,
            registry=registry,
        )
        print(
            "Imported seeded snapshot pool: "
            f"count={len(imported_policy_ids)} "
            f"source_run_dir={source_run_dir.as_posix()}"
        )
    return imported_policy_ids


def _ensure_noleague_baseline_anchor(
    *,
    stack: StackConfig,
    training_paths: TrainingPaths,
    run_dir: Path,
    learner: ImpalaLearner,
    device: torch.device,
    config_hash256: str,
    spec_hash256: str | None = None,
    baseline_run_dir: Path | None = None,
    permit_current_run_alias: bool = False,
    source_checkpoint_path: Path | None = None,
    update: int | None = None,
) -> str | None:
    league = stack.config.league
    training_config = stack.config.training
    experiment_role = _experiment_role(stack)
    requires_anchor = bool(
        league is not None
        and league.enabled
        and league.promotion_gate_enabled
        and _PROMOTION_GATE_NOLEAGUE_BASELINE_NAME in league.promotion_anchor_set_v1.required
    )
    if not requires_anchor and not permit_current_run_alias:
        return None
    if learner.model is None:
        raise RuntimeError("Cannot ensure the NoLeague baseline anchor without a learner model")

    registry_path = training_paths.snapshots_dir / REGISTRY_FILENAME
    registry = SnapshotRegistry.load(registry_path)
    _sync_snapshot_registry_retention(stack, registry)
    available_policy_ids = {snapshot.policy_id for snapshot in registry.snapshots}
    existing_policy_id = next(
        (
            candidate
            for candidate in _promotion_anchor_policy_id_candidates(_PROMOTION_GATE_NOLEAGUE_BASELINE_NAME)
            if candidate in available_policy_ids
        ),
        None,
    )
    if existing_policy_id is not None and baseline_run_dir is None and permit_current_run_alias:
        existing_snapshot = next(
            (snapshot for snapshot in registry.snapshots if snapshot.policy_id == existing_policy_id),
            None,
        )
        resolved_update = int(learner.update_count if update is None else update)
        if existing_snapshot is None or int(existing_snapshot.update) < resolved_update:
            existing_policy_id = None
    if existing_policy_id is not None:
        registry.pin_snapshot(existing_policy_id)
        _save_snapshot_registry_with_retention(
            stack=stack,
            training_paths=training_paths,
            run_dir=run_dir,
            registry=registry,
        )
        return existing_policy_id

    if baseline_run_dir is not None:
        weights_path, weights_sha256, imported_update = _import_noleague_baseline_anchor(
            training_paths=training_paths,
            run_dir=run_dir,
            baseline_run_dir=baseline_run_dir,
            expected_model_state_dict=learner.model.state_dict(),
            expected_config_canonical=canonical_config_dict(stack),
            expected_spec_hash256=spec_hash256,
        )
        registry.add_snapshot(
            policy_id=_PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID,
            update=int(imported_update),
            weights_sha256=weights_sha256,
            path=weights_path.relative_to(run_dir).as_posix(),
        )
        registry.pin_snapshot(_PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID)
        _save_snapshot_registry_with_retention(
            stack=stack,
            training_paths=training_paths,
            run_dir=run_dir,
            registry=registry,
        )
        print(
            "Imported promotion anchor: "
            f"anchor={_PROMOTION_GATE_NOLEAGUE_BASELINE_NAME} "
            f"policy_id={_PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID} "
            f"source_run_dir={Path(baseline_run_dir).resolve()}"
        )
        return _PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID

    if not permit_current_run_alias:
        if requires_anchor:
            raise RuntimeError(
                "The canonical B1 NoLeague baseline is required for this training run. "
                "Pass --b1-baseline-run-dir pointing at a completed baseline_noleague run."
            )
        return None

    resolved_update = int(learner.update_count if update is None else update)
    checkpoint_path = (
        training_paths.checkpoints_dir / _PROMOTION_GATE_NOLEAGUE_BASELINE_CHECKPOINT
        if source_checkpoint_path is None
        else Path(source_checkpoint_path)
    )
    if source_checkpoint_path is None:
        _write_checkpoint(
            checkpoint_path=checkpoint_path,
            learner=learner,
            stack=stack,
            device=device,
            algorithm=str(training_config.algorithm).strip() if training_config is not None else None,
            spec_hash256=spec_hash256,
        )
    weights_path, weights_sha256 = _write_snapshot_artifact(
        snapshots_dir=training_paths.snapshots_dir,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        policy_id=_PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID,
        update=resolved_update,
        config_hash256=config_hash256,
        device=device,
        model_state_dict=learner.model.state_dict(),
    )
    registry.add_snapshot(
        policy_id=_PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID,
        update=resolved_update,
        weights_sha256=weights_sha256,
        path=weights_path.relative_to(run_dir).as_posix(),
    )
    registry.pin_snapshot(_PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID)
    _save_snapshot_registry_with_retention(
        stack=stack,
        training_paths=training_paths,
        run_dir=run_dir,
        registry=registry,
    )
    print(
        "Persisted canonical B1 baseline alias: "
        f"anchor={_PROMOTION_GATE_NOLEAGUE_BASELINE_NAME} "
        f"policy_id={_PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID} "
        f"experiment_role={experiment_role or 'unknown'} update={resolved_update}"
    )
    return _PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID


def _resolve_promotion_anchor_policy_ids(
    *,
    stack: StackConfig,
    registry: SnapshotRegistry,
) -> tuple[dict[str, str], tuple[str, ...]]:
    league = stack.config.league
    if league is None:
        return {}, ()

    available_policy_ids = {snapshot.policy_id for snapshot in registry.snapshots}
    resolved: dict[str, str] = {}
    missing_required: list[str] = []
    anchor_names = [
        *league.promotion_anchor_set_v1.required,
        *league.promotion_anchor_set_v1.optional_if_available,
    ]
    required_names = set(league.promotion_anchor_set_v1.required)

    for anchor_name in anchor_names:
        policy_id = _resolve_symbolic_promotion_anchor_policy_id(anchor_name, registry=registry)
        if policy_id is None:
            candidates = _promotion_anchor_policy_id_candidates(anchor_name)
            policy_id = next((candidate for candidate in candidates if candidate in available_policy_ids), None)
        if policy_id is None and anchor_name == _PROMOTION_GATE_RANDOMLEGAL_NAME:
            policy_id = _PROMOTION_GATE_RANDOMLEGAL_POLICY_ID
        if policy_id is None and heuristic_public_profile_name_for_policy_id(anchor_name) is not None:
            policy_id = anchor_name
        if policy_id is not None:
            resolved[anchor_name] = policy_id
            continue
        if anchor_name in required_names:
            missing_required.append(anchor_name)

    return resolved, tuple(missing_required)


def _snapshot_meta_by_policy_id(registry: SnapshotRegistry) -> dict[str, Any]:
    return {snapshot.policy_id: snapshot for snapshot in registry.snapshots}


def _load_snapshot_eval_model(
    *,
    run_dir: Path,
    snapshot_path: str,
    observation_dim: int,
    action_dim: int,
    stack: StackConfig,
    observation_spec: dict[str, Any] | None = None,
    spec_bundle: dict[str, Any] | None = None,
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


class _PromotionGateRunner:
    def __init__(
        self,
        *,
        stack: StackConfig,
        focal_policy_id: str,
        focal_model: PolicyValueModel,
        anchor_models: dict[str, PolicyValueModel],
        heuristic_policies: dict[str, HeuristicPublicPolicy],
        observation_dim: int,
        action_dim: int,
        pass_action_id: int,
        artifact_dir: Path,
        require_sorted_legal_ids: bool,
    ) -> None:
        self.stack = stack
        self.focal_policy_id = focal_policy_id
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.pass_action_id = pass_action_id
        self.artifact_dir = artifact_dir
        self.require_sorted_legal_ids = require_sorted_legal_ids
        self._policy_models = {focal_policy_id: focal_model, **anchor_models}
        self._heuristic_policies = dict(heuristic_policies)
        self._baseline_logits = np.zeros((action_dim,), dtype=np.float32)
        self._device = torch.device("cpu")

    def run_game(self, scheduled_game: ScheduledGame):
        env = _build_ids_eval_env(
            self.stack,
            seed=scheduled_game.episode_seed,
            pass_action_id=self.pass_action_id,
        )
        seat_hidden = {
            seat: self._initial_hidden(scheduled_game.seat0_policy_id if seat == 0 else scheduled_game.seat1_policy_id)
            for seat in (0, 1)
        }
        seat_rngs = {
            seat: Pcg32XshRrV1(_promotion_gate_rng_seed(scheduled_game=scheduled_game, seat=seat)) for seat in (0, 1)
        }
        last_acting_seat: int | None = None

        try:
            batch = env.reset(seed=scheduled_game.episode_seed)
            self._abort_on_fault(batch)
            while True:
                if bool(batch.terminated[0]) or bool(batch.truncated[0]):
                    return game_result_from_step(
                        batch,
                        env_index=0,
                        acting_seat=last_acting_seat,
                        episode_seed=scheduled_game.episode_seed,
                    )

                current_seat = int(batch.actor[0])
                current_policy_id = (
                    scheduled_game.seat0_policy_id if current_seat == 0 else scheduled_game.seat1_policy_id
                )
                action, next_hidden = self._select_action(
                    batch=batch,
                    current_seat=current_seat,
                    current_policy_id=current_policy_id,
                    seat_hidden=seat_hidden[current_seat],
                    rng=seat_rngs[current_seat],
                )
                seat_hidden[current_seat] = next_hidden
                last_acting_seat = current_seat
                batch = env.step(np.asarray([action], dtype=np.uint32))
                self._abort_on_fault(batch)
        finally:
            env.close()

    def _initial_hidden(self, policy_id: str) -> torch.Tensor | None:
        model = self._policy_models.get(policy_id)
        if model is None:
            return None
        return model.initial_seat_hidden(1, device=self._device)

    def _select_action(
        self,
        *,
        batch: DecisionBoundaryBatch,
        current_seat: int,
        current_policy_id: str,
        seat_hidden: torch.Tensor | None,
        rng: Pcg32XshRrV1,
    ) -> tuple[int, torch.Tensor | None]:
        legal_ids = _legal_ids_for_env_row(
            batch=batch,
            env_index=0,
            require_sorted=self.require_sorted_legal_ids,
        )
        heuristic_policy = self._heuristic_policies.get(current_policy_id)
        if heuristic_policy is not None:
            action = heuristic_policy.choose_action(
                np.asarray(batch.obs[0], dtype=np.float32),
                legal_ids,
            )
            return int(action), seat_hidden
        model = self._policy_models.get(current_policy_id)
        if model is None:
            if current_policy_id != _PROMOTION_GATE_RANDOMLEGAL_POLICY_ID:
                raise RuntimeError(f"Unsupported promotion-gate policy_id: {current_policy_id}")
            action, _ = sample_action_pinned(
                self._baseline_logits,
                legal_ids,
                rng=rng,
                pass_action_id=self.pass_action_id,
            )
            return action, seat_hidden

        if seat_hidden is None:
            raise RuntimeError(f"Missing hidden state for promotion-gate policy_id: {current_policy_id}")

        with torch.inference_mode():
            logits_tensor, _value_tensor, next_seat_hidden = model.forward_seat_aware(
                torch.as_tensor(np.asarray(batch.obs, dtype=np.float32), device=self._device),
                torch.as_tensor([current_seat], device=self._device, dtype=torch.long),
                seat_hidden,
                scoring_mode="learner",
            )
        logits = logits_tensor[0].detach().cpu().numpy().astype(np.float32, copy=False)
        action, _ = sample_action_pinned(
            logits,
            legal_ids,
            rng=rng,
            pass_action_id=self.pass_action_id,
        )
        return action, next_seat_hidden

    def _abort_on_fault(self, batch: DecisionBoundaryBatch) -> None:
        abort_on_engine_fault_eval(
            run_dir=self.artifact_dir,
            engine_status=batch.engine_status,
            decision_id=batch.decision_id,
            episode_key=batch.episode_key,
            note="engine_status!=0 during promotion gate",
        )


def _evaluation_config_or_raise(stack: StackConfig):
    evaluation = stack.config.evaluation
    if evaluation is None:
        raise RuntimeError("The locked stack is missing the evaluation config block")
    return evaluation


def _validate_periodic_dev_eval_contract(stack: StackConfig) -> Any:
    evaluation = _evaluation_config_or_raise(stack)
    if not evaluation.seat_swap:
        raise RuntimeError("Periodic dev eval requires evaluation.seat_swap=true")
    if evaluation.eval_device != "cpu":
        raise RuntimeError(f"Periodic dev eval requires evaluation.eval_device='cpu', got {evaluation.eval_device!r}")
    if not evaluation.eval_inference_mode:
        raise RuntimeError("Periodic dev eval requires evaluation.eval_inference_mode=true")
    if evaluation.eval_sampling_algorithm != "pinned_cdf_pcg_v1":
        raise RuntimeError(
            "Periodic dev eval requires evaluation.eval_sampling_algorithm='pinned_cdf_pcg_v1', "
            f"got {evaluation.eval_sampling_algorithm!r}"
        )
    return evaluation


def _resolve_repo_path(root: Path, path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else root / path


def _resolve_periodic_dev_eval_seed_file(stack: StackConfig) -> tuple[Path, dict[str, str]]:
    evaluation = _evaluation_config_or_raise(stack)
    reproducibility = stack.config.reproducibility
    resolved_paths: dict[str, Path] = {}
    if "dev_eval" in stack.seed_sets:
        resolved_paths["stack.seed_sets.dev_eval"] = stack.seed_sets["dev_eval"]
    if "dev_eval" in evaluation.seed_files:
        resolved_paths["evaluation.seed_files.dev_eval"] = _resolve_repo_path(
            stack.root,
            evaluation.seed_files["dev_eval"],
        )
    if reproducibility is not None and "dev_eval" in reproducibility.seed_files:
        resolved_paths["reproducibility.seed_files.dev_eval"] = _resolve_repo_path(
            stack.root,
            reproducibility.seed_files["dev_eval"],
        )
    if not resolved_paths:
        raise RuntimeError("Periodic dev eval requires a configured dev_eval seed file")

    unique_paths = {path.resolve() for path in resolved_paths.values()}
    if len(unique_paths) != 1:
        mismatch = {name: _json_relative_path(path, root=stack.root) for name, path in resolved_paths.items()}
        raise RuntimeError(f"Periodic dev eval seed file mismatch: {mismatch}")

    seed_file = next(iter(resolved_paths.values()))
    return seed_file, {name: _json_relative_path(path, root=stack.root) for name, path in resolved_paths.items()}


def _periodic_dev_eval_schedule(stack: StackConfig) -> tuple[Path, dict[str, str], list[int], str]:
    evaluation = _validate_periodic_dev_eval_contract(stack)
    seed_file, validated_sources = _resolve_periodic_dev_eval_seed_file(stack)
    all_paired_seeds = parse_seed_file(seed_file)
    required_pairs = int(evaluation.periodic_dev_eval_paired_seeds)
    if len(all_paired_seeds) < required_pairs:
        raise RuntimeError(
            f"Periodic dev eval requires {required_pairs} paired seeds, found {len(all_paired_seeds)} in {seed_file}"
        )
    return seed_file, validated_sources, all_paired_seeds[:required_pairs], hash_seed_file(seed_file)


def _legal_ids_for_env_row(
    *,
    batch: DecisionBoundaryBatch,
    env_index: int,
    require_sorted: bool,
) -> np.ndarray:
    if batch.ids_offsets is None:
        raise RuntimeError("Expected ids_offsets legality during periodic dev eval")
    legal_ids, legal_offsets = batch.ids_offsets
    start = int(legal_offsets[env_index])
    end = int(legal_offsets[env_index + 1])
    row = np.asarray(legal_ids[start:end], dtype=np.uint32)
    if require_sorted:
        assert_strictly_increasing_legal_ids(row)
    return row


def _periodic_dev_eval_rng_seed(*, scheduled_game: ScheduledGame, seat: int) -> int:
    payload = canonical_json_bytes(
        {
            "kind": "periodic_dev_eval_rng_v1",
            "pair_index": scheduled_game.pair_index,
            "swap_index": scheduled_game.swap_index,
            "episode_seed": scheduled_game.episode_seed,
            "seat": int(seat),
            "seat_policy_id": scheduled_game.seat0_policy_id if seat == 0 else scheduled_game.seat1_policy_id,
        }
    )
    return stable_hash64(payload)


def _promotion_gate_rng_seed(*, scheduled_game: ScheduledGame, seat: int) -> int:
    payload = canonical_json_bytes(
        {
            "kind": "promotion_gate_rng_v1",
            "pair_index": scheduled_game.pair_index,
            "swap_index": scheduled_game.swap_index,
            "episode_seed": scheduled_game.episode_seed,
            "seat": int(seat),
            "seat_policy_id": scheduled_game.seat0_policy_id if seat == 0 else scheduled_game.seat1_policy_id,
        }
    )
    return stable_hash64(payload)


def _periodic_dev_eval_bootstrap_seed(*, update_count: int, policy_version: int) -> int:
    return stable_hash64(
        canonical_json_bytes(
            {
                "kind": "periodic_dev_eval_bootstrap_v1",
                "update_count": int(update_count),
                "policy_version": int(policy_version),
            }
        )
    )


def _promotion_gate_bootstrap_seed(*, update_count: int, policy_version: int) -> int:
    return stable_hash64(
        canonical_json_bytes(
            {
                "kind": "promotion_gate_bootstrap_v1",
                "update_count": int(update_count),
                "policy_version": int(policy_version),
            }
        )
    )


def _clone_cpu_eval_model(
    *,
    learner_model: PolicyValueModel,
    observation_dim: int,
    action_dim: int,
    stack: StackConfig,
    observation_spec: dict[str, Any] | None = None,
    spec_bundle: dict[str, Any] | None = None,
) -> PolicyValueModel:
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
    cpu_state_dict = {name: value.detach().cpu().clone() for name, value in learner_model.state_dict().items()}
    eval_model.load_state_dict(cpu_state_dict)
    eval_model.eval()
    return eval_model


def _current_focal_policy_id(*, learner: ImpalaLearner) -> str:
    return f"train_u{int(learner.update_count)}_p{int(learner.get_policy_version())}"


def _checkpoint_path_for_update(checkpoints_dir: Path, *, update_count: int) -> Path:
    return checkpoints_dir / f"checkpoint_{update_count}.pt"


def _ensure_current_checkpoint(
    *,
    training_paths: TrainingPaths,
    learner: ImpalaLearner,
    stack: StackConfig,
    device: torch.device,
    spec_hash256: str | None = None,
    algorithm: str | None = None,
) -> Path:
    checkpoint_path = _checkpoint_path_for_update(
        training_paths.checkpoints_dir,
        update_count=int(learner.update_count),
    )
    if checkpoint_path.is_file():
        return checkpoint_path

    _write_checkpoint(
        checkpoint_path=checkpoint_path,
        learner=learner,
        stack=stack,
        device=device,
        spec_hash256=spec_hash256,
        algorithm=algorithm,
    )
    return checkpoint_path


def _should_run_periodic_dev_eval(stack: StackConfig, *, update_count: int) -> bool:
    evaluation = stack.config.evaluation
    if evaluation is None:
        return False
    interval = int(evaluation.periodic_dev_eval_interval_updates)
    return interval > 0 and update_count % interval == 0


def _periodic_dev_eval_summaries_path(training_paths: TrainingPaths) -> Path:
    return training_paths.logs_dir / "periodic_dev_eval_summaries.json"


def _stall_monitor_state_path(training_paths: TrainingPaths) -> Path:
    return training_paths.logs_dir / "stall_monitor.json"


def _periodic_dev_eval_opponents(
    *,
    stack: StackConfig,
    contract: SimulatorContract,
    run_dir: Path,
    observation_dim: int,
    action_dim: int,
) -> list[tuple[str, str, PolicyValueModel | None, HeuristicPublicPolicy | None]]:
    registry_path = ArtifactLayout.from_run_dir(run_dir).training_snapshots_dir / REGISTRY_FILENAME
    registry = SnapshotRegistry.load(registry_path) if registry_path.is_file() else SnapshotRegistry()
    anchor_policy_ids, missing_required = _resolve_promotion_anchor_policy_ids(
        stack=stack,
        registry=registry,
    )
    if missing_required:
        missing_text = ",".join(missing_required)
        raise RuntimeError(f"Periodic dev eval is missing required anchors: {missing_text}")

    league = stack.config.league
    anchor_names: list[str]
    if league is None:
        anchor_names = [_PROMOTION_GATE_RANDOMLEGAL_NAME, _PROMOTION_GATE_NOLEAGUE_BASELINE_NAME]
    else:
        anchor_names = [
            *league.promotion_anchor_set_v1.required,
            *league.promotion_anchor_set_v1.optional_if_available,
        ]

    snapshot_index = _snapshot_meta_by_policy_id(registry)
    observation_spec = cast(dict[str, Any] | None, contract.spec_bundle.get("observation"))
    spec_bundle = cast(dict[str, Any] | None, contract.spec_bundle)
    opponents: list[tuple[str, str, PolicyValueModel | None, HeuristicPublicPolicy | None]] = []
    for anchor_name in anchor_names:
        policy_id = anchor_policy_ids.get(anchor_name)
        if policy_id is None:
            continue
        if policy_id == _PROMOTION_GATE_RANDOMLEGAL_POLICY_ID:
            opponents.append((policy_id, anchor_name, None, None))
            continue
        heuristic_profile = heuristic_public_profile_name_for_policy_id(policy_id)
        if heuristic_profile is not None:
            try:
                heuristic_policy = _build_heuristic_public_policy(
                    contract.spec_bundle,
                    scoring_profile=heuristic_profile,
                )
            except Exception as exc:
                if league is not None and anchor_name in league.promotion_anchor_set_v1.required:
                    raise RuntimeError(
                        f"Periodic dev eval requires a heuristic-compatible simulator contract for {policy_id}"
                    ) from exc
                continue
            opponents.append((policy_id, anchor_name, None, heuristic_policy))
            continue
        snapshot = snapshot_index.get(policy_id)
        if snapshot is None:
            if league is not None and anchor_name in league.promotion_anchor_set_v1.required:
                raise RuntimeError(f"Periodic dev eval could not resolve required snapshot anchor {anchor_name!r}")
            continue
        opponents.append(
            (
                policy_id,
                anchor_name,
                _load_snapshot_eval_model(
                    run_dir=run_dir,
                    snapshot_path=snapshot.path,
                    stack=stack,
                    observation_dim=observation_dim,
                    action_dim=action_dim,
                    observation_spec=observation_spec,
                    spec_bundle=spec_bundle,
                ),
                None,
            )
        )
    return opponents


def _persist_periodic_dev_eval_summary(
    *,
    training_paths: TrainingPaths,
    payload: Mapping[str, Any],
) -> None:
    focal_policy_id = str(payload.get("policy_id", "")).strip()
    if not focal_policy_id:
        return
    path = _periodic_dev_eval_summaries_path(training_paths)
    summaries = _load_json_object(path, label="periodic dev-eval summaries") if path.is_file() else {}
    summaries[focal_policy_id] = {
        "aggregate_score": float(payload.get("aggregate_score", 0.0)),
        "anchor_scores": dict(cast(Mapping[str, Any], payload.get("anchor_scores", {}))),
        "update_count": int(payload.get("update_count", 0)),
        "policy_version": int(payload.get("policy_version", 0)),
    }
    _write_json(path, summaries)


def _update_stall_monitor(
    *,
    stack: StackConfig,
    training_paths: TrainingPaths,
    update_count: int,
    summary_payload: Mapping[str, Any],
) -> dict[str, Any] | None:
    curriculum = stack.config.curriculum
    if curriculum is None or not curriculum.stall_monitor.enabled:
        return None
    threshold = float(curriculum.stall_monitor.truncation_rate_threshold)
    required_consecutive = int(curriculum.stall_monitor.consecutive_evals)
    anchors_raw = summary_payload.get("anchors", {})
    if not isinstance(anchors_raw, Mapping):
        return None

    anchor_truncation_rates: dict[str, float] = {}
    anchor_no_progress_rates: dict[str, float] = {}
    anchor_natural_timeout_rates: dict[str, float] = {}
    anchor_stall_rates: dict[str, float] = {}
    for anchor_name, anchor_payload in anchors_raw.items():
        if not isinstance(anchor_payload, Mapping):
            continue
        matchup_summary = anchor_payload.get("summary", {})
        if not isinstance(matchup_summary, Mapping):
            continue
        truncation_rate = _summary_rate(matchup_summary, "truncations")
        no_progress_rate = _summary_rate(matchup_summary, "no_progress_timeouts")
        natural_timeout_rate = _summary_rate(matchup_summary, "natural_timeouts")
        if truncation_rate is None and no_progress_rate is None and natural_timeout_rate is None:
            continue
        anchor_truncation_rates[anchor_name] = 0.0 if truncation_rate is None else truncation_rate
        anchor_no_progress_rates[anchor_name] = 0.0 if no_progress_rate is None else no_progress_rate
        anchor_natural_timeout_rates[anchor_name] = 0.0 if natural_timeout_rate is None else natural_timeout_rate
        anchor_stall_rates[anchor_name] = (
            anchor_no_progress_rates[anchor_name]
            if no_progress_rate is not None
            else anchor_truncation_rates[anchor_name]
        )
    if not anchor_stall_rates:
        return None

    state_path = _stall_monitor_state_path(training_paths)
    state = _load_json_object(state_path, label="stall monitor state") if state_path.is_file() else {}
    previous_consecutive = int(state.get("consecutive_trigger_count", 0))
    worst_anchor = max(anchor_stall_rates, key=anchor_stall_rates.get)
    worst_rate = float(anchor_stall_rates[worst_anchor])
    consecutive = previous_consecutive + 1 if worst_rate >= threshold else 0
    stall_risk = consecutive >= required_consecutive
    payload = {
        "enabled": True,
        "update_count": int(update_count),
        "threshold": threshold,
        "required_consecutive_evals": required_consecutive,
        "consecutive_trigger_count": consecutive,
        "stall_risk": stall_risk,
        "worst_anchor": worst_anchor,
        "stall_indicator_kind": (
            "no_progress_timeout" if anchor_no_progress_rates.get(worst_anchor, 0.0) > 0.0 else "truncation_fallback"
        ),
        "worst_stall_rate": worst_rate,
        "worst_truncation_rate": float(anchor_truncation_rates.get(worst_anchor, 0.0)),
        "worst_no_progress_timeout_rate": float(anchor_no_progress_rates.get(worst_anchor, 0.0)),
        "worst_natural_timeout_rate": float(anchor_natural_timeout_rates.get(worst_anchor, 0.0)),
        "anchor_truncation_rates": anchor_truncation_rates,
        "anchor_no_progress_timeout_rates": anchor_no_progress_rates,
        "anchor_natural_timeout_rates": anchor_natural_timeout_rates,
    }
    _write_json(state_path, payload)
    return payload


def _maybe_rollback_to_best_checkpoint(
    *,
    stack: StackConfig,
    training_paths: TrainingPaths,
    artifacts: RunArtifacts,
    runtime: QueueRuntime,
    learner: ImpalaLearner,
    model: PolicyValueModel,
    device: torch.device,
    spec_hash256: str,
    algorithm: str,
    latest_metrics: Mapping[str, float] | None,
    dev_eval_summary: Mapping[str, Any] | None,
    last_rollback_update: int | None,
) -> dict[str, Any] | None:
    curriculum = stack.config.curriculum
    if curriculum is None:
        return None
    checkpoint_guard = curriculum.checkpoint_guard
    if not checkpoint_guard.enabled or dev_eval_summary is None:
        return None
    if last_rollback_update is not None and (int(learner.update_count) - int(last_rollback_update)) < int(
        checkpoint_guard.cooldown_updates
    ):
        return None

    current_score = _dev_eval_aggregate_score(dev_eval_summary)
    if current_score is None:
        return None
    worst_truncation_rate = _dev_eval_worst_truncation_rate(dev_eval_summary)
    worst_stall_rate = _dev_eval_worst_stall_rate(dev_eval_summary)
    worst_no_progress_timeout_rate = _dev_eval_worst_no_progress_timeout_rate(dev_eval_summary)
    worst_natural_timeout_rate = _dev_eval_worst_natural_timeout_rate(dev_eval_summary)
    tracker = _load_checkpoint_tracker(training_paths)
    best_record = tracker.get("best")
    if not isinstance(best_record, Mapping):
        return None
    best_metric_kind = str(best_record.get("metric_kind", "")).strip()
    best_metric_value = best_record.get("metric_value")
    best_update_count = best_record.get("update_count")
    if best_metric_kind != "dev_eval_mean":
        return None
    if not isinstance(best_metric_value, (int, float)) or not np.isfinite(float(best_metric_value)):
        return None
    if not isinstance(best_update_count, int) or int(best_update_count) >= int(learner.update_count):
        return None
    best_score = float(best_metric_value)
    if best_score < float(checkpoint_guard.min_best_score):
        return None

    confidence = _dev_eval_confidence_stats(dev_eval_summary)
    rollback_reasons: list[str] = []
    if current_score <= best_score - float(checkpoint_guard.rollback_score_margin):
        rollback_reasons.append("score_drop")
    if worst_stall_rate is not None and (
        worst_stall_rate >= float(checkpoint_guard.rollback_truncation_rate_threshold)
    ):
        rollback_reasons.append("truncation")
    max_prob_lt_half = confidence["max_prob_lt_half"]
    if max_prob_lt_half is not None and (float(max_prob_lt_half) >= float(checkpoint_guard.rollback_max_prob_lt_half)):
        rollback_reasons.append("confidence")
    if not rollback_reasons:
        return None

    best_checkpoint_path = training_paths.best_checkpoint_path
    _restore_learner_from_checkpoint(
        checkpoint_path=best_checkpoint_path,
        learner=learner,
        stack=stack,
        device=device,
        expected_spec_hash256=spec_hash256,
        algorithm=algorithm,
        restore_counters=False,
    )
    demoted_champions = _demote_registry_champions_newer_than(
        training_paths,
        update_count=int(best_update_count),
    )
    publish_metrics = runtime.maybe_publish_snapshot(
        learner_model=model,
        learner_update_count=int(learner.update_count),
        force=True,
    )
    runtime.reset_outcome_tracker()
    runtime.refresh_opponent_pool()
    _write_checkpoint(
        checkpoint_path=training_paths.latest_checkpoint_path,
        learner=learner,
        stack=stack,
        device=device,
        spec_hash256=spec_hash256,
        algorithm=algorithm,
    )
    tracker["latest"] = _build_checkpoint_record(
        alias_name="latest",
        alias_path=training_paths.latest_checkpoint_path,
        source_checkpoint_path=best_checkpoint_path,
        artifacts=artifacts,
        learner=learner,
        metric_kind="dev_eval_mean",
        metric_value=best_score,
    )
    _write_checkpoint_tracker(training_paths, tracker)

    payload = {
        "format": "checkpoint_guard_event_v1",
        "action": "rollback_to_best",
        "update_count": int(learner.update_count),
        "policy_version": int(learner.get_policy_version()),
        "current_score": current_score,
        "best_score": best_score,
        "best_update_count": int(best_update_count),
        "worst_stall_rate": worst_stall_rate,
        "worst_truncation_rate": worst_truncation_rate,
        "worst_no_progress_timeout_rate": worst_no_progress_timeout_rate,
        "worst_natural_timeout_rate": worst_natural_timeout_rate,
        "min_prob_gt_half": confidence["min_prob_gt_half"],
        "max_prob_lt_half": confidence["max_prob_lt_half"],
        "max_ci_half_width": confidence["max_ci_half_width"],
        "reasons": rollback_reasons,
        "best_checkpoint_path": _relative_path_text(best_checkpoint_path, root=artifacts.run_dir),
        "latest_checkpoint_path": _relative_path_text(training_paths.latest_checkpoint_path, root=artifacts.run_dir),
        "snapshot_publish_latency_ms": publish_metrics.get("snapshot_publish_latency_ms", 0.0),
        "snapshot_apply_latency_ms": publish_metrics.get("snapshot_apply_latency_ms", 0.0),
        "latest_loss": None if latest_metrics is None else latest_metrics.get("loss"),
        "demoted_champions": demoted_champions,
    }
    _append_checkpoint_guard_event(training_paths, payload)
    return payload


def _maybe_finalize_from_best_checkpoint(
    *,
    stack: StackConfig,
    training_paths: TrainingPaths,
    artifacts: RunArtifacts,
    runtime: QueueRuntime,
    learner: ImpalaLearner,
    device: torch.device,
    spec_hash256: str,
    algorithm: str,
    latest_metrics: Mapping[str, float] | None,
    dev_eval_summary: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    curriculum = stack.config.curriculum
    if curriculum is None or not curriculum.checkpoint_guard.enabled:
        return None
    best_record = _best_checkpoint_record(training_paths)
    if best_record is None:
        return None
    best_metric_kind = str(best_record.get("metric_kind", "")).strip()
    best_metric_value = best_record.get("metric_value")
    best_update_count = best_record.get("update_count")
    if best_metric_kind != "dev_eval_mean":
        return None
    if not isinstance(best_metric_value, (int, float)) or not np.isfinite(float(best_metric_value)):
        return None
    if not isinstance(best_update_count, int):
        return None
    current_score = _dev_eval_aggregate_score(dev_eval_summary)
    best_score = float(best_metric_value)
    if current_score is None or current_score >= best_score:
        return None
    confidence = _dev_eval_confidence_stats(dev_eval_summary)
    best_checkpoint_path = training_paths.best_checkpoint_path
    _restore_learner_from_checkpoint(
        checkpoint_path=best_checkpoint_path,
        learner=learner,
        stack=stack,
        device=device,
        expected_spec_hash256=spec_hash256,
        algorithm=algorithm,
        restore_counters=False,
    )
    demoted_champions = _demote_registry_champions_newer_than(
        training_paths,
        update_count=int(best_update_count),
    )
    runtime.reset_outcome_tracker()
    runtime.refresh_opponent_pool()
    final_checkpoint_path = _ensure_current_checkpoint(
        training_paths=training_paths,
        learner=learner,
        stack=stack,
        device=device,
        spec_hash256=spec_hash256,
        algorithm=algorithm,
    )
    tracker = _load_checkpoint_tracker(training_paths)
    tracker["latest"] = _build_checkpoint_record(
        alias_name="latest",
        alias_path=training_paths.latest_checkpoint_path,
        source_checkpoint_path=best_checkpoint_path,
        artifacts=artifacts,
        learner=learner,
        metric_kind="dev_eval_mean",
        metric_value=best_score,
    )
    shutil.copy2(final_checkpoint_path, training_paths.latest_checkpoint_path)
    _write_checkpoint_tracker(training_paths, tracker)
    payload = {
        "format": "checkpoint_guard_event_v1",
        "action": "finalize_to_best",
        "update_count": int(learner.update_count),
        "policy_version": int(learner.get_policy_version()),
        "current_score": current_score,
        "best_score": best_score,
        "best_update_count": int(best_update_count),
        "min_prob_gt_half": confidence["min_prob_gt_half"],
        "max_prob_lt_half": confidence["max_prob_lt_half"],
        "max_ci_half_width": confidence["max_ci_half_width"],
        "latest_loss": None if latest_metrics is None else latest_metrics.get("loss"),
        "best_checkpoint_path": _relative_path_text(best_checkpoint_path, root=artifacts.run_dir),
        "latest_checkpoint_path": _relative_path_text(training_paths.latest_checkpoint_path, root=artifacts.run_dir),
        "demoted_champions": demoted_champions,
    }
    _append_checkpoint_guard_event(training_paths, payload)
    return payload


def _run_periodic_dev_eval(
    *,
    stack: StackConfig,
    contract: SimulatorContract,
    artifacts: Any,
    training_paths: TrainingPaths,
    learner: ImpalaLearner,
    device: torch.device,
    run_id256: str,
    config_hash256: str,
    spec_hash256: str,
    artifact_dir_name: str = "dev_eval",
    artifact_scope: str = "periodic_dev_eval",
    paired_seeds_override: Sequence[int] | None = None,
    persist_summary: bool = True,
    update_stall_monitor: bool = True,
) -> dict[str, Any]:
    if learner.model is None:
        raise RuntimeError("Periodic dev eval requires an attached learner model")

    evaluation = _validate_periodic_dev_eval_contract(stack)
    seed_file, validated_sources, scheduled_paired_seeds, seed_file_sha256 = _periodic_dev_eval_schedule(stack)
    paired_seeds = (
        [int(seed) for seed in paired_seeds_override]
        if paired_seeds_override is not None
        else [int(seed) for seed in scheduled_paired_seeds]
    )
    if not paired_seeds:
        raise RuntimeError("Periodic dev eval requires at least one paired seed")
    observation_dim, action_dim = _spec_dimensions(contract)
    pass_action_id = int(contract.spec_bundle["action"]["pass_action_id"])
    update_count = int(learner.update_count)
    policy_version = int(learner.get_policy_version())
    focal_policy_id = _current_focal_policy_id(learner=learner)
    checkpoint_path = _ensure_current_checkpoint(
        training_paths=training_paths,
        learner=learner,
        stack=stack,
        device=device,
        spec_hash256=spec_hash256,
        algorithm=str(stack.config.training.algorithm).strip() if stack.config.training is not None else None,
    )

    update_dir = artifacts.run_dir / "eval" / artifact_dir_name / f"update_{update_count}"
    matchup_dir = update_dir / "b0_randomlegal"
    eval_model = _clone_cpu_eval_model(
        learner_model=cast(PolicyValueModel, learner.model),
        observation_dim=observation_dim,
        action_dim=action_dim,
        stack=stack,
        observation_spec=cast(dict[str, Any] | None, contract.spec_bundle.get("observation")),
        spec_bundle=cast(dict[str, Any] | None, contract.spec_bundle),
    )
    opponents = _periodic_dev_eval_opponents(
        stack=stack,
        contract=contract,
        run_dir=artifacts.run_dir,
        observation_dim=observation_dim,
        action_dim=action_dim,
    )

    anchor_payloads: dict[str, dict[str, Any]] = {}
    anchor_scores: dict[str, float] = {}
    primary_summary: dict[str, Any] | None = None
    for opponent_policy_id, display_name, opponent_model, heuristic_policy in opponents:
        matchup_dir = update_dir / opponent_policy_id
        runner = _PeriodicDevEvalRunner(
            stack=stack,
            model=eval_model,
            opponent_policy_id=opponent_policy_id,
            opponent_model=opponent_model,
            heuristic_policy=heuristic_policy,
            observation_dim=observation_dim,
            action_dim=action_dim,
            pass_action_id=pass_action_id,
            artifact_dir=matchup_dir,
            focal_policy_id=focal_policy_id,
            require_sorted_legal_ids=bool(evaluation.eval_assert_sorted_legal_ids),
        )

        seed_usage_payload = {
            "seed_set": "dev_eval",
            "seed_file": {
                "path": _json_relative_path(seed_file, root=stack.root),
                "sha256": seed_file_sha256,
                "validated_sources": validated_sources,
            },
            "artifact_scope": artifact_scope,
            "seed_schedule": {
                "configured_paired_seed_count": len(scheduled_paired_seeds),
                "requested_paired_seed_count": len(paired_seeds),
                "expanded_beyond_seed_file": len(paired_seeds) > len(scheduled_paired_seeds),
            },
            "paired_seed_count": len(paired_seeds),
            "paired_seeds": list(paired_seeds),
            "protocol": {
                "seat_swap": bool(evaluation.seat_swap),
                "eval_device": evaluation.eval_device,
                "eval_inference_mode": bool(evaluation.eval_inference_mode),
                "eval_sampling_algorithm": evaluation.eval_sampling_algorithm,
                "eval_assert_sorted_legal_ids": bool(evaluation.eval_assert_sorted_legal_ids),
            },
            "focal_policy": {
                "policy_id": focal_policy_id,
                "update_count": update_count,
                "policy_version": policy_version,
                "checkpoint_path": (
                    None if checkpoint_path is None else _json_relative_path(checkpoint_path, root=artifacts.run_dir)
                ),
            },
            "opponent_policy": {
                "policy_id": opponent_policy_id,
                "display_name": display_name,
            },
        }
        _write_json(matchup_dir / "seed_usage.json", seed_usage_payload)

        matchup = run_seat_swapped_matchup(
            focal_policy_id=focal_policy_id,
            opponent_policy_id=opponent_policy_id,
            paired_seeds=paired_seeds,
            runner=runner,
            episodes_path=matchup_dir / "episodes.jsonl",
            run_id256=run_id256,
            config_hash256=config_hash256,
            spec_hash256=spec_hash256,
        )

        matchup_payload = build_matchup_export(
            matchup.records,
            stop_rules=evaluation.stop_rules,
            max_paired_seeds=len(paired_seeds),
            scheme=cast(PayoffFoldScheme, evaluation.final_policy_set_selection.folding),
            sample_count=1000,
            seed=_periodic_dev_eval_bootstrap_seed(update_count=update_count, policy_version=policy_version),
        )
        matchup_payload["evaluation_context"] = {
            "artifact_scope": artifact_scope,
            "update_count": update_count,
            "policy_version": policy_version,
            "checkpoint_path": (
                None if checkpoint_path is None else _json_relative_path(checkpoint_path, root=artifacts.run_dir)
            ),
            "seed_usage_path": _json_relative_path(matchup_dir / "seed_usage.json", root=artifacts.run_dir),
            "anchor_display_name": display_name,
        }
        write_matchup_summary_json(matchup_dir / "matchup_summary.json", matchup_payload)
        write_matchup_summary_csv(matchup_dir / "matchup_summary.csv", matchup_payload)
        write_matchup_diagnostics_json(
            matchup_dir / "diagnostics.json",
            build_seat_advantage_diagnostics(matchup.records),
        )
        anchor_payloads[display_name] = matchup_payload
        anchor_scores[display_name] = float(matchup_payload["uncertainty"]["mean"])
        if primary_summary is None or opponent_policy_id == "b0_randomlegal":
            primary_summary = matchup_payload

    if primary_summary is None:
        raise RuntimeError("Periodic dev eval did not produce any matchup summaries")

    aggregate_score = sum(anchor_scores.values()) / max(1, len(anchor_scores))
    summary_payload = dict(primary_summary)
    summary_payload.update(
        {
            "policy_id": focal_policy_id,
            "update_count": update_count,
            "policy_version": policy_version,
            "aggregate_score": aggregate_score,
            "anchor_scores": anchor_scores,
            "anchors": anchor_payloads,
        }
    )
    if persist_summary:
        _persist_periodic_dev_eval_summary(training_paths=training_paths, payload=summary_payload)
    if update_stall_monitor:
        stall_monitor = _update_stall_monitor(
            stack=stack,
            training_paths=training_paths,
            update_count=update_count,
            summary_payload=summary_payload,
        )
        if stall_monitor is not None:
            summary_payload["stall_monitor"] = stall_monitor
            if bool(stall_monitor.get("stall_risk", False)):
                print(
                    "Stall monitor warning: "
                    f"update={update_count} worst_anchor={stall_monitor['worst_anchor']} "
                    f"stall_rate={float(stall_monitor['worst_stall_rate']):.3f} "
                    f"no_progress_rate={float(stall_monitor['worst_no_progress_timeout_rate']):.3f} "
                    f"truncation_rate={float(stall_monitor['worst_truncation_rate']):.3f} "
                    f"consecutive={int(stall_monitor['consecutive_trigger_count'])}"
                )
    _write_json(update_dir / "summary.json", summary_payload)
    return summary_payload


def _run_snapshot_promotion_gate(
    *,
    stack: StackConfig,
    contract: SimulatorContract,
    artifacts: Any,
    training_paths: TrainingPaths,
    learner: ImpalaLearner,
    candidate_policy_id: str,
    update_count: int,
    league_reference_update: int | None,
    policy_version: int,
    run_id256: str,
    config_hash256: str,
    spec_hash256: str,
) -> bool | None:
    league = stack.config.league
    if league is None or not league.enabled or not league.promotion_gate_enabled:
        return None
    reference_update = int(update_count if league_reference_update is None else league_reference_update)
    if reference_update < int(league.warmup.first_updates):
        print(
            "Promotion gate skipped during league warmup: "
            f"update={update_count} effective_update={reference_update} threshold={int(league.warmup.first_updates)} "
            f"candidate={candidate_policy_id}"
        )
        return None
    if learner.model is None:
        raise RuntimeError("Promotion gate requires an attached learner model")

    evaluation = _validate_periodic_dev_eval_contract(stack)
    registry_path = training_paths.snapshots_dir / REGISTRY_FILENAME
    registry = SnapshotRegistry.load(registry_path)
    anchor_policy_ids, missing_required = _resolve_promotion_anchor_policy_ids(
        stack=stack,
        registry=registry,
    )
    if missing_required:
        print(
            "Promotion gate skipped: "
            f"update={update_count} candidate={candidate_policy_id} "
            f"missing_anchors={','.join(missing_required)}"
        )
        return None

    observation_dim, action_dim = _spec_dimensions(contract)
    snapshot_index = _snapshot_meta_by_policy_id(registry)
    anchor_models = {
        policy_id: _load_snapshot_eval_model(
            run_dir=artifacts.run_dir,
            snapshot_path=snapshot_index[policy_id].path,
            observation_dim=observation_dim,
            action_dim=action_dim,
            stack=stack,
            observation_spec=cast(dict[str, Any] | None, contract.spec_bundle.get("observation")),
            spec_bundle=cast(dict[str, Any] | None, contract.spec_bundle),
        )
        for policy_id in set(anchor_policy_ids.values())
        if policy_id != _PROMOTION_GATE_RANDOMLEGAL_POLICY_ID
        and heuristic_public_profile_name_for_policy_id(policy_id) is None
    }
    heuristic_policies: dict[str, HeuristicPublicPolicy] = {}
    heuristic_policy_ids = {
        policy_id
        for policy_id in set(anchor_policy_ids.values())
        if heuristic_public_profile_name_for_policy_id(policy_id) is not None
    }
    if heuristic_policy_ids:
        try:
            heuristic_policies = {
                policy_id: _build_heuristic_public_policy(
                    contract.spec_bundle,
                    scoring_profile=cast(str, heuristic_public_profile_name_for_policy_id(policy_id)),
                )
                for policy_id in heuristic_policy_ids
            }
        except Exception as exc:
            assert league is not None
            missing_required = [
                policy_id for policy_id in heuristic_policy_ids if policy_id in league.promotion_anchor_set_v1.required
            ]
            if missing_required:
                missing_text = ", ".join(missing_required)
                raise RuntimeError(
                    f"Promotion gate requires a heuristic-compatible simulator contract for {missing_text}"
                ) from exc
            anchor_policy_ids = {
                anchor_name: policy_id
                for anchor_name, policy_id in anchor_policy_ids.items()
                if heuristic_public_profile_name_for_policy_id(policy_id) is None
            }
            print(
                "Promotion gate note: skipping optional heuristic-public anchors because the active simulator contract "
                f"does not expose the required public action/observation metadata ({exc})."
            )
    runner = _PromotionGateRunner(
        stack=stack,
        focal_policy_id=candidate_policy_id,
        focal_model=_clone_cpu_eval_model(
            learner_model=cast(PolicyValueModel, learner.model),
            observation_dim=observation_dim,
            action_dim=action_dim,
            stack=stack,
            observation_spec=cast(dict[str, Any] | None, contract.spec_bundle.get("observation")),
            spec_bundle=cast(dict[str, Any] | None, contract.spec_bundle),
        ),
        anchor_models=anchor_models,
        heuristic_policies=heuristic_policies,
        observation_dim=observation_dim,
        action_dim=action_dim,
        pass_action_id=int(contract.spec_bundle["action"]["pass_action_id"]),
        artifact_dir=artifacts.run_dir / "eval" / "promotion_gate" / f"update_{update_count}",
        require_sorted_legal_ids=bool(evaluation.eval_assert_sorted_legal_ids),
    )
    result = run_promotion_gate(
        stack=stack,
        run_dir=artifacts.run_dir / "eval" / "promotion_gate" / f"update_{update_count}",
        focal_policy_id=candidate_policy_id,
        anchor_policy_ids=anchor_policy_ids,
        runner=runner,
        run_id256=run_id256,
        config_hash256=config_hash256,
        spec_hash256=spec_hash256,
        bootstrap_seed=_promotion_gate_bootstrap_seed(
            update_count=update_count,
            policy_version=policy_version,
        ),
    )
    if result.passed:
        registry.add_champion(candidate_policy_id)
        _save_snapshot_registry_with_retention(
            stack=stack,
            training_paths=training_paths,
            run_dir=artifacts.run_dir,
            registry=registry,
        )
        print(
            "Promotion gate passed: "
            f"update={update_count} candidate={candidate_policy_id} "
            f"anchors={','.join(result.ordered_opponents)}"
        )
        return True

    reason_codes = ",".join(str(reason.get("code", "unknown")) for reason in result.reasons) or "unknown"
    print(f"Promotion gate failed: update={update_count} candidate={candidate_policy_id} reasons={reason_codes}")
    return False


def _run_minimal_training(
    *,
    stack: StackConfig,
    contract: SimulatorContract,
    artifacts: Any,
    num_envs: int,
    unroll_length: int,
    max_updates: int,
    profile: str,
    device: torch.device,
    seed: int,
    checkpoint_interval_updates: int,
    run_id256: str,
    config_hash256: str,
    spec_hash256: str,
    runtime_mode: QueueRuntimeMode,
    b1_baseline_run_dir: Path | None,
    seed_snapshot_run_dir: Path | None = None,
    profile_timers: bool = False,
    torch_profiler: bool = False,
    resume_checkpoint_path: Path | None = None,
    tensorboard_logger: TensorBoardLogger | None = None,
) -> dict[str, float]:
    _configure_torch_threads(stack)
    torch.manual_seed(seed)
    np.random.seed(seed & 0xFFFF_FFFF)

    observation_dim, action_dim = _spec_dimensions(contract)
    training_config = stack.config.training
    model_config = stack.config.model
    environment_config = stack.config.environment
    rewards_config = stack.config.rewards
    experiment_role = _experiment_role(stack)
    if training_config is None or model_config is None or environment_config is None or rewards_config is None:
        raise RuntimeError("The locked stack is missing training, model, environment, or rewards config")

    training_paths = _training_paths(artifacts.run_dir)
    pass_action_id = int(contract.spec_bundle["action"]["pass_action_id"])
    algorithm = str(training_config.algorithm).strip()
    _validate_algorithm_model_contract(
        algorithm=algorithm,
        recurrent_core=model_config.recurrent_core,
        encoder_kind=model_config.encoder_kind,
    )
    model = build_policy_value_model(
        observation_dim=observation_dim,
        config=model_config,
        action_dim=action_dim,
        observation_spec=contract.spec_bundle.get("observation"),
        spec_bundle=contract.spec_bundle,
    ).to(device)
    compiled_model = _maybe_compile_learner_model(
        model=model,
        training_config=training_config,
        device=device,
    )
    learner = _build_training_learner(
        algorithm=algorithm,
        model=model,
        compiled_model=compiled_model,
        training_config=training_config,
        training_paths=training_paths,
        pass_action_id=pass_action_id,
        checkpoint_interval_updates=checkpoint_interval_updates,
    )
    resume_state = None
    if resume_checkpoint_path is not None:
        resume_state = _restore_learner_from_checkpoint(
            checkpoint_path=resume_checkpoint_path,
            learner=learner,
            stack=stack,
            device=device,
            expected_spec_hash256=spec_hash256,
            algorithm=algorithm,
        )
        print(
            "Resumed learner state: "
            f"checkpoint={resume_state.checkpoint_path} "
            f"update={resume_state.update_count} "
            f"policy_version={resume_state.policy_version}"
        )

    config_hash256 = compute_config_hash256(stack)
    _ensure_noleague_baseline_anchor(
        stack=stack,
        training_paths=training_paths,
        run_dir=artifacts.run_dir,
        learner=learner,
        device=device,
        config_hash256=config_hash256,
        spec_hash256=spec_hash256,
        baseline_run_dir=b1_baseline_run_dir,
    )
    if seed_snapshot_run_dir is not None:
        _import_seed_snapshot_pool(
            stack=stack,
            training_paths=training_paths,
            run_dir=artifacts.run_dir,
            seed_snapshot_run_dir=seed_snapshot_run_dir,
            expected_model_state_dict=learner.model.state_dict(),
            expected_config_canonical=canonical_config_dict(stack),
            expected_spec_hash256=spec_hash256,
        )
    runtime_config = build_runtime_config(
        stack=stack,
        num_envs=num_envs,
        unroll_length=unroll_length,
        profile=profile,
        seed=seed,
        pass_action_id=pass_action_id,
        runtime_mode=runtime_mode,
    )
    runtime = QueueRuntime(
        stack=stack,
        config=runtime_config,
        model=model,
        observation_dim=observation_dim,
        action_dim=action_dim,
        observation_spec=cast(dict[str, Any] | None, contract.spec_bundle.get("observation")),
        spec_bundle=cast(dict[str, Any], contract.spec_bundle),
        run_dir=artifacts.run_dir,
        performance_log_path=training_paths.performance_log_path,
        learner_device=device,
    )
    actor_torch_threads = _central_runtime_actor_torch_threads(stack, runtime)
    learner_torch_threads = None if stack.config.system is None else int(stack.config.system.learner_torch_threads)
    latest_metrics: dict[str, float] = {}
    last_checkpoint_guard_rollback_update: int | None = None
    last_dev_eval_summary: Mapping[str, Any] | None = None
    last_dev_eval_update_count: int | None = None
    start_time = time.time()
    profiler, profiler_context, profiler_trace_dir = _build_training_profiler(
        enabled=bool(torch_profiler),
        run_dir=artifacts.run_dir,
        device=device,
    )
    with profiler_context:
        if int(learner.update_count) == 0:
            latest_metrics = _run_structured_warmstart(
                learner=learner,
                runtime=runtime,
                algorithm=algorithm,
                training_config=training_config,
                rewards_config=rewards_config,
                training_paths=training_paths,
                tensorboard_logger=tensorboard_logger,
                start_time=start_time,
                profile_timers=bool(profile_timers),
                actor_torch_threads=actor_torch_threads,
                learner_torch_threads=learner_torch_threads,
            )
        if int(learner.update_count) >= max_updates:
            raise RuntimeError(
                f"Resume checkpoint is already at update {learner.update_count}, which is >= --max-updates {max_updates}"
            )
        try:
            for _update_index in range(int(learner.update_count), max_updates):
                learner.set_entropy_coef(
                    _entropy_coef_for_next_update(training_config, update_count=int(learner.update_count) + 1)
                )
                with (
                    _profile_block(profile_timers, "collect_update_batch"),
                    _torch_num_threads_scope(actor_torch_threads),
                ):
                    runtime_batch = _collect_training_batch(
                        runtime=runtime,
                        algorithm=algorithm,
                        training_config=training_config,
                        rewards_config=rewards_config,
                    )
                with _profile_block(profile_timers, "learner_update"), _torch_num_threads_scope(learner_torch_threads):
                    latest_metrics = learner.update(runtime_batch.learner_batch)
                with _profile_block(profile_timers, "runtime_snapshot_publish"):
                    latest_metrics.update(
                        runtime.maybe_publish_snapshot(
                            learner_model=model,
                            learner_update_count=int(learner.update_count),
                        )
                    )
                _write_scalars_record(
                    scalars_path=training_paths.scalars_path,
                    learner=learner,
                    metrics=latest_metrics,
                    start_time=start_time,
                )
                if tensorboard_logger is not None:
                    tensorboard_logger.log_training_step(
                        update_count=int(learner.update_count),
                        policy_version=int(learner.get_policy_version()),
                        wall_clock_seconds=time.time() - start_time,
                        metrics=latest_metrics,
                    )
                if learner.update_count % checkpoint_interval_updates == 0:
                    ckpt_path = training_paths.checkpoints_dir / f"checkpoint_{learner.update_count}.pt"
                    _write_checkpoint(
                        checkpoint_path=ckpt_path,
                        learner=learner,
                        stack=stack,
                        device=device,
                        spec_hash256=spec_hash256,
                        algorithm=algorithm,
                    )
                    tracker_payload = _publish_checkpoint_aliases(
                        stack=stack,
                        training_paths=training_paths,
                        artifacts=artifacts,
                        checkpoint_path=ckpt_path,
                        learner=learner,
                        latest_metrics=latest_metrics,
                    )
                    _maybe_log_structured_mainmove_guard(
                        training_paths=training_paths,
                        learner=learner,
                        latest_metrics=latest_metrics,
                        dev_eval_summary=last_dev_eval_summary,
                    )
                    if tensorboard_logger is not None:
                        tensorboard_logger.log_checkpoint_tracker(tracker_payload, step=int(learner.update_count))

                    if learner.model is None:
                        raise RuntimeError("Cannot persist a snapshot registry entry without a learner model")
                    candidate_policy_id = _persist_snapshot_registry_entry(
                        stack=stack,
                        training_paths=training_paths,
                        run_dir=artifacts.run_dir,
                        checkpoint_path=ckpt_path,
                        model_state_dict=learner.model.state_dict(),
                        config_hash256=config_hash256,
                        device=device,
                        update=int(learner.update_count),
                        policy_version=int(learner.get_policy_version()),
                    )
                    if _is_noleague_baseline_role(experiment_role):
                        _ensure_noleague_baseline_anchor(
                            stack=stack,
                            training_paths=training_paths,
                            run_dir=artifacts.run_dir,
                            learner=learner,
                            device=device,
                            config_hash256=config_hash256,
                            permit_current_run_alias=True,
                            source_checkpoint_path=ckpt_path,
                            update=int(learner.update_count),
                        )
                    runtime.refresh_opponent_pool()
                    promotion_passed = _run_snapshot_promotion_gate(
                        stack=stack,
                        contract=contract,
                        artifacts=artifacts,
                        training_paths=training_paths,
                        learner=learner,
                        candidate_policy_id=candidate_policy_id,
                        update_count=int(learner.update_count),
                        league_reference_update=(
                            None
                            if "league_effective_update" not in latest_metrics
                            else int(latest_metrics["league_effective_update"])
                        ),
                        policy_version=int(learner.get_policy_version()),
                        run_id256=run_id256,
                        config_hash256=config_hash256,
                        spec_hash256=spec_hash256,
                    )
                    if promotion_passed:
                        runtime.refresh_opponent_pool()

                if _should_run_periodic_dev_eval(stack, update_count=int(learner.update_count)):
                    summary_payload = _run_periodic_dev_eval(
                        stack=stack,
                        contract=contract,
                        artifacts=artifacts,
                        training_paths=training_paths,
                        learner=learner,
                        device=device,
                        run_id256=run_id256,
                        config_hash256=config_hash256,
                        spec_hash256=spec_hash256,
                    )
                    anchor_keys = sorted(cast(dict[str, Any], summary_payload["anchor_scores"]).keys())
                    opponent_fragment = f" opponent={_slug_policy_id(anchor_keys[0])}" if anchor_keys else ""
                    print(
                        "Periodic dev eval: "
                        f"update={learner.update_count}{opponent_fragment} "
                        f"aggregate={summary_payload['aggregate_score']:.4f} "
                        f"anchors={','.join(anchor_keys)}"
                    )
                    effective_summary = summary_payload
                    tracker_before_dev_eval = _load_checkpoint_tracker(training_paths)
                    existing_best_record = tracker_before_dev_eval.get("best")
                    if not isinstance(existing_best_record, Mapping):
                        existing_best_record = None
                    confirmatory_request = _confirmatory_dev_eval_request(
                        stack=stack,
                        existing_best_record=cast(Mapping[str, Any] | None, existing_best_record),
                        dev_eval_summary=summary_payload,
                    )
                    if confirmatory_request is not None:
                        seed_file, _validated_sources, base_paired_seeds, seed_file_sha256 = (
                            _periodic_dev_eval_schedule(stack)
                        )
                        confirmatory_pairs = _expand_periodic_dev_eval_paired_seeds(
                            base_paired_seeds,
                            requested_pairs=int(confirmatory_request["target_pairs"]),
                            seed_file_sha256=seed_file_sha256,
                            update_count=int(learner.update_count),
                            policy_version=int(learner.get_policy_version()),
                            scope="periodic_dev_eval_confirmatory",
                        )
                        effective_summary = _run_periodic_dev_eval(
                            stack=stack,
                            contract=contract,
                            artifacts=artifacts,
                            training_paths=training_paths,
                            learner=learner,
                            device=device,
                            run_id256=run_id256,
                            config_hash256=config_hash256,
                            spec_hash256=spec_hash256,
                            artifact_dir_name="dev_eval_confirmatory",
                            artifact_scope="periodic_dev_eval_confirmatory",
                            paired_seeds_override=confirmatory_pairs,
                            persist_summary=False,
                            update_stall_monitor=False,
                        )
                        print(
                            "Confirmatory dev eval: "
                            f"update={learner.update_count} paired_seeds={len(confirmatory_pairs)} "
                            f"aggregate={effective_summary['aggregate_score']:.4f} "
                            f"reasons={','.join(cast(list[str], confirmatory_request['reasons']))} "
                            f"seed_file={seed_file.name}"
                        )
                    last_dev_eval_summary = effective_summary
                    last_dev_eval_update_count = int(learner.update_count)
                    ckpt_path = _ensure_current_checkpoint(
                        training_paths=training_paths,
                        learner=learner,
                        stack=stack,
                        device=device,
                        spec_hash256=spec_hash256,
                        algorithm=algorithm,
                    )
                    tracker_payload = _publish_checkpoint_aliases(
                        stack=stack,
                        training_paths=training_paths,
                        artifacts=artifacts,
                        checkpoint_path=ckpt_path,
                        learner=learner,
                        latest_metrics=latest_metrics,
                        dev_eval_summary=effective_summary,
                    )
                    _maybe_log_structured_mainmove_guard(
                        training_paths=training_paths,
                        learner=learner,
                        latest_metrics=latest_metrics,
                        dev_eval_summary=effective_summary,
                    )
                    guard_event = _maybe_rollback_to_best_checkpoint(
                        stack=stack,
                        training_paths=training_paths,
                        artifacts=artifacts,
                        runtime=runtime,
                        learner=learner,
                        model=model,
                        device=device,
                        spec_hash256=spec_hash256,
                        algorithm=algorithm,
                        latest_metrics=latest_metrics,
                        dev_eval_summary=effective_summary,
                        last_rollback_update=last_checkpoint_guard_rollback_update,
                    )
                    if guard_event is not None:
                        last_checkpoint_guard_rollback_update = int(learner.update_count)
                        print(
                            "Checkpoint guard rollback: "
                            f"update={guard_event['update_count']} "
                            f"best_update={guard_event['best_update_count']} "
                            f"current_score={float(guard_event['current_score']):.4f} "
                            f"best_score={float(guard_event['best_score']):.4f} "
                            f"reasons={','.join(cast(list[str], guard_event['reasons']))}"
                        )
                    if tensorboard_logger is not None:
                        tensorboard_logger.log_periodic_dev_eval(effective_summary, step=int(learner.update_count))
                        tensorboard_logger.log_checkpoint_tracker(tracker_payload, step=int(learner.update_count))
        finally:
            runtime.close()

    if profiler is not None and profiler_trace_dir is not None:
        trace_path = profiler_trace_dir / "trace.json"
        profiler.export_chrome_trace(str(trace_path))
        print(f"Wrote torch profiler trace: {trace_path}")

    if _is_noleague_baseline_role(experiment_role):
        _ensure_noleague_baseline_anchor(
            stack=stack,
            training_paths=training_paths,
            run_dir=artifacts.run_dir,
            learner=learner,
            device=device,
            config_hash256=config_hash256,
            permit_current_run_alias=True,
            update=int(learner.update_count),
        )

    if not latest_metrics:
        raise RuntimeError("The canonical single-node run finished without producing learner metrics")
    final_checkpoint_path = _ensure_current_checkpoint(
        training_paths=training_paths,
        learner=learner,
        stack=stack,
        device=device,
        spec_hash256=spec_hash256,
        algorithm=algorithm,
    )
    final_dev_eval_summary = last_dev_eval_summary if last_dev_eval_update_count == int(learner.update_count) else None
    tracker_payload = _publish_checkpoint_aliases(
        stack=stack,
        training_paths=training_paths,
        artifacts=artifacts,
        checkpoint_path=final_checkpoint_path,
        learner=learner,
        latest_metrics=latest_metrics,
        dev_eval_summary=final_dev_eval_summary,
    )
    finalize_guard_event = _maybe_finalize_from_best_checkpoint(
        stack=stack,
        training_paths=training_paths,
        artifacts=artifacts,
        runtime=runtime,
        learner=learner,
        device=device,
        spec_hash256=spec_hash256,
        algorithm=algorithm,
        latest_metrics=latest_metrics,
        dev_eval_summary=final_dev_eval_summary,
    )
    if finalize_guard_event is not None:
        print(
            "Checkpoint guard final selection: "
            f"update={finalize_guard_event['update_count']} "
            f"best_update={finalize_guard_event['best_update_count']} "
            f"current_score={float(finalize_guard_event['current_score']):.4f} "
            f"best_score={float(finalize_guard_event['best_score']):.4f}"
        )
        tracker_payload = _load_checkpoint_tracker(training_paths)
    if tensorboard_logger is not None:
        tensorboard_logger.log_checkpoint_tracker(tracker_payload, step=int(learner.update_count))
    return latest_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical single-node thesis training entrypoint")
    parser.add_argument("--stack-config", type=Path, required=True)
    parser.add_argument("--spec-hash", type=str, default="", help="Expected spec hash or spec bundle SHA-256")
    parser.add_argument(
        "--public-demo",
        action="store_true",
        help="Stage the built-in public-safe toy catalog/policy bundle instead of probing weiss_sim.",
    )
    parser.add_argument(
        "--config-hash",
        type=str,
        default="",
        help="Expected config_hash256 for contract validation",
    )
    parser.add_argument("--run-label", type=str, default="", help="Optional run directory label override")
    parser.add_argument("--run-id", dest="run_id_alias", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument(
        "--override",
        "--config-override",
        dest="config_override",
        action="append",
        default=None,
        help="Deterministic config override in KEY=JSON_VALUE form, e.g. training.optimizer.learning_rate=0.0001",
    )
    parser.add_argument("--num-envs", type=int, default=2, help="Env count for the single-node training run")
    parser.add_argument("--unroll-length", type=int, default=4, help="Tiny rollout length for the smoke run")
    parser.add_argument("--max-updates", type=int, default=1, help="Number of learner updates to run")
    parser.add_argument(
        "--runtime-mode",
        type=str,
        default="train_ordered",
        choices=("train_ordered", "train_async_fast"),
        help="Queue runtime mode: deterministic ordered collection or throughput-oriented async-fast collection",
    )
    parser.add_argument(
        "--profile-timers",
        action="store_true",
        help="Enable cheap runtime/learner timers and record_function ranges without emitting a torch profiler trace",
    )
    parser.add_argument(
        "--torch-profiler",
        action="store_true",
        help="Emit a torch profiler trace under profiling/torch_profiler/trace.json",
    )
    parser.add_argument("--profile", type=str, default="", help="Optional simulator profile override")
    parser.add_argument("--device", type=str, default="", help="Optional learner device override")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override")
    parser.add_argument(
        "--checkpoint-interval-updates",
        type=int,
        default=None,
        help="Optional checkpoint cadence override for the single-node training run",
    )
    parser.add_argument(
        "--snapshot-registry-json",
        type=Path,
        default=None,
        help="Optional snapshot registry JSON used to resolve the deterministic final policy set in the manifest",
    )
    parser.add_argument(
        "--dev-eval-summaries-json",
        type=Path,
        default=None,
        help="Optional dev-eval summaries JSON used to resolve the deterministic final policy set in the manifest",
    )
    parser.add_argument(
        "--b1-baseline-run-dir",
        type=Path,
        default=None,
        help="Completed baseline_noleague run directory used to import the canonical B1 baseline anchor",
    )
    parser.add_argument(
        "--seed-snapshot-run-dir",
        type=Path,
        default=None,
        help="Optional completed run directory whose snapshot registry should be imported into the current training league before update 1",
    )
    parser.add_argument(
        "--resume-run-dir",
        type=Path,
        default=None,
        help="Resume training in-place inside an existing run directory",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default="",
        help="Checkpoint path or alias (`latest`/`best`) to restore before continuing training",
    )
    args = parser.parse_args()
    run_label = _resolve_run_label(parser, args.run_label, args.run_id_alias)

    num_envs = _require_positive_int("--num-envs", args.num_envs)
    unroll_length = _require_positive_int("--unroll-length", args.unroll_length)
    max_updates = _require_positive_int("--max-updates", args.max_updates)
    stack = load_stack_config(args.stack_config)
    stack = apply_stack_overrides(stack, parse_override_tokens(args.config_override))
    stack = _apply_training_flag_overrides(
        stack,
        enable_profile_timers=bool(args.profile_timers),
        enable_torch_profiler=bool(args.torch_profiler),
    )
    training_config = stack.config.training
    manifest_only_reason = _manifest_scaffold_only_reason(stack)
    if training_config is None and manifest_only_reason is None:
        parser.error("stack config is missing training")

    public_demo_enabled = bool(args.public_demo)
    resume_run_dir = None if args.resume_run_dir is None else args.resume_run_dir.resolve()
    resume_checkpoint_path = _resolve_resume_checkpoint_path(
        resume_from=str(args.resume_from),
        resume_run_dir=resume_run_dir,
    )
    if public_demo_enabled and (resume_run_dir is not None or resume_checkpoint_path is not None):
        parser.error("Public demo mode does not support checkpoint resume")
    if public_demo_enabled:
        public_demo_bundle = public_demo_spec_bundle()
        assert_spec_bundle_contract(args.spec_hash, public_demo_bundle)
        spec_bundle = public_demo_bundle
        spec_hash256 = public_demo_spec_hash256()
        simulator_info = public_demo_simulator_info()
    else:
        simulator_contract = load_verified_simulator_contract(stack.root, expected_spec_hash=args.spec_hash)
        spec_bundle = simulator_contract.spec_bundle
        spec_hash256 = simulator_contract.spec_hash256
        simulator_info = simulator_contract.simulator
    config_hash256 = compute_config_hash256(stack)
    _require_matching_hash(
        flag_name="--config-hash",
        expected=_expected_sha256(args.config_hash, flag_name="--config-hash"),
        actual=config_hash256,
    )

    git_commit = _git_commit()
    start_nonce = _start_nonce()
    manifest_dict: dict[str, Any] | None = None
    if resume_run_dir is None:
        run_id256 = compute_run_id256(spec_hash256, config_hash256, git_commit or None, start_nonce)
        run_id64 = f"{compute_run_id64(spec_hash256, config_hash256, git_commit or None, start_nonce):016x}"
        run_dir_name = run_label or default_run_dir_name(run_id64)
    else:
        artifacts = _run_artifacts_from_existing_run_dir(resume_run_dir)
        manifest_dict = _load_json_object(artifacts.manifest_path, label="resume manifest")
        run_id256 = str(manifest_dict.get("run_id256", "")).strip().lower()
        run_id64 = str(manifest_dict.get("run_id64", "")).strip().lower()
        run_dir_name = artifacts.run_dir_name
        existing_spec_hash = str(manifest_dict.get("spec_hash256", "")).strip().lower()
        existing_config_hash = str(manifest_dict.get("config_hash256", "")).strip().lower()
        if existing_spec_hash != spec_hash256:
            raise RuntimeError(
                f"resume run spec hash mismatch: expected {spec_hash256}, found {existing_spec_hash} in {artifacts.manifest_path}"
            )
        if existing_config_hash != config_hash256:
            raise RuntimeError(
                f"resume run config hash mismatch: expected {config_hash256}, found {existing_config_hash} in {artifacts.manifest_path}"
            )

    print_startup_banner(
        spec_hash256,
        config_hash256,
        run_id64=run_id64,
        run_id256=run_id256,
        run_label=run_label or ("" if resume_run_dir is None else run_dir_name),
        run_dir_name=run_dir_name,
        spec_mismatch_policy=_spec_mismatch_policy(stack),
    )
    spec_bundle_message = (
        "Loaded synthetic public-demo spec bundle: " if public_demo_enabled else "Verified runtime spec bundle: "
    )
    print(spec_bundle_message + f"compat={simulator_info.get('compatibility_hash', '')} sha256={spec_hash256}")
    print(f"Loaded stack config with {len(stack.components)} components")

    device = _resolve_device(stack, args.device)
    profile = _resolve_runtime_profile(stack, args.profile)
    seed = _resolve_seed(stack, args.seed)
    actor_device_layout = _manifest_actor_device_layout(
        stack=stack,
        num_envs=num_envs,
        unroll_length=unroll_length,
        profile=profile,
        seed=seed,
        pass_action_id=int(spec_bundle["action"]["pass_action_id"]),
        runtime_mode=cast(QueueRuntimeMode, args.runtime_mode),
        learner_device=device,
    )
    policy_set_selection, policy_set_selection_details = _resolve_policy_set_selection(
        stack,
        snapshot_registry_path=args.snapshot_registry_json,
        dev_eval_summaries_path=args.dev_eval_summaries_json,
    )
    manifest = RunManifest(
        run_id256=run_id256,
        run_id64=run_id64,
        start_nonce=start_nonce,
        git_commit=git_commit,
        git_dirty=_git_dirty(),
        spec_hash256=spec_hash256,
        config_hash256=config_hash256,
        simulator=simulator_info,
        spec_bundle=spec_bundle,
        config_canonical=canonical_config_dict(stack),
        seed_files=build_seed_file_manifest(stack.seed_sets, root=stack.root),
        hardware=_hardware_summary(
            device,
            actor_device=("cpu" if stack.config.system is None else stack.config.system.actor_device),
            actor_device_layout=actor_device_layout,
        ),
        evaluation_pinning=_evaluation_pinning(stack),
        policy_set_selection=policy_set_selection,
        policy_set_selection_details=policy_set_selection_details,
    )
    if resume_run_dir is None:
        artifacts = write_run_artifacts(
            stack.root / "runs",
            manifest,
            run_label=run_label or None,
        )
    else:
        artifacts = _run_artifacts_from_existing_run_dir(resume_run_dir)
    run_summary_payload = _load_json_object(artifacts.run_summary_path, label="run summary")
    run_summary_payload["runtime_mode"] = "public_demo" if public_demo_enabled else str(args.runtime_mode)
    run_summary_payload["policy_set_selection_mode"] = policy_set_selection_details.get("mode", "unresolved")
    if training_config is not None:
        run_summary_payload["training_controls"] = {
            "profile_timers": bool(training_config.profile_timers),
            "torch_profiler": bool(training_config.torch_profiler),
            "structured_metrics_mode": str(training_config.structured_metrics_mode),
            "teacher_aux_mode": str(training_config.teacher_aux_mode),
            "fixed_opponent_backend": str(training_config.fixed_opponent_backend),
        }
    if args.b1_baseline_run_dir is not None:
        run_summary_payload["b1_baseline_run_dir"] = args.b1_baseline_run_dir.resolve().as_posix()
    if args.seed_snapshot_run_dir is not None:
        run_summary_payload["seed_snapshot_run_dir"] = args.seed_snapshot_run_dir.resolve().as_posix()
    if resume_checkpoint_path is not None:
        run_summary_payload["resume"] = {
            "enabled": True,
            "resume_run_dir": None if resume_run_dir is None else resume_run_dir.as_posix(),
            "resume_checkpoint_path": resume_checkpoint_path.as_posix(),
        }
    _write_json(artifacts.run_summary_path, run_summary_payload)

    determinism_payload = _load_json_object(artifacts.determinism_report_path, label="determinism report")
    determinism_payload["runtime_mode"] = "public_demo" if public_demo_enabled else str(args.runtime_mode)
    determinism_payload["policy_selection_mode"] = policy_set_selection_details.get("mode", "unresolved")
    if training_config is not None:
        determinism_payload["training_controls"] = {
            "profile_timers": bool(training_config.profile_timers),
            "torch_profiler": bool(training_config.torch_profiler),
            "structured_metrics_mode": str(training_config.structured_metrics_mode),
            "teacher_aux_mode": str(training_config.teacher_aux_mode),
            "fixed_opponent_backend": str(training_config.fixed_opponent_backend),
        }
    if args.b1_baseline_run_dir is not None:
        determinism_payload["b1_baseline_run_dir"] = args.b1_baseline_run_dir.resolve().as_posix()
    if args.seed_snapshot_run_dir is not None:
        determinism_payload["seed_snapshot_run_dir"] = args.seed_snapshot_run_dir.resolve().as_posix()
    if resume_checkpoint_path is not None:
        determinism_payload["resume_checkpoint_path"] = resume_checkpoint_path.as_posix()
    _write_json(artifacts.determinism_report_path, determinism_payload)

    environment_payload = _load_json_object(artifacts.environment_path, label="environment manifest")
    environment_payload["cwd"] = stack.root.as_posix()
    environment_payload["argv"] = sys.argv
    environment_payload["hardware"] = manifest.hardware
    if resume_checkpoint_path is not None:
        environment_payload["resume_checkpoint_path"] = resume_checkpoint_path.as_posix()
    _write_json(artifacts.environment_path, environment_payload)
    tensorboard_logger = TensorBoardLogger(artifacts.layout.tensorboard_dir)
    if not tensorboard_logger.enabled:
        unavailable_reason = tensorboard_unavailable_reason()
        print(
            "TensorBoard logging is disabled: "
            + ("SummaryWriter unavailable" if unavailable_reason is None else unavailable_reason),
            file=sys.stderr,
        )
    else:
        tensorboard_logger.log_run_context(
            manifest=manifest.to_dict(),
            environment=environment_payload,
            run_summary=run_summary_payload,
            determinism_report=determinism_payload,
        )
    if resume_run_dir is None:
        print(f"Wrote manifest: {artifacts.manifest_path}")
    else:
        print(f"Resuming existing run directory: {artifacts.run_dir}")

    try:
        if public_demo_enabled:
            staged = stage_public_demo_run(artifacts.run_dir)
            print(
                "Staged public-demo toy catalog and policy bundle: "
                f"mode={PUBLIC_DEMO_MODE} policy_count={len(staged.policy_ids)} "
                f"catalog={staged.catalog_path}"
            )
            print(
                "Public demo mode is intentionally synthetic and demo-only. "
                "It does not execute simulator training or claim thesis-grade results."
            )
            return

        if manifest_only_reason is not None:
            _print_manifest_only_message(manifest_only_reason)
            return

        runtime_prerequisite_failure = _runtime_training_prerequisite_failure(stack)
        if runtime_prerequisite_failure is not None:
            _raise_runtime_prerequisite_failure(runtime_prerequisite_failure)

        assert training_config is not None
        checkpoint_interval_updates = _require_positive_int(
            "--checkpoint-interval-updates",
            args.checkpoint_interval_updates
            if args.checkpoint_interval_updates is not None
            else int(training_config.checkpoint_interval_updates),
        )

        profile_timers = bool(training_config.profile_timers)
        torch_profiler = bool(training_config.torch_profiler)
        if profile_timers or torch_profiler:
            print(
                "Structured profiling enabled: "
                f"profile_timers={profile_timers} "
                f"torch_profiler={torch_profiler} "
                f"structured_metrics_mode={training_config.structured_metrics_mode} "
                f"teacher_aux_mode={training_config.teacher_aux_mode} "
                f"fixed_opponent_backend={training_config.fixed_opponent_backend}"
            )

        metrics = _run_minimal_training(
            stack=stack,
            contract=simulator_contract,
            artifacts=artifacts,
            num_envs=num_envs,
            unroll_length=unroll_length,
            max_updates=max_updates,
            profile=profile,
            device=device,
            seed=seed,
            checkpoint_interval_updates=checkpoint_interval_updates,
            run_id256=run_id256,
            config_hash256=config_hash256,
            spec_hash256=spec_hash256,
            runtime_mode=cast(QueueRuntimeMode, args.runtime_mode),
            b1_baseline_run_dir=None if args.b1_baseline_run_dir is None else args.b1_baseline_run_dir.resolve(),
            seed_snapshot_run_dir=None if args.seed_snapshot_run_dir is None else args.seed_snapshot_run_dir.resolve(),
            profile_timers=profile_timers,
            torch_profiler=torch_profiler,
            resume_checkpoint_path=resume_checkpoint_path,
            tensorboard_logger=tensorboard_logger,
        )
        print(
            "Completed canonical single-node training run: "
            f"loss={metrics.get('loss', 0.0):.6f} "
            f"policy_loss={metrics.get('policy_loss', 0.0):.6f} "
            f"value_loss={metrics.get('value_loss', 0.0):.6f} "
            f"entropy={metrics.get('entropy', 0.0):.6f}"
        )
    finally:
        tensorboard_logger.close()


if __name__ == "__main__":
    main()
