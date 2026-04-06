from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
import hashlib

import numpy as np
import torch

from weiss_rl.cli_banner import print_startup_banner
from weiss_rl.config import StackConfig, canonical_config_dict, compute_config_hash256, load_stack_config
from weiss_rl.envs.decision_env import DecisionBoundaryBatch, DecisionBoundaryEnv
from weiss_rl.envs.pool_factory import make_env_pool_from_config
from weiss_rl.eval import (
    Pcg32XshRrV1,
    PayoffFoldScheme,
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
from weiss_rl.learners.impala_learner import ImpalaLearner
from weiss_rl.learners.vtrace import VTraceTargets, compute_vtrace_targets
from weiss_rl.league import run_promotion_gate
from weiss_rl.manifest import RunManifest, build_seed_file_manifest, default_run_dir_name, write_run_artifacts
from weiss_rl.masking import assert_strictly_increasing_legal_ids, masked_logp_from_mask, sample_actions_from_mask
from weiss_rl.model import PolicyValueModel
from weiss_rl.repro import (
    canonical_json_bytes,
    compute_run_id64,
    compute_run_id256,
    hash_seed_file,
    parse_seed_file,
    stable_hash64,
)
from weiss_rl.simulator_contract import SimulatorContract, load_simulator_contract
from weiss_rl.spec import assert_spec_bundle_contract
from weiss_rl.league.registry import (
    REGISTRY_FILENAME,
    SNAPSHOT_METADATA_FILENAME,
    SnapshotRegistry,
    snapshot_weights_relpath,
)

_SHA256_HEX_LENGTH = 64
_U64_MASK = (1 << 64) - 1
_PROMOTION_GATE_RANDOMLEGAL_NAME = "B0 RandomLegal"
_PROMOTION_GATE_RANDOMLEGAL_POLICY_ID = "b0_randomlegal"


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
    scalars_path: Path


class _PeriodicDevEvalRunner:
    def __init__(
        self,
        *,
        stack: StackConfig,
        model: PolicyValueModel,
        observation_dim: int,
        action_dim: int,
        pass_action_id: int,
        artifact_dir: Path,
        focal_policy_id: str,
        require_sorted_legal_ids: bool,
    ) -> None:
        self.stack = stack
        self.model = model
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
        seat_hidden = self.model.initial_seat_hidden(1, device=self._device)
        seat_rngs = {
            seat: Pcg32XshRrV1(_periodic_dev_eval_rng_seed(scheduled_game=scheduled_game, seat=seat))
            for seat in (0, 1)
        }
        last_acting_seat: int | None = None

        try:
            batch = env.reset()
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
                action, seat_hidden = self._select_action(
                    batch=batch,
                    scheduled_game=scheduled_game,
                    current_seat=current_seat,
                    seat_hidden=seat_hidden,
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
        seat_hidden: torch.Tensor,
        rng: Pcg32XshRrV1,
    ) -> tuple[int, torch.Tensor]:
        legal_ids = _legal_ids_for_env_row(
            batch=batch,
            env_index=0,
            require_sorted=self.require_sorted_legal_ids,
        )
        current_policy_id = scheduled_game.seat0_policy_id if current_seat == 0 else scheduled_game.seat1_policy_id
        if current_policy_id != self.focal_policy_id:
            action, _ = sample_action_pinned(
                self._baseline_logits,
                legal_ids,
                rng=rng,
                pass_action_id=self.pass_action_id,
            )
            return action, seat_hidden

        with torch.inference_mode():
            logits_tensor, _value_tensor, next_seat_hidden = self.model.forward_seat_aware(
                torch.as_tensor(np.asarray(batch.obs, dtype=np.float32), device=self._device),
                torch.as_tensor([current_seat], device=self._device, dtype=torch.long),
                seat_hidden,
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


def _persist_snapshot_registry_entry(
    *,
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
    reg.add_snapshot(
        policy_id=policy_id,
        update=int(update),
        weights_sha256=weights_sha256,
        path=weights_path.relative_to(run_dir).as_posix(),
    )
    reg.save(registry_path)
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


def _hardware_summary() -> dict[str, str | int]:
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count() or 0,
    }


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


def _policy_set_selection(stack: StackConfig) -> list[str]:
    if stack.config.evaluation is None:
        return []
    selection = stack.config.evaluation.final_policy_set_selection
    return [*selection.fixed_anchor_set_v1.required, *selection.fixed_anchor_set_v1.optional_if_available]


def _spec_mismatch_policy(stack: StackConfig) -> str:
    reproducibility = stack.config.reproducibility
    if reproducibility is None:
        return "hard_fail"
    return "hard_fail" if reproducibility.spec_bundle.fail_on_spec_mismatch else "disabled"


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
        return "balanced"
    return system_config.profile.local_iteration


def _resolve_device(stack: StackConfig, device_override: str) -> torch.device:
    requested = device_override.strip()
    if not requested:
        system_config = stack.config.system
        requested = "cpu" if system_config is None else system_config.learner_device
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("Requested CUDA device is unavailable; falling back to cpu for the minimal train smoke.", file=sys.stderr)
        requested = "cpu"
    return torch.device(requested)


def _resolve_seed(stack: StackConfig, seed_override: int | None) -> int:
    if seed_override is not None:
        return int(seed_override)
    reproducibility = stack.config.reproducibility
    if reproducibility is None:
        return 7
    return int(reproducibility.seed_derivation.base_seed64)


def _minimal_training_prerequisite_failure(stack: StackConfig) -> str | None:
    missing_blocks: list[str] = []
    if stack.config.environment is None:
        missing_blocks.append("environment")
    if stack.config.training_family_a is None:
        missing_blocks.append("training_family_a")
    if stack.config.model is None:
        missing_blocks.append("model")
    if missing_blocks:
        return f"missing config blocks: {', '.join(missing_blocks)}"

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


def _training_paths(run_dir: Path) -> TrainingPaths:
    training_dir = run_dir / "training"
    checkpoints_dir = training_dir / "checkpoints"
    logs_dir = training_dir / "logs"
    snapshots_dir = training_dir / "snapshots"
    for path in (training_dir, checkpoints_dir, logs_dir, snapshots_dir):
        path.mkdir(parents=True, exist_ok=True)
    return TrainingPaths(
        training_dir=training_dir,
        checkpoints_dir=checkpoints_dir,
        logs_dir=logs_dir,
        snapshots_dir=snapshots_dir,
        scalars_path=logs_dir / "scalars.jsonl",
    )


def _configure_torch_threads(stack: StackConfig) -> None:
    system_config = stack.config.system
    if system_config is None:
        return
    torch.set_num_threads(int(system_config.learner_torch_threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _spec_dimensions(contract: SimulatorContract) -> tuple[int, int]:
    observation_dim = int(contract.spec_bundle["observation"]["obs_len"])
    action_dim = int(contract.spec_bundle["action"]["action_space_size"])
    return observation_dim, action_dim


def _env_pool_config(stack: StackConfig, *, seed: int) -> dict[str, int | str]:
    environment_config = stack.config.environment
    if environment_config is None:
        raise RuntimeError("The locked stack is missing the environment config block")
    return {
        "max_decisions": int(environment_config.max_decisions),
        "max_ticks": int(environment_config.max_ticks),
        "observation_visibility": environment_config.observation_visibility,
        "seed": int(seed),
    }



def _build_env(
    stack: StackConfig,
    *,
    profile: str,
    num_envs: int,
    seed: int,
) -> DecisionBoundaryEnv:
    pool, layout_name = make_env_pool_from_config(
        _env_pool_config(stack, seed=seed),
        profile=profile,  # type: ignore[arg-type]
        num_envs=num_envs,
    )
    if layout_name != "mask":
        raise RuntimeError(
            "The minimal M3-08 training path expects mask legality because ImpalaLearner consumes legal_mask. "
            f"Profile {profile!r} resolved to layout {layout_name!r}."
        )
    return DecisionBoundaryEnv(pool, legality="mask", engine_status_policy="hard_fail")



def _build_ids_eval_env(
    stack: StackConfig,
    *,
    seed: int,
    pass_action_id: int,
) -> DecisionBoundaryEnv:
    pool, layout_name = make_env_pool_from_config(
        _env_pool_config(stack, seed=seed),
        profile="fast",
        num_envs=1,
    )
    if layout_name != "i16_legal_ids":
        raise RuntimeError(
            "Periodic dev eval requires ids-based legality for the pinned eval protocol. "
            f"Profile 'fast' resolved to layout {layout_name!r}."
        )
    return DecisionBoundaryEnv(
        pool,
        legality="ids_offsets",
        pass_action_id=pass_action_id,
        engine_status_policy="hard_fail",
    )


def _collect_rollout(
    env: DecisionBoundaryEnv,
    model: PolicyValueModel,
    *,
    unroll_length: int,
    num_envs: int,
    observation_dim: int,
    action_dim: int,
    device: torch.device,
    rng: np.random.Generator,
    pass_action_id: int,
) -> tuple[MinimalRollout, torch.Tensor, torch.Tensor]:
    batch = env.reset()
    seat_hidden = model.initial_seat_hidden(num_envs, device=device)
    initial_hidden_state = seat_hidden.detach().clone()

    obs = np.zeros((unroll_length, num_envs, observation_dim), dtype=np.float32)
    legal_mask = np.zeros((unroll_length, num_envs, action_dim), dtype=bool)
    actions = np.zeros((unroll_length, num_envs), dtype=np.int64)
    rewards = np.zeros((unroll_length, num_envs), dtype=np.float32)
    terminated = np.zeros((unroll_length, num_envs), dtype=bool)
    truncated = np.zeros((unroll_length, num_envs), dtype=bool)
    to_play_seat = np.zeros((unroll_length, num_envs), dtype=np.int64)
    behavior_logp = np.zeros((unroll_length, num_envs), dtype=np.float32)
    logits = np.zeros((unroll_length, num_envs, action_dim), dtype=np.float32)
    values = np.zeros((unroll_length, num_envs), dtype=np.float32)

    final_batch = None

    model.eval()
    with torch.inference_mode():
        for step_index in range(unroll_length):
            obs_step = np.asarray(batch.obs, dtype=np.float32)
            actor_step = np.asarray(batch.actor, dtype=np.int64)
            mask_step = np.asarray(batch.mask, dtype=bool)

            if obs_step.shape != (num_envs, observation_dim):
                raise RuntimeError(f"Expected obs shape {(num_envs, observation_dim)}, got {tuple(obs_step.shape)}")
            if mask_step.shape != (num_envs, action_dim):
                raise RuntimeError(f"Expected legal mask shape {(num_envs, action_dim)}, got {tuple(mask_step.shape)}")
            if np.any((actor_step != 0) & (actor_step != 1)):
                raise RuntimeError(
                    "The minimal collector only supports live decision-boundary rows during the rollout window. "
                    f"Received actor rows {actor_step.tolist()} at step {step_index}."
                )

            logits_tensor, value_tensor, seat_hidden = model.forward_seat_aware(
                torch.as_tensor(obs_step, device=device),
                torch.as_tensor(actor_step, device=device, dtype=torch.long),
                seat_hidden,
            )
            logits_step = logits_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
            value_step = value_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
            action_step, logp_step, _entropy = sample_actions_from_mask(
                logits_step,
                mask_step,
                rng=rng,
                pass_action_id=pass_action_id,
            )

            next_batch = env.step(action_step.astype(np.uint32, copy=False))
            done = np.logical_or(next_batch.terminated, next_batch.truncated)
            if np.any(done) and step_index != unroll_length - 1:
                raise RuntimeError(
                    "A row terminated/truncated inside the tiny smoke rollout before the final bootstrap step. "
                    "Reduce --unroll-length or revisit the collector before using longer runs."
                )

            obs[step_index] = obs_step
            legal_mask[step_index] = mask_step
            actions[step_index] = action_step
            rewards[step_index] = np.asarray(next_batch.reward, dtype=np.float32)
            terminated[step_index] = np.asarray(next_batch.terminated, dtype=bool)
            truncated[step_index] = np.asarray(next_batch.truncated, dtype=bool)
            to_play_seat[step_index] = actor_step
            behavior_logp[step_index] = logp_step
            logits[step_index] = logits_step
            values[step_index] = value_step

            final_batch = next_batch
            batch = next_batch

    if final_batch is None:
        raise RuntimeError("Failed to collect the final bootstrap batch")

    return (
        MinimalRollout(
            obs=obs,
            legal_mask=legal_mask,
            actions=actions,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            to_play_seat=to_play_seat,
            behavior_logp=behavior_logp,
            logits=logits,
            values=values,
            bootstrap_obs=np.asarray(final_batch.obs, dtype=np.float32),
            bootstrap_actor=np.asarray(final_batch.actor, dtype=np.int64),
        ),
        initial_hidden_state,
        seat_hidden.detach().clone(),
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
    training_config = stack.config.training_family_a
    environment_config = stack.config.environment
    if training_config is None or environment_config is None:
        raise RuntimeError("The minimal M3-08 path requires training and environment config blocks")

    target_logp = masked_logp_from_mask(
        rollout.logits.reshape(-1, action_dim),
        rollout.legal_mask.reshape(-1, action_dim),
        rollout.actions.reshape(-1),
        pass_action_id=pass_action_id,
    ).reshape(rollout.actions.shape)

    discounts = np.logical_not(rollout.terminated).astype(np.float32) * float(training_config.gamma)
    if not bool(environment_config.truncation_bootstrap_value):
        discounts *= np.logical_not(rollout.truncated).astype(np.float32)

    values = np.concatenate([rollout.values, bootstrap_value[np.newaxis, :]], axis=0)
    vtrace_result: VTraceTargets = compute_vtrace_targets(
        rollout.rewards,
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
        "rewards": rollout.rewards,
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
) -> None:
    if learner.model is None:
        raise RuntimeError("Cannot write a checkpoint without a learner model")

    payload = {
        "format": "minimal_train_checkpoint_v1",
        "update_count": int(learner.update_count),
        "policy_version": int(learner.get_policy_version()),
        "device": str(device),
        "config_hash256": compute_config_hash256(stack),
        "model_state_dict": learner.model.state_dict(),
        "optimizer_state_dict": None if learner.optimizer is None else learner.optimizer.state_dict(),
    }
    torch.save(payload, checkpoint_path)


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
    normalized = _slug_policy_id(anchor_name)
    if not normalized:
        return ()
    return tuple(dict.fromkeys((normalized, anchor_name)))


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
        candidates = _promotion_anchor_policy_id_candidates(anchor_name)
        policy_id = next((candidate for candidate in candidates if candidate in available_policy_ids), None)
        if policy_id is None and anchor_name == _PROMOTION_GATE_RANDOMLEGAL_NAME:
            policy_id = _PROMOTION_GATE_RANDOMLEGAL_POLICY_ID
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
) -> PolicyValueModel:
    payload = torch.load(run_dir / snapshot_path, map_location="cpu", weights_only=True)
    model_state_dict = payload.get("model_state_dict")
    if not isinstance(model_state_dict, dict):
        raise RuntimeError(f"Snapshot weights payload missing model_state_dict: {snapshot_path}")

    model_config = stack.config.model
    if model_config is None:
        raise RuntimeError("The locked stack is missing the model config block")

    eval_model = PolicyValueModel(
        observation_dim=observation_dim,
        config=model_config,
        action_dim=action_dim,
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
        self._baseline_logits = np.zeros((action_dim,), dtype=np.float32)
        self._device = torch.device("cpu")

    def run_game(self, scheduled_game: ScheduledGame):
        env = _build_ids_eval_env(
            self.stack,
            seed=scheduled_game.episode_seed,
            pass_action_id=self.pass_action_id,
        )
        seat_hidden = {
            seat: self._initial_hidden(
                scheduled_game.seat0_policy_id if seat == 0 else scheduled_game.seat1_policy_id
            )
            for seat in (0, 1)
        }
        seat_rngs = {
            seat: Pcg32XshRrV1(_promotion_gate_rng_seed(scheduled_game=scheduled_game, seat=seat))
            for seat in (0, 1)
        }
        last_acting_seat: int | None = None

        try:
            batch = env.reset()
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
        raise RuntimeError(
            "Periodic dev eval requires evaluation.eval_device='cpu', "
            f"got {evaluation.eval_device!r}"
        )
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
) -> PolicyValueModel:
    model_config = stack.config.model
    if model_config is None:
        raise RuntimeError("The locked stack is missing the model config block")
    eval_model = PolicyValueModel(
        observation_dim=observation_dim,
        config=model_config,
        action_dim=action_dim,
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
    )
    return checkpoint_path


def _should_run_periodic_dev_eval(stack: StackConfig, *, update_count: int) -> bool:
    evaluation = stack.config.evaluation
    if evaluation is None:
        return False
    interval = int(evaluation.periodic_dev_eval_interval_updates)
    return interval > 0 and update_count % interval == 0


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
) -> dict[str, Any]:
    if learner.model is None:
        raise RuntimeError("Periodic dev eval requires an attached learner model")

    evaluation = _validate_periodic_dev_eval_contract(stack)
    seed_file, validated_sources, paired_seeds, seed_file_sha256 = _periodic_dev_eval_schedule(stack)
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
    )

    update_dir = artifacts.run_dir / "eval" / "dev_eval" / f"update_{update_count}"
    matchup_dir = update_dir / "b0_randomlegal"
    eval_model = _clone_cpu_eval_model(
        learner_model=cast(PolicyValueModel, learner.model),
        observation_dim=observation_dim,
        action_dim=action_dim,
        stack=stack,
    )
    runner = _PeriodicDevEvalRunner(
        stack=stack,
        model=eval_model,
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
                None
                if checkpoint_path is None
                else _json_relative_path(checkpoint_path, root=artifacts.run_dir)
            ),
        },
        "opponent_policy": {
            "policy_id": "b0_randomlegal",
            "display_name": "B0 RandomLegal",
        },
    }
    _write_json(update_dir / "seed_usage.json", seed_usage_payload)

    matchup = run_seat_swapped_matchup(
        focal_policy_id=focal_policy_id,
        opponent_policy_id="b0_randomlegal",
        paired_seeds=paired_seeds,
        runner=runner,
        episodes_path=matchup_dir / "episodes.jsonl",
        run_id256=run_id256,
        config_hash256=config_hash256,
        spec_hash256=spec_hash256,
    )

    summary_payload = build_matchup_export(
        matchup.records,
        stop_rules=evaluation.stop_rules,
        max_paired_seeds=len(paired_seeds),
        scheme=cast(PayoffFoldScheme, evaluation.final_policy_set_selection.folding),
        sample_count=1000,
        seed=_periodic_dev_eval_bootstrap_seed(update_count=update_count, policy_version=policy_version),
    )
    summary_payload["evaluation_context"] = {
        "artifact_scope": "periodic_dev_eval",
        "update_count": update_count,
        "policy_version": policy_version,
        "checkpoint_path": (
            None
            if checkpoint_path is None
            else _json_relative_path(checkpoint_path, root=artifacts.run_dir)
        ),
        "seed_usage_path": _json_relative_path(update_dir / "seed_usage.json", root=artifacts.run_dir),
    }
    write_matchup_summary_json(matchup_dir / "matchup_summary.json", summary_payload)
    write_matchup_summary_csv(matchup_dir / "matchup_summary.csv", summary_payload)
    write_matchup_diagnostics_json(
        matchup_dir / "diagnostics.json",
        build_seat_advantage_diagnostics(matchup.records),
    )
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
    policy_version: int,
    run_id256: str,
    config_hash256: str,
    spec_hash256: str,
) -> bool | None:
    league = stack.config.league
    if league is None or not league.enabled or not league.promotion_gate_enabled:
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
        )
        for policy_id in set(anchor_policy_ids.values())
        if policy_id != _PROMOTION_GATE_RANDOMLEGAL_POLICY_ID
    }
    runner = _PromotionGateRunner(
        stack=stack,
        focal_policy_id=candidate_policy_id,
        focal_model=_clone_cpu_eval_model(
            learner_model=cast(PolicyValueModel, learner.model),
            observation_dim=observation_dim,
            action_dim=action_dim,
            stack=stack,
        ),
        anchor_models=anchor_models,
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
        if candidate_policy_id not in registry.champion_snapshots:
            registry.add_champion(candidate_policy_id)
            registry.save(registry_path)
        print(
            "Promotion gate passed: "
            f"update={update_count} candidate={candidate_policy_id} "
            f"anchors={','.join(result.ordered_opponents)}"
        )
        return True

    reason_codes = ",".join(str(reason.get("code", "unknown")) for reason in result.reasons) or "unknown"
    print(
        "Promotion gate failed: "
        f"update={update_count} candidate={candidate_policy_id} reasons={reason_codes}"
    )
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
) -> dict[str, float]:
    _configure_torch_threads(stack)
    torch.manual_seed(seed)
    np.random.seed(seed & 0xFFFF_FFFF)

    observation_dim, action_dim = _spec_dimensions(contract)
    training_config = stack.config.training_family_a
    model_config = stack.config.model
    if training_config is None or model_config is None:
        raise RuntimeError("The locked stack is missing training_family_a or model config")

    training_paths = _training_paths(artifacts.run_dir)
    pass_action_id = int(contract.spec_bundle["action"]["pass_action_id"])
    model = PolicyValueModel(
        observation_dim=observation_dim,
        config=model_config,
        action_dim=action_dim,
    ).to(device)
    learner = ImpalaLearner(
        model=model,
        learning_rate=training_config.learning_rate,
        value_loss_coef=training_config.value_loss_coef,
        entropy_coef=training_config.entropy_coef,
        grad_norm_clip=training_config.grad_norm_clip,
        checkpoint_dir=training_paths.checkpoints_dir,
        checkpoint_interval_updates=checkpoint_interval_updates,
        logs_dir=training_paths.logs_dir,
        logging_interval_updates=1,
        vtrace_rho_bar=training_config.vtrace_rho_bar,
        vtrace_c_bar=training_config.vtrace_c_bar,
        pass_action_id=pass_action_id,
    )

    env = _build_env(stack, profile=profile, num_envs=num_envs, seed=seed)
    config_hash256 = compute_config_hash256(stack)
    latest_metrics: dict[str, float] = {}
    start_time = time.time()
    try:
        for update_index in range(max_updates):
            rollout, initial_hidden_state, final_seat_hidden = _collect_rollout(
                env,
                model,
                unroll_length=unroll_length,
                num_envs=num_envs,
                observation_dim=observation_dim,
                action_dim=action_dim,
                device=device,
                rng=np.random.default_rng(seed + update_index),
                pass_action_id=pass_action_id,
            )
            bootstrap_value = _bootstrap_values(
                model,
                rollout,
                final_seat_hidden,
                device=device,
            )
            learner_batch = _build_learner_batch(
                stack,
                rollout,
                bootstrap_value,
                action_dim=action_dim,
                initial_hidden_state=initial_hidden_state,
                pass_action_id=pass_action_id,
            )
            latest_metrics = learner.update(learner_batch)
            _write_scalars_record(
                scalars_path=training_paths.scalars_path,
                learner=learner,
                metrics=latest_metrics,
                start_time=start_time,
            )
            if learner.update_count % checkpoint_interval_updates == 0:
                ckpt_path = training_paths.checkpoints_dir / f"checkpoint_{learner.update_count}.pt"
                _write_checkpoint(
                    checkpoint_path=ckpt_path,
                    learner=learner,
                    stack=stack,
                    device=device,
                )

                if learner.model is None:
                    raise RuntimeError("Cannot persist a snapshot registry entry without a learner model")
                candidate_policy_id = _persist_snapshot_registry_entry(
                    training_paths=training_paths,
                    run_dir=artifacts.run_dir,
                    checkpoint_path=ckpt_path,
                    model_state_dict=learner.model.state_dict(),
                    config_hash256=config_hash256,
                    device=device,
                    update=int(learner.update_count),
                    policy_version=int(learner.get_policy_version()),
                )
                _run_snapshot_promotion_gate(
                    stack=stack,
                    contract=contract,
                    artifacts=artifacts,
                    training_paths=training_paths,
                    learner=learner,
                    candidate_policy_id=candidate_policy_id,
                    update_count=int(learner.update_count),
                    policy_version=int(learner.get_policy_version()),
                    run_id256=run_id256,
                    config_hash256=config_hash256,
                    spec_hash256=spec_hash256,
                )

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
                print(
                    "Periodic dev eval: "
                    f"update={learner.update_count} opponent=b0_randomlegal "
                    f"games={summary_payload['summary']['games']} "
                    f"mean={summary_payload['uncertainty']['mean']:.4f} "
                    f"stop_reason={summary_payload['stop_reason']}"
                )
    finally:
        env.close()

    if not latest_metrics:
        raise RuntimeError("The minimal train smoke finished without producing learner metrics")
    return latest_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal end-to-end M3-08 train smoke entrypoint")
    parser.add_argument("--stack-config", type=Path, required=True)
    parser.add_argument("--spec-hash", type=str, default="", help="Expected spec hash or spec bundle SHA-256")
    parser.add_argument(
        "--config-hash",
        type=str,
        default="",
        help="Expected config_hash256 for contract validation",
    )
    parser.add_argument("--run-label", type=str, default="", help="Optional run directory label override")
    parser.add_argument("--run-id", dest="run_id_alias", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--num-envs", type=int, default=2, help="Tiny env count for the minimal smoke run")
    parser.add_argument("--unroll-length", type=int, default=4, help="Tiny rollout length for the smoke run")
    parser.add_argument("--max-updates", type=int, default=1, help="Number of learner updates to run")
    parser.add_argument("--profile", type=str, default="", help="Optional simulator profile override")
    parser.add_argument("--device", type=str, default="", help="Optional learner device override")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override")
    parser.add_argument(
        "--checkpoint-interval-updates",
        type=int,
        default=1,
        help="Checkpoint cadence for the minimal smoke run",
    )
    args = parser.parse_args()
    run_label = _resolve_run_label(parser, args.run_label, args.run_id_alias)

    num_envs = _require_positive_int("--num-envs", args.num_envs)
    unroll_length = _require_positive_int("--unroll-length", args.unroll_length)
    max_updates = _require_positive_int("--max-updates", args.max_updates)
    checkpoint_interval_updates = _require_positive_int(
        "--checkpoint-interval-updates",
        args.checkpoint_interval_updates,
    )

    stack = load_stack_config(args.stack_config)
    simulator_contract = load_simulator_contract(stack.root)
    assert_spec_bundle_contract(args.spec_hash, simulator_contract.spec_bundle)

    spec_hash256 = simulator_contract.spec_hash256
    config_hash256 = compute_config_hash256(stack)
    _require_matching_hash(
        flag_name="--config-hash",
        expected=_expected_sha256(args.config_hash, flag_name="--config-hash"),
        actual=config_hash256,
    )

    git_commit = _git_commit()
    start_nonce = _start_nonce()
    run_id256 = compute_run_id256(spec_hash256, config_hash256, git_commit or None, start_nonce)
    run_id64 = f"{compute_run_id64(spec_hash256, config_hash256, git_commit or None, start_nonce):016x}"
    run_dir_name = run_label or default_run_dir_name(run_id64)

    print_startup_banner(
        spec_hash256,
        config_hash256,
        run_id64=run_id64,
        run_id256=run_id256,
        run_label=run_label,
        run_dir_name=run_dir_name,
        spec_mismatch_policy=_spec_mismatch_policy(stack),
    )
    print(
        "Verified runtime spec bundle: "
        f"compat={simulator_contract.simulator.get('compatibility_hash', '')} sha256={spec_hash256}"
    )
    print(f"Loaded stack config with {len(stack.components)} components")

    manifest = RunManifest(
        run_id256=run_id256,
        run_id64=run_id64,
        start_nonce=start_nonce,
        git_commit=git_commit,
        git_dirty=_git_dirty(),
        spec_hash256=spec_hash256,
        config_hash256=config_hash256,
        simulator=simulator_contract.simulator,
        spec_bundle=simulator_contract.spec_bundle,
        config_canonical=canonical_config_dict(stack),
        seed_files=build_seed_file_manifest(stack.seed_sets, root=stack.root),
        hardware=_hardware_summary(),
        evaluation_pinning=_evaluation_pinning(stack),
        policy_set_selection=_policy_set_selection(stack),
    )
    artifacts = write_run_artifacts(
        stack.root / "runs",
        manifest,
        run_label=run_label or None,
    )
    print(f"Wrote manifest: {artifacts.manifest_path}")

    manifest_only_reason = _minimal_training_prerequisite_failure(stack)
    if manifest_only_reason is not None:
        _print_manifest_only_message(manifest_only_reason)
        return

    profile = _resolve_runtime_profile(stack, args.profile)
    device = _resolve_device(stack, args.device)
    seed = _resolve_seed(stack, args.seed)

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
    )
    print(
        "Completed minimal training run: "
        f"loss={metrics.get('loss', 0.0):.6f} "
        f"policy_loss={metrics.get('policy_loss', 0.0):.6f} "
        f"value_loss={metrics.get('value_loss', 0.0):.6f} "
        f"entropy={metrics.get('entropy', 0.0):.6f}"
    )


if __name__ == "__main__":
    main()
