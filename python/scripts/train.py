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
from typing import Any

import numpy as np
import torch

from weiss_rl.cli_banner import print_startup_banner
from weiss_rl.config import StackConfig, canonical_config_dict, compute_config_hash256, load_stack_config
from weiss_rl.envs.decision_env import DecisionBoundaryEnv
from weiss_rl.envs.pool_factory import make_env_pool_from_config
from weiss_rl.learners.impala_learner import ImpalaLearner
from weiss_rl.learners.vtrace import VTraceTargets, compute_vtrace_targets
from weiss_rl.manifest import RunManifest, build_seed_file_manifest, default_run_dir_name, write_run_artifacts
from weiss_rl.masking import masked_logp_from_mask, sample_actions_from_mask
from weiss_rl.model import PolicyValueModel
from weiss_rl.repro import compute_run_id64, compute_run_id256
from weiss_rl.simulator_contract import SimulatorContract, load_simulator_contract
from weiss_rl.spec import assert_spec_bundle_contract

_SHA256_HEX_LENGTH = 64
_U64_MASK = (1 << 64) - 1


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


def _build_env(
    stack: StackConfig,
    *,
    profile: str,
    num_envs: int,
    seed: int,
) -> DecisionBoundaryEnv:
    environment_config = stack.config.environment
    if environment_config is None:
        raise RuntimeError("The locked stack is missing the environment config block")

    pool, layout_name = make_env_pool_from_config(
        {
            "max_decisions": int(environment_config.max_decisions),
            "max_ticks": int(environment_config.max_ticks),
            "observation_visibility": environment_config.observation_visibility,
            "seed": int(seed),
        },
        profile=profile,  # type: ignore[arg-type]
        num_envs=num_envs,
    )
    if layout_name != "mask":
        raise RuntimeError(
            "The minimal M3-08 training path expects mask legality because ImpalaLearner consumes legal_mask. "
            f"Profile {profile!r} resolved to layout {layout_name!r}."
        )
    return DecisionBoundaryEnv(pool, legality="mask", engine_status_policy="hard_fail")


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
                _write_checkpoint(
                    checkpoint_path=training_paths.checkpoints_dir / f"checkpoint_{learner.update_count}.pt",
                    learner=learner,
                    stack=stack,
                    device=device,
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
