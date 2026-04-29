"""CLI bootstrap helpers for the public training entrypoint."""

from __future__ import annotations

import argparse
import importlib
import math
import sys
from pathlib import Path
from typing import Any

import torch

from weiss_rl.config import apply_stack_overrides

_SHA256_HEX_LENGTH = 64


def build_train_arg_parser() -> argparse.ArgumentParser:
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
        "--autoscale",
        action="store_true",
        help="Resolve training topology from training.scaling and the selected hardware profile",
    )
    parser.add_argument(
        "--autoscale-dry-run",
        action="store_true",
        help="Print resolved scaling topology and exit before writing run artifacts",
    )
    parser.add_argument(
        "--hardware-profile",
        type=str,
        default="local",
        help="Autoscale hardware profile: local, uc1-l40-3, uc1-l40-4, 8gpu-l40, or gpu<N>",
    )
    parser.add_argument(
        "--ddp",
        action="store_true",
        help="Enable torch.distributed learner gradient averaging; intended for torchrun",
    )
    parser.add_argument(
        "--ddp-backend",
        type=str,
        default="auto",
        choices=("auto", "nccl", "gloo"),
        help="torch.distributed backend for --ddp or WORLD_SIZE>1",
    )
    parser.add_argument(
        "--ddp-timeout-seconds",
        type=int,
        default=1800,
        help=(
            "torch.distributed process-group timeout. Keep this comfortably above the longest "
            "rank-0-only evaluation or checkpoint gate on multi-GPU server runs."
        ),
    )
    parser.add_argument(
        "--max-wall-clock-minutes",
        type=float,
        default=None,
        help="Optional wall-clock budget in minutes; the run stops cleanly between updates when the budget is reached",
    )
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
    parser.add_argument(
        "--resume-allow-config-mismatch",
        action="store_true",
        help=(
            "Allow loading a checkpoint whose config hash differs from the current stack. "
            "Spec hash and algorithm are still checked; use only for explicit research continuations."
        ),
    )
    parser.add_argument(
        "--resume-reset-optimizer",
        action="store_true",
        help="Load checkpoint model weights and counters but start with a fresh optimizer and grad scaler state.",
    )
    return parser


def resolve_run_label(parser: argparse.ArgumentParser, run_label: str, run_id_alias: str) -> str:
    normalized_label = run_label.strip()
    normalized_alias = run_id_alias.strip()
    if normalized_label and normalized_alias and normalized_label != normalized_alias:
        parser.error("--run-label and deprecated --run-id must match when both are provided")
    if normalized_alias:
        print("Warning: --run-id is deprecated; use --run-label instead.", file=sys.stderr)
    return normalized_label or normalized_alias


def require_positive_int(name: str, value: int) -> int:
    number = int(value)
    if number < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return number


def require_positive_optional_float(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a finite number > 0, got {value}")
    return number


def normalize_sha256(value: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != _SHA256_HEX_LENGTH:
        return ""
    if any(char not in "0123456789abcdef" for char in normalized):
        return ""
    return normalized


def expected_sha256(value: str, *, flag_name: str) -> str:
    if not value.strip():
        return ""
    normalized = normalize_sha256(value)
    if not normalized:
        raise ValueError(f"{flag_name} must be a 64-character lowercase or uppercase SHA-256 hex string")
    return normalized


def require_matching_hash(*, flag_name: str, expected: str, actual: str) -> None:
    if expected and expected != actual:
        raise RuntimeError(f"{flag_name} mismatch: expected {expected}, observed {actual}")


def apply_training_flag_overrides(
    stack: Any,
    *,
    enable_profile_timers: bool,
    enable_torch_profiler: bool,
) -> Any:
    training_config = stack.config.training
    if training_config is None:
        return stack
    overrides: dict[str, Any] = {}
    if enable_profile_timers and not bool(training_config.profile_timers):
        overrides["training.profile_timers"] = True
    if enable_torch_profiler and not bool(training_config.torch_profiler):
        overrides["training.torch_profiler"] = True
    return apply_stack_overrides(stack, overrides)


def resolve_runtime_profile(stack: Any, profile_override: str) -> str:
    if profile_override.strip():
        return profile_override.strip()
    system_config = stack.config.system
    if system_config is None:
        return "fast"
    return system_config.profile.local_iteration


def format_cuda_unavailable_fallback_message() -> str:
    return "Requested CUDA device is unavailable; falling back to cpu for the canonical single-node run."


def resolve_device(stack: Any, device_override: str) -> torch.device:
    requested = device_override.strip()
    if not requested:
        system_config = stack.config.system
        requested = "cpu" if system_config is None else getattr(system_config, "learner_device", "cpu")
    normalized = str(requested).strip().lower()
    if normalized in {"auto", "cuda:auto"}:
        requested = "cuda:0" if torch.cuda.is_available() and int(torch.cuda.device_count()) > 0 else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(format_cuda_unavailable_fallback_message(), file=sys.stderr)
        requested = "cpu"
    return torch.device(requested)


def resolve_seed(stack: Any, seed_override: int | None) -> int:
    if seed_override is not None:
        return int(seed_override)
    reproducibility = stack.config.reproducibility
    if reproducibility is None:
        return 7
    return int(reproducibility.seed_derivation.base_seed64)


def manifest_scaffold_only_reason(stack: Any) -> str | None:
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


def runtime_training_prerequisite_failure(stack: Any) -> str | None:
    if manifest_scaffold_only_reason(stack) is not None:
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


def format_spec_bundle_status_message(
    *,
    public_demo_enabled: bool,
    compatibility_hash: object,
    spec_hash256: str,
) -> str:
    prefix = "Loaded synthetic public-demo spec bundle: " if public_demo_enabled else "Verified runtime spec bundle: "
    return prefix + f"compat={compatibility_hash} sha256={spec_hash256}"


def format_loaded_stack_config_message(component_count: int) -> str:
    return f"Loaded stack config with {int(component_count)} components"


def format_manifest_written_message(manifest_path: Path) -> str:
    return f"Wrote manifest: {manifest_path}"


def format_resume_run_dir_message(run_dir: Path) -> str:
    return f"Resuming existing run directory: {run_dir}"


def format_manifest_scaffold_only_message(reason: str) -> tuple[str, str]:
    return (
        "Manifest scaffold only: no learner training or rollout collection was executed.",
        f"Reason: {reason}.",
    )


def format_tensorboard_disabled_message(unavailable_reason: str | None) -> str:
    return "TensorBoard logging is disabled: " + (
        "SummaryWriter unavailable" if unavailable_reason is None else str(unavailable_reason)
    )


def format_public_demo_staged_message(
    *,
    mode: str,
    policy_count: int,
    catalog_path: Path,
) -> str:
    return (
        "Staged public-demo toy catalog and policy bundle: "
        f"mode={mode} policy_count={int(policy_count)} catalog={catalog_path}"
    )


def format_public_demo_disclaimer_message() -> str:
    return (
        "Public demo mode is intentionally synthetic and demo-only. "
        "It does not execute simulator training or claim thesis-grade results."
    )
