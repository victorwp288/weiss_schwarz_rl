from __future__ import annotations

import importlib
import sys
from typing import Any

import torch

from weiss_rl.config.overrides import apply_stack_overrides
from weiss_rl.core.simulator_contract import validate_imported_weiss_sim_runtime


def resolve_runtime_profile(stack: Any, profile_override: str) -> str:
    if profile_override.strip():
        return profile_override.strip()
    system_config = stack.config.system
    if system_config is None:
        return "fast"
    return system_config.profile.local_iteration


def resolve_device(stack: Any, device_override: str) -> torch.device:
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


def resolve_seed(stack: Any, seed_override: int | None) -> int:
    if seed_override is not None:
        return int(seed_override)
    reproducibility = stack.config.reproducibility
    if reproducibility is None:
        return 7
    return int(reproducibility.seed_derivation.base_seed64)


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

    try:
        validate_imported_weiss_sim_runtime(weiss_sim)
    except RuntimeError as exc:
        return str(exc)

    return None


def noleague_training_prerequisite_failure(stack: Any) -> str | None:
    """Return a hard-fail reason when B1 NoLeague is configured as disguised heuristic imitation."""

    experiment = getattr(stack.config, "experiment", None)
    role = "" if experiment is None else str(getattr(experiment, "role", "")).strip()
    if role != "baseline_noleague":
        return None

    training = getattr(stack.config, "training", None)
    model = getattr(stack.config, "model", None)
    league = getattr(stack.config, "league", None)
    if training is None or model is None or league is None:
        return None

    if bool(getattr(league, "enabled", False)):
        return "B1 NoLeague requires league.enabled=false"
    if str(getattr(training, "actor_policy_backend", "model")).strip().lower() != "model":
        return "B1 NoLeague must use training.actor_policy_backend=model"
    if float(getattr(training, "actor_heuristic_fraction", 0.0)) != 0.0:
        return "B1 NoLeague must set training.actor_heuristic_fraction=0.0"
    if float(getattr(training, "actor_heuristic_final_fraction", 0.0)) != 0.0:
        return "B1 NoLeague must set training.actor_heuristic_final_fraction=0.0"
    if str(getattr(training, "teacher_aux_mode", "off")).strip().lower() != "off":
        return "B1 NoLeague must set training.teacher_aux.mode=off"
    if bool(getattr(training, "structured_warmstart_enabled", False)):
        return "B1 NoLeague must disable training.structured_warmstart"
    if float(getattr(training, "teacher_public_heuristic_coef", 0.0)) != 0.0:
        return "B1 NoLeague must set training.structured_aux.teacher_public_heuristic_coef=0.0"
    if float(getattr(model, "public_heuristic_logit_bias_scale", 0.0)) != 0.0:
        return "B1 NoLeague must set model.public_heuristic_logit_bias_scale=0.0"
    actor_bias = float(getattr(model, "public_heuristic_actor_logit_bias_scale", 0.0))
    if actor_bias not in {-1.0, 0.0}:
        return "B1 NoLeague must not use actor-side public heuristic logit bias"

    sampling = getattr(league, "sampling", None)
    if sampling is not None:
        nonzero_sampling_fields = {
            "heuristic_public_mix_fraction": float(getattr(sampling, "heuristic_public_mix_fraction", 0.0)),
            "heuristic_public_variant_mix_fraction": float(
                getattr(sampling, "heuristic_public_variant_mix_fraction", 0.0)
            ),
            "noleague_baseline_mix_fraction": float(getattr(sampling, "noleague_baseline_mix_fraction", 0.0)),
            "heuristic_public_reserved_envs_per_actor": float(
                getattr(sampling, "heuristic_public_reserved_envs_per_actor", 0)
            ),
            "noleague_baseline_reserved_envs_per_actor": float(
                getattr(sampling, "noleague_baseline_reserved_envs_per_actor", 0)
            ),
        }
        active = {key: value for key, value in nonzero_sampling_fields.items() if value != 0.0}
        if active:
            fields = ", ".join(f"{key}={value:g}" for key, value in sorted(active.items()))
            return f"B1 NoLeague must not inherit fixed-opponent/league sampling pressure: {fields}"

    return None


def print_manifest_only_message(reason: str) -> None:
    print("Manifest scaffold only: no learner training or rollout collection was executed.")
    print(f"Reason: {reason}.")


def raise_runtime_prerequisite_failure(reason: str) -> None:
    raise RuntimeError(
        "Canonical simulator-backed training requires a weiss_sim runtime with stepping support. "
        f"Startup failed because {reason}."
    )


def raise_noleague_training_prerequisite_failure(reason: str) -> None:
    raise RuntimeError(
        f"Canonical B1 NoLeague training route is structurally invalid. Startup failed because {reason}."
    )
