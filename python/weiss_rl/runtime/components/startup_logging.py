"""Startup performance logging for QueueRuntime."""

from __future__ import annotations

from typing import Any


def log_runtime_startup(
    runtime: Any,
    *,
    model_kind: str,
    training_config: Any,
    structured_warmstart_cfg: Any,
) -> None:
    performance_logger = getattr(runtime, "_performance_logger", None)
    if performance_logger is None:
        return

    rows_per_actor_unroll = int(runtime.config.unroll_length) * int(runtime.config.envs_per_actor)
    batch_env_steps = int(runtime.config.batch_unrolls_per_update) * rows_per_actor_unroll
    performance_logger.log(
        {
            "kind": "runtime_startup_v1",
            "actor_device": runtime._device.type,
            "actor_device_layout": list(dict.fromkeys(runtime._process_actor_device_names))
            if runtime._use_process_collectors
            else [str(runtime._device)],
            "compile_actor_inference": bool(runtime._compile_actor_inference),
            "fixed_opponent_backend": runtime._fixed_opponent_backend,
            "actor_policy_backend": runtime._actor_policy_backend,
            "actor_heuristic_fraction": float(runtime._actor_heuristic_fraction),
            "actor_sampling_temperature": float(runtime.config.actor_sampling_temperature),
            "runtime_actor_count": int(runtime.config.actor_count),
            "runtime_envs_per_actor": int(runtime.config.envs_per_actor),
            "runtime_total_envs": int(runtime.config.total_envs),
            "runtime_unroll_length": int(runtime.config.unroll_length),
            "runtime_rows_per_actor_unroll": int(rows_per_actor_unroll),
            "runtime_batch_unrolls_per_update": int(runtime.config.batch_unrolls_per_update),
            "runtime_batch_env_steps": int(batch_env_steps),
            "runtime_queue_capacity_unrolls": int(runtime.config.queue_capacity_unrolls),
            "collection_backend": runtime._collection_backend,
            "league_enabled": bool(runtime._league_enabled),
            "model_kind": model_kind,
            "structured_fixed_opponents_expected": bool(runtime._structured_fixed_opponents_expected),
            "structured_warmstart_enabled": bool(
                training_config is not None and bool(getattr(training_config, "structured_warmstart_enabled", False))
            ),
            "structured_warmstart_flag_enabled": bool(
                structured_warmstart_cfg is not None and bool(getattr(structured_warmstart_cfg, "enabled", False))
            ),
            "use_central_batched_collection": bool(runtime._use_central_batched_collection),
            "use_process_collectors": bool(runtime._use_process_collectors),
        }
    )


__all__ = ["log_runtime_startup"]
