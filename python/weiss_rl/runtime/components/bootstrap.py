"""Bootstrap value computation for runtime unrolls."""

from __future__ import annotations

import time
from typing import Any, NamedTuple

import numpy as np
import torch


class CollectorBootstrapFields(NamedTuple):
    obs: np.ndarray
    actor: np.ndarray
    value: np.ndarray


def bootstrap_fields_from_batch(batch: Any) -> CollectorBootstrapFields:
    obs = np.asarray(batch.obs, dtype=np.float32)
    actor = np.asarray(batch.actor, dtype=np.int64)
    return CollectorBootstrapFields(
        obs=obs,
        actor=actor,
        value=np.zeros((obs.shape[0],), dtype=np.float32),
    )


def bootstrap_fields_from_batches(batches: list[Any]) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    fields = [bootstrap_fields_from_batch(batch) for batch in batches]
    return (
        [field.obs for field in fields],
        [field.actor for field in fields],
        [field.value for field in fields],
    )


def live_bootstrap_rows(bootstrap_actor: np.ndarray) -> np.ndarray:
    actor_array = np.asarray(bootstrap_actor, dtype=np.int64)
    return (actor_array == 0) | (actor_array == 1)


def model_bootstrap_values(
    *,
    bootstrap_obs: np.ndarray,
    bootstrap_actor: np.ndarray,
    hidden_state: Any,
    actor_model: Any,
    bootstrap_device: torch.device,
    actor_amp_enabled: bool,
) -> np.ndarray:
    obs_array = np.asarray(bootstrap_obs, dtype=np.float32)
    actor_array = np.asarray(bootstrap_actor, dtype=np.int64)
    bootstrap_value = np.zeros((obs_array.shape[0],), dtype=np.float32)
    valid_rows = live_bootstrap_rows(actor_array)
    if not np.any(valid_rows):
        return bootstrap_value
    with (
        torch.inference_mode(),
        torch.amp.autocast(
            device_type=bootstrap_device.type,
            enabled=bool(actor_amp_enabled and bootstrap_device.type == "cuda"),
        ),
    ):
        value_seat_aware = getattr(actor_model, "value_seat_aware", None)
        if callable(value_seat_aware):
            value_tensor = value_seat_aware(
                torch.as_tensor(obs_array[valid_rows], device=bootstrap_device),
                torch.as_tensor(actor_array[valid_rows], device=bootstrap_device, dtype=torch.long),
                torch.as_tensor(hidden_state[valid_rows], device=bootstrap_device),
            )
        else:
            _, value_tensor, _ = actor_model.forward_seat_aware(
                torch.as_tensor(obs_array[valid_rows], device=bootstrap_device),
                torch.as_tensor(actor_array[valid_rows], device=bootstrap_device, dtype=torch.long),
                torch.as_tensor(hidden_state[valid_rows], device=bootstrap_device),
            )
    bootstrap_value[valid_rows] = value_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    return bootstrap_value


def collector_bootstrap_fields_for_actor(
    *,
    batch: Any,
    actor: Any,
    actor_model: Any,
    bootstrap_device: torch.device,
    actor_amp_enabled: bool,
    values_required: bool,
    counters: dict[str, int],
) -> CollectorBootstrapFields:
    fields = bootstrap_fields_from_batch(batch)
    if bool(values_required) and np.any(live_bootstrap_rows(fields.actor)):
        bootstrap_started = time.perf_counter()
        fields = CollectorBootstrapFields(
            obs=fields.obs,
            actor=fields.actor,
            value=model_bootstrap_values(
                bootstrap_obs=fields.obs,
                bootstrap_actor=fields.actor,
                hidden_state=actor.seat_hidden,
                actor_model=actor_model,
                bootstrap_device=bootstrap_device,
                actor_amp_enabled=actor_amp_enabled,
            ),
        )
        counters["actor_bootstrap_ms"] += int((time.perf_counter() - bootstrap_started) * 1000.0)
    return fields


def add_shared_elapsed_ms(
    *,
    counters: list[dict[str, int]],
    key: str,
    started_at: float,
    divisor: int,
) -> None:
    elapsed_ms = int(((time.perf_counter() - started_at) * 1000.0) / max(int(divisor), 1))
    for counter in counters:
        counter[key] += elapsed_ms


def bootstrap_values_for_unroll(
    *,
    unroll: Any,
    actor_model: Any,
    bootstrap_device: torch.device,
    actor_amp_enabled: bool,
) -> np.ndarray:
    return model_bootstrap_values(
        bootstrap_obs=unroll.bootstrap_obs,
        bootstrap_actor=unroll.bootstrap_actor,
        hidden_state=unroll.final_hidden_state,
        actor_model=actor_model,
        bootstrap_device=bootstrap_device,
        actor_amp_enabled=actor_amp_enabled,
    )
