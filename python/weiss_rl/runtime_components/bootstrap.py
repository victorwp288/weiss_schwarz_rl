"""Bootstrap value computation for runtime unrolls."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def bootstrap_values_for_unroll(
    *,
    unroll: Any,
    actor_model: Any,
    bootstrap_device: torch.device,
    actor_amp_enabled: bool,
) -> np.ndarray:
    bootstrap_value = np.zeros((unroll.bootstrap_obs.shape[0],), dtype=np.float32)
    valid_rows = (unroll.bootstrap_actor == 0) | (unroll.bootstrap_actor == 1)
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
                torch.as_tensor(unroll.bootstrap_obs[valid_rows], device=bootstrap_device),
                torch.as_tensor(unroll.bootstrap_actor[valid_rows], device=bootstrap_device, dtype=torch.long),
                torch.as_tensor(unroll.final_hidden_state[valid_rows], device=bootstrap_device),
            )
        else:
            _, value_tensor, _ = actor_model.forward_seat_aware(
                torch.as_tensor(unroll.bootstrap_obs[valid_rows], device=bootstrap_device),
                torch.as_tensor(unroll.bootstrap_actor[valid_rows], device=bootstrap_device, dtype=torch.long),
                torch.as_tensor(unroll.final_hidden_state[valid_rows], device=bootstrap_device),
            )
    bootstrap_value[valid_rows] = value_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    return bootstrap_value
