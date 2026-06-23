from __future__ import annotations

from typing import Any, cast

import numpy as np
import torch
from weiss_rl.runtime import QueueRuntime


def bare_queue_runtime() -> QueueRuntime:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._device = torch.device("cpu")
    runtime_any._actor_amp_enabled = False
    return runtime


class FixedRng:
    def __init__(self, values: tuple[float, ...]) -> None:
        self.values = np.asarray(values, dtype=np.float64)

    def random(self, size: int) -> np.ndarray:
        assert size <= self.values.shape[0]
        return self.values[:size]
