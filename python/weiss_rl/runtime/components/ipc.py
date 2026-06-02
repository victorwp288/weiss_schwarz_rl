"""IPC serialization helpers for runtime model state payloads."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch


def serialize_state_dict_for_ipc(state_dict: dict[str, Any]) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            serialized[str(key)] = np.array(value.detach().cpu().numpy(), copy=True)
        else:
            serialized[str(key)] = copy.deepcopy(value)
    return serialized


def deserialize_state_dict_from_ipc(state_dict: dict[str, Any]) -> dict[str, Any]:
    restored: dict[str, Any] = {}
    for key, value in state_dict.items():
        if isinstance(value, np.ndarray):
            restored[str(key)] = torch.from_numpy(np.array(value, copy=True))
        else:
            restored[str(key)] = copy.deepcopy(value)
    return restored
