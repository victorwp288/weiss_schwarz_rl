"""Typed inputs for central actor action execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from weiss_rl.runtime.components.collector_state import CollectorUnrollState
from weiss_rl.runtime.components.teacher_labels import TeacherLabelArrays


@dataclass(frozen=True, slots=True)
class CentralActorActionInputs:
    actor: Any
    batch: Any
    state: CollectorUnrollState
    obs_step: np.ndarray
    focal_rows: np.ndarray
    logits_step: np.ndarray | None
    config: Any
    action_family_index: Mapping[str, int] | None


@dataclass(frozen=True, slots=True)
class PackedCentralActorActionMode:
    actor_index: int
    structured_central_packed: bool
    structured_action_steps: Sequence[np.ndarray] | None
    structured_logp_steps: Sequence[np.ndarray] | None


@dataclass(frozen=True, slots=True)
class PackedCentralActorActionCallbacks:
    ensure_legal_action_meta: Callable[..., np.ndarray | None]
    teacher_labels_from_ids: Callable[..., TeacherLabelArrays]


@dataclass(frozen=True, slots=True)
class MaskCentralActorActionCallbacks:
    teacher_labels_from_mask: Callable[..., TeacherLabelArrays]


__all__ = [
    "CentralActorActionInputs",
    "MaskCentralActorActionCallbacks",
    "PackedCentralActorActionCallbacks",
    "PackedCentralActorActionMode",
]
