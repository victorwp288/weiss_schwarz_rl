"""Runtime-facing teacher guidance settings resolved at QueueRuntime startup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RuntimeTeacherSettings:
    teacher_guidance_enabled: bool
    teacher_aux_mode: str
    teacher_label_profiles: tuple[Any, ...]
    teacher_label_profile_mode: str
    teacher_label_profiles_end_updates: int
    teacher_guidance_warmstart_updates: int


def resolve_runtime_teacher_settings(*, training_config: Any) -> RuntimeTeacherSettings:
    structured_warmstart_cfg = (
        None if training_config is None else getattr(training_config, "structured_warmstart", None)
    )
    teacher_guidance_warmstart_updates = 0
    if structured_warmstart_cfg is not None and bool(getattr(structured_warmstart_cfg, "enabled", False)):
        teacher_guidance_warmstart_updates = max(0, int(getattr(structured_warmstart_cfg, "updates", 0)))

    return RuntimeTeacherSettings(
        teacher_guidance_enabled=bool(
            training_config is not None and bool(getattr(training_config, "structured_aux_enabled", False))
        ),
        teacher_aux_mode=(
            "always"
            if training_config is None
            else str(getattr(training_config, "teacher_aux_mode", "always")).strip().lower()
        ),
        teacher_label_profiles=(
            ("base",)
            if training_config is None
            else tuple(getattr(training_config, "teacher_public_heuristic_profiles", ())) or ("base",)
        ),
        teacher_label_profile_mode=(
            "mixture"
            if training_config is None
            else str(getattr(training_config, "teacher_public_heuristic_profile_mode", "mixture")).strip().lower()
        ),
        teacher_label_profiles_end_updates=(
            -1
            if training_config is None
            else int(getattr(training_config, "teacher_public_heuristic_profiles_end_updates", -1))
        ),
        teacher_guidance_warmstart_updates=teacher_guidance_warmstart_updates,
    )


__all__ = ["RuntimeTeacherSettings", "resolve_runtime_teacher_settings"]
