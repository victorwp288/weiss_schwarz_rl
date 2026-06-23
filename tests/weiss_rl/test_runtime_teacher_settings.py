from __future__ import annotations

from types import SimpleNamespace

from weiss_rl.runtime.components.teacher_settings import resolve_runtime_teacher_settings


def test_runtime_teacher_settings_defaults_without_training_config() -> None:
    settings = resolve_runtime_teacher_settings(training_config=None)

    assert settings.teacher_guidance_enabled is False
    assert settings.teacher_aux_mode == "always"
    assert settings.teacher_label_profiles == ("base",)
    assert settings.teacher_label_profile_mode == "mixture"
    assert settings.teacher_label_profiles_end_updates == -1
    assert settings.teacher_guidance_warmstart_updates == 0


def test_runtime_teacher_settings_normalizes_training_config_modes() -> None:
    training_config = SimpleNamespace(
        structured_aux_enabled=True,
        teacher_aux_mode=" Warmstart ",
        teacher_public_heuristic_profiles=("base", "aggro"),
        teacher_public_heuristic_profile_mode=" Cycling ",
        teacher_public_heuristic_profiles_end_updates="17",
        structured_warmstart=SimpleNamespace(enabled=True, updates="8"),
    )

    settings = resolve_runtime_teacher_settings(training_config=training_config)

    assert settings.teacher_guidance_enabled is True
    assert settings.teacher_aux_mode == "warmstart"
    assert settings.teacher_label_profiles == ("base", "aggro")
    assert settings.teacher_label_profile_mode == "cycling"
    assert settings.teacher_label_profiles_end_updates == 17
    assert settings.teacher_guidance_warmstart_updates == 8


def test_runtime_teacher_settings_falls_back_to_base_profile_for_empty_profile_list() -> None:
    training_config = SimpleNamespace(teacher_public_heuristic_profiles=())

    settings = resolve_runtime_teacher_settings(training_config=training_config)

    assert settings.teacher_label_profiles == ("base",)


def test_runtime_teacher_settings_clamps_negative_warmstart_updates() -> None:
    training_config = SimpleNamespace(
        structured_warmstart=SimpleNamespace(enabled=True, updates=-5),
    )

    settings = resolve_runtime_teacher_settings(training_config=training_config)

    assert settings.teacher_guidance_warmstart_updates == 0


def test_runtime_teacher_settings_ignores_disabled_warmstart_updates() -> None:
    training_config = SimpleNamespace(
        structured_warmstart=SimpleNamespace(enabled=False, updates=9),
    )

    settings = resolve_runtime_teacher_settings(training_config=training_config)

    assert settings.teacher_guidance_warmstart_updates == 0
