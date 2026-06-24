from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from inspect import signature
from typing import Any, Literal

from weiss_rl.envs.env_config import build_env_config_from_stack as build_env_config_from_stack

Profile = Literal["debug", "balanced", "fast"]
LayoutName = Literal["mask", "i16_legal_ids"]

REQUIRED_ENV_CONFIG_KEYS = ("max_decisions", "max_ticks", "observation_visibility")
PROFILE_ORDER = ("debug", "balanced", "fast")


@dataclass(frozen=True)
class _ProfileSettings:
    entrypoint: Literal["fast", "inspect"]
    legal_repr: str
    obs_dtype: str
    layout_name: LayoutName


PROFILE_SETTINGS: dict[str, _ProfileSettings] = {
    "debug": _ProfileSettings(
        entrypoint="inspect",
        legal_repr="mask_u8",
        obs_dtype="i32",
        layout_name="mask",
    ),
    "balanced": _ProfileSettings(
        entrypoint="fast",
        legal_repr="mask_u8",
        obs_dtype="i16",
        layout_name="mask",
    ),
    "fast": _ProfileSettings(
        entrypoint="fast",
        legal_repr="ids_u16",
        obs_dtype="i16",
        layout_name="i16_legal_ids",
    ),
}


def _resolve_profile_settings(profile: str) -> _ProfileSettings:
    settings = PROFILE_SETTINGS.get(profile)
    if settings is None:
        expected = ", ".join(PROFILE_ORDER)
        raise ValueError(f"Unknown profile {profile!r}. Expected one of: {expected}.")
    return settings


def _resolve_num_envs(kwargs: dict[str, Any], explicit_num_envs: int | None) -> int:
    config_num_envs = kwargs.pop("num_envs", None)

    if explicit_num_envs is not None and config_num_envs is not None:
        raise ValueError("num_envs was provided twice. Pass it either in env_config or as num_envs=, not both.")

    raw_num_envs = explicit_num_envs if explicit_num_envs is not None else config_num_envs
    if raw_num_envs is None:
        raise ValueError("num_envs is required. Pass it via num_envs= or include it in env_config.")

    num_envs = int(raw_num_envs)
    if num_envs < 1:
        raise ValueError(f"num_envs must be >= 1, got {num_envs}.")
    return num_envs


def _validate_env_config(kwargs: Mapping[str, Any]) -> None:
    reserved_keys = [key for key in ("legal_repr", "obs_dtype") if key in kwargs]
    if reserved_keys:
        raise ValueError(f"env_config cannot override profile-managed keys: {reserved_keys}")

    missing = [key for key in REQUIRED_ENV_CONFIG_KEYS if key not in kwargs]
    if missing:
        raise ValueError(f"env_config missing required keys: {missing}")


def make_env_pool_from_config(
    env_config: Mapping[str, Any],
    *,
    profile: Profile,
    num_envs: int | None = None,
) -> tuple[Any, LayoutName]:
    """Build a `weiss_sim` pool using profile-derived simulator settings.

    `env_config` should contain the shared simulator settings. `num_envs` may be
    passed explicitly, or omitted when it is already present in `env_config`.
    Providing both is rejected so the call site stays unambiguous.
    """

    settings = _resolve_profile_settings(profile)

    kwargs = dict(env_config)
    _validate_env_config(kwargs)
    kwargs["num_envs"] = _resolve_num_envs(kwargs, num_envs)
    kwargs["legal_repr"] = settings.legal_repr
    kwargs["obs_dtype"] = settings.obs_dtype

    import weiss_sim

    factory = getattr(weiss_sim, settings.entrypoint)
    parameters = signature(factory).parameters
    uses_kwargs_wrapper = any(parameter.kind == parameter.VAR_KEYWORD for parameter in parameters.values())
    if uses_kwargs_wrapper and hasattr(weiss_sim, "make"):
        translated_kwargs = dict(kwargs)
        curriculum_json = translated_kwargs.pop("curriculum_json", None)
        if curriculum_json is not None:
            translated_kwargs["curriculum"] = json.loads(str(curriculum_json))
        env = weiss_sim.make(mode=settings.entrypoint, **translated_kwargs)
    elif "curriculum_json" not in parameters and "curriculum" in parameters:
        translated_kwargs = dict(kwargs)
        curriculum_json = translated_kwargs.pop("curriculum_json", None)
        if curriculum_json is not None:
            translated_kwargs["curriculum"] = json.loads(str(curriculum_json))
        env = factory(**translated_kwargs)
    else:
        env = factory(**kwargs)
    return env.pool, settings.layout_name
