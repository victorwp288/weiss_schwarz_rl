from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from weiss_rl.envs.pool_factory import make_env_pool_from_config

BASE_ENV_CONFIG = {
    "max_decisions": 480,
    "max_ticks": 100_000,
    "observation_visibility": "public",
}


def _install_stub_weiss_sim(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, object]]]:
    calls: list[tuple[str, dict[str, object]]] = []
    stub = ModuleType("weiss_sim")

    def make_factory(entrypoint: str):
        def factory(**kwargs: object) -> SimpleNamespace:
            calls.append((entrypoint, dict(kwargs)))
            return SimpleNamespace(pool=f"{entrypoint}-pool")

        return factory

    stub.inspect = make_factory("inspect")  # type: ignore[attr-defined]
    stub.fast = make_factory("fast")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "weiss_sim", stub)
    return calls


@pytest.mark.parametrize(
    ("profile", "entrypoint", "legal_repr", "obs_dtype", "layout_name"),
    [
        ("debug", "inspect", "mask_u8", "i32", "mask"),
        ("balanced", "fast", "mask_u8", "i16", "mask"),
        ("fast", "fast", "ids_u16", "i16", "i16_legal_ids"),
    ],
)
def test_make_env_pool_from_config_maps_profiles_to_simulator_settings(
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
    entrypoint: str,
    legal_repr: str,
    obs_dtype: str,
    layout_name: str,
) -> None:
    calls = _install_stub_weiss_sim(monkeypatch)

    pool, actual_layout_name = make_env_pool_from_config(
        BASE_ENV_CONFIG,
        profile=profile,  # type: ignore[arg-type]
        num_envs=7,
    )

    assert pool == f"{entrypoint}-pool"
    assert actual_layout_name == layout_name
    assert calls == [
        (
            entrypoint,
            {
                **BASE_ENV_CONFIG,
                "num_envs": 7,
                "legal_repr": legal_repr,
                "obs_dtype": obs_dtype,
            },
        )
    ]


def test_make_env_pool_from_config_rejects_unknown_profile_before_import() -> None:
    with pytest.raises(ValueError, match=r"Unknown profile 'turbo'"):
        make_env_pool_from_config(
            BASE_ENV_CONFIG,
            profile="turbo",  # type: ignore[arg-type]
            num_envs=4,
        )


def test_make_env_pool_from_config_rejects_duplicate_num_envs() -> None:
    with pytest.raises(ValueError, match="provided twice"):
        make_env_pool_from_config(
            {**BASE_ENV_CONFIG, "num_envs": 4},
            profile="fast",
            num_envs=8,
        )
