from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from pathlib import Path

from weiss_rl.config import load_stack_config
from weiss_rl.envs.pool_factory import build_env_config_from_stack, make_env_pool_from_config


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_build_env_config_from_stack_emits_reward_json() -> None:
    stack = load_stack_config(_repo_root() / "configs" / "presets" / "typed_local.yaml")

    env_config = build_env_config_from_stack(stack, seed=17)

    assert env_config["seed"] == 17
    reward_payload = json.loads(env_config["reward_json"])
    assert reward_payload == {
        "board_reward": 0.0,
        "damage_reward": 0.05,
        "enable_shaping": True,
        "level_reward": 0.0,
        "no_progress_penalty": 0.0,
        "terminal_draw": 0.0,
        "terminal_loss": -1.0,
        "terminal_timeout": -0.1,
        "terminal_win": 1.0,
    }
    assert "curriculum_json" not in env_config


def test_build_env_config_from_stack_emits_curriculum_json_when_present(tmp_path: Path) -> None:
    repo_root = _repo_root()
    fake_repo = tmp_path
    (fake_repo / "configs").mkdir()
    (fake_repo / "python").mkdir()
    thesis_path = fake_repo / "configs" / "typed_thesis_locked.yaml"
    thesis_path.write_text(
        (repo_root / "configs/presets/typed_thesis_locked.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    local_path = fake_repo / "configs" / "typed_local.yaml"
    local_path.write_text(
        (
            (repo_root / "configs/presets/typed_local.yaml").read_text(encoding="utf-8")
            + "\ncurriculum:\n  simulator:\n    max_no_progress_decisions: 192\n    phase: opening\n    staged_decks:\n      - trial\n"
        ),
        encoding="utf-8",
    )
    stack = load_stack_config(local_path)

    env_config = build_env_config_from_stack(stack, seed=23)

    curriculum_payload = json.loads(env_config["curriculum_json"])
    assert curriculum_payload == {
        "max_no_progress_decisions": 192,
        "phase": "opening",
        "staged_decks": ["trial"],
    }


def test_make_env_pool_from_config_adapts_curriculum_for_high_level_api(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_make(
        *,
        mode: str,
        num_envs: int,
        max_decisions: int,
        max_ticks: int,
        observation_visibility: str,
        seed: int,
        curriculum: dict[str, object] | None = None,
        reward_json: str | None = None,
        legal_repr: str | None = None,
        obs_dtype: str | None = None,
    ):
        captured.update(
            {
                "mode": mode,
                "num_envs": num_envs,
                "max_decisions": max_decisions,
                "max_ticks": max_ticks,
                "observation_visibility": observation_visibility,
                "seed": seed,
                "curriculum": curriculum,
                "reward_json": reward_json,
                "legal_repr": legal_repr,
                "obs_dtype": obs_dtype,
            }
        )
        return SimpleNamespace(pool="fake-pool")

    def fake_fast(**kwargs):
        return fake_make(mode="fast", **kwargs)

    monkeypatch.setitem(sys.modules, "weiss_sim", SimpleNamespace(fast=fake_fast, make=fake_make))

    pool, layout = make_env_pool_from_config(
        {
            "max_decisions": 200,
            "max_ticks": 1000,
            "observation_visibility": "public",
            "seed": 7,
            "reward_json": '{"terminal_win": 1.0}',
            "curriculum_json": '{"max_no_progress_decisions": 192}',
        },
        profile="fast",
        num_envs=3,
    )

    assert pool == "fake-pool"
    assert layout == "i16_legal_ids"
    assert captured["mode"] == "fast"
    assert captured["curriculum"] == {"max_no_progress_decisions": 192}
    assert captured["reward_json"] == '{"terminal_win": 1.0}'
