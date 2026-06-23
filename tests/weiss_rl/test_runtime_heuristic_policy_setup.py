from __future__ import annotations

from types import SimpleNamespace

import pytest
from weiss_rl.eval.policies.set import HEURISTIC_PUBLIC_POLICY_ID
from weiss_rl.runtime.components import heuristic_policy_setup as setup
from weiss_rl.runtime.components.teacher_settings import RuntimeTeacherSettings


class _FakeHeuristicPublicPolicy:
    calls: list[tuple[object, object]] = []

    @classmethod
    def from_spec_bundle(cls, spec_bundle: object, scoring_profile: object = None) -> tuple[object, object]:
        cls.calls.append((spec_bundle, scoring_profile))
        return spec_bundle, scoring_profile


def _teacher_settings(*, enabled: bool = False, profiles: tuple[str, ...] = ("base",)) -> RuntimeTeacherSettings:
    return RuntimeTeacherSettings(
        teacher_guidance_enabled=enabled,
        teacher_aux_mode="always",
        teacher_label_profiles=profiles,
        teacher_label_profile_mode="mixture",
        teacher_label_profiles_end_updates=-1,
        teacher_guidance_warmstart_updates=0,
    )


def test_runtime_heuristic_policy_setup_builds_profiled_teacher_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeHeuristicPublicPolicy.calls = []
    monkeypatch.setattr(setup, "HeuristicPublicPolicy", _FakeHeuristicPublicPolicy)
    action_catalog = SimpleNamespace(
        families=(SimpleNamespace(name="main_play_character"), SimpleNamespace(name="attack")),
        attack_type_names=("front", "direct"),
    )

    policies = setup.build_runtime_heuristic_policy_setup(
        spec_bundle={"contract": "fake"},
        action_catalog=action_catalog,
        teacher_settings=_teacher_settings(enabled=True, profiles=("base", "Aggressive", "control")),
        actor_policy_backend="model",
        league_config=None,
        diverse_opponent_actor_count=0,
        actor_count=4,
    )

    assert policies.teacher_policy == ({"contract": "fake"}, None)
    assert policies.teacher_policy_by_profile == {
        "base": ({"contract": "fake"}, None),
        "aggressive": ({"contract": "fake"}, "aggressive"),
        "control": ({"contract": "fake"}, "control"),
    }
    assert policies.teacher_action_catalog is action_catalog
    assert policies.teacher_family_index == {"main_play_character": 0, "attack": 1}
    assert policies.teacher_attack_type_index == {"front": 0, "direct": 1}
    assert policies.opponent_heuristic_policies == {}


def test_runtime_heuristic_policy_setup_builds_league_opponent_policies(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeHeuristicPublicPolicy.calls = []
    monkeypatch.setattr(setup, "HeuristicPublicPolicy", _FakeHeuristicPublicPolicy)
    monkeypatch.setattr(
        setup, "heuristic_public_policy_ids", lambda *, include_base: ("variant:aggro", "variant:control")
    )
    monkeypatch.setattr(
        setup,
        "heuristic_public_profile_name_for_policy_id",
        lambda policy_id: str(policy_id).split(":", maxsplit=1)[1],
    )
    league_config = SimpleNamespace(
        sampling=SimpleNamespace(
            heuristic_public_mix_fraction=0.25,
            heuristic_public_variant_mix_fraction=0.0,
            heuristic_public_variant_final_mix_fraction=0.5,
        )
    )

    policies = setup.build_runtime_heuristic_policy_setup(
        spec_bundle={"contract": "fake"},
        action_catalog=None,
        teacher_settings=_teacher_settings(),
        actor_policy_backend="model",
        league_config=league_config,
        diverse_opponent_actor_count=0,
        actor_count=4,
    )

    assert policies.teacher_policy is None
    assert policies.opponent_heuristic_policies == {
        HEURISTIC_PUBLIC_POLICY_ID: ({"contract": "fake"}, None),
        "variant:aggro": ({"contract": "fake"}, "aggro"),
        "variant:control": ({"contract": "fake"}, "control"),
    }


def test_runtime_heuristic_policy_setup_requires_spec_for_heuristic_actor_backend() -> None:
    with pytest.raises(RuntimeError, match="training.actor_policy_backend=heuristic_public requires"):
        setup.build_runtime_heuristic_policy_setup(
            spec_bundle=None,
            action_catalog=None,
            teacher_settings=_teacher_settings(),
            actor_policy_backend="heuristic_public",
            league_config=None,
            diverse_opponent_actor_count=0,
            actor_count=1,
        )
