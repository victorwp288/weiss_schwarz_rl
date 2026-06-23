from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from weiss_rl.learners.structured_auxiliary import (
    active_public_heuristic_profiles,
    mix_public_heuristic_profile_logits,
    normalize_public_heuristic_profile_mode,
    normalize_public_heuristic_profiles,
    score_public_heuristic_target_logits,
    selected_public_heuristic_profiles,
)


def test_public_heuristic_profiles_are_normalized_deduplicated_and_defaulted() -> None:
    assert normalize_public_heuristic_profiles(None) == ("base",)
    assert normalize_public_heuristic_profiles([" Base ", "aggressive", "base", ""]) == ("base", "aggressive")

    with pytest.raises(ValueError, match="unsupported profiles: unknown"):
        normalize_public_heuristic_profiles(("unknown",))


def test_public_heuristic_profile_mode_is_normalized_and_validated() -> None:
    assert normalize_public_heuristic_profile_mode(None) == "mixture"
    assert normalize_public_heuristic_profile_mode(" Cycle ") == "cycle"

    with pytest.raises(ValueError, match="teacher_public_heuristic_profile_mode"):
        normalize_public_heuristic_profile_mode("round_robin")


def test_active_public_heuristic_profiles_default_and_end_update_rules() -> None:
    assert active_public_heuristic_profiles((), update_count=10, end_updates=-1) == ("base",)
    assert active_public_heuristic_profiles(("base", "control"), update_count=10, end_updates=-1) == (
        "base",
        "control",
    )
    assert active_public_heuristic_profiles(("base", "control"), update_count=11, end_updates=10) == ("base",)


def test_selected_public_heuristic_profiles_preserves_cycle_after_end_update_fallback() -> None:
    assert selected_public_heuristic_profiles(
        ("base", "aggressive", "control"),
        profile_mode="cycle",
        update_count=4,
        end_updates=-1,
    ) == ("aggressive",)
    assert selected_public_heuristic_profiles(
        ("base", "aggressive", "control"),
        profile_mode="cycle",
        update_count=4,
        end_updates=3,
    ) == ("base",)
    assert selected_public_heuristic_profiles(
        ("base", "aggressive"),
        profile_mode="mixture",
        update_count=1,
        end_updates=-1,
    ) == ("base", "aggressive")


def test_mix_public_heuristic_profile_logits_preserves_rowwise_mixture_normalization() -> None:
    offsets = torch.tensor([0, 2, 5], dtype=torch.long)
    profile_a = torch.tensor([3.0, 1.0, 0.0, 2.0, 4.0], dtype=torch.float32)
    profile_b = torch.tensor([1.0, 3.0, 4.0, 2.0, 0.0], dtype=torch.float32)

    mixed = mix_public_heuristic_profile_logits(
        [profile_a, profile_b],
        offsets=offsets,
        temperature=1.0,
        device=torch.device("cpu"),
    )

    expected_rows: list[torch.Tensor] = []
    for start, end in ((0, 2), (2, 5)):
        log_probs = torch.stack(
            [
                profile_a[start:end] - torch.logsumexp(profile_a[start:end], dim=0),
                profile_b[start:end] - torch.logsumexp(profile_b[start:end], dim=0),
            ],
            dim=0,
        )
        expected_rows.append(torch.logsumexp(log_probs, dim=0) - torch.log(torch.tensor(2.0)))
    torch.testing.assert_close(mixed, torch.cat(expected_rows))


def test_mix_public_heuristic_profile_logits_handles_empty_and_single_profile_cases() -> None:
    single = torch.tensor([1.0, 2.0])

    assert (
        mix_public_heuristic_profile_logits(
            [],
            offsets=torch.tensor([0], dtype=torch.long),
            temperature=1.0,
            device=torch.device("cpu"),
        ).numel()
        == 0
    )
    assert (
        mix_public_heuristic_profile_logits(
            [single],
            offsets=torch.tensor([0, 2], dtype=torch.long),
            temperature=1.0,
            device=torch.device("cpu"),
        )
        is single
    )
    torch.testing.assert_close(
        mix_public_heuristic_profile_logits(
            [single, single + 1.0],
            offsets=torch.tensor([0], dtype=torch.long),
            temperature=1.0,
            device=torch.device("cpu"),
        ),
        single,
    )


def test_score_public_heuristic_target_logits_selects_profiles_and_passes_context() -> None:
    offsets = torch.tensor([0, 2, 4], dtype=torch.long)
    legal_actions = SimpleNamespace(offsets=offsets)
    obs_rows = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    context = {"stage": torch.ones((2, 1))}
    calls: list[tuple[str, torch.Tensor, object, object]] = []

    class _Scorer:
        def score_packed_public_heuristic_candidates(
            self,
            obs: torch.Tensor,
            legal: object,
            *,
            observation_context: object,
            scoring_profile: str,
        ) -> torch.Tensor:
            calls.append((scoring_profile, obs, legal, observation_context))
            values = {
                "base": torch.tensor([1.0, 0.0, 0.0, 1.0]),
                "aggressive": torch.tensor([0.0, 1.0, 2.0, 0.0]),
                "control": torch.tensor([2.0, 0.0, 0.0, 2.0]),
            }
            return values[scoring_profile]

    logits = score_public_heuristic_target_logits(
        forward_model=_Scorer(),
        obs_rows=obs_rows,
        legal_actions=legal_actions,
        observation_context=context,
        profiles=("base", "aggressive", "control"),
        profile_mode="cycle",
        update_count=1,
        end_updates=-1,
        temperature=1.0,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(logits, torch.tensor([0.0, 1.0, 2.0, 0.0]))
    assert len(calls) == 1
    assert calls[0][0] == "aggressive"
    assert calls[0][1] is obs_rows
    assert calls[0][2] is legal_actions
    assert calls[0][3] is context
