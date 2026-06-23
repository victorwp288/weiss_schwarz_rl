from __future__ import annotations

from types import SimpleNamespace

import pytest
from weiss_rl.runtime.components.opponents import (
    active_actor_heuristic_fraction,
    active_heuristic_public_mix_fraction,
    active_heuristic_public_variant_mix_fraction,
    active_mirror_mix_fraction,
    active_noleague_baseline_mix_fraction,
    active_warmup_snapshot_mix_fraction,
)


def test_active_mix_fractions_preserve_anneal_and_expiry_rules() -> None:
    league_config = SimpleNamespace(
        warmup=SimpleNamespace(first_updates=10),
        sampling=SimpleNamespace(
            heuristic_public_mix_fraction=1.0,
            heuristic_public_mix_end_updates=5,
            heuristic_public_final_mix_fraction=0.25,
            heuristic_public_variant_mix_fraction=0.4,
            heuristic_public_variant_mix_end_updates=4,
            heuristic_public_variant_final_mix_fraction=0.1,
            mirror_mix_fraction=0.6,
            mirror_mix_end_updates=6,
            mirror_final_mix_fraction=0.3,
            noleague_baseline_mix_fraction=0.3,
            noleague_baseline_mix_end_updates=3,
            warmup_snapshot_mix_fraction=0.2,
        ),
    )

    assert active_heuristic_public_mix_fraction(league_config=league_config, reference_update=3) == pytest.approx(0.55)
    assert active_heuristic_public_mix_fraction(league_config=league_config, reference_update=5) == pytest.approx(0.25)
    assert active_heuristic_public_variant_mix_fraction(
        league_config=league_config, reference_update=2
    ) == pytest.approx(0.25)
    assert active_heuristic_public_variant_mix_fraction(
        league_config=league_config, reference_update=4
    ) == pytest.approx(0.1)
    assert active_mirror_mix_fraction(league_config=league_config, reference_update=3) == pytest.approx(0.45)
    assert active_mirror_mix_fraction(league_config=league_config, reference_update=6) == pytest.approx(0.3)
    assert active_noleague_baseline_mix_fraction(league_config=league_config, reference_update=2) == pytest.approx(0.3)
    assert active_noleague_baseline_mix_fraction(league_config=league_config, reference_update=3) == pytest.approx(0.0)
    assert active_warmup_snapshot_mix_fraction(
        league_config=league_config,
        reference_update=9,
        has_opponent_candidates=True,
        has_opponent_models=True,
    ) == pytest.approx(0.2)
    assert active_warmup_snapshot_mix_fraction(
        league_config=league_config,
        reference_update=10,
        has_opponent_candidates=True,
        has_opponent_models=True,
    ) == pytest.approx(0.0)


def test_active_actor_heuristic_fraction_clamps_and_respects_delayed_start() -> None:
    assert active_actor_heuristic_fraction(
        initial_fraction=1.5,
        final_fraction=-0.5,
        start_updates=4,
        end_updates=8,
        reference_update=6,
    ) == pytest.approx(0.5)
