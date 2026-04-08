from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Literal

import pytest

from weiss_rl.eval import EvalGameRecord
from weiss_rl.metagame.payoff import (
    build_p_mean_and_counts,
    write_p_mean_csv,
    write_payoff_counts_json,
)

_CONFIG_HASH256 = "ab" * 32
_SPEC_HASH256 = "cd" * 32
OutcomeToken = Literal["W", "L", "D", "T"]


def _pair(pair_index: int, outcome_a: OutcomeToken, outcome_b: OutcomeToken) -> list[EvalGameRecord]:
    episode_seed = pair_index + 100
    return [
        _record(pair_index, 0, outcome_a, episode_seed=episode_seed),
        _record(pair_index, 1, outcome_b, episode_seed=episode_seed),
    ]


def _duplicate_seed_runs(episode_seed: int, *outcomes: tuple[OutcomeToken, OutcomeToken]) -> list[EvalGameRecord]:
    records: list[EvalGameRecord] = []
    for pair_index, (outcome_a, outcome_b) in enumerate(outcomes):
        records.extend(
            [
                _record(pair_index, 0, outcome_a, episode_seed=episode_seed),
                _record(pair_index, 1, outcome_b, episode_seed=episode_seed),
            ]
        )
    return records


def _record(
    pair_index: int,
    swap_index: int,
    outcome: OutcomeToken,
    *,
    episode_seed: int | None = None,
    focal_policy_id: str = "champion",
    opponent_policy_id: str = "baseline",
) -> EvalGameRecord:
    normalized_swap_index = int(swap_index)
    if normalized_swap_index == 0:
        seat0_policy_id = focal_policy_id
        seat1_policy_id = opponent_policy_id
        focal_seat = 0
    else:
        seat0_policy_id = opponent_policy_id
        seat1_policy_id = focal_policy_id
        focal_seat = 1

    episode_seed_value = pair_index if episode_seed is None else episode_seed
    episode_index = pair_index * 2 + normalized_swap_index
    episode_key64 = episode_index + 1
    return EvalGameRecord(
        pair_index=pair_index,
        swap_index=normalized_swap_index,
        episode_index=episode_index,
        episode_seed=episode_seed_value,
        episode_key=f"{episode_key64:064x}",
        episode_key64=episode_key64,
        config_hash256=_CONFIG_HASH256,
        spec_hash256=_SPEC_HASH256,
        focal_policy_id=focal_policy_id,
        opponent_policy_id=opponent_policy_id,
        seat0_policy_id=seat0_policy_id,
        seat1_policy_id=seat1_policy_id,
        focal_seat=focal_seat,
        outcome=outcome,
        terminated=outcome != "T",
        truncated=outcome == "T",
        engine_status=0,
    )


def test_build_p_mean_and_counts_produces_symmetric_matrix() -> None:
    records = [
        *_pair(0, "W", "W"),
        *_pair(1, "D", "T"),
    ]
    p_mean, counts, policy_ids = build_p_mean_and_counts(records, scheme="S0")

    assert tuple(policy_ids) == ("baseline", "champion") or tuple(policy_ids) == ("champion", "baseline")
    hero_index = policy_ids.index("champion")
    foe_index = policy_ids.index("baseline")

    expected = 0.75
    assert p_mean[hero_index, foe_index] == expected
    assert p_mean[foe_index, hero_index] == 1.0 - expected
    assert p_mean[hero_index, hero_index] == 0.5
    assert p_mean[foe_index, foe_index] == 0.5
    assert counts[hero_index, foe_index] == 2
    assert counts[foe_index, hero_index] == 2
    assert counts[hero_index, hero_index] == 0
    assert counts[foe_index, foe_index] == 0


def test_build_p_mean_and_counts_combines_bidirectional_evidence() -> None:
    records = [*_pair(0, "W", "W")]
    baseline_vs_champion = [
        _record(1, 0, "L", focal_policy_id="baseline", opponent_policy_id="champion"),
        _record(1, 1, "L", focal_policy_id="baseline", opponent_policy_id="champion"),
    ]
    records.extend(baseline_vs_champion)

    p_mean, counts, policy_ids = build_p_mean_and_counts(records, scheme="S0")
    champion = policy_ids.index("champion")
    baseline = policy_ids.index("baseline")

    assert p_mean[champion, baseline] == pytest.approx(1.0)
    assert p_mean[baseline, champion] == pytest.approx(0.0)
    assert counts[champion, baseline] == 2
    assert counts[baseline, champion] == 2


def test_build_p_mean_and_counts_splits_reused_pair_index_by_episode_seed() -> None:
    records = [
        *_pair(0, "W", "L"),
        _record(0, 0, "W", episode_seed=250),
        _record(0, 1, "W", episode_seed=250),
    ]

    p_mean, counts, policy_ids = build_p_mean_and_counts(records, scheme="S0")
    champion = policy_ids.index("champion")
    baseline = policy_ids.index("baseline")

    assert p_mean[champion, baseline] == pytest.approx(0.75)
    assert p_mean[baseline, champion] == pytest.approx(0.25)
    assert counts[champion, baseline] == 2
    assert counts[baseline, champion] == 2


def test_build_p_mean_and_counts_aggregates_duplicate_same_seed_runs() -> None:
    records = _duplicate_seed_runs(250, ("W", "L"), ("W", "W"))

    p_mean, counts, policy_ids = build_p_mean_and_counts(records, scheme="S0")
    champion = policy_ids.index("champion")
    baseline = policy_ids.index("baseline")

    assert p_mean[champion, baseline] == pytest.approx(0.75)
    assert p_mean[baseline, champion] == pytest.approx(0.25)
    assert counts[champion, baseline] == 1
    assert counts[baseline, champion] == 1


def test_write_payoff_artifacts(tmp_path: Path) -> None:
    records = [*_pair(0, "W", "L")]
    p_mean, counts, policy_ids = build_p_mean_and_counts(records, scheme="S0")

    p_mean_path = tmp_path / "p_mean.csv"
    counts_path = tmp_path / "payoff_counts.json"

    write_p_mean_csv(p_mean_path, p_mean, policy_ids)
    write_payoff_counts_json(counts_path, counts, policy_ids)

    with p_mean_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    assert rows[0] == ["", "baseline", "champion"]
    assert rows[1][0] == "baseline"
    assert rows[2][0] == "champion"

    with counts_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["champion"]["baseline"] == 1
    assert payload["baseline"]["champion"] == 1
