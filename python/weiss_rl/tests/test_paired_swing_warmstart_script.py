from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from scripts.paired_swing_warmstart import (
    _opponent_context_indices_for_episodes,
    _sample_episode_indices,
    _source_opponent_policy_ids_by_episode,
)


def test_paired_swing_warmstart_maps_source_opponents_to_context_indices() -> None:
    dataset = SimpleNamespace(
        episode_count=3,
        metadata={
            "selected_bundles": [
                {"source_opponent_policy_id": "B2"},
                {"source_opponent_policy_id": ""},
                {"source_opponent_policy_id": "policy_000004"},
            ]
        },
    )
    model = _ContextModel({"B2": 4, "policy_000004": 9})

    indices = _opponent_context_indices_for_episodes(model, dataset, episode_indices=[2, 0, 1])

    assert indices.tolist() == [9, 4, 0]
    assert _source_opponent_policy_ids_by_episode(dataset) == ["B2", "", "policy_000004"]


def test_paired_swing_warmstart_returns_zero_context_without_model_support() -> None:
    dataset = SimpleNamespace(
        episode_count=2,
        metadata={"selected_bundles": [{"source_opponent_policy_id": "B2"}, {"source_opponent_policy_id": "B4"}]},
    )

    indices = _opponent_context_indices_for_episodes(object(), dataset, episode_indices=[0, 1])

    np.testing.assert_array_equal(indices, np.zeros((2,), dtype=np.int64))


def test_paired_swing_warmstart_samples_retention_indices_with_replacement_when_needed() -> None:
    rng = np.random.default_rng(123)

    indices = _sample_episode_indices(rng, episode_count=2, batch_episodes=5)

    assert len(indices) == 5
    assert set(indices).issubset({0, 1})


class _ContextModel:
    def __init__(self, mapping: dict[str, int]) -> None:
        self._mapping = dict(mapping)

    def opponent_context_indices_for_policy_ids(self, policy_ids: list[str]) -> list[int]:
        return [self._mapping.get(policy_id, 0) for policy_id in policy_ids]
