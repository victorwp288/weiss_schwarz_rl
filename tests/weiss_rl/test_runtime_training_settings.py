from __future__ import annotations

from types import SimpleNamespace

import pytest
from weiss_rl.runtime.components.training_settings import resolve_runtime_training_settings


def test_runtime_training_settings_defaults_to_model_actor_policy() -> None:
    settings = resolve_runtime_training_settings(training_config=None, actor_count=4)

    assert settings.actor_policy_backend == "model"
    assert settings.actor_heuristic_fraction == 1.0
    assert settings.actor_heuristic_start_updates == 0
    assert settings.actor_heuristic_end_updates == -1
    assert settings.actor_heuristic_final_fraction == 1.0
    assert settings.train_on_heuristic_actor_rows is True
    assert settings.diverse_opponent_actor_count == 0
    assert settings.diverse_model_actor_count == 0
    assert settings.diverse_opponent_batch_fraction == 0.0
    assert settings.diverse_opponent_batch_wait_ms == 0
    assert settings.heuristic_actor_hidden_state_tracking is True
    assert settings.trajectory_retention_enabled is False
    assert settings.trajectory_retention_policy_ids == ()
    assert settings.trajectory_retention_sources == ()
    assert settings.actor_behavior_values_required is False


def test_runtime_training_settings_normalizes_custom_training_config() -> None:
    training_config = SimpleNamespace(
        actor_policy_backend=" heuristic_public ",
        actor_heuristic_fraction=0.6,
        actor_heuristic_start_updates=2,
        actor_heuristic_end_updates=5,
        actor_heuristic_final_fraction=0.1,
        train_on_heuristic_actor_rows=False,
        diverse_opponent_actor_count=6,
        diverse_model_actor_count=4,
        diverse_opponent_batch_fraction=0.5,
        diverse_opponent_batch_wait_ms=7,
        heuristic_actor_hidden_state_tracking=False,
        trajectory_retention_coef=0.2,
        trajectory_retention_policy_ids=(" champ ", "", "seed"),
        trajectory_retention_sources=("Champions", " warmup_snapshots "),
        algorithm=" ppo_lite ",
    )

    settings = resolve_runtime_training_settings(
        training_config=training_config,
        actor_count=3,
    )

    assert settings.actor_policy_backend == "heuristic_public"
    assert settings.actor_heuristic_fraction == 0.6
    assert settings.actor_heuristic_start_updates == 2
    assert settings.actor_heuristic_end_updates == 5
    assert settings.actor_heuristic_final_fraction == 0.1
    assert settings.train_on_heuristic_actor_rows is False
    assert settings.diverse_opponent_actor_count == 3
    assert settings.diverse_model_actor_count == 3
    assert settings.diverse_opponent_batch_fraction == 0.5
    assert settings.diverse_opponent_batch_wait_ms == 7
    assert settings.heuristic_actor_hidden_state_tracking is False
    assert settings.trajectory_retention_enabled is True
    assert settings.trajectory_retention_policy_ids == ("champ", "seed")
    assert settings.trajectory_retention_sources == ("champions", "warmup_snapshots")
    assert settings.actor_behavior_values_required is True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("actor_policy_backend", "legacy", "training.actor_policy_backend must be one of"),
        ("actor_heuristic_fraction", 1.1, "training.actor_heuristic_fraction must be between"),
        ("actor_heuristic_start_updates", -1, "training.actor_heuristic_start_updates must be >= 0"),
        ("actor_heuristic_end_updates", -2, "training.actor_heuristic_end_updates must be >= -1"),
        ("actor_heuristic_final_fraction", -0.1, "training.actor_heuristic_final_fraction must be between"),
        ("diverse_opponent_actor_count", -1, "training.diverse_opponent_actor_count must be >= 0"),
        ("diverse_model_actor_count", -1, "training.diverse_model_actor_count must be >= 0"),
        ("diverse_opponent_batch_fraction", 1.1, "training.diverse_opponent_batch_fraction must be between"),
        ("diverse_opponent_batch_wait_ms", -1, "training.diverse_opponent_batch_wait_ms must be >= 0"),
    ],
)
def test_runtime_training_settings_rejects_invalid_values(
    field: str,
    value: object,
    message: str,
) -> None:
    training_config = SimpleNamespace(**{field: value})

    with pytest.raises(ValueError, match=message):
        resolve_runtime_training_settings(training_config=training_config, actor_count=4)


def test_runtime_training_settings_rejects_inverted_heuristic_schedule() -> None:
    training_config = SimpleNamespace(
        actor_heuristic_start_updates=5,
        actor_heuristic_end_updates=4,
    )

    with pytest.raises(
        ValueError,
        match="training.actor_heuristic_end_updates must be >= training.actor_heuristic_start_updates",
    ):
        resolve_runtime_training_settings(training_config=training_config, actor_count=4)
