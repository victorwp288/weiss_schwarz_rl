from __future__ import annotations

import pytest
from weiss_rl.config.sections.sections_core import parse_experiment_config, parse_system_config


def _system_body() -> dict[str, object]:
    return {
        "profile": {
            "training": "full",
            "local_iteration": "local",
            "ci_invariant_testing": "ci",
        },
        "mp_start_method": "spawn",
        "learner_device": "cpu",
        "actor_device": "cpu",
        "actor_process_count": 1,
        "envs_per_actor": 2,
        "total_envs": 2,
        "actor_torch_threads": 1,
        "learner_torch_threads": 2,
        "actor_queue_capacity_unrolls": 3,
        "learner_prefetch_batches": 1,
    }


def test_parse_experiment_config_accepts_known_roles() -> None:
    config = parse_experiment_config({"role": "guided_league_bootstrap"})

    assert config.role == "guided_league_bootstrap"


def test_parse_experiment_config_rejects_unknown_keys_and_roles() -> None:
    with pytest.raises(ValueError, match="experiment has unsupported keys: extra"):
        parse_experiment_config({"role": "main", "extra": True})

    with pytest.raises(
        ValueError,
        match="experiment.role must be one of: ablation_discount, ablation_guided, ablation_reward, baseline_noleague, "
        "baseline_norecurrence, baseline_ppo_lite, guided_league_bootstrap, guided_league_seed, main",
    ):
        parse_experiment_config({"role": "new_algorithm"})


def test_parse_system_config_defaults_collection_backend() -> None:
    config = parse_system_config(_system_body())

    assert config.collection_backend == "auto"
    assert config.profile.training == "full"
    assert config.profile.local_iteration == "local"
    assert config.profile.ci_invariant_testing == "ci"
    assert config.actor_process_count == 1
    assert config.learner_prefetch_batches == 1


def test_parse_system_config_accepts_explicit_collection_backend() -> None:
    body = _system_body()
    body["collection_backend"] = "central"

    assert parse_system_config(body).collection_backend == "central"


def test_parse_system_config_preserves_validation_messages() -> None:
    bad_backend = _system_body()
    bad_backend["collection_backend"] = "other"
    with pytest.raises(ValueError, match="system.collection_backend must be one of: auto, central, process"):
        parse_system_config(bad_backend)

    bad_profile = _system_body()
    bad_profile["profile"] = {"training": "full", "local_iteration": "local", "extra": "x"}
    with pytest.raises(ValueError, match="system.profile has unsupported keys: extra"):
        parse_system_config(bad_profile)

    bad_minimum = _system_body()
    bad_minimum["learner_prefetch_batches"] = 0
    with pytest.raises(ValueError, match="system.learner_prefetch_batches must be >= 1, got 0"):
        parse_system_config(bad_minimum)
