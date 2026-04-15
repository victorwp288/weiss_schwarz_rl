from __future__ import annotations

import json
from pathlib import Path

import pytest

from weiss_rl.config import canonical_config_dict, canonical_config_json, compute_config_hash256, load_stack_config
from weiss_rl.repro import canonical_json_bytes, sha256_hex


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _temp_repo(tmp_path: Path) -> Path:
    (tmp_path / "configs").mkdir()
    (tmp_path / "python").mkdir()
    return tmp_path


def test_load_stack_config_reads_typed_thesis_preset() -> None:
    repo_root = _repo_root()
    stack = load_stack_config(repo_root / "configs/presets/typed_thesis_locked.yaml")

    assert stack.root == repo_root
    assert stack.schema_version == 2
    assert stack.components == {}
    assert stack.config.experiment is not None
    assert stack.config.experiment.role == "main"
    assert stack.config.model is not None
    assert stack.config.model.encoder_kind == "typed_v1"
    assert stack.config.training is not None
    assert stack.config.training.algorithm == "impala_vtrace_gru"
    assert stack.config.training.profile_timers is False
    assert stack.config.training.torch_profiler is False
    assert stack.config.training.compile_actor_inference is False
    assert stack.config.training.structured_metrics_mode == "off"
    assert stack.config.training.teacher_aux_mode == "always"
    assert stack.config.training.fixed_opponent_backend == "python_scalar"
    assert stack.config.rewards is not None
    assert stack.config.rewards.gamma == pytest.approx(1.0)
    assert stack.config.league is not None
    assert stack.config.league.warmup.first_updates == 200000
    assert stack.seed_sets["dev_eval"] == repo_root / "configs/seeds/dev_eval_seeds.txt"


def test_load_stack_config_applies_extends_for_typed_local() -> None:
    repo_root = _repo_root()
    stack = load_stack_config(repo_root / "configs/presets/typed_local.yaml")

    assert stack.config.training is not None
    assert stack.config.training.entropy_coef == pytest.approx(0.03)
    assert stack.config.training.checkpoint_interval_updates == 20
    assert stack.config.rewards is not None
    assert stack.config.rewards.objective == "terminal_pm1"
    assert stack.config.rewards.gamma == pytest.approx(0.99)
    assert stack.config.rewards.shaping.enable_damage_shaping is True
    assert stack.config.rewards.shaping.level_reward == pytest.approx(0.0)
    assert stack.config.rewards.shaping.board_reward == pytest.approx(0.0)
    assert stack.config.rewards.shaping.no_progress_penalty == pytest.approx(0.0)
    assert stack.config.rewards.truncation.reward == pytest.approx(-0.1)
    assert stack.config.curriculum is not None
    assert stack.config.curriculum.stall_monitor.enabled is True
    assert stack.config.curriculum.checkpoint_guard.enabled is True
    assert stack.config.curriculum.checkpoint_guard.rollback_score_margin == pytest.approx(0.15)
    assert stack.config.curriculum.checkpoint_guard.rollback_max_prob_lt_half == pytest.approx(0.7)
    assert stack.config.curriculum.checkpoint_guard.min_best_score == pytest.approx(0.55)
    assert stack.config.curriculum.checkpoint_guard.promote_min_prob_gt_half == pytest.approx(0.6)
    assert stack.config.curriculum.checkpoint_guard.promote_max_ci_half_width == pytest.approx(0.24)
    assert stack.config.curriculum.simulator == {}
    assert stack.config.league is not None
    assert stack.config.league.pool.champion_max_age_updates == 120
    assert stack.config.league.sampling.heuristic_public_start_updates == 100
    assert stack.config.league.sampling.heuristic_public_mix_fraction == pytest.approx(0.1)
    assert stack.config.league.sampling.heuristic_public_reserved_envs_per_actor == 0
    assert stack.config.league.sampling.noleague_baseline_reserved_envs_per_actor == 0
    assert stack.config.league.sampling.champion_mix_fraction == pytest.approx(0.35)
    assert stack.config.league.sampling.hard_negative_mix_fraction == pytest.approx(0.25)
    assert stack.config.league.sampling.hard_negative_min_samples == 16
    assert stack.config.league.warmup.first_updates == 200
    assert stack.config.league.promotion.seed_file == "configs/seeds/local_promotion_eval_seeds.txt"
    assert stack.config.evaluation is not None
    assert stack.config.evaluation.periodic_dev_eval_paired_seeds == 8
    assert stack.seed_sets["dev_eval"] == repo_root / "configs/seeds/local_dev_eval_seeds.txt"
    assert stack.seed_sets["promotion_gate"] == repo_root / "configs/seeds/local_promotion_eval_seeds.txt"


def test_load_stack_config_supports_baselines_and_ablations() -> None:
    repo_root = _repo_root()

    noleague = load_stack_config(repo_root / "configs/presets/baselines/noleague_impala.yaml")
    assert noleague.config.experiment is not None
    assert noleague.config.experiment.role == "baseline_noleague"
    assert noleague.config.league is not None
    assert noleague.config.league.enabled is False

    norecurrence = load_stack_config(repo_root / "configs/presets/baselines/norecurrence_impala.yaml")
    assert norecurrence.config.experiment is not None
    assert norecurrence.config.experiment.role == "baseline_norecurrence"
    assert norecurrence.config.model is not None
    assert norecurrence.config.model.recurrent_core == "none"
    assert norecurrence.config.training is not None
    assert norecurrence.config.training.algorithm == "impala_vtrace_ff"

    ppo = load_stack_config(repo_root / "configs/presets/baselines/ppo_lite.yaml")
    assert ppo.config.experiment is not None
    assert ppo.config.experiment.role == "baseline_ppo_lite"
    assert ppo.config.training is not None
    assert ppo.config.training.algorithm == "ppo_lite_masked_v1"
    assert ppo.config.training.ppo_epochs == 4

    discount = load_stack_config(repo_root / "configs/presets/ablations/discount_gamma099.yaml")
    assert discount.config.experiment is not None
    assert discount.config.experiment.role == "ablation_discount"
    assert discount.config.rewards is not None
    assert discount.config.rewards.gamma == pytest.approx(0.99)

    shaping = load_stack_config(repo_root / "configs/presets/ablations/reward_shaping.yaml")
    assert shaping.config.experiment is not None
    assert shaping.config.experiment.role == "ablation_reward"
    assert shaping.config.rewards is not None
    assert shaping.config.rewards.shaping.enable_damage_shaping is True
    assert shaping.config.rewards.shaping.level_reward == pytest.approx(0.05)
    assert shaping.config.rewards.shaping.no_progress_penalty == pytest.approx(0.005)


def test_load_stack_config_applies_antistall_v2_overrides() -> None:
    repo_root = _repo_root()
    stack = load_stack_config(repo_root / "configs/presets/typed_local_antistall_v2.yaml")

    assert stack.config.rewards is not None
    assert stack.config.rewards.shaping.damage_reward == pytest.approx(0.015)
    assert stack.config.rewards.shaping.no_progress_penalty == pytest.approx(0.015)
    assert stack.config.rewards.truncation.reward == pytest.approx(-0.6)
    assert stack.config.curriculum is not None
    assert stack.config.curriculum.simulator["max_no_progress_decisions"] == 64


def test_load_stack_config_applies_longhorizon_v1_overrides() -> None:
    repo_root = _repo_root()
    stack = load_stack_config(repo_root / "configs/presets/typed_local_longhorizon_v1.yaml")

    assert stack.config.rewards is not None
    assert stack.config.rewards.gamma == pytest.approx(1.0)
    assert stack.config.rewards.shaping.damage_reward == pytest.approx(0.0)
    assert stack.config.rewards.shaping.no_progress_penalty == pytest.approx(0.01)


def test_load_stack_config_supports_structured_v2_preset() -> None:
    repo_root = _repo_root()
    stack = load_stack_config(repo_root / "configs/presets/structured_v2.yaml")

    assert stack.config.model is not None
    assert stack.config.model.encoder_kind == "structured_v2"
    assert stack.config.model.recurrent_core == "gru"
    assert stack.config.training is not None
    assert stack.config.training.algorithm == "structured_v2"
    assert stack.seed_sets["dev_eval"] == repo_root / "configs/seeds/local_dev_eval_seeds.txt"


def test_load_stack_config_supports_typed_structured_v2_preset() -> None:
    repo_root = _repo_root()
    stack = load_stack_config(repo_root / "configs/presets/typed_structured_v2.yaml")

    assert stack.config.model is not None
    assert stack.config.model.encoder_kind == "structured_v2"
    assert stack.config.training is not None
    assert stack.config.training.algorithm == "impala_vtrace_structured_v1"
    assert stack.config.training.profile_timers is False
    assert stack.config.training.torch_profiler is False
    assert stack.config.training.compile_actor_inference is False
    assert stack.config.training.structured_metrics_mode == "sampled"
    assert stack.config.training.teacher_aux_mode == "always"
    assert stack.config.training.fixed_opponent_backend == "python_scalar"
    assert stack.config.training.structured_aux_enabled is True
    assert stack.config.training.teacher_family_coef == pytest.approx(0.20)
    assert stack.config.training.teacher_slot_coef == pytest.approx(0.10)
    assert stack.config.training.teacher_attack_type_coef == pytest.approx(0.05)
    assert stack.config.training.structured_warmstart_enabled is True
    assert stack.config.training.structured_warmstart.updates == 32
    assert stack.config.training.structured_warmstart.teacher_family_coef == pytest.approx(0.75)
    assert stack.config.training.structured_warmstart.teacher_slot_coef == pytest.approx(0.35)
    assert stack.config.training.structured_warmstart.teacher_attack_type_coef == pytest.approx(0.20)
    assert stack.seed_sets["dev_eval"] == repo_root / "configs/seeds/local_dev_eval_seeds.txt"


def test_load_stack_config_supports_structured_dev_fast_and_acceptance_presets() -> None:
    repo_root = _repo_root()

    dev_fast = load_stack_config(repo_root / "configs/presets/structured_dev_fast.yaml")
    assert dev_fast.config.league is not None
    assert dev_fast.config.league.enabled is False
    assert dev_fast.config.training is not None
    assert dev_fast.config.training.profile_timers is False
    assert dev_fast.config.training.torch_profiler is False
    assert dev_fast.config.training.compile_actor_inference is False
    assert dev_fast.config.training.structured_metrics_mode == "off"
    assert dev_fast.config.training.teacher_aux_mode == "warmstart_only"
    assert dev_fast.config.training.fixed_opponent_backend == "python_batched"
    assert dev_fast.config.training.structured_warmstart.updates == 1

    acceptance = load_stack_config(repo_root / "configs/presets/structured_acceptance.yaml")
    assert acceptance.config.training is not None
    assert acceptance.config.training.profile_timers is False
    assert acceptance.config.training.torch_profiler is False
    assert acceptance.config.training.compile_actor_inference is False
    assert acceptance.config.training.structured_metrics_mode == "sampled"
    assert acceptance.config.training.teacher_aux_mode == "always"
    assert acceptance.config.training.fixed_opponent_backend == "python_batched"
    assert acceptance.config.training.structured_warmstart.updates == 32


def test_load_stack_config_applies_anchorlane_v1_overrides() -> None:
    repo_root = _repo_root()
    stack = load_stack_config(repo_root / "configs/presets/typed_local_anchorlanes_v1.yaml")

    assert stack.config.league is not None
    assert stack.config.league.sampling.heuristic_public_reserved_envs_per_actor == 1
    assert stack.config.league.sampling.noleague_baseline_reserved_envs_per_actor == 1


def test_canonical_config_hash_is_stable() -> None:
    repo_root = _repo_root()
    stack = load_stack_config(repo_root / "configs/presets/typed_thesis_locked.yaml")

    canonical = canonical_config_json(stack)
    assert '"experiment":{"role":"main"}' in canonical
    assert compute_config_hash256(stack) == compute_config_hash256(
        load_stack_config(repo_root / "configs/presets/typed_thesis_locked.yaml")
    )


def test_load_stack_config_reads_canonical_run_artifact_json(tmp_path: Path) -> None:
    repo_root = _repo_root()
    baseline = load_stack_config(repo_root / "configs/presets/typed_local_longhorizon_v1.yaml")
    fake_repo = _temp_repo(tmp_path)
    artifact_path = fake_repo / "runs" / "example_run" / "config_canonical.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(json.dumps(canonical_config_dict(baseline)))
    payload["config"]["league"]["sampling"].pop("heuristic_public_reserved_envs_per_actor", None)
    payload["config"]["league"]["sampling"].pop("noleague_baseline_reserved_envs_per_actor", None)
    artifact_path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_stack_config(artifact_path)

    assert loaded.root == fake_repo
    assert loaded.config.rewards is not None
    assert loaded.config.rewards.gamma == pytest.approx(1.0)
    assert loaded.seed_sets["dev_eval"] == fake_repo / "configs/seeds/local_dev_eval_seeds.txt"
    assert canonical_config_dict(loaded) == payload
    assert compute_config_hash256(loaded) == sha256_hex(canonical_json_bytes(payload))


def test_config_hash_changes_when_resolved_semantics_change(tmp_path: Path) -> None:
    repo_root = _repo_root()
    fake_repo = _temp_repo(tmp_path)
    config_path = fake_repo / "configs" / "typed_local.yaml"
    thesis_path = fake_repo / "configs" / "typed_thesis_locked.yaml"
    thesis_path.write_text(
        (repo_root / "configs/presets/typed_thesis_locked.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    config_path.write_text(
        (repo_root / "configs/presets/typed_local.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    baseline = load_stack_config(config_path)

    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace("gamma: 0.99", "gamma: 0.98"),
        encoding="utf-8",
    )
    changed = load_stack_config(config_path)

    assert changed.config.rewards is not None
    assert changed.config.rewards.gamma == pytest.approx(0.98)
    assert compute_config_hash256(changed) != compute_config_hash256(baseline)


def test_load_stack_config_rejects_unknown_top_level_keys(tmp_path: Path) -> None:
    fake_repo = _temp_repo(tmp_path)
    stack_path = fake_repo / "configs" / "typed_local.yaml"
    stack_path.write_text(
        "\n".join(
            (
                "schema_version: 2",
                "description: broken",
                "experiment:",
                "  role: main",
                "mystery: true",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported keys: mystery"):
        load_stack_config(stack_path)


def test_load_stack_config_rejects_unknown_experiment_role(tmp_path: Path) -> None:
    repo_root = _repo_root()
    fake_repo = _temp_repo(tmp_path)
    thesis_path = fake_repo / "configs" / "typed_thesis_locked.yaml"
    thesis_path.write_text(
        (repo_root / "configs/presets/typed_thesis_locked.yaml")
        .read_text(encoding="utf-8")
        .replace("role: main", "role: mystery_role", 1),
        encoding="utf-8",
    )
    stack_path = fake_repo / "configs" / "typed_local.yaml"
    stack_path.write_text(
        (repo_root / "configs/presets/typed_local.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="experiment.role must be one of"):
        load_stack_config(stack_path)


def test_load_stack_config_rejects_unsupported_pfsp_stats_source(tmp_path: Path) -> None:
    repo_root = _repo_root()
    fake_repo = _temp_repo(tmp_path)
    thesis_path = fake_repo / "configs" / "typed_thesis_locked.yaml"
    thesis_path.write_text(
        (repo_root / "configs/presets/typed_thesis_locked.yaml")
        .read_text(encoding="utf-8")
        .replace("pfsp_stats_source: online_outcomes", "pfsp_stats_source: registry_snapshots", 1),
        encoding="utf-8",
    )
    stack_path = fake_repo / "configs" / "typed_local.yaml"
    stack_path.write_text(
        (repo_root / "configs/presets/typed_local.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="pfsp_stats_source currently only supports 'online_outcomes'"):
        load_stack_config(stack_path)
