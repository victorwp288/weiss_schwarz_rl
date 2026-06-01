from __future__ import annotations

import json
from pathlib import Path

import pytest

from weiss_rl.artifacts.reproducibility import canonical_json_bytes, parse_seed_file, sha256_hex
from weiss_rl.config import (
    TrainingConfig,
    canonical_config_dict,
    canonical_config_json,
    compute_config_hash256,
    load_stack_config,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _temp_repo(tmp_path: Path) -> Path:
    (tmp_path / "configs").mkdir()
    (tmp_path / "python").mkdir()
    return tmp_path


def _require_training_config(training: TrainingConfig | None) -> TrainingConfig:
    assert training is not None
    return training


def _a075_context_config(repo_root: Path, filename: str) -> Path:
    return repo_root / "configs" / "thesis" / "_shared" / "hardneg_a075_context" / filename


def _a050_context_width128_config(repo_root: Path, filename: str) -> Path:
    return repo_root / "configs" / "thesis" / "_shared" / "hardneg_a050_context_width128" / filename


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
    assert stack.config.model.candidate_scoring_chunk_size == 65536
    assert stack.config.model.cuda_learner_candidate_scoring_chunk_size == 262144
    assert stack.config.model.public_heuristic_logit_bias_scale == pytest.approx(0.0)
    assert stack.config.model.public_heuristic_actor_logit_bias_scale == pytest.approx(-1.0)
    assert stack.config.training is not None
    assert stack.config.training.algorithm == "impala_vtrace_gru"
    assert stack.config.training.profile_timers is False
    assert stack.config.training.torch_profiler is False
    assert stack.config.training.compile_actor_inference is False
    assert stack.config.training.structured_metrics_mode == "off"
    assert stack.config.training.teacher_aux_mode == "always"
    assert stack.config.training.fixed_opponent_backend == "python_scalar"
    assert stack.config.training.actor_policy_backend == "model"
    assert stack.config.training.actor_heuristic_fraction == pytest.approx(1.0)
    assert stack.config.training.heuristic_actor_hidden_state_tracking is True
    assert stack.config.system is not None
    assert stack.config.system.collection_backend == "auto"
    assert stack.config.rewards is not None
    assert stack.config.rewards.gamma == pytest.approx(1.0)
    assert stack.config.league is not None
    assert stack.config.league.warmup.first_updates == 200000
    assert stack.seed_sets["dev_eval"] == repo_root / "configs/seeds/dev_eval_seeds.txt"


def test_canonical_config_dict_normalizes_yaml_sequences_to_json_lists() -> None:
    repo_root = _repo_root()
    stack = load_stack_config(repo_root / "configs/presets/typed_thesis_locked.yaml")

    payload = canonical_config_dict(stack)

    assert json.loads(json.dumps(payload)) == payload
    assert isinstance(payload["config"]["model"]["public_heuristic_logit_bias_families"], list)


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
    assert stack.config.league.sampling.noleague_baseline_mix_fraction == pytest.approx(0.0)
    assert stack.config.league.sampling.noleague_baseline_mix_end_updates == -1
    assert stack.config.league.sampling.warmup_snapshot_mix_fraction == pytest.approx(0.0)
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

    no_gru = load_stack_config(repo_root / "configs/thesis/ablations/no_gru.yaml")
    assert no_gru.config.experiment is not None
    assert no_gru.config.experiment.role == "baseline_norecurrence"
    assert no_gru.config.model is not None
    assert no_gru.config.model.recurrent_core == "none"
    assert no_gru.config.training is not None
    assert no_gru.config.training.algorithm == "impala_vtrace_ff"

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


def test_load_stack_config_supports_guided_b1_ablation() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root / "configs/thesis/_shared/guided_teacher/public_teacher_tactical_mulliganguard_reward.yaml"
    )

    assert stack.config.experiment is not None
    assert stack.config.experiment.role == "ablation_guided"
    assert stack.config.model is not None
    assert stack.config.model.public_heuristic_logit_bias_scale == pytest.approx(0.0)
    assert stack.config.model.public_heuristic_actor_logit_bias_scale == pytest.approx(0.0)
    assert stack.config.training is not None
    assert stack.config.training.actor_policy_backend == "model"
    assert stack.config.training.actor_heuristic_fraction == pytest.approx(0.0)
    assert stack.config.training.teacher_aux_mode == "always"
    assert stack.config.training.structured_aux_enabled is True
    assert stack.config.training.structured_warmstart_enabled is False
    assert stack.config.training.teacher_family_coef == pytest.approx(0.02)
    assert stack.config.training.teacher_same_family_action_coef == pytest.approx(0.03)
    assert stack.config.training.teacher_public_heuristic_coef == pytest.approx(0.05)
    assert stack.config.training.teacher_public_heuristic_end_updates == 75
    assert stack.config.training.teacher_public_heuristic_final_coef == pytest.approx(0.0)
    assert stack.config.training.teacher_public_heuristic_temperature == pytest.approx(8.0)
    assert stack.config.training.teacher_public_heuristic_families == (
        "main_play_character",
        "attack",
        "main_move",
        "clock_from_hand",
    )
    assert stack.config.training.teacher_public_heuristic_profiles == ("base", "aggressive", "control")
    assert stack.config.training.teacher_public_heuristic_profile_mode == "cycle"
    assert stack.config.training.mulligan_force_confirm_after_select is True
    assert stack.config.league is not None
    assert stack.config.league.enabled is False
    assert stack.config.environment is not None
    assert stack.config.environment.deck_pool == ("preset:main_deck_5hy_yotsuba_v1",)
    assert stack.config.environment.opponent_deck_pool == ("preset:main_deck_5hy_yotsuba_v1",)


def test_load_stack_config_supports_passaware_guided_b1_ablation() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root / "configs/thesis/_shared/guided_teacher/public_teacher_passaware_mulliganguard_reward.yaml"
    )

    assert stack.config.experiment is not None
    assert stack.config.experiment.role == "ablation_guided"
    assert stack.config.training is not None
    assert stack.config.training.teacher_public_heuristic_end_updates == 150
    assert stack.config.training.teacher_public_heuristic_final_coef == pytest.approx(0.01)
    assert stack.config.training.teacher_public_heuristic_families == (
        "pass",
        "main_play_character",
        "attack",
        "main_move",
        "clock_from_hand",
    )
    assert stack.config.training.mulligan_force_confirm_after_select is True
    assert stack.config.training.actor_heuristic_fraction == pytest.approx(0.0)
    assert stack.config.league is not None
    assert stack.config.league.enabled is False


def test_load_stack_config_supports_main_b1only_p2_trust_region_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root / "configs/thesis/_shared/main_b1only_p2/main_b1only_p2_trust_region_probe.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.optimizer.learning_rate == pytest.approx(0.00005)
    assert stack.config.training.entropy_coef == pytest.approx(0.003)
    assert stack.config.training.fixed_model_opponent_action_selection == "sample"
    assert stack.config.training.policy_anchor_coef == pytest.approx(0.15)
    assert stack.config.training.policy_anchor_top_action_coef == pytest.approx(0.0)
    assert stack.config.training.policy_anchor_temperature == pytest.approx(0.75)
    assert stack.config.training.trajectory_bc_enabled is False
    assert stack.config.league is not None
    assert stack.config.league.sampling.noleague_baseline_mix_fraction == pytest.approx(1.0)
    assert stack.config.league.sampling.noleague_baseline_reserved_envs_per_actor == 8
    assert stack.config.league.sampling.heuristic_public_final_mix_fraction == pytest.approx(0.0)
    assert stack.config.league.sampling.champion_mix_fraction == pytest.approx(0.0)


def test_load_stack_config_supports_main_b1only_p2_trust_region_no_warmup_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root / "configs/thesis/_shared/main_b1only_p2/main_b1only_p2_trust_region_no_warmup_probe.yaml"
    )

    assert stack.config.league is not None
    assert stack.config.league.sampling.noleague_baseline_mix_fraction == pytest.approx(1.0)
    assert stack.config.league.sampling.warmup_snapshot_mix_fraction == pytest.approx(0.0)
    assert stack.config.training is not None
    assert stack.config.training.policy_anchor_coef == pytest.approx(0.15)


def test_load_stack_config_supports_main_b1only_p2_trust_region_argmax_opp_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root / "configs/thesis/_shared/main_b1only_p2/main_b1only_p2_trust_region_argmax_opp_probe.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.fixed_model_opponent_action_selection == "argmax"
    assert stack.config.training.policy_anchor_coef == pytest.approx(0.15)
    assert stack.config.league is not None
    assert stack.config.league.sampling.noleague_baseline_mix_fraction == pytest.approx(1.0)


def test_load_stack_config_supports_main_b1only_p2_free_argmax_opp_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root / "configs/thesis/_shared/main_b1only_p2/main_b1only_p2_free_argmax_opp_probe.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.fixed_model_opponent_action_selection == "argmax"
    assert stack.config.training.policy_anchor_coef == pytest.approx(0.0)
    assert stack.config.training.policy_anchor_top_action_coef == pytest.approx(0.0)
    assert stack.config.league is not None
    assert stack.config.league.sampling.noleague_baseline_mix_fraction == pytest.approx(1.0)


def test_load_stack_config_supports_main_league_champion_hardneg_long_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root / "configs/thesis/_shared/hardneg_core/main_league_champion_hardneg_long_probe.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.optimizer.learning_rate == pytest.approx(0.00004)
    assert stack.config.training.fixed_model_opponent_action_selection == "argmax"
    assert stack.config.training.policy_anchor_coef == pytest.approx(0.12)
    assert stack.config.training.policy_anchor_top_action_coef == pytest.approx(0.01)
    assert stack.config.training.trajectory_bc_enabled is False
    assert stack.config.league is not None
    assert stack.config.league.pool.seed_snapshot_import_filter == "all"
    assert stack.config.league.pool.seed_snapshot_champion_import == "all"
    assert stack.config.league.sampling.noleague_baseline_mix_fraction == pytest.approx(0.30)
    assert stack.config.league.sampling.noleague_baseline_reserved_envs_per_actor == 4
    assert stack.config.league.sampling.champion_mix_fraction == pytest.approx(0.25)
    assert stack.config.league.sampling.hard_negative_mix_fraction == pytest.approx(0.15)
    assert stack.config.league.sampling.hard_negative_max_win_rate == pytest.approx(0.50)


def test_load_stack_config_supports_main_league_champion_hardneg_rehearsal_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root / "configs/thesis/_shared/hardneg_core/main_league_champion_hardneg_rehearsal_probe.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.trajectory_bc_enabled is True
    assert stack.config.training.policy_anchor_coef == pytest.approx(0.20)
    assert stack.config.training.policy_anchor_top_action_coef == pytest.approx(0.02)
    assert stack.config.training.trajectory_bc_dataset_path.endswith("trajectory_bc_direct_b2_b3_b4_win_64.npz")
    assert stack.config.training.structured_aux.trajectory_bc_every_updates == 1
    assert stack.config.training.structured_aux.trajectory_bc_teacher_action_coef == pytest.approx(0.18)
    assert stack.config.training.structured_aux.trajectory_bc_teacher_same_family_action_coef == pytest.approx(0.55)
    assert stack.config.league is not None
    assert stack.config.league.pool.seed_snapshot_import_filter == "all"
    assert stack.config.league.pool.seed_snapshot_champion_import == "all"
    assert stack.config.league.sampling.noleague_baseline_mix_fraction == pytest.approx(0.20)
    assert stack.config.league.sampling.noleague_baseline_reserved_envs_per_actor == 2
    assert stack.config.league.sampling.heuristic_public_mix_fraction == pytest.approx(0.08)
    assert stack.config.league.sampling.heuristic_public_variant_mix_fraction == pytest.approx(0.20)
    assert stack.config.league.sampling.champion_mix_fraction == pytest.approx(0.20)
    assert stack.config.league.sampling.hard_negative_mix_fraction == pytest.approx(0.12)


def test_load_stack_config_supports_main_league_champion_hardneg_consolidation_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root / "configs/thesis/_shared/hardneg_core/main_league_champion_hardneg_consolidation_probe.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.trajectory_bc_enabled is True
    assert stack.config.training.policy_anchor_coef == pytest.approx(0.16)
    assert stack.config.training.structured_aux.trajectory_bc_every_updates == 2
    assert stack.config.training.structured_aux.trajectory_bc_teacher_action_coef == pytest.approx(0.14)
    assert stack.config.training.structured_aux.trajectory_bc_teacher_same_family_action_coef == pytest.approx(0.45)
    assert stack.config.league is not None
    assert stack.config.league.pool.champion_size == 6
    assert stack.config.league.sampling.noleague_baseline_mix_fraction == pytest.approx(0.28)
    assert stack.config.league.sampling.noleague_baseline_reserved_envs_per_actor == 4
    assert stack.config.league.sampling.heuristic_public_mix_fraction == pytest.approx(0.06)
    assert stack.config.league.sampling.heuristic_public_variant_mix_fraction == pytest.approx(0.14)
    assert stack.config.league.sampling.champion_mix_fraction == pytest.approx(0.20)
    assert stack.config.league.sampling.hard_negative_mix_fraction == pytest.approx(0.16)
    assert stack.config.league.sampling.hard_negative_min_samples == 2
    assert stack.config.league.sampling.hard_negative_max_win_rate == pytest.approx(0.55)


def test_load_stack_config_supports_main_league_champion_hardneg_stable_long_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root / "configs/thesis/_shared/hardneg_core/main_league_champion_hardneg_stable_long_probe.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.optimizer.learning_rate == pytest.approx(0.00003)
    assert stack.config.training.trajectory_bc_enabled is True
    assert stack.config.training.policy_anchor_coef == pytest.approx(0.18)
    assert stack.config.training.structured_aux.trajectory_bc_every_updates == 1
    assert stack.config.training.structured_aux.trajectory_bc_teacher_action_coef == pytest.approx(0.10)
    assert stack.config.training.structured_aux.trajectory_bc_teacher_same_family_action_coef == pytest.approx(0.35)
    assert stack.config.league is not None
    assert stack.config.league.pool.champion_size == 6
    assert stack.config.league.sampling.noleague_baseline_mix_fraction == pytest.approx(0.32)
    assert stack.config.league.sampling.heuristic_public_mix_fraction == pytest.approx(0.06)
    assert stack.config.league.sampling.heuristic_public_variant_mix_fraction == pytest.approx(0.14)
    assert stack.config.league.sampling.champion_mix_fraction == pytest.approx(0.18)
    assert stack.config.league.sampling.hard_negative_mix_fraction == pytest.approx(0.14)
    assert stack.config.league.sampling.hard_negative_min_samples == 2
    assert stack.config.league.sampling.hard_negative_max_win_rate == pytest.approx(0.55)


def test_load_stack_config_supports_main_league_champion_hardneg_polish_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root / "configs/thesis/_shared/hardneg_core/main_league_champion_hardneg_polish_probe.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.optimizer.learning_rate == pytest.approx(0.00001)
    assert stack.config.training.trajectory_bc_enabled is False
    assert stack.config.training.policy_anchor_coef == pytest.approx(0.25)
    assert stack.config.training.policy_anchor_top_action_coef == pytest.approx(0.03)
    assert stack.config.league is not None
    assert stack.config.league.pool.champion_size == 6
    assert stack.config.league.sampling.noleague_baseline_mix_fraction == pytest.approx(0.25)
    assert stack.config.league.sampling.heuristic_public_mix_fraction == pytest.approx(0.04)
    assert stack.config.league.sampling.heuristic_public_variant_mix_fraction == pytest.approx(0.08)
    assert stack.config.league.sampling.champion_mix_fraction == pytest.approx(0.35)
    assert stack.config.league.sampling.hard_negative_mix_fraction == pytest.approx(0.20)
    assert stack.config.league.sampling.hard_negative_min_samples == 2
    assert stack.config.league.sampling.hard_negative_max_win_rate == pytest.approx(0.55)


def test_load_stack_config_supports_main_league_champion_hardneg_multiobjective_guard_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root / "configs/thesis/_shared/hardneg_core/main_league_champion_hardneg_multiobjective_guard_probe.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.optimizer.learning_rate == pytest.approx(0.00002)
    assert stack.config.training.policy_anchor_coef == pytest.approx(0.22)
    assert stack.config.training.policy_anchor_top_action_coef == pytest.approx(0.03)
    assert stack.config.training.trajectory_bc_enabled is True
    assert stack.config.training.structured_aux.trajectory_bc_teacher_same_family_action_coef == pytest.approx(0.40)
    assert stack.config.league is not None
    assert stack.config.league.pool.champion_size == 8
    assert stack.config.league.pool.seed_snapshot_import_filter == "all"
    assert stack.config.league.pool.seed_snapshot_champion_import == "source_champions"
    assert stack.config.league.warmup.first_updates == 1
    assert stack.config.league.sampling.noleague_baseline_mix_fraction == pytest.approx(0.30)
    assert stack.config.league.sampling.champion_mix_fraction == pytest.approx(0.24)
    assert stack.config.league.sampling.hard_negative_mix_fraction == pytest.approx(0.18)
    assert stack.config.league.sampling.hard_negative_max_win_rate == pytest.approx(0.60)


def test_load_stack_config_supports_main_league_champion_hardneg_multiobjective_retention_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/hardneg_retention/main_league_champion_hardneg_multiobjective_retention_probe.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.optimizer.learning_rate == pytest.approx(0.00002)
    assert stack.config.training.policy_anchor_coef == pytest.approx(0.30)
    assert stack.config.training.policy_anchor_top_action_coef == pytest.approx(0.05)
    assert stack.config.training.trajectory_retention_coef == pytest.approx(0.08)
    assert stack.config.training.trajectory_retention_sources == (
        "champions",
        "hard_negatives",
        "warmup_snapshots",
    )
    assert stack.config.training.trajectory_bc_enabled is False
    assert stack.config.league is not None
    assert stack.config.league.sampling.champion_mix_fraction == pytest.approx(0.28)
    assert stack.config.league.sampling.hard_negative_mix_fraction == pytest.approx(0.20)


def test_load_stack_config_supports_main_league_champion_hardneg_replaybc_retention_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/hardneg_retention/main_league_champion_hardneg_replaybc_retention_probe.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.trajectory_retention_coef == pytest.approx(0.08)
    assert stack.config.training.trajectory_bc_enabled is True
    assert stack.config.training.trajectory_bc_dataset_path.endswith(
        "trajectory_bc_main_a015_vs_imported_champions_win8.npz"
    )
    assert stack.config.league is not None
    assert stack.config.league.sampling.hard_negative_min_samples == 2
    assert stack.config.league.sampling.hard_negative_max_win_rate == pytest.approx(0.60)


def test_load_stack_config_supports_main_league_champion_hardneg_balanced_replaybc_retention_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/hardneg_retention/main_league_champion_hardneg_balanced_replaybc_retention_probe.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.trajectory_retention_coef == pytest.approx(0.08)
    assert stack.config.training.trajectory_bc_enabled is True
    assert stack.config.training.trajectory_bc_dataset_path.endswith(
        "trajectory_bc_balanced_fixed_b2b3b4_and_learned_a015_win8.npz"
    )
    assert stack.config.league is not None
    assert stack.config.league.sampling.champion_mix_fraction == pytest.approx(0.28)
    assert stack.config.league.sampling.hard_negative_mix_fraction == pytest.approx(0.20)


def test_load_stack_config_supports_main_league_champion_hardneg_weighted_replaybc_win32_retention_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/hardneg_retention/main_league_champion_hardneg_weighted_replaybc_win32_retention_probe.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.trajectory_retention_coef == pytest.approx(0.08)
    assert stack.config.training.trajectory_bc_enabled is True
    assert stack.config.training.trajectory_bc_dataset_path.endswith(
        "trajectory_bc_weighted_fixed2x_b2b3b4_and_learned_a015_win32.npz"
    )
    assert stack.config.league is not None
    assert stack.config.league.sampling.champion_mix_fraction == pytest.approx(0.28)
    assert stack.config.league.sampling.hard_negative_mix_fraction == pytest.approx(0.20)


def test_load_stack_config_supports_main_league_champion_hardneg_selected_retention_b4guard_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/hardneg_selected/main_league_champion_hardneg_selected_retention_b4guard_probe.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.trajectory_retention_coef == pytest.approx(0.08)
    assert stack.config.training.trajectory_bc_enabled is False
    assert stack.config.training.policy_anchor_coef == pytest.approx(0.30)
    assert stack.config.league is not None
    assert stack.config.league.sampling.noleague_baseline_mix_fraction == pytest.approx(0.32)
    assert stack.config.league.sampling.heuristic_public_mix_fraction == pytest.approx(0.06)
    assert stack.config.league.sampling.heuristic_public_variant_mix_fraction == pytest.approx(0.16)
    assert stack.config.league.sampling.champion_mix_fraction == pytest.approx(0.24)
    assert stack.config.league.sampling.hard_negative_mix_fraction == pytest.approx(0.18)
    assert stack.config.league.sampling.hard_negative_min_samples == 2
    assert stack.config.league.sampling.hard_negative_max_win_rate == pytest.approx(0.60)


def test_load_stack_config_supports_main_league_champion_hardneg_selected_alloutcome_replaybc_b4guard_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/hardneg_selected/main_league_champion_hardneg_selected_alloutcome_replaybc_b4guard_probe.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.trajectory_retention_coef == pytest.approx(0.0)
    assert stack.config.training.trajectory_bc_enabled is True
    assert stack.config.training.trajectory_bc_dataset_path.endswith(
        "trajectory_bc_selected_a015_vs_imported_champions_all32.npz"
    )
    assert stack.config.training.policy_anchor_coef == pytest.approx(0.45)
    assert stack.config.training.policy_anchor_top_action_coef == pytest.approx(0.10)
    assert stack.config.league is not None
    assert stack.config.league.sampling.champion_mix_fraction == pytest.approx(0.24)
    assert stack.config.league.sampling.hard_negative_mix_fraction == pytest.approx(0.18)


def test_load_stack_config_supports_main_league_champion_hardneg_selected_alloutcome_b2repair_b4guard_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/hardneg_selected_alloutcome/main_league_champion_hardneg_selected_alloutcome_b2repair_b4guard_probe.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.optimizer.learning_rate == pytest.approx(0.0000075)
    assert stack.config.training.trajectory_retention_coef == pytest.approx(0.0)
    assert stack.config.training.trajectory_bc_enabled is True
    assert stack.config.training.trajectory_bc_dataset_path.endswith(
        "trajectory_bc_selected_a015_vs_imported_champions_all32.npz"
    )
    assert stack.config.training.teacher_public_heuristic_coef == pytest.approx(0.025)
    assert stack.config.training.teacher_public_heuristic_profiles == ("base",)
    assert stack.config.training.policy_anchor_coef == pytest.approx(0.50)
    assert stack.config.training.policy_anchor_top_action_coef == pytest.approx(0.12)
    assert stack.config.training.structured_aux.trajectory_bc_batch_episodes == 10
    assert stack.config.league is not None
    assert stack.config.league.sampling.heuristic_public_mix_fraction == pytest.approx(0.08)
    assert stack.config.league.sampling.heuristic_public_variant_mix_fraction == pytest.approx(0.14)
    assert stack.config.league.sampling.champion_mix_fraction == pytest.approx(0.24)
    assert stack.config.league.sampling.hard_negative_mix_fraction == pytest.approx(0.20)


def test_load_stack_config_supports_main_league_champion_hardneg_selected_alloutcome_learnedfloor_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / (
            "configs/thesis/_shared/hardneg_selected_alloutcome/"
            "main_league_champion_hardneg_selected_alloutcome_learnedfloor_b4b2guard_probe.yaml"
        )
    )

    assert stack.config.training is not None
    assert stack.config.training.optimizer.learning_rate == pytest.approx(0.000006)
    assert stack.config.training.entropy_coef == pytest.approx(0.0010)
    assert stack.config.training.actor_sampling_temperature == pytest.approx(0.84)
    assert stack.config.training.teacher_public_heuristic_coef == pytest.approx(0.018)
    assert stack.config.training.teacher_public_heuristic_final_coef == pytest.approx(0.018)
    assert stack.config.training.policy_anchor_coef == pytest.approx(0.55)
    assert stack.config.training.policy_anchor_top_action_coef == pytest.approx(0.14)
    assert stack.config.training.structured_aux.trajectory_bc_batch_episodes == 14
    assert stack.config.training.structured_aux.trajectory_bc_teacher_action_coef == pytest.approx(0.18)
    assert stack.config.training.structured_aux.trajectory_bc_teacher_same_family_action_coef == pytest.approx(0.68)
    assert stack.config.training.structured_aux.trajectory_bc_teacher_same_family_action_margin_coef == pytest.approx(
        0.10
    )
    assert stack.config.league is not None
    assert stack.config.league.sampling.noleague_baseline_mix_fraction == pytest.approx(0.30)
    assert stack.config.league.sampling.heuristic_public_mix_fraction == pytest.approx(0.06)
    assert stack.config.league.sampling.heuristic_public_variant_mix_fraction == pytest.approx(0.12)
    assert stack.config.league.sampling.champion_mix_fraction == pytest.approx(0.26)
    assert stack.config.league.sampling.hard_negative_mix_fraction == pytest.approx(0.22)


def test_load_stack_config_supports_main_league_champion_hardneg_selected_alloutcome_learnedpush_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / (
            "configs/thesis/_shared/hardneg_selected_alloutcome/"
            "main_league_champion_hardneg_selected_alloutcome_learnedpush_b4b2guard_probe.yaml"
        )
    )

    assert stack.config.training is not None
    assert stack.config.training.optimizer.learning_rate == pytest.approx(0.0000065)
    assert stack.config.training.entropy_coef == pytest.approx(0.0014)
    assert stack.config.training.actor_sampling_temperature == pytest.approx(0.88)
    assert stack.config.training.teacher_public_heuristic_coef == pytest.approx(0.012)
    assert stack.config.training.teacher_public_heuristic_final_coef == pytest.approx(0.012)
    assert stack.config.training.policy_anchor_coef == pytest.approx(0.42)
    assert stack.config.training.policy_anchor_top_action_coef == pytest.approx(0.08)
    assert stack.config.training.policy_anchor_temperature == pytest.approx(0.70)
    assert stack.config.training.structured_aux.trajectory_bc_batch_episodes == 8
    assert stack.config.training.structured_aux.trajectory_bc_teacher_action_coef == pytest.approx(0.12)
    assert stack.config.training.structured_aux.trajectory_bc_teacher_same_family_action_coef == pytest.approx(0.52)
    assert stack.config.training.structured_aux.trajectory_bc_teacher_same_family_action_margin_coef == pytest.approx(
        0.06
    )
    assert stack.config.league is not None
    assert stack.config.league.sampling.noleague_baseline_mix_fraction == pytest.approx(0.26)
    assert stack.config.league.sampling.heuristic_public_mix_fraction == pytest.approx(0.06)
    assert stack.config.league.sampling.heuristic_public_variant_mix_fraction == pytest.approx(0.12)
    assert stack.config.league.sampling.champion_mix_fraction == pytest.approx(0.27)
    assert stack.config.league.sampling.hard_negative_mix_fraction == pytest.approx(0.22)
    assert stack.config.league.sampling.hard_negative_max_win_rate == pytest.approx(0.66)


def test_load_stack_config_supports_main_league_champion_hardneg_selected_alloutcome_swingrepair_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / (
            "configs/thesis/_shared/hardneg_selected_alloutcome/"
            "main_league_champion_hardneg_selected_alloutcome_swingrepair_b4b2guard_probe.yaml"
        )
    )

    assert stack.config.training is not None
    assert stack.config.training.optimizer.learning_rate == pytest.approx(0.000006)
    assert stack.config.training.trajectory_bc_dataset_path.endswith(
        "trajectory_bc_selected_a015_all32_plus_swing4x.npz"
    )
    assert stack.config.training.policy_anchor_coef == pytest.approx(0.44)
    assert stack.config.training.policy_anchor_top_action_coef == pytest.approx(0.09)
    assert stack.config.training.policy_anchor_temperature == pytest.approx(0.68)
    assert stack.config.training.structured_aux.trajectory_bc_batch_episodes == 10
    assert stack.config.training.structured_aux.trajectory_bc_teacher_action_coef == pytest.approx(0.13)
    assert stack.config.training.structured_aux.trajectory_bc_teacher_same_family_action_coef == pytest.approx(0.56)
    assert stack.config.training.structured_aux.trajectory_bc_teacher_same_family_action_margin_coef == pytest.approx(
        0.07
    )
    assert stack.config.league is not None
    assert stack.config.league.sampling.champion_mix_fraction == pytest.approx(0.27)
    assert stack.config.league.sampling.hard_negative_mix_fraction == pytest.approx(0.22)


def test_load_stack_config_supports_main_league_champion_hardneg_selected_alloutcome_disjointrepair_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / (
            "configs/thesis/_shared/hardneg_selected_alloutcome/"
            "main_league_champion_hardneg_selected_alloutcome_disjointrepair_b4b2guard_probe.yaml"
        )
    )

    assert stack.config.training is not None
    assert stack.config.training.trajectory_bc_dataset_path.endswith(
        "trajectory_bc_selected_a015_all32_plus_disjointrepair4x.npz"
    )
    repair_seed_file = stack.seed_sets["hardneg_repair_train"]
    assert repair_seed_file == repo_root / "configs/seeds/hardneg_repair_train_seeds_20260518.txt"
    repair_seeds = set(parse_seed_file(repair_seed_file))
    assert len(repair_seeds) == 32
    eval_seed_files = (
        repo_root / "configs/seeds/dev_eval_seeds.txt",
        repo_root / "configs/seeds/promotion_eval_seeds.txt",
        repo_root / "configs/seeds/report_eval_seeds.txt",
    )
    eval_seeds = set().union(*(set(parse_seed_file(path)) for path in eval_seed_files))
    assert repair_seeds.isdisjoint(eval_seeds)


def test_load_stack_config_supports_main_league_champion_hardneg_selected_alloutcome_focusoldhn_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / (
            "configs/thesis/_shared/hardneg_selected_alloutcome/"
            "main_league_champion_hardneg_selected_alloutcome_focusoldhn_b4b2guard_probe.yaml"
        )
    )

    assert stack.config.training is not None
    assert stack.config.training.optimizer.learning_rate == pytest.approx(0.0000065)
    assert stack.config.league is not None
    assert stack.config.league.sampling.champion_mix_fraction == pytest.approx(0.27)
    assert stack.config.league.sampling.hard_negative_mix_fraction == pytest.approx(0.22)
    assert stack.config.league.sampling.hard_negative_focus_policy_ids == (
        "seed_c3aac2f9dc_checkpoint_000025",
        "seed_c3aac2f9dc_main_bestresponse_u25_devbest",
        "seed_c3aac2f9dc_policy_000002",
    )
    assert stack.config.league.sampling.hard_negative_focus_weight_multiplier == pytest.approx(4.0)


def test_load_stack_config_supports_main_league_champion_hardneg_selected_alloutcome_focusoldhn_strong_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / (
            "configs/thesis/_shared/hardneg_selected_alloutcome/"
            "main_league_champion_hardneg_selected_alloutcome_focusoldhn_strong_b4b2guard_probe.yaml"
        )
    )

    assert stack.config.training is not None
    assert stack.config.training.optimizer.learning_rate == pytest.approx(0.0000055)
    assert stack.config.league is not None
    assert stack.config.league.sampling.hard_negative_focus_policy_ids == (
        "seed_c3aac2f9dc_checkpoint_000025",
        "seed_c3aac2f9dc_main_bestresponse_u25_devbest",
        "seed_c3aac2f9dc_policy_000002",
    )
    assert stack.config.league.sampling.hard_negative_focus_weight_multiplier == pytest.approx(8.0)


def test_load_stack_config_supports_main_league_champion_hardneg_selected_alloutcome_focusoldhn_b2retention_probe() -> (
    None
):
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / (
            "configs/thesis/_shared/hardneg_selected_alloutcome/"
            "main_league_champion_hardneg_selected_alloutcome_focusoldhn_b2retention_b4b2guard_probe.yaml"
        )
    )

    assert stack.config.training is not None
    assert stack.config.training.optimizer.learning_rate == pytest.approx(0.0000048)
    assert stack.config.training.teacher_public_heuristic_coef == pytest.approx(0.026)
    assert stack.config.training.policy_anchor_coef == pytest.approx(0.52)
    assert stack.config.training.structured_aux.trajectory_bc_batch_episodes == 12
    assert stack.config.league is not None
    assert stack.config.league.sampling.noleague_baseline_mix_fraction == pytest.approx(0.30)
    assert stack.config.league.sampling.heuristic_public_mix_fraction == pytest.approx(0.10)
    assert stack.config.league.sampling.heuristic_public_variant_mix_fraction == pytest.approx(0.16)
    assert stack.config.league.sampling.champion_mix_fraction == pytest.approx(0.24)
    assert stack.config.league.sampling.hard_negative_mix_fraction == pytest.approx(0.20)
    assert stack.config.league.sampling.hard_negative_focus_policy_ids == (
        "seed_c3aac2f9dc_checkpoint_000025",
        "seed_c3aac2f9dc_main_bestresponse_u25_devbest",
        "seed_c3aac2f9dc_policy_000002",
    )
    assert stack.config.league.sampling.hard_negative_focus_weight_multiplier == pytest.approx(6.0)


def test_load_stack_config_supports_main_league_champion_hardneg_selected_alloutcome_extensionrepair_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / (
            "configs/thesis/_shared/hardneg_selected_alloutcome/"
            "main_league_champion_hardneg_selected_alloutcome_focusoldhn_extensionrepair_b4b2guard_probe.yaml"
        )
    )

    assert stack.config.training is not None
    assert stack.config.training.optimizer.learning_rate == pytest.approx(0.0000038)
    assert stack.config.training.policy_anchor_coef == pytest.approx(0.56)
    assert stack.config.training.structured_aux.trajectory_bc_batch_episodes == 14
    assert stack.config.training.structured_aux.trajectory_bc_dataset_path.endswith(
        "trajectory_bc_selected_a015_all32_plus_confirm128ext16x3.npz"
    )
    assert stack.config.training.structured_aux.trajectory_bc_teacher_action_coef == pytest.approx(0.17)
    assert stack.config.training.structured_aux.trajectory_bc_teacher_same_family_action_coef == pytest.approx(0.70)
    assert stack.config.league is not None
    assert stack.config.league.sampling.hard_negative_focus_weight_multiplier == pytest.approx(6.0)


def test_load_stack_config_supports_main_league_champion_hardneg_selected_alloutcome_winnerrepair_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / (
            "configs/thesis/_shared/hardneg_selected_alloutcome/"
            "main_league_champion_hardneg_selected_alloutcome_winnerrepair_b4b2guard_probe.yaml"
        )
    )

    assert stack.config.training is not None
    assert stack.config.training.optimizer.learning_rate == pytest.approx(0.0000045)
    assert stack.config.training.policy_anchor_coef == pytest.approx(0.54)
    assert stack.config.training.policy_anchor_top_action_coef == pytest.approx(0.13)
    assert stack.config.training.structured_aux.trajectory_bc_batch_episodes == 14
    assert stack.config.training.structured_aux.trajectory_bc_dataset_path.endswith(
        "trajectory_bc_selected_a015_disjoint4x_plus_extensionp5win2x.npz"
    )
    repair_seed_file = stack.seed_sets["hardneg_repair_train"]
    assert repair_seed_file == repo_root / "configs/seeds/hardneg_repair_train_seeds_20260518.txt"
    eval_seed_files = (
        repo_root / "configs/seeds/dev_eval_seeds.txt",
        repo_root / "configs/seeds/promotion_eval_seeds.txt",
        repo_root / "configs/seeds/report_eval_seeds.txt",
    )
    repair_seeds = set(parse_seed_file(repair_seed_file))
    eval_seeds = set().union(*(set(parse_seed_file(path)) for path in eval_seed_files))
    assert repair_seeds.isdisjoint(eval_seeds)
    assert stack.config.league is not None
    assert stack.config.league.sampling.champion_mix_fraction == pytest.approx(0.24)
    assert stack.config.league.sampling.hard_negative_mix_fraction == pytest.approx(0.20)
    assert stack.config.league.sampling.hard_negative_focus_weight_multiplier == pytest.approx(6.0)


def test_load_stack_config_supports_main_league_champion_hardneg_stratifiedwinnerrepair_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / (
            "configs/thesis/_shared/hardneg_selected_stratified/"
            "main_league_champion_hardneg_selected_alloutcome_stratifiedwinnerrepair_b4b2guard_probe.yaml"
        )
    )

    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert stack.config.training.optimizer.learning_rate == pytest.approx(0.0000038)
    assert structured_aux.trajectory_bc_aux_updates == 2
    assert structured_aux.trajectory_bc_batch_episodes == 16
    assert structured_aux.trajectory_bc_focus_source_labels == (
        "extensionrepair_p5_hardneg_disjoint_win16x6_a",
        "extensionrepair_p5_hardneg_disjoint_win16x6_b",
    )
    assert structured_aux.trajectory_bc_focus_fraction == pytest.approx(0.50)
    assert structured_aux.trajectory_bc_teacher_same_family_action_coef == pytest.approx(0.72)
    assert stack.config.league is not None
    assert stack.config.league.sampling.hard_negative_overlaps_champions is False


def test_load_stack_config_supports_main_league_champion_hardneg_overlap_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / (
            "configs/thesis/_shared/hardneg_selected_stratified/"
            "main_league_champion_hardneg_selected_alloutcome_stratifiedwinnerrepair_overlap_b4b2guard_probe.yaml"
        )
    )

    assert stack.config.training is not None
    assert stack.config.training.structured_aux.trajectory_bc_focus_fraction == pytest.approx(0.50)
    assert stack.config.league is not None
    assert stack.config.league.sampling.hard_negative_overlaps_champions is True


def test_load_stack_config_supports_main_league_b1_loss_topaction_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / (
            "configs/thesis/_shared/hardneg_selected_stratified/"
            "main_league_champion_hardneg_selected_alloutcome_stratifiedwinnerrepair_overlap_b1losstopaction_b4b2guard_probe.yaml"
        )
    )

    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert structured_aux.policy_anchor_coef == pytest.approx(0.60)
    assert structured_aux.trajectory_bc_dataset_path.endswith(
        "trajectory_bc_winnerrepair_plus_b1_lossstate_policyb_topaction.npz"
    )
    assert structured_aux.trajectory_bc_focus_source_labels == ("b1_lossstate_policyb_topaction",)
    assert structured_aux.trajectory_bc_focus_fraction == pytest.approx(0.25)
    assert structured_aux.trajectory_bc_teacher_same_family_action_coef == pytest.approx(0.78)
    assert stack.config.league is not None
    assert stack.config.league.sampling.hard_negative_overlaps_champions is True


def test_load_stack_config_supports_main_league_b1_hardneg_loss_topaction_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / (
            "configs/thesis/_shared/hardneg_selected_stratified/"
            "main_league_champion_hardneg_selected_alloutcome_stratifiedwinnerrepair_overlap_"
            "b1hnlosstopaction_b4b2guard_probe.yaml"
        )
    )

    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert structured_aux.trajectory_bc_dataset_path.endswith(
        "trajectory_bc_winnerrepair_plus_b1_plus_hn_lossstate_tv010.npz"
    )
    assert structured_aux.trajectory_bc_focus_source_labels == (
        "b1_lossstate_policyb_topaction",
        "hn_lossstate_policy000002_topaction",
        "hn_lossstate_checkpoint000025_topaction",
        "hn_lossstate_bestresponse_topaction",
        "hn_lossstate_mainleague_topaction",
        "hn_lossstate_policy000003_topaction",
        "hn_lossstate_policy000004_topaction",
    )
    assert structured_aux.trajectory_bc_focus_fraction == pytest.approx(0.35)
    assert structured_aux.trajectory_bc_teacher_same_family_action_coef == pytest.approx(0.78)
    assert stack.config.league is not None
    assert stack.config.league.sampling.hard_negative_overlaps_champions is True


def test_load_stack_config_supports_main_league_b1_hardneg_preserved_winner_focus_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / (
            "configs/thesis/_shared/hardneg_selected_stratified/"
            "main_league_champion_hardneg_selected_alloutcome_stratifiedwinnerrepair_overlap_"
            "b1hnlosstopaction_preservewinnerfocus_b4b2guard_probe.yaml"
        )
    )

    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert structured_aux.trajectory_bc_dataset_path.endswith(
        "trajectory_bc_winnerrepair_preserved_plus_b1_plus_hn_lossstate_tv010.npz"
    )
    assert structured_aux.trajectory_bc_focus_source_labels == (
        "extensionrepair_p5_hardneg_disjoint_win16x6_a",
        "extensionrepair_p5_hardneg_disjoint_win16x6_b",
        "b1_lossstate_policyb_topaction",
        "hn_lossstate_policy000002_topaction",
        "hn_lossstate_checkpoint000025_topaction",
        "hn_lossstate_bestresponse_topaction",
        "hn_lossstate_mainleague_topaction",
        "hn_lossstate_policy000003_topaction",
        "hn_lossstate_policy000004_topaction",
    )
    assert structured_aux.trajectory_bc_focus_fraction == pytest.approx(0.50)
    assert structured_aux.trajectory_bc_teacher_same_family_action_coef == pytest.approx(0.78)
    assert stack.config.league is not None
    assert stack.config.league.sampling.hard_negative_overlaps_champions is True


def test_load_stack_config_supports_main_league_b1_hardneg_preserved_winner_b1b3repair_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / (
            "configs/thesis/_shared/hardneg_selected_stratified/"
            "main_league_champion_hardneg_selected_alloutcome_stratifiedwinnerrepair_overlap_"
            "b1hnlosstopaction_preservewinnerfocus_b1b3repair_b4b2guard_probe.yaml"
        )
    )

    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert structured_aux.trajectory_bc_dataset_path.endswith(
        "trajectory_bc_winnerrepair_preserved_plus_b1_hn_plus_confirm128_b1b3_lossstate.npz"
    )
    assert structured_aux.trajectory_bc_focus_source_labels == (
        "extensionrepair_p5_hardneg_disjoint_win16x6_a",
        "extensionrepair_p5_hardneg_disjoint_win16x6_b",
        "b1_lossstate_policyb_topaction",
        "hn_lossstate_policy000002_topaction",
        "hn_lossstate_checkpoint000025_topaction",
        "hn_lossstate_bestresponse_topaction",
        "hn_lossstate_mainleague_topaction",
        "hn_lossstate_policy000003_topaction",
        "hn_lossstate_policy000004_topaction",
        "b1_lossstate_confirm128_policyb_topaction",
        "b3_lossstate_confirm128_policyb_topaction",
    )
    assert structured_aux.trajectory_bc_focus_fraction == pytest.approx(0.55)
    assert structured_aux.trajectory_bc_teacher_same_family_action_coef == pytest.approx(0.78)
    assert stack.config.league is not None
    assert stack.config.league.sampling.hard_negative_overlaps_champions is True


def test_load_stack_config_supports_main_league_grouped_b1b3_hardneg_repair_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / (
            "configs/thesis/_shared/hardneg_selected_stratified/"
            "main_league_champion_hardneg_selected_alloutcome_stratifiedwinnerrepair_overlap_"
            "b1hnlosstopaction_preservewinnerfocus_b1b3repair_grouped_b4b2guard_probe.yaml"
        )
    )

    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert structured_aux.trajectory_bc_dataset_path.endswith(
        "trajectory_bc_winnerrepair_preserved_plus_b1_hn_plus_confirm128_b1b3_lossstate.npz"
    )
    assert structured_aux.trajectory_bc_focus_source_labels == ()
    assert structured_aux.trajectory_bc_focus_fraction == pytest.approx(0.0)
    assert tuple(group.name for group in structured_aux.trajectory_bc_focus_groups) == (
        "winner_extension_repair",
        "hard_negative_lossstate_repair",
        "fixed_b1b3_lossstate_repair",
    )
    assert tuple(group.fraction for group in structured_aux.trajectory_bc_focus_groups) == pytest.approx(
        (0.20, 0.15, 0.20)
    )
    assert structured_aux.trajectory_bc_focus_groups[1].source_labels == (
        "hn_lossstate_policy000002_topaction",
        "hn_lossstate_checkpoint000025_topaction",
        "hn_lossstate_bestresponse_topaction",
        "hn_lossstate_mainleague_topaction",
        "hn_lossstate_policy000003_topaction",
        "hn_lossstate_policy000004_topaction",
    )
    assert structured_aux.trajectory_bc_teacher_same_family_action_coef == pytest.approx(0.78)
    assert stack.config.league is not None
    assert stack.config.league.sampling.hard_negative_overlaps_champions is True


def test_load_stack_config_supports_main_league_grouped_fixedwin_repair_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / (
            "configs/thesis/_shared/hardneg_selected_stratified/"
            "main_league_champion_hardneg_selected_alloutcome_stratifiedwinnerrepair_overlap_"
            "b1hnlosstopaction_preservewinnerfocus_b1b3repair_grouped_fixedwin_b4b2guard_probe.yaml"
        )
    )

    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert structured_aux.trajectory_bc_dataset_path.endswith("trajectory_bc_grouped_b1hn_b1b3_plus_fixedwins.npz")
    assert tuple(group.name for group in structured_aux.trajectory_bc_focus_groups) == (
        "fixed_win_rehearsal",
        "fixed_lossstate_repair",
        "hard_negative_lossstate_repair",
        "winner_extension_repair",
    )
    assert tuple(group.fraction for group in structured_aux.trajectory_bc_focus_groups) == pytest.approx(
        (0.20, 0.15, 0.15, 0.15)
    )
    assert structured_aux.trajectory_bc_focus_groups[0].source_labels == (
        "fixed_b1_win64_rehearsal",
        "direct_b2_wins64",
        "direct_b3_wins64",
        "direct_b4_wins64",
    )
    assert structured_aux.trajectory_bc_focus_groups[2].source_labels == (
        "hn_lossstate_policy000002_topaction",
        "hn_lossstate_checkpoint000025_topaction",
        "hn_lossstate_bestresponse_topaction",
        "hn_lossstate_mainleague_topaction",
        "hn_lossstate_policy000003_topaction",
        "hn_lossstate_policy000004_topaction",
    )
    assert stack.config.league is not None
    assert stack.config.league.sampling.hard_negative_overlaps_champions is True


def test_load_stack_config_supports_main_league_grouped_b2split_fixedwin_repair_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / (
            "configs/thesis/_shared/hardneg_selected_stratified/"
            "main_league_champion_hardneg_selected_alloutcome_stratifiedwinnerrepair_overlap_"
            "b1hnlosstopaction_preservewinnerfocus_b1b3repair_grouped_b2split_fixedwin_b4b2guard_probe.yaml"
        )
    )

    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert structured_aux.trajectory_bc_dataset_path.endswith("trajectory_bc_grouped_b1hn_b1b3_plus_fixedwins.npz")
    assert tuple(group.name for group in structured_aux.trajectory_bc_focus_groups) == (
        "fixed_b2_win_rehearsal",
        "fixed_other_win_rehearsal",
        "fixed_lossstate_repair",
        "hard_negative_lossstate_repair",
        "winner_extension_repair",
    )
    assert tuple(group.fraction for group in structured_aux.trajectory_bc_focus_groups) == pytest.approx(
        (0.16, 0.12, 0.14, 0.14, 0.14)
    )
    assert structured_aux.trajectory_bc_focus_groups[0].source_labels == ("direct_b2_wins64",)
    assert structured_aux.trajectory_bc_focus_groups[1].source_labels == (
        "fixed_b1_win64_rehearsal",
        "direct_b3_wins64",
        "direct_b4_wins64",
    )
    assert stack.config.league is not None
    assert stack.config.league.sampling.hard_negative_overlaps_champions is True


def test_load_stack_config_supports_main_league_grouped_b2loss_fixedwin_repair_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / (
            "configs/thesis/_shared/hardneg_selected_stratified/"
            "main_league_champion_hardneg_selected_alloutcome_stratifiedwinnerrepair_overlap_"
            "b1hnlosstopaction_preservewinnerfocus_b1b3repair_grouped_b2loss_fixedwin_b4b2guard_probe.yaml"
        )
    )

    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert structured_aux.trajectory_bc_dataset_path.endswith(
        "trajectory_bc_grouped_b1hn_b1b3_b2loss_plus_fixedwins.npz"
    )
    assert tuple(group.name for group in structured_aux.trajectory_bc_focus_groups) == (
        "fixed_b2_lossstate_repair",
        "fixed_other_win_rehearsal",
        "fixed_b1b3_lossstate_repair",
        "hard_negative_lossstate_repair",
        "winner_extension_repair",
    )
    assert tuple(group.fraction for group in structured_aux.trajectory_bc_focus_groups) == pytest.approx(
        (0.16, 0.12, 0.14, 0.14, 0.14)
    )
    assert structured_aux.trajectory_bc_focus_groups[0].source_labels == ("b2_lossstate_grouped_u2_topaction",)
    assert structured_aux.trajectory_bc_focus_groups[2].source_labels == (
        "b1_lossstate_policyb_topaction",
        "b1_lossstate_confirm128_policyb_topaction",
        "b3_lossstate_confirm128_policyb_topaction",
    )
    assert stack.config.league is not None
    assert stack.config.league.sampling.hard_negative_overlaps_champions is True


def test_load_stack_config_supports_main_league_selected_conservative_online_guard_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/hardneg_selected/main_league_champion_hardneg_selected_conservative_online_guard_probe.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.optimizer.learning_rate == pytest.approx(0.000012)
    assert stack.config.training.exploration.entropy_coef == pytest.approx(0.0012)
    assert stack.config.training.exploration.actor_sampling_temperature == pytest.approx(0.86)
    structured_aux = stack.config.training.structured_aux
    assert structured_aux.policy_anchor_coef == pytest.approx(0.38)
    assert structured_aux.policy_anchor_top_action_coef == pytest.approx(0.08)
    assert structured_aux.policy_anchor_temperature == pytest.approx(0.66)
    assert structured_aux.trajectory_retention_coef == pytest.approx(0.05)
    assert structured_aux.trajectory_bc_every_updates == 0
    assert structured_aux.trajectory_bc_dataset_path == ""
    assert stack.config.league is not None
    sampling = stack.config.league.sampling
    assert sampling.noleague_baseline_reserved_envs_per_actor == 5
    assert sampling.noleague_baseline_mix_fraction == pytest.approx(0.36)
    assert sampling.heuristic_public_mix_fraction == pytest.approx(0.08)
    assert sampling.heuristic_public_variant_mix_fraction == pytest.approx(0.20)
    assert sampling.champion_mix_fraction == pytest.approx(0.18)
    assert sampling.hard_negative_mix_fraction == pytest.approx(0.12)
    assert sampling.hard_negative_overlaps_champions is True


def test_load_stack_config_supports_main_league_lowpressure_pairloss_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/hardneg_selected_paired/main_league_champion_hardneg_selected_lowpressure_pairloss_b4b2guard_probe.yaml"
    )

    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert structured_aux.trajectory_bc_dataset_path.endswith(
        "trajectory_bc_grouped_b1hn_b1b3_b2loss_plus_fixedwins.npz"
    )
    assert structured_aux.trajectory_bc_every_updates == 1
    assert structured_aux.trajectory_bc_aux_updates == 1
    assert structured_aux.trajectory_bc_batch_episodes == 12
    assert structured_aux.trajectory_bc_teacher_action_coef == pytest.approx(0.10)
    assert structured_aux.trajectory_bc_teacher_same_family_action_coef == pytest.approx(0.45)
    assert structured_aux.trajectory_bc_teacher_same_family_action_margin_coef == pytest.approx(0.05)
    assert tuple(group.name for group in structured_aux.trajectory_bc_focus_groups) == (
        "hard_negative_lossstate_repair",
        "fixed_b1b3_lossstate_repair",
        "fixed_b2_lossstate_repair",
    )
    assert tuple(group.fraction for group in structured_aux.trajectory_bc_focus_groups) == pytest.approx(
        (0.08, 0.08, 0.04)
    )
    assert structured_aux.trajectory_bc_focus_groups[0].source_labels == (
        "hn_lossstate_policy000002_topaction",
        "hn_lossstate_checkpoint000025_topaction",
        "hn_lossstate_bestresponse_topaction",
        "hn_lossstate_mainleague_topaction",
    )
    assert structured_aux.trajectory_bc_focus_groups[1].source_labels == (
        "b1_lossstate_confirm128_policyb_topaction",
        "b3_lossstate_confirm128_policyb_topaction",
    )
    assert structured_aux.trajectory_bc_focus_groups[2].source_labels == ("b2_lossstate_grouped_u2_topaction",)
    assert stack.config.training.optimizer.learning_rate == pytest.approx(0.000012)
    assert stack.config.league is not None
    sampling = stack.config.league.sampling
    assert sampling.noleague_baseline_mix_fraction == pytest.approx(0.36)
    assert sampling.champion_mix_fraction == pytest.approx(0.18)
    assert sampling.hard_negative_mix_fraction == pytest.approx(0.12)
    assert sampling.hard_negative_overlaps_champions is True


def test_load_stack_config_supports_main_league_paired_swing_contrastive_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/hardneg_selected_paired/main_league_champion_hardneg_selected_paired_swing_contrastive_probe.yaml"
    )

    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert structured_aux.trajectory_bc_every_updates == 0
    assert structured_aux.trajectory_bc_dataset_path == ""
    assert structured_aux.paired_swing_dataset_path.endswith(
        "trajectory_bc_grouped_b1hn_b1b3_b2loss_plus_fixedwins.npz"
    )
    assert structured_aux.paired_swing_every_updates == 1
    assert structured_aux.paired_swing_aux_updates == 1
    assert structured_aux.paired_swing_batch_episodes == 16
    assert structured_aux.paired_swing_margin == pytest.approx(0.25)
    assert structured_aux.paired_swing_coef == pytest.approx(0.05)
    assert structured_aux.paired_swing_positive_action_source == "teacher_action"
    assert structured_aux.paired_swing_negative_action_source == "actions"
    assert tuple(group.name for group in structured_aux.paired_swing_focus_groups) == (
        "fixed_b1b2b3_lossstate_contrast",
        "hard_negative_lossstate_contrast",
    )
    assert tuple(group.fraction for group in structured_aux.paired_swing_focus_groups) == pytest.approx((0.18, 0.06))
    assert structured_aux.paired_swing_focus_groups[0].source_labels == (
        "b1_lossstate_confirm128_policyb_topaction",
        "b2_lossstate_grouped_u2_topaction",
        "b3_lossstate_confirm128_policyb_topaction",
    )
    assert structured_aux.paired_swing_focus_groups[1].source_labels == (
        "hn_lossstate_policy000002_topaction",
        "hn_lossstate_checkpoint000025_topaction",
        "hn_lossstate_bestresponse_topaction",
        "hn_lossstate_mainleague_topaction",
    )


def test_load_stack_config_supports_main_league_paired_flipbc_conservative_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/hardneg_selected_paired/main_league_champion_hardneg_selected_paired_flipbc_conservative_probe.yaml"
    )

    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert structured_aux.trajectory_bc_dataset_path.endswith(
        "paired_flip_bc_balanced_fixed_preserve_learned_repair.npz"
    )
    assert structured_aux.trajectory_bc_every_updates == 1
    assert structured_aux.trajectory_bc_aux_updates == 1
    assert structured_aux.trajectory_bc_batch_episodes == 8
    assert structured_aux.trajectory_bc_teacher_action_coef == pytest.approx(0.08)
    assert structured_aux.trajectory_bc_teacher_same_family_action_coef == pytest.approx(0.35)
    assert structured_aux.trajectory_bc_teacher_same_family_action_margin_coef == pytest.approx(0.04)
    assert tuple(group.name for group in structured_aux.trajectory_bc_focus_groups) == (
        "fixed_b3b4_selected_preserve",
        "learned_champion_hardneg_repair",
    )
    assert tuple(group.fraction for group in structured_aux.trajectory_bc_focus_groups) == pytest.approx((0.16, 0.16))
    assert structured_aux.trajectory_bc_focus_groups[0].source_labels == (
        "fixed_preserve_B3 HeuristicPublicAggro",
        "fixed_preserve_B4 HeuristicPublicControl",
    )
    assert structured_aux.trajectory_bc_focus_groups[1].source_labels == (
        "learned_repair_seed_b8c698d26a_seed_c3aac2f9dc_main_league_selected",
        "learned_repair_seed_b8c698d26a_seed_c3aac2f9dc_policy_000003",
        "learned_repair_seed_b8c698d26a_seed_c3aac2f9dc_policy_000004",
        "learned_repair_seed_b8c698d26a_seed_c3aac2f9dc_policy_000005",
    )
    assert stack.config.training.optimizer.learning_rate == pytest.approx(0.000012)
    assert stack.config.league is not None
    sampling = stack.config.league.sampling
    assert sampling.noleague_baseline_mix_fraction == pytest.approx(0.36)
    assert sampling.heuristic_public_variant_mix_fraction == pytest.approx(0.20)
    assert sampling.champion_mix_fraction == pytest.approx(0.18)
    assert sampling.hard_negative_mix_fraction == pytest.approx(0.12)


def test_load_stack_config_supports_main_league_paired_flipbc_focusoldhn_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / (
            "configs/thesis/_shared/hardneg_selected_paired/"
            "main_league_champion_hardneg_selected_paired_flipbc_focusoldhn_conservative_probe.yaml"
        )
    )

    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert structured_aux.trajectory_bc_dataset_path.endswith(
        "paired_flip_bc_balanced_fixed_preserve_learned_repair.npz"
    )
    assert structured_aux.trajectory_bc_every_updates == 1
    assert tuple(group.name for group in structured_aux.trajectory_bc_focus_groups) == (
        "fixed_b3b4_selected_preserve",
        "learned_champion_hardneg_repair",
    )
    assert stack.config.league is not None
    sampling = stack.config.league.sampling
    assert sampling.champion_mix_fraction == pytest.approx(0.14)
    assert sampling.hard_negative_mix_fraction == pytest.approx(0.16)
    assert sampling.hard_negative_focus_policy_ids == (
        "seed_c3aac2f9dc_policy_000002",
        "seed_c3aac2f9dc_checkpoint_000025",
        "seed_c3aac2f9dc_main_bestresponse_u25_devbest",
    )
    assert sampling.hard_negative_focus_weight_multiplier == pytest.approx(3.0)
    assert sampling.hard_negative_overlaps_champions is True


def test_load_stack_config_supports_main_league_grouped128_paired_flipbc_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / (
            "configs/thesis/_shared/hardneg_selected_paired/"
            "main_league_champion_hardneg_selected_grouped128_paired_flipbc_focusoldhn_probe.yaml"
        )
    )

    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert structured_aux.trajectory_bc_dataset_path.endswith("paired_flip_bc_grouped128_balanced_fixed_learned.npz")
    assert structured_aux.trajectory_bc_batch_episodes == 16
    assert structured_aux.trajectory_bc_teacher_action_coef == pytest.approx(0.07)
    assert structured_aux.trajectory_bc_teacher_same_family_action_coef == pytest.approx(0.32)
    assert tuple(group.name for group in structured_aux.trajectory_bc_focus_groups) == (
        "grouped128_fixed_preserve",
        "grouped128_learned_repair",
    )
    assert tuple(group.fraction for group in structured_aux.trajectory_bc_focus_groups) == pytest.approx((0.25, 0.25))
    assert structured_aux.trajectory_bc_focus_groups[0].source_labels == (
        "grouped128_fixed_preserve_B1 NoLeague baseline",
        "grouped128_fixed_preserve_B2 HeuristicPublic",
        "grouped128_fixed_preserve_B3 HeuristicPublicAggro",
        "grouped128_fixed_preserve_B4 HeuristicPublicControl",
    )
    assert "grouped128_learned_repair_seed_b8c698d26a_seed_c3aac2f9dc_policy_000002" in (
        structured_aux.trajectory_bc_focus_groups[1].source_labels
    )
    assert "grouped128_learned_repair_seed_b8c698d26a_seed_c3aac2f9dc_checkpoint_000025" in (
        structured_aux.trajectory_bc_focus_groups[1].source_labels
    )
    assert stack.config.league is not None
    sampling = stack.config.league.sampling
    assert sampling.champion_mix_fraction == pytest.approx(0.14)
    assert sampling.hard_negative_mix_fraction == pytest.approx(0.16)
    assert sampling.hard_negative_focus_weight_multiplier == pytest.approx(3.0)


def test_load_stack_config_supports_main_league_outcome_contrastive_focusoldhn_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/hardneg_selected_outcome_contrastive/main_league_champion_hardneg_selected_outcome_contrastive_focusoldhn_probe.yaml"
    )

    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert structured_aux.trajectory_bc_every_updates == 0
    assert structured_aux.trajectory_bc_dataset_path == ""
    assert structured_aux.trajectory_bc_focus_groups == ()
    assert structured_aux.paired_swing_dataset_path.endswith(
        "paired_outcome_contrastive_grouped128_fixed_learned_probe.npz"
    )
    assert structured_aux.paired_swing_every_updates == 1
    assert structured_aux.paired_swing_batch_episodes == 16
    assert structured_aux.paired_swing_margin == pytest.approx(0.20)
    assert structured_aux.paired_swing_coef == pytest.approx(0.04)
    assert structured_aux.paired_swing_positive_action_source == "actions"
    assert structured_aux.paired_swing_negative_action_source == "teacher_action"
    assert tuple(group.name for group in structured_aux.paired_swing_focus_groups) == (
        "outcome_fixed_preserve",
        "outcome_learned_repair",
    )
    assert tuple(group.fraction for group in structured_aux.paired_swing_focus_groups) == pytest.approx((0.50, 0.50))
    assert structured_aux.paired_swing_focus_groups[0].source_labels == (
        "grouped128_fixed_preserve_B1 NoLeague baseline",
        "grouped128_fixed_preserve_B2 HeuristicPublic",
        "grouped128_fixed_preserve_B3 HeuristicPublicAggro",
        "grouped128_fixed_preserve_B4 HeuristicPublicControl",
    )
    assert "grouped128_learned_repair_seed_b8c698d26a_seed_c3aac2f9dc_policy_000002" in (
        structured_aux.paired_swing_focus_groups[1].source_labels
    )
    assert "grouped128_learned_repair_seed_b8c698d26a_seed_c3aac2f9dc_checkpoint_000025" in (
        structured_aux.paired_swing_focus_groups[1].source_labels
    )
    assert stack.config.league is not None
    sampling = stack.config.league.sampling
    assert sampling.champion_mix_fraction == pytest.approx(0.14)
    assert sampling.hard_negative_mix_fraction == pytest.approx(0.16)
    assert sampling.hard_negative_focus_weight_multiplier == pytest.approx(3.0)


def test_load_stack_config_supports_main_league_outcome_contrastive_full_focusoldhn_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / (
            "configs/thesis/_shared/hardneg_selected_outcome_contrastive/"
            "main_league_champion_hardneg_selected_outcome_contrastive_full_focusoldhn_probe.yaml"
        )
    )

    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert structured_aux.paired_swing_dataset_path.endswith(
        "paired_outcome_contrastive_grouped128_fixed_learned_full.npz"
    )
    assert structured_aux.paired_swing_positive_action_source == "actions"
    assert structured_aux.paired_swing_negative_action_source == "teacher_action"
    assert tuple(group.name for group in structured_aux.paired_swing_focus_groups) == (
        "outcome_fixed_preserve",
        "outcome_learned_repair",
    )


def test_load_stack_config_supports_main_league_outcome_contrastive_edgehn_focus_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/hardneg_selected_outcome_contrastive/main_league_champion_hardneg_selected_outcome_contrastive_edgehn_focus_probe.yaml"
    )

    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert structured_aux.policy_anchor_coef == pytest.approx(0.36)
    assert structured_aux.policy_anchor_top_action_coef == pytest.approx(0.075)
    assert structured_aux.paired_swing_dataset_path.endswith(
        "paired_outcome_contrastive_grouped128_fixed_learned_full.npz"
    )
    assert structured_aux.paired_swing_batch_episodes == 20
    assert structured_aux.paired_swing_coef == pytest.approx(0.05)
    assert tuple(group.name for group in structured_aux.paired_swing_focus_groups) == (
        "outcome_fixed_preserve",
        "outcome_old_hardneg_repair",
        "outcome_edge_hardneg_repair",
    )
    assert tuple(group.fraction for group in structured_aux.paired_swing_focus_groups) == pytest.approx(
        (0.40, 0.30, 0.30)
    )
    assert "grouped128_learned_repair_seed_b8c698d26a_seed_c3aac2f9dc_policy_000001" in (
        structured_aux.paired_swing_focus_groups[2].source_labels
    )
    assert "grouped128_learned_repair_seed_b8c698d26a_seed_c3aac2f9dc_policy_000005" in (
        structured_aux.paired_swing_focus_groups[2].source_labels
    )
    assert stack.config.league is not None
    sampling = stack.config.league.sampling
    assert sampling.hard_negative_focus_policy_ids == (
        "seed_c3aac2f9dc_policy_000001",
        "seed_c3aac2f9dc_policy_000002",
        "seed_c3aac2f9dc_checkpoint_000025",
        "seed_c3aac2f9dc_main_bestresponse_u25_devbest",
        "seed_c3aac2f9dc_policy_000005",
    )
    assert sampling.hard_negative_focus_weight_multiplier == pytest.approx(2.5)


def test_load_stack_config_supports_main_league_outcome_contrastive_edgehn_b1b2focus_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / (
            "configs/thesis/_shared/hardneg_selected_outcome_contrastive/"
            "main_league_champion_hardneg_selected_outcome_contrastive_edgehn_b1b2focus_probe.yaml"
        )
    )

    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert structured_aux.policy_anchor_coef == pytest.approx(0.38)
    assert structured_aux.policy_anchor_top_action_coef == pytest.approx(0.08)
    assert structured_aux.paired_swing_batch_episodes == 20
    assert structured_aux.paired_swing_coef == pytest.approx(0.045)
    assert tuple(group.name for group in structured_aux.paired_swing_focus_groups) == (
        "outcome_fixed_b1b2_preserve",
        "outcome_fixed_b3b4_preserve",
        "outcome_old_hardneg_repair",
        "outcome_edge_hardneg_repair",
    )
    assert tuple(group.fraction for group in structured_aux.paired_swing_focus_groups) == pytest.approx(
        (0.30, 0.15, 0.25, 0.30)
    )
    assert structured_aux.paired_swing_focus_groups[0].source_labels == (
        "grouped128_fixed_preserve_B1 NoLeague baseline",
        "grouped128_fixed_preserve_B2 HeuristicPublic",
    )
    assert structured_aux.paired_swing_focus_groups[3].source_labels == (
        "grouped128_learned_repair_seed_b8c698d26a_seed_c3aac2f9dc_policy_000001",
        "grouped128_learned_repair_seed_b8c698d26a_seed_c3aac2f9dc_policy_000005",
    )


def test_load_stack_config_supports_main_league_outcome_contrastive_extpreserve_a0375_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / (
            "configs/thesis/_shared/hardneg_selected_outcome_contrastive/"
            "main_league_champion_hardneg_selected_outcome_contrastive_extpreserve_a0375_probe.yaml"
        )
    )

    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert stack.config.training.optimizer.learning_rate == pytest.approx(0.000008)
    assert structured_aux.policy_anchor_coef == pytest.approx(0.42)
    assert structured_aux.policy_anchor_top_action_coef == pytest.approx(0.09)
    assert structured_aux.paired_swing_dataset_path.endswith(
        "paired_outcome_contrastive_grouped128_plus_extpreserve_a0375.npz"
    )
    assert tuple(group.name for group in structured_aux.paired_swing_focus_groups) == (
        "grouped_fixed_preserve",
        "grouped_learned_repair",
        "ext256_fixed_preserve",
        "ext256_learned_preserve",
    )
    assert tuple(group.fraction for group in structured_aux.paired_swing_focus_groups) == pytest.approx(
        (0.25, 0.25, 0.25, 0.25)
    )
    assert "ext256_fixed_preserveB1 NoLeague baseline" in (structured_aux.paired_swing_focus_groups[2].source_labels)
    assert "ext256_learned_preserveseed_b8c698d26a_seed_c3aac2f9dc_policy_000005" in (
        structured_aux.paired_swing_focus_groups[3].source_labels
    )


def test_load_stack_config_supports_main_league_outcome_contrastive_rawext256_allpreserve_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / (
            "configs/thesis/_shared/hardneg_selected_outcome_contrastive/"
            "main_league_champion_hardneg_selected_outcome_contrastive_rawext256_allpreserve_repair_probe.yaml"
        )
    )

    assert stack.config.training is not None
    assert stack.config.league is not None
    structured_aux = stack.config.training.structured_aux
    assert stack.config.training.optimizer.learning_rate == pytest.approx(0.000004)
    assert stack.config.league.sampling.hard_negative_focus_policy_ids == (
        "seed_b8c698d26a_seed_c3aac2f9dc_policy_000001",
        "seed_b8c698d26a_seed_c3aac2f9dc_policy_000002",
        "seed_b8c698d26a_seed_c3aac2f9dc_checkpoint_000025",
        "seed_b8c698d26a_seed_c3aac2f9dc_main_bestresponse_u25_devbest",
        "seed_b8c698d26a_seed_c3aac2f9dc_policy_000005",
    )
    assert structured_aux.policy_anchor_coef == pytest.approx(0.42)
    assert structured_aux.paired_swing_coef == pytest.approx(0.055)
    assert structured_aux.paired_swing_dataset_path.endswith(
        "paired_outcome_contrastive_grouped128_plus_rawext256_allpreserve.npz"
    )
    assert tuple(group.name for group in structured_aux.paired_swing_focus_groups) == (
        "grouped_fixed_preserve",
        "grouped_learned_repair",
        "rawext256_fixed_preserve",
        "rawext256_learned_preserve",
    )
    assert tuple(group.fraction for group in structured_aux.paired_swing_focus_groups) == pytest.approx(
        (0.20, 0.20, 0.30, 0.30)
    )
    assert "rawext256_allfixed_preserve_B2 HeuristicPublic" in (
        structured_aux.paired_swing_focus_groups[2].source_labels
    )
    assert "rawext256_alllearned_preserve_seed_b8c698d26a_seed_c3aac2f9dc_policy_000001" in (
        structured_aux.paired_swing_focus_groups[3].source_labels
    )


def test_load_stack_config_supports_main_league_outcome_contrastive_rawext256_b2_policy1_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / (
            "configs/thesis/_shared/hardneg_selected_outcome_contrastive/"
            "main_league_champion_hardneg_selected_outcome_contrastive_rawext256_b2_policy1_repair_probe.yaml"
        )
    )

    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert structured_aux.paired_swing_coef == pytest.approx(0.05)
    assert tuple(group.name for group in structured_aux.paired_swing_focus_groups) == (
        "grouped_fixed_preserve",
        "grouped_learned_repair",
        "rawext256_b2_fixed_preserve",
        "rawext256_policy1_learned_preserve",
    )
    assert tuple(group.fraction for group in structured_aux.paired_swing_focus_groups) == pytest.approx(
        (0.15, 0.50, 0.25, 0.10)
    )
    assert structured_aux.paired_swing_focus_groups[2].source_labels == (
        "rawext256_allfixed_preserve_B2 HeuristicPublic",
    )
    assert structured_aux.paired_swing_focus_groups[3].source_labels == (
        "rawext256_alllearned_preserve_seed_b8c698d26a_seed_c3aac2f9dc_policy_000001",
    )


def test_load_stack_config_supports_main_league_outcome_contrastive_rawext256_b2_oldhn_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / (
            "configs/thesis/_shared/hardneg_selected_outcome_contrastive/"
            "main_league_champion_hardneg_selected_outcome_contrastive_rawext256_b2_oldhn_repair_probe.yaml"
        )
    )

    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert structured_aux.paired_swing_coef == pytest.approx(0.05)
    assert tuple(group.name for group in structured_aux.paired_swing_focus_groups) == (
        "grouped_fixed_preserve",
        "grouped_learned_repair",
        "rawext256_b2_fixed_preserve",
        "rawext256_old_checkpoint_preserve",
        "rawext256_old_bestresponse_preserve",
    )
    assert tuple(group.fraction for group in structured_aux.paired_swing_focus_groups) == pytest.approx(
        (0.15, 0.40, 0.25, 0.10, 0.10)
    )
    assert structured_aux.paired_swing_focus_groups[2].source_labels == (
        "rawext256_allfixed_preserve_B2 HeuristicPublic",
    )
    assert structured_aux.paired_swing_focus_groups[3].source_labels == (
        "rawext256_alllearned_preserve_seed_b8c698d26a_seed_c3aac2f9dc_checkpoint_000025",
    )
    assert structured_aux.paired_swing_focus_groups[4].source_labels == (
        "rawext256_alllearned_preserve_seed_b8c698d26a_seed_c3aac2f9dc_main_bestresponse_u25_devbest",
    )


def test_load_stack_config_supports_main_league_interp_a050_continue_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/hardneg_interp_a050/main_league_champion_hardneg_interp_a050_b2_oldhn_continue_probe.yaml"
    )

    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert stack.config.training.optimizer.learning_rate == pytest.approx(0.000002)
    assert structured_aux.policy_anchor_coef == pytest.approx(0.46)
    assert structured_aux.policy_anchor_top_action_coef == pytest.approx(0.10)
    assert structured_aux.paired_swing_coef == pytest.approx(0.04)
    assert tuple(group.name for group in structured_aux.paired_swing_focus_groups) == (
        "grouped_fixed_preserve",
        "grouped_learned_repair",
        "rawext256_b2_fixed_preserve",
        "rawext256_old_checkpoint_preserve",
        "rawext256_old_bestresponse_preserve",
    )


def test_load_stack_config_supports_main_league_interp_a050_u1_nowarm_b2guard_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/hardneg_interp_a050/main_league_champion_hardneg_interp_a050_u1_nowarm_b2guard_continue_probe.yaml"
    )

    assert stack.config.league is not None
    assert stack.config.league.warmup.first_updates == 0
    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert stack.config.training.optimizer.learning_rate == pytest.approx(0.000002)
    assert structured_aux.policy_anchor_coef == pytest.approx(0.48)
    assert structured_aux.policy_anchor_top_action_coef == pytest.approx(0.12)
    assert structured_aux.trajectory_retention_coef == pytest.approx(0.08)
    assert structured_aux.paired_swing_coef == pytest.approx(0.04)
    assert tuple(group.name for group in structured_aux.paired_swing_focus_groups) == (
        "grouped_fixed_preserve",
        "grouped_learned_repair",
        "rawext256_b2_fixed_preserve",
        "rawext256_old_checkpoint_preserve",
        "rawext256_old_bestresponse_preserve",
    )
    assert tuple(group.fraction for group in structured_aux.paired_swing_focus_groups) == pytest.approx(
        (0.20, 0.30, 0.30, 0.10, 0.10)
    )


def test_load_stack_config_supports_main_league_interp_a050_u1_nowarm_balanced_b2guard_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / (
            "configs/thesis/_shared/hardneg_interp_a050/"
            "main_league_champion_hardneg_interp_a050_u1_nowarm_balanced_b2guard_continue_probe.yaml"
        )
    )

    assert stack.config.league is not None
    assert stack.config.league.warmup.first_updates == 0
    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert structured_aux.policy_anchor_coef == pytest.approx(0.46)
    assert structured_aux.policy_anchor_top_action_coef == pytest.approx(0.11)
    assert structured_aux.trajectory_retention_coef == pytest.approx(0.06)
    assert tuple(group.name for group in structured_aux.paired_swing_focus_groups) == (
        "grouped_fixed_preserve",
        "grouped_learned_repair",
        "rawext256_b2_fixed_preserve",
        "rawext256_old_checkpoint_preserve",
        "rawext256_old_bestresponse_preserve",
    )
    assert tuple(group.fraction for group in structured_aux.paired_swing_focus_groups) == pytest.approx(
        (0.15, 0.40, 0.30, 0.075, 0.075)
    )


def test_load_stack_config_supports_main_league_interp_a050_p1p2_a025_b2exact_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / (
            "configs/thesis/_shared/hardneg_interp_a050/"
            "main_league_champion_hardneg_interp_a050_p1p2_a025_b2exact_nowarm_continue_probe.yaml"
        )
    )

    assert stack.config.league is not None
    assert stack.config.league.warmup.first_updates == 0
    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert stack.config.training.optimizer.learning_rate == pytest.approx(0.000001)
    assert structured_aux.policy_anchor_coef == pytest.approx(0.50)
    assert structured_aux.policy_anchor_top_action_coef == pytest.approx(0.12)
    assert structured_aux.trajectory_retention_coef == pytest.approx(0.10)
    assert structured_aux.paired_swing_coef == pytest.approx(0.045)
    assert (
        structured_aux.paired_swing_dataset_path
        == "runs/paired_outcome_contrastive_grouped128_rawext256_plus_a025_b2exact_20260520/"
        "paired_outcome_contrastive_grouped128_rawext256_plus_a025_b2exact.npz"
    )
    assert tuple(group.name for group in structured_aux.paired_swing_focus_groups) == (
        "grouped_fixed_preserve",
        "grouped_learned_repair",
        "rawext256_b2_fixed_preserve",
        "a025_exact_b2_selected_preserve",
        "rawext256_old_checkpoint_preserve",
        "rawext256_old_bestresponse_preserve",
    )
    assert tuple(group.fraction for group in structured_aux.paired_swing_focus_groups) == pytest.approx(
        (0.15, 0.30, 0.20, 0.25, 0.05, 0.05)
    )
    assert structured_aux.paired_swing_focus_groups[3].source_labels == ("a025_b2_selectedwin_loss_B2 HeuristicPublic",)


def test_load_stack_config_supports_main_league_interp_a050_p1p2_a025_b2exact_learnedp16_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / (
            "configs/thesis/_shared/hardneg_interp_a050/"
            "main_league_champion_hardneg_interp_a050_p1p2_a025_b2exact_learnedp16_nowarm_continue_probe.yaml"
        )
    )

    assert stack.config.league is not None
    assert stack.config.league.warmup.first_updates == 0
    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert structured_aux.paired_swing_batch_episodes == 24
    assert (
        structured_aux.paired_swing_dataset_path
        == "runs/paired_outcome_contrastive_grouped128_rawext256_b2exact_plus_learnedp16_20260520/"
        "paired_outcome_contrastive_grouped128_rawext256_b2exact_plus_learnedp16.npz"
    )
    assert tuple(group.name for group in structured_aux.paired_swing_focus_groups) == (
        "grouped_fixed_preserve",
        "grouped_learned_repair",
        "rawext256_b2_fixed_preserve",
        "a025_exact_b2_selected_preserve",
        "a025_exact_learned_p16_preserve",
        "rawext256_old_checkpoint_preserve",
        "rawext256_old_bestresponse_preserve",
    )
    assert tuple(group.fraction for group in structured_aux.paired_swing_focus_groups) == pytest.approx(
        (0.12, 0.22, 0.15, 0.23, 0.23, 0.025, 0.025)
    )
    assert structured_aux.paired_swing_focus_groups[4].source_labels == (
        "a025_learned_preserve_seed_b8c698d26a_seed_c3aac2f9dc_policy_000002",
        "a025_learned_preserve_seed_b8c698d26a_seed_c3aac2f9dc_policy_000004",
        "a025_learned_preserve_seed_b8c698d26a_seed_c3aac2f9dc_policy_000005",
    )


def test_load_stack_config_supports_main_league_interp_a050_p1p2_a025_b2pair70_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / (
            "configs/thesis/_shared/hardneg_interp_a050/"
            "main_league_champion_hardneg_interp_a050_p1p2_a025_b2pair70_nowarm_continue_probe.yaml"
        )
    )

    assert stack.config.league is not None
    assert stack.config.league.warmup.first_updates == 0
    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert stack.config.training.optimizer.learning_rate == pytest.approx(0.000001)
    assert structured_aux.policy_anchor_coef == pytest.approx(0.50)
    assert structured_aux.policy_anchor_top_action_coef == pytest.approx(0.12)
    assert structured_aux.trajectory_retention_coef == pytest.approx(0.10)
    assert (
        structured_aux.paired_swing_dataset_path
        == "runs/paired_outcome_contrastive_grouped128_rawext256_plus_b2pair70_20260520/"
        "paired_outcome_contrastive_grouped128_rawext256_plus_b2pair70.npz"
    )
    assert tuple(group.name for group in structured_aux.paired_swing_focus_groups) == (
        "grouped_fixed_preserve",
        "grouped_learned_repair",
        "rawext256_b2_fixed_preserve",
        "a025_exact_b2_pair70_selected_preserve",
        "rawext256_old_checkpoint_preserve",
        "rawext256_old_bestresponse_preserve",
    )
    assert tuple(group.fraction for group in structured_aux.paired_swing_focus_groups) == pytest.approx(
        (0.14, 0.34, 0.18, 0.24, 0.05, 0.05)
    )
    assert structured_aux.paired_swing_focus_groups[3].source_labels == (
        "a025_b2_pair70_selected_preserve_B2 HeuristicPublic",
    )


def test_load_stack_config_supports_main_league_a075_nonconflict_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/hardneg_a050_a075_followup/main_league_champion_hardneg_interp_a075_nonconflict_continue_probe.yaml"
    )

    assert stack.config.league is not None
    assert stack.config.league.warmup.first_updates == 0
    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert stack.config.training.optimizer.learning_rate == pytest.approx(0.00000075)
    assert structured_aux.policy_anchor_coef == pytest.approx(0.55)
    assert structured_aux.policy_anchor_top_action_coef == pytest.approx(0.14)
    assert structured_aux.trajectory_retention_coef == pytest.approx(0.12)
    assert structured_aux.paired_swing_coef == pytest.approx(0.035)
    assert structured_aux.paired_swing_batch_episodes == 7
    assert (
        structured_aux.paired_swing_dataset_path
        == "runs/paired_outcome_contrastive_a075_nonconflict_confirm256_micro_20260520/"
        "paired_outcome_contrastive_a075_nonconflict_confirm256_micro.npz"
    )
    assert tuple(group.name for group in structured_aux.paired_swing_focus_groups) == (
        "a075_nonconflict_fixed_preserve",
        "a075_nonconflict_learned_preserve",
        "a075_nonconflict_learned_repair",
    )
    assert tuple(group.fraction for group in structured_aux.paired_swing_focus_groups) == pytest.approx(
        (0.45, 0.30, 0.25)
    )
    assert "a075_nonconflict_b2_pair229_preserve_B2 HeuristicPublic" in (
        structured_aux.paired_swing_focus_groups[0].source_labels
    )
    assert "a075_nonconflict_p5_pair68_repair_seed_b8c698d26a_seed_c3aac2f9dc_policy_000005" in (
        structured_aux.paired_swing_focus_groups[2].source_labels
    )


def test_load_stack_config_supports_main_league_a075_broad_conflictfilter_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/hardneg_a050_a075_followup/main_league_champion_hardneg_interp_a075_broad_conflictfilter_continue_probe.yaml"
    )

    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert structured_aux.paired_swing_conflict_filter == "current_state"
    assert (
        structured_aux.paired_swing_dataset_path
        == "runs/paired_outcome_contrastive_grouped128_rawext256_plus_b2pair70_20260520/"
        "paired_outcome_contrastive_grouped128_rawext256_plus_b2pair70.npz"
    )
    assert structured_aux.paired_swing_coef == pytest.approx(0.045)


def test_load_stack_config_supports_main_league_a075_episodepref_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / (
            "configs/thesis/_shared/hardneg_a050_a075_followup/"
            "main_league_champion_hardneg_interp_a075_broad_conflictfilter_episodepref_probe.yaml"
        )
    )

    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert structured_aux.paired_swing_conflict_filter == "current_state"
    assert structured_aux.paired_swing_loss_scope == "episode_mean"
    assert structured_aux.paired_swing_aux_updates == 4


def test_load_stack_config_supports_main_league_a075_context_episodepref_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        _a075_context_config(
            repo_root,
            "main_league_champion_hardneg_a075_context_broad_conflictfilter_episodepref_probe.yaml",
        )
    )

    assert stack.config.model is not None
    assert stack.config.model.opponent_context_hidden_scale == pytest.approx(0.0)
    assert stack.config.model.opponent_context_trainable_hidden_scale == pytest.approx(1.0)
    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert structured_aux.paired_swing_conflict_filter == "current_state"
    assert structured_aux.paired_swing_loss_scope == "episode_mean"
    assert tuple(group.name for group in structured_aux.paired_swing_focus_groups) == (
        "grouped_fixed_preserve",
        "grouped_learned_repair",
        "rawext256_b2_fixed_preserve",
        "a025_exact_b2_pair70_selected_preserve",
        "rawext256_old_checkpoint_preserve",
        "rawext256_old_bestresponse_preserve",
    )


def test_load_stack_config_supports_main_league_a075_context_labelmean_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        _a075_context_config(
            repo_root,
            "main_league_champion_hardneg_a075_context_broad_conflictfilter_labelmean_probe.yaml",
        )
    )

    assert stack.config.model is not None
    assert stack.config.model.opponent_context_trainable_hidden_scale == pytest.approx(1.0)
    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert structured_aux.paired_swing_conflict_filter == "current_state"
    assert structured_aux.paired_swing_loss_scope == "label_mean"


def test_load_stack_config_supports_main_league_a075_context_hidden_adapteronly_labelmean_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        _a075_context_config(
            repo_root,
            "main_league_champion_hardneg_a075_context_hidden_adapteronly_broad_labelmean_probe.yaml",
        )
    )

    assert stack.config.model is not None
    assert stack.config.model.opponent_context_trainable_hidden_scale == pytest.approx(1.0)
    assert stack.config.model.opponent_context_adapter_train_only is True
    assert stack.config.model.opponent_context_adapter_lr_multiplier == pytest.approx(2000.0)
    assert stack.config.training is not None
    assert stack.config.training.optimizer.learning_rate == pytest.approx(0.0000001)
    assert stack.config.training.structured_aux.paired_swing_loss_scope == "label_mean"


def test_load_stack_config_supports_main_league_a075_preference_balanced_micro_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/hardneg_a050_a075_followup/main_league_champion_hardneg_a075_preference_balanced_micro_probe.yaml"
    )

    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert stack.config.training.optimizer.learning_rate == pytest.approx(0.0000005)
    assert structured_aux.paired_swing_dataset_path == ""
    assert structured_aux.paired_swing_every_updates == 0
    assert (
        structured_aux.paired_outcome_preference_dataset_path
        == "runs/paired_outcome_preference_a075_balanced_micro_20260520/"
        "paired_outcome_preference_a075_balanced_micro.npz"
    )
    assert structured_aux.paired_outcome_preference_every_updates == 1
    assert structured_aux.paired_outcome_preference_batch_episodes == 22
    assert structured_aux.paired_outcome_preference_coef == pytest.approx(0.08)
    assert structured_aux.paired_outcome_preference_beta == pytest.approx(0.20)


def test_load_stack_config_supports_main_league_a075_preference_groupbalanced_micro_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/hardneg_a050_a075_followup/main_league_champion_hardneg_a075_preference_groupbalanced_micro_probe.yaml"
    )

    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert structured_aux.paired_outcome_preference_group_balance is True
    assert structured_aux.paired_outcome_preference_batch_episodes == 22
    assert structured_aux.paired_outcome_preference_coef == pytest.approx(0.08)


def test_load_stack_config_supports_main_league_a075_context_preference_groupbalanced_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        _a075_context_config(
            repo_root,
            "main_league_champion_hardneg_a075_context_preference_groupbalanced_micro_probe.yaml",
        )
    )

    assert stack.config.model is not None
    assert stack.config.model.opponent_context_trainable_candidate_residual_scale == pytest.approx(1.0)
    assert stack.config.model.opponent_context_candidate_residual_mode == "bilinear"
    assert stack.config.model.opponent_context_adapter_train_only is True
    assert stack.config.training is not None
    assert stack.config.training.optimizer.learning_rate == pytest.approx(0.0000001)
    structured_aux = stack.config.training.structured_aux
    assert structured_aux.paired_outcome_preference_group_balance is True
    assert structured_aux.policy_anchor_coef == pytest.approx(0.0)
    assert structured_aux.trajectory_retention_coef == pytest.approx(0.0)


def test_load_stack_config_supports_main_league_a075_context_candres_episodepref_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        _a075_context_config(
            repo_root,
            "main_league_champion_hardneg_a075_context_candres_broad_conflictfilter_episodepref_probe.yaml",
        )
    )

    assert stack.config.model is not None
    assert stack.config.model.opponent_context_trainable_hidden_scale == pytest.approx(0.0)
    assert stack.config.model.opponent_context_trainable_candidate_residual_scale == pytest.approx(1.0)
    assert stack.config.model.opponent_context_candidate_residual_mode == "bilinear"
    assert stack.config.model.opponent_context_adapter_train_only is True
    assert stack.config.model.opponent_context_adapter_lr_multiplier == pytest.approx(1500.0)
    assert stack.config.training is not None
    assert stack.config.training.optimizer.learning_rate == pytest.approx(0.0000001)
    structured_aux = stack.config.training.structured_aux
    assert structured_aux.paired_swing_conflict_filter == "current_state"
    assert structured_aux.paired_swing_loss_scope == "episode_mean"
    assert structured_aux.paired_swing_aux_updates == 6
    assert structured_aux.paired_swing_coef == pytest.approx(0.10)


def test_load_stack_config_supports_main_league_a075_context_candres_mechpush_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        _a075_context_config(
            repo_root,
            "main_league_champion_hardneg_a075_context_candres_broad_conflictfilter_episodepref_mechpush_probe.yaml",
        )
    )

    assert stack.config.model is not None
    assert stack.config.model.opponent_context_adapter_train_only is True
    assert stack.config.model.opponent_context_adapter_lr_multiplier == pytest.approx(5000.0)
    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert structured_aux.paired_swing_loss_scope == "episode_mean"
    assert structured_aux.paired_swing_aux_updates == 16
    assert structured_aux.paired_swing_coef == pytest.approx(0.25)


def test_load_stack_config_supports_main_league_a075_context_candres_topother_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        _a075_context_config(
            repo_root,
            "main_league_champion_hardneg_a075_context_candres_broad_conflictfilter_topother_probe.yaml",
        )
    )

    assert stack.config.model is not None
    assert stack.config.model.opponent_context_adapter_train_only is True
    assert stack.config.model.opponent_context_adapter_lr_multiplier == pytest.approx(2500.0)
    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert structured_aux.paired_swing_conflict_filter == "current_state"
    assert structured_aux.paired_swing_loss_scope == "row"
    assert structured_aux.paired_swing_compare_to == "top_other"
    assert structured_aux.paired_swing_margin == pytest.approx(0.02)


def test_load_stack_config_supports_main_league_a075_context_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(_a075_context_config(repo_root, "main_league_champion_hardneg_a075_context_probe.yaml"))

    assert stack.config.model is not None
    assert stack.config.model.opponent_context_hidden_scale == pytest.approx(0.75)
    assert "policy_000001" in stack.config.model.opponent_context_eval_policy_ids
    assert "B4 HeuristicPublicControl" in stack.config.model.opponent_context_policy_ids
    assert "seed_c3aac2f9dc_policy_000004" in stack.config.model.opponent_context_policy_ids
    assert stack.config.training is not None
    assert stack.config.training.structured_aux.paired_swing_batch_episodes == 7


def test_load_stack_config_supports_main_league_a075_context_conflict_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        _a075_context_config(repo_root, "main_league_champion_hardneg_a075_context_conflict_probe.yaml")
    )

    assert stack.config.league is not None
    assert stack.config.league.pool.seed_snapshot_import_filter == "source_champions"
    assert stack.config.league.pool.seed_snapshot_champion_import == "source_champions"
    assert stack.config.model is not None
    assert stack.config.model.opponent_context_hidden_scale == pytest.approx(0.75)
    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert (
        structured_aux.paired_swing_dataset_path
        == "runs/paired_outcome_contrastive_a075_context_conflict_full_20260520/"
        "paired_outcome_contrastive_a075_context_conflict_full.npz"
    )
    assert structured_aux.paired_swing_coef == pytest.approx(0.04)
    assert structured_aux.paired_swing_batch_episodes == 11
    assert tuple(group.name for group in structured_aux.paired_swing_focus_groups) == (
        "a075_context_b2_preserve",
        "a075_context_p4_preserve",
        "a075_context_learned_repair",
    )
    assert tuple(group.fraction for group in structured_aux.paired_swing_focus_groups) == pytest.approx(
        (0.27, 0.18, 0.55)
    )
    assert structured_aux.paired_swing_focus_groups[0].source_labels == ("a075_loss_preserve_B2 HeuristicPublic",)


def test_load_stack_config_supports_main_league_a075_context_conflict_strong_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        _a075_context_config(repo_root, "main_league_champion_hardneg_a075_context_conflict_strong_probe.yaml")
    )

    assert stack.config.model is not None
    assert stack.config.model.opponent_context_hidden_scale == pytest.approx(1.50)
    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert structured_aux.paired_swing_aux_updates == 2
    assert structured_aux.paired_swing_coef == pytest.approx(0.08)
    assert structured_aux.paired_swing_batch_episodes == 11


def test_load_stack_config_supports_main_league_a075_context_trainable_conflict_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        _a075_context_config(repo_root, "main_league_champion_hardneg_a075_context_trainable_conflict_probe.yaml")
    )

    assert stack.config.model is not None
    assert stack.config.model.opponent_context_hidden_scale == pytest.approx(0.0)
    assert stack.config.model.opponent_context_trainable_hidden_scale == pytest.approx(1.0)
    assert stack.config.training is not None
    assert stack.config.training.structured_aux.paired_swing_dataset_path.endswith(
        "paired_outcome_contrastive_a075_context_conflict_full.npz"
    )


def test_load_stack_config_supports_main_league_a075_context_trainable_conflict_s64_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        _a075_context_config(repo_root, "main_league_champion_hardneg_a075_context_trainable_conflict_s64_probe.yaml")
    )

    assert stack.config.model is not None
    assert stack.config.model.opponent_context_hidden_scale == pytest.approx(0.0)
    assert stack.config.model.opponent_context_trainable_hidden_scale == pytest.approx(64.0)


def test_load_stack_config_supports_main_league_a075_context_trainable_lrmul_conflict_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        _a075_context_config(repo_root, "main_league_champion_hardneg_a075_context_trainable_lrmul_conflict_probe.yaml")
    )

    assert stack.config.model is not None
    assert stack.config.model.opponent_context_trainable_hidden_scale == pytest.approx(1.0)
    assert stack.config.model.opponent_context_adapter_lr_multiplier == pytest.approx(10000.0)


def test_load_stack_config_supports_main_league_a075_context_actionbias_conflict_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        _a075_context_config(repo_root, "main_league_champion_hardneg_a075_context_actionbias_conflict_probe.yaml")
    )

    assert stack.config.model is not None
    assert stack.config.model.opponent_context_trainable_hidden_scale == pytest.approx(0.0)
    assert stack.config.model.opponent_context_trainable_action_bias_scale == pytest.approx(1.0)
    assert stack.config.model.opponent_context_adapter_lr_multiplier == pytest.approx(1000.0)


def test_load_stack_config_supports_main_league_a075_context_actionbias_pair205_adapteronly_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        _a075_context_config(
            repo_root,
            "main_league_champion_hardneg_a075_context_actionbias_pair205_adapteronly_probe.yaml",
        )
    )

    assert stack.config.model is not None
    assert stack.config.model.opponent_context_trainable_hidden_scale == pytest.approx(0.0)
    assert stack.config.model.opponent_context_trainable_action_bias_scale == pytest.approx(1.0)
    assert stack.config.model.opponent_context_adapter_train_only is True
    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert (
        structured_aux.paired_swing_dataset_path == "runs/paired_outcome_contrastive_a075_pair205_only_20260520/"
        "paired_outcome_contrastive_a075_pair205_only.npz"
    )
    assert structured_aux.paired_swing_positive_action_source == "actions"
    assert structured_aux.paired_swing_negative_action_source == "teacher_action"
    assert structured_aux.paired_swing_batch_episodes == 7
    assert tuple(group.name for group in structured_aux.paired_swing_focus_groups) == (
        "pair205_fixed_preserve",
        "pair205_learned_repair",
    )


def test_load_stack_config_supports_main_league_a075_context_actionbias_pair205_strong_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        _a075_context_config(
            repo_root,
            "main_league_champion_hardneg_a075_context_actionbias_pair205_adapteronly_strong_probe.yaml",
        )
    )

    assert stack.config.model is not None
    assert stack.config.model.opponent_context_adapter_train_only is True
    assert stack.config.model.opponent_context_adapter_lr_multiplier == pytest.approx(10000.0)
    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert structured_aux.policy_anchor_coef == pytest.approx(0.0)
    assert structured_aux.trajectory_retention_coef == pytest.approx(0.0)
    assert structured_aux.paired_swing_aux_updates == 32
    assert structured_aux.paired_swing_coef == pytest.approx(0.40)


def test_load_stack_config_supports_main_league_a075_context_actionbias_pair205_minflip_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        _a075_context_config(
            repo_root,
            "main_league_champion_hardneg_a075_context_actionbias_pair205_adapteronly_minflip_probe.yaml",
        )
    )

    assert stack.config.model is not None
    assert stack.config.model.opponent_context_adapter_train_only is True
    assert stack.config.model.opponent_context_adapter_lr_multiplier == pytest.approx(2000.0)
    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert structured_aux.paired_swing_aux_updates == 8
    assert structured_aux.paired_swing_coef == pytest.approx(0.20)
    assert structured_aux.paired_swing_positive_action_source == "actions"
    assert structured_aux.trajectory_retention_coef == pytest.approx(0.0)


def test_load_stack_config_supports_main_league_a075_context_candidate_residual_pair205_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        _a075_context_config(
            repo_root,
            "main_league_champion_hardneg_a075_context_candidate_residual_pair205_adapteronly_minflip_probe.yaml",
        )
    )

    assert stack.config.model is not None
    assert stack.config.model.opponent_context_adapter_train_only is True
    assert stack.config.model.opponent_context_trainable_action_bias_scale == pytest.approx(0.0)
    assert stack.config.model.opponent_context_trainable_candidate_residual_scale == pytest.approx(1.0)
    assert stack.config.model.opponent_context_candidate_residual_width == 32
    assert stack.config.model.opponent_context_adapter_lr_multiplier == pytest.approx(2000.0)
    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert structured_aux.paired_swing_aux_updates == 8
    assert structured_aux.paired_swing_coef == pytest.approx(0.20)
    assert structured_aux.paired_swing_positive_action_source == "actions"


def test_load_stack_config_supports_main_league_a075_bilinear_candidate_residual_pair205_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        _a075_context_config(
            repo_root,
            "main_league_champion_hardneg_a075_context_candidate_residual_pair205_bilinear_adapteronly_probe.yaml",
        )
    )

    assert stack.config.model is not None
    assert stack.config.model.opponent_context_trainable_action_bias_scale == pytest.approx(0.0)
    assert stack.config.model.opponent_context_trainable_candidate_residual_scale == pytest.approx(1.0)
    assert stack.config.model.opponent_context_candidate_residual_mode == "bilinear"
    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert structured_aux.paired_swing_margin == pytest.approx(0.05)
    assert structured_aux.paired_swing_coef == pytest.approx(0.08)


def test_load_stack_config_supports_main_league_a075_actionid_candidate_residual_pair205_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        _a075_context_config(
            repo_root,
            "main_league_champion_hardneg_a075_context_candidate_residual_pair205_actionids_adapteronly_probe.yaml",
        )
    )

    assert stack.config.model is not None
    assert stack.config.model.opponent_context_trainable_action_bias_scale == pytest.approx(0.0)
    assert stack.config.model.opponent_context_trainable_candidate_residual_scale == pytest.approx(1.0)
    assert stack.config.model.opponent_context_candidate_residual_mode == "rich"
    assert stack.config.model.opponent_context_candidate_residual_action_ids == (104, 124)
    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert structured_aux.paired_swing_aux_updates == 16
    assert structured_aux.paired_swing_margin == pytest.approx(0.05)
    assert structured_aux.paired_swing_coef == pytest.approx(0.12)
    assert tuple(group.name for group in structured_aux.paired_swing_focus_groups) == (
        "pair205_b2_fixed_preserve",
        "pair205_policy0004_fixed_preserve",
        "pair205_learned_repair",
    )


def test_load_stack_config_supports_main_league_a075_rowbalanced_actionid_candidate_residual_pair205_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        _a075_context_config(
            repo_root,
            "main_league_champion_hardneg_a075_context_candidate_residual_pair205_actionids_rowbalanced_adapteronly_probe.yaml",
        )
    )

    assert stack.config.model is not None
    assert stack.config.model.opponent_context_candidate_residual_mode == "rich"
    assert stack.config.model.opponent_context_candidate_residual_action_ids == (104, 124)
    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert structured_aux.paired_swing_aux_updates == 16
    groups = structured_aux.paired_swing_focus_groups
    assert len(groups) == 7
    assert sum(group.fraction for group in groups) == pytest.approx(1.0)
    assert all(len(group.source_labels) == 1 for group in groups)


def test_load_stack_config_supports_main_league_a075_richbilinear_fullconflict_topother_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        _a075_context_config(
            repo_root,
            "main_league_champion_hardneg_a075_context_richbilinear_fullconflict_topother_adapteronly_probe.yaml",
        )
    )

    assert stack.config.model is not None
    assert stack.config.model.opponent_context_adapter_train_only is True
    assert stack.config.model.opponent_context_candidate_residual_mode == "rich_bilinear"
    assert stack.config.model.opponent_context_candidate_residual_width == 64
    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert (
        structured_aux.paired_swing_dataset_path
        == "runs/paired_outcome_contrastive_a075_context_conflict_full_20260520/"
        "paired_outcome_contrastive_a075_context_conflict_full.npz"
    )
    assert structured_aux.paired_swing_compare_to == "top_other"
    assert structured_aux.paired_swing_batch_episodes == 11
    assert tuple(group.name for group in structured_aux.paired_swing_focus_groups) == (
        "a075_context_b2_preserve",
        "a075_context_p4_preserve",
        "a075_context_learned_repair",
    )


def test_load_stack_config_supports_main_league_a075_actionid_richbilinear_fullconflict_topother_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        _a075_context_config(
            repo_root,
            "main_league_champion_hardneg_a075_context_richbilinear_fullconflict_actionids_topother_adapteronly_probe.yaml",
        )
    )

    assert stack.config.model is not None
    assert stack.config.model.opponent_context_candidate_residual_mode == "rich_bilinear"
    assert stack.config.model.opponent_context_candidate_residual_action_ids == (
        104,
        113,
        123,
        124,
        472,
        480,
        483,
        485,
        494,
        495,
    )
    assert stack.config.training is not None
    assert stack.config.training.structured_aux.paired_swing_compare_to == "top_other"


def test_load_stack_config_supports_main_league_a050_width128_preference_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        _a050_context_width128_config(
            repo_root,
            "main_league_champion_hardneg_a050_context_preference_width128_probe.yaml",
        )
    )

    assert stack.config.model is not None
    assert stack.config.model.opponent_context_adapter_train_only is True
    assert stack.config.model.opponent_context_trainable_candidate_residual_scale == pytest.approx(1.0)
    assert stack.config.model.opponent_context_candidate_residual_mode == "bilinear"
    assert stack.config.model.opponent_context_candidate_residual_width == 128
    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert structured_aux.paired_outcome_preference_group_balance is True
    assert structured_aux.paired_outcome_preference_aggregation == "mean"


def test_load_stack_config_supports_main_league_a050_width128_additive_preference_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        _a050_context_width128_config(
            repo_root,
            "main_league_champion_hardneg_a050_context_preference_width128_additive_probe.yaml",
        )
    )

    assert stack.config.model is not None
    assert stack.config.model.opponent_context_adapter_train_only is True
    assert stack.config.model.opponent_context_trainable_candidate_residual_scale == pytest.approx(1.0)
    assert stack.config.model.opponent_context_candidate_residual_mode == "additive"
    assert stack.config.model.opponent_context_candidate_residual_width == 128


def test_load_stack_config_supports_main_league_a050_width128_rich_preference_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        _a050_context_width128_config(
            repo_root,
            "main_league_champion_hardneg_a050_context_preference_width128_rich_probe.yaml",
        )
    )

    assert stack.config.model is not None
    assert stack.config.model.opponent_context_adapter_train_only is True
    assert stack.config.model.opponent_context_trainable_candidate_residual_scale == pytest.approx(1.0)
    assert stack.config.model.opponent_context_candidate_residual_mode == "rich"
    assert stack.config.model.opponent_context_candidate_residual_width == 128


def test_load_stack_config_supports_main_league_a050_width128_rich_exactaliases_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        _a050_context_width128_config(
            repo_root,
            "main_league_champion_hardneg_a050_context_preference_width128_rich_exactaliases_probe.yaml",
        )
    )

    assert stack.config.model is not None
    assert stack.config.model.opponent_context_candidate_residual_mode == "rich"
    assert stack.config.model.opponent_context_candidate_residual_width == 128
    assert "seed_b8c698d26a_seed_c3aac2f9dc_policy_000004" in stack.config.model.opponent_context_policy_ids
    assert "seed_c3aac2f9dc_policy_000004" in stack.config.model.opponent_context_policy_ids


def test_load_stack_config_supports_main_league_a050balanced_live_rowgate_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/hardneg_a050_a075_followup/main_league_champion_hardneg_a050balanced_live_rowgate_probe.yaml"
    )

    assert stack.config.league is not None
    sampling = stack.config.league.sampling
    assert sampling.champion_mix_fraction == pytest.approx(0.22)
    assert sampling.hard_negative_mix_fraction == pytest.approx(0.18)
    assert sampling.heuristic_public_variant_mix_fraction == pytest.approx(0.18)
    assert "seed_c3aac2f9dc_policy_000004" in sampling.hard_negative_focus_policy_ids
    assert sampling.hard_negative_focus_weight_multiplier == pytest.approx(2.5)
    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert structured_aux.trajectory_bc_every_updates == 0
    assert structured_aux.trajectory_bc_dataset_path == ""
    assert structured_aux.paired_swing_every_updates == 0
    assert structured_aux.paired_swing_dataset_path == ""
    assert structured_aux.paired_swing_coef == pytest.approx(0.0)
    assert structured_aux.policy_anchor_coef == pytest.approx(0.52)
    assert structured_aux.trajectory_retention_coef == pytest.approx(0.08)


def test_load_stack_config_supports_main_league_a050p2_live_learnedpush_rowgate_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/hardneg_a050p2_live/main_league_champion_hardneg_a050p2_live_learnedpush_rowgate_probe.yaml"
    )

    assert stack.config.league is not None
    sampling = stack.config.league.sampling
    assert sampling.champion_mix_fraction == pytest.approx(0.25)
    assert sampling.hard_negative_mix_fraction == pytest.approx(0.22)
    assert sampling.heuristic_public_variant_mix_fraction == pytest.approx(0.16)
    assert "seed_c3aac2f9dc_policy_000001" in sampling.hard_negative_focus_policy_ids
    assert "seed_c3aac2f9dc_policy_000004" in sampling.hard_negative_focus_policy_ids
    assert sampling.hard_negative_focus_weight_multiplier == pytest.approx(3.0)
    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert structured_aux.paired_swing_every_updates == 0
    assert structured_aux.paired_swing_dataset_path == ""
    assert structured_aux.paired_swing_coef == pytest.approx(0.0)
    assert structured_aux.policy_anchor_coef == pytest.approx(0.46)
    assert structured_aux.trajectory_retention_coef == pytest.approx(0.04)


def test_load_stack_config_supports_main_league_a050p2_live_rowdeficit_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/hardneg_a050p2_live/main_league_champion_hardneg_a050p2_live_rowdeficit_probe.yaml"
    )

    assert stack.config.league is not None
    sampling = stack.config.league.sampling
    assert sampling.champion_mix_fraction == pytest.approx(0.25)
    assert sampling.hard_negative_mix_fraction == pytest.approx(0.22)
    assert sampling.hard_negative_overlaps_champions is True
    assert sampling.hard_negative_focus_weight_multiplier == pytest.approx(3.0)
    assert sampling.row_deficit_policy_weights == (
        ("seed_c3aac2f9dc_main_league_selected", pytest.approx(1.5)),
        ("seed_c3aac2f9dc_policy_000001", pytest.approx(2.0)),
        ("seed_c3aac2f9dc_policy_000003", pytest.approx(2.0)),
        ("seed_c3aac2f9dc_policy_000004", pytest.approx(3.0)),
        ("seed_c3aac2f9dc_policy_000005", pytest.approx(2.0)),
    )
    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert structured_aux.paired_swing_every_updates == 0
    assert structured_aux.trajectory_bc_every_updates == 0


def test_load_stack_config_supports_main_league_a050p2_live_unlocked_rowdeficit_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/hardneg_a050p2_live/main_league_champion_hardneg_a050p2_live_unlocked_rowdeficit_probe.yaml"
    )

    assert stack.config.league is not None
    sampling = stack.config.league.sampling
    assert sampling.row_deficit_policy_weights == (
        ("seed_c3aac2f9dc_main_league_selected", pytest.approx(1.5)),
        ("seed_c3aac2f9dc_policy_000001", pytest.approx(2.0)),
        ("seed_c3aac2f9dc_policy_000003", pytest.approx(2.0)),
        ("seed_c3aac2f9dc_policy_000004", pytest.approx(3.0)),
        ("seed_c3aac2f9dc_policy_000005", pytest.approx(2.0)),
    )
    assert stack.config.training is not None
    assert stack.config.training.optimizer.learning_rate == pytest.approx(0.0000024)
    assert stack.config.training.exploration.entropy_coef == pytest.approx(0.0012)
    structured_aux = stack.config.training.structured_aux
    assert structured_aux.policy_anchor_coef == pytest.approx(0.18)
    assert structured_aux.policy_anchor_top_action_coef == pytest.approx(0.03)
    assert structured_aux.policy_anchor_temperature == pytest.approx(0.75)
    assert structured_aux.trajectory_retention_coef == pytest.approx(0.01)
    assert structured_aux.paired_swing_every_updates == 0
    assert structured_aux.trajectory_bc_every_updates == 0


def test_load_stack_config_supports_main_league_a050p2_live_unlocked_learned_recovery_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/hardneg_a050p2_live/main_league_champion_hardneg_a050p2_live_unlocked_learned_recovery_probe.yaml"
    )

    assert stack.config.league is not None
    sampling = stack.config.league.sampling
    assert sampling.noleague_baseline_reserved_envs_per_actor == 2
    assert sampling.noleague_baseline_mix_fraction == pytest.approx(0.20)
    assert sampling.heuristic_public_mix_fraction == pytest.approx(0.035)
    assert sampling.heuristic_public_variant_mix_fraction == pytest.approx(0.10)
    assert sampling.champion_mix_fraction == pytest.approx(0.34)
    assert sampling.hard_negative_mix_fraction == pytest.approx(0.32)
    assert sampling.hard_negative_min_samples == 3
    assert sampling.hard_negative_max_win_rate == pytest.approx(0.62)
    assert sampling.hard_negative_focus_weight_multiplier == pytest.approx(4.5)
    assert sampling.row_deficit_policy_weights == (
        ("seed_c3aac2f9dc_main_league_selected", pytest.approx(3.0)),
        ("seed_c3aac2f9dc_policy_000001", pytest.approx(4.0)),
        ("seed_c3aac2f9dc_policy_000003", pytest.approx(4.0)),
        ("seed_c3aac2f9dc_policy_000004", pytest.approx(4.0)),
        ("seed_c3aac2f9dc_policy_000005", pytest.approx(4.0)),
    )
    assert stack.config.training is not None
    assert stack.config.training.optimizer.learning_rate == pytest.approx(0.0000032)
    assert stack.config.training.exploration.entropy_coef == pytest.approx(0.0015)
    structured_aux = stack.config.training.structured_aux
    assert structured_aux.policy_anchor_coef == pytest.approx(0.06)
    assert structured_aux.policy_anchor_top_action_coef == pytest.approx(0.01)
    assert structured_aux.policy_anchor_temperature == pytest.approx(0.78)
    assert structured_aux.trajectory_retention_coef == pytest.approx(0.0)
    assert structured_aux.paired_swing_every_updates == 0
    assert structured_aux.trajectory_bc_every_updates == 0


def test_load_stack_config_supports_main_league_a075_context_recurrent_conflict_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        _a075_context_config(repo_root, "main_league_champion_hardneg_a075_context_recurrent_conflict_probe.yaml")
    )

    assert stack.config.model is not None
    assert stack.config.model.opponent_context_trainable_hidden_scale == pytest.approx(0.0)
    assert stack.config.model.opponent_context_trainable_recurrent_scale == pytest.approx(1.0)
    assert stack.config.model.opponent_context_adapter_lr_multiplier == pytest.approx(1000.0)


def test_load_stack_config_supports_main_league_a075_context_trainable_selectedpreserve_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        _a075_context_config(
            repo_root,
            "main_league_champion_hardneg_a075_context_trainable_conflict_selectedpreserve_probe.yaml",
        )
    )

    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert structured_aux.trajectory_bc_every_updates == 1
    assert structured_aux.trajectory_bc_batch_episodes == 5
    assert structured_aux.trajectory_bc_focus_groups[0].name == "ctx_u1_selected_learned_preserve"
    assert structured_aux.paired_swing_batch_episodes == 11


def test_load_stack_config_supports_main_league_a075_context_u1_nonconflict_b2_learnedpreserve_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        _a075_context_config(
            repo_root,
            "main_league_champion_hardneg_a075_context_u1_nonconflict_b2_learnedpreserve_probe.yaml",
        )
    )

    assert stack.config.model is not None
    assert stack.config.model.opponent_context_trainable_hidden_scale == pytest.approx(1.0)
    assert stack.config.training is not None
    structured_aux = stack.config.training.structured_aux
    assert (
        structured_aux.paired_swing_dataset_path
        == "runs/paired_outcome_contrastive_a075_nonconflict_confirm256_micro_20260520/"
        "paired_outcome_contrastive_a075_nonconflict_confirm256_micro.npz"
    )
    assert structured_aux.paired_swing_positive_action_source == "actions"
    assert structured_aux.paired_swing_negative_action_source == "teacher_action"
    assert structured_aux.paired_swing_coef == pytest.approx(0.02)
    assert structured_aux.paired_swing_batch_episodes == 2
    assert tuple(group.name for group in structured_aux.paired_swing_focus_groups) == ("b2_nonconflict_confirm256",)
    assert structured_aux.paired_swing_focus_groups[0].source_labels == (
        "a075_nonconflict_b2_pair70_preserve_B2 HeuristicPublic",
        "a075_nonconflict_b2_pair229_preserve_B2 HeuristicPublic",
    )
    assert structured_aux.trajectory_bc_every_updates == 1
    assert structured_aux.trajectory_bc_batch_episodes == 2
    assert structured_aux.trajectory_bc_teacher_action_coef == pytest.approx(0.03)
    assert structured_aux.trajectory_bc_teacher_same_family_action_coef == pytest.approx(0.10)
    assert structured_aux.trajectory_bc_teacher_same_family_action_margin_coef == pytest.approx(0.02)
    assert structured_aux.trajectory_bc_focus_groups[0].name == "learned_p1_main_p3_preserve"


def test_load_stack_config_supports_mainmoveguard_guided_b1_ablation() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/guided_teacher/public_teacher_passaware_mainmoveguard_mulliganguard_reward.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.mulligan_force_confirm_after_select is True
    assert stack.config.training.force_pass_over_main_move_only is True
    assert stack.config.training.teacher_public_heuristic_families[0] == "pass"
    assert stack.config.training.actor_heuristic_fraction == pytest.approx(0.0)
    assert stack.config.league is not None
    assert stack.config.league.enabled is False


def test_load_stack_config_supports_b2exact_guided_b1_ablation() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/guided_teacher/public_teacher_b2exact_mainmoveguard_mulliganguard_reward.yaml"
    )

    assert stack.config.experiment is not None
    assert stack.config.experiment.role == "ablation_guided"
    assert stack.config.training is not None
    assert stack.config.training.mulligan_force_confirm_after_select is True
    assert stack.config.training.force_pass_over_main_move_only is True
    assert stack.config.training.teacher_action_coef == pytest.approx(0.03)
    assert stack.config.training.teacher_same_family_action_coef == pytest.approx(0.12)
    assert stack.config.training.teacher_slot_coef == pytest.approx(0.08)
    assert stack.config.training.teacher_public_heuristic_coef == pytest.approx(0.10)
    assert stack.config.training.teacher_public_heuristic_temperature == pytest.approx(2.0)
    assert stack.config.training.teacher_public_heuristic_families == ()
    assert stack.config.training.teacher_public_heuristic_profiles == ("base",)
    assert stack.config.training.actor_heuristic_fraction == pytest.approx(0.0)
    assert stack.config.league is not None
    assert stack.config.league.enabled is False


def test_load_stack_config_supports_b2exact_lowentropy_guided_b1_ablation() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/guided_teacher/public_teacher_b2exact_lowentropy_mainmoveguard_mulliganguard_reward.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.entropy_coef == pytest.approx(0.003)
    assert stack.config.training.entropy_anneal_to == pytest.approx(0.0)
    assert stack.config.training.entropy_anneal_steps_updates == 75
    assert stack.config.training.teacher_same_family_action_coef == pytest.approx(0.12)
    assert stack.config.training.force_pass_over_main_move_only is True


def test_load_stack_config_supports_b2exact_argmaxdev_guided_b1_ablation() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/guided_teacher/public_teacher_b2exact_argmaxdev_mainmoveguard_mulliganguard_reward.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.teacher_same_family_action_coef == pytest.approx(0.12)
    assert stack.config.training.force_pass_over_main_move_only is True
    assert stack.config.evaluation is not None
    assert stack.config.evaluation.eval_sampling_algorithm == "model_argmax_pinned_v1"


def test_load_stack_config_supports_b2exact_lowentropy_argmaxdev_guided_b1_ablation() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/guided_teacher/public_teacher_b2exact_lowentropy_argmaxdev_mainmoveguard_mulliganguard_reward.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.entropy_coef == pytest.approx(0.003)
    assert stack.config.training.entropy_anneal_to == pytest.approx(0.0)
    assert stack.config.training.entropy_anneal_steps_updates == 75
    assert stack.config.training.teacher_same_family_action_coef == pytest.approx(0.12)
    assert stack.config.training.force_pass_over_main_move_only is True
    assert stack.config.evaluation is not None
    assert stack.config.evaluation.eval_sampling_algorithm == "model_argmax_pinned_v1"


def test_load_stack_config_supports_b2exact_attackguard_argmaxdev_guided_b1_ablation() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/guided_teacher/public_teacher_b2exact_argmaxdev_attackguard_mainmoveguard_mulliganguard_reward.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.mulligan_force_confirm_after_select is True
    assert stack.config.training.force_pass_over_main_move_only is True
    assert stack.config.training.force_attack_over_pass_when_attack_legal is True
    assert stack.config.training.teacher_same_family_action_coef == pytest.approx(0.12)
    assert stack.config.evaluation is not None
    assert stack.config.evaluation.eval_sampling_algorithm == "model_argmax_pinned_v1"


def test_load_stack_config_supports_b2exact_sharp_attackguard_argmaxdev_guided_b1_ablation() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/guided_teacher/public_teacher_b2exact_sharp_argmaxdev_attackguard_mainmoveguard_mulliganguard_reward.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.force_attack_over_pass_when_attack_legal is True
    assert stack.config.training.teacher_action_coef == pytest.approx(0.08)
    assert stack.config.training.teacher_same_family_action_coef == pytest.approx(0.25)
    assert stack.config.training.teacher_public_heuristic_coef == pytest.approx(0.20)
    assert stack.config.training.teacher_public_heuristic_final_coef == pytest.approx(0.08)
    assert stack.config.training.teacher_public_heuristic_end_updates == 300
    assert stack.config.training.teacher_public_heuristic_temperature == pytest.approx(0.75)
    assert stack.config.evaluation is not None
    assert stack.config.evaluation.eval_sampling_algorithm == "model_argmax_pinned_v1"


def test_load_stack_config_supports_b2exact_margin_attackguard_argmaxdev_guided_b1_ablation() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/guided_teacher/public_teacher_b2exact_margin_argmaxdev_attackguard_mainmoveguard_mulliganguard_reward.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.force_attack_over_pass_when_attack_legal is True
    assert stack.config.training.teacher_action_margin_coef == pytest.approx(0.08)
    assert stack.config.training.teacher_action_margin == pytest.approx(0.75)
    assert stack.config.training.teacher_action_coef == pytest.approx(0.03)
    assert stack.config.training.teacher_public_heuristic_temperature == pytest.approx(2.0)
    assert stack.config.evaluation is not None
    assert stack.config.evaluation.eval_sampling_algorithm == "model_argmax_pinned_v1"


def test_load_stack_config_supports_b2exact_margin_multianchor_guided_b1_ablation() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / (
            "configs/thesis/_shared/guided_teacher/"
            "public_teacher_b2exact_margin_multianchor_argmaxdev_attackguard_mainmoveguard_mulliganguard_reward.yaml"
        )
    )

    assert stack.config.training is not None
    assert stack.config.training.teacher_action_margin_coef == pytest.approx(0.08)
    assert stack.config.training.teacher_action_margin == pytest.approx(0.75)
    assert stack.config.league is not None
    assert stack.config.league.promotion_anchor_set_v1.required == ("B0 RandomLegal",)
    assert stack.config.league.promotion_anchor_set_v1.optional_if_available == (
        "B1 NoLeague baseline",
        "B2 HeuristicPublic",
        "B3 HeuristicPublicAggro",
        "B4 HeuristicPublicControl",
    )
    assert stack.config.evaluation is not None
    assert stack.config.evaluation.final_policy_set_selection.fixed_anchor_set_v1.required == ("B0 RandomLegal",)
    assert stack.config.evaluation.final_policy_set_selection.fixed_anchor_set_v1.optional_if_available == (
        "B1 NoLeague baseline",
        "B2 HeuristicPublic",
        "B3 HeuristicPublicAggro",
        "B4 HeuristicPublicControl",
    )


def test_load_stack_config_supports_b2exact_margin_multianchor_teacherfade_guided_b1_ablation() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / (
            "configs/thesis/_shared/guided_teacher/"
            "public_teacher_b2exact_margin_multianchor_teacherfade_argmaxdev_attackguard_mainmoveguard_mulliganguard_reward.yaml"
        )
    )

    assert stack.config.training is not None
    assert stack.config.training.teacher_action_margin_coef == pytest.approx(0.08)
    assert stack.config.training.teacher_public_heuristic_end_updates == 75
    assert stack.config.training.teacher_public_heuristic_final_coef == pytest.approx(0.0)
    assert stack.config.training.teacher_supervised_start_updates == 45
    assert stack.config.training.teacher_supervised_end_updates == 75
    assert stack.config.training.teacher_supervised_final_scale == pytest.approx(0.25)
    assert stack.config.league is not None
    assert stack.config.league.promotion_anchor_set_v1.optional_if_available == (
        "B1 NoLeague baseline",
        "B2 HeuristicPublic",
        "B3 HeuristicPublicAggro",
        "B4 HeuristicPublicControl",
    )


def test_load_stack_config_supports_b2exact_margin_multianchor_latefloor_guided_b1_ablation() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / (
            "configs/thesis/_shared/guided_teacher/"
            "public_teacher_b2exact_margin_multianchor_latefloor_argmaxdev_attackguard_mainmoveguard_mulliganguard_reward.yaml"
        )
    )

    assert stack.config.training is not None
    assert stack.config.training.teacher_action_margin_coef == pytest.approx(0.08)
    assert stack.config.training.teacher_public_heuristic_end_updates == 100
    assert stack.config.training.teacher_public_heuristic_final_coef == pytest.approx(0.02)
    assert stack.config.training.teacher_supervised_start_updates == 50
    assert stack.config.training.teacher_supervised_end_updates == 100
    assert stack.config.training.teacher_supervised_final_scale == pytest.approx(0.50)
    assert stack.config.league is not None
    assert stack.config.league.promotion_anchor_set_v1.optional_if_available == (
        "B1 NoLeague baseline",
        "B2 HeuristicPublic",
        "B3 HeuristicPublicAggro",
        "B4 HeuristicPublicControl",
    )


def test_load_stack_config_supports_b2exact_margin_multianchor_hold75_guided_b1_ablation() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / (
            "configs/thesis/_shared/guided_teacher/"
            "public_teacher_b2exact_margin_multianchor_hold75_argmaxdev_attackguard_mainmoveguard_mulliganguard_reward.yaml"
        )
    )

    assert stack.config.training is not None
    assert stack.config.training.teacher_action_margin_coef == pytest.approx(0.08)
    assert stack.config.training.teacher_public_heuristic_end_updates == 75
    assert stack.config.training.teacher_public_heuristic_final_coef == pytest.approx(0.04)
    assert stack.config.training.teacher_supervised_start_updates == 50
    assert stack.config.training.teacher_supervised_end_updates == 75
    assert stack.config.training.teacher_supervised_final_scale == pytest.approx(0.75)
    assert stack.config.league is not None
    assert stack.config.league.promotion_anchor_set_v1.optional_if_available == (
        "B1 NoLeague baseline",
        "B2 HeuristicPublic",
        "B3 HeuristicPublicAggro",
        "B4 HeuristicPublicControl",
    )


def test_load_stack_config_supports_hold75_mulligan02_samefamilymargin_guided_b1_ablation() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / (
            "configs/thesis/_shared/guided_teacher/"
            "public_teacher_b2exact_margin_multianchor_hold75_mulligan02_samefamilymargin_argmaxdev_attackguard_mainmoveguard_mulliganguard_reward.yaml"
        )
    )

    assert stack.config.rewards is not None
    assert stack.config.training is not None
    assert stack.config.rewards.shaping.mulligan_select_with_confirm_penalty == pytest.approx(0.02)
    assert stack.config.training.teacher_same_family_action_margin_coef == pytest.approx(0.04)
    assert stack.config.training.teacher_same_family_action_margin == pytest.approx(0.50)
    assert stack.config.training.teacher_supervised_final_scale == pytest.approx(0.75)
    assert stack.config.training.teacher_public_heuristic_final_coef == pytest.approx(0.04)


def test_load_stack_config_supports_first_class_guided_b1_league_seed() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(repo_root / "configs/thesis/b1_guided_seed.yaml")

    assert stack.config.experiment is not None
    assert stack.config.experiment.role == "guided_league_seed"
    assert stack.config.model is not None
    assert stack.config.model.gru_hidden_size == 64
    assert stack.config.model.encoder_mlp_width == 64
    assert stack.config.training is not None
    assert stack.config.training.teacher_aux_mode == "always"
    assert stack.config.training.structured_aux_enabled is True
    assert stack.config.training.entropy_coef == pytest.approx(0.003)
    assert stack.config.training.entropy_anneal_to == pytest.approx(0.0)
    assert stack.config.training.teacher_public_heuristic_final_coef == pytest.approx(0.10)
    assert stack.config.training.teacher_public_nonpass_over_pass_coef == pytest.approx(0.02)
    assert stack.config.training.teacher_exact_action_families == (
        "attack",
        "main_move",
        "encore_pay",
        "encore_decline",
        "mulligan_confirm",
        "pass",
    )
    assert stack.config.training.actor_heuristic_fraction == pytest.approx(0.0)
    assert stack.config.league is not None
    assert stack.config.league.enabled is False
    assert stack.config.league.promotion_anchor_set_v1.optional_if_available == (
        "B1 NoLeague baseline",
        "B2 HeuristicPublic",
        "B3 HeuristicPublicAggro",
        "B4 HeuristicPublicControl",
    )
    assert stack.config.environment is not None
    assert stack.config.environment.deck_pool == ("preset:main_deck_5hy_yotsuba_v1",)
    assert stack.config.environment.opponent_deck_pool == ("preset:main_deck_5hy_yotsuba_v1",)


def test_load_stack_config_supports_b2exact_margin_multianchor_profiles_guided_b1_ablation() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / (
            "configs/thesis/_shared/guided_teacher/"
            "public_teacher_b2exact_margin_multianchor_profiles_argmaxdev_attackguard_mainmoveguard_mulliganguard_reward.yaml"
        )
    )

    assert stack.config.training is not None
    assert stack.config.training.teacher_action_margin_coef == pytest.approx(0.08)
    assert stack.config.training.teacher_public_heuristic_profiles == ("base", "aggressive", "control")
    assert stack.config.training.teacher_public_heuristic_profile_mode == "cycle"
    assert stack.config.training.teacher_public_heuristic_profiles_end_updates == 150
    assert stack.config.league is not None
    assert stack.config.league.promotion_anchor_set_v1.optional_if_available == (
        "B1 NoLeague baseline",
        "B2 HeuristicPublic",
        "B3 HeuristicPublicAggro",
        "B4 HeuristicPublicControl",
    )


def test_load_stack_config_supports_factorized_profiles_guided_b1_ablation() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / (
            "configs/thesis/_shared/guided_teacher/"
            "public_teacher_b2exact_margin_multianchor_hold75_playstrong_factorized_profiles_argmaxdev_attackguard_mainmoveguard_mulliganguard_reward.yaml"
        )
    )

    assert stack.config.model is not None
    assert stack.config.model.structured_policy_contract == "factorized_v1"
    assert stack.config.training is not None
    assert stack.config.training.actor_sampling_temperature == pytest.approx(0.75)
    assert stack.config.training.teacher_action_coef == pytest.approx(0.08)
    assert stack.config.training.teacher_same_family_action_coef == pytest.approx(0.30)
    assert stack.config.training.teacher_action_margin_coef == pytest.approx(0.10)
    assert stack.config.training.teacher_same_family_action_margin_coef == pytest.approx(0.08)
    assert stack.config.training.teacher_public_heuristic_profiles == ("base", "aggressive", "control")
    assert stack.config.training.teacher_public_heuristic_profile_mode == "cycle"
    assert stack.config.training.teacher_public_heuristic_profiles_end_updates == 150


def test_load_stack_config_supports_factorized_publicmix_guided_b1_ablation() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / (
            "configs/thesis/_shared/guided_teacher/"
            "public_teacher_b2exact_margin_multianchor_hold75_playstrong_factorized_publicmix_argmaxdev_attackguard_mainmoveguard_mulliganguard_reward.yaml"
        )
    )

    assert stack.config.model is not None
    assert stack.config.model.structured_policy_contract == "factorized_v1"
    assert stack.config.training is not None
    assert stack.config.training.teacher_action_coef == pytest.approx(0.08)
    assert stack.config.training.teacher_same_family_action_coef == pytest.approx(0.30)
    assert stack.config.training.teacher_public_heuristic_profiles == ("base", "aggressive", "control")
    assert stack.config.training.teacher_public_heuristic_profile_mode == "mixture"
    assert stack.config.training.teacher_public_heuristic_profiles_end_updates == 150


def test_load_stack_config_supports_trajbc_anchor_nopublic_continuation() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root / "configs/thesis/main_league_guided_bootstrap_selected_trajbc_b4win_anchor_nopublic.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.teacher_public_heuristic_coef == pytest.approx(0.0)
    assert stack.config.training.teacher_public_heuristic_final_coef == pytest.approx(0.0)
    assert stack.config.training.teacher_public_heuristic_end_updates == 0
    assert stack.config.training.teacher_supervised_end_updates == 0
    assert stack.config.training.teacher_supervised_final_scale == pytest.approx(0.0)
    assert stack.config.training.policy_anchor_coef == pytest.approx(0.8)
    assert stack.config.training.policy_anchor_top_action_coef == pytest.approx(0.06)
    assert stack.config.training.policy_anchor_temperature == pytest.approx(0.5)
    assert stack.config.training.trajectory_bc_enabled is True
    assert stack.config.training.trajectory_bc_dataset_path.endswith("trajectory_bc_b4_win_full.npz")
    assert stack.config.league is not None
    assert stack.config.league.pool.seed_snapshot_champion_import == "pinned"


def test_load_stack_config_supports_trajbc_anchor_nopublic_stability_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root / "configs/thesis/main_league_guided_bootstrap_selected_trajbc_b4win_anchor_nopublic_stability.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.learning_rate == pytest.approx(0.00005)
    assert stack.config.training.actor_sampling_temperature == pytest.approx(1.0)
    assert stack.config.training.policy_anchor_coef == pytest.approx(0.8)
    assert stack.config.training.teacher_public_heuristic_coef == pytest.approx(0.0)
    assert stack.config.training.trajectory_bc_enabled is True


def test_load_stack_config_supports_direct_b2b3b4_trajbc_anchor_nopublic_continuation() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root / "configs/thesis/main_league_guided_bootstrap_selected_trajbc_direct_b2b3b4_anchor_nopublic.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.teacher_public_heuristic_coef == pytest.approx(0.0)
    assert stack.config.training.teacher_supervised_end_updates == 0
    assert stack.config.training.policy_anchor_coef == pytest.approx(0.8)
    assert stack.config.training.actor_reload_interval_updates == 1
    assert stack.config.training.trajectory_bc_enabled is True
    assert stack.config.training.trajectory_bc_dataset_path.endswith("trajectory_bc_direct_b2_b3_b4_win_64.npz")
    assert stack.config.league is not None
    assert stack.config.league.pool.seed_snapshot_import_filter == "pinned"
    assert stack.config.league.sampling.noleague_baseline_mix_fraction == pytest.approx(0.15)
    assert stack.config.league.sampling.noleague_baseline_reserved_envs_per_actor == 1
    assert stack.config.league.promotion_anchor_set_v1.required == (
        "B0 RandomLegal",
        "B1 NoLeague baseline",
        "B2 HeuristicPublic",
        "B3 HeuristicPublicAggro",
        "B4 HeuristicPublicControl",
    )


def test_load_stack_config_supports_b2exact_noexact_attackguard_argmaxdev_guided_b1_ablation() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/guided_teacher/public_teacher_b2exact_noexact_argmaxdev_attackguard_mainmoveguard_mulliganguard_reward.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.force_attack_over_pass_when_attack_legal is True
    assert stack.config.training.teacher_action_coef == pytest.approx(0.0)
    assert stack.config.training.teacher_same_family_action_coef == pytest.approx(0.0)
    assert stack.config.training.teacher_action_margin_coef == pytest.approx(0.0)
    assert stack.config.training.teacher_family_coef == pytest.approx(0.03)
    assert stack.config.evaluation is not None
    assert stack.config.evaluation.eval_sampling_algorithm == "model_argmax_pinned_v1"


def test_load_stack_config_supports_b2exact_filteredexact_attackguard_argmaxdev_guided_b1_ablation() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/guided_teacher/public_teacher_b2exact_filteredexact_argmaxdev_attackguard_mainmoveguard_mulliganguard_reward.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.force_attack_over_pass_when_attack_legal is True
    assert stack.config.training.teacher_action_coef == pytest.approx(0.03)
    assert stack.config.training.teacher_same_family_action_coef == pytest.approx(0.12)
    assert stack.config.training.teacher_exact_action_families == (
        "attack",
        "main_move",
        "encore_pay",
        "encore_decline",
        "mulligan_confirm",
        "pass",
    )
    assert stack.config.evaluation is not None
    assert stack.config.evaluation.eval_sampling_algorithm == "model_argmax_pinned_v1"


def test_load_stack_config_supports_b2exact_filteredexact_constpublic_guided_b1_ablation() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/guided_teacher/public_teacher_b2exact_filteredexact_constpublic_argmaxdev_attackguard_mainmoveguard_mulliganguard_reward.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.teacher_exact_action_families == (
        "attack",
        "main_move",
        "encore_pay",
        "encore_decline",
        "mulligan_confirm",
        "pass",
    )
    assert stack.config.training.teacher_public_heuristic_coef == pytest.approx(0.10)
    assert stack.config.training.teacher_public_heuristic_final_coef == pytest.approx(0.10)
    assert stack.config.training.teacher_public_heuristic_end_updates == 300
    assert stack.config.training.teacher_public_heuristic_profiles_end_updates == 300


def test_load_stack_config_supports_b2exact_filteredexact_constpublic_antipass_guided_b1_ablation() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/guided_teacher/public_teacher_b2exact_filteredexact_constpublic_antipass_argmaxdev_attackguard_mainmoveguard_mulliganguard_reward.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.teacher_exact_action_families == (
        "attack",
        "main_move",
        "encore_pay",
        "encore_decline",
        "mulligan_confirm",
        "pass",
    )
    assert stack.config.training.teacher_public_heuristic_final_coef == pytest.approx(0.10)
    assert stack.config.training.teacher_public_nonpass_over_pass_coef == pytest.approx(0.08)
    assert stack.config.training.teacher_public_nonpass_over_pass_margin == pytest.approx(0.50)


def test_load_stack_config_supports_b2exact_filteredexact_constpublic_light_antipass_guided_b1_ablation() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/guided_teacher/public_teacher_b2exact_filteredexact_constpublic_antipass02_argmaxdev_attackguard_mainmoveguard_mulliganguard_reward.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.teacher_public_heuristic_final_coef == pytest.approx(0.10)
    assert stack.config.training.teacher_public_nonpass_over_pass_coef == pytest.approx(0.02)
    assert stack.config.training.teacher_public_nonpass_over_pass_margin == pytest.approx(0.50)


def test_load_stack_config_supports_b2exact_filteredexact_constpublic_light_antipass_lowentropy_guided_b1_ablation() -> (
    None
):
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/guided_teacher/public_teacher_b2exact_filteredexact_constpublic_antipass02_lowentropy_argmaxdev_attackguard_mainmoveguard_mulliganguard_reward.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.entropy_coef == pytest.approx(0.003)
    assert stack.config.training.entropy_anneal_to == pytest.approx(0.0)
    assert stack.config.training.entropy_anneal_steps_updates == 75
    assert stack.config.training.teacher_public_nonpass_over_pass_coef == pytest.approx(0.02)


def test_load_stack_config_supports_filteredexact_lowentropy_multianchor_guided_b1_ablation() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / (
            "configs/thesis/_shared/guided_teacher/"
            "public_teacher_b2exact_filteredexact_constpublic_antipass02_lowentropy_multianchor_argmaxdev_attackguard_mainmoveguard_mulliganguard_reward.yaml"
        )
    )

    assert stack.config.training is not None
    assert stack.config.training.entropy_coef == pytest.approx(0.003)
    assert stack.config.training.teacher_exact_action_families == (
        "attack",
        "main_move",
        "encore_pay",
        "encore_decline",
        "mulligan_confirm",
        "pass",
    )
    assert stack.config.training.teacher_public_heuristic_final_coef == pytest.approx(0.10)
    assert stack.config.league is not None
    assert stack.config.league.promotion_anchor_set_v1.optional_if_available == (
        "B1 NoLeague baseline",
        "B2 HeuristicPublic",
        "B3 HeuristicPublicAggro",
        "B4 HeuristicPublicControl",
    )


def test_load_stack_config_supports_b2exact_filteredexact_constpublic_light_antipass_lowentropy_groupstrong_guided_b1_ablation() -> (
    None
):
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/guided_teacher/public_teacher_b2exact_filteredexact_constpublic_antipass02_lowentropy_groupstrong_argmaxdev_attackguard_mainmoveguard_mulliganguard_reward.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.teacher_family_coef == pytest.approx(0.06)
    assert stack.config.training.teacher_slot_coef == pytest.approx(0.16)
    assert stack.config.training.teacher_move_source_coef == pytest.approx(0.06)
    assert stack.config.training.teacher_attack_type_coef == pytest.approx(0.06)
    assert stack.config.training.teacher_public_nonpass_over_pass_coef == pytest.approx(0.02)


def test_load_stack_config_supports_b2exact_filteredexact_constpublic_light_antipass_lowentropy_samefamilymargin_guided_b1_ablation() -> (
    None
):
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/guided_teacher/public_teacher_b2exact_filteredexact_constpublic_antipass02_lowentropy_samefamilymargin_argmaxdev_attackguard_mainmoveguard_mulliganguard_reward.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.teacher_same_family_action_margin_coef == pytest.approx(0.04)
    assert stack.config.training.teacher_same_family_action_margin == pytest.approx(0.50)
    assert stack.config.training.teacher_public_nonpass_over_pass_coef == pytest.approx(0.02)
    assert stack.config.training.teacher_exact_action_families == (
        "attack",
        "main_move",
        "encore_pay",
        "encore_decline",
        "mulligan_confirm",
        "pass",
    )


def test_load_stack_config_supports_b2exact_filteredexact_constpublic_light_antipass_lowentropy_choiceexactmargin_guided_b1_ablation() -> (
    None
):
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/guided_teacher/public_teacher_b2exact_filteredexact_constpublic_antipass02_lowentropy_choiceexactmargin_argmaxdev_attackguard_mainmoveguard_mulliganguard_reward.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.teacher_same_family_action_margin_coef == pytest.approx(0.04)
    assert stack.config.training.teacher_same_family_action_margin == pytest.approx(0.50)
    assert stack.config.training.teacher_exact_action_families == (
        "attack",
        "main_move",
        "encore_pay",
        "encore_decline",
        "mulligan_confirm",
        "pass",
        "choice_select",
        "climax_play",
    )


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
    assert stack.config.model.candidate_scoring_chunk_size == 65536
    assert stack.config.model.cuda_learner_candidate_scoring_chunk_size == 262144
    assert stack.config.model.public_heuristic_logit_bias_scale == pytest.approx(0.0)
    assert stack.config.model.public_heuristic_actor_logit_bias_scale == pytest.approx(-1.0)
    assert stack.config.training is not None
    assert stack.config.training.algorithm == "structured_v2"
    assert stack.seed_sets["dev_eval"] == repo_root / "configs/seeds/local_dev_eval_seeds.txt"


def test_load_stack_config_supports_typed_structured_v2_preset() -> None:
    repo_root = _repo_root()
    stack = load_stack_config(repo_root / "configs/presets/typed_structured_v2.yaml")

    assert stack.config.model is not None
    assert stack.config.model.encoder_kind == "structured_v2"
    assert stack.config.model.candidate_scoring_chunk_size == 65536
    assert stack.config.model.cuda_learner_candidate_scoring_chunk_size == 1048576
    assert stack.config.training is not None
    assert stack.config.training.algorithm == "impala_vtrace_structured_v1"
    assert stack.config.training.profile_timers is False
    assert stack.config.training.torch_profiler is False
    assert stack.config.training.compile_actor_inference is False
    assert stack.config.training.structured_metrics_mode == "sampled"
    assert stack.config.training.teacher_aux_mode == "always"
    assert stack.config.training.fixed_opponent_backend == "python_scalar"
    assert stack.config.training.actor_policy_backend == "model"
    assert stack.config.training.actor_heuristic_fraction == pytest.approx(1.0)
    assert stack.config.training.heuristic_actor_hidden_state_tracking is True
    assert stack.config.training.structured_aux_enabled is True
    assert stack.config.training.teacher_family_coef == pytest.approx(0.20)
    assert stack.config.training.teacher_slot_coef == pytest.approx(0.10)
    assert stack.config.training.teacher_attack_type_coef == pytest.approx(0.05)
    assert stack.config.training.teacher_action_coef == pytest.approx(0.10)


def test_load_stack_config_supports_local_learning_liveleague_preset() -> None:
    repo_root = _repo_root()
    stack = load_stack_config(repo_root / "configs/presets/structured_acceptance_local_learning_liveleague.yaml")

    assert stack.config.system is not None
    assert stack.config.system.collection_backend == "central"
    assert stack.config.training is not None
    assert stack.config.training.actor_policy_backend == "heuristic_public"
    assert stack.config.training.actor_heuristic_end_updates == 5
    assert stack.config.training.snapshot_interval_updates == 5
    assert stack.config.training.actor_reload_interval_updates == 10
    assert stack.config.league is not None
    assert stack.config.league.warmup.first_updates == 5
    assert stack.config.league.warmup.initial_window_episodes == 512


def test_load_stack_config_supports_local_learning_liveleague_terminal_preset() -> None:
    repo_root = _repo_root()
    stack = load_stack_config(
        repo_root / "configs/presets/structured_acceptance_local_learning_liveleague_terminal.yaml"
    )

    assert stack.config.rewards is not None
    assert stack.config.rewards.objective == "terminal_only_pm1"
    assert stack.config.rewards.gamma == pytest.approx(1.0)
    assert stack.config.rewards.shaping.enable_damage_shaping is False
    assert stack.config.rewards.shaping.damage_reward == pytest.approx(0.0)
    assert stack.config.rewards.truncation.reward == pytest.approx(0.0)
    assert stack.config.rewards.truncation.bootstrap_value is True
    training = _require_training_config(stack.config.training)
    assert training.teacher_same_family_action_coef == pytest.approx(0.0)
    assert training.teacher_public_heuristic_coef == pytest.approx(0.1)
    assert training.teacher_public_heuristic_temperature == pytest.approx(32.0)
    assert training.structured_warmstart_enabled is True
    assert training.structured_warmstart.updates == 1
    assert training.structured_warmstart.teacher_family_coef == pytest.approx(0.75)
    assert training.structured_warmstart.teacher_slot_coef == pytest.approx(0.35)
    assert training.structured_warmstart.teacher_attack_type_coef == pytest.approx(0.20)
    assert training.structured_warmstart.teacher_action_coef == pytest.approx(0.50)
    assert training.structured_warmstart.teacher_same_family_action_coef == pytest.approx(0.0)
    assert training.structured_warmstart.teacher_public_heuristic_coef == pytest.approx(0.5)
    assert training.structured_warmstart.teacher_public_heuristic_temperature == pytest.approx(32.0)
    assert stack.seed_sets["dev_eval"] == repo_root / "configs/seeds/local_dev_eval_seeds.txt"


def test_load_stack_config_supports_local_learning_thesis_eval_preset() -> None:
    repo_root = _repo_root()
    stack = load_stack_config(repo_root / "configs/presets/structured_acceptance_local_learning_thesis_eval.yaml")

    assert stack.config.league is not None
    assert stack.config.league.promotion_anchor_set_v1.optional_if_available == (
        "B2 HeuristicPublic",
        "Previous champion snapshot",
        "Previous recent snapshot",
    )
    assert stack.config.evaluation is not None
    assert stack.config.evaluation.final_policy_set_selection.fixed_anchor_set_v1.optional_if_available == (
        "B2 HeuristicPublic",
        "Previous champion snapshot",
        "Previous recent snapshot",
    )


def test_thesis_main_and_b1_use_medium64_model_surface() -> None:
    repo_root = _repo_root()

    main = load_stack_config(repo_root / "configs/thesis/main_league.yaml")
    b1 = load_stack_config(repo_root / "configs/thesis/b1_noleague.yaml")
    auto_gpu = load_stack_config(repo_root / "configs/thesis/main_league_auto_gpu.yaml")
    final_eval = load_stack_config(repo_root / "configs/thesis/final_eval.yaml")
    final_eval_no_replay = load_stack_config(repo_root / "configs/thesis/final_eval_no_replay.yaml")

    for stack in (main, b1, auto_gpu, final_eval, final_eval_no_replay):
        assert stack.config.model is not None
        assert stack.config.model.gru_hidden_size == 64
        assert stack.config.model.encoder_mlp_width == 64
        assert stack.config.model.typed_feature_width == 16
        assert stack.config.model.public_heuristic_logit_bias_scale == pytest.approx(0.0)
        assert stack.config.model.public_heuristic_actor_logit_bias_scale == pytest.approx(0.0)
        assert stack.config.training is not None
        assert stack.config.training.actor_policy_backend == "model"
        assert stack.config.training.actor_heuristic_fraction == pytest.approx(0.0)
        assert stack.config.training.actor_heuristic_final_fraction == pytest.approx(0.0)
        assert stack.config.training.train_on_heuristic_actor_rows is False
        assert stack.config.training.structured_warmstart_enabled is False
        assert stack.config.training.structured_warmstart.updates == 0
        assert stack.config.training.teacher_public_heuristic_coef == pytest.approx(0.0)
        assert stack.config.training.structured_warmstart.teacher_public_heuristic_coef == pytest.approx(0.0)

    for stack in (main, b1, auto_gpu):
        assert stack.config.training is not None
        assert stack.config.training.teacher_aux_mode == "off"
        assert stack.config.training.structured_aux.enabled is False
        assert stack.config.training.teacher_family_coef == pytest.approx(0.0)
        assert stack.config.training.teacher_action_coef == pytest.approx(0.0)
    for stack in (final_eval, final_eval_no_replay):
        assert stack.config.training is not None
        assert stack.config.training.teacher_aux_mode == "always"
        assert stack.config.training.structured_aux.enabled is True

    assert b1.config.league is not None
    assert b1.config.league.enabled is False
    assert b1.config.league.sampling.heuristic_public_mix_fraction == pytest.approx(0.0)
    assert b1.config.league.sampling.heuristic_public_variant_mix_fraction == pytest.approx(0.0)
    assert b1.config.league.sampling.noleague_baseline_mix_fraction == pytest.approx(0.0)
    assert b1.config.evaluation.periodic_dev_eval_interval_updates == 25
    assert b1.config.training.entropy_coef == pytest.approx(0.01)
    assert b1.config.training.mulligan_force_confirm_after_select is True
    assert b1.config.training.force_pass_over_main_move_only is True
    assert b1.config.training.force_attack_over_pass_when_attack_legal is True
    assert b1.config.rewards.shaping.enable_damage_shaping is True
    assert b1.config.rewards.shaping.damage_reward == pytest.approx(0.05)
    assert b1.config.rewards.shaping.level_reward == pytest.approx(0.05)
    assert b1.config.rewards.shaping.board_reward == pytest.approx(0.02)
    assert b1.config.rewards.shaping.no_progress_penalty == pytest.approx(0.005)
    assert b1.config.rewards.shaping.pass_with_nonpass_penalty == pytest.approx(0.02)
    assert b1.config.league.promotion_anchor_set_v1.required == ("B0 RandomLegal",)
    assert b1.config.league.promotion_anchor_set_v1.optional_if_available == (
        "B2 HeuristicPublic",
        "B3 HeuristicPublicAggro",
        "B4 HeuristicPublicControl",
    )
    assert b1.config.evaluation.final_policy_set_selection.fixed_anchor_set_v1.required == ("B0 RandomLegal",)
    assert b1.config.evaluation.final_policy_set_selection.fixed_anchor_set_v1.optional_if_available == (
        "B2 HeuristicPublic",
        "B3 HeuristicPublicAggro",
        "B4 HeuristicPublicControl",
    )
    assert main.config.league is not None
    assert main.config.league.sampling.heuristic_public_mix_fraction == pytest.approx(0.2)
    assert main.config.league.sampling.noleague_baseline_mix_fraction == pytest.approx(0.15)
    assert main.config.training.diverse_opponent_actor_count == 999
    assert main.config.training.diverse_model_actor_count == 999
    assert main.config.training.diverse_opponent_batch_fraction == pytest.approx(1.0)
    assert auto_gpu.config.system is not None
    assert auto_gpu.config.system.learner_device == "cuda:auto"
    assert auto_gpu.config.system.actor_device == "cuda:auto"
    assert auto_gpu.config.system.collection_backend == "process"
    assert final_eval_no_replay.config.evaluation is not None
    assert final_eval_no_replay.config.evaluation.replay_capture_rate_eval == pytest.approx(0.0)
    assert final_eval_no_replay.config.evaluation.regression_capture_count == 0


def test_load_stack_config_supports_strict_b1_factorized_sync5_familyentropy_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(repo_root / "configs/thesis/ablations/b1_noleague_factorized_sync5_familyentropy.yaml")

    assert stack.config.experiment is not None
    assert stack.config.experiment.role == "baseline_noleague"
    assert stack.config.model is not None
    assert stack.config.model.structured_policy_contract == "factorized_v1"
    assert stack.config.training is not None
    assert stack.config.training.teacher_aux_mode == "off"
    assert stack.config.training.structured_aux_enabled is False
    assert stack.config.training.entropy_scope == "family"
    assert stack.config.training.actor_reload_interval_updates == 5
    assert stack.config.training.actor_policy_backend == "model"
    assert stack.config.training.actor_heuristic_fraction == pytest.approx(0.0)
    assert stack.config.league is not None
    assert stack.config.league.enabled is False


def test_load_stack_config_supports_guided_factorized_league_probe_without_b1_anchor() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root / "configs/thesis/_shared/guided_factorized/main_league_guided_factorized_no_b1_anchor_probe.yaml"
    )

    assert stack.config.experiment is not None
    assert stack.config.experiment.role == "ablation_guided"
    assert stack.config.model is not None
    assert stack.config.model.structured_policy_contract == "factorized_v1"
    assert stack.config.training is not None
    assert stack.config.training.teacher_aux_mode == "off"
    assert stack.config.training.structured_aux_enabled is False
    assert stack.config.training.actor_policy_backend == "model"
    assert stack.config.training.diverse_opponent_actor_count == 999
    assert stack.config.training.diverse_model_actor_count == 999
    assert stack.config.training.diverse_opponent_batch_fraction == pytest.approx(1.0)
    assert stack.config.league is not None
    assert stack.config.league.enabled is True
    assert stack.config.league.promotion_anchor_set_v1.required == ("B0 RandomLegal",)
    assert stack.config.league.promotion_anchor_set_v1.optional_if_available == (
        "B2 HeuristicPublic",
        "B3 HeuristicPublicAggro",
        "B4 HeuristicPublicControl",
    )
    assert stack.config.league.sampling.noleague_baseline_mix_fraction == pytest.approx(0.0)
    assert stack.config.league.sampling.warmup_snapshot_mix_fraction == pytest.approx(0.15)
    assert stack.config.evaluation.final_policy_set_selection.fixed_anchor_set_v1.required == ("B0 RandomLegal",)
    assert stack.config.evaluation.final_policy_set_selection.fixed_anchor_set_v1.optional_if_available == (
        "B2 HeuristicPublic",
        "B3 HeuristicPublicAggro",
        "B4 HeuristicPublicControl",
    )


def test_load_stack_config_supports_guided_factorized_league_continuation_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/guided_factorized/main_league_guided_factorized_continuation_no_b1_anchor_probe.yaml"
    )

    assert stack.config.experiment is not None
    assert stack.config.experiment.role == "ablation_guided"
    assert stack.config.model is not None
    assert stack.config.model.structured_policy_contract == "factorized_v1"
    assert stack.config.training is not None
    assert stack.config.training.teacher_aux_mode == "always"
    assert stack.config.training.structured_aux_enabled is True
    assert stack.config.training.entropy_coef == pytest.approx(0.01)
    assert stack.config.training.entropy_scope == "family"
    assert stack.config.training.actor_reload_interval_updates == 5
    assert stack.config.training.teacher_action_coef == pytest.approx(0.08)
    assert stack.config.training.teacher_same_family_action_coef == pytest.approx(0.30)
    assert stack.config.training.teacher_public_heuristic_coef == pytest.approx(0.20)
    assert stack.config.training.teacher_public_heuristic_final_coef == pytest.approx(0.08)
    assert stack.config.training.actor_policy_backend == "model"
    assert stack.config.training.diverse_opponent_actor_count == 999
    assert stack.config.training.diverse_model_actor_count == 999
    assert stack.config.training.diverse_opponent_batch_fraction == pytest.approx(1.0)
    assert stack.config.league is not None
    assert stack.config.league.enabled is True
    assert stack.config.league.sampling.heuristic_public_mix_fraction == pytest.approx(0.2)
    assert stack.config.league.sampling.heuristic_public_variant_mix_fraction == pytest.approx(0.0)
    assert stack.config.league.sampling.noleague_baseline_mix_fraction == pytest.approx(0.0)
    assert stack.config.league.sampling.warmup_snapshot_mix_fraction == pytest.approx(0.15)
    assert stack.config.league.promotion_anchor_set_v1.required == ("B0 RandomLegal",)
    assert stack.config.league.promotion_anchor_set_v1.optional_if_available == (
        "B2 HeuristicPublic",
        "B3 HeuristicPublicAggro",
        "B4 HeuristicPublicControl",
    )


def test_load_stack_config_supports_guided_factorized_league_continuation_allow_main_move_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/guided_factorized/main_league_guided_factorized_continuation_allow_main_move_probe.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.force_pass_over_main_move_only is False
    assert stack.config.training.mulligan_force_confirm_after_select is True
    assert stack.config.training.force_attack_over_pass_when_attack_legal is True
    assert stack.config.training.teacher_aux_mode == "always"
    assert stack.config.training.structured_aux_enabled is True
    assert stack.config.training.teacher_same_family_action_coef == pytest.approx(0.30)
    assert stack.config.league is not None
    assert stack.config.league.enabled is True
    assert stack.config.league.sampling.heuristic_public_mix_fraction == pytest.approx(0.2)
    assert stack.config.league.sampling.noleague_baseline_mix_fraction == pytest.approx(0.0)
    assert stack.config.league.sampling.warmup_snapshot_mix_fraction == pytest.approx(0.15)


def test_load_stack_config_supports_guided_factorized_league_handaux_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/guided_factorized/main_league_guided_factorized_continuation_handaux_probe.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.teacher_aux_mode == "always"
    assert stack.config.training.structured_aux_enabled is True
    assert stack.config.training.teacher_hand_coef == pytest.approx(0.12)
    assert stack.config.training.teacher_slot_coef == pytest.approx(0.08)
    assert stack.config.training.teacher_same_family_action_coef == pytest.approx(0.30)
    assert stack.config.league is not None
    assert stack.config.league.enabled is True
    assert stack.config.league.sampling.heuristic_public_mix_fraction == pytest.approx(0.2)


def test_load_stack_config_supports_guided_factorized_league_teacherfade_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/guided_factorized/main_league_guided_factorized_continuation_teacherfade_probe.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.teacher_aux_mode == "always"
    assert stack.config.training.structured_aux_enabled is True
    assert stack.config.training.teacher_supervised_start_updates == 0
    assert stack.config.training.teacher_supervised_end_updates == 50
    assert stack.config.training.teacher_supervised_final_scale == pytest.approx(0.25)
    assert stack.config.training.teacher_public_heuristic_end_updates == 50
    assert stack.config.training.teacher_public_heuristic_final_coef == pytest.approx(0.0)
    assert stack.config.training.teacher_public_heuristic_profiles_end_updates == 50
    assert stack.config.curriculum.checkpoint_guard.stop_after_rollback is True
    assert stack.config.league is not None
    assert stack.config.league.enabled is True
    assert stack.config.league.sampling.heuristic_public_mix_fraction == pytest.approx(0.2)


def test_load_stack_config_supports_guided_factorized_liveleague_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root / "configs/thesis/_shared/guided_factorized/main_league_guided_factorized_liveleague_probe.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.teacher_supervised_end_updates == 50
    assert stack.config.curriculum.checkpoint_guard.stop_after_rollback is True
    assert stack.config.league is not None
    assert stack.config.league.enabled is True
    assert stack.config.league.warmup.first_updates == 5
    assert stack.config.league.warmup.initial_window_episodes == 512
    assert stack.config.league.warmup.ramp_target_updates == 20
    assert stack.config.league.warmup.ramp_target_window_episodes == 4096
    assert stack.config.league.sampling.heuristic_public_mix_fraction == pytest.approx(0.2)
    assert stack.config.league.sampling.noleague_baseline_mix_fraction == pytest.approx(0.0)


def test_load_stack_config_supports_guided_factorized_liveleague_mirror_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/guided_factorized/main_league_guided_factorized_liveleague_mirror_probe.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.teacher_supervised_end_updates == 50
    assert stack.config.league is not None
    assert stack.config.league.enabled is True
    assert stack.config.league.warmup.first_updates == 5
    assert stack.config.league.sampling.mirror_mix_fraction == pytest.approx(0.5)
    assert stack.config.league.sampling.mirror_mix_end_updates == 100
    assert stack.config.league.sampling.mirror_final_mix_fraction == pytest.approx(0.25)
    assert stack.config.league.sampling.champion_mix_fraction == pytest.approx(0.0)
    assert stack.config.league.sampling.hard_negative_mix_fraction == pytest.approx(0.0)


def test_load_stack_config_supports_main_league_guided_bootstrap() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(repo_root / "configs/thesis/main_league_guided_bootstrap.yaml")

    assert stack.config.experiment is not None
    assert stack.config.experiment.role == "guided_league_bootstrap"
    assert stack.config.training is not None
    assert stack.config.training.teacher_supervised_end_updates == 50
    assert stack.config.curriculum.checkpoint_guard.stop_after_rollback is True
    assert stack.config.league is not None
    assert stack.config.league.enabled is True
    assert stack.config.league.warmup.first_updates == 5
    assert stack.config.league.sampling.heuristic_public_mix_fraction == pytest.approx(0.2)
    assert stack.config.league.sampling.noleague_baseline_mix_fraction == pytest.approx(0.0)
    assert stack.config.league.sampling.mirror_mix_fraction == pytest.approx(0.5)
    assert stack.config.league.sampling.mirror_mix_end_updates == 100
    assert stack.config.league.sampling.mirror_final_mix_fraction == pytest.approx(0.25)


def test_load_stack_config_supports_main_league_guided_bootstrap_vtrace() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(repo_root / "configs/thesis/main_league_guided_bootstrap_vtrace.yaml")

    assert stack.config.experiment is not None
    assert stack.config.experiment.role == "guided_league_bootstrap"
    assert stack.config.training is not None
    assert stack.config.training.vtrace_rho_bar == pytest.approx(1.0)
    assert stack.config.training.vtrace_c_bar == pytest.approx(1.0)
    assert stack.config.curriculum.checkpoint_guard.stop_after_rollback is True
    assert stack.config.league is not None
    assert stack.config.league.enabled is True
    assert stack.config.league.sampling.mirror_mix_fraction == pytest.approx(0.5)
    assert stack.config.league.sampling.mirror_final_mix_fraction == pytest.approx(0.25)


def test_load_stack_config_supports_main_league_guided_bootstrap_seedchampion() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(repo_root / "configs/thesis/main_league_guided_bootstrap_seedchampion.yaml")

    assert stack.config.experiment is not None
    assert stack.config.experiment.role == "guided_league_bootstrap"
    assert stack.config.league is not None
    assert stack.config.league.pool.seed_snapshot_champion_import == "all"
    assert stack.config.league.pool.champion_max_age_updates == 0
    assert stack.config.league.sampling.mirror_mix_fraction == pytest.approx(0.3)
    assert stack.config.league.sampling.mirror_final_mix_fraction == pytest.approx(0.15)
    assert stack.config.league.sampling.champion_mix_fraction == pytest.approx(0.35)
    assert stack.config.league.sampling.hard_negative_mix_fraction == pytest.approx(0.10)


def test_load_stack_config_supports_main_league_guided_bootstrap_selected() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(repo_root / "configs/thesis/main_league_guided_bootstrap_selected.yaml")

    assert stack.config.experiment is not None
    assert stack.config.experiment.role == "guided_league_bootstrap"
    assert stack.config.league is not None
    assert stack.config.league.pool.seed_snapshot_champion_import == "pinned"
    assert stack.config.league.sampling.mirror_mix_fraction == pytest.approx(0.3)
    assert stack.config.league.sampling.champion_mix_fraction == pytest.approx(0.35)


def test_load_stack_config_supports_main_league_guided_bootstrap_selected_anchor() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(repo_root / "configs/thesis/main_league_guided_bootstrap_selected_anchor.yaml")

    assert stack.config.experiment is not None
    assert stack.config.experiment.role == "guided_league_bootstrap"
    assert stack.config.training is not None
    assert stack.config.training.policy_anchor_coef == pytest.approx(0.2)
    assert stack.config.training.policy_anchor_temperature == pytest.approx(1.0)
    assert stack.config.training.policy_anchor_top_action_coef == pytest.approx(0.0)
    assert stack.config.league is not None
    assert stack.config.league.pool.seed_snapshot_champion_import == "pinned"


def test_load_stack_config_supports_main_league_guided_bootstrap_selected_anchor_floor() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(repo_root / "configs/thesis/main_league_guided_bootstrap_selected_anchor_floor.yaml")

    assert stack.config.training is not None
    assert stack.config.training.policy_anchor_coef == pytest.approx(0.2)
    assert stack.config.training.teacher_public_heuristic_coef == pytest.approx(0.04)
    assert stack.config.training.teacher_public_heuristic_final_coef == pytest.approx(0.04)
    assert stack.config.training.teacher_public_heuristic_end_updates == -1
    assert stack.config.training.teacher_public_heuristic_profiles_end_updates == -1
    assert stack.config.league is not None
    assert stack.config.league.pool.seed_snapshot_champion_import == "pinned"


def test_load_stack_config_supports_main_league_guided_bootstrap_selected_anchor_floor_stability() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root / "configs/thesis/main_league_guided_bootstrap_selected_anchor_floor_stability.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.learning_rate == pytest.approx(0.00005)
    assert stack.config.training.actor_sampling_temperature == pytest.approx(1.0)
    assert stack.config.training.policy_anchor_coef == pytest.approx(0.2)
    assert stack.config.training.teacher_public_heuristic_coef == pytest.approx(0.04)
    assert stack.config.league is not None
    assert stack.config.league.pool.seed_snapshot_champion_import == "pinned"


def test_load_stack_config_supports_main_league_guided_bootstrap_selected_anchor_profile_floor() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root / "configs/thesis/main_league_guided_bootstrap_selected_anchor_profile_floor.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.policy_anchor_coef == pytest.approx(0.2)
    assert stack.config.training.teacher_public_heuristic_coef == pytest.approx(0.04)
    assert stack.config.training.teacher_public_heuristic_profiles == ("base", "aggressive", "control")
    assert stack.config.training.teacher_public_heuristic_profile_mode == "cycle"
    assert stack.config.training.teacher_public_heuristic_profiles_end_updates == -1
    assert stack.config.league is not None
    assert stack.config.league.pool.seed_snapshot_champion_import == "pinned"


def test_load_stack_config_supports_guided_factorized_postbest_floor_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root / "configs/thesis/_shared/guided_factorized/main_league_guided_factorized_postbest_floor_probe.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.teacher_aux_mode == "always"
    assert stack.config.training.structured_aux_enabled is True
    assert stack.config.training.teacher_supervised_start_updates == 0
    assert stack.config.training.teacher_supervised_end_updates == 0
    assert stack.config.training.teacher_supervised_final_scale == pytest.approx(0.25)
    assert stack.config.training.teacher_public_heuristic_coef == pytest.approx(0.04)
    assert stack.config.training.teacher_public_heuristic_end_updates == -1
    assert stack.config.training.teacher_public_heuristic_final_coef == pytest.approx(0.04)
    assert stack.config.training.teacher_public_heuristic_profiles_end_updates == -1
    assert stack.config.curriculum.checkpoint_guard.stop_after_rollback is True
    assert stack.config.league is not None
    assert stack.config.league.enabled is True
    assert stack.config.league.sampling.heuristic_public_mix_fraction == pytest.approx(0.2)


def test_load_stack_config_supports_guided_factorized_postbest_stability_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/guided_factorized/main_league_guided_factorized_postbest_stability_probe.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.learning_rate == pytest.approx(0.00005)
    assert stack.config.training.actor_sampling_temperature == pytest.approx(1.0)
    assert stack.config.training.teacher_supervised_final_scale == pytest.approx(0.25)
    assert stack.config.training.teacher_public_heuristic_coef == pytest.approx(0.04)
    assert stack.config.curriculum.checkpoint_guard.stop_after_rollback is True
    assert stack.config.league is not None
    assert stack.config.league.enabled is True


def test_load_stack_config_supports_guided_factorized_postbest_vtrace_clamp_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/guided_factorized/main_league_guided_factorized_postbest_vtrace_clamp_probe.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.vtrace_rho_bar == pytest.approx(1.0)
    assert stack.config.training.vtrace_c_bar == pytest.approx(1.0)
    assert stack.config.training.teacher_supervised_final_scale == pytest.approx(0.25)
    assert stack.config.training.teacher_public_heuristic_coef == pytest.approx(0.04)
    assert stack.config.curriculum.checkpoint_guard.stop_after_rollback is True
    assert stack.config.league is not None
    assert stack.config.league.enabled is True


def test_load_stack_config_supports_guided_factorized_postbest_vtrace_profile_floor_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/guided_factorized/main_league_guided_factorized_postbest_vtrace_profile_floor_probe.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.vtrace_rho_bar == pytest.approx(1.0)
    assert stack.config.training.vtrace_c_bar == pytest.approx(1.0)
    assert stack.config.training.teacher_public_heuristic_coef == pytest.approx(0.04)
    assert stack.config.training.teacher_public_heuristic_final_coef == pytest.approx(0.04)
    assert stack.config.training.teacher_public_heuristic_profiles == ("base", "aggressive", "control")
    assert stack.config.training.teacher_public_heuristic_profile_mode == "cycle"
    assert stack.config.training.teacher_public_heuristic_profiles_end_updates == -1
    assert stack.config.curriculum.checkpoint_guard.stop_after_rollback is True
    assert stack.config.league is not None
    assert stack.config.league.enabled is True


def test_load_stack_config_supports_guided_factorized_postbest_vtrace_variant005_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/guided_factorized/main_league_guided_factorized_postbest_vtrace_variant005_probe.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.vtrace_rho_bar == pytest.approx(1.0)
    assert stack.config.training.vtrace_c_bar == pytest.approx(1.0)
    assert stack.config.training.teacher_public_heuristic_profiles == ("base",)
    assert stack.config.training.teacher_public_heuristic_coef == pytest.approx(0.04)
    assert stack.config.league is not None
    assert stack.config.league.enabled is True
    assert stack.config.league.sampling.heuristic_public_mix_fraction == pytest.approx(0.2)
    assert stack.config.league.sampling.heuristic_public_variant_mix_fraction == pytest.approx(0.05)
    assert stack.config.league.sampling.heuristic_public_variant_final_mix_fraction == pytest.approx(0.05)
    assert stack.config.league.sampling.heuristic_public_variant_mix_end_updates == -1


def test_load_stack_config_supports_guided_factorized_postbest_vtrace_control_floor_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/guided_factorized/main_league_guided_factorized_postbest_vtrace_control_floor_probe.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.vtrace_rho_bar == pytest.approx(1.0)
    assert stack.config.training.vtrace_c_bar == pytest.approx(1.0)
    assert stack.config.training.teacher_public_heuristic_coef == pytest.approx(0.04)
    assert stack.config.training.teacher_public_heuristic_final_coef == pytest.approx(0.04)
    assert stack.config.training.teacher_public_heuristic_profiles == ("control",)
    assert stack.config.training.teacher_public_heuristic_profile_mode == "mixture"
    assert stack.config.training.teacher_public_heuristic_profiles_end_updates == -1
    assert stack.config.league is not None
    assert stack.config.league.enabled is True


def test_load_stack_config_supports_guided_factorized_postbest_profile_floor_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/guided_factorized/main_league_guided_factorized_postbest_profile_floor_probe.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.teacher_public_heuristic_coef == pytest.approx(0.04)
    assert stack.config.training.teacher_public_heuristic_final_coef == pytest.approx(0.04)
    assert stack.config.training.teacher_public_heuristic_profiles == ("base", "aggressive", "control")
    assert stack.config.training.teacher_public_heuristic_profile_mode == "cycle"
    assert stack.config.training.teacher_public_heuristic_profiles_end_updates == -1
    assert stack.config.curriculum.checkpoint_guard.stop_after_rollback is True
    assert stack.config.league is not None
    assert stack.config.league.enabled is True
    assert stack.config.league.sampling.heuristic_public_variant_mix_fraction == pytest.approx(0.0)


def test_load_stack_config_supports_guided_factorized_postbest_mainmove_once_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/guided_factorized/main_league_guided_factorized_postbest_mainmove_once_probe.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.force_pass_over_main_move_only is True
    assert stack.config.training.main_move_only_max_consecutive == 1
    assert stack.config.training.teacher_public_heuristic_coef == pytest.approx(0.04)
    assert stack.config.curriculum.checkpoint_guard.stop_after_rollback is True
    assert stack.config.league is not None
    assert stack.config.league.enabled is True


def test_load_stack_config_supports_guided_factorized_postbest_b4_exact_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/guided_factorized/main_league_guided_factorized_postbest_b4_exact_probe.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.teacher_supervised_final_scale == pytest.approx(0.5)
    assert stack.config.training.teacher_exact_action_families == (
        "main_play_character",
        "clock_from_hand",
        "attack",
    )
    assert stack.config.training.teacher_slot_coef == pytest.approx(0.12)
    assert stack.config.training.teacher_hand_coef == pytest.approx(0.12)
    assert stack.config.training.teacher_attack_type_coef == pytest.approx(0.08)
    assert stack.config.training.teacher_action_coef == pytest.approx(0.08)
    assert stack.config.training.teacher_same_family_action_coef == pytest.approx(0.45)
    assert stack.config.training.teacher_same_family_action_margin_coef == pytest.approx(0.12)
    assert stack.config.training.teacher_same_family_action_margin == pytest.approx(0.75)
    assert stack.config.training.teacher_public_heuristic_profiles == ("base", "aggressive", "control")
    assert stack.config.curriculum.checkpoint_guard.stop_after_rollback is True
    assert stack.config.league is not None
    assert stack.config.league.enabled is True


def test_load_stack_config_supports_guided_factorized_postbest_anchor_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root / "configs/thesis/_shared/guided_factorized/main_league_guided_factorized_postbest_anchor_probe.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.policy_anchor_coef == pytest.approx(0.08)
    assert stack.config.training.policy_anchor_temperature == pytest.approx(1.0)
    assert stack.config.training.teacher_public_heuristic_coef == pytest.approx(0.04)
    assert stack.config.curriculum.checkpoint_guard.stop_after_rollback is True
    assert stack.config.league is not None
    assert stack.config.league.enabled is True


def test_load_stack_config_supports_guided_factorized_postbest_anchor_strong_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/guided_factorized/main_league_guided_factorized_postbest_anchor_strong_probe.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.policy_anchor_coef == pytest.approx(0.8)
    assert stack.config.training.policy_anchor_temperature == pytest.approx(0.5)
    assert stack.config.training.teacher_public_heuristic_coef == pytest.approx(0.04)
    assert stack.config.curriculum.checkpoint_guard.stop_after_rollback is True
    assert stack.config.league is not None
    assert stack.config.league.enabled is True


def test_load_stack_config_supports_guided_factorized_postbest_anchor_topaction_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/guided_factorized/main_league_guided_factorized_postbest_anchor_topaction_probe.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.policy_anchor_coef == pytest.approx(0.2)
    assert stack.config.training.policy_anchor_top_action_coef == pytest.approx(0.06)
    assert stack.config.training.policy_anchor_temperature == pytest.approx(0.75)
    assert stack.config.curriculum.checkpoint_guard.stop_after_rollback is True
    assert stack.config.league is not None
    assert stack.config.league.enabled is True


def test_load_stack_config_supports_guided_factorized_league_variant_profiles_probe() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs/thesis/_shared/guided_factorized/main_league_guided_factorized_continuation_variant_profiles_probe.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.force_pass_over_main_move_only is True
    assert stack.config.training.teacher_aux_mode == "always"
    assert stack.config.training.structured_aux_enabled is True
    assert stack.config.training.teacher_public_heuristic_profiles == ("base", "aggressive", "control")
    assert stack.config.training.teacher_public_heuristic_profile_mode == "cycle"
    assert stack.config.league is not None
    assert stack.config.league.enabled is True
    assert stack.config.league.sampling.heuristic_public_mix_fraction == pytest.approx(0.2)
    assert stack.config.league.sampling.heuristic_public_variant_mix_fraction == pytest.approx(0.2)
    assert stack.config.league.sampling.heuristic_public_variant_final_mix_fraction == pytest.approx(0.2)
    assert stack.config.league.sampling.noleague_baseline_mix_fraction == pytest.approx(0.0)
    assert stack.config.league.sampling.warmup_snapshot_mix_fraction == pytest.approx(0.15)


def test_b1_tight_sync_probes_remain_clean_noleague_routes() -> None:
    repo_root = _repo_root()

    sync20 = load_stack_config(repo_root / "configs/thesis/ablations/b1_noleague_tight_sync20.yaml")
    sync20_temp050 = load_stack_config(repo_root / "configs/thesis/ablations/b1_noleague_tight_sync20_temp050.yaml")

    for stack in (sync20, sync20_temp050):
        assert stack.config.experiment is not None
        assert stack.config.experiment.role == "baseline_noleague"
        assert stack.config.league is not None
        assert stack.config.league.enabled is False
        assert stack.config.league.sampling.heuristic_public_mix_fraction == pytest.approx(0.0)
        assert stack.config.league.sampling.heuristic_public_variant_mix_fraction == pytest.approx(0.0)
        assert stack.config.league.sampling.noleague_baseline_mix_fraction == pytest.approx(0.0)
        assert stack.config.model is not None
        assert stack.config.model.public_heuristic_logit_bias_scale == pytest.approx(0.0)
        assert stack.config.model.public_heuristic_actor_logit_bias_scale == pytest.approx(0.0)
        assert stack.config.training is not None
        assert stack.config.training.actor_policy_backend == "model"
        assert stack.config.training.actor_heuristic_fraction == pytest.approx(0.0)
        assert stack.config.training.actor_heuristic_final_fraction == pytest.approx(0.0)
        assert stack.config.training.teacher_aux_mode == "off"
        assert stack.config.training.structured_aux.enabled is False
        assert stack.config.training.structured_warmstart_enabled is False
        assert stack.config.training.teacher_public_heuristic_coef == pytest.approx(0.0)
        assert stack.config.training.structured_warmstart.teacher_public_heuristic_coef == pytest.approx(0.0)
        assert stack.config.training.actor_reload_interval_updates == 20
        assert stack.config.training.mulligan_force_confirm_after_select is True
        assert stack.config.training.force_pass_over_main_move_only is True
        assert stack.config.training.force_attack_over_pass_when_attack_legal is True
        assert stack.config.evaluation.final_policy_set_selection.fixed_anchor_set_v1.optional_if_available == (
            "B2 HeuristicPublic",
            "B3 HeuristicPublicAggro",
            "B4 HeuristicPublicControl",
        )

    assert sync20.config.training.actor_sampling_temperature == pytest.approx(1.0)
    assert sync20_temp050.config.training.actor_sampling_temperature == pytest.approx(0.5)


def test_thesis_reward_ablations_are_isolated_b1_routes() -> None:
    repo_root = _repo_root()
    expected = {
        "terminal_only_reward.yaml": ("terminal_only_pm1", False, 0.0, 0.0, 0.0, 0.0, 0.02, 0.0, 1.0, 0.0, True),
        "damage_only_reward.yaml": ("terminal_pm1", True, 0.05, 0.0, 0.0, 0.0, 0.02, 0.0, 0.99, -0.1, False),
        "level_only_reward.yaml": ("terminal_pm1", True, 0.0, 0.05, 0.0, 0.0, 0.02, 0.0, 0.99, -0.1, False),
        "board_only_reward.yaml": ("terminal_pm1", True, 0.0, 0.0, 0.02, 0.0, 0.02, 0.0, 0.99, -0.1, False),
        "no_progress_penalty_reward.yaml": (
            "terminal_pm1",
            True,
            0.0,
            0.0,
            0.0,
            0.005,
            0.02,
            0.0,
            0.99,
            -0.1,
            False,
        ),
        "damage_level_reward.yaml": ("terminal_pm1", True, 0.05, 0.05, 0.0, 0.0, 0.02, 0.0, 0.99, -0.1, False),
        "full_shaping_reward.yaml": ("terminal_pm1", True, 0.05, 0.05, 0.02, 0.005, 0.02, 0.0, 0.99, -0.1, False),
        "full_shaping_sync10_reward.yaml": (
            "terminal_pm1",
            True,
            0.05,
            0.05,
            0.02,
            0.005,
            0.02,
            0.0,
            0.99,
            -0.1,
            False,
        ),
        "full_shaping_entropy01_reward.yaml": (
            "terminal_pm1",
            True,
            0.05,
            0.05,
            0.02,
            0.005,
            0.02,
            0.0,
            0.99,
            -0.1,
            False,
        ),
        "full_shaping_entropy01_sync20_reward.yaml": (
            "terminal_pm1",
            True,
            0.05,
            0.05,
            0.02,
            0.005,
            0.02,
            0.0,
            0.99,
            -0.1,
            False,
        ),
        "full_shaping_entropy01_sync5_reward.yaml": (
            "terminal_pm1",
            True,
            0.05,
            0.05,
            0.02,
            0.005,
            0.02,
            0.0,
            0.99,
            -0.1,
            False,
        ),
        "full_shaping_family_entropy_sync5_reward.yaml": (
            "terminal_pm1",
            True,
            0.05,
            0.05,
            0.02,
            0.005,
            0.02,
            0.0,
            0.99,
            -0.1,
            False,
        ),
        "full_shaping_family_entropy_sync5_passpenalty02_reward.yaml": (
            "terminal_pm1",
            True,
            0.05,
            0.05,
            0.02,
            0.005,
            0.02,
            0.0,
            0.99,
            -0.1,
            False,
        ),
        "full_shaping_family_entropy_sync5_passpenalty02_mulligan02_reward.yaml": (
            "terminal_pm1",
            True,
            0.05,
            0.05,
            0.02,
            0.005,
            0.02,
            0.02,
            0.99,
            -0.1,
            False,
        ),
        "full_shaping_family_entropy_sync5_passpenalty02_mulliganguard_reward.yaml": (
            "terminal_pm1",
            True,
            0.05,
            0.05,
            0.02,
            0.005,
            0.02,
            0.0,
            0.99,
            -0.1,
            False,
        ),
        "full_shaping_entropy01_sync1_reward.yaml": (
            "terminal_pm1",
            True,
            0.05,
            0.05,
            0.02,
            0.005,
            0.02,
            0.0,
            0.99,
            -0.1,
            False,
        ),
        "full_shaping_entropy01_sync20_noprog02_reward.yaml": (
            "terminal_pm1",
            True,
            0.05,
            0.05,
            0.02,
            0.02,
            0.02,
            0.0,
            0.99,
            -0.1,
            False,
        ),
        "full_shaping_entropy01_sync5_passpenalty02_reward.yaml": (
            "terminal_pm1",
            True,
            0.05,
            0.05,
            0.02,
            0.005,
            0.02,
            0.0,
            0.99,
            -0.1,
            False,
        ),
        "full_shaping_entropy01_sync5_mulligan02_reward.yaml": (
            "terminal_pm1",
            True,
            0.05,
            0.05,
            0.02,
            0.005,
            0.02,
            0.02,
            0.99,
            -0.1,
            False,
        ),
    }

    for filename, (
        objective,
        enabled,
        damage,
        level,
        board,
        no_progress,
        pass_penalty,
        mulligan_penalty,
        gamma,
        truncation_reward,
        bootstrap_value,
    ) in expected.items():
        config_dir = (
            repo_root / "configs" / "thesis" / "ablations"
            if filename == "terminal_only_reward.yaml"
            else repo_root / "configs" / "archive" / "thesis_reward_ablations_20260513"
        )
        stack = load_stack_config(config_dir / filename)
        assert stack.config.experiment is not None
        assert stack.config.experiment.role == "ablation_reward"
        assert stack.config.training is not None
        assert stack.config.training.actor_policy_backend == "model"
        assert stack.config.training.actor_heuristic_fraction == pytest.approx(0.0)
        if filename == "full_shaping_sync10_reward.yaml":
            assert stack.config.training.actor_reload_interval_updates == 10
        if filename == "full_shaping_entropy01_reward.yaml":
            assert stack.config.training.entropy_coef == pytest.approx(0.01)
            assert stack.config.training.entropy_anneal_to == pytest.approx(0.003)
        if filename == "full_shaping_entropy01_sync20_reward.yaml":
            assert stack.config.training.entropy_coef == pytest.approx(0.01)
            assert stack.config.training.entropy_anneal_to == pytest.approx(0.003)
            assert stack.config.training.actor_reload_interval_updates == 20
        if filename == "full_shaping_entropy01_sync5_reward.yaml":
            assert stack.config.training.entropy_coef == pytest.approx(0.01)
            assert stack.config.training.entropy_anneal_to == pytest.approx(0.003)
            assert stack.config.training.actor_reload_interval_updates == 5
        if filename == "full_shaping_family_entropy_sync5_reward.yaml":
            assert stack.config.training.entropy_coef == pytest.approx(0.01)
            assert stack.config.training.entropy_scope == "family"
            assert stack.config.training.actor_reload_interval_updates == 5
        if filename == "full_shaping_family_entropy_sync5_passpenalty02_reward.yaml":
            assert stack.config.training.entropy_coef == pytest.approx(0.01)
            assert stack.config.training.entropy_scope == "family"
            assert stack.config.training.actor_reload_interval_updates == 5
        if filename == "full_shaping_family_entropy_sync5_passpenalty02_mulligan02_reward.yaml":
            assert stack.config.training.entropy_coef == pytest.approx(0.01)
            assert stack.config.training.entropy_scope == "family"
            assert stack.config.training.actor_reload_interval_updates == 5
        if filename == "full_shaping_family_entropy_sync5_passpenalty02_mulliganguard_reward.yaml":
            assert stack.config.training.entropy_coef == pytest.approx(0.01)
            assert stack.config.training.entropy_scope == "family"
            assert stack.config.training.actor_reload_interval_updates == 5
            assert stack.config.training.mulligan_force_confirm_after_select is True
        if filename == "full_shaping_entropy01_sync1_reward.yaml":
            assert stack.config.training.entropy_coef == pytest.approx(0.01)
            assert stack.config.training.entropy_anneal_to == pytest.approx(0.003)
            assert stack.config.training.actor_reload_interval_updates == 1
        if filename == "full_shaping_entropy01_sync20_noprog02_reward.yaml":
            assert stack.config.training.entropy_coef == pytest.approx(0.01)
            assert stack.config.training.entropy_anneal_to == pytest.approx(0.003)
            assert stack.config.training.actor_reload_interval_updates == 20
        assert stack.config.training.teacher_aux_mode == "off"
        assert stack.config.league is not None
        assert stack.config.league.enabled is False
        assert stack.config.league.promotion_anchor_set_v1.required == ("B0 RandomLegal",)
        assert stack.config.league.promotion_anchor_set_v1.optional_if_available == (
            "B1 NoLeague baseline",
            "B2 HeuristicPublic",
        )
        assert stack.config.environment is not None
        assert stack.config.environment.deck_pool == ("preset:main_deck_5hy_yotsuba_v1",)
        assert stack.config.environment.opponent_deck_pool == ("preset:main_deck_5hy_yotsuba_v1",)
        assert stack.config.rewards is not None
        assert stack.config.rewards.objective == objective
        assert stack.config.rewards.gamma == pytest.approx(gamma)
        assert stack.config.rewards.shaping.enable_damage_shaping is enabled
        assert stack.config.rewards.shaping.damage_reward == pytest.approx(damage)
        assert stack.config.rewards.shaping.level_reward == pytest.approx(level)
        assert stack.config.rewards.shaping.board_reward == pytest.approx(board)
        assert stack.config.rewards.shaping.no_progress_penalty == pytest.approx(no_progress)
        assert stack.config.rewards.shaping.pass_with_nonpass_penalty == pytest.approx(pass_penalty)
        assert stack.config.rewards.shaping.mulligan_select_with_confirm_penalty == pytest.approx(mulligan_penalty)
        assert stack.config.rewards.truncation.reward == pytest.approx(truncation_reward)
        assert stack.config.rewards.truncation.bootstrap_value is bootstrap_value


def test_load_stack_config_supports_structured_dev_fast_and_acceptance_presets() -> None:
    repo_root = _repo_root()

    dev_fast = load_stack_config(repo_root / "configs/presets/structured_dev_fast.yaml")
    assert dev_fast.config.league is not None
    assert dev_fast.config.league.enabled is False
    assert dev_fast.config.model is not None
    assert dev_fast.config.model.cuda_learner_candidate_scoring_chunk_size == 1048576
    assert dev_fast.config.model.public_heuristic_logit_bias_scale == pytest.approx(0.0)
    assert dev_fast.config.model.public_heuristic_actor_logit_bias_scale == pytest.approx(-1.0)
    assert dev_fast.config.training is not None
    assert dev_fast.config.training.profile_timers is False
    assert dev_fast.config.training.torch_profiler is False
    assert dev_fast.config.training.compile_actor_inference is False
    assert dev_fast.config.training.structured_metrics_mode == "off"
    assert dev_fast.config.training.teacher_aux_mode == "warmstart_only"
    assert dev_fast.config.training.fixed_opponent_backend == "python_batched"
    assert dev_fast.config.training.actor_policy_backend == "model"
    assert dev_fast.config.training.actor_heuristic_fraction == pytest.approx(1.0)
    assert dev_fast.config.training.heuristic_actor_hidden_state_tracking is True
    assert dev_fast.config.training.teacher_public_heuristic_coef == pytest.approx(0.0)
    assert dev_fast.config.training.structured_warmstart.updates == 1

    acceptance = load_stack_config(repo_root / "configs/presets/structured_acceptance.yaml")
    assert acceptance.config.model is not None
    assert acceptance.config.model.cuda_learner_candidate_scoring_chunk_size == 1048576
    assert acceptance.config.model.public_heuristic_logit_bias_scale == pytest.approx(0.0)
    assert acceptance.config.model.public_heuristic_actor_logit_bias_scale == pytest.approx(-1.0)
    assert acceptance.config.training is not None
    assert acceptance.config.training.profile_timers is False
    assert acceptance.config.training.torch_profiler is False
    assert acceptance.config.training.compile_actor_inference is False
    assert acceptance.config.training.structured_metrics_mode == "sampled"
    assert acceptance.config.training.teacher_aux_mode == "always"
    assert acceptance.config.training.fixed_opponent_backend == "python_batched"
    assert acceptance.config.training.actor_policy_backend == "model"
    assert acceptance.config.training.actor_heuristic_fraction == pytest.approx(1.0)
    assert acceptance.config.training.heuristic_actor_hidden_state_tracking is True
    assert acceptance.config.training.teacher_public_heuristic_coef == pytest.approx(0.0)
    assert acceptance.config.training.structured_warmstart.updates == 32

    linux_frontier = load_stack_config(repo_root / "configs/presets/structured_acceptance_linux_frontier.yaml")
    assert linux_frontier.config.system is not None
    assert linux_frontier.config.system.collection_backend == "process"
    assert linux_frontier.config.training is not None
    assert linux_frontier.config.training.actor_policy_backend == "heuristic_public"
    assert linux_frontier.config.training.actor_heuristic_fraction == pytest.approx(1.0)
    assert linux_frontier.config.training.heuristic_actor_hidden_state_tracking is True
    assert linux_frontier.config.training.fixed_opponent_backend == "simulator_native"
    assert linux_frontier.config.training.structured_warmstart.updates == 1
    assert linux_frontier.config.model is not None
    assert linux_frontier.config.model.gru_hidden_size == 32
    assert linux_frontier.config.model.encoder_mlp_width == 32
    assert linux_frontier.config.model.typed_feature_width == 8
    assert linux_frontier.config.league is not None
    assert linux_frontier.config.league.sampling.noleague_baseline_mix_fraction == pytest.approx(0.0)
    assert linux_frontier.config.league.sampling.noleague_baseline_mix_end_updates == -1

    linux_hybrid_frontier = load_stack_config(
        repo_root / "configs/presets/structured_acceptance_linux_hybrid_frontier.yaml"
    )
    assert linux_hybrid_frontier.config.model is not None
    assert linux_hybrid_frontier.config.model.public_heuristic_logit_bias_scale == pytest.approx(1.0)
    assert linux_hybrid_frontier.config.model.public_heuristic_actor_logit_bias_scale == pytest.approx(1.0)
    assert linux_hybrid_frontier.config.training is not None
    assert linux_hybrid_frontier.config.training.teacher_public_heuristic_coef == pytest.approx(0.1)
    assert linux_hybrid_frontier.config.training.structured_warmstart.teacher_public_heuristic_coef == pytest.approx(
        0.5
    )

    linux_learning_frontier = load_stack_config(
        repo_root / "configs/presets/structured_acceptance_linux_learning_frontier.yaml"
    )
    assert linux_learning_frontier.config.model is not None
    assert linux_learning_frontier.config.model.gru_hidden_size == 64
    assert linux_learning_frontier.config.model.encoder_mlp_width == 64
    assert linux_learning_frontier.config.model.typed_feature_width == 16
    assert linux_learning_frontier.config.league is not None
    assert linux_learning_frontier.config.league.sampling.noleague_baseline_mix_fraction == pytest.approx(0.15)
    assert linux_learning_frontier.config.league.sampling.noleague_baseline_mix_end_updates == -1
    assert linux_learning_frontier.config.league.sampling.warmup_snapshot_mix_fraction == pytest.approx(0.0)

    linux_learning_curriculum = load_stack_config(
        repo_root / "configs/presets/structured_acceptance_linux_learning_curriculum.yaml"
    )
    assert linux_learning_curriculum.config.league is not None
    assert linux_learning_curriculum.config.league.sampling.noleague_baseline_mix_fraction == pytest.approx(0.15)
    assert linux_learning_curriculum.config.league.sampling.noleague_baseline_mix_end_updates == 5

    tactical_multideck = load_stack_config(
        repo_root
        / "configs/presets/structured_acceptance_tiny32_fast_pubcycle3_b1mix015_b1end10_tacticalbias_multideck.yaml"
    )
    assert tactical_multideck.config.environment is not None
    assert tactical_multideck.config.environment.deck_pool == (
        "preset:starter_deck_ws02_v1",
        "preset:main_deck_5hy_yotsuba_v1",
        "preset:aggro_deck_5hy_nino_v1",
        "preset:control_deck_jj_s66_v1",
    )
    assert tactical_multideck.config.environment.opponent_deck_pool == (
        "preset:control_deck_jj_s66_v1",
        "preset:aggro_deck_5hy_nino_v1",
        "preset:main_deck_5hy_yotsuba_v1",
        "preset:starter_deck_ws02_v1",
    )

    standard = load_stack_config(repo_root / "configs/presets/structured_acceptance_standard.yaml")
    assert standard.config.model is not None
    assert standard.config.model.public_heuristic_logit_bias_scale == pytest.approx(2.0)
    assert standard.config.league is not None
    assert standard.config.league.sampling.noleague_baseline_mix_end_updates == 10
    assert standard.config.curriculum is not None
    assert standard.config.curriculum.checkpoint_guard.rollback_score_margin == pytest.approx(0.01)
    assert standard.config.environment is not None
    assert standard.config.environment.deck_pool == ("preset:main_deck_5hy_yotsuba_v1",)
    assert standard.config.environment.opponent_deck_pool == ("preset:main_deck_5hy_yotsuba_v1",)

    standard_auto_gpu = load_stack_config(repo_root / "configs/presets/structured_acceptance_standard_auto_gpu.yaml")
    assert standard_auto_gpu.config.system is not None
    assert standard_auto_gpu.config.system.learner_device == "cuda:auto"
    assert standard_auto_gpu.config.system.actor_device == "cuda:auto"
    assert standard_auto_gpu.config.system.collection_backend == "process"

    standard_eval = load_stack_config(repo_root / "configs/presets/structured_acceptance_standard_thesis_eval.yaml")
    assert standard_eval.config.evaluation is not None
    assert standard_eval.config.environment is not None
    assert standard_eval.config.environment.deck_pool == ("preset:main_deck_5hy_yotsuba_v1",)
    assert standard_eval.config.environment.opponent_deck_pool == ("preset:main_deck_5hy_yotsuba_v1",)
    assert (
        "B3 HeuristicPublicAggro"
        in standard_eval.config.evaluation.final_policy_set_selection.fixed_anchor_set_v1.optional_if_available
    )

    standard_multideck = load_stack_config(repo_root / "configs/presets/structured_acceptance_standard_multideck.yaml")
    assert standard_multideck.config.environment is not None
    assert "preset:main_deck_5hy_yotsuba_v1" in standard_multideck.config.environment.deck_pool

    teacher_fade = load_stack_config(repo_root / "configs/presets/ablations/standard_teacher_fade.yaml")
    assert teacher_fade.config.training is not None
    assert teacher_fade.config.training.actor_heuristic_start_updates == 40
    assert teacher_fade.config.training.actor_heuristic_end_updates == 140
    assert teacher_fade.config.training.actor_heuristic_final_fraction == pytest.approx(0.25)
    assert teacher_fade.config.training.teacher_public_heuristic_start_updates == 40
    assert teacher_fade.config.training.teacher_public_heuristic_end_updates == 140
    assert teacher_fade.config.training.teacher_public_heuristic_final_coef == pytest.approx(0.0)
    assert teacher_fade.config.model is not None
    assert teacher_fade.config.model.public_heuristic_logit_bias_start_updates == 40
    assert teacher_fade.config.model.public_heuristic_logit_bias_end_updates == 140
    assert teacher_fade.config.model.public_heuristic_logit_bias_final_scale == pytest.approx(0.5)

    teacher_fade_auto_gpu = load_stack_config(
        repo_root / "configs/presets/ablations/standard_teacher_fade_auto_gpu.yaml"
    )
    assert teacher_fade_auto_gpu.config.system is not None
    assert teacher_fade_auto_gpu.config.system.learner_device == "cuda:auto"
    assert teacher_fade_auto_gpu.config.system.actor_device == "cuda:auto"
    assert teacher_fade_auto_gpu.config.system.collection_backend == "process"

    ablate_no_tactical_bias = load_stack_config(repo_root / "configs/presets/ablations/standard_no_tactical_bias.yaml")
    assert ablate_no_tactical_bias.config.model is not None
    assert (
        ablate_no_tactical_bias.config.model.public_heuristic_logit_bias_scale
        < standard.config.model.public_heuristic_logit_bias_scale
    )

    ablate_no_b1_cutoff = load_stack_config(repo_root / "configs/presets/ablations/standard_no_b1_cutoff.yaml")
    assert ablate_no_b1_cutoff.config.league is not None
    assert ablate_no_b1_cutoff.config.league.sampling.noleague_baseline_mix_end_updates == -1


def test_mulliganguard_final_eval_preserves_action_surface_policy_behavior() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(repo_root / "configs" / "thesis" / "ablations" / "final_eval_mulliganguard.yaml")

    assert stack.config.training is not None
    assert stack.config.training.mulligan_force_confirm_after_select is True
    assert stack.config.evaluation is not None
    assert stack.config.evaluation.final_policy_set_selection.include_heuristic_public_anchors_b2_b3_b4 is True


def test_mainmoveguard_final_eval_preserves_action_surface_policy_behavior() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root / "configs" / "thesis" / "ablations" / "final_eval_mainmoveguard_mulliganguard.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.mulligan_force_confirm_after_select is True
    assert stack.config.training.force_pass_over_main_move_only is True
    assert stack.config.evaluation is not None
    assert stack.config.evaluation.final_policy_set_selection.include_heuristic_public_anchors_b2_b3_b4 is True


def test_argmax_final_eval_preserves_guarded_surface_and_sampling_mode() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root / "configs" / "thesis" / "ablations" / "final_eval_argmax_mainmoveguard_mulliganguard.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.mulligan_force_confirm_after_select is True
    assert stack.config.training.force_pass_over_main_move_only is True
    assert stack.config.evaluation is not None
    assert stack.config.evaluation.eval_sampling_algorithm == "model_argmax_pinned_v1"
    assert stack.config.evaluation.final_policy_set_selection.include_heuristic_public_anchors_b2_b3_b4 is True


def test_attackguard_argmax_final_eval_preserves_guarded_surface_and_sampling_mode() -> None:
    repo_root = _repo_root()

    stack = load_stack_config(
        repo_root
        / "configs"
        / "thesis"
        / "ablations"
        / "final_eval_argmax_attackguard_mainmoveguard_mulliganguard.yaml"
    )

    assert stack.config.training is not None
    assert stack.config.training.mulligan_force_confirm_after_select is True
    assert stack.config.training.force_pass_over_main_move_only is True
    assert stack.config.training.force_attack_over_pass_when_attack_legal is True
    assert stack.config.evaluation is not None
    assert stack.config.evaluation.eval_sampling_algorithm == "model_argmax_pinned_v1"
    assert stack.config.evaluation.final_policy_set_selection.include_heuristic_public_anchors_b2_b3_b4 is True


def test_temperature_final_eval_preserves_guarded_surface_and_sampling_temperature() -> None:
    repo_root = _repo_root()

    for config_name, expected_temperature, expected_attack_guard in (
        ("final_eval_temp025_mainmoveguard_mulliganguard.yaml", 0.25, True),
        ("final_eval_temp005_mainmoveguard_mulliganguard.yaml", 0.05, True),
        ("final_eval_temp005_attackguard_mainmoveguard_mulliganguard.yaml", 0.05, True),
    ):
        stack = load_stack_config(repo_root / "configs" / "thesis" / "ablations" / config_name)

        assert stack.config.training is not None
        assert stack.config.training.mulligan_force_confirm_after_select is True
        assert stack.config.training.force_pass_over_main_move_only is True
        assert stack.config.training.force_attack_over_pass_when_attack_legal is expected_attack_guard
        assert stack.config.evaluation is not None
        assert stack.config.evaluation.eval_sampling_algorithm == "pinned_cdf_pcg_v1"
        assert stack.config.evaluation.model_sampling_temperature == pytest.approx(expected_temperature)
        assert stack.config.evaluation.final_policy_set_selection.include_heuristic_public_anchors_b2_b3_b4 is True


def test_actor_temperature_ablation_preserves_lowentropy_surface() -> None:
    repo_root = _repo_root()
    stack = load_stack_config(
        repo_root
        / "configs"
        / "thesis"
        / "_shared"
        / "guided_teacher"
        / (
            "public_teacher_b2exact_filteredexact_constpublic_antipass02_lowentropy_actortemp025_"
            "argmaxdev_attackguard_mainmoveguard_mulliganguard_reward.yaml"
        )
    )

    assert stack.config.training is not None
    assert stack.config.training.entropy_coef == pytest.approx(0.003)
    assert stack.config.training.entropy_anneal_to == pytest.approx(0.0)
    assert stack.config.training.actor_sampling_temperature == pytest.approx(0.25)
    assert stack.config.training.force_attack_over_pass_when_attack_legal is True
    assert stack.config.evaluation is not None
    assert stack.config.evaluation.eval_sampling_algorithm == "model_argmax_pinned_v1"


def test_load_stack_config_applies_anchorlane_v1_overrides() -> None:
    repo_root = _repo_root()
    stack = load_stack_config(repo_root / "configs/presets/typed_local_anchorlanes_v1.yaml")

    assert stack.config.league is not None
    assert stack.config.league.sampling.heuristic_public_reserved_envs_per_actor == 1
    assert stack.config.league.sampling.noleague_baseline_reserved_envs_per_actor == 1


def test_load_stack_config_supports_warmup_snapshot_mix_override(tmp_path: Path) -> None:
    repo_root = _repo_root()
    fake_repo = _temp_repo(tmp_path)
    (fake_repo / "configs" / "typed_thesis_locked.yaml").write_text(
        (repo_root / "configs/presets/typed_thesis_locked.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    local_path = fake_repo / "configs" / "typed_local.yaml"
    local_path.write_text(
        (repo_root / "configs/presets/typed_local.yaml")
        .read_text(encoding="utf-8")
        .replace(
            "    heuristic_public_mix_fraction: 0.1\n",
            "    heuristic_public_mix_fraction: 0.1\n    warmup_snapshot_mix_fraction: 0.35\n",
            1,
        ),
        encoding="utf-8",
    )

    stack = load_stack_config(local_path)

    assert stack.config.league is not None
    assert stack.config.league.sampling.warmup_snapshot_mix_fraction == pytest.approx(0.35)


def test_load_stack_config_supports_environment_deck_pool_override(tmp_path: Path) -> None:
    repo_root = _repo_root()
    fake_repo = _temp_repo(tmp_path)
    thesis_path = fake_repo / "configs" / "typed_thesis_locked.yaml"
    thesis_path.write_text(
        (repo_root / "configs/presets/typed_thesis_locked.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    config_path = fake_repo / "configs" / "typed_local.yaml"
    config_path.write_text(
        (
            (repo_root / "configs/presets/typed_local.yaml").read_text(encoding="utf-8")
            + "\nenvironment:\n"
            + "  deck_pool:\n"
            + "    - preset:main_deck_5hy_yotsuba_v1\n"
            + "    - preset:aggro_deck_5hy_nino_v1\n"
            + "  opponent_deck_pool:\n"
            + "    - preset:control_deck_jj_s66_v1\n"
        ),
        encoding="utf-8",
    )

    stack = load_stack_config(config_path)

    assert stack.config.environment is not None
    assert stack.config.environment.deck_pool == (
        "preset:main_deck_5hy_yotsuba_v1",
        "preset:aggro_deck_5hy_nino_v1",
    )
    assert stack.config.environment.opponent_deck_pool == ("preset:control_deck_jj_s66_v1",)


def test_load_stack_config_supports_diverse_batch_quota_overrides(tmp_path: Path) -> None:
    repo_root = _repo_root()
    fake_repo = _temp_repo(tmp_path)
    for preset_name in [
        "structured_acceptance.yaml",
        "typed_structured_v2.yaml",
        "typed_local.yaml",
        "typed_thesis_locked.yaml",
    ]:
        (fake_repo / "configs" / preset_name).write_text(
            (repo_root / f"configs/presets/{preset_name}").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    local_path = fake_repo / "configs" / "structured_acceptance_local_learning_frontier.yaml"
    local_path.write_text(
        (repo_root / "configs/presets/structured_acceptance_local_learning_frontier.yaml")
        .read_text(encoding="utf-8")
        .replace(
            "  actor_heuristic_final_fraction: 0.0\n",
            "  actor_heuristic_final_fraction: 0.0\n"
            "  train_on_heuristic_actor_rows: false\n"
            "  diverse_opponent_actor_count: 4\n"
            "  diverse_model_actor_count: 2\n"
            "  diverse_opponent_batch_fraction: 0.125\n"
            "  diverse_opponent_batch_wait_ms: 250\n",
            1,
        ),
        encoding="utf-8",
    )

    stack = load_stack_config(local_path)

    training = _require_training_config(stack.config.training)
    assert training.diverse_opponent_actor_count == 4
    assert training.diverse_model_actor_count == 2
    assert training.train_on_heuristic_actor_rows is False
    assert training.diverse_opponent_batch_fraction == pytest.approx(0.125)
    assert training.diverse_opponent_batch_wait_ms == 250


def test_canonical_config_hash_is_stable() -> None:
    repo_root = _repo_root()
    stack = load_stack_config(repo_root / "configs/presets/typed_thesis_locked.yaml")

    canonical = canonical_config_json(stack)
    assert '"experiment":{"role":"main"}' in canonical
    assert compute_config_hash256(stack) == compute_config_hash256(
        load_stack_config(repo_root / "configs/presets/typed_thesis_locked.yaml")
    )


@pytest.mark.parametrize(
    ("preset", "expected_hash", "expected_json_length"),
    [
        (
            "structured_acceptance_standard.yaml",
            "da462235409cf8d81fb22e52a868e6ac408c67121fd95925e56d820180197285",
            12508,
        ),
        (
            "structured_acceptance_standard_auto_gpu.yaml",
            "cb66aae32a7864c0af98ff11da5582ffe756ced1fd90f79618fd7806875ae38b",
            12519,
        ),
        (
            "structured_acceptance_standard_thesis_eval.yaml",
            "092f265d655f144d06bddde414744b5d09edcd4866bdbe63844899c6bd4dcb07",
            12727,
        ),
        (
            "structured_acceptance_standard_multideck.yaml",
            "917e5dfa56deef836ac46cc1a1a70f1cafb3d0723b484cf5a63a9cf6aae2d5fe",
            12696,
        ),
    ],
)
def test_structured_acceptance_public_preset_canonical_hashes_are_pinned(
    preset: str,
    expected_hash: str,
    expected_json_length: int,
) -> None:
    repo_root = _repo_root()
    stack = load_stack_config(repo_root / "configs" / "presets" / preset)
    canonical = canonical_config_json(stack)

    assert compute_config_hash256(stack) == expected_hash
    assert len(canonical) == expected_json_length
    assert canonical.startswith('{"config":{"curriculum":{"checkpoint_guard":')


def test_load_stack_config_reads_canonical_run_artifact_json(tmp_path: Path) -> None:
    repo_root = _repo_root()
    baseline = load_stack_config(repo_root / "configs/presets/typed_local_longhorizon_v1.yaml")
    fake_repo = _temp_repo(tmp_path)
    artifact_path = fake_repo / "runs" / "example_run" / "config_canonical.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(json.dumps(canonical_config_dict(baseline)))
    payload["config"]["league"]["sampling"].pop("heuristic_public_reserved_envs_per_actor", None)
    payload["config"]["league"]["sampling"].pop("noleague_baseline_mix_fraction", None)
    payload["config"]["league"]["sampling"].pop("noleague_baseline_mix_end_updates", None)
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


def test_load_stack_config_supports_model_candidate_scoring_chunk_overrides(tmp_path: Path) -> None:
    repo_root = _repo_root()
    fake_repo = _temp_repo(tmp_path)
    thesis_path = fake_repo / "configs" / "typed_thesis_locked.yaml"
    thesis_path.write_text(
        (repo_root / "configs/presets/typed_thesis_locked.yaml")
        .read_text(encoding="utf-8")
        .replace(
            "  recurrent_core: gru\n",
            "  recurrent_core: gru\n"
            "  candidate_scoring_chunk_size: 131072\n"
            "  cuda_learner_candidate_scoring_chunk_size: 524288\n",
            1,
        ),
        encoding="utf-8",
    )
    stack_path = fake_repo / "configs" / "typed_local.yaml"
    stack_path.write_text(
        (repo_root / "configs/presets/typed_local.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    stack = load_stack_config(stack_path)

    assert stack.config.model is not None
    assert stack.config.model.candidate_scoring_chunk_size == 131072
    assert stack.config.model.cuda_learner_candidate_scoring_chunk_size == 524288


def test_load_stack_config_supports_structured_policy_contract_override(tmp_path: Path) -> None:
    repo_root = _repo_root()
    fake_repo = _temp_repo(tmp_path)
    (fake_repo / "configs" / "typed_local.yaml").write_text(
        (repo_root / "configs/presets/typed_local.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (fake_repo / "configs" / "typed_thesis_locked.yaml").write_text(
        (repo_root / "configs/presets/typed_thesis_locked.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    structured_path = fake_repo / "configs" / "typed_structured_v2.yaml"
    structured_path.write_text(
        (repo_root / "configs/presets/typed_structured_v2.yaml")
        .read_text(encoding="utf-8")
        .replace(
            "  encoder_kind: structured_v2\n",
            "  encoder_kind: structured_v2\n  structured_policy_contract: factorized_v1\n",
            1,
        ),
        encoding="utf-8",
    )

    stack = load_stack_config(structured_path)

    assert stack.config.model is not None
    assert stack.config.model.structured_policy_contract == "factorized_v1"


def test_load_stack_config_supports_public_heuristic_bias_override(tmp_path: Path) -> None:
    repo_root = _repo_root()
    fake_repo = _temp_repo(tmp_path)
    (fake_repo / "configs" / "typed_local.yaml").write_text(
        (repo_root / "configs/presets/typed_local.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (fake_repo / "configs" / "typed_thesis_locked.yaml").write_text(
        (repo_root / "configs/presets/typed_thesis_locked.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    structured_path = fake_repo / "configs" / "typed_structured_v2.yaml"
    structured_path.write_text(
        (repo_root / "configs/presets/typed_structured_v2.yaml")
        .read_text(encoding="utf-8")
        .replace(
            "  encoder_kind: structured_v2\n",
            "  encoder_kind: structured_v2\n"
            "  public_heuristic_logit_bias_scale: 0.75\n"
            "  public_heuristic_logit_bias_families:\n"
            "    - attack\n"
            "    - main_move\n",
            1,
        ),
        encoding="utf-8",
    )

    stack = load_stack_config(structured_path)

    assert stack.config.model is not None
    assert stack.config.model.public_heuristic_logit_bias_scale == pytest.approx(0.75)
    assert stack.config.model.public_heuristic_actor_logit_bias_scale == pytest.approx(-1.0)
    assert stack.config.model.public_heuristic_logit_bias_families == ("attack", "main_move")


def test_load_stack_config_supports_public_actor_heuristic_bias_override(tmp_path: Path) -> None:
    repo_root = _repo_root()
    fake_repo = _temp_repo(tmp_path)
    (fake_repo / "configs" / "typed_local.yaml").write_text(
        (repo_root / "configs/presets/typed_local.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (fake_repo / "configs" / "typed_thesis_locked.yaml").write_text(
        (repo_root / "configs/presets/typed_thesis_locked.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    structured_path = fake_repo / "configs" / "typed_structured_v2.yaml"
    structured_path.write_text(
        (repo_root / "configs/presets/typed_structured_v2.yaml")
        .read_text(encoding="utf-8")
        .replace(
            "  encoder_kind: structured_v2\n",
            "  encoder_kind: structured_v2\n"
            "  public_heuristic_logit_bias_scale: 0.75\n"
            "  public_heuristic_actor_logit_bias_scale: 0.25\n",
            1,
        ),
        encoding="utf-8",
    )

    stack = load_stack_config(structured_path)

    assert stack.config.model is not None
    assert stack.config.model.public_heuristic_logit_bias_scale == pytest.approx(0.75)
    assert stack.config.model.public_heuristic_actor_logit_bias_scale == pytest.approx(0.25)


def test_load_stack_config_supports_guidance_anneal_overrides(tmp_path: Path) -> None:
    repo_root = _repo_root()
    fake_repo = _temp_repo(tmp_path)
    for preset_name in [
        "structured_acceptance_standard.yaml",
        "structured_acceptance.yaml",
        "structured_acceptance_tiny32_fast_pubcycle3_b1mix015_b1end10_tacticalbias_eval16_nostall_noconfirmdrop.yaml",
        "structured_acceptance_tiny32_fast_pubcycle3_b1mix015_b1end10_tacticalbias_eval16_nostall.yaml",
        "structured_acceptance_tiny32_fast_pubcycle3_b1mix015_b1end10_tacticalbias.yaml",
        "structured_acceptance_tiny32_fast_pubcycle3_b1mix015_b1end10.yaml",
        "structured_acceptance_tiny32_fast_pubcycle3_b1mix015.yaml",
        "structured_acceptance_tiny32_fast_pubcycle3.yaml",
        "structured_acceptance_tiny32_fast_packed.yaml",
        "structured_acceptance_linux_frontier.yaml",
        "typed_structured_v2.yaml",
        "typed_local.yaml",
        "typed_thesis_locked.yaml",
    ]:
        (fake_repo / "configs" / preset_name).write_text(
            (repo_root / f"configs/presets/{preset_name}").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    config_path = fake_repo / "configs" / "structured_acceptance_standard.yaml"
    config_path.write_text(
        (config_path.read_text(encoding="utf-8"))
        + "\nmodel:\n"
        + "  public_heuristic_logit_bias_start_updates: 12\n"
        + "  public_heuristic_logit_bias_end_updates: 48\n"
        + "  public_heuristic_logit_bias_final_scale: 0.25\n"
        + "training:\n"
        + "  actor_heuristic_start_updates: 8\n"
        + "  actor_heuristic_end_updates: 40\n"
        + "  actor_heuristic_final_fraction: 0.5\n"
        + "  structured_aux:\n"
        + "    teacher_public_heuristic_start_updates: 10\n"
        + "    teacher_public_heuristic_end_updates: 60\n"
        + "    teacher_public_heuristic_final_coef: 0.01\n",
        encoding="utf-8",
    )

    stack = load_stack_config(config_path)

    assert stack.config.model is not None
    assert stack.config.model.public_heuristic_logit_bias_start_updates == 12
    assert stack.config.model.public_heuristic_logit_bias_end_updates == 48
    assert stack.config.model.public_heuristic_logit_bias_final_scale == pytest.approx(0.25)
    assert stack.config.training is not None
    assert stack.config.training.actor_heuristic_start_updates == 8
    assert stack.config.training.actor_heuristic_end_updates == 40
    assert stack.config.training.actor_heuristic_final_fraction == pytest.approx(0.5)
    assert stack.config.training.teacher_public_heuristic_start_updates == 10
    assert stack.config.training.teacher_public_heuristic_end_updates == 60
    assert stack.config.training.teacher_public_heuristic_final_coef == pytest.approx(0.01)


def test_load_stack_config_supports_supervised_teacher_fade_override(tmp_path: Path) -> None:
    repo_root = _repo_root()
    fake_repo = _temp_repo(tmp_path)
    for preset_name in [
        "structured_acceptance_standard.yaml",
        "structured_acceptance.yaml",
        "structured_acceptance_tiny32_fast_pubcycle3_b1mix015_b1end10_tacticalbias_eval16_nostall_noconfirmdrop.yaml",
        "structured_acceptance_tiny32_fast_pubcycle3_b1mix015_b1end10_tacticalbias_eval16_nostall.yaml",
        "structured_acceptance_tiny32_fast_pubcycle3_b1mix015_b1end10_tacticalbias.yaml",
        "structured_acceptance_tiny32_fast_pubcycle3_b1mix015_b1end10.yaml",
        "structured_acceptance_tiny32_fast_pubcycle3_b1mix015.yaml",
        "structured_acceptance_tiny32_fast_pubcycle3.yaml",
        "structured_acceptance_tiny32_fast_packed.yaml",
        "structured_acceptance_linux_frontier.yaml",
        "typed_structured_v2.yaml",
        "typed_local.yaml",
        "typed_thesis_locked.yaml",
    ]:
        (fake_repo / "configs" / preset_name).write_text(
            (repo_root / f"configs/presets/{preset_name}").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    config_path = fake_repo / "configs" / "structured_acceptance_standard.yaml"
    config_path.write_text(
        (config_path.read_text(encoding="utf-8"))
        + "\ntraining:\n"
        + "  structured_aux:\n"
        + "    teacher_supervised_start_updates: 25\n"
        + "    teacher_supervised_end_updates: 75\n"
        + "    teacher_supervised_final_scale: 0.2\n",
        encoding="utf-8",
    )

    stack = load_stack_config(config_path)

    assert stack.config.training is not None
    assert stack.config.training.teacher_supervised_start_updates == 25
    assert stack.config.training.teacher_supervised_end_updates == 75
    assert stack.config.training.teacher_supervised_final_scale == pytest.approx(0.2)


def test_load_stack_config_supports_actor_policy_backend_override(tmp_path: Path) -> None:
    repo_root = _repo_root()
    fake_repo = _temp_repo(tmp_path)
    (fake_repo / "configs" / "typed_local.yaml").write_text(
        (repo_root / "configs/presets/typed_local.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (fake_repo / "configs" / "typed_thesis_locked.yaml").write_text(
        (repo_root / "configs/presets/typed_thesis_locked.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    structured_path = fake_repo / "configs" / "typed_structured_v2.yaml"
    structured_path.write_text(
        (repo_root / "configs/presets/typed_structured_v2.yaml")
        .read_text(encoding="utf-8")
        .replace(
            "  fixed_opponent_backend: python_scalar\n",
            "  fixed_opponent_backend: python_scalar\n"
            "  actor_policy_backend: heuristic_public\n"
            "  actor_heuristic_fraction: 0.5\n",
            1,
        ),
        encoding="utf-8",
    )

    stack = load_stack_config(structured_path)

    assert stack.config.training is not None
    assert stack.config.training.actor_policy_backend == "heuristic_public"
    assert stack.config.training.actor_heuristic_fraction == pytest.approx(0.5)
    assert stack.config.training.actor_heuristic_end_updates == -1
    assert stack.config.training.actor_heuristic_final_fraction == pytest.approx(0.5)


def test_load_stack_config_supports_actor_heuristic_schedule_override(tmp_path: Path) -> None:
    repo_root = _repo_root()
    fake_repo = _temp_repo(tmp_path)
    (fake_repo / "configs" / "typed_local.yaml").write_text(
        (repo_root / "configs/presets/typed_local.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (fake_repo / "configs" / "typed_thesis_locked.yaml").write_text(
        (repo_root / "configs/presets/typed_thesis_locked.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    structured_path = fake_repo / "configs" / "typed_structured_v2.yaml"
    structured_path.write_text(
        (repo_root / "configs/presets/typed_structured_v2.yaml")
        .read_text(encoding="utf-8")
        .replace(
            "  fixed_opponent_backend: python_scalar\n",
            "  fixed_opponent_backend: python_scalar\n"
            "  actor_policy_backend: heuristic_public\n"
            "  actor_heuristic_fraction: 1.0\n"
            "  actor_heuristic_end_updates: 5\n"
            "  actor_heuristic_final_fraction: 0.25\n",
            1,
        ),
        encoding="utf-8",
    )

    stack = load_stack_config(structured_path)

    assert stack.config.training is not None
    assert stack.config.training.actor_policy_backend == "heuristic_public"
    assert stack.config.training.actor_heuristic_fraction == pytest.approx(1.0)
    assert stack.config.training.actor_heuristic_end_updates == 5
    assert stack.config.training.actor_heuristic_final_fraction == pytest.approx(0.25)


def test_load_stack_config_supports_noleague_baseline_mix_override(tmp_path: Path) -> None:
    repo_root = _repo_root()
    fake_repo = _temp_repo(tmp_path)
    (fake_repo / "configs" / "typed_local.yaml").write_text(
        (repo_root / "configs/presets/typed_local.yaml")
        .read_text(encoding="utf-8")
        .replace(
            "    heuristic_public_mix_fraction: 0.1\n",
            "    heuristic_public_mix_fraction: 0.1\n"
            "    noleague_baseline_mix_fraction: 0.2\n"
            "    noleague_baseline_mix_end_updates: 5\n",
            1,
        ),
        encoding="utf-8",
    )
    (fake_repo / "configs" / "typed_thesis_locked.yaml").write_text(
        (repo_root / "configs/presets/typed_thesis_locked.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    structured_path = fake_repo / "configs" / "typed_structured_v2.yaml"
    structured_path.write_text(
        (repo_root / "configs/presets/typed_structured_v2.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    stack = load_stack_config(structured_path)

    assert stack.config.league is not None
    assert stack.config.league.sampling.noleague_baseline_mix_fraction == pytest.approx(0.2)
    assert stack.config.league.sampling.noleague_baseline_mix_end_updates == 5


def test_load_stack_config_supports_heuristic_public_mix_schedule_override(tmp_path: Path) -> None:
    repo_root = _repo_root()
    fake_repo = _temp_repo(tmp_path)
    (fake_repo / "configs" / "typed_local.yaml").write_text(
        (repo_root / "configs/presets/typed_local.yaml")
        .read_text(encoding="utf-8")
        .replace(
            "    heuristic_public_mix_fraction: 0.1\n",
            "    heuristic_public_mix_fraction: 0.1\n"
            "    heuristic_public_mix_end_updates: 5\n"
            "    heuristic_public_final_mix_fraction: 0.25\n"
            "    heuristic_public_variant_mix_fraction: 0.2\n"
            "    heuristic_public_variant_mix_end_updates: 7\n"
            "    heuristic_public_variant_final_mix_fraction: 0.05\n",
            1,
        ),
        encoding="utf-8",
    )
    (fake_repo / "configs" / "typed_thesis_locked.yaml").write_text(
        (repo_root / "configs/presets/typed_thesis_locked.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    structured_path = fake_repo / "configs" / "typed_structured_v2.yaml"
    structured_path.write_text(
        (repo_root / "configs/presets/typed_structured_v2.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    stack = load_stack_config(structured_path)

    assert stack.config.league is not None
    assert stack.config.league.sampling.heuristic_public_mix_fraction == pytest.approx(0.1)
    assert stack.config.league.sampling.heuristic_public_mix_end_updates == 5
    assert stack.config.league.sampling.heuristic_public_final_mix_fraction == pytest.approx(0.25)
    assert stack.config.league.sampling.heuristic_public_variant_mix_fraction == pytest.approx(0.2)
    assert stack.config.league.sampling.heuristic_public_variant_mix_end_updates == 7
    assert stack.config.league.sampling.heuristic_public_variant_final_mix_fraction == pytest.approx(0.05)


def test_load_stack_config_supports_heuristic_actor_hidden_state_tracking_override(tmp_path: Path) -> None:
    repo_root = _repo_root()
    fake_repo = _temp_repo(tmp_path)
    (fake_repo / "configs" / "typed_local.yaml").write_text(
        (repo_root / "configs/presets/typed_local.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (fake_repo / "configs" / "typed_thesis_locked.yaml").write_text(
        (repo_root / "configs/presets/typed_thesis_locked.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    structured_path = fake_repo / "configs" / "typed_structured_v2.yaml"
    structured_path.write_text(
        (repo_root / "configs/presets/typed_structured_v2.yaml")
        .read_text(encoding="utf-8")
        .replace(
            "  fixed_opponent_backend: python_scalar\n",
            "  fixed_opponent_backend: python_scalar\n  heuristic_actor_hidden_state_tracking: false\n",
            1,
        ),
        encoding="utf-8",
    )

    stack = load_stack_config(structured_path)

    assert stack.config.training is not None
    assert stack.config.training.heuristic_actor_hidden_state_tracking is False


def test_load_stack_config_supports_collection_backend_override(tmp_path: Path) -> None:
    repo_root = _repo_root()
    fake_repo = _temp_repo(tmp_path)
    (fake_repo / "configs" / "typed_local.yaml").write_text(
        (repo_root / "configs/presets/typed_local.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (fake_repo / "configs" / "typed_thesis_locked.yaml").write_text(
        (repo_root / "configs/presets/typed_thesis_locked.yaml")
        .read_text(encoding="utf-8")
        .replace(
            "  mp_start_method: spawn\n",
            "  mp_start_method: spawn\n  collection_backend: process\n",
            1,
        ),
        encoding="utf-8",
    )
    structured_path = fake_repo / "configs" / "typed_structured_v2.yaml"
    structured_path.write_text(
        (repo_root / "configs/presets/typed_structured_v2.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    stack = load_stack_config(structured_path)

    assert stack.config.system is not None
    assert stack.config.system.collection_backend == "process"


def test_load_stack_config_supports_public_heuristic_teacher_override(tmp_path: Path) -> None:
    repo_root = _repo_root()
    fake_repo = _temp_repo(tmp_path)
    (fake_repo / "configs" / "typed_local.yaml").write_text(
        (repo_root / "configs/presets/typed_local.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (fake_repo / "configs" / "typed_thesis_locked.yaml").write_text(
        (repo_root / "configs/presets/typed_thesis_locked.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    structured_path = fake_repo / "configs" / "typed_structured_v2.yaml"
    structured_path.write_text(
        (repo_root / "configs/presets/typed_structured_v2.yaml")
        .read_text(encoding="utf-8")
        .replace(
            "    teacher_action_coef: 0.10\n",
            "    teacher_action_coef: 0.10\n"
            "    teacher_public_heuristic_coef: 0.30\n"
            "    teacher_public_heuristic_temperature: 24.0\n"
            "    teacher_public_heuristic_families:\n"
            "      - attack\n"
            "      - main_move\n"
            "    teacher_public_heuristic_profiles:\n"
            "      - aggressive\n"
            "      - control\n"
            "    teacher_public_heuristic_profile_mode: cycle\n"
            "    teacher_public_heuristic_profiles_end_updates: 20\n",
            1,
        )
        .replace(
            "    teacher_action_coef: 0.50\n",
            "    teacher_action_coef: 0.50\n"
            "    teacher_public_heuristic_coef: 0.80\n"
            "    teacher_public_heuristic_temperature: 12.0\n"
            "    teacher_public_heuristic_families:\n"
            "      - main_play_character\n"
            "    teacher_public_heuristic_profiles:\n"
            "      - base\n"
            "      - aggressive\n"
            "    teacher_public_heuristic_profile_mode: mixture\n"
            "    teacher_public_heuristic_profiles_end_updates: 1\n",
            1,
        ),
        encoding="utf-8",
    )

    stack = load_stack_config(structured_path)

    assert stack.config.training is not None
    assert stack.config.training.teacher_public_heuristic_coef == pytest.approx(0.30)
    assert stack.config.training.teacher_public_heuristic_temperature == pytest.approx(24.0)
    assert stack.config.training.teacher_public_heuristic_families == ("attack", "main_move")
    assert stack.config.training.teacher_public_heuristic_profiles == ("aggressive", "control")
    assert stack.config.training.teacher_public_heuristic_profile_mode == "cycle"
    assert stack.config.training.teacher_public_heuristic_profiles_end_updates == 20
    assert stack.config.training.structured_warmstart.teacher_public_heuristic_coef == pytest.approx(0.80)
    assert stack.config.training.structured_warmstart.teacher_public_heuristic_temperature == pytest.approx(12.0)
    assert stack.config.training.structured_warmstart.teacher_public_heuristic_families == ("main_play_character",)
    assert stack.config.training.structured_warmstart.teacher_public_heuristic_profiles == ("base", "aggressive")
    assert stack.config.training.structured_warmstart.teacher_public_heuristic_profile_mode == "mixture"
    assert stack.config.training.structured_warmstart.teacher_public_heuristic_profiles_end_updates == 1


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
