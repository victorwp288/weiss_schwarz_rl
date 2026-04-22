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
    thesis_model = load_stack_config(repo_root / "configs/presets/structured_acceptance_thesis_model_auto_gpu.yaml")
    assert thesis_model.config.evaluation is not None
    assert thesis_model.seed_sets["dev_eval"] == repo_root / "configs/seeds/dev_eval_seeds.txt"
    assert thesis_model.seed_sets["promotion_gate"] == repo_root / "configs/seeds/promotion_eval_seeds.txt"
    assert thesis_model.seed_sets["report_eval"] == repo_root / "configs/seeds/report_eval_seeds.txt"

    noleague = load_stack_config(repo_root / "configs/presets/baselines/noleague_impala.yaml")
    assert noleague.config.experiment is not None
    assert noleague.config.experiment.role == "baseline_noleague"
    assert noleague.config.league is not None
    assert noleague.config.league.enabled is False
    assert noleague.config.evaluation is not None
    assert noleague.config.evaluation.periodic_dev_eval_interval_updates == 10
    assert noleague.config.curriculum is not None
    assert noleague.config.curriculum.early_cutoff.enabled is True
    assert noleague.config.model is not None
    assert thesis_model.config.model is not None
    assert noleague.config.model.gru_hidden_size == thesis_model.config.model.gru_hidden_size
    assert noleague.config.model.encoder_mlp_width == thesis_model.config.model.encoder_mlp_width
    assert noleague.config.model.typed_feature_width == thesis_model.config.model.typed_feature_width

    norecurrence = load_stack_config(repo_root / "configs/presets/baselines/norecurrence_impala.yaml")
    assert norecurrence.config.experiment is not None
    assert norecurrence.config.experiment.role == "baseline_norecurrence"
    assert norecurrence.config.model is not None
    assert norecurrence.config.model.recurrent_core == "none"
    assert norecurrence.config.model.encoder_kind == thesis_model.config.model.encoder_kind
    assert norecurrence.config.model.gru_hidden_size == thesis_model.config.model.gru_hidden_size
    assert norecurrence.config.model.encoder_mlp_width == thesis_model.config.model.encoder_mlp_width
    assert norecurrence.config.model.typed_feature_width == thesis_model.config.model.typed_feature_width
    assert norecurrence.config.training is not None
    assert norecurrence.config.training.algorithm == "impala_vtrace_ff"

    norecurrence_noleague = load_stack_config(repo_root / "configs/presets/baselines/norecurrence_impala_noleague.yaml")
    assert norecurrence_noleague.config.experiment is not None
    assert norecurrence_noleague.config.experiment.role == "baseline_noleague"
    assert norecurrence_noleague.config.league is not None
    assert norecurrence_noleague.config.league.enabled is False
    assert norecurrence_noleague.config.model is not None
    assert norecurrence_noleague.config.model.recurrent_core == "none"
    assert norecurrence_noleague.config.training is not None
    assert norecurrence_noleague.config.training.algorithm == "impala_vtrace_ff"
    assert norecurrence.config.curriculum is not None
    assert norecurrence.config.curriculum.early_cutoff.enabled is True

    ppo = load_stack_config(repo_root / "configs/presets/baselines/ppo_lite.yaml")
    assert ppo.config.experiment is not None
    assert ppo.config.experiment.role == "baseline_ppo_lite"
    assert ppo.config.model is not None
    assert ppo.config.model.encoder_kind == thesis_model.config.model.encoder_kind
    assert ppo.config.model.gru_hidden_size == thesis_model.config.model.gru_hidden_size
    assert ppo.config.model.encoder_mlp_width == thesis_model.config.model.encoder_mlp_width
    assert ppo.config.model.typed_feature_width == thesis_model.config.model.typed_feature_width
    assert ppo.config.training is not None
    assert ppo.config.training.algorithm == "ppo_lite_masked_v1"
    assert ppo.config.training.ppo_epochs == 4

    reward = load_stack_config(
        repo_root / "configs/presets/ablations/structured_acceptance_thesis_model_reward_shaping_auto_gpu.yaml"
    )
    assert reward.config.experiment is not None
    assert reward.config.experiment.role == "ablation_reward"
    assert reward.config.rewards is not None
    assert reward.config.rewards.objective == "terminal_pm1"
    assert reward.config.rewards.shaping.enable_damage_shaping is True
    assert reward.config.rewards.shaping.damage_reward == pytest.approx(0.05)
    assert reward.config.rewards.truncation.reward == pytest.approx(-0.1)
    assert reward.config.rewards.truncation.bootstrap_value is False

    reward_noleague = load_stack_config(
        repo_root / "configs/presets/baselines/structured_acceptance_thesis_model_reward_shaping_auto_gpu_noleague.yaml"
    )
    assert reward_noleague.config.experiment is not None
    assert reward_noleague.config.experiment.role == "baseline_noleague"
    assert reward_noleague.config.league is not None
    assert reward_noleague.config.league.enabled is False
    assert ppo.config.curriculum is not None
    assert ppo.config.curriculum.early_cutoff.enabled is True


def test_load_stack_config_applies_antistall_v2_overrides(tmp_path: Path) -> None:
    repo_root = _repo_root()
    fake_repo = _temp_repo(tmp_path)
    (fake_repo / "configs" / "typed_thesis_locked.yaml").write_text(
        (repo_root / "configs/presets/typed_thesis_locked.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    config_path = fake_repo / "configs" / "typed_local_antistall_v2.yaml"
    config_path.write_text(
        (repo_root / "configs/presets/typed_local.yaml").read_text(encoding="utf-8")
        + "\nrewards:\n"
        + "  shaping:\n"
        + "    damage_reward: 0.015\n"
        + "    no_progress_penalty: 0.015\n"
        + "  truncation:\n"
        + "    reward: -0.6\n"
        + "curriculum:\n"
        + "  simulator:\n"
        + "    max_no_progress_decisions: 64\n",
        encoding="utf-8",
    )
    stack = load_stack_config(config_path)

    assert stack.config.rewards is not None
    assert stack.config.rewards.shaping.damage_reward == pytest.approx(0.015)
    assert stack.config.rewards.shaping.no_progress_penalty == pytest.approx(0.015)
    assert stack.config.rewards.truncation.reward == pytest.approx(-0.6)
    assert stack.config.curriculum is not None
    assert stack.config.curriculum.simulator["max_no_progress_decisions"] == 64


def test_load_stack_config_applies_longhorizon_v1_overrides(tmp_path: Path) -> None:
    repo_root = _repo_root()
    fake_repo = _temp_repo(tmp_path)
    (fake_repo / "configs" / "typed_thesis_locked.yaml").write_text(
        (repo_root / "configs/presets/typed_thesis_locked.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    config_path = fake_repo / "configs" / "typed_local_longhorizon_v1.yaml"
    config_path.write_text(
        (repo_root / "configs/presets/typed_local.yaml").read_text(encoding="utf-8")
        + "\nrewards:\n"
        + "  discount:\n"
        + "    gamma: 1.0\n"
        + "  shaping:\n"
        + "    damage_reward: 0.0\n"
        + "    no_progress_penalty: 0.01\n",
        encoding="utf-8",
    )
    stack = load_stack_config(config_path)

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


def test_load_stack_config_supports_frozen_thesis_model_auto_gpu_presets() -> None:
    repo_root = _repo_root()

    thesis_model = load_stack_config(repo_root / "configs/presets/structured_acceptance_thesis_model_auto_gpu.yaml")
    assert thesis_model.config.system is not None
    assert thesis_model.config.system.learner_device == "cuda:auto"
    assert thesis_model.config.system.actor_device == "cuda:auto"
    assert thesis_model.config.model is not None
    assert thesis_model.config.model.gru_hidden_size == 248
    assert thesis_model.config.model.encoder_mlp_width == 248
    assert thesis_model.config.model.typed_feature_width == 62
    assert thesis_model.config.evaluation is not None
    assert thesis_model.config.evaluation.async_periodic_dev_eval_enabled is True
    assert thesis_model.config.evaluation.periodic_dev_eval_interval_updates == 10
    assert thesis_model.config.evaluation.periodic_dev_eval_parallel_workers == 6
    assert thesis_model.config.evaluation.periodic_dev_eval_parallel_worker_devices == (
        "cuda:0",
        "cuda:1",
        "cuda:2",
        "cuda:0",
        "cuda:1",
        "cuda:2",
    )
    assert thesis_model.config.curriculum.early_cutoff.enabled is True
    assert thesis_model.config.curriculum.early_cutoff.warmup_updates == 120
    assert thesis_model.config.curriculum.early_cutoff.patience_updates == 120
    assert thesis_model.config.curriculum.early_cutoff.min_improvement == pytest.approx(0.01)
    assert thesis_model.config.curriculum.early_cutoff.stall_patience_evals == 4
    assert thesis_model.config.curriculum.early_cutoff.stall_rate_threshold == pytest.approx(0.25)
    assert thesis_model.config.league is not None
    assert thesis_model.config.league.promotion_gate.async_enabled is False
    assert thesis_model.config.league.promotion_gate.parallel_workers == 6
    assert thesis_model.config.league.promotion_gate.parallel_worker_devices == (
        "cuda:0",
        "cuda:1",
        "cuda:2",
        "cuda:0",
        "cuda:1",
        "cuda:2",
    )

    thesis_model_noleague = load_stack_config(
        repo_root / "configs/presets/baselines/structured_acceptance_thesis_model_auto_gpu_noleague.yaml"
    )
    assert thesis_model_noleague.config.experiment is not None
    assert thesis_model_noleague.config.experiment.role == "baseline_noleague"
    assert thesis_model_noleague.config.league is not None
    assert thesis_model_noleague.config.league.enabled is False
    assert thesis_model_noleague.config.evaluation is not None
    assert thesis_model_noleague.config.evaluation.periodic_dev_eval_interval_updates == 10
    assert thesis_model_noleague.config.curriculum is not None
    assert thesis_model_noleague.config.curriculum.early_cutoff.enabled is True
    assert thesis_model_noleague.seed_sets["dev_eval"] == repo_root / "configs/seeds/dev_eval_seeds.txt"
    assert thesis_model_noleague.seed_sets["promotion_gate"] == repo_root / "configs/seeds/promotion_eval_seeds.txt"
    assert thesis_model_noleague.config.model is not None
    assert thesis_model_noleague.config.model.gru_hidden_size == 248
    assert thesis_model_noleague.config.model.encoder_mlp_width == 248
    assert thesis_model_noleague.config.model.typed_feature_width == 62

    thesis_model_eval = load_stack_config(
        repo_root / "configs/presets/structured_acceptance_thesis_model_eval_auto_gpu.yaml"
    )
    assert thesis_model_eval.config.evaluation is not None
    assert thesis_model_eval.config.evaluation.eval_device == "cuda:auto"
    assert (
        "B3 HeuristicPublicAggro"
        in thesis_model_eval.config.evaluation.final_policy_set_selection.fixed_anchor_set_v1.optional_if_available
    )

def test_load_stack_config_supports_current_thesis_facing_presets() -> None:
    repo_root = _repo_root()

    thesis_multideck = load_stack_config(
        repo_root / "configs/presets/structured_acceptance_thesis_model_multideck_auto_gpu.yaml"
    )
    assert thesis_multideck.config.environment is not None
    assert thesis_multideck.config.environment.deck_pool == (
        "preset:starter_v1",
        "preset:quints_balanced_v2",
        "preset:quints_ichika_focus_v1",
        "preset:quints_yotsuba_focus_v1",
        "preset:quints_support_mix_v1",
    )
    assert thesis_multideck.config.environment.opponent_deck_pool == (
        "preset:quints_support_mix_v1",
        "preset:quints_yotsuba_focus_v1",
        "preset:quints_balanced_v2",
        "preset:starter_v1",
        "preset:quints_ichika_focus_v1",
    )
    assert thesis_multideck.config.curriculum is not None
    assert thesis_multideck.config.curriculum.early_cutoff.enabled is True
    assert thesis_multideck.seed_sets["dev_eval"] == repo_root / "configs/seeds/dev_eval_seeds.txt"
    assert thesis_multideck.seed_sets["promotion_gate"] == repo_root / "configs/seeds/promotion_eval_seeds.txt"

    thesis_multideck_eval = load_stack_config(
        repo_root / "configs/presets/structured_acceptance_thesis_model_multideck_eval_auto_gpu.yaml"
    )
    assert thesis_multideck_eval.config.evaluation is not None
    assert (
        "B4 HeuristicPublicControl"
        in thesis_multideck_eval.config.evaluation.final_policy_set_selection.fixed_anchor_set_v1.optional_if_available
    )

    teacher_fade = load_stack_config(
        repo_root / "configs/presets/ablations/structured_acceptance_thesis_model_teacher_fade_auto_gpu.yaml"
    )
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
    assert teacher_fade.config.system is not None
    assert teacher_fade.config.system.learner_device == "cuda:auto"
    assert teacher_fade.config.system.actor_device == "cuda:auto"
    assert teacher_fade.config.system.collection_backend == "process"
    assert teacher_fade.config.curriculum is not None
    assert teacher_fade.config.curriculum.early_cutoff.enabled is True

    teacher_fade_eval = load_stack_config(
        repo_root / "configs/presets/ablations/structured_acceptance_thesis_model_teacher_fade_eval_auto_gpu.yaml"
    )
    assert teacher_fade_eval.config.evaluation is not None
    assert (
        "B3 HeuristicPublicAggro"
        in teacher_fade_eval.config.evaluation.final_policy_set_selection.fixed_anchor_set_v1.optional_if_available
    )

    ablate_no_tactical_bias = load_stack_config(
        repo_root / "configs/presets/ablations/structured_acceptance_thesis_model_no_tactical_bias_auto_gpu.yaml"
    )
    assert ablate_no_tactical_bias.config.model is not None
    assert ablate_no_tactical_bias.config.model.public_heuristic_logit_bias_scale == pytest.approx(1.0)
    assert ablate_no_tactical_bias.config.model.public_heuristic_logit_bias_families == ()
    assert ablate_no_tactical_bias.config.curriculum is not None
    assert ablate_no_tactical_bias.config.curriculum.early_cutoff.enabled is True

    ablate_no_tactical_bias_eval = load_stack_config(
        repo_root / "configs/presets/ablations/structured_acceptance_thesis_model_no_tactical_bias_eval_auto_gpu.yaml"
    )
    assert ablate_no_tactical_bias_eval.config.model is not None
    assert ablate_no_tactical_bias_eval.config.model.public_heuristic_logit_bias_families == ()

    ablate_no_b1_cutoff = load_stack_config(
        repo_root / "configs/presets/ablations/structured_acceptance_thesis_model_no_b1_cutoff_auto_gpu.yaml"
    )
    assert ablate_no_b1_cutoff.config.league is not None
    assert ablate_no_b1_cutoff.config.league.sampling.noleague_baseline_mix_end_updates == -1
    assert ablate_no_b1_cutoff.config.curriculum is not None
    assert ablate_no_b1_cutoff.config.curriculum.early_cutoff.enabled is True

    ablate_no_b1_cutoff_eval = load_stack_config(
        repo_root / "configs/presets/ablations/structured_acceptance_thesis_model_no_b1_cutoff_eval_auto_gpu.yaml"
    )
    assert ablate_no_b1_cutoff_eval.config.evaluation is not None

    thesis_multideck_noleague = load_stack_config(
        repo_root / "configs/presets/baselines/structured_acceptance_thesis_model_multideck_auto_gpu_noleague.yaml"
    )
    assert thesis_multideck_noleague.config.experiment is not None
    assert thesis_multideck_noleague.config.experiment.role == "baseline_noleague_multideck"
    assert thesis_multideck_noleague.config.league is not None
    assert thesis_multideck_noleague.config.league.enabled is False
    assert thesis_multideck_noleague.config.evaluation is not None
    assert thesis_multideck_noleague.config.evaluation.periodic_dev_eval_interval_updates == 10
    assert thesis_multideck_noleague.config.curriculum is not None
    assert thesis_multideck_noleague.config.curriculum.early_cutoff.enabled is True

    thesis_no_tactical_bias_noleague = load_stack_config(
        repo_root
        / "configs/presets/baselines/structured_acceptance_thesis_model_no_tactical_bias_auto_gpu_noleague.yaml"
    )
    assert thesis_no_tactical_bias_noleague.config.experiment is not None
    assert thesis_no_tactical_bias_noleague.config.experiment.role == "baseline_noleague_ablation_no_tactical_bias"
    assert thesis_no_tactical_bias_noleague.config.league is not None
    assert thesis_no_tactical_bias_noleague.config.league.enabled is False
    assert thesis_no_tactical_bias_noleague.config.evaluation is not None
    assert thesis_no_tactical_bias_noleague.config.evaluation.periodic_dev_eval_interval_updates == 10
    assert thesis_no_tactical_bias_noleague.config.curriculum is not None
    assert thesis_no_tactical_bias_noleague.config.curriculum.early_cutoff.enabled is True

    thesis_teacher_fade_noleague = load_stack_config(
        repo_root
        / "configs/presets/baselines/structured_acceptance_thesis_model_teacher_fade_auto_gpu_noleague.yaml"
    )
    assert thesis_teacher_fade_noleague.config.experiment is not None
    assert thesis_teacher_fade_noleague.config.experiment.role == "baseline_noleague_ablation_teacher_fade"
    assert thesis_teacher_fade_noleague.config.league is not None
    assert thesis_teacher_fade_noleague.config.league.enabled is False
    assert thesis_teacher_fade_noleague.config.evaluation is not None
    assert thesis_teacher_fade_noleague.config.evaluation.periodic_dev_eval_interval_updates == 10
    assert thesis_teacher_fade_noleague.config.curriculum is not None
    assert thesis_teacher_fade_noleague.config.curriculum.early_cutoff.enabled is True


def test_load_stack_config_applies_anchorlane_v1_overrides(tmp_path: Path) -> None:
    repo_root = _repo_root()
    fake_repo = _temp_repo(tmp_path)
    (fake_repo / "configs" / "typed_thesis_locked.yaml").write_text(
        (repo_root / "configs/presets/typed_thesis_locked.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    config_path = fake_repo / "configs" / "typed_local_anchorlanes_v1.yaml"
    config_path.write_text(
        (repo_root / "configs/presets/typed_local.yaml").read_text(encoding="utf-8")
        + "\nleague:\n"
        + "  sampling:\n"
        + "    heuristic_public_reserved_envs_per_actor: 1\n"
        + "    noleague_baseline_reserved_envs_per_actor: 1\n",
        encoding="utf-8",
    )
    stack = load_stack_config(config_path)

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
            + "    - preset:quints_balanced_v2\n"
            + "    - preset:quints_ichika_focus_v1\n"
            + "  opponent_deck_pool:\n"
            + "    - preset:quints_support_mix_v1\n"
        ),
        encoding="utf-8",
    )

    stack = load_stack_config(config_path)

    assert stack.config.environment is not None
    assert stack.config.environment.deck_pool == (
        "preset:quints_balanced_v2",
        "preset:quints_ichika_focus_v1",
    )
    assert stack.config.environment.opponent_deck_pool == ("preset:quints_support_mix_v1",)


def test_load_stack_config_supports_diverse_batch_quota_overrides(tmp_path: Path) -> None:
    repo_root = _repo_root()
    fake_repo = _temp_repo(tmp_path)
    for preset_name in [
        "typed_structured_v2.yaml",
        "typed_local.yaml",
        "typed_thesis_locked.yaml",
    ]:
        (fake_repo / "configs" / preset_name).write_text(
            (repo_root / f"configs/presets/{preset_name}").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    local_path = fake_repo / "configs" / "typed_local_diverse.yaml"
    local_path.write_text(
        (repo_root / "configs/presets/typed_local.yaml")
        .read_text(encoding="utf-8")
        .replace(
            "  checkpointing:\n",
            "  train_on_heuristic_actor_rows: false\n"
            "  diverse_opponent_actor_count: 4\n"
            "  diverse_model_actor_count: 2\n"
            "  diverse_opponent_batch_fraction: 0.125\n"
            "  diverse_opponent_batch_wait_ms: 250\n"
            "  checkpointing:\n",
            1,
        ),
        encoding="utf-8",
    )

    stack = load_stack_config(local_path)

    assert stack.config.training is not None
    assert stack.config.training.diverse_opponent_actor_count == 4
    assert stack.config.training.diverse_model_actor_count == 2
    assert stack.config.training.train_on_heuristic_actor_rows is False
    assert stack.config.training.diverse_opponent_batch_fraction == pytest.approx(0.125)
    assert stack.config.training.diverse_opponent_batch_wait_ms == 250


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
    fake_repo = _temp_repo(tmp_path)
    (fake_repo / "configs" / "typed_thesis_locked.yaml").write_text(
        (repo_root / "configs/presets/typed_thesis_locked.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    baseline_path = fake_repo / "configs" / "typed_local_longhorizon_v1.yaml"
    baseline_path.write_text(
        (repo_root / "configs/presets/typed_local.yaml").read_text(encoding="utf-8")
        + "\nrewards:\n"
        + "  discount:\n"
        + "    gamma: 1.0\n"
        + "  shaping:\n"
        + "    damage_reward: 0.0\n"
        + "    no_progress_penalty: 0.01\n",
        encoding="utf-8",
    )
    baseline = load_stack_config(baseline_path)
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
        "structured_acceptance_thesis_model_auto_gpu.yaml",
        "typed_structured_v2.yaml",
        "typed_local.yaml",
        "typed_thesis_locked.yaml",
    ]:
        (fake_repo / "configs" / preset_name).write_text(
            (repo_root / f"configs/presets/{preset_name}").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    config_path = fake_repo / "configs" / "structured_acceptance_thesis_model_auto_gpu.yaml"
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


def test_load_stack_config_supports_prefetch_and_native_rollout_overrides(tmp_path: Path) -> None:
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
            "  heuristic_native_rollout_enabled: true\n"
            "  collect_batch_prefetch_enabled: true\n"
            "  heuristic_actor_hidden_state_tracking: false\n",
            1,
        ),
        encoding="utf-8",
    )

    stack = load_stack_config(structured_path)

    assert stack.config.training is not None
    assert stack.config.training.collect_batch_prefetch_enabled is True
    assert stack.config.training.heuristic_native_rollout_enabled is True
    assert stack.config.training.heuristic_actor_hidden_state_tracking is False


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
