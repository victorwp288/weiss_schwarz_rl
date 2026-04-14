from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_config_example_module():
    module_path = Path(__file__).resolve().parents[3] / "examples" / "config_example.py"
    spec = importlib.util.spec_from_file_location("test_config_example_module", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_example_config_accepts_random_legal_policy(tmp_path: Path) -> None:
    module = _load_config_example_module()

    module.repo_root = lambda: tmp_path

    preset_path = tmp_path / "configs" / "preset.yaml"
    loop_path = tmp_path / "configs" / "minimal_loop.yaml"
    preset_path.parent.mkdir(parents=True)

    preset_path.write_text(
        "schema_version: 1\n"
        "experiment:\n"
        "  role: main\n"
        "system:\n"
        "  num_envs: 2\n"
        "model:\n"
        "  encoder_kind: typed_v1\n"
        "  encoder_width: 256\n"
        "  encoder_depth: 2\n"
        "  typed_feature_width: 64\n"
        "  recurrent_core: gru\n"
        "  hidden_size: 256\n"
        "  policy_head_dim: 256\n"
        "  value_head_dim: 256\n"
        "  dropout:\n"
        "    family_a: 0.0\n"
        "training:\n"
        "  algorithm: impala_vtrace_gru\n"
        "  rollout:\n"
        "    unroll_length: 8\n"
        "    batch_unrolls_per_update: 4\n"
        "  optimizer:\n"
        "    learning_rate: 0.0002\n"
        "    adam_betas: [0.9, 0.999]\n"
        "    adam_eps: 1.0e-08\n"
        "    weight_decay: 0.0\n"
        "    grad_norm_clip: 40.0\n"
        "    value_loss_coef: 0.5\n"
        "  exploration:\n"
        "    entropy_coef: 0.01\n"
        "    entropy_anneal_to: 0.001\n"
        "    entropy_anneal_steps_updates: 1000\n"
        "  precision:\n"
        "    mixed_precision: false\n"
        "  checkpointing:\n"
        "    checkpoint_interval_updates: 10\n"
        "    snapshot_interval_updates: 20\n"
        "    actor_reload_interval_updates: 5\n"
        "  vtrace:\n"
        "    rho_clip: 1.0\n"
        "    c_clip: 1.0\n"
        "environment:\n"
        "  observation_visibility: public\n"
        "  max_decisions: 2000\n"
        "  max_ticks: 100000\n"
        "  truncate_on_max_steps: true\n"
        "  deck_set: all\n"
        "  deck_set_sizes:\n"
        "    target: 10\n"
        "    asymmetry: 0\n"
        "rewards:\n"
        "  objective: terminal_pm1\n"
        "  discount:\n"
        "    gamma: 0.99\n"
        "  shaping:\n"
        "    enable_damage_shaping: false\n"
        "    damage_reward: 0.0\n"
        "  truncation:\n"
        "    reward: 0.0\n"
        "    bootstrap_value: true\n"
        "curriculum:\n"
        "  simulator: {}\n"
        "league:\n"
        "  enabled: false\n"
        "  warmup:\n"
        "    first_updates: 0\n"
        "    initial_window_episodes: 0\n"
        "    ramp_target_updates: 0\n"
        "    ramp_target_window_episodes: 0\n"
        "  pool:\n"
        "    recent_size: 4\n"
        "    champion_size: 1\n"
        "  sampling:\n"
        "    pfsp_power: 1.0\n"
        "    pfsp_epsilon_uniform: 0.2\n"
        "    pfsp_window_episodes: 100\n"
        "  promotion:\n"
        "    enabled: false\n"
        "    paired_seeds: 16\n"
        "    gate:\n"
        "      uncertainty_method: bayesian_bootstrap_seedlevel_v1\n"
        "      promotion_score_threshold: 0.55\n"
        "      max_truncation_rate: 0.2\n"
        "      anchor_sets:\n"
        "        required: [B0 RandomLegal]\n"
        "        optional: []\n"
        "      guardrails:\n"
        "        random_legal_max_prob_lt_half: 0.1\n"
        "        noleague_max_prob_lt_half: 0.1\n"
        "        heuristic_public_max_prob_lt_half: 0.1\n"
        "evaluation:\n"
        "  eval_device: cpu\n"
        "  eval_sampling_algorithm: pinned_cpu_cdf\n"
        "  eval_inference_mode: inference_mode\n"
        "  seat_swap: true\n"
        "  periodic_dev_eval_interval_updates: 0\n"
        "  periodic_dev_eval_paired_seeds: 16\n"
        "  legal_fingerprint_checks:\n"
        "    version: legal_fingerprint_v1\n"
        "    mismatch_policy: hard_fail\n"
        "  stop_rules:\n"
        "    max_updates: 10\n"
        "    max_samples: 0\n"
        "    max_wallclock_seconds: 0\n"
        "  decision_kind_tagging:\n"
        "    enabled: true\n"
        "  fixed_anchor_sets:\n"
        "    dev: [B0 RandomLegal]\n"
        "    report: [B0 RandomLegal]\n"
        "  final_policy_set_selection:\n"
        "    include_random_legal_baseline_b0: true\n"
        "    include_no_league_baseline_b1: false\n"
        "    include_heuristic_public_b2_if_exists: false\n"
        "    include_final_champion_snapshot: false\n"
        "    include_spaced_snapshots_near_percent_updates: []\n"
        "  policy_set_size: 2\n"
        "reproducibility:\n"
        "  ids:\n"
        "    run_id: run_id256\n"
        "    config_hash: config_hash256\n"
        "    spec_hash: spec_hash256\n"
        "  seed_derivation:\n"
        "    generator: splitmix64\n"
        "    actor_stream: actor_seed_v1\n"
        "    eval_stream: eval_seed_v1\n"
        "  legal_fingerprint:\n"
        "    version: legal_fingerprint_v1\n"
        "  spec_bundle_policy:\n"
        "    capture_observation_spec: true\n"
        "    capture_action_spec: true\n",
        encoding="utf-8",
    )
    loop_path.write_text(
        "minimal_loop:\n  action_policy: random_legal\n",
        encoding="utf-8",
    )

    config = module.load_example_config(preset_config_path=preset_path, loop_config_path=loop_path)

    assert config.action_policy == "random_legal"
