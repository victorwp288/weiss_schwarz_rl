# B1 Anchor Diagnosis Prompt For GPT Pro

I need help diagnosing and improving a weak RL baseline anchor run in my thesis repo. Please analyze this as a serious RL systems / training stability problem, not as a generic tuning question.

I want:
1. A diagnosis of why the B1 anchor is plateauing.
2. A ranked list of hypotheses.
3. A concrete experiment plan with the next 3-5 highest-value experiments.
4. Separation between:
   - low-value micro-tuning
   - structural changes likely to matter
5. Specific suggestions for what code/config I should change.
6. What evidence would confirm or falsify each hypothesis.

Please be concrete. Assume I care about scientific defensibility and reproducibility, not random hacks.

==================================================
PROJECT CONTEXT
==================================================

Project:
- Weiss Schwarz RL thesis project

Target artifact:
- a strong `B1 NoLeague baseline` anchor

Seed:
- `20260421` used consistently

Training family:
- structured acceptance / no-league training setup
- async periodic dev eval
- fixed-opponent baseline regime
- structured warmstart enabled

Primary problem:
- I fixed the catastrophic collapse / truncation issue.
- But the anchor is still not clearly strong against `B1 NoLeague baseline`.
- It plateaus early.
- Best `B1` score so far is only about `0.5625`.

What I want from you:
- Read the results and code/config excerpts below.
- Tell me why this is happening.
- Tell me what to try next.
- Tell me what NOT to waste time on.

==================================================
HIGH-LEVEL OBSERVATIONS
==================================================

What changed:
- The original long 200-update anchor collapsed badly later in training.
- Recovery runs with anti-stall shaping fixed the collapse.
- Truncations are now `0` in the best recovery/probe runs.
- But `B1` does not meaningfully improve beyond the mid-0.5s.

Observed pattern:
- Aggregate score can improve somewhat.
- `B3 HeuristicPublicAggro` can improve somewhat.
- `B4` is generally already strong.
- `B1` is the hard matchup and remains stubbornly flat.

My current interpretation:
- The bottleneck may no longer be optimizer tuning.
- The recipe may be hitting an early local ceiling.
- The first 10 updates may still be dominated by warmstart / teacher-guided behavior.
- Small LR / entropy changes do not seem to break through.

Please challenge or confirm that interpretation.

==================================================
BASE TRAINING / EVAL SETUP
==================================================

Base training entrypoint shape:

```bash
uv run python python/scripts/train.py \
  --stack-config configs/presets/baselines/structured_acceptance_thesis_model_auto_gpu_noleague.yaml \
  --run-label <label> \
  --num-envs 2048 \
  --unroll-length 64 \
  --runtime-mode train_async_fast \
  --max-updates <N> \
  --seed 20260421 \
  [config overrides]
```

Common environment facts from runtime logs:
- model kind: `structured_v2`
- collection backend: `process`
- fixed opponent backend: `simulator_native`
- actor policy backend during warm behavior: `heuristic_public`
- league disabled
- structured warmstart enabled
- async periodic dev eval every 10 updates
- dev eval anchors:
  - `B0 RandomLegal`
  - `B1 NoLeague baseline`
  - `B2 HeuristicPublic`
  - `B3 HeuristicPublicAggro`
  - `B4 HeuristicPublicControl`

Throughput:
- bounded probes usually reach roughly `25k–47k samples/s`
- so this does NOT look like a throughput bottleneck
- this looks like a policy-quality / training-dynamics problem

Eval budget in short probes:
- periodic dev eval paired seeds: `16`

==================================================
RESULTS SUMMARY
==================================================

Original long anchor:
- run label:
  `b1_anchor_thesis_model_seed20260421`
- went to 200 updates
- best early checkpoint:
  - update 10 aggregate: `0.8375`
  - `B1`: `0.546875`
  - `B3`: `0.671875`
  - `B4`: `0.96875`
  - truncations: `0`
- later collapsed badly and became unusable as final anchor

Recovery / probe experiments:

1. Strong anti-stall recovery
Run:
- `b1_anchor_thesis_model_seed20260421_recovery100_antistall`

Overrides:
- `training.optimizer.learning_rate=0.0003`
- `training.exploration.entropy_coef=0.02`
- `evaluation.periodic_dev_eval_paired_seeds=16`
- `curriculum.early_cutoff.warmup_updates=20`
- `curriculum.early_cutoff.patience_updates=20`
- `curriculum.early_cutoff.stall_patience_evals=2`
- `curriculum.early_cutoff.stall_rate_threshold=0.2`
- `rewards.shaping.no_progress_penalty=0.015`
- `rewards.truncation.reward=-0.6`

Results:
- update 10:
  - aggregate `0.825`
  - `B1=0.5625`
  - `B3=0.59375`
  - `B4=0.96875`
  - truncations `0`
- later plateau:
  - `B1=0.50–0.53125`
  - aggregate around `0.78–0.80`
- stable, but not “super solid”

2. Low-damage anti-stall variant
Run:
- `b1_anchor_thesis_model_seed20260421_recovery100_lowdamage_antistall`

Overrides:
- `learning_rate=0.0002`
- `entropy_coef=0.03`
- `damage_reward=0.015`
- `no_progress_penalty=0.015`
- `truncation.reward=-0.6`

Results:
- update 10:
  - aggregate `0.79375`
  - `B1=0.46875`
  - truncations `0`
- worse

3. Default-damage anti-stall variant
Run:
- `b1_anchor_thesis_model_seed20260421_recovery100_default_antistall`

Overrides:
- `learning_rate=0.0002`
- `entropy_coef=0.03`
- `damage_reward=0.05`
- `no_progress_penalty=0.015`
- `truncation.reward=-0.6`

Results:
- update 10:
  - aggregate `0.79375`
  - `B1=0.46875`
- update 20:
  - aggregate `0.78646`
  - `B1=0.53125`
- update 30:
  - aggregate `0.79167`
  - `B1=0.5000`
- update 40:
  - aggregate `0.79167`
  - `B1=0.53125`
- update 50:
  - aggregate `0.78646`
  - `B1=0.53125`
- truncations `0`
- stable but below best run

4. Mild truncation penalty probe
Run:
- `b1_anchor_thesis_model_seed20260421_probe_modtrunc10`

Overrides:
- `learning_rate=0.0003`
- `entropy_coef=0.02`
- `evaluation.periodic_dev_eval_paired_seeds=16`
- `rewards.shaping.no_progress_penalty=0.015`
- `rewards.truncation.reward=-0.3`

Results:
- update 10:
  - aggregate `0.825`
  - `B1=0.5625`
  - `B3=0.59375`
  - `B4=0.96875`
  - truncations `0`

5. Softer anti-stall probe
Run:
- `b1_anchor_thesis_model_seed20260421_probe_softantistall10`

Overrides:
- `learning_rate=0.0003`
- `entropy_coef=0.02`
- `evaluation.periodic_dev_eval_paired_seeds=16`
- `rewards.shaping.no_progress_penalty=0.01`
- `rewards.truncation.reward=-0.3`

Results:
- update 10:
  - aggregate `0.83125`
  - `B1=0.5625`
  - `B3=0.625`
  - `B4=0.96875`
  - truncations `0`

This is currently the best recovery/probe result.

6. Same soft-antistall recipe but lower entropy
Run:
- `b1_anchor_thesis_model_seed20260421_probe_softantistall_lowent10`

Overrides:
- `learning_rate=0.0003`
- `entropy_coef=0.015`
- `no_progress_penalty=0.01`
- `truncation.reward=-0.3`

Results:
- update 10:
  - aggregate `0.825`
  - `B1=0.53125`
  - `B3=0.625`
  - truncations `0`

7. Same soft-antistall recipe but higher LR
Run:
- `b1_anchor_thesis_model_seed20260421_probe_softantistall_hilr10`

Overrides:
- `learning_rate=0.00035`
- `entropy_coef=0.02`
- `no_progress_penalty=0.01`
- `truncation.reward=-0.3`

Results:
- update 10:
  - aggregate `0.825`
  - `B1=0.53125`
  - `B3=0.59375`
  - `B4=1.0`
  - truncations `0`

Current best recipe:
- `learning_rate=0.0003`
- `entropy_coef=0.02`
- `evaluation.periodic_dev_eval_paired_seeds=16`
- `rewards.shaping.no_progress_penalty=0.01`
- `rewards.truncation.reward=-0.3`

Current continuation:
- I also launched a 30-update continuation of the best recipe:
  - run label:
    `b1_anchor_thesis_model_seed20260421_softantistall30`
- It advanced through at least `checkpoint_20.pt`
- Training throughput remained healthy
- Async dev eval for update 10 was lagging behind training artifact creation
- I do not yet have the final update-20 B1 eval result from this continuation in the material below
- So please reason mainly from the completed results above

==================================================
INTERPRETATION I WANT YOU TO EVALUATE
==================================================

My working interpretation:
- Anti-stall shaping fixed collapse and truncation.
- But the anchor’s B1 matchup is still capped by the training recipe itself.
- Small LR / entropy changes do not change the first-10-update policy enough.
- The main remaining bottleneck may be structural:
  - warmstart / teacher influence
  - opponent mix / curriculum
  - reward design
  - termination policy
  - or the overall anchor training objective itself

Please tell me whether that is likely correct.

==================================================
RELEVANT CODE / CONFIG EXCERPTS
==================================================

------------------------------
1. B1 anchor preset
File:
`configs/presets/baselines/structured_acceptance_thesis_model_auto_gpu_noleague.yaml`
Purpose:
- main no-league baseline preset used for B1 anchor training

```yaml
schema_version: 2
description: canonical frozen structured_acceptance thesis model no-league baseline preset for Linux servers
extends: ../structured_acceptance_thesis_model_auto_gpu.yaml
experiment:
  role: baseline_noleague
evaluation:
  periodic_dev_eval_interval_updates: 10
curriculum:
  early_cutoff:
    enabled: true
    warmup_updates: 120
    patience_updates: 120
    min_improvement: 0.01
    stall_patience_evals: 4
    stall_rate_threshold: 0.25
league:
  enabled: false
```

------------------------------
2. Parent preset
File:
`configs/presets/structured_acceptance_thesis_model_auto_gpu.yaml`
Purpose:
- server thesis preset that adds multi-GPU actor sharding, warmstart, teacher/public-heuristic guidance, periodic dev eval, league sampling, early cutoff, and checkpoint guard settings

```yaml
schema_version: 2
description: canonical frozen structured_acceptance thesis model preset for Linux servers with automatic multi-GPU actor sharding
extends: typed_structured_v2.yaml
system:
  learner_device: cuda:auto
  actor_device: cuda:auto
  actor_process_count: 32
  learner_torch_threads: 16
  collection_backend: process
model:
  gru_hidden_size: 248
  encoder_mlp_width: 248
  typed_feature_width: 62
  public_heuristic_logit_bias_scale: 2.0
  public_heuristic_actor_logit_bias_scale: 1.0
  public_heuristic_logit_bias_final_scale: 2.0
  public_heuristic_logit_bias_families:
    - main_play_character
    - main_move
    - attack
training:
  profile_timers: true
  fixed_opponent_backend: simulator_native
  actor_policy_backend: heuristic_public
  actor_heuristic_fraction: 1.0
  structured_aux:
    teacher_public_heuristic_coef: 0.10
    teacher_public_heuristic_final_coef: 0.10
    teacher_public_heuristic_profiles:
      - base
      - aggressive
      - control
    teacher_public_heuristic_profile_mode: cycle
  structured_warmstart:
    updates: 1
    teacher_public_heuristic_coef: 0.50
    teacher_public_heuristic_profiles:
      - base
      - aggressive
      - control
    teacher_public_heuristic_profile_mode: cycle
evaluation:
  seed_files:
    dev_eval: configs/seeds/dev_eval_seeds.txt
    report_eval: configs/seeds/report_eval_seeds.txt
    promotion_gate: configs/seeds/promotion_eval_seeds.txt
  periodic_dev_eval_paired_seeds: 32
  eval_device: cuda:auto
  async_periodic_dev_eval_enabled: true
  periodic_dev_eval_interval_updates: 10
  periodic_dev_eval_parallel_workers: 6
  periodic_dev_eval_parallel_worker_devices:
    - cuda:0
    - cuda:1
    - cuda:2
    - cuda:0
    - cuda:1
    - cuda:2
league:
  sampling:
    heuristic_public_start_updates: 0
    heuristic_public_mix_fraction: 1.0
    heuristic_public_final_mix_fraction: 1.0
    noleague_baseline_mix_fraction: 0.15
    noleague_baseline_mix_end_updates: 10
  promotion:
    paired_seeds: 16
    seed_file: configs/seeds/promotion_eval_seeds.txt
    anchor_set_v1:
      required:
        - B0 RandomLegal
        - B1 NoLeague baseline
      optional_if_available:
        - B2 HeuristicPublic
        - B3 HeuristicPublicAggro
        - B4 HeuristicPublicControl
        - Previous champion snapshot
        - Previous recent snapshot
    gate:
      parallel_workers: 6
      parallel_worker_devices:
        - cuda:0
        - cuda:1
        - cuda:2
        - cuda:0
        - cuda:1
        - cuda:2
curriculum:
  stall_monitor:
    enabled: false
  early_cutoff:
    enabled: true
    warmup_updates: 120
    patience_updates: 120
    min_improvement: 0.01
    stall_patience_evals: 4
    stall_rate_threshold: 0.25
  checkpoint_guard:
    rollback_score_margin: 0.01
reproducibility:
  seed_files:
    dev_eval: configs/seeds/dev_eval_seeds.txt
    promotion_gate: configs/seeds/promotion_eval_seeds.txt
```

------------------------------
3. Inherited structured learner preset
File:
`configs/presets/typed_structured_v2.yaml`
Purpose:
- base structured IMPALA preset that defines algorithm, teacher auxiliary defaults, and warmstart defaults

```yaml
schema_version: 2
description: typed_structured_v2 local preset
extends: typed_local.yaml
model:
  encoder_kind: structured_v2
  cuda_learner_candidate_scoring_chunk_size: 1048576
training:
  algorithm: impala_vtrace_structured_v1
  profile_timers: false
  structured_metrics:
    mode: sampled
  teacher_aux:
    mode: always
  fixed_opponent_backend: python_scalar
  structured_aux:
    enabled: true
    teacher_family_coef: 0.20
    teacher_slot_coef: 0.10
    teacher_attack_type_coef: 0.05
    teacher_action_coef: 0.10
  structured_warmstart:
    enabled: true
    updates: 32
    teacher_family_coef: 0.75
    teacher_slot_coef: 0.35
    teacher_attack_type_coef: 0.20
    teacher_action_coef: 0.50
```

------------------------------
4. Lower-level local defaults
File:
`configs/presets/typed_local.yaml`
Purpose:
- inherited optimizer, exploration, reward shaping, truncation, checkpoint guard, league, and periodic dev eval defaults

```yaml
schema_version: 2
extends: typed_thesis_locked.yaml
description: "Typed GRU local default with stronger exploration and conservative anti-stall defaults"
training:
  rollout:
    batch_unrolls_per_update: 64
  exploration:
    entropy_coef: 0.03
    entropy_anneal_to: 0.01
    entropy_anneal_steps_updates: 300000
  checkpointing:
    checkpoint_interval_updates: 20
    snapshot_interval_updates: 20
    actor_reload_interval_updates: 50
rewards:
  objective: terminal_pm1
  discount:
    gamma: 0.99
  shaping:
    enable_damage_shaping: true
    damage_reward: 0.05
    level_reward: 0.0
    board_reward: 0.0
    no_progress_penalty: 0.0
  truncation:
    reward: -0.1
    bootstrap_value: false
curriculum:
  stall_monitor:
    enabled: true
    truncation_rate_threshold: 0.25
    consecutive_evals: 2
  checkpoint_guard:
    enabled: true
    rollback_score_margin: 0.15
    rollback_truncation_rate_threshold: 0.25
    rollback_max_prob_lt_half: 0.7
    min_best_score: 0.55
    promote_min_prob_gt_half: 0.6
    promote_max_ci_half_width: 0.24
    cooldown_updates: 20
league:
  pool:
    champion_max_age_updates: 120
  sampling:
    pfsp_power: 1.5
    pfsp_epsilon_uniform: 0.3
    pfsp_window_episodes: 25000
    heuristic_public_start_updates: 100
    heuristic_public_mix_fraction: 0.1
    champion_mix_fraction: 0.35
    hard_negative_mix_fraction: 0.25
    hard_negative_min_samples: 16
    hard_negative_max_win_rate: 0.45
  warmup:
    first_updates: 200
    initial_window_episodes: 5000
    ramp_target_updates: 600
    ramp_target_window_episodes: 15000
  promotion:
    paired_seeds: 8
    seed_file: configs/seeds/local_promotion_eval_seeds.txt
evaluation:
  seed_files:
    dev_eval: configs/seeds/local_dev_eval_seeds.txt
    promotion_gate: configs/seeds/local_promotion_eval_seeds.txt
  periodic_dev_eval_interval_updates: 20
  periodic_dev_eval_paired_seeds: 8
reproducibility:
  seed_files:
    dev_eval: configs/seeds/local_dev_eval_seeds.txt
    promotion_gate: configs/seeds/local_promotion_eval_seeds.txt
```

------------------------------
5. Locked thesis-safe base preset
File:
`configs/presets/typed_thesis_locked.yaml`
Purpose:
- deepest inherited base defining optimizer defaults, exploration schedule, environment caps, reward objective, league defaults, and evaluation defaults

```yaml
schema_version: 2
description: "Typed GRU thesis-safe locked preset"
experiment:
  role: main
system:
  profile:
    training: fast
    local_iteration: fast
    ci_invariant_testing: debug
  mp_start_method: spawn
  learner_device: cuda
  actor_device: cpu
  actor_process_count: 12
  envs_per_actor: 8
  total_envs: 96
  actor_torch_threads: 1
  learner_torch_threads: 4
  actor_queue_capacity_unrolls: 256
  learner_prefetch_batches: 4
model:
  gru_hidden_size: 256
  encoder_mlp_width: 256
  encoder_mlp_layers: 2
  encoder_kind: typed_v1
  typed_feature_width: 64
  recurrent_core: gru
  layer_norm: true
  dropout:
    family_a: 0.0
    ablation: 0.1
training:
  algorithm: impala_vtrace_gru
  rollout:
    unroll_length: 64
    batch_unrolls_per_update: 128
  optimizer:
    name: Adam
    learning_rate: 0.0002
    grad_norm_clip: 40.0
    value_loss_coef: 0.5
  exploration:
    entropy_coef: 0.01
    entropy_anneal_to: 0.001
    entropy_anneal_steps_updates: 2000000
  precision:
    mixed_precision: true
    compile_learner: false
    compile_actor_inference: false
    masking_math_float32: true
  checkpointing:
    checkpoint_interval_updates: 50000
    snapshot_interval_updates: 100000
    actor_reload_interval_updates: 1000
  vtrace:
    rho_bar: 1.0
    c_bar: 1.0
  ppo:
    clip_epsilon: 0.2
    value_clip_epsilon: 0.2
    gae_lambda: 0.95
    epochs: 4
    target_kl: 0.03
    normalize_advantages: true
environment:
  observation_visibility: public
  visibility: public
  truncate_on_max_steps: true
  max_raw_decisions_per_episode: 4000
  max_decisions: 2000
  max_decisions_per_episode: 2000
  max_learner_steps_per_episode: 2000
  max_ticks: 100000
  deck_set_size:
    bring_up: 2
    paper: 8
rewards:
  objective: terminal_only_pm1
  discount:
    gamma: 1.0
  shaping:
    enable_damage_shaping: false
    damage_reward: 0.0
    level_reward: 0.0
    board_reward: 0.0
    no_progress_penalty: 0.0
  truncation:
    reward: 0.0
    bootstrap_value: true
    bootstrap_rule: "if truncated and not terminated, bootstrap from V(next_state); if terminated, no bootstrap"
curriculum:
  simulator: {}
  stall_monitor:
    enabled: false
    truncation_rate_threshold: 0.25
    consecutive_evals: 2
league:
  enabled: true
  pool:
    recent_size: 24
    champion_size: 4
    champion_max_age_updates: 0
  sampling:
    opponent_sampling: PFSP
    pfsp_power: 2.0
    pfsp_epsilon_uniform: 0.2
    pfsp_stats_source: online_outcomes
    pfsp_window_episodes: 50000
    heuristic_public_start_updates: 0
    heuristic_public_mix_fraction: 0.0
    champion_mix_fraction: 0.35
    hard_negative_mix_fraction: 0.15
    hard_negative_min_samples: 32
    hard_negative_max_win_rate: 0.45
  warmup:
    first_updates: 200000
    initial_window_episodes: 10000
    ramp_target_updates: 1000000
    ramp_target_window_episodes: 50000
  promotion:
    enabled: true
    paired_seeds: 64
    threshold: "P(p_anchor > 0.55) > 0.95 using AnchorSet_v1"
    anchor_set_v1:
      required:
        - B0 RandomLegal
        - B1 NoLeague baseline
      optional_if_available:
        - B2 HeuristicPublic
    seed_file: configs/seeds/promotion_eval_seeds.txt
    gate:
      uncertainty_method: bayesian_bootstrap_seedlevel_v1
      weighting: uniform_across_anchors
      seat_swap: true
      folding: S0
      guardrails:
        max_prob_anchor_loss_below_0_45: 0.05
        max_truncation_rate: 0.05
      record_file: promotion_gate.json
evaluation:
  seat_swap: true
  eval_device: cpu
  eval_inference_mode: true
  eval_sampling_algorithm: pinned_cdf_pcg_v1
  eval_assert_sorted_legal_ids: true
  seed_files:
    dev_eval: configs/seeds/dev_eval_seeds.txt
    report_eval: configs/seeds/report_eval_seeds.txt
    promotion_gate: configs/seeds/promotion_eval_seeds.txt
  periodic_dev_eval_interval_updates: 50000
  periodic_dev_eval_paired_seeds: 64
  final_policy_set_size: 10
  final_matrix_stage1_paired_seeds: 64
  final_matrix_stage2_adaptive_max_paired_seeds: 256
  stop_rules:
    stop_delta_ci_half_width: 0.03
    stop_confidence: 0.95
  replay_capture_rate_eval: 0.001
  regression_capture_count: 50
  legal_fingerprint_checks:
    enabled: true
    version: legal_fingerprint_v1
    require_strictly_increasing_legal_ids: true
    mismatch_policy: hard_fail
  decision_kind_tagging:
    required_for_training: false
    enable_python_derived_debug_tag: false
  final_policy_set_selection:
    version: deterministic_v1
    include_random_legal_baseline_b0: true
    include_no_league_baseline_b1: true
    include_heuristic_public_b2_if_exists: true
    include_final_champion_snapshot: true
    include_spaced_snapshots_near_percent_updates: [25, 50, 75]
    remaining_slots_strategy: top_dev_performers_vs_anchor_set_v1
    fixed_anchor_set_v1:
      required:
        - B0 RandomLegal
        - B1 NoLeague baseline
      optional_if_available:
        - B2 HeuristicPublic
    seed_file: configs/seeds/dev_eval_seeds.txt
    folding: S0
    seat_swap: true
    tie_break: lowest_policy_id
reproducibility:
  spec_bundle:
    require_export_spec_bundle: true
    persist_in_manifest: true
    fail_on_spec_mismatch: true
  ids:
    run_id_hash: sha256
    config_hash: sha256
    spec_hash: sha256
    store_full_256_bit_ids: true
    store_short_64_bit_ids_for_filenames: true
  seed_derivation:
    base_seed64: 20260212
    actor_seed_formula: "hash64(base_seed64, actor_id)"
    episode_seed_formula: "hash64(actor_seed64, env_id, episode_index)"
  seed_files:
    dev_eval: configs/seeds/dev_eval_seeds.txt
    report_eval: configs/seeds/report_eval_seeds.txt
    promotion_gate: configs/seeds/promotion_eval_seeds.txt
  determinism_requirements:
    - simulator_build_and_spec_bundle_match
    - environment_config_match
    - episode_seeds_match
    - action_sequence_match
    - evaluation_sampler_rng_and_algorithm_match
  legal_fingerprint:
    version: legal_fingerprint_v1
    compute_in_rl_layer: true
    canonical_bytes:
      - "b'legal_fp_v1'"
      - spec_hash256
      - u32_le_decision_id
      - u32_le_legal_ids_len
      - u32_le_each_legal_id
    replay_eval_mismatch_policy: hard_fail
```

------------------------------
6. Reference ablation showing intended teacher fade idea
File:
`configs/presets/ablations/structured_acceptance_thesis_model_teacher_fade_auto_gpu.yaml`
Purpose:
- useful reference for a structural change that fades heuristic guidance later in training

```yaml
schema_version: 2
description: thesis-model recipe with late anneal of heuristic guidance and tactical learner-side public bias
extends: ../structured_acceptance_thesis_model_auto_gpu.yaml
experiment:
  role: ablation_teacher_fade
curriculum:
  early_cutoff:
    enabled: true
    warmup_updates: 120
    patience_updates: 120
    min_improvement: 0.01
    stall_patience_evals: 4
    stall_rate_threshold: 0.25
model:
  public_heuristic_logit_bias_start_updates: 40
  public_heuristic_logit_bias_end_updates: 140
  public_heuristic_logit_bias_final_scale: 0.5
training:
  actor_heuristic_start_updates: 40
  actor_heuristic_end_updates: 140
  actor_heuristic_final_fraction: 0.25
  structured_aux:
    teacher_public_heuristic_start_updates: 40
    teacher_public_heuristic_end_updates: 140
    teacher_public_heuristic_final_coef: 0.0
```

------------------------------
7. train.py CLI and override handling
File:
`python/scripts/train.py`
Purpose:
- how `--config-override` is accepted
- how B1 baseline imports are passed in
- how run metadata tracks the training controls

```python
    parser.add_argument("--run-label", type=str, default="", help="Optional run directory label override")
    parser.add_argument("--run-id", dest="run_id_alias", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument(
        "--override",
        "--config-override",
        dest="config_override",
        action="append",
        default=None,
        help="Deterministic config override in KEY=JSON_VALUE form, e.g. training.optimizer.learning_rate=0.0001",
    )
    parser.add_argument("--num-envs", type=int, default=2, help="Env count for the single-node training run")
    parser.add_argument("--unroll-length", type=int, default=4, help="Tiny rollout length for the smoke run")
    parser.add_argument("--max-updates", type=int, default=1, help="Number of learner updates to run")
    parser.add_argument(
        "--runtime-mode",
        type=str,
        default="train_ordered",
        choices=("train_ordered", "train_async_fast"),
        help="Queue runtime mode: deterministic ordered collection or throughput-oriented async-fast collection",
    )
    parser.add_argument(
        "--profile-timers",
        action="store_true",
        help="Enable cheap runtime/learner timers and record_function ranges without emitting a torch profiler trace",
    )
    parser.add_argument(
        "--torch-profiler",
        action="store_true",
        help="Emit a torch profiler trace under profiling/torch_profiler/trace.json",
    )
    parser.add_argument("--profile", type=str, default="", help="Optional simulator profile override")
    parser.add_argument("--device", type=str, default="", help="Optional learner device override")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override")
    parser.add_argument(
        "--checkpoint-interval-updates",
        type=int,
        default=None,
        help="Optional checkpoint cadence override for the single-node training run",
    )
    parser.add_argument(
        "--snapshot-registry-json",
        type=Path,
        default=None,
        help="Optional snapshot registry JSON used to resolve the deterministic final policy set in the manifest",
    )
    parser.add_argument(
        "--dev-eval-summaries-json",
        type=Path,
        default=None,
        help="Optional dev-eval summaries JSON used to resolve the deterministic final policy set in the manifest",
    )
    parser.add_argument(
        "--b1-baseline-run-dir",
        type=Path,
        default=None,
        help="Completed baseline_noleague run directory used to import the canonical B1 baseline anchor",
    )
    parser.add_argument(
        "--seed-snapshot-run-dir",
        type=Path,
        default=None,
        help="Optional completed run directory whose snapshot registry should be imported into the current training league before update 1",
    )
    parser.add_argument(
        "--resume-run-dir",
        type=Path,
        default=None,
        help="Resume training in-place inside an existing run directory",
    )
```

```python
    run_summary_payload = _load_json_object(artifacts.run_summary_path, label="run summary")
    run_summary_payload["runtime_mode"] = "public_demo" if public_demo_enabled else str(args.runtime_mode)
    run_summary_payload["policy_set_selection_mode"] = policy_set_selection_details.get("mode", "unresolved")
    if training_config is not None:
        run_summary_payload["training_controls"] = {
            "profile_timers": bool(training_config.profile_timers),
            "torch_profiler": bool(training_config.torch_profiler),
            "structured_metrics_mode": str(training_config.structured_metrics_mode),
            "teacher_aux_mode": str(training_config.teacher_aux_mode),
            "fixed_opponent_backend": str(training_config.fixed_opponent_backend),
        }
    if args.b1_baseline_run_dir is not None:
        run_summary_payload["b1_baseline_run_dir"] = args.b1_baseline_run_dir.resolve().as_posix()
    if args.seed_snapshot_run_dir is not None:
        run_summary_payload["seed_snapshot_run_dir"] = args.seed_snapshot_run_dir.resolve().as_posix()
    if resume_checkpoint_path is not None:
        run_summary_payload["resume"] = {
            "enabled": True,
            "resume_run_dir": None if resume_run_dir is None else resume_run_dir.as_posix(),
            "resume_checkpoint_path": resume_checkpoint_path.as_posix(),
        }
    _write_json(artifacts.run_summary_path, run_summary_payload)

    determinism_payload = _load_json_object(artifacts.determinism_report_path, label="determinism report")
    determinism_payload["runtime_mode"] = "public_demo" if public_demo_enabled else str(args.runtime_mode)
    determinism_payload["policy_selection_mode"] = policy_set_selection_details.get("mode", "unresolved")
    if training_config is not None:
        determinism_payload["training_controls"] = {
            "profile_timers": bool(training_config.profile_timers),
            "torch_profiler": bool(training_config.torch_profiler),
            "structured_metrics_mode": str(training_config.structured_metrics_mode),
            "teacher_aux_mode": str(training_config.teacher_aux_mode),
            "fixed_opponent_backend": str(training_config.fixed_opponent_backend),
        }
    if args.b1_baseline_run_dir is not None:
        determinism_payload["b1_baseline_run_dir"] = args.b1_baseline_run_dir.resolve().as_posix()
    if args.seed_snapshot_run_dir is not None:
        determinism_payload["seed_snapshot_run_dir"] = args.seed_snapshot_run_dir.resolve().as_posix()
    if resume_checkpoint_path is not None:
        determinism_payload["resume_checkpoint_path"] = resume_checkpoint_path.as_posix()
```

------------------------------
8. Learner config plumbing for teacher and optimizer pieces
File:
`python/scripts/train.py`
Purpose:
- shows what learner knobs are actually passed through and scheduled

```python
        "grad_norm_clip": training_config.grad_norm_clip,
        "mixed_precision": bool(training_config.mixed_precision),
        "checkpoint_dir": training_paths.checkpoints_dir,
        "checkpoint_interval_updates": int(checkpoint_interval_updates),
        "logs_dir": training_paths.logs_dir,
        "logging_interval_updates": 1,
        "pass_action_id": pass_action_id,
        "teacher_family_coef": training_config.teacher_family_coef,
        "teacher_slot_coef": training_config.teacher_slot_coef,
        "teacher_move_source_coef": training_config.teacher_move_source_coef,
        "teacher_attack_type_coef": training_config.teacher_attack_type_coef,
        "teacher_action_coef": training_config.teacher_action_coef,
        "teacher_same_family_action_coef": training_config.teacher_same_family_action_coef,
        "teacher_public_heuristic_coef": training_config.teacher_public_heuristic_coef,
        "teacher_public_heuristic_temperature": training_config.teacher_public_heuristic_temperature,
        "teacher_public_heuristic_families": training_config.teacher_public_heuristic_families,
        "teacher_public_heuristic_profiles": training_config.teacher_public_heuristic_profiles,
        "teacher_public_heuristic_profile_mode": training_config.teacher_public_heuristic_profile_mode,
        "teacher_public_heuristic_profiles_end_updates": training_config.teacher_public_heuristic_profiles_end_updates,
        "profile_timers": bool(getattr(training_config, "profile_timers", False)),
        "structured_metrics_mode": str(getattr(training_config, "structured_metrics_mode", "full")),
        "teacher_aux_mode": str(getattr(training_config, "teacher_aux_mode", "always")),
    }
    if algorithm in _IMPALA_ALGORITHMS:
        return ImpalaLearner(
            **common_kwargs,
            vtrace_rho_bar=training_config.vtrace_rho_bar,
            vtrace_c_bar=training_config.vtrace_c_bar,
        )
```

```python
def _entropy_coef_for_next_update(training_config: Any, *, update_count: int) -> float:
    start = float(training_config.entropy_coef)
    target = float(training_config.entropy_anneal_to)
    steps = max(1, int(training_config.entropy_anneal_steps_updates))
    progress = min(max(int(update_count), 0), steps) / float(steps)
    return float(start + (target - start) * progress)


def _teacher_public_heuristic_coef_for_next_update(training_config: Any, *, update_count: int) -> float:
    return float(
        linear_anneal_value(
            initial_value=float(training_config.teacher_public_heuristic_coef),
            final_value=float(getattr(training_config, "teacher_public_heuristic_final_coef", 0.0)),
            start_update=int(getattr(training_config, "teacher_public_heuristic_start_updates", 0)),
            end_update=int(getattr(training_config, "teacher_public_heuristic_end_updates", -1)),
            update_count=int(update_count),
        )
    )


def _public_heuristic_logit_bias_scale_for_next_update(model_config: Any, *, update_count: int) -> float:
    return float(
        linear_anneal_value(
            initial_value=float(getattr(model_config, "public_heuristic_logit_bias_scale", 0.0)),
            final_value=float(
                getattr(
                    model_config,
                    "public_heuristic_logit_bias_final_scale",
                    getattr(model_config, "public_heuristic_logit_bias_scale", 0.0),
                )
            ),
            start_update=int(getattr(model_config, "public_heuristic_logit_bias_start_updates", 0)),
            end_update=int(getattr(model_config, "public_heuristic_logit_bias_end_updates", -1)),
            update_count=int(update_count),
        )
    )


def _apply_guidance_schedule_for_next_update(
    *,
    learner: ImpalaLearner,
    model: PolicyValueModel | None,
    stack: StackConfig,
    update_count: int,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    training_config = stack.config.training
    if training_config is not None:
        teacher_coef = _teacher_public_heuristic_coef_for_next_update(training_config, update_count=update_count)
        learner.set_teacher_aux_coefs(public_heuristic=teacher_coef)
        metrics["teacher_public_heuristic_coef_active"] = float(teacher_coef)
    model_config = stack.config.model
    if model is not None and model_config is not None:
        set_bias_scale = getattr(model, "set_public_heuristic_logit_bias_scale", None)
        get_bias_scale = getattr(model, "get_public_heuristic_logit_bias_scale", None)
        if callable(set_bias_scale):
            actor_bias_scale: float | None = None
            if callable(get_bias_scale):
                actor_bias_scale = float(get_bias_scale(scoring_mode="actor"))
            learner_bias_scale = _public_heuristic_logit_bias_scale_for_next_update(
                model_config,
                update_count=update_count,
            )
            set_bias_scale(learner_bias_scale, actor_value=actor_bias_scale)
            metrics["public_heuristic_logit_bias_scale_active"] = float(learner_bias_scale)
```

------------------------------
9. Structured warmstart implementation
File:
`python/scripts/train.py`
Purpose:
- shows exactly how strong early teacher guidance is applied before ordinary training

```python
def _run_structured_warmstart(
    *,
    learner: ImpalaLearner,
    runtime: QueueRuntime,
    algorithm: str,
    training_config: Any,
    rewards_config: Any,
    training_paths: TrainingPaths,
    tensorboard_logger: TensorBoardLogger | None,
    start_time: float,
    profile_timers: bool = False,
    actor_torch_threads: int | None = None,
    learner_torch_threads: int | None = None,
) -> dict[str, float]:
    if not bool(getattr(training_config, "structured_warmstart_enabled", False)):
        return {}
    if algorithm not in _IMPALA_ALGORITHMS:
        raise RuntimeError("structured warmstart currently supports only IMPALA learners")
    warmstart_cfg = training_config.structured_warmstart
    updates = int(warmstart_cfg.updates)
    if updates <= 0:
        return {}

    previous_family = float(training_config.teacher_family_coef)
    previous_slot = float(training_config.teacher_slot_coef)
    previous_move_source = float(training_config.teacher_move_source_coef)
    previous_attack_type = float(training_config.teacher_attack_type_coef)
    previous_action = float(training_config.teacher_action_coef)
    previous_same_family_action = float(training_config.teacher_same_family_action_coef)
    previous_public_heuristic = float(training_config.teacher_public_heuristic_coef)
    previous_public_heuristic_temperature = float(training_config.teacher_public_heuristic_temperature)
    previous_public_heuristic_families = tuple(training_config.teacher_public_heuristic_families)
    previous_public_heuristic_profiles = tuple(training_config.teacher_public_heuristic_profiles)
    previous_public_heuristic_profile_mode = str(training_config.teacher_public_heuristic_profile_mode)
    previous_public_heuristic_profiles_end_updates = int(training_config.teacher_public_heuristic_profiles_end_updates)
    learner.set_teacher_aux_coefs(
        family=float(warmstart_cfg.teacher_family_coef),
        slot=float(warmstart_cfg.teacher_slot_coef),
        move_source=float(warmstart_cfg.teacher_move_source_coef),
        attack_type=float(warmstart_cfg.teacher_attack_type_coef),
        action=float(warmstart_cfg.teacher_action_coef),
        same_family_action=float(warmstart_cfg.teacher_same_family_action_coef),
        public_heuristic=float(warmstart_cfg.teacher_public_heuristic_coef),
        public_heuristic_temperature=float(warmstart_cfg.teacher_public_heuristic_temperature),
        public_heuristic_families=tuple(warmstart_cfg.teacher_public_heuristic_families),
        public_heuristic_profiles=tuple(warmstart_cfg.teacher_public_heuristic_profiles),
        public_heuristic_profile_mode=str(warmstart_cfg.teacher_public_heuristic_profile_mode),
        public_heuristic_profiles_end_updates=int(warmstart_cfg.teacher_public_heuristic_profiles_end_updates),
    )
    latest_metrics: dict[str, float] = {}
    try:
        with (
            runtime.structured_warmstart_source_mix() as warmstart_source_metrics,
            runtime.disable_mirror_policy_fusion(),
        ):
            for warmstart_step in range(updates):
                with (
                    _profile_block(profile_timers, "collect_training_batch"),
                    _torch_num_threads_scope(actor_torch_threads),
                ):
                    runtime_batch = _collect_training_batch(
                        runtime=runtime,
                        algorithm=algorithm,
                        training_config=training_config,
                        rewards_config=rewards_config,
                    )
                with (
                    _profile_block(profile_timers, "learner_auxiliary_update"),
                    _torch_num_threads_scope(learner_torch_threads),
                ):
                    latest_metrics = learner.auxiliary_update(runtime_batch.learner_batch)
                latest_metrics.update(runtime_batch.runtime_metrics)
                latest_metrics.update(warmstart_source_metrics)
                latest_metrics["warmstart_phase"] = 1.0
                latest_metrics["warmstart_step"] = float(warmstart_step + 1)
                _write_scalars_record(
                    scalars_path=training_paths.scalars_path,
                    learner=learner,
                    metrics=latest_metrics,
                    start_time=start_time,
                )
                if tensorboard_logger is not None:
                    tensorboard_logger.log_training_step(
                        update_count=int(learner.update_count),
                        policy_version=int(learner.get_policy_version()),
                        wall_clock_seconds=time.time() - start_time,
                        metrics=latest_metrics,
                    )
    finally:
        learner.set_teacher_aux_coefs(
            family=previous_family,
            slot=previous_slot,
            move_source=previous_move_source,
            attack_type=previous_attack_type,
            action=previous_action,
            same_family_action=previous_same_family_action,
```

------------------------------
10. Early cutoff logic
File:
`python/scripts/train.py`
Purpose:
- shows why no-improvement cutoff cannot trigger early under the current preset

```python
def _update_early_cutoff(
    *,
    stack: StackConfig,
    training_paths: TrainingPaths,
    update_count: int,
    summary_payload: Mapping[str, Any],
) -> dict[str, Any] | None:
    curriculum = stack.config.curriculum
    if curriculum is None or not curriculum.early_cutoff.enabled:
        return None
    current_score = _dev_eval_aggregate_score(summary_payload)
    if current_score is None:
        return None

    early_cutoff = curriculum.early_cutoff
    state_path = _early_cutoff_state_path(training_paths)
    state = _load_json_object(state_path, label="early cutoff state") if state_path.is_file() else {}
    previous_best_score = state.get("best_score")
    previous_best_update = state.get("best_update_count")
    previous_consecutive_stall = int(state.get("consecutive_stall_evals", 0))

    improved = False
    if isinstance(previous_best_score, (int, float)) and np.isfinite(float(previous_best_score)):
        best_score = float(previous_best_score)
        best_update_count = (
            int(previous_best_update) if isinstance(previous_best_update, int) else int(update_count)
        )
        if float(current_score) > best_score + float(early_cutoff.min_improvement):
            best_score = float(current_score)
            best_update_count = int(update_count)
            improved = True
    else:
        best_score = float(current_score)
        best_update_count = int(update_count)
        improved = True

    patience_reference_update = max(int(best_update_count), int(early_cutoff.warmup_updates))
    no_improvement_updates = max(0, int(update_count) - patience_reference_update)
    worst_stall_rate = _dev_eval_worst_stall_rate(summary_payload)
    if (
        worst_stall_rate is not None
        and worst_stall_rate >= float(early_cutoff.stall_rate_threshold)
    ):
        consecutive_stall_evals = previous_consecutive_stall + 1
    else:
        consecutive_stall_evals = 0

    reasons: list[str] = []
    if (
        int(early_cutoff.patience_updates) > 0
        and int(update_count) >= int(early_cutoff.warmup_updates)
        and no_improvement_updates >= int(early_cutoff.patience_updates)
    ):
        reasons.append("no_improvement")
    if (
        int(early_cutoff.stall_patience_evals) > 0
        and consecutive_stall_evals >= int(early_cutoff.stall_patience_evals)
    ):
        reasons.append("stall")

    payload = {
        "enabled": True,
        "update_count": int(update_count),
        "current_score": float(current_score),
        "best_score": float(best_score),
        "best_update_count": int(best_update_count),
        "improved": bool(improved),
        "min_improvement": float(early_cutoff.min_improvement),
        "warmup_updates": int(early_cutoff.warmup_updates),
        "patience_updates": int(early_cutoff.patience_updates),
        "no_improvement_updates": int(no_improvement_updates),
        "stall_patience_evals": int(early_cutoff.stall_patience_evals),
        "stall_rate_threshold": float(early_cutoff.stall_rate_threshold),
        "worst_stall_rate": None if worst_stall_rate is None else float(worst_stall_rate),
        "consecutive_stall_evals": int(consecutive_stall_evals),
        "should_stop": bool(reasons),
        "reasons": reasons,
    }
```

------------------------------
11. Checkpoint guard rollback logic
File:
`python/scripts/train.py`
Purpose:
- shows how later regressions can be rolled back to best checkpoint without changing the underlying training recipe

```python
def _maybe_rollback_to_best_checkpoint(
    *,
    stack: StackConfig,
    dev_eval_summary: Mapping[str, Any] | None,
    last_rollback_update: int | None,
) -> dict[str, Any] | None:
    curriculum = stack.config.curriculum
    if curriculum is None:
        return None
    checkpoint_guard = curriculum.checkpoint_guard
    if not checkpoint_guard.enabled or dev_eval_summary is None:
        return None
    if last_rollback_update is not None and (int(learner.update_count) - int(last_rollback_update)) < int(
        checkpoint_guard.cooldown_updates
    ):
        return None

    current_score = _dev_eval_aggregate_score(dev_eval_summary)
    if current_score is None:
        return None
    worst_truncation_rate = _dev_eval_worst_truncation_rate(dev_eval_summary)
    worst_stall_rate = _dev_eval_worst_stall_rate(dev_eval_summary)
    worst_no_progress_timeout_rate = _dev_eval_worst_no_progress_timeout_rate(dev_eval_summary)
    worst_natural_timeout_rate = _dev_eval_worst_natural_timeout_rate(dev_eval_summary)
    tracker = _load_checkpoint_tracker(training_paths)
    best_record = tracker.get("best")
    if not isinstance(best_record, Mapping):
        return None
    best_metric_kind = str(best_record.get("metric_kind", "")).strip()
    best_metric_value = best_record.get("metric_value")
    best_update_count = best_record.get("update_count")
    if best_metric_kind != "dev_eval_mean":
        return None
    if not isinstance(best_metric_value, (int, float)) or not np.isfinite(float(best_metric_value)):
        return None
    if not isinstance(best_update_count, int) or int(best_update_count) >= int(learner.update_count):
        return None
    best_score = float(best_metric_value)
    if best_score < float(checkpoint_guard.min_best_score):
        return None

    confidence = _dev_eval_confidence_stats(dev_eval_summary)
    rollback_reasons: list[str] = []
    if current_score <= best_score - float(checkpoint_guard.rollback_score_margin):
        rollback_reasons.append("score_drop")
    if worst_stall_rate is not None and (
        worst_stall_rate >= float(checkpoint_guard.rollback_truncation_rate_threshold)
    ):
        rollback_reasons.append("truncation")
    max_prob_lt_half = confidence["max_prob_lt_half"]
    if max_prob_lt_half is not None and (float(max_prob_lt_half) >= float(checkpoint_guard.rollback_max_prob_lt_half)):
        rollback_reasons.append("confidence")
    if not rollback_reasons:
        return None

    best_checkpoint_path = training_paths.best_checkpoint_path
    _restore_checkpoint_to_latest_alias(
        checkpoint_path=best_checkpoint_path,
        training_paths=training_paths,
        learner=learner,
        stack=stack,
        device=device,
        expected_spec_hash256=spec_hash256,
        algorithm=algorithm,
    )
    demoted_champions = _demote_registry_champions_newer_than(
        training_paths,
        update_count=int(best_update_count),
    )
    publish_metrics = runtime.maybe_publish_snapshot(
        learner_model=model,
        learner_update_count=int(learner.update_count),
        force=True,
    )
    runtime.reset_outcome_tracker()
    runtime.refresh_opponent_pool()
    tracker["latest"] = _build_checkpoint_record(
        alias_name="latest",
        alias_path=training_paths.latest_checkpoint_path,
        source_checkpoint_path=best_checkpoint_path,
        artifacts=artifacts,
        learner=learner,
        metric_kind="dev_eval_mean",
        metric_value=best_score,
    )
    _write_checkpoint_tracker(training_paths, tracker)

    payload = {
        "format": "checkpoint_guard_event_v1",
        "action": "rollback_to_best",
        "update_count": int(learner.update_count),
        "policy_version": int(learner.get_policy_version()),
        "current_score": current_score,
        "best_score": best_score,
        "best_update_count": int(best_update_count),
        "worst_stall_rate": worst_stall_rate,
        "worst_truncation_rate": worst_truncation_rate,
        "worst_no_progress_timeout_rate": worst_no_progress_timeout_rate,
        "worst_natural_timeout_rate": worst_natural_timeout_rate,
        "min_prob_gt_half": confidence["min_prob_gt_half"],
        "max_prob_lt_half": confidence["max_prob_lt_half"],
        "max_ci_half_width": confidence["max_ci_half_width"],
        "reasons": rollback_reasons,
        "best_checkpoint_path": _relative_path_text(best_checkpoint_path, root=artifacts.run_dir),
        "latest_checkpoint_path": _relative_path_text(training_paths.latest_checkpoint_path, root=artifacts.run_dir),
        "snapshot_publish_latency_ms": publish_metrics.get("snapshot_publish_latency_ms", 0.0),
        "snapshot_apply_latency_ms": publish_metrics.get("snapshot_apply_latency_ms", 0.0),
        "latest_loss": None if latest_metrics is None else latest_metrics.get("loss"),
        "demoted_champions": demoted_champions,
    }
```

------------------------------
12. Periodic dev eval scheduling in the main training loop
File:
`python/scripts/train.py`
Purpose:
- shows how async periodic dev eval is scheduled and processed

```python
                if pending_periodic_dev_eval is not None and pending_periodic_dev_eval.future.done():
                    completed_summary, guard_event = _process_completed_periodic_dev_eval(
                        pending_eval=pending_periodic_dev_eval,
                        stack=stack,
                        contract=contract,
                        artifacts=artifacts,
                        training_paths=training_paths,
                        runtime=runtime,
                        learner=learner,
                        device=device,
                        run_id256=run_id256,
                        config_hash256=config_hash256,
                        spec_hash256=spec_hash256,
                        last_rollback_update=last_checkpoint_guard_rollback_update,
                        tensorboard_logger=tensorboard_logger,
                    )
                    pending_periodic_dev_eval = None
                    last_dev_eval_summary = completed_summary
                    last_dev_eval_update_count = int(completed_summary["update_count"])
                    anchor_keys = sorted(cast(dict[str, Any], completed_summary["anchor_scores"]).keys())
                    opponent_fragment = f" opponent={_slug_policy_id(anchor_keys[0])}" if anchor_keys else ""
                    print(
                        "Periodic dev eval complete: "
                        f"update={int(completed_summary['update_count'])}{opponent_fragment} "
                        f"aggregate={completed_summary['aggregate_score']:.4f} "
                        f"anchors={','.join(anchor_keys)}"
                    )
                    if guard_event is not None:
                        last_checkpoint_guard_rollback_update = int(learner.update_count)
                        pending_promotion_gate = _drop_stale_pending_promotion_gate(
                            stack=stack,
                            training_paths=training_paths,
                            run_dir=artifacts.run_dir,
                            pending_gate=pending_promotion_gate,
                            rollback_best_update_count=int(guard_event["best_update_count"]),
                        )
                        prefetched_runtime_batch = None
                        print(
                            "Checkpoint guard rollback: "
                            f"update={guard_event['update_count']} "
                            f"best_update={guard_event['best_update_count']} "
                            f"current_score={float(guard_event['current_score']):.4f} "
                            f"best_score={float(guard_event['best_score']):.4f} "
                            f"reasons={','.join(cast(list[str], guard_event['reasons']))}"
                        )
                    early_cutoff_payload = _update_early_cutoff(
                        stack=stack,
                        training_paths=training_paths,
                        update_count=int(completed_summary["update_count"]),
                        summary_payload=completed_summary,
                    )
                    if early_cutoff_payload is not None and bool(early_cutoff_payload.get("should_stop", False)):
                        latest_metrics.update(
                            {
                                "early_cutoff_triggered": 1.0,
                                "early_cutoff_best_score": float(early_cutoff_payload["best_score"]),
                                "early_cutoff_current_score": float(early_cutoff_payload["current_score"]),
                                "early_cutoff_no_improvement_updates": float(
                                    early_cutoff_payload["no_improvement_updates"]
                                ),
                                "early_cutoff_consecutive_stall_evals": float(
                                    early_cutoff_payload["consecutive_stall_evals"]
                                ),
                            }
                        )
                        print(
                            "Early cutoff triggered: "
                            f"update={int(completed_summary['update_count'])} "
                            f"best_update={int(early_cutoff_payload['best_update_count'])} "
                            f"best_score={float(early_cutoff_payload['best_score']):.4f} "
                            f"current_score={float(early_cutoff_payload['current_score']):.4f} "
                            f"reasons={','.join(cast(list[str], early_cutoff_payload['reasons']))}"
                        )
                        stop_requested = True
```

```python
                        opponent_specs, pinned_snapshot_ids = _resolve_periodic_dev_eval_opponent_specs(
                            stack=stack,
                            run_dir=artifacts.run_dir,
                        )
                        newly_pinned_snapshot_ids = _pin_snapshot_ids(
                            stack=stack,
                            training_paths=training_paths,
                            run_dir=artifacts.run_dir,
                            snapshot_ids=pinned_snapshot_ids,
                        )
                        request = AsyncPeriodicDevEvalRequest(
                            stack=stack,
                            checkpoint_path=checkpoint_path,
                            focal_policy_id=_current_focal_policy_id(learner=learner),
                            update_count=int(learner.update_count),
                            policy_version=int(learner.get_policy_version()),
                            run_dir=artifacts.run_dir,
                            run_id256=run_id256,
                            config_hash256=config_hash256,
                            spec_hash256=spec_hash256,
                            artifact_dir_name="dev_eval",
                            artifact_scope="periodic_dev_eval",
                            paired_seeds=tuple(_periodic_dev_eval_schedule(stack)[2]),
                            opponents=tuple(opponent_specs),
                            eval_device_override=None,
                            parallel_workers=max(
                                1,
                                int(
                                    getattr(
                                        stack.config.evaluation,
                                        "periodic_dev_eval_parallel_workers",
                                        1,
                                    )
                                ),
                            ),
                            parallel_worker_devices=_resolved_periodic_dev_eval_worker_devices(
                                stack=stack,
                                parallel_workers=max(
                                    1,
                                    int(
                                        getattr(
                                            stack.config.evaluation,
                                            "periodic_dev_eval_parallel_workers",
                                            1,
                                        )
                                    ),
                                ),
                                explicit_worker_devices=tuple(
                                    getattr(
                                        stack.config.evaluation,
                                        "periodic_dev_eval_parallel_worker_devices",
                                        (),
                                    )
                                ),
                                eval_device=str(
                                    getattr(
                                        stack.config.evaluation,
                                        "eval_device",
                                        "cpu",
                                    )
                                ),
                                learner_device=device,
                            ),
                        )
                        pending_periodic_dev_eval = PendingPeriodicDevEval(
                            future=async_periodic_dev_eval_executor.submit(
                                _run_async_periodic_dev_eval_worker,
                                request,
                            ),
                            request=request,
                            pinned_snapshot_ids=tuple(newly_pinned_snapshot_ids),
                            latest_metrics=dict(latest_metrics),
                        )
                        print(
                            "Periodic dev eval scheduled: "
                            f"update={int(learner.update_count)} "
                            f"devices={','.join(request.parallel_worker_devices) or str(stack.config.evaluation.eval_device)} "
                            f"anchors={','.join(spec.display_name for spec in request.opponents)}"
                        )
```

==================================================
QUESTIONS I WANT ANSWERED
==================================================

1. Why would anti-stall shaping fix collapse but fail to improve the hardest `B1` matchup?
2. Do these results suggest that warmstart / teacher behavior dominates the first 10 updates?
3. Is paired-seed 16 too noisy/coarse to detect real gains here, or is the plateau too consistent for that to be the main explanation?
4. What structural change is most likely to improve `B1` specifically?
5. Should I prioritize:
   - reward shaping changes
   - termination / no-progress policy changes
   - warmstart / teacher coefficient changes
   - opponent mix / curriculum changes
   - longer training with best current recipe and early stopping
   - a dedicated B1 anchor config
6. If you had to choose only the next 3 experiments, what exactly would you run?
7. Which ideas are most likely to produce only cosmetic aggregate gains while leaving `B1` unchanged?
8. What signals in the code/config would tell you that this anchor objective is misaligned with the `B1` matchup I care about?

==================================================
OUTPUT FORMAT I WANT
==================================================

Please answer with these sections:

1. Diagnosis
- What you think is happening overall

2. Ranked Hypotheses
- ordered strongest to weakest
- each with:
  - why it fits the evidence
  - what evidence would falsify it

3. What To Stop Doing
- which kinds of tweaks are low-value based on the current evidence

4. Recommended Next Experiments
- top 3 to 5 experiments
- include exact settings / code-level levers to change
- explain why each is high value

5. Structural Changes Most Likely To Matter
- what bigger design changes could actually move `B1`

6. Minimal Rescue Plan
- if I only had time for one careful next experiment, what should it be?

Do not give me generic RL advice.
Ground your reasoning in the actual results and code/config I pasted.
