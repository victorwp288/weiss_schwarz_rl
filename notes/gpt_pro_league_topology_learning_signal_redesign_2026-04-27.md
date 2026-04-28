# GPT Pro Prompt: Redesign the League Topology and Learning Signal for Weiss Schwarz RL

You are helping us in the same ongoing Weiss Schwarz RL thesis rescue/debugging session.

Please assume you have the earlier prompt and your earlier outputs, but this prompt is meant to be self-contained enough that you can reason without seeing the codebase directly.

We are no longer asking only:

```text
Why did the latest run fail?
```

We are asking the bigger question:

```text
How should we redesign the league topology and learning signal so this project can actually produce a thesis-worthy agent?
```

We have a decent B1 no-league anchor. We expected that a league/self-play system, with enough compute, should eventually make a model clearly stronger than that anchor. Instead, many variants plateau at 0.50 against B1 and often fail to improve against recent/previous champions. Something feels structurally wrong.

We are very open to changing the league system, objective, promotion mechanics, opponent pools, eval surfaces, counterfactual search, or model training setup if that is what fixes the issue. Please do not treat the current implementation as sacred.

## High-Level Goal

We want a strong, thesis-worthy Weiss Schwarz RL system:

```text
- B1 no-league anchor is useful but should not be the ceiling.
- League/self-play should produce upward trends over longer training.
- The final model should become clearly better than B1 on meaningful surfaces.
- The system should be honest about what comes from learned policy vs hand-coded heuristic prior.
- It should scale to multi-GPU Linux L40 server runs after local diagnostics prove the direction.
```

The local Windows machine is only for correctness, relative comparisons, artifact validation, and short diagnostics. Final serious training target is multi-GPU Linux L40, but we do not want to burn server compute until the learning signal is credible.

## Current Important Runs and Artifacts

Strong B1 no-league anchor:

```text
runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425
```

Important B1 checkpoint:

```text
runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425/training/checkpoints/checkpoint_450.pt
```

Previously best local small-model league checkpoint:

```text
runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/training/checkpoints/checkpoint_480.pt
```

But that checkpoint is no longer trusted as a main parent because lower-bias/no-bias diagnostics showed its learned body is weaker than B1.

Latest B1/S1 restart branch:

```text
runs/b1_s1_retrain_u450_to_u460_fix2_20260427
```

Candidate checkpoint:

```text
runs/b1_s1_retrain_u450_to_u460_fix2_20260427/training/checkpoints/checkpoint_460.pt
```

Verdict:

```text
Mechanically correct run, but discarded as a candidate.
It did not improve against B1 on S1 and did not preserve raw/S0 body.
```

## The Two Big Problems We Think We Have

### Problem 1: League topology/bookkeeping was not semantically honest

Earlier, the league did not really behave like an evolving population.

Examples:

```text
- champion_snapshots was empty in important runs.
- "Previous recent snapshot" could resolve to an imported B1-history seed snapshot.
- PFSP recent pool could contain imported B1 seed history rather than learned local recents.
- Promotion/champion semantics were too binary and too tied to noisy eval gates.
- Seed imports, B1 baseline anchors, local candidates, admitted recents, champions, and rejected policies were not cleanly separated.
```

We have implemented repairs:

```text
- source_kind fields such as local, league_import, seed_import, baseline_anchor.
- seed imports no longer become active champions.
- seed history can be quarantined from true PFSP recent/champion lanes.
- latest local/champion selectors avoid seed imports and rejected snapshots.
- true local/league_import recents can now appear in PFSP after continuation.
```

Post-fix topology evidence:

```text
policy_000011 update 480 source_kind league_import
policy_000012 update 500 source_kind league_import
policy_000013 update 520 source_kind local rejected
policy_000014 update 540 source_kind local champion/provisional artifact

PFSP after handoff:
  pfsp_sampling_ready=1
  pfsp_recent_pool_size=2
  pfsp_champion_pool_size=0
  pfsp_warmup_snapshot_pool_size=7
  pfsp_sampling_weight_recent=0.40
  pfsp_sampling_weight_warmup_snapshot=0.0
```

So the old "recent secretly means B1 seed import" problem is materially improved.

But we still worry the topology is not ideal for a serious league:

```text
- We still infer too much from source_kind + champion_snapshots + rejected_snapshots.
- We do not yet have first-class roles like main, exploiter, hard_negative, admitted_recent, provisional_champion, confirmed_champion.
- A policy can be bad as a main checkpoint but useful as a hard negative, and the current design does not express that cleanly.
- 8-pair promotion gates can create noisy "champions" that should be provisional only.
- Imported B1-history snapshots can still be visible as registry artifacts even if not sampled in active lanes.
```

### Problem 2: The learning signal is not producing B1-best-response behavior

This now seems like the larger issue.

The official eval surface was originally:

```text
S3 official/effective surface:
  scoring_mode = learner
  public heuristic logit bias scale = 3.0
```

Diagnostics showed S3 is saturated:

```text
B1/u480/u540 exact 0.50 vs each other
all pair classes 1-1
both-greedy still exact 0.50
physical seat/seed winner pattern is unchanged
```

Trace probe showed the public heuristic bias often overrides raw network preference:

```text
S3 B1 raw-vs-final top action mismatch:   about 33%
S3 u540 raw-vs-final top action mismatch: about 35%
S3 u540 raw-vs-final top family mismatch: about 27%
```

So S3 is not "network plus a tiny prior." It is often a strong wrapper that dominates action selection.

Bias sweep showed:

```text
S0/no-bias:
  B1 crushes u480/u540.

S0.5:
  B1 still crushes u480/u540.

S1/low-bias:
  differences become visible, but u480/u540 do not clearly beat B1.

S3:
  B1-family policies collapse to exact 0.50.
```

This implies:

```text
The league continuations were not hidden better policies.
They were S3-wrapper-preserved policies with weaker raw/low-bias bodies.
```

## Current Eval Surfaces

We now think every serious eval should report at least:

```text
S3 official_s3:
  scoring_mode = learner
  common public heuristic bias scale = 3.0
  Interpretation: deployment/effective policy with strong heuristic prior.

S1 lowbias_s1:
  scoring_mode = learner
  common public heuristic bias scale = 1.0
  Interpretation: diagnostic/training surface that exposes policy differences while keeping some heuristic structure.

S0 raw_s0:
  scoring_mode = learner
  common public heuristic bias scale = 0.0
  Interpretation: learned neural body without public heuristic wrapper.
```

Important: current actor-mode results are not clean because B1 actor bias and learner bias historically differed.

Example actor/learner mismatch:

```text
B1 actor bias scale:   1.0
B1 learner bias scale: 3.0
u480 actor bias:       3.0
u480 learner bias:     3.0
```

This made actor-mode u480/u540 vs B1 look good, but mostly because wrappers were unequal.

We now want actor/eval target consistency:

```text
training B1 effective surface == eval B1 effective surface
current actor bias == current learner bias for the branch
B1 actor/eval bias == same named B1 anchor surface
```

## Recent B1/S1 Restart Experiment

Your previous recommendation was:

```text
- restart from B1 checkpoint_450, not u480/u540;
- train with common actor/learner S1 bias parity;
- preserve S0/raw B1 behavior;
- track S3 only as deployment sanity.
```

We implemented the simplest version of that.

Config:

```text
configs/presets/pass3_b1_s1_retrain_from_u450_rawprotect.yaml
```

Full config:

```yaml
schema_version: 2
description: Pass 3 B1 restart from u450 with common low-bias S1 actor/learner parity and raw-body preservation by gate
extends: structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_selfplay_localpromo.yaml
model:
  public_heuristic_logit_bias_profile: aggressive
  public_heuristic_logit_bias_scale: 1.0
  public_heuristic_actor_logit_bias_scale: 1.0
  public_heuristic_logit_bias_start_updates: 450
  public_heuristic_logit_bias_end_updates: -1
  public_heuristic_logit_bias_final_scale: 1.0
training:
  behavior_action_bc_coef: 0.0
  reference_policy_id: b1_noleague_baseline
  reference_policy_top_action_bc_coef: 0.0
  reference_policy_top_action_bc_final_coef: 0.0
  reference_policy_top_action_bc_start_updates: 450
  reference_policy_top_action_bc_end_updates: 520
  reference_policy_top_action_family_bc_coef: 0.03
  reference_policy_top_action_family_bc_final_coef: 0.0
  reference_policy_top_action_family_bc_start_updates: 450
  reference_policy_top_action_family_bc_end_updates: 520
  optimizer:
    learning_rate: 0.00001
  exploration:
    entropy_coef: 0.04
    entropy_anneal_to: 0.02
    entropy_anneal_steps_updates: 300000
  structured_aux:
    teacher_family_coef: 0.05
    teacher_slot_coef: 0.025
    teacher_action_coef: 0.025
    teacher_attack_type_coef: 0.01
    teacher_public_heuristic_coef: 0.0
    teacher_public_heuristic_final_coef: 0.0
    teacher_public_main_move_coef: 0.0
league:
  sampling:
    heuristic_public_mix_fraction: 0.10
    heuristic_public_final_mix_fraction: 0.10
    heuristic_public_mix_end_updates: -1
    heuristic_public_variant_mix_fraction: 0.15
    heuristic_public_variant_final_mix_fraction: 0.15
    heuristic_public_variant_mix_end_updates: -1
    noleague_baseline_mix_fraction: 0.50
    noleague_baseline_mix_end_updates: -1
    noleague_baseline_reward_scale: 1.0
    noleague_baseline_force_focal_seat: -1
    warmup_snapshot_mix_fraction: 0.0
    exclude_seed_snapshots_from_pfsp: true
    mirror_mix_fraction: 0.25
    champion_mix_fraction: 0.0
    hard_negative_mix_fraction: 0.0
  promotion:
    enabled: false
evaluation:
  periodic_dev_eval_interval_updates: 0
  periodic_dev_eval_paired_seeds: 8
curriculum:
  checkpoint_guard:
    enabled: false
```

Important caveat:

```text
"rawprotect" was only gate-only plus a small B1 family rail.
It did NOT implement true raw/S0 KL distillation.
```

Run command:

```powershell
uv run python python/scripts/train.py `
  --stack-config configs/presets/pass3_b1_s1_retrain_from_u450_rawprotect.yaml `
  --run-label b1_s1_retrain_u450_to_u460_fix2_20260427 `
  --runtime-mode train_async_fast `
  --autoscale `
  --hardware-profile local `
  --resume-from runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425/training/checkpoints/checkpoint_450.pt `
  --resume-allow-config-mismatch `
  --b1-baseline-run-dir runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425 `
  --max-updates 460 `
  --checkpoint-interval-updates 5 `
  --profile-timers
```

Runtime lane evidence:

```text
pfsp_sampling_weight_noleague_baseline: 0.5
pfsp_sampling_weight_heuristic_public: 0.1
pfsp_sampling_weight_heuristic_public_variant: 0.15
pfsp_sampling_weight_mirror: 0.25
pfsp_sampling_weight_recent: 0.0
pfsp_sampling_weight_champion: 0.0
pfsp_sampling_weight_warmup_snapshot: 0.0
pfsp_sampling_weight_hard_negative: 0.0

collector_b1_opponent_env_steps: about 8k/update
collector_b1_opponent_train_rows: about 4k/update
```

Training tail at u460:

```json
{
  "loss": 0.3731667995452881,
  "policy_loss": 0.007906601764261723,
  "value_loss": 0.00552163552492857,
  "entropy": 0.6152474880218506,
  "reference_policy_top_action_bc_coef": 0.0,
  "reference_policy_top_action_family_bc_coef": 0.025714285714285714,
  "policy_train_fraction": 0.509033203125,
  "reward_mean": -0.0021991729736328125,
  "reward_abs_mean": 0.02392578125,
  "target_mean": 0.358154296875,
  "target_abs_mean": 0.5439453125,
  "vtrace_rho_p95": 314.3132019042969,
  "vtrace_rho_p99": 353750.8125
}
```

The vtrace spike at u460 may be update/checkpoint boundary lag, but should be monitored.

## B1/S1 Restart Eval Results

Candidate:

```text
runs/b1_s1_retrain_u450_to_u460_fix2_20260427/training/checkpoints/checkpoint_460.pt
```

### S1 low-bias, 16 pairs

Artifact:

```text
runs/b1_s1_retrain_u450_to_u460_fix2_20260427/eval/b1_s1_retrain_eval_s1_p16_20260427
```

Results:

```text
B1 -> u460:
  mean 0.50
  pair classes: 1-1:16

u460 -> B1:
  mean 0.50
  pair classes: 1-1:16

u460 -> B3 HeuristicPublicAggro:
  mean 0.50
  pair classes: 1-1:16

u460 -> B4 HeuristicPublicControl:
  mean 0.6875
  pair classes: 2-0:6, 1-1:10
```

Interpretation:

```text
No B1 movement on intended S1 surface.
Still exact paired split parity.
Not a promising candidate.
```

### S0 raw/no-bias, 8 pairs

Artifact:

```text
runs/b1_s1_retrain_u450_to_u460_fix2_20260427/eval/b1_s1_retrain_eval_s0_p8_20260427
```

Results:

```text
B1 -> u460:
  mean 0.6875

u460 -> B1:
  mean 0.1875

u460 -> B3 HeuristicPublicAggro:
  mean 0.0

u460 -> B4 HeuristicPublicControl:
  mean 0.0
```

Interpretation:

```text
Raw/S0 body is still worse than B1.
Raw/no-bias policy collapses against B3/B4.
Gate-only raw protection did not protect the raw body.
```

### S3 official/deployment wrapper, 8 pairs

Artifact:

```text
runs/b1_s1_retrain_u450_to_u460_fix2_20260427/eval/b1_s1_retrain_eval_s3_p8_20260427
```

Results:

```text
B1 -> u460:
  mean 0.50
  pair classes: 1-1:8

u460 -> B1:
  mean 0.50
  pair classes: 1-1:8

u460 -> B3 HeuristicPublicAggro:
  mean 1.0

u460 -> B4 HeuristicPublicControl:
  mean 1.0
```

Interpretation:

```text
S3 still looks fine because wrapper bias scale 3.0 dominates.
It does not prove learned-policy improvement.
```

## Current Code Shape Relevant to Reference BC

The B1 anchor import path was fixed so B1 is imported when reference BC needs it.

Current reference attachment concept in `python/scripts/train.py`:

```python
def _attach_reference_policy_model_if_configured(...):
    coef = float(getattr(training_config, "reference_policy_top_action_bc_coef", 0.0))
    family_coef = float(getattr(training_config, "reference_policy_top_action_family_bc_coef", 0.0))
    if coef == 0.0 and family_coef == 0.0:
        return
    policy_id = str(getattr(training_config, "reference_policy_id", "") or "").strip()
    if not policy_id:
        policy_id = _PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID
    weights_path = training_paths.snapshots_dir / policy_id / SNAPSHOT_WEIGHTS_FILENAME
    payload = torch.load(weights_path, map_location=device, weights_only=True)
    reference_model = build_policy_value_model(...)
    reference_model.load_state_dict(payload["model_state_dict"])
    _restore_model_guidance_from_payload(reference_model, payload)
    reference_model.eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)
    learner.reference_policy_model = reference_model
```

Potential issue:

```text
_restore_model_guidance_from_payload(reference_model, payload) likely restores B1 checkpoint guidance/bias.
So the reference policy is not necessarily raw/S0.
```

Current reference BC/family BC conceptual code in `ImpalaLearner`:

```python
reference_result = self._evaluate_factorized_model_time_major(
    reference_model,
    batch,
    obs=obs,
    packed_legal=packed_legal,
    actions=None,
)
reference_top_actions = reference_result.top_action_ids

current_factorized_result = self._evaluate_factorized_model_time_major(
    forward_model,
    batch,
    obs=obs,
    packed_legal=packed_legal,
    actions=safe_reference_actions,
)
current_reference_logp = current_factorized_result.action_logp
current_family_log_probs = current_factorized_result.family_log_probs
```

So the family rail in the B1/S1 branch likely trained:

```text
student effective S1 policy toward B1 effective restored-guidance family choices
```

It did not train:

```text
student raw/S0 logits toward B1 raw/S0 logits
```

That may explain why raw/S0 was not protected.

## Counterfactual Evidence So Far

We built action trace and forced-action support in:

```text
python/scripts/b1_artifact_matrix.py
```

Destructive control:

```text
Baseline pair:
  physical seat 0 wins both swaps.

Forced pass on winning physical seat:
  physical seat 1 wins both swaps.
```

So:

```text
The simulator/action path is live.
Legal action interventions can flip the winner.
Physical-seat invariance is not immutable deck fate.
```

Constructive probes:

```text
- one-action forced alternatives for losing physical seat did not flip winner;
- exact-index multi-action sequences changed trajectory substantially but did not flip winner;
- exact-index approach is brittle because earlier interventions shift later decisions.
```

Current conclusion:

```text
We need prefix-replay, state-conditioned search if we want useful counterfactual labels.
Match by seat + phase + legal_action_fingerprint + decision window, not exact absolute decision index.
```

## Current Best Interpretation

Our current interpretation is:

```text
1. Old league topology was broken/misleading. We fixed much of that.
2. S3 official eval is saturated by public heuristic wrapper scale 3.0.
3. League continuations survived by wrapper strength, not by learning stronger raw bodies.
4. B1/S1 restart from B1 is mechanically correct but gate-only raw protection is insufficient.
5. Self-play/PFSP among B1-like policies is not creating best-response signal.
6. We need either true raw/S0 preservation, explicit B1 best-response labels, or a redesigned league with exploiter roles and typed hard negatives.
```

We are not asking Pro to merely tune LR. If LR matters, say so, but please treat this as a possible structural design failure first.

## What We Need From You

Please help us redesign the system.

We want:

```text
A. A better league topology.
B. A better learning signal.
C. A minimal implementation plan that can be tested locally before server compute.
```

Please be concrete and code-aware.

## Part A: League Topology Design Questions

Please propose a clean league population model.

Current concepts:

```text
source_kind:
  local
  league_import
  seed_import
  baseline_anchor
```

Current registry arrays:

```text
snapshots
champion_snapshots
rejected_snapshots
pinned_snapshots
```

This feels too weak for thesis-grade league semantics.

Should we add a sidecar like:

```json
{
  "policy_id": "policy_000014",
  "role": "main_candidate",
  "status": "admitted_recent",
  "pool_roles": ["recent", "diagnostic"],
  "promotion_tier": "provisional",
  "confirmed_eval_pairs": 8,
  "rejected_for_main": false,
  "hard_negative_eligible": false,
  "parent_policy_id": "policy_000012",
  "lineage_id": "main_b1_s1",
  "surface_profile": "lowbias_s1",
  "last_confirmed_eval": {
    "S3_B1": 0.50,
    "S1_B1": 0.50,
    "S0_B1": 0.1875
  }
}
```

Recommended roles/statuses could include:

```text
fixed_anchor
seed_history
main_candidate
admitted_recent
provisional_frontier_champion
confirmed_frontier_champion
b1_exploiter_candidate
b1_exploiter_confirmed
hard_negative
diagnostic_only
rejected_main
rejected_all
retired
```

Questions:

1. What exact roles/statuses do you recommend?
2. Should this be embedded in snapshot metadata or a mutable `league_state.json` sidecar?
3. What selectors should runtime use?
4. What should PFSP sample from?
5. How should main policy, B1 exploiter, champion exploiter, and hard negatives interact?
6. Should the main policy train against exploiters immediately, or only after exploiters are confirmed?
7. How should imported seed/B1-history snapshots be used, if at all, after B1 initialization?
8. Should we run separate branches/roles in one league or separate training jobs that exchange snapshots?

## Part B: Learning Signal Design Questions

We need to stop producing wrapper-preserved weak raw policies.

Potential options:

### Option 1: True raw/S0 B1 distillation

Goal:

```text
Preserve B1 raw neural body while training on S1.
```

Possible loss:

```text
teacher = B1 checkpoint with public heuristic bias scale 0.0
student = current model with public heuristic bias scale 0.0
loss = KL or CE over legal actions/top-k legal actions
coef = small, e.g. 0.02-0.05
```

Questions:

1. Should teacher and student both be raw/S0?
2. Should this distill full legal-action distribution, top-k logits, top action, or family?
3. How do we prevent it from freezing B1 and blocking best-response improvement?
4. Should distillation apply only to states without counterfactual positive labels?
5. Should it be phase-limited, e.g. preserve raw B1 on mulligan/clock/level/main basics but allow attack/climax deviations?
6. Should it use KL(B1||student), CE top-k, or advantage-weighted behavior cloning?

### Option 2: Prefix-replay counterfactual labels

Goal:

```text
Find state-conditioned legal deviations that improve losing-seat outcomes vs B1.
```

Minimal algorithm:

```text
1. Generate S1 traces for B1-vs-B1, candidate-vs-B1, B1-vs-candidate.
2. Identify losing-seat high-impact decisions.
3. Replay from same seed.
4. Match target by:
   - actor seat
   - phase
   - legal action fingerprint
   - decision window
   - optional public state digest
5. Force one candidate action.
6. Continue normally.
7. Record terminal delta, winner flip, pair-class improvement.
8. Keep positive labels.
```

Questions:

1. Should we prioritize this before more training?
2. What candidate actions should be enumerated?
3. What counts as a positive label if winner does not flip?
4. Should we train an exploiter from these labels or the main policy directly?
5. Should labels be generated under S1 or S0?
6. Should we search losing physical seat, focal seat 1, or both?

### Option 3: Dedicated B1 exploiter

Goal:

```text
Train a role whose job is not broad strength, but finding B1 weaknesses.
```

Possible branch:

```yaml
league:
  role: b1_exploiter
  sampling:
    noleague_baseline_mix_fraction: 0.70
    heuristic_public_mix_fraction: 0.10
    heuristic_public_variant_mix_fraction: 0.10
    mirror_mix_fraction: 0.10
training:
  public_bias_surface: S1
  exact_action_bc_to_b1: 0.0
  raw_b1_distill: small or phase-limited
  counterfactual_teacher: enabled if labels exist
```

Questions:

1. Should B1 exploiter optimize S1 B1, S0 B1, or S3 B1?
2. Should it be allowed to lose B3/B4?
3. How should it be imported into the main league?
4. When does it become a hard negative?
5. Should multiple exploiters be maintained?

### Option 4: Change reward/value targets

Current rewards/targets often show:

```text
reward_abs_mean is small
target_abs_mean can be much larger
win/loss signal may be sparse and weak for early decisions
```

Questions:

1. Is sparse terminal/self-play reward enough here?
2. Should we add B1-specific reward shaping?
3. Should we add pair-class reward, seat-1 reward, or best-response reward?
4. Would this corrupt general play?
5. Should we instead use counterfactual value labels for selected states?

## Part C: Promotion/Eval Redesign Questions

Current concern:

```text
8-pair local eval is too noisy for confirmed champions.
S3 alone is saturated and misleading.
S0 alone may be too harsh.
```

Please propose:

1. Candidate admission gate.
2. Admitted recent gate.
3. Provisional champion gate.
4. Confirmed frontier champion gate.
5. B1 exploiter gate.
6. Thesis-relevant champion gate.

We think gates should include:

```text
S3 B1/B3/B4 sanity
S1 B1 improvement or pair-class improvement
S0 raw preservation
pair classes 2-0 / 1-1 / 0-2
seat splits
swapped-label complement check
trace/logit divergence report
no ambiguous B1 alias without surface label
```

Please give concrete thresholds for local 8/16/32-pair diagnostics and server 64+ pair confirmations.

## Part D: Exact Next Implementation Plan

Please give us a ranked plan for the next coding and experiment loop.

We want something like:

```text
Patch 1:
  implement X
  tests Y
  run Z
  accept/discard criteria

Patch 2:
  implement A
  tests B
  run C
  accept/discard criteria
```

Please be explicit about whether the next step should be:

```text
1. implement true raw/S0 B1 distillation first;
2. implement prefix-replay counterfactual labels first;
3. redesign league state/roles first;
4. train a B1 exploiter first;
5. or something else.
```

## Current Files/Code Paths You May Want in a Follow-Up

We can provide any of these in the next prompt:

```text
python/weiss_rl/learners/impala_learner.py
  _reference_policy_top_action_bc_losses
  _auxiliary_loss_and_metrics
  _evaluate_factorized_model_time_major
  _forward_time_major

python/weiss_rl/model.py or model package
  PolicyValueModel.forward_seat_aware
  set_public_heuristic_logit_bias_scale
  get_public_heuristic_logit_bias_scale
  public heuristic bias application code

python/scripts/train.py
  _attach_reference_policy_model_if_configured
  _ensure_noleague_baseline_anchor
  _apply_scheduled_training_guidance
  learner construction
  snapshot import/resume code
  promotion gate code

python/weiss_rl/runtime.py
  _refresh_opponent_pool
  _sample_opponent_policy_ids
  _assign_episode_roles
  B1 opponent mask/reward scaling
  PFSP lane metrics

python/weiss_rl/league/registry.py
  SnapshotRegistry
  typed selectors
  source_kind/status fields

python/scripts/b1_artifact_matrix.py
  bias sweep
  action traces
  forced-action intervention support

configs/presets/pass3_b1_s1_retrain_from_u450_rawprotect.yaml
configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_selfplay_localpromo.yaml
```

## Most Useful Artifacts We Can Provide Next

Tell us what you want most:

```text
1. Full matrix summaries for S3/S1/S0.
2. One full action trace row set for same seed under B1/u460.
3. B1 raw top-k vs u460 raw top-k for representative states.
4. Reference BC code and model bias override code.
5. Runtime PFSP pool composition logs.
6. Registry JSON examples.
7. Prefix-replay counterfactual probe outputs.
```

## Please Keep This in Mind

We are very open to major changes.

If the right answer is:

```text
The current league topology is too ambiguous; replace it with explicit roles and league_state.json.
```

say that.

If the right answer is:

```text
Do not run more training until you implement raw/S0 distillation and counterfactual labels.
```

say that.

If the right answer is:

```text
The hand-coded public heuristic wrapper is too dominant for a learning thesis unless it is treated as a fixed prior with explicit ablations.
```

say that.

We need a path that can actually lead to improvement, not a nicer story around flat 0.50 results.

## Desired Output Format

Please answer with:

```text
1. Ranked diagnosis.
2. Proposed league topology.
3. Proposed learning objective.
4. Promotion/admission/eval gates.
5. Minimal next code patches.
6. Exact next local experiments.
7. Criteria for server pilot.
8. What code/artifacts you want copied into the next prompt.
```

