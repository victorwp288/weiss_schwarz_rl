# GPT Pro Follow-Up: B1/S1 Restart Was Mechanically Correct but Did Not Improve

You are in the same ongoing Weiss Schwarz RL thesis rescue/debugging session. You should assume you have the previous prompt, your previous two answers, and the broad context:

- We had a league/self-play system that was supposed to become much stronger than a B1 no-league anchor.
- Old league results were misleading because previous/recent/champion topology was partly broken.
- We repaired seed quarantine and true local recent/champion selection.
- Then we found S3 official eval with public heuristic bias scale `3.0` is saturated:
  - B1/u480/u540 exact 0.50 vs each other, all 1-1 pairs.
  - Both-greedy still exact 0.50.
  - Bias sweep showed lower-bias surfaces expose differences.
  - No-bias showed u480/u540 raw bodies are weaker than B1.
- We then tested destructive and constructive counterfactuals:
  - destructive forced-pass controls can flip the physical winner, so simulator/action intervention path is live.
  - one-action and exact-index multi-action constructive probes did not flip the losing physical seat.
  - conclusion: exact-index constructive probes are too brittle/weak, not proof that B1 is unexploitable.
- Your most recent recommendation was:
  - stop continuing u480/u540/S3 PFSP;
  - restart from the B1 checkpoint_450;
  - train with common actor/learner public-bias parity at S1 (`1.0`);
  - preserve raw/S0 B1 behavior;
  - evaluate S3/S1/S0;
  - keep counterfactual search as a smaller diagnostic track.

This follow-up reports that we implemented and tested that direction. It did not work in its simplest gate-only form.

## Current Best B1 Parent and Checkpoints

Strong B1 no-league anchor run:

```text
runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425
```

Parent checkpoint used for this experiment:

```text
runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425/training/checkpoints/checkpoint_450.pt
```

This is not u480/u540. We explicitly restarted from B1, per your recommendation.

## Code Changes Made

### 1. Added explicit eval surface labels to matrix diagnostics

File:

```text
python/scripts/b1_artifact_matrix.py
```

Patch concept:

```python
parser.add_argument(
    "--surface-name",
    default="",
    help="Optional eval-surface label, e.g. official_s3/lowbias_s1/raw_s0.",
)
```

Persisted into:

```text
policy_load_manifest.json
per-matchup evaluation_context
matrix_summary.json
```

Example persisted fields:

```json
{
  "surface_name": "lowbias_s1",
  "scoring_mode": "learner",
  "public_heuristic_bias_override_requested": true,
  "public_heuristic_bias_override_scale": 1.0
}
```

### 2. Found/fixed B1 anchor import bug for reference-policy BC

The first smoke failed before training:

```text
FileNotFoundError:
reference policy weights not found for policy_id='b1_noleague_baseline':
runs/b1_s1_retrain_smoke_u450_to_u452_20260427/training/snapshots/b1_noleague_baseline/weights.pt
```

Cause:

```text
_ensure_noleague_baseline_anchor(...) imported the B1 anchor only when promotion gating required B1,
or when the current run was itself permitted to become the B1 alias.

It did not import B1 when B1 was needed only as a frozen reference policy for reference_policy_top_action_family_bc_coef.
```

Patch concept in:

```text
python/scripts/train.py
```

```python
reference_policy_id = str(getattr(training_config, "reference_policy_id", "") or "").strip()
if not reference_policy_id:
    reference_policy_id = _PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID

reference_needs_b1_anchor = (
    float(getattr(training_config, "reference_policy_top_action_bc_coef", 0.0)) != 0.0
    or float(getattr(training_config, "reference_policy_top_action_family_bc_coef", 0.0)) != 0.0
    or float(getattr(training_config, "b1_opponent_reference_policy_top_action_bc_coef", 0.0)) != 0.0
) and reference_policy_id in _promotion_anchor_policy_id_candidates(_PROMOTION_GATE_NOLEAGUE_BASELINE_NAME)

requires_anchor = bool(
    reference_needs_b1_anchor
    or (
        league is not None
        and league.enabled
        and league.promotion_gate_enabled
        and _PROMOTION_GATE_NOLEAGUE_BASELINE_NAME in league.promotion_anchor_set_v1.required
    )
)
```

Regression test added:

```text
python/weiss_rl/tests/test_snapshot_registry.py::test_ensure_noleague_baseline_anchor_imports_for_reference_bc_without_promotion
```

Validation passed:

```text
uv run python -m py_compile python/scripts/train.py python/scripts/b1_artifact_matrix.py

uv run pytest -q \
  python/weiss_rl/tests/test_snapshot_registry.py::test_ensure_noleague_baseline_anchor_imports_for_reference_bc_without_promotion \
  python/weiss_rl/tests/test_snapshot_registry.py::test_guidance_schedule_applies_configured_actor_bias_after_resume \
  --tb=short
```

## New Config Tested

File:

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

Important note:

```text
This config says "rawprotect" but the protection is gate-only plus a small B1 family rail.
It does NOT implement a true raw/S0 KL or top-k distillation loss.
```

## Why the Sampler Was Adjusted

First successful 2-update smoke revealed:

```text
lane weights summed to 0.90
runtime assigned leftover 0.10 to recent
recent pool contained imported B1-history snapshots from the B1 run
```

That was not sampled in the inspected tail, but it made the branch semantically messy.

Fix:

```yaml
mirror_mix_fraction: 0.25
```

Now fixed lanes sum to exactly:

```text
B1 baseline 0.50
heuristic public 0.10
heuristic variant 0.15
mirror 0.25
= 1.00
```

Runtime tail confirmed:

```json
{
  "pfsp_sampling_weight_noleague_baseline": 0.5,
  "pfsp_sampling_weight_heuristic_public": 0.1,
  "pfsp_sampling_weight_heuristic_public_variant": 0.15,
  "pfsp_sampling_weight_mirror": 0.25,
  "pfsp_sampling_weight_recent": 0.0,
  "pfsp_sampling_weight_champion": 0.0,
  "pfsp_sampling_weight_warmup_snapshot": 0.0,
  "pfsp_sampling_weight_hard_negative": 0.0
}
```

B1 pressure was live:

```text
tail collector_b1_opponent_env_steps: about 8k/update
tail collector_b1_opponent_train_rows: about 4k/update
```

## Training Run

Run:

```text
runs/b1_s1_retrain_u450_to_u460_fix2_20260427
```

Command:

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

Completed:

```text
checkpoint_455.pt
checkpoint_460.pt
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

The vtrace spike at u460 may be update/checkpoint boundary lag, but it is worth monitoring if this branch is revisited.

## Multi-Surface Eval Results

Candidate:

```text
runs/b1_s1_retrain_u450_to_u460_fix2_20260427/training/checkpoints/checkpoint_460.pt
```

### S1 low-bias, 16 pairs

Artifact:

```text
runs/b1_s1_retrain_u450_to_u460_fix2_20260427/eval/b1_s1_retrain_eval_s1_p16_20260427
```

Command surface:

```text
--surface-name lowbias_s1
--public-heuristic-bias-scale 1.0
--pairs 16
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
The intended S1 branch did NOT move against B1.
It is still exact physical-seat split parity.
It also did not clearly preserve broader S1 heuristic strength.
```

### S0 raw/no-bias, 8 pairs

Artifact:

```text
runs/b1_s1_retrain_u450_to_u460_fix2_20260427/eval/b1_s1_retrain_eval_s0_p8_20260427
```

Command surface:

```text
--surface-name raw_s0
--disable-public-heuristic-bias
--pairs 8
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
It collapses versus B3/B4 without wrapper.
Gate-only raw protection did not actually protect raw body.
```

### S3 official/deployment wrapper, 8 pairs

Artifact:

```text
runs/b1_s1_retrain_u450_to_u460_fix2_20260427/eval/b1_s1_retrain_eval_s3_p8_20260427
```

Command surface:

```text
--surface-name official_s3
--public-heuristic-bias-scale 3.0
--pairs 8
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
S3 still looks "good" because it is saturated by wrapper bias scale 3.0.
It gives no evidence of learned B1 improvement.
```

## Current Verdict

This branch is mechanically correct and useful as a negative result, but not a candidate.

It shows:

```text
B1 checkpoint_450 -> common S1 training for 10 updates
with B1 pressure, heuristic sanity lanes, mirror lane, and small B1 family rail
does NOT produce immediate S1 B1 movement.
```

More importantly:

```text
It does NOT protect raw/S0 body.
```

So the simplest version of your recommendation:

```text
B1 parent + S1 parity + gate-only raw protection
```

is not sufficient.

## Key Suspicion

The config's "rawprotect" name is misleading. It did not implement true raw/S0 preservation.

Current reference policy attachment:

```python
reference_model = build_policy_value_model(...)
reference_model.load_state_dict(payload["model_state_dict"])
_restore_model_guidance_from_payload(reference_model, payload)
reference_model.eval()
learner.reference_policy_model = reference_model
```

Potential issue:

```text
The reference model restores B1 checkpoint guidance from payload.
For the B1 checkpoint, that likely means B1's effective guidance/bias, not raw S0.
```

Learner reference BC/family BC appears to use effective model scoring:

```python
reference_result = self._evaluate_factorized_model_time_major(reference_model, ...)
...
current_factorized_result = self._evaluate_factorized_model_time_major(forward_model, ...)
```

So the small family rail probably does NOT implement:

```text
teacher = B1 raw/S0 logits
student = current raw/S0 logits
loss = legal-action KL/top-k CE
```

It is closer to:

```text
teacher = B1 effective policy after restored guidance
student = current effective policy under current S1 guidance
family CE only
```

That may not preserve the raw body at all.

## What We Need Help With Now

Please analyze the result and recommend the next concrete step.

I see three plausible paths:

### Path A: Implement true raw/S0 B1 distillation

Add an auxiliary loss that explicitly evaluates:

```text
B1 teacher with public heuristic bias scale 0.0
current student with public heuristic bias scale 0.0
```

on the same legal-action rows, and applies a small:

```text
top-k KL
or top-k CE
or family+top-action mixed loss
```

only over legal actions.

Question:

```text
How exactly should this be implemented in this code shape without corrupting actor/eval guidance state?
```

Candidate design:

```yaml
training:
  raw_b1_distill:
    enabled: true
    teacher_policy_id: b1_noleague_baseline
    teacher_public_heuristic_bias_scale: 0.0
    student_public_heuristic_bias_scale: 0.0
    coef: 0.05
    final_coef: 0.02
    top_k: 8
```

Implementation questions:

```text
Should the learner temporarily set model bias scales to 0.0 during the auxiliary forward and restore afterward?
Should it build a separate no-bias clone of the current model each update? That seems expensive.
Can forward_seat_aware accept a scoring/bias override argument cleanly?
Should this be a reference-policy loss inside ImpalaLearner or a structured auxiliary loss?
Should it distill top-k raw logits or only top action/family?
How do we avoid simply freezing B1 and preventing S1 improvement?
```

### Path B: Lower-LR/stronger-rail S1 retry

Try:

```text
LR 2e-6 or 5e-6
stronger B1 family rail
maybe exact-action BC to B1 raw/low-bias
```

But this may just freeze the B1 clone and still not improve B1.

Question:

```text
Is this worth doing before true raw distill/counterfactual labels,
or is it likely a time sink?
```

### Path C: Prefix-replay counterfactual labels first

Continue your state-conditioned S1 prefix-replay search:

```text
match target by seat + phase + legal_fingerprint + decision window,
not exact global decision index.
```

Then use positive labels as:

```text
preserve B1 raw where no evidence,
deviate where counterfactual says better.
```

Question:

```text
Given the S1 retrain failure, should we prioritize counterfactual positive labels before any more training?
```

## Ask for GPT Pro

Please give a ranked diagnosis and a concrete next patch/experiment plan.

Be specific. We want code-level guidance, not just conceptual advice.

Please answer:

1. Did this B1/S1 result falsify your proposed "B1 checkpoint_450 -> common S1 retrain" branch, or only the gate-only/no-true-raw-distill version?
2. Is the most likely failure now:
   - missing true raw/S0 preservation;
   - still no B1 best-response signal;
   - S1 too weak/strong as a wrapper;
   - LR too high;
   - reference/family rail mis-specified;
   - or something else?
3. What is the minimum correct raw B1 distillation implementation for this codebase?
4. Should raw distill use:
   - KL over all legal actions;
   - top-k KL;
   - top-action CE;
   - family CE;
   - or advantage/counterfactual-weighted labels?
5. Should the student raw forward be no-bias, or should only the teacher be no-bias?
6. Should we train from B1 again with true raw distill, or stop training until prefix-replay counterfactual search finds positive B1 deviations?
7. What should the next 10-update local experiment look like exactly?
8. What surface/gate thresholds should determine accept/discard?

## Code Blocks That Would Be Useful Next If You Need Them

Tell us which of these you want copied into the next prompt:

```text
python/weiss_rl/learners/impala_learner.py
  _reference_policy_top_action_bc_losses
  _auxiliary_loss_and_metrics
  _evaluate_factorized_model_time_major
  _forward_time_major

python/weiss_rl/model.py or relevant model file
  PolicyValueModel.forward_seat_aware
  set_public_heuristic_logit_bias_scale
  get_public_heuristic_logit_bias_scale
  public heuristic bias application code

python/scripts/train.py
  _attach_reference_policy_model_if_configured
  _apply_scheduled_training_guidance
  learner construction where ImpalaLearner receives BC coefficients

python/weiss_rl/config/models.py and parse.py
  TrainingConfig fields
  parsing additions for raw_b1_distill

runtime/training batch fields:
  legal_actions
  packed legal action meta
  b1_opponent_mask
  policy_train_mask
  to_play_seat/actor
```

Most useful artifact to request:

```text
For one batch row or trace state:
  B1 raw/S0 top-k,
  u460 raw/S0 top-k,
  B1 S1 top-k,
  u460 S1 top-k,
  public heuristic top-k,
  final chosen action,
  action family,
  whether raw student drifted away from B1.
```
