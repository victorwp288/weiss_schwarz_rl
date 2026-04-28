# Follow-up Prompt For GPT Pro - Same Session

You are in the same session and have all prior context. We implemented your suggested sharp control: zero-update/near-zero-change B1 parent sanity plus a no-RL raw-distill-only branch, then evaluated S0/S1/S3. Please interpret this as a decision point, not another coefficient-tuning loop.

## Current Big Picture

The project is Weiss Schwarz RL in:

```text
C:\Users\Bruger\Desktop\this one\weiss_schwarz_rl
```

Prior findings you already saw:

```text
1. The original league topology bug is mostly fixed.
   Seed imports are no longer active PFSP recents/champions after handoff.

2. S3 official eval is saturated:
   scoring_mode = learner
   public heuristic bias scale = 3.0
   B1/u480/u540 collapse to exact paired 0.50, all 1-1 pair classes.

3. u480/u540/u460 are not good parents.
   They are wrapper-preserved but weak on raw/low-bias surfaces.

4. The best known parent remains B1 checkpoint 450:
   runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425/training/checkpoints/checkpoint_450.pt

5. True raw/S0 B1 distillation was implemented:
   teacher and student forward with public heuristic bias 0.0,
   legal-action KL,
   optional raw top-action CE,
   metrics for raw top1/topk/family/KL,
   tests verifying bias restore and nonzero loss when student is perturbed.

6. Normal S1 RL branches from B1, with weak/strong/topCE raw distill, did not improve:
   S1 B1 stayed 0.50/all 1-1,
   S0 candidate-vs-B1 was poor,
   B3/B4 raw collapsed,
   S3 stayed deceptively strong.
```

Your last recommendation was: before fully pivoting to counterfactual labels + frozen-B1 residual, run a no-RL/distill-only control. Done.

## New Code Patch

Added a reusable config/learner switch:

```text
training.policy_loss_coef
```

Touched:

```text
python/weiss_rl/config/models.py
python/weiss_rl/config/parse.py
python/weiss_rl/learners/impala_learner.py
python/scripts/train.py
python/weiss_rl/tests/test_impala_learner.py
```

The learner total loss now scales the RL policy loss:

```python
total_loss = (
    (float(self.policy_loss_coef) * policy_loss)
    + (self.value_loss_coef * value_loss)
    - (self.entropy_coef * entropy_mean)
)
```

`policy_loss_coef` is logged in update metrics and custom TensorBoard metrics.

Added test:

```text
test_impala_learner_policy_loss_coef_can_disable_rl_policy_loss
```

Validation:

```text
uv run python -m py_compile python/weiss_rl/learners/impala_learner.py python/weiss_rl/config/models.py python/weiss_rl/config/parse.py python/scripts/train.py

uv run pytest -q \
  python/weiss_rl/tests/test_impala_learner.py::test_impala_learner_policy_loss_coef_can_disable_rl_policy_loss \
  python/weiss_rl/tests/test_impala_learner.py::test_impala_learner_raw_b1_distill_uses_zero_bias_and_restores_scales \
  python/weiss_rl/tests/test_impala_learner.py::test_impala_learner_raw_b1_distill_penalizes_perturbed_student_raw_logits \
  --tb=short
```

Result:

```text
3 passed
```

## New Control Config

Added:

```text
configs/presets/pass3_b1_s1_retrain_from_u450_distillonly_control.yaml
```

Config intent:

```text
B1 checkpoint 450 parent
common S1 actor/learner bias scale = 1.0
policy_loss_coef = 0.0
value_loss_coef = 0.0
entropy_coef = 0.0
behavior/reference BC off
structured teacher aux off
only raw/S0 B1 distillation active
```

Key parsed values:

```text
policy_loss_coef      0.0
value_loss_coef       0.0
entropy_coef          0.0
teacher_family_coef   0.0
teacher_action_coef   0.0
raw_enabled           True
raw_coef              1.0
raw_topce             0.0
```

## Control Run

Command concept:

```text
train.py
  --stack-config configs/presets/pass3_b1_s1_retrain_from_u450_distillonly_control.yaml
  --run-label b1_s1_distillonly_u450_to_u455_20260427
  --runtime-mode train_async_fast
  --autoscale --hardware-profile local
  --resume-from runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425/training/checkpoints/checkpoint_450.pt
  --resume-allow-config-mismatch
  --b1-baseline-run-dir runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425
  --max-updates 455
  --checkpoint-interval-updates 5
  --profile-timers
```

Run:

```text
runs/b1_s1_distillonly_u450_to_u455_20260427
```

Final metrics at u455:

```text
loss                         9.99e-7
policy_loss                  9.85e-4
policy_loss_coef             0.0
value_loss                   0.0102
entropy                      0.6076
entropy_coef                 0.0
raw_b1_distill_loss/raw_kl    9.99e-7
raw_b1_top1_match            0.9789
raw_b1_topk_overlap          0.9997
raw_b1_family_match          0.9926
teacher_aux_loss             0.0
```

Parameter movement from B1 parent checkpoint 450 to distill-only u455:

```text
float params compared  2,965,340
L2 distance            0.00220
max abs delta          0.000102
changed tensors        187 / 291
```

So it is not perfectly frozen, but the movement is tiny.

## New Matrix Artifacts

```text
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_matrix_distillonly_s0_p8_20260427
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_matrix_distillonly_s1_p8_20260427
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_matrix_distillonly_s3_p8_20260427
```

Important manifest fact:

```text
B1 NoLeague baseline vs b1parent l2_distance = 0.0
B1 NoLeague baseline vs u455     l2_distance = 0.0021956
b1parent vs u455                 l2_distance = 0.0021956
```

Note: because this matrix uses stochastic sampling, B1 vs b1parent at S0/S1 is not always exactly 0.50 even though weights match. Complement checks pass.

## S0 p8 Results

```text
b1parent vs B1  0.625
B1 vs b1parent  0.5625
u455 vs B1      0.625
B1 vs u455      0.3125
b1parent vs B3  0.0
u455 vs B3      0.0
b1parent vs B4  0.0
u455 vs B4      0.0
```

## S1 p8 Results

```text
b1parent vs B1  0.5625
B1 vs b1parent  0.4375
u455 vs B1      0.5
B1 vs u455      0.5
b1parent vs B3  0.5
u455 vs B3      0.5
b1parent vs B4  0.5625
u455 vs B4      0.625
```

## S3 p8 Results

```text
b1parent vs B1  0.5
B1 vs b1parent  0.5
u455 vs B1      0.5
B1 vs u455      0.5
b1parent vs B3  0.9375
u455 vs B3      1.0
b1parent vs B4  1.0
u455 vs B4      1.0
```

## My Current Interpretation

The no-RL/distill-only branch did not reproduce the catastrophic raw/S0 closed-loop weakness seen in normal S1 RL branches. It stayed extremely close to B1 in parameter space and retained essentially the same broad S0/S1/S3 profile as the parent on this tiny matrix.

That means:

```text
raw distillation plumbing is probably not the main bug;
generic S1 RL/self-play updates are the destructive component;
the current learner can move important decisions enough to hurt closed-loop raw play,
but it still does not discover positive B1 best-response deviations.
```

So this supports your recommendation:

```text
stop tuning raw-distill coefficients;
pivot to state-conditioned positive labels;
use an identity-preserving policy form, likely frozen B1 + residual deviation/exploiter head.
```

## Questions For You

Please give a decisive next plan under time pressure. We are open to radical changes.

1. Does this no-RL control conclusively separate raw-distill mechanics from RL destructive drift, or is there still a hidden recurrent/distill issue worth auditing before pivot?

2. Given B1 parent is S3-strong but S0 weak vs B3/B4, should the next target be:
   - S1 B1 exploiter only,
   - S1 main policy,
   - raw/S0 rebuild,
   - or a fresh-from-scratch training redesign?

3. Please design the minimal frozen-B1 residual policy patch:
   - where to wrap the model;
   - whether residual acts on raw logits before public bias or final logits after public bias;
   - how to zero-init it;
   - what metrics/gates prove it is identity-preserving;
   - what exact loss to use first.

4. Please design the minimal counterfactual label generator in codebase-aware terms:
   - should it extend `python/scripts/b1_artifact_matrix.py` forced-action machinery or be a new script?
   - what fields are essential for label rows?
   - how many target states/action candidates are enough for a first proof?
   - what success/failure threshold should decide whether to train the residual exploiter?

5. If we are running out of time and need a defensible thesis result fast, what is the fastest honest result path?
   - a defensible negative/diagnostic result about S3 saturation and sparse self-play failure?
   - a B1 exploiter result on S1?
   - a redesigned lower-bias agent?
   - a fresh supervised/counterfactual best-response experiment?

## Code/Artifact Blocks You May Want Next

The most useful snippets to request next:

```text
python/weiss_rl/model.py
  PolicyValueModel.forward_seat_aware
  public heuristic bias application

python/weiss_rl/learners/impala_learner.py
  _evaluate_factorized_model_time_major
  _forward_time_major
  raw distill helper code
  loss assembly

python/scripts/b1_artifact_matrix.py
  forced action / forced pass path
  matrix runner selection logic
  pair-table writing

python/weiss_rl/runtime.py
  actor/opponent policy scoring-mode and public-bias handling
```

The most useful artifacts to request next:

```text
1. One S1 action trace for B1-vs-B1 or B1-vs-u455 with raw/final top-k.
2. First-divergence trace between normal RL u460 and B1 under S0/S1.
3. The no-RL distill-only u455 policy load manifest.
4. A forced-pass destructive-control artifact proving intervention still works.
```

Bottom line I need you to help decide:

```text
Do we now commit hard to counterfactual labels + frozen-B1 residual exploiter,
or is there a faster/riskier radical route that is more likely to produce a defensible result quickly?
```
