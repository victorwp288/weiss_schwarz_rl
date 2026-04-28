# GPT Pro follow-up: true raw/S0 distillation was implemented and still did not fix league learning

You are continuing the same GPT Pro debugging session about the Weiss Schwarz RL thesis league system. You have the prior prompt and your prior answers in context. This follow-up gives new evidence from Codex after implementing your recommended "true raw/S0 B1 distillation" patch.

The high-level problem remains:

We hoped that a sound league/self-play setup, initialized from a useful B1 no-league anchor and given enough compute, would trend upward and eventually become much stronger than the B1 anchor. Instead, the system keeps plateauing. The old league topology bug has mostly been repaired, but model-vs-model B1 remains flat at 0.50 on the official S3 surface, and lower-bias/raw surfaces show the learned continuations are not stronger than B1.

Your last recommendation was:

1. Stop continuing u480/u540 as parents.
2. Restart from the strong B1 checkpoint at update 450.
3. Use S1 as the main learning surface:
   - `scoring_mode=learner`
   - common public heuristic bias scale `1.0`
   - actor and learner bias parity.
4. Preserve raw/S0 B1 behavior with true no-bias distillation:
   - teacher = B1 checkpoint with public heuristic bias scale `0.0`
   - student = current model with public heuristic bias scale `0.0`
   - masked legal-action KL/top-k style loss.
5. Do not rely on S3 as the learning surface, because S3 is saturated by public heuristic bias scale `3.0`.
6. If raw/S0 preservation plus S1 retraining does not move, go toward prefix-replay counterfactual labels / B1 exploiter roles.

Codex implemented and tested that recommendation. It did not produce a useful learning improvement.

## Current code changes made by Codex

### Config model

Added a raw B1 distillation config:

```python
@dataclass(frozen=True, slots=True)
class TrainingRawB1DistillConfig:
    enabled: bool = False
    teacher_policy_id: str = "b1_noleague_baseline"
    teacher_surface: str = "raw_s0"
    student_surface: str = "raw_s0"
    coef: float = 0.0
    final_coef: float = 0.0
    start_updates: int = 0
    end_updates: int = -1
    top_k: int = 16
    temperature: float = 1.5
    top_action_ce_coef: float = 0.0
    teacher_public_heuristic_bias_scale: float = 0.0
    student_public_heuristic_bias_scale: float = 0.0
```

This is now a field on `TrainingConfig`:

```python
raw_b1_distill: TrainingRawB1DistillConfig = field(default_factory=TrainingRawB1DistillConfig)
```

### Config parser

`training.raw_b1_distill` now parses:

```yaml
training:
  raw_b1_distill:
    enabled: true
    teacher_policy_id: b1_noleague_baseline
    teacher_surface: raw_s0
    student_surface: raw_s0
    coef: 0.05
    final_coef: 0.02
    start_updates: 450
    end_updates: 650
    top_k: 16
    temperature: 1.5
    top_action_ce_coef: 0.0
    teacher_public_heuristic_bias_scale: 0.0
    student_public_heuristic_bias_scale: 0.0
```

Validation currently requires `teacher_surface=raw_s0` and `student_surface=raw_s0`.

### Learner fields

`ImpalaLearner` now has:

```python
raw_b1_distill_coef: float = 0.0
raw_b1_distill_teacher_bias_scale: float = 0.0
raw_b1_distill_student_bias_scale: float = 0.0
raw_b1_distill_top_k: int = 16
raw_b1_distill_temperature: float = 1.5
raw_b1_distill_top_action_ce_coef: float = 0.0
```

and:

```python
def set_raw_b1_distill_coef(self, value: float) -> None:
    self.raw_b1_distill_coef = float(value)
```

### Critical implementation detail: temporary no-bias forwards

Codex added helper methods:

```python
def _set_public_heuristic_bias_scale_if_supported(self, model: Any, value: float) -> tuple[float, float] | None:
    get_bias_scale = getattr(model, "get_public_heuristic_logit_bias_scale", None)
    set_bias_scale = getattr(model, "set_public_heuristic_logit_bias_scale", None)
    if not callable(get_bias_scale) or not callable(set_bias_scale):
        return None
    previous = (
        float(get_bias_scale(scoring_mode="learner")),
        float(get_bias_scale(scoring_mode="actor")),
    )
    set_bias_scale(float(value), actor_value=float(value))
    return previous

def _restore_public_heuristic_bias_scale_if_supported(
    self,
    model: Any,
    previous: tuple[float, float] | None,
) -> None:
    if previous is None:
        return
    set_bias_scale = getattr(model, "set_public_heuristic_logit_bias_scale", None)
    if callable(set_bias_scale):
        set_bias_scale(float(previous[0]), actor_value=float(previous[1]))
```

The raw distill path:

1. Temporarily sets frozen B1 teacher public bias to `0.0`.
2. Temporarily sets current student public bias to `0.0`.
3. Runs teacher forward under `torch.no_grad()`.
4. Runs student forward with gradients.
5. Computes raw/S0 legal-action KL on packed legal candidates.
6. Optionally adds raw teacher top-action CE.
7. Restores both models to their previous learner/actor public-bias scales.

The implementation intentionally does **not** use the S3 restored-guidance policy for the distillation target.

### Loss shape

For packed legal candidates, it computes segment log-softmax over legal candidates per row:

```python
temperature = max(float(self.raw_b1_distill_temperature), 1.0e-6)
scaled_student = student_logits / temperature
scaled_teacher = teacher_logits.detach() / temperature
student_log_z = _segment_logsumexp(scaled_student, row_indices, row_count)
teacher_log_z = _segment_logsumexp(scaled_teacher, row_indices, row_count)
student_logp = scaled_student - student_log_z.index_select(0, row_indices)
teacher_logp = scaled_teacher - teacher_log_z.index_select(0, row_indices)
teacher_probs = torch.exp(teacher_logp)
row_kl_terms = teacher_probs * (teacher_logp - student_logp)
row_kl.scatter_add_(0, row_indices, row_kl_terms)
kl_loss = (row_kl * active_mask).sum() / denominator
```

Optional top-action CE:

```python
teacher_top_actions = self._packed_top_action_ids(teacher_logits.detach(), ids, offsets)
teacher_top_logp = _packed_selected_action_logp(
    scaled_student,
    ids,
    offsets,
    teacher_top_actions,
    pass_action_id=self.pass_action_id,
    strict=False,
)
top_action_ce = -(
    torch.where(ce_mask > 0.0, teacher_top_logp, torch.zeros_like(teacher_top_logp)) * ce_mask
).sum() / ce_denominator
loss = kl_loss + top_action_ce_coef * top_action_ce
```

This raw distill loss is then multiplied by `raw_b1_distill_coef` and added to the normal IMPALA loss.

### Metrics emitted

Training logs now include:

```text
raw_b1_distill_loss
raw_b1_distill_coef
raw_b1_distill_teacher_bias_scale
raw_b1_distill_student_bias_scale
raw_b1_distill_temperature
raw_b1_distill_top_k
raw_b1_distill_top_action_ce_coef
raw_b1_distill_row_fraction
raw_b1_top1_match
raw_b1_topk_overlap
raw_b1_family_match
raw_b1_kl
raw_b1_top_action_ce
```

### Train script plumbing

`python/scripts/train.py` now:

- passes raw distill config into `ImpalaLearner`;
- schedules `raw_b1_distill_coef`;
- attaches frozen B1 reference policy even when normal reference BC is off;
- imports the B1 baseline anchor for raw distill.

The startup log confirms this:

```text
Attached frozen reference policy: policy_id=b1_noleague_baseline coef=0 family_coef=0 raw_b1_distill=True
```

### Tests passed

New tests:

```text
python/weiss_rl/tests/test_impala_learner.py::test_impala_learner_raw_b1_distill_uses_zero_bias_and_restores_scales
python/weiss_rl/tests/test_impala_learner.py::test_impala_learner_raw_b1_distill_penalizes_perturbed_student_raw_logits
```

They verify:

- teacher/student are evaluated under no-bias for the auxiliary;
- original learner/actor public-bias scales are restored afterward;
- if student equals B1, raw distill loss is near zero;
- if student raw logits are perturbed, raw distill loss increases.

Also passed:

```text
uv run python -m py_compile python/weiss_rl/learners/impala_learner.py python/weiss_rl/config/models.py python/weiss_rl/config/parse.py python/scripts/train.py
```

## Configs used

### Prior gate-only rawprotect baseline, already discarded

File:

```text
configs/presets/pass3_b1_s1_retrain_from_u450_rawprotect.yaml
```

Key shape:

```yaml
model:
  public_heuristic_logit_bias_scale: 1.0
  public_heuristic_actor_logit_bias_scale: 1.0
  public_heuristic_logit_bias_start_updates: 450
  public_heuristic_logit_bias_end_updates: -1
  public_heuristic_logit_bias_final_scale: 1.0
training:
  behavior_action_bc_coef: 0.0
  reference_policy_id: b1_noleague_baseline
  reference_policy_top_action_bc_coef: 0.0
  reference_policy_top_action_family_bc_coef: 0.03
  reference_policy_top_action_family_bc_final_coef: 0.0
  reference_policy_top_action_family_bc_start_updates: 450
  reference_policy_top_action_family_bc_end_updates: 520
  optimizer:
    learning_rate: 0.00001
  exploration:
    entropy_coef: 0.04
    entropy_anneal_to: 0.02
  structured_aux:
    teacher_public_heuristic_coef: 0.0
    teacher_public_heuristic_final_coef: 0.0
    teacher_public_main_move_coef: 0.0
league:
  sampling:
    noleague_baseline_mix_fraction: 0.50
    heuristic_public_mix_fraction: 0.10
    heuristic_public_variant_mix_fraction: 0.15
    mirror_mix_fraction: 0.25
    warmup_snapshot_mix_fraction: 0.0
    champion_mix_fraction: 0.0
    hard_negative_mix_fraction: 0.0
  promotion:
    enabled: false
curriculum:
  checkpoint_guard:
    enabled: false
```

Result: mechanically correct, but did not preserve raw body.

### True raw KL distill

File:

```text
configs/presets/pass3_b1_s1_retrain_from_u450_true_rawdistill.yaml
```

Key shape:

```yaml
extends: pass3_b1_s1_retrain_from_u450_rawprotect.yaml
training:
  reference_policy_top_action_family_bc_coef: 0.0
  reference_policy_top_action_family_bc_final_coef: 0.0
  raw_b1_distill:
    enabled: true
    teacher_policy_id: b1_noleague_baseline
    teacher_surface: raw_s0
    student_surface: raw_s0
    coef: 0.05
    final_coef: 0.02
    start_updates: 450
    end_updates: 650
    top_k: 16
    temperature: 1.5
    teacher_public_heuristic_bias_scale: 0.0
    student_public_heuristic_bias_scale: 0.0
```

### Strong raw KL distill

File:

```text
configs/presets/pass3_b1_s1_retrain_from_u450_true_rawdistill_strong.yaml
```

Key change:

```yaml
raw_b1_distill:
  coef: 2.0
  final_coef: 1.0
```

### Raw KL plus raw top-action CE

File:

```text
configs/presets/pass3_b1_s1_retrain_from_u450_true_rawdistill_topce.yaml
```

Key change:

```yaml
raw_b1_distill:
  coef: 1.0
  final_coef: 0.5
  top_action_ce_coef: 1.0
```

## New runs and results

All runs use parent:

```text
runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425/training/checkpoints/checkpoint_450.pt
```

All are short local B1/S1 restarts, not thesis-grade final runs.

All use:

```text
common actor/learner public heuristic bias = 1.0
promotion disabled
checkpoint guard disabled
sampling lanes:
  B1 baseline: 0.50
  heuristic public: 0.10
  heuristic public variant: 0.15
  mirror: 0.25
  recent/champion/warmup/hard-negative: 0.0
```

### Run A: weak raw KL distill

Run dirs:

```text
runs/b1_s1_rawdistill_u450_to_u455_smoke_20260427
runs/b1_s1_rawdistill_u455_to_u460_20260427
```

u460 training metrics:

```text
raw_b1_distill_loss: 0.00510567
raw_b1_distill_coef: 0.0485
raw_b1_teacher_bias: 0.0
raw_b1_student_bias: 0.0
raw_b1_top1_match: 0.9363
raw_b1_topk_overlap: 0.9965
raw_b1_family_match: 0.9823
raw_b1_kl: 0.00510567
```

S1 p8 eval:

```text
B1 vs u460:   0.50, all 1-1
u460 vs B1:   0.50, all 1-1
u460 vs B3:   0.50, all 1-1
u460 vs B4:   0.625, 2-0:2, 1-1:6
```

S0 p8 eval:

```text
B1 vs u460:   0.6875
u460 vs B1:   0.1875
u460 vs B3:   0.0
u460 vs B4:   0.0
```

S3 p8 eval:

```text
B1 vs u460:   0.50, all 1-1
u460 vs B1:   0.50, all 1-1
u460 vs B3:   1.0
u460 vs B4:   1.0
```

Verdict:

Mechanically correct but not useful. Raw KL preserved top-k/family fairly well but not game strength.

### Run B: strong raw KL distill

Run dir:

```text
runs/b1_s1_rawdistillstrong_u450_to_u460_20260427
```

u460 training metrics:

```text
raw_b1_distill_loss: 0.0037669
raw_b1_distill_coef: 1.95
raw_b1_top1_match: 0.9384
raw_b1_topk_overlap: 0.9967
raw_b1_family_match: 0.9824
raw_b1_kl: 0.0037669
```

S0 p8 eval:

```text
B1 vs u460:   0.625
u460 vs B1:   0.1875
u460 vs B3:   0.0
u460 vs B4:   0.0
```

S1 p8 eval:

```text
B1 vs u460:   0.50, all 1-1
u460 vs B1:   0.50, all 1-1
u460 vs B3:   0.50, all 1-1
u460 vs B4:   0.625
```

Verdict:

Stronger KL did not fix it. It maybe improved the reverse raw B1-vs-candidate matchup slightly but candidate raw focal strength is unchanged.

### Run C: raw KL plus raw top-action CE

Run dir:

```text
runs/b1_s1_rawdistilltopce_u450_to_u460_20260427
```

u460 training metrics:

```text
raw_b1_distill_loss: 0.9749
raw_b1_distill_coef: 0.975
raw_b1_distill_top_action_ce_coef: 1.0
raw_b1_kl: 0.00870
raw_b1_top_action_ce: 0.9662
raw_b1_top1_match: 0.9158
raw_b1_topk_overlap: 0.9966
raw_b1_family_match: 0.9760
```

S0 p8 eval:

```text
B1 vs u460:   0.6875
u460 vs B1:   0.25
u460 vs B3:   0.0
u460 vs B4:   0.0
```

S1 p8 eval:

```text
B1 vs u460:   0.50, all 1-1
u460 vs B1:   0.50, all 1-1
u460 vs B3:   0.50, all 1-1
u460 vs B4:   0.625
```

Verdict:

Top-action CE marginally improved raw `u460 vs B1` from `0.1875` to `0.25`, but it worsened raw top-1 match and did not create S1 progress. It appears to fight the RL objective without solving the game-level problem.

## Important observation

Even when raw top-k overlap is around `0.996`, raw/no-bias game performance can be terrible:

```text
u460 vs B3 raw S0: 0.0
u460 vs B4 raw S0: 0.0
```

So "looks close by top-k" is not enough. A few raw top-action/family deviations may be catastrophic, or the eval surface is highly sensitive to particular early decisions. This also means raw KL metrics may be a poor proxy unless bucketed by phase/state importance.

## Current diagnosis after these experiments

The implementation has moved the uncertainty:

Old uncertainty:

```text
Maybe rawprotect failed because it was not true raw/S0 distillation.
```

New evidence:

```text
True raw/S0 distillation exists and is active.
Weak KL, strong KL, and KL+top-action CE still do not produce a useful B1/S1 branch.
```

Therefore auxiliary preservation alone is probably not the missing learning signal.

The S1 self-play/RL objective appears to push the model into tiny raw decision shifts that:

- do not break B1 parity under S1;
- remain hidden under saturated S3;
- destroy or fail to preserve S0/raw game strength;
- do not improve against B3/B4 under S1 enough to matter.

This suggests the next breakthrough probably requires positive best-response signal, not just preservation:

- prefix-replay counterfactual labels;
- explicit B1 exploiter role;
- state/phase-targeted labels;
- freezing most of the policy and only allowing selected deviations;
- lower-level action trace bucketing to identify the catastrophic raw drift states;
- or a more radical learning objective.

## What I want from you now

Please treat this as a design/debugging question, not as a request for small coefficient tuning.

Given that true raw/S0 distillation did not fix the issue:

1. What is now the most likely root cause?
2. Are these raw distill results enough to conclude S1 generic self-play is the wrong next step?
3. Should we pivot immediately to prefix-replay counterfactual labels, or is there one sharper ablation we should do first?
4. If we do prefix-replay counterfactual labels, please design the exact minimal algorithm:
   - what traces to collect;
   - target-state selection;
   - trigger matching;
   - candidate action generation;
   - one-step vs two-step search;
   - terminal margin function;
   - filtering thresholds;
   - how many examples are enough for a local proof;
   - what training loss should consume the labels.
5. Should the main policy be frozen except for certain heads/layers during B1/S1 retraining?
6. Would you recommend policy-head-only training, adapter/LoRA-style residuals, or a residual "deviation head" on top of frozen B1?
7. Should we stop trying to improve the main policy directly and first train a B1 exploiter/hard-negative?
8. How should the league topology use such an exploiter without destabilizing the main policy?
9. How should S0/S1/S3 gates be changed in light of the fact that top-k raw similarity did not preserve raw performance?
10. Is there any sign here of a bug in the raw distillation implementation, or do the results look like a real optimization/signal problem?

## Candidate next designs I am considering

Please critique these directly.

### Option A: Prefix-replay counterfactual label generator

Use S1 traces and losing physical-seat decisions. Match target decisions by:

```text
actor_seat
phase
legal_action_fingerprint
decision_index window
optional public_state_digest
```

Generate candidate actions from:

```text
B1 raw top-k
B1 S1 top-k
current raw top-k
current S1 top-k
public heuristic top-k
one representative per action family
pass/non-pass alternative
clock/no-clock alternatives
main_play_character alternatives
main_move alternatives
climax/event alternatives
attack alternatives
small random legal sample
```

Roll out after forced action under S1. Keep labels if:

```text
winner flips
or terminal margin improves
or pair class improves
```

Then train a B1 exploiter or main residual head on those labels.

### Option B: Frozen B1 trunk/policy residual

Freeze most of B1. Add trainable residual logits:

```text
final_logits = B1_raw_logits + alpha * residual_logits + public_bias(S1)
```

Train only residual with:

```text
RL S1 loss
small residual norm penalty
counterfactual positive labels if available
strict S0/S1/S3 eval gates
```

This prevents the raw body from drifting catastrophically but allows selected deviations.

### Option C: Policy-head-only S1 retraining

Freeze trunk/recurrent body and train only policy head under S1 with raw B1 KL/CE. This is cheaper than a residual architecture but may still wreck action choices.

### Option D: B1 exploiter-first

Do not try to make the main policy better yet. Train a role-specific B1 exploiter:

```text
target surface: S1
main opponent: B1 S1
raw/S0 preservation: weak, only sanity
B3/B4 allowed to degrade
counterfactual labels prioritized
```

If it improves S1 B1 or pair-score, import it as:

```text
role=b1_exploiter
status=b1_exploiter_confirmed
pool_roles=["b1_exploiter", "hard_negative"]
rejected_for_main=true
```

Then train the main policy against B1 + B1 exploiter + B3/B4.

### Option E: Diagnose catastrophic raw drift first

Before more training, compare B1 raw vs u460 raw decisions on the same states and find which phases/action families account for the raw S0 collapse:

```text
mulligan
clock
level-up
main_play_character
main_move
climax/event
attack
encore
pass with non-pass available
```

Then apply phase-specific preservation or labels. This might be necessary because global KL/top-k is not predictive.

## What code/artifacts would help you in the next packet?

Tell Codex exactly what to provide. For example:

```text
1. Full raw distill helper code.
2. Full Impala loss section around total_loss.
3. Policy head public-bias application code.
4. S0/S1 trace rows comparing B1 and rawdistill u460.
5. Pair tables for rawdistill variants.
6. Training metrics around vtrace rho by update.
7. Config YAMLs for all three rawdistill variants.
8. Action-family mismatch summaries for B1 raw vs u460 raw.
```

The next action should be something that can break the plateau, not another tiny config tweak unless you think one specific ablation is genuinely diagnostic.
