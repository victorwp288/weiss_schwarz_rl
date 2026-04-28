# GPT Pro follow-up prompt: counterfactual-label smoke and next radical move

Same session/context as before. We implemented your recommended pivot toward S1 counterfactual labels before building the frozen-B1 residual exploiter. Please analyze the new evidence and help choose the fastest defensible next step.

## Current high-level context

We are in the Weiss Schwarz RL thesis project. The core blocker remains:

```text
S3 official eval = learner scoring + public heuristic bias scale 3.0
S3 collapses B1/u480/u540 to exact paired 0.50/all 1-1.
S1 = learner scoring + common public heuristic bias scale 1.0.
S0 = raw/no public heuristic bias.
```

Earlier evidence:

```text
1. Original league topology bug was mostly fixed:
   seed imports no longer occupy true recent/champion lanes.

2. S3 is saturated:
   public heuristic wrapper often overrides raw network preferences.

3. u480/u540/u460 are not good main parents:
   they are S3-wrapper-preserved but raw/low-bias weak.

4. True raw/S0 B1 distillation was implemented and tested.
   Normal S1 RL branches with weak/strong/topCE raw distill did not improve B1.

5. No-RL/distill-only control separated the failure mode:
   policy_loss_coef = 0
   value_loss_coef = 0
   entropy_coef = 0
   raw_b1_distill_loss ≈ 1e-6
   param L2 movement from B1 ≈ 0.0022
   S0/S1/S3 profile stayed broadly parent-like on a tiny matrix.

Conclusion from last round:
   raw distill plumbing is probably not the main bug;
   generic S1 RL/self-play updates are the destructive/low-signal component.
```

You recommended:

```text
Stop tuning raw distill coefficients.
Stop full-model S1 self-play.
Build S1 B1 counterfactual labels.
Then train a frozen-B1 residual B1 exploiter.
```

## New code implemented

We added minimal replay/intervention support:

```text
python/scripts/b1_artifact_matrix.py
  + action traces now emit actual legal_ids, not just legal_ids_sha256.
  + forced action can be restricted by --force-action-pair-index.

python/scripts/b1_counterfactual_labels.py
  + runs S1 B1-vs-B1 baseline traces.
  + targets losing physical-seat decisions.
  + filters to high-impact selected-action families:
      clock_from_hand
      level_up
      main_play_character
      main_move
      attack
      encore_pay
      encore_decline
  + builds candidate actions from:
      pass if legal
      raw top-k
      S1 final top-k
      legal action fallback
  + replays one forced legal action at a time.
  + writes counterfactual_labels.jsonl and counterfactual_summary.json.
  + initially used subprocess-per-action, then added in-process mode that loads policies once and runs repeated matrix matchups in one process.
```

Validation:

```text
uv run python -m py_compile python/scripts/b1_artifact_matrix.py python/scripts/b1_counterfactual_labels.py
```

passed.

## Destructive-control result

Artifact:

```text
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_cf_destructive_pass_s1_smoke_p1_20260427
```

Same S1 seed scope as the label smoke.

Baseline pair 0:

```text
episode_seed = 338610598310627562
focal_as_seat0 winner_seat = 0
focal_as_seat1 winner_seat = 0
pair_class = 1-1
```

Forced destructive intervention:

```text
--force-pass-seat 0
--force-pass-max-per-episode 3
forced_pass_decisions = 6
```

Result:

```text
focal_as_seat0 winner_seat = 1
focal_as_seat1 winner_seat = 1
pair_class still 1-1 because B1-vs-B1 symmetric, but physical winner changed
```

Interpretation:

```text
The intervention path is live.
Outcomes are not immutable deck/seat fate.
Bad forced actions can flip the physical winner.
```

## Constructive one-step search results

### Initial tiny smoke

Artifact:

```text
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_cf_labels_s1_smoke_p1_20260427
```

Summary:

```text
pairs                  1
trace_rows             120
target_states          4
attempted replays      8
forced_misses          0
winner_flip_labels     0
```

This spent early budget on the first losing-seat decisions, including mulligan/clock.

### Focused single-seed search

Artifact:

```text
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_cf_labels_s1_focused_p1_20260427
```

Summary:

```text
pairs                  1
trace_rows             160
target_states          4
attempted replays      12
forced_misses          0
winner_flip_labels     0
target families        clock/main/attack/level-up/encore
```

### Focused multi-seed search

Artifact:

```text
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_cf_labels_s1_focused_p3_20260427
```

Summary:

```text
pairs                  3
trace_rows             480
target_states          6
attempted replays      18
forced_misses          0
winner_flip_labels     0
target families        clock/main/attack/level-up/encore
```

Interpretation:

```text
Cheap one-step top-k/family/pass alternatives did not find positive winner-flip labels in the small local budget.
This does NOT prove B1 is unexploitable, because destructive controls flip outcomes.
It suggests useful deviations may require:
  deeper two-step/multi-step search,
  better candidate generation,
  terminal-margin labels instead of flip-only labels,
  search from states closer to swing points,
  or a different learning signal/reward.
```

## In-process search mode

Artifact:

```text
runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_cf_labels_s1_inproc_smoke_p1_20260427
```

Command used:

```text
python/scripts/b1_counterfactual_labels.py
  --execution-mode in_process
  --pairs 1
  --max-target-states 1
  --max-actions-per-state 2
  --max-forced-replays 2
```

Result:

```text
trace_rows             120
target_states          1
attempted replays      2
forced_misses          0
winner_flip_labels     0
execution_mode         in_process
```

It works mechanically. It still requires full game rollouts per candidate, so it is not free, but it avoids model reload/subprocess overhead and is the right base for a larger search.

## Current concern/question

We are running out of time and need defensible results quickly. The key uncertainty is whether to:

```text
A. continue counterfactual search, but upgrade to two-step beam / terminal-margin labels;
B. build frozen-B1 residual anyway using negative/near-positive/no-flip labels;
C. add reward shaping directly to S1 B1 exploiter training;
D. train from fresh with a better objective/surface;
E. suspect model/algorithm fundamentals and pivot thesis framing.
```

The user is worried that the base model or algorithm is fundamentally broken and asks whether we need reward shaping. My current view is:

```text
The base model is not obviously trash: B1 is a usable anchor.
The blocker is sparse terminal reward plus same-policy self-play plus S3 wrapper saturation.
Broad hand-shaped rewards are risky, but targeted counterfactual/value labels or role-specific B1 exploiter reward may be needed.
```

## Specific things to answer

1. Given destructive interventions flip outcomes but one-step constructive search found zero winner flips, what is the most likely blocker?

2. Should we add terminal-margin labels now instead of requiring winner flips?
   If yes, define a robust terminal-margin function from available episode fields:
   currently pair_table has winner_seat, decision_count, tick_count, termination_reason, but not rich final level/clock/stock unless we add it.

3. Should the next patch be:

```text
two-step beam in b1_counterfactual_labels.py
in-process batched candidate search
forced-action trace/terminal-summary enrichment
frozen-B1 residual wrapper
reward shaping for a B1 exploiter
or something more radical?
```

4. If two-step beam: specify exact algorithm and candidate-selection priorities that are likely to find labels fastest.

5. If reward shaping: specify minimal role-specific B1-exploiter rewards that are least likely to corrupt broad play.

6. If fresh training: explain why it is likely to outperform B1 restart under current evidence, because so far B1 is the best raw/low-bias parent.

7. If base model/algo may be fundamentally weak, what fast falsification tests should we run in code now?

8. What exact artifact would convince you to proceed to frozen residual exploiter training?

## Code paths now most relevant

```text
python/scripts/b1_counterfactual_labels.py
python/scripts/b1_artifact_matrix.py
python/weiss_rl/eval/simulator_runner.py
python/weiss_rl/eval/harness.py
python/weiss_rl/model.py
python/weiss_rl/learners/impala_learner.py
python/scripts/train.py
```

## Current bottom line

We have evidence that:

```text
1. S1 interventions can change winners.
2. Simple one-step losing-seat substitutions do not easily flip winners.
3. Full-model RL/self-play is destructive/flat.
4. The next positive result probably needs either better counterfactual search or an explicit role-specific shaped B1 exploiter.
```

Please recommend the fastest credible next 1-2 code patches and the exact acceptance/discard criteria.
