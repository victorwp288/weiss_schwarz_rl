# GPT Pro Follow-Up: Bias Sweep + Trace Probe + Counterfactual Intervention Results

This is a follow-up in the same Weiss Schwarz RL league debugging session.

Assume you have the previous prompts and your prior outputs. This packet adds the newest diagnostics after your recommendation to:

1. run common-bias sweeps;
2. add action trace/logit probes;
3. run destructive forced-action controls;
4. then try constructive counterfactuals before more PFSP/training.

## Current high-level state

The league topology bug is mostly repaired:

```text
seed imports are quarantined from true PFSP recent/champion lanes
league_import/local policies can appear as true recent/champion artifacts
old "previous recent secretly means seed B1" failure is materially fixed
```

But the learning-quality blocker is now sharper:

```text
official learner-mode public heuristic bias scale 3.0 saturates B1-family policies
B1/u480/u540 collapse to exact paired 0.50 under S3
u480/u540 are weaker than B1 on low-bias/raw surfaces
actor-mode wins for u480/u540 over B1 are wrapper-mismatch artifacts
```

We have not resumed PFSP/server training. We are still diagnosing the learning signal.

## Key checkpoints

B1 anchor:

```text
runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425/training/checkpoints/checkpoint_450.pt
```

Current best old league checkpoint:

```text
runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/training/checkpoints/checkpoint_480.pt
```

Topology-fixed but not better continuation:

```text
runs/league_typedpools_seedquarantine_u500_to_u540_20260427/training/checkpoints/checkpoint_540.pt
```

## New/extended scripts

### `python/scripts/b1_artifact_matrix.py`

Already had:

```text
policy load manifests
pair tables
builtin contrast policies
common public-bias override
learner/actor scoring-mode override
both-greedy mode
hard decision counters
```

New additions:

```text
--emit-action-traces
--trace-top-k
--trace-max-decisions-per-episode
--force-pass-seat
--force-pass-max-per-episode
--force-action-seat
--force-action-decision-index
--force-action-id
--force-action-swap-index
--force-action-sequence
```

Trace rows include:

```json
{
  "pair_index": 0,
  "swap_index": 0,
  "episode_seed": 338610598310627562,
  "decision_index": 15,
  "actor_seat": 1,
  "policy_id": "u540",
  "scoring_mode": "learner",
  "selection_mode": "sample",
  "legal_action_count": 12,
  "selected_action": {
    "action_id": 119,
    "family": "main_play_character",
    "label": "main_play_character(hand_index=3, stage_slot=2)"
  },
  "raw_topk_no_public_bias": [
    {
      "action_id": 415,
      "family": "main_move",
      "label": "main_move(from_slot=3, to_slot=1)",
      "logit": 9.436505317687988
    }
  ],
  "final_topk": [
    {
      "action_id": 117,
      "family": "main_play_character",
      "label": "main_play_character(hand_index=3, stage_slot=0)",
      "logit": 18.90178871154785
    }
  ],
  "raw_top_action_matches_final": false,
  "raw_top_family_matches_final": false,
  "public_bias_report": {
    "effective_learner": 3.0,
    "effective_actor": 3.0
  }
}
```

### `python/scripts/b1_bias_sweep_matrix.py`

New wrapper script. It repeatedly invokes `b1_artifact_matrix.py` over common bias scales and writes:

```text
bias_sweep_summary.json
```

## Validation

Compile checks passed:

```text
uv run python -m py_compile python/scripts/b1_artifact_matrix.py python/scripts/b1_bias_sweep_matrix.py
```

No Python train/eval processes were left running after diagnostics.

## Bias sweep results

Command:

```text
uv run python python/scripts/b1_bias_sweep_matrix.py \
  --stack-config configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_b1init_devtargetsharp_actorparity3_shift450_vlowlr_frontierweighted_seedpool_lag10_evalgated_lowlr_evalguard_localpromo.yaml \
  --run-dir runs/league_typedpools_seedquarantine_u500_to_u540_20260427 \
  --b1-baseline-run-dir runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425 \
  --checkpoint-policy u480=runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/training/checkpoints/checkpoint_480.pt \
  --checkpoint-policy u540=runs/league_typedpools_seedquarantine_u500_to_u540_20260427/training/checkpoints/checkpoint_540.pt \
  --pairs 8 \
  --artifact-dir-name b1_bias_sweep_missing_mid_p8_20260427 \
  --device cuda:0 \
  --scales 0.5,1.5,2.0,3.0
```

Artifact:

```text
runs/league_typedpools_seedquarantine_u500_to_u540_20260427/eval/b1_bias_sweep_missing_mid_p8_20260427/bias_sweep_summary.json
```

Combined with prior scale `0.0` and `1.0` results:

### Scale 0.0

```text
B1 vs u480: 0.8125
B1 vs u540: 0.9375
u480 vs B1: 0.1875
u540 vs B1: 0.25
```

Interpretation:

```text
raw/no-bias B1 is much stronger than u480/u540
```

### Scale 0.5

```text
B1 vs u480: 1.0, pair 2-0:8
u480 vs B1: 0.0, pair 0-2:8
B1 vs u540: 1.0, pair 2-0:8
u540 vs B1: 0.0, pair 0-2:8
u480 vs u540: 0.6875
u540 vs u480: 0.3125
```

Interpretation:

```text
low common bias still has B1 crushing the league continuations
```

### Scale 1.0

```text
B1 vs u480: 0.5
B1 vs u540: 0.625
u480 vs B1: 0.5
u540 vs B1: 0.4375
u480 vs u540: 0.8125
u540 vs u480: 0.3125
```

Interpretation:

```text
exact invariance breaks, but u480/u540 still do not clearly beat B1
```

### Scale 1.5

```text
B1 vs u480: 0.5
B1 vs u540: 0.4375
u480 vs B1: 0.5
u540 vs B1: 0.5625
u480 vs u540: 0.4375
u540 vs u480: 0.5
```

### Scale 2.0

```text
B1 vs u480: 0.4375
u480 vs B1: 0.4375
B1 vs u540: 0.4375
u540 vs B1: 0.5625
u480 vs u540: 0.4375
u540 vs u480: 0.4375
```

### Scale 3.0

```text
all six B1/u480/u540 ordered matchups:
  mean 0.5
  pair classes 1-1:8
```

Interpretation:

```text
S3 is fully saturated for B1-family matchups
```

## Trace probe results

### S3 trace: official learner scale 3.0

Command:

```text
uv run python python/scripts/b1_artifact_matrix.py \
  --stack-config <b1-init league config> \
  --run-dir runs/league_typedpools_seedquarantine_u500_to_u540_20260427 \
  --b1-baseline-run-dir runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425 \
  --checkpoint-policy u480=<u480 checkpoint> \
  --checkpoint-policy u540=<u540 checkpoint> \
  --pairs 1 \
  --artifact-dir-name b1_trace_probe_s3_p1_20260427 \
  --device cuda:0 \
  --matchup B1=u540 \
  --emit-action-traces \
  --trace-top-k 5 \
  --trace-max-decisions-per-episode 80
```

Artifact:

```text
runs/league_typedpools_seedquarantine_u500_to_u540_20260427/eval/b1_trace_probe_s3_p1_20260427/B1_NoLeague_baseline__vs__u540/action_trace.jsonl
```

Result:

```text
pair result:
  B1 vs u540 = 0.5
  pair class 1-1

trace rows:
  160

raw-vs-final top action:
  match 106
  mismatch 54

raw-vs-final top family:
  match 127
  mismatch 33
```

Per policy:

```text
B1:
  action match rate 0.6709
  family match rate 0.8608

u540:
  action match rate 0.6543
  family match rate 0.7284
```

Examples:

```text
B1 raw main_play_character -> final pass
u540 raw main_move -> final main_play_character
B1 raw main_move -> final main_play_character
```

Interpretation:

```text
S3 public heuristic bias is not a mild tie-breaker.
It frequently changes top action and sometimes changes top action family.
For u540, family-level overrides are especially common.
```

### S1 trace: common learner/actor scale 1.0

Command:

```text
same as above but:
  --public-heuristic-bias-scale 1.0
  --artifact-dir-name b1_trace_probe_s1_p1_20260427
```

Artifact:

```text
runs/league_typedpools_seedquarantine_u500_to_u540_20260427/eval/b1_trace_probe_s1_p1_20260427/B1_NoLeague_baseline__vs__u540/action_trace.jsonl
```

Result:

```text
pair result:
  B1 vs u540 = 1.0
  pair class 2-0:1

trace rows:
  160

raw-vs-final top action:
  match 116
  mismatch 44

raw-vs-final top family:
  match 154
  mismatch 6
```

Per policy:

```text
B1:
  action match rate 0.7531
  family match rate 0.9877

u540:
  action match rate 0.6962
  family match rate 0.9367
```

Interpretation:

```text
S1 still nudges exact action choice, but family-level overrides are much rarer.
S1 is a more useful diagnostic surface than S3.
```

## Destructive forced-action control

Purpose:

```text
Verify that forced legal deviations can alter terminal winners.
```

Command:

```text
uv run python python/scripts/b1_artifact_matrix.py \
  --stack-config <config> \
  --run-dir runs/league_typedpools_seedquarantine_u500_to_u540_20260427 \
  --b1-baseline-run-dir runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425 \
  --checkpoint-policy u480=<u480 checkpoint> \
  --checkpoint-policy u540=<u540 checkpoint> \
  --pairs 1 \
  --artifact-dir-name b1_forced_pass_seat0_s3_p1_20260427 \
  --device cuda:0 \
  --matchup B1=u540 \
  --emit-action-traces \
  --trace-top-k 3 \
  --trace-max-decisions-per-episode 80 \
  --force-pass-seat 0 \
  --force-pass-max-per-episode 10
```

Baseline same seed:

```text
swap 0:
  focal B1 as seat0 wins
  physical winner seat 0

swap 1:
  focal B1 as seat1 loses
  physical winner seat 0
```

Forced-pass result:

```text
swap 0:
  focal B1 as seat0 loses
  physical winner seat 1

swap 1:
  focal B1 as seat1 wins
  physical winner seat 1

forced_pass_decisions: 20
```

Interpretation:

```text
The intervention path works.
B1-family physical-seat outcomes are not immutable deck fate.
Legal deviations can flip terminal winners.
```

## Constructive one-action probes

Target:

```text
matchup B1 vs u540
pair 0
swap 0
physical seat 1 = u540
physical seat 1 loses in baseline
```

We forced one legal action at a time for the losing physical seat.

Candidates from S3 trace:

```text
decision 15 -> action 415 main_move(from_slot=3, to_slot=1)
decision 16 -> action 415
decision 17 -> action 402 main_move(from_slot=0, to_slot=1)
decision 18 -> action 402
decision 44 -> action 417 main_move(from_slot=3, to_slot=4)

decision 12 -> action 57 clock_from_hand(hand_index=5)
decision 13 -> action 128 main_play_character(hand_index=5, stage_slot=1)
decision 14 -> action 117 main_play_character(hand_index=3, stage_slot=0)
decision 14 -> action 103 main_play_character(hand_index=0, stage_slot=1)
decision 15 -> action 117
decision 16 -> action 117
decision 20 -> action 475 attack(slot=1, attack_type=frontal)
decision 21 -> action 475
```

Result:

```text
all 13 probes:
  forced_action_decisions: 1
  forced_action_missed_decisions: 0
  swap 0 remained focal B1 win / physical winner seat 0
```

Interpretation:

```text
Single plausible constructive deviations did not flip the losing seat.
```

## Constructive multi-action sequence probes

We then added `--force-action-sequence` to force multiple actions in one game.

### Sequence 1: raw main-move sequence

```json
[
  {"swap_index": 0, "seat": 1, "decision_index": 15, "action_id": 415},
  {"swap_index": 0, "seat": 1, "decision_index": 16, "action_id": 415},
  {"swap_index": 0, "seat": 1, "decision_index": 17, "action_id": 402},
  {"swap_index": 0, "seat": 1, "decision_index": 18, "action_id": 402},
  {"swap_index": 0, "seat": 1, "decision_index": 44, "action_id": 417}
]
```

Result:

```text
forced 1
missed 4
swap 0 still physical winner seat 0
decision/tick changed from baseline 133/416 to 134/414
```

Because early forced action shifted the trajectory, later exact decision-index actions missed.

### Sequence 2: final top-action sequence

```json
[
  {"swap_index": 0, "seat": 1, "decision_index": 12, "action_id": 57},
  {"swap_index": 0, "seat": 1, "decision_index": 13, "action_id": 128},
  {"swap_index": 0, "seat": 1, "decision_index": 14, "action_id": 117},
  {"swap_index": 0, "seat": 1, "decision_index": 15, "action_id": 117},
  {"swap_index": 0, "seat": 1, "decision_index": 16, "action_id": 117},
  {"swap_index": 0, "seat": 1, "decision_index": 20, "action_id": 475},
  {"swap_index": 0, "seat": 1, "decision_index": 21, "action_id": 475}
]
```

Result:

```text
forced 5
missed 2
swap 0 still physical winner seat 0
decision/tick changed from baseline 133/416 to 125/387
```

### Sequence 3: mixed play/attack

```json
[
  {"swap_index": 0, "seat": 1, "decision_index": 13, "action_id": 128},
  {"swap_index": 0, "seat": 1, "decision_index": 14, "action_id": 103},
  {"swap_index": 0, "seat": 1, "decision_index": 15, "action_id": 117},
  {"swap_index": 0, "seat": 1, "decision_index": 20, "action_id": 475},
  {"swap_index": 0, "seat": 1, "decision_index": 21, "action_id": 475}
]
```

Result:

```text
forced 3
missed 2
swap 0 still physical winner seat 0
decision/tick changed from baseline 133/416 to 125/384
```

Interpretation:

```text
Multi-action constructive probes altered trajectories substantially but did not flip the winner.
Manual exact-index sequence forcing is brittle because earlier forced actions shift later decision opportunities.
```

## Current interpretation

What we know now:

```text
1. S3 official eval is saturated and wrapper-dominated.
2. S1 is less saturated and more useful diagnostically.
3. B1 raw/low-bias body is stronger than u480/u540.
4. Destructive forced actions can flip winners, so the simulator/action path responds.
5. One-action constructive probes did not find an exploitable move.
6. Naive multi-action exact-index constructive sequences changed trajectories but did not flip.
```

The key remaining unknown:

```text
Do constructive B1-beating deviations exist but require search over state-conditioned action sequences,
or is B1 genuinely much stronger than u480/u540 under low-bias/raw surfaces?
```

## What we need from you now

Please advise the next implementation step.

Possible directions:

### Option A: Build a real state-conditioned counterfactual search

Instead of exact decision-index sequences, search at runtime:

```text
at each losing-seat decision:
  enumerate legal actions or family representatives
  force one action
  continue rollout
  evaluate terminal outcome or terminal delta
```

Problems:

```text
we currently only replay from initial seed and force by decision index
we do not clone simulator state
later decision indices shift after interventions
```

Question:

```text
Should we invest in state cloning / replay-to-prefix / forked rollout machinery?
```

### Option B: Train from B1 with common S1 bias parity

Since B1 raw/low-bias is stronger than u480/u540:

```text
restart from B1 checkpoint_450
train with common actor/learner bias scale 1.0
use S3/S1/S0 eval gates
remove exact-action BC
keep small B3/B4 sanity lanes
do not continue u540
```

Question:

```text
Is this now more promising than spending more time on counterfactual search?
```

### Option C: Add an auxiliary loss to preserve/improve B1 raw policy while reducing wrapper reliance

Maybe the league branch degraded raw network strength because the wrapper carried performance.

Potential objective:

```text
distill B1 raw/S0 into current policy
train RL at S1
track S3 for broad strength
slowly anneal bias 3.0 -> 1.0
```

Question:

```text
Should the next branch explicitly protect B1 raw/S0 behavior before trying to improve?
```

### Option D: Give up on u480/u540 as parents

Evidence:

```text
u480/u540 are weaker than B1 at S0/S0.5
u540 is not reliably better than u480 at S1
S3 hides the weakness
```

Question:

```text
Should we discard them as main-policy parents and only use them as diagnostics/history?
```

## Requested answer format

Please give:

1. Ranked diagnosis after the counterfactual results.
2. Whether to keep pushing counterfactual search or switch to B1/S1 retraining.
3. If counterfactual search: exact algorithm to implement next without overbuilding.
4. If retraining: exact config changes and gates.
5. Whether u480/u540 should be discarded as parents.
6. What artifact would convince you to allow a server/L40 pilot.

## Code/artifacts we can provide next if useful

Ask for specific snippets if needed:

```text
python/scripts/b1_artifact_matrix.py
python/scripts/b1_bias_sweep_matrix.py
python/weiss_rl/model.py public heuristic bias functions
python/weiss_rl/eval/simulator_runner.py run_game/select_action
python/weiss_rl/runtime.py actor-side policy scoring/bias restore
python/scripts/train.py promotion gate and B1 import/eval logic
current b1-init league config
trace rows for S3/S1
forced action sequence artifact summaries
```

