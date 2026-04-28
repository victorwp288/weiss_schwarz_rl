# GPT Pro Follow-Up: B1 Artifact Matrix Implemented, Exact Per-Seed Symmetry Found

Assume this is the same GPT Pro session. You have already seen:

1. The original long prompt about the Weiss Schwarz RL thesis league system.
2. Your diagnosis that the old league had topology/bookkeeping flaws and a separate B1 best-response problem.
3. The follow-up prompt showing that typed league pools / seed-history quarantine fixed the topology enough that PFSP sampled true local recents, but u540 still did not improve quality.
4. Your most recent answer recommending artifact triage first, then B1 best-response/counterfactual work.

This prompt reports what Codex implemented and tested from your latest recommendations.

The short version:

```text
We implemented a no-training B1 artifact matrix with policy load manifests and pair tables.
It rules out "same checkpoint accidentally loaded" for B1/u480/u540.
But it found something even more suspicious:
for the first 8 paired seeds, every model-vs-model matchup has exactly the same per-seed winner-seat pattern as B1-vs-B1.
Greedy focal action mode does not change it.
```

We need your next diagnosis and the next concrete implementation target.

## Workspace

```text
C:\Users\Bruger\Desktop\this one\weiss_schwarz_rl
```

Date:

```text
2026-04-27
```

## What We Implemented

### New Diagnostic Script

File:

```text
python/scripts/b1_artifact_matrix.py
```

Purpose:

```text
Run a small policy matrix without training.
Load B1, u480, u540 as independent eval policies.
Emit policy load manifests, state-dict hashes, parameter distances, pair tables, seat diagnostics, and complement checks.
```

It is explicitly diagnostic, not a thesis comparison surface.

### Important Script Behavior

It loads:

```text
B1 NoLeague baseline
u480 checkpoint
u540 checkpoint
```

It writes:

```text
policy_load_manifest.json
resolved_policies.json
matrix_summary.json
<focal>__vs__<opponent>/episodes.jsonl
<focal>__vs__<opponent>/matchup_summary.json
<focal>__vs__<opponent>/seat_diagnostics.json
<focal>__vs__<opponent>/pair_class_summary.json
<focal>__vs__<opponent>/pair_table.jsonl
```

It computes:

```text
source file SHA256
model object id
loaded model state_dict SHA256
state_dict parameter L2 norm
pairwise parameter L2 distances
public heuristic logit bias scales
ordered matchup mean/win/loss/draw/truncation
pair classes: 2-0 / 1-1 / 0-2 / mixed
swapped-label complement checks
```

It supports:

```text
--focal-action-mode sample
--focal-action-mode greedy
```

Greedy mode makes the focal policy choose argmax over legal logits while the opponent remains normal sampled eval.

### Core Code Shape

The script builds the matrix using existing canonical eval primitives:

```python
from weiss_rl.eval.harness import run_seat_swapped_matchup
from weiss_rl.eval.simulator_runner import SimulatorEvalRunner, resolve_eval_policies
```

It loads B1 through the existing resolver:

```python
b1_resolved = resolve_eval_policies(
    stack=stack,
    policy_ids=[NO_LEAGUE_POLICY_ID],
    run_dir=args.run_dir,
    observation_dim=observation_dim,
    action_dim=action_dim,
    spec_bundle=spec_bundle,
    b1_baseline_run_dir=args.b1_baseline_run_dir,
    eval_device=args.device,
)[NO_LEAGUE_POLICY_ID]
```

It loads u480/u540 directly from checkpoints:

```python
model = train_script._load_checkpoint_eval_model(
    checkpoint_path=checkpoint_path,
    observation_dim=observation_dim,
    action_dim=action_dim,
    stack=stack,
    eval_device=args.device,
    observation_spec=observation_spec,
    spec_bundle=spec_bundle,
)
policies[alias] = ResolvedEvalPolicy(
    policy_id=alias,
    kind="checkpoint",
    source_run_dir=checkpoint_path.parent.parent.parent.as_posix(),
    snapshot_path=checkpoint_path.as_posix(),
    model=model,
)
```

It runs every ordered pair:

```python
result = run_seat_swapped_matchup(
    focal_policy_id=focal_policy_id,
    opponent_policy_id=opponent_policy_id,
    paired_seeds=paired_seeds,
    runner=runner,
    episodes_path=matchup_dir / "episodes.jsonl",
    run_id256=manifest_run_id,
    config_hash256=manifest_config_hash,
    spec_hash256=spec_hash,
)
```

Pair class construction:

```python
if outcomes == ("W", "W"):
    pair_class = "2-0"
elif outcomes == ("L", "L"):
    pair_class = "0-2"
elif set(outcomes) == {"W", "L"}:
    pair_class = "1-1"
else:
    pair_class = "mixed"
```

Greedy focal override:

```python
class _GreedyFocalSimulatorEvalRunner(SimulatorEvalRunner):
    def _select_action(self, **kwargs):
        current_policy_id = str(kwargs.get("current_policy_id"))
        if current_policy_id != self._greedy_policy_id:
            return super()._select_action(**kwargs)
        policy = self.policies.get(current_policy_id)
        if policy is None or policy.model is None:
            return super()._select_action(**kwargs)
        logits_tensor, _value_tensor, next_seat_hidden = policy.model.forward_seat_aware(...)
        legal_logits = logits[legal_ids.astype(np.int64, copy=False)]
        return int(legal_ids[int(np.argmax(legal_logits))]), next_seat_hidden
```

## Extra Low-Risk Registry Fix

You flagged that `latest_active_champion_ids(...)` should exclude rejected snapshots too.

We patched:

```text
python/weiss_rl/league/registry.py
```

Now `latest_active_champion_ids(...)` excludes `rejected_snapshots`, and `normalize()` strips rejected IDs from active champion state.

Test updated:

```text
python/weiss_rl/tests/test_snapshot_registry.py::test_snapshot_registry_typed_selectors_keep_seed_history_out_of_active_champions
```

## Validation

Compile:

```text
uv run python -m py_compile python/weiss_rl/league/registry.py python/scripts/b1_artifact_matrix.py
```

Focused test:

```text
uv run pytest -q python/weiss_rl/tests/test_snapshot_registry.py::test_snapshot_registry_typed_selectors_keep_seed_history_out_of_active_champions --tb=short
```

Result:

```text
1 passed
```

Broader focused regression:

```text
uv run pytest -q \
  python/weiss_rl/tests/test_snapshot_registry.py::test_import_seed_snapshot_pool_imports_external_snapshots_as_seed_history_not_champions \
  python/weiss_rl/tests/test_snapshot_registry.py::test_import_resume_league_snapshot_pool_preserves_local_recents_and_champions \
  python/weiss_rl/tests/test_snapshot_registry.py::test_snapshot_registry_typed_selectors_keep_seed_history_out_of_active_champions \
  python/weiss_rl/tests/test_runtime.py::test_refresh_opponent_pool_never_treats_seed_history_as_active_champion_or_recent \
  python/weiss_rl/tests/test_runtime.py::test_refresh_opponent_pool_keeps_champions_out_of_recent_lane \
  --tb=short
```

Result:

```text
5 passed
```

## Artifact Matrix Runs

### Tiny Smoke

Command:

```text
uv run python python/scripts/b1_artifact_matrix.py \
  --stack-config configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_b1init_devtargetsharp_actorparity3_shift450_vlowlr_frontierweighted_seedpool_lag10_evalgated_lowlr_evalguard_localpromo.yaml \
  --run-dir runs/league_typedpools_seedquarantine_u500_to_u540_20260427 \
  --b1-baseline-run-dir runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425 \
  --checkpoint-policy u480=runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/training/checkpoints/checkpoint_480.pt \
  --checkpoint-policy u540=runs/league_typedpools_seedquarantine_u500_to_u540_20260427/training/checkpoints/checkpoint_540.pt \
  --pairs 1 \
  --artifact-dir-name b1_artifact_matrix_smoke_p1_20260427 \
  --device cuda:0 \
  --include-self
```

Result:

```text
All 9 ordered/self matchups split 1-1 on the single paired seed.
No load/runtime errors.
```

Artifact dir:

```text
runs/league_typedpools_seedquarantine_u500_to_u540_20260427/eval/b1_artifact_matrix_smoke_p1_20260427
```

### Main Sampled Matrix

Command:

```text
uv run python python/scripts/b1_artifact_matrix.py \
  --stack-config configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_b1init_devtargetsharp_actorparity3_shift450_vlowlr_frontierweighted_seedpool_lag10_evalgated_lowlr_evalguard_localpromo.yaml \
  --run-dir runs/league_typedpools_seedquarantine_u500_to_u540_20260427 \
  --b1-baseline-run-dir runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425 \
  --checkpoint-policy u480=runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/training/checkpoints/checkpoint_480.pt \
  --checkpoint-policy u540=runs/league_typedpools_seedquarantine_u500_to_u540_20260427/training/checkpoints/checkpoint_540.pt \
  --pairs 8 \
  --artifact-dir-name b1_artifact_matrix_p8_20260427 \
  --device cuda:0 \
  --include-self
```

Wall time:

```text
~472 seconds local Windows
```

Artifact dir:

```text
runs/league_typedpools_seedquarantine_u500_to_u540_20260427/eval/b1_artifact_matrix_p8_20260427
```

Result:

```text
B1 NoLeague baseline vs B1 NoLeague baseline: mean=0.5 wins=8 losses=8 pair_classes={'2-0': 0, '1-1': 8, '0-2': 0, 'mixed': 0}
B1 NoLeague baseline vs u480:                 mean=0.5 wins=8 losses=8 pair_classes={'2-0': 0, '1-1': 8, '0-2': 0, 'mixed': 0}
B1 NoLeague baseline vs u540:                 mean=0.5 wins=8 losses=8 pair_classes={'2-0': 0, '1-1': 8, '0-2': 0, 'mixed': 0}
u480 vs B1 NoLeague baseline:                 mean=0.5 wins=8 losses=8 pair_classes={'2-0': 0, '1-1': 8, '0-2': 0, 'mixed': 0}
u480 vs u480:                                  mean=0.5 wins=8 losses=8 pair_classes={'2-0': 0, '1-1': 8, '0-2': 0, 'mixed': 0}
u480 vs u540:                                  mean=0.5 wins=8 losses=8 pair_classes={'2-0': 0, '1-1': 8, '0-2': 0, 'mixed': 0}
u540 vs B1 NoLeague baseline:                 mean=0.5 wins=8 losses=8 pair_classes={'2-0': 0, '1-1': 8, '0-2': 0, 'mixed': 0}
u540 vs u480:                                  mean=0.5 wins=8 losses=8 pair_classes={'2-0': 0, '1-1': 8, '0-2': 0, 'mixed': 0}
u540 vs u540:                                  mean=0.5 wins=8 losses=8 pair_classes={'2-0': 0, '1-1': 8, '0-2': 0, 'mixed': 0}
```

Complement checks:

```text
B1 vs u480 and u480 vs B1: forward_mean=0.5 reverse_mean=0.5 sum=1.0
B1 vs u540 and u540 vs B1: forward_mean=0.5 reverse_mean=0.5 sum=1.0
u480 vs u540 and u540 vs u480: forward_mean=0.5 reverse_mean=0.5 sum=1.0
```

### Greedy-Focal Matrix

Command:

```text
uv run python python/scripts/b1_artifact_matrix.py \
  --stack-config configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_b1init_devtargetsharp_actorparity3_shift450_vlowlr_frontierweighted_seedpool_lag10_evalgated_lowlr_evalguard_localpromo.yaml \
  --run-dir runs/league_typedpools_seedquarantine_u500_to_u540_20260427 \
  --b1-baseline-run-dir runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425 \
  --checkpoint-policy u480=runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/training/checkpoints/checkpoint_480.pt \
  --checkpoint-policy u540=runs/league_typedpools_seedquarantine_u500_to_u540_20260427/training/checkpoints/checkpoint_540.pt \
  --pairs 8 \
  --artifact-dir-name b1_artifact_matrix_greedyfocal_p8_20260427 \
  --device cuda:0 \
  --focal-action-mode greedy
```

Wall time:

```text
~315 seconds local Windows
```

Artifact dir:

```text
runs/league_typedpools_seedquarantine_u500_to_u540_20260427/eval/b1_artifact_matrix_greedyfocal_p8_20260427
```

Result:

```text
B1 NoLeague baseline vs u480: mean=0.5 wins=8 losses=8 pair_classes={'2-0': 0, '1-1': 8, '0-2': 0, 'mixed': 0}
B1 NoLeague baseline vs u540: mean=0.5 wins=8 losses=8 pair_classes={'2-0': 0, '1-1': 8, '0-2': 0, 'mixed': 0}
u480 vs B1 NoLeague baseline: mean=0.5 wins=8 losses=8 pair_classes={'2-0': 0, '1-1': 8, '0-2': 0, 'mixed': 0}
u480 vs u540:                 mean=0.5 wins=8 losses=8 pair_classes={'2-0': 0, '1-1': 8, '0-2': 0, 'mixed': 0}
u540 vs B1 NoLeague baseline: mean=0.5 wins=8 losses=8 pair_classes={'2-0': 0, '1-1': 8, '0-2': 0, 'mixed': 0}
u540 vs u480:                 mean=0.5 wins=8 losses=8 pair_classes={'2-0': 0, '1-1': 8, '0-2': 0, 'mixed': 0}
```

Greedy focal did not change the paired result.

## Load Manifest Findings

From:

```text
runs/league_typedpools_seedquarantine_u500_to_u540_20260427/eval/b1_artifact_matrix_p8_20260427/policy_load_manifest.json
```

### B1

```text
policy_id: B1 NoLeague baseline
source_path:
  runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425/training/snapshots/b1_noleague_baseline/weights.pt
source_file_sha256:
  8861bd04db5882ff2878a775c1487438df556cb914ff996edf81a086eb99c310
state_dict_sha256:
  620bf243792eb2c25d40539c1f035e7e1efafe4d35a3e7e40c576d1ccf514759
state_dict_param_l2_norm:
  1452.0640368167014
public_heuristic_logit_bias_scale_learner:
  3.0
public_heuristic_logit_bias_scale_actor:
  1.0
```

### u480

```text
policy_id: u480
source_path:
  runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/training/checkpoints/checkpoint_480.pt
source_file_sha256:
  1d23190a9c201bb35aee04745e656e2bbef48ca298e21e702036c618f1b372e4
state_dict_sha256:
  aa67db4a97a028b65cb532d3a5415b131e4586c7ee0dd81ccd7cf12e06abea46
state_dict_param_l2_norm:
  1452.0640157170303
public_heuristic_logit_bias_scale_learner:
  3.0
public_heuristic_logit_bias_scale_actor:
  3.0
```

### u540

```text
policy_id: u540
source_path:
  runs/league_typedpools_seedquarantine_u500_to_u540_20260427/training/checkpoints/checkpoint_540.pt
source_file_sha256:
  505372cc61b109b02a66ed941ca41fe9a6db24fa2ea0da412c14130a045c43dc
state_dict_sha256:
  6b48fd712a03a7a6d9ba09c9642fdc55bb1fb34e562a09df6279227a6d75a1b3
state_dict_param_l2_norm:
  1452.0640236808495
public_heuristic_logit_bias_scale_learner:
  3.0
public_heuristic_logit_bias_scale_actor:
  3.0
```

### Pairwise Parameter Distances

```text
B1 vs u480:
  compared_float_params: 2,954,524
  l2_distance: 0.3942730264801399

B1 vs u540:
  compared_float_params: 2,954,524
  l2_distance: 0.46297084975380637

u480 vs u540:
  compared_float_params: 2,954,524
  l2_distance: 0.11901920451535859
```

Interpretation:

```text
The policies are not literally the same checkpoint or same state_dict.
The stale-identical-file hypothesis is mostly ruled out.
However, parameter distances are tiny relative to total norm ~1452.
The policies may be functionally near-identical.
```

One additional oddity:

```text
B1 actor public heuristic bias scale is 1.0, while u480/u540 actor scale is 3.0.
Learner scale is 3.0 for all three.
Eval uses scoring_mode="learner", so learner scale may dominate eval behavior.
```

Please assess whether this actor/learner guidance mismatch matters for training/eval interpretation.

## Most Suspicious New Finding

For the first 8 paired seeds, the per-seed pattern is identical across B1-vs-B1, u480-vs-B1, u540-vs-B1, u480-vs-u540, and swapped labels.

We ran a small comparison over the generated `pair_table.jsonl` files.

For these matchups:

```text
B1_NoLeague_baseline__vs__B1_NoLeague_baseline
u480__vs__B1_NoLeague_baseline
u540__vs__B1_NoLeague_baseline
u480__vs__u540
u540__vs__u480
```

The pattern for each pair was:

```text
pair 0: focal_as_seat0 W, focal_as_seat1 L, winner seats 0/0
pair 1: focal_as_seat0 L, focal_as_seat1 W, winner seats 1/1
pair 2: focal_as_seat0 W, focal_as_seat1 L, winner seats 0/0
pair 3: focal_as_seat0 W, focal_as_seat1 L, winner seats 0/0
pair 4: focal_as_seat0 W, focal_as_seat1 L, winner seats 0/0
pair 5: focal_as_seat0 L, focal_as_seat1 W, winner seats 1/1
pair 6: focal_as_seat0 L, focal_as_seat1 W, winner seats 1/1
pair 7: focal_as_seat0 W, focal_as_seat1 L, winner seats 0/0
```

Equivalently:

```text
For each paired seed, the same physical seat wins both swaps, regardless of whether the policies are B1, u480, or u540.
```

Example from `u540__vs__B1_NoLeague_baseline/pair_table.jsonl`:

```json
{
  "pair_index": 0,
  "episode_seed": 338610598310627562,
  "pair_class": "1-1",
  "focal_as_seat0": {
    "outcome": "W",
    "winner_seat": 0,
    "seat0_policy_id": "u540",
    "seat1_policy_id": "B1 NoLeague baseline",
    "decision_count": 134
  },
  "focal_as_seat1": {
    "outcome": "L",
    "winner_seat": 0,
    "seat0_policy_id": "B1 NoLeague baseline",
    "seat1_policy_id": "u540",
    "decision_count": 134
  }
}
```

Another seed:

```json
{
  "pair_index": 1,
  "episode_seed": 703675363409817853,
  "pair_class": "1-1",
  "focal_as_seat0": {
    "outcome": "L",
    "winner_seat": 1,
    "seat0_policy_id": "u540",
    "seat1_policy_id": "B1 NoLeague baseline"
  },
  "focal_as_seat1": {
    "outcome": "W",
    "winner_seat": 1,
    "seat0_policy_id": "B1 NoLeague baseline",
    "seat1_policy_id": "u540"
  }
}
```

This same physical-seat-wins pattern also appears in:

```text
B1 vs B1
u480 vs B1
u540 vs B1
u480 vs u540
u540 vs u480
```

Greedy focal matrix produced the same outcome pattern for the tested non-self ordered matchups.

## Current Interpretation

We now think:

```text
1. "Same file / same weights accidentally loaded" is unlikely.
2. "B1 alias resolves to current checkpoint" is unlikely.
3. "Swapped-label complement broken" is unlikely for this matrix, because forward/reverse means sum to 1.
4. "Model policies are functionally near-identical on this eval surface" is likely.
5. "For these model-like policies, paired-seed outcome is dominated by physical seat/seed, not by policy identity" is also likely.
6. Therefore B1 0.50 is not merely a scalar-noise problem. It is a surface/protocol/signal problem.
```

The hardest new question:

```text
If B1, u480, and u540 all produce the exact same physical-seat winner pattern on model-vs-model eval,
what is the right next experiment to discover whether any B1-beating policy exists?
```

## What This Does And Does Not Prove

It proves:

```text
B1, u480, and u540 were loaded from distinct files/state_dicts.
The paired matrix is internally consistent.
The exact 0.50 comes from every pair splitting 1-1, not from aggregate rounding.
Greedy focal action mode did not reveal a hidden winning deterministic policy mode.
The first 8 paired seeds have policy-invariant physical-seat winners among these model-like policies.
```

It does not yet prove:

```text
All possible policies are irrelevant.
B1 is a true local equilibrium.
The simulator is wrong.
The policy cannot improve against B1 with a different objective/search method.
The same invariance holds against heuristic B3/B4 or random policies.
The same invariance holds on all seed sets or unpaired/non-seat-swapped protocols.
```

We know from scalar eval that policies can differ against B3/B4/B0, so the entire evaluator is not obviously dead. But model-vs-model appears suspiciously invariant.

## Questions For You Now

Please answer as a continuation, not from scratch.

### 1. Diagnosis Of The Matrix

How should we interpret this?

```text
Distinct model weights, but identical per-seed physical-seat winner patterns for B1/u480/u540.
```

Most likely explanations:

```text
A. B1/u480/u540 are functionally identical because public heuristic learner bias dominates eval.
B. Their parameter distances are too small to matter.
C. The game/eval protocol on these seeds is seat/seed dominated for near-clone policies.
D. The action sampler RNG depends on seat_policy_id in a way that preserves physical-seat outcomes for near-clones.
E. The model is only changing within action families that do not affect game result.
F. The eval state/action trace is deterministic enough that all B1-family models induce the same game.
G. There is still a hidden eval bug not caught by load hashes.
```

Please rank these and say what evidence would separate them.

### 2. Next Fastest Artifact Tests

Should we next implement/run:

```text
1. action trace digest comparison for B1/u480/u540 on identical games;
2. logit/top-k/action-family probe on shared states;
3. no-public-heuristic-bias eval override;
4. unpaired / shuffled-seat eval;
5. include B3/B4/random in the artifact matrix as contrast;
6. run the same matrix on a different seed schedule;
7. run a matrix with older B1 snapshots u300/u400/u450;
8. run exact same matrix with focal/opponent both greedy, not just focal greedy;
9. inspect simulator terminal summaries to see why same physical seat wins.
```

Please pick the best next 2-3, not everything.

### 3. Does This Change The B1 Best-Response Plan?

You recommended counterfactual rollout from B1-seat/loss states.

Given this new evidence, should we still do that immediately?

Or should we first prove that forced action deviations can actually change the terminal winner in these suspicious paired seeds?

For example:

```text
Take pair 0 where physical seat0 wins regardless of B1/u480/u540.
At selected decision states, force legal alternatives for seat1.
Check whether any forced action flips winner from seat0 to seat1.
```

Is that effectively the counterfactual pilot, or should it be even simpler?

### 4. Is Public Heuristic Bias A Prime Suspect?

Eval uses:

```text
model.forward_seat_aware(..., scoring_mode="learner")
```

Manifest says:

```text
B1 learner bias scale: 3.0
u480 learner bias scale: 3.0
u540 learner bias scale: 3.0
B1 actor bias scale: 1.0
u480/u540 actor bias scale: 3.0
```

Questions:

```text
Could learner-side public heuristic bias be collapsing all policies to the same effective policy at eval?
Should we add an eval override that sets public_heuristic_logit_bias_scale=0 for all loaded policies?
Would that be a diagnostic only, or should thesis eval include the bias if it was part of the policy?
If no-bias eval changes the matrix, what does that imply?
If no-bias eval still gives identical physical-seat winners, what does that imply?
```

### 5. What Should Codex Implement Next?

Please give exact next coding tasks.

Candidate tasks:

```text
A. Extend b1_artifact_matrix.py with:
   --disable-public-heuristic-bias
   --both-greedy
   --include-policy-id B3/B4/random
   --seed-scope or --seed-offset for independent seed schedules

B. Add action trace digest:
   record action ids, action families, legal fingerprints, policy id, actor seat, decision id;
   compare traces across B1/u480/u540 for the same seed/swap.

C. Add shared-state logit probe:
   collect states from one B1-vs-B1 run;
   evaluate B1/u480/u540 logits on same batch;
   report KL, top-action match, top-family match, public heuristic agreement.

D. Build minimal forced-action counterfactual script:
   pick a few states in losing seat;
   enumerate legal alternatives;
   force first action only;
   roll out rest normally;
   see if terminal winner can flip.
```

Please rank these by expected information gain.

### 6. What Would Convince You To Discard More League Training?

We do not want to burn compute on clone self-play.

Given this artifact matrix, what evidence would be enough to say:

```text
Do not run more PFSP/league training until action trace/logit/counterfactual diagnostics are fixed.
```

Conversely, what evidence would justify a short B1-exploiter training run?

### 7. What Would You Like To See Next?

At the end of your answer, please list exactly what extra context would help you most in the next follow-up, for example:

```text
specific code blocks from SimulatorEvalRunner;
the PolicyValueModel public heuristic bias implementation;
action-family decoding code;
b1_disagreement_audit.py internals;
sample action trace rows;
no-bias matrix results;
both-greedy matrix results;
forced-action counterfactual results;
configs controlling public heuristic bias;
terminal summaries for invariant physical-seat winners;
```

Please be specific so Codex can produce the most useful next packet.
