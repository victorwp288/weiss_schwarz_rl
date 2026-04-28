# GPT Pro Follow-Up: B1 0.50 Is Now a Public-Bias / Effective-Policy Problem

This is a follow-up in the same debugging session as the previous prompt and your last response.

Assume you already saw:

- The long context prompt about our Weiss Schwarz RL thesis league system.
- Your diagnosis that the old league topology was partly broken because "recent" was resolving to seed/B1 history and champions were empty.
- Our topology repair evidence: true `league_import` and `local` policies can now enter the recent lane, seed history is quarantined from PFSP recent/champion, and a u540 local candidate/champion artifact was created.
- Our first artifact matrix: B1, u480, and u540 had distinct weight hashes but every B1-family model-vs-model matchup produced exact paired `0.50` with pair classes `1-1: 8`.
- Your latest recommendation to extend `b1_artifact_matrix.py` with built-in contrast policies, no-bias and scoring-mode overrides, both-greedy, action RNG salt controls, and hard counters proving these toggles actually fire.

This prompt adds the implementation details and new results from that requested follow-up.

## Core question now

We need your help deciding the next structural fix.

The old bookkeeping bug is now probably not the main blocker. The key problem is:

> Official learner-mode eval with public heuristic logit bias scale `3.0` makes B1, u480, and u540 land at exact paired `0.50`, but changing the effective bias/scoring wrapper reveals large differences.

This suggests that our "B1 is always 0.50" issue may not mean "the learned policy equals B1" in all modes. It may mean:

1. official eval is dominated by a common hand-coded public heuristic bias wrapper;
2. B1/u480/u540 learned networks are weak or tiny-different, and the wrapper collapses them into nearly the same effective policy;
3. actor/learner scoring mismatch is huge because B1 actor scale is `1.0` while u480/u540 actor scales are `3.0`;
4. the current thesis evaluation surface may be rewarding the wrapper more than learning;
5. league/self-play at official bias may be clone self-play inside the wrapper's basin.

Please help us reason about what to implement next.

## Current key artifacts

Base B1 anchor:

```text
runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425/training/checkpoints/checkpoint_450.pt
```

Best prior local league checkpoint:

```text
runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/training/checkpoints/checkpoint_480.pt
```

Topology-fixed continuation candidate:

```text
runs/league_typedpools_seedquarantine_u500_to_u540_20260427/training/checkpoints/checkpoint_540.pt
```

Matrix artifacts from this packet:

```text
runs/league_typedpools_seedquarantine_u500_to_u540_20260427/eval/b1_artifact_matrix_controls_smoke_p1_20260427
runs/league_typedpools_seedquarantine_u500_to_u540_20260427/eval/b1_artifact_matrix_builtincontrast_p8_20260427
runs/league_typedpools_seedquarantine_u500_to_u540_20260427/eval/b1_artifact_matrix_nobias_p8_20260427
runs/league_typedpools_seedquarantine_u500_to_u540_20260427/eval/b1_artifact_matrix_bothgreedy_p8_20260427
runs/league_typedpools_seedquarantine_u500_to_u540_20260427/eval/b1_artifact_matrix_actor_p8_20260427
runs/league_typedpools_seedquarantine_u500_to_u540_20260427/eval/b1_artifact_matrix_bias1_p8_20260427
```

## Code changes made for this packet

We extended:

```text
python/scripts/b1_artifact_matrix.py
```

It now supports:

```text
--include-builtin B0_RandomLegal
--include-builtin B2_HeuristicPublic
--include-builtin B3_HeuristicAggro
--include-builtin B4_HeuristicControl
--matchup focal=opponent
--disable-public-heuristic-bias
--public-heuristic-bias-scale <float>
--scoring-mode learner|actor
--both-greedy
--seed-scope <string>
--seed-offset <int>
--action-rng-salt-mode shared|policy|matchup
```

It writes:

```text
policy_load_manifest.json
resolved_policies.json
matrix_summary.json
<matchup>/episodes.jsonl
<matchup>/matchup_summary.json
<matchup>/seat_diagnostics.json
<matchup>/pair_class_summary.json
<matchup>/pair_table.jsonl
```

The manifest now records:

```json
{
  "scoring_mode": "learner",
  "both_greedy": false,
  "action_rng_salt_mode": "shared",
  "public_heuristic_bias_override_requested": true,
  "public_heuristic_bias_override_scale": 0.0,
  "public_heuristic_bias_override_report": {
    "B1 NoLeague baseline": {
      "before_learner": 3.0,
      "before_actor": 1.0,
      "effective_learner": 0.0,
      "effective_actor": 0.0
    },
    "u480": {
      "before_learner": 3.0,
      "before_actor": 3.0,
      "effective_learner": 0.0,
      "effective_actor": 0.0
    },
    "u540": {
      "before_learner": 3.0,
      "before_actor": 3.0,
      "effective_learner": 0.0,
      "effective_actor": 0.0
    }
  }
}
```

Each matchup summary now includes hard counters:

```json
{
  "matrix_runner_counters": {
    "model_decisions": 2128,
    "heuristic_decisions": 0,
    "random_legal_decisions": 0,
    "sample_decisions": 0,
    "greedy_override_decisions": 2128,
    "fallback_to_parent_decisions": 0,
    "scoring_mode": "learner",
    "greedy_policy_ids": [
      "B1 NoLeague baseline",
      "u540"
    ],
    "greedy_override_requested": true,
    "action_rng_salt_mode": "shared"
  }
}
```

The script hard-fails if:

```text
greedy is requested for a model policy but greedy_override_decisions == 0
bias override is requested but effective learner/actor scales do not match the requested scale
a requested builtin policy alias cannot resolve
```

## Relevant code excerpt: bias override

```python
def _apply_public_heuristic_bias_override(
    policies: Mapping[str, ResolvedEvalPolicy],
    *,
    override_scale: float | None,
) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    if override_scale is None:
        for policy_id, policy in policies.items():
            report[policy_id] = {
                "requested": False,
                "effective_learner": _read_bias_scale(policy, scoring_mode="learner"),
                "effective_actor": _read_bias_scale(policy, scoring_mode="actor"),
            }
        return report
    expected = float(override_scale)
    for policy_id, policy in policies.items():
        before = {
            "learner": _read_bias_scale(policy, scoring_mode="learner"),
            "actor": _read_bias_scale(policy, scoring_mode="actor"),
        }
        if policy.model is not None:
            set_bias_scale = getattr(policy.model, "set_public_heuristic_logit_bias_scale", None)
            if not callable(set_bias_scale):
                raise RuntimeError(f"bias override requested, but policy {policy_id!r} has no bias setter")
            set_bias_scale(expected, actor_value=expected)
        after = {
            "learner": _read_bias_scale(policy, scoring_mode="learner"),
            "actor": _read_bias_scale(policy, scoring_mode="actor"),
        }
        if policy.model is not None:
            for mode, value in after.items():
                if value is None or abs(float(value) - expected) > 1e-6:
                    raise RuntimeError(
                        f"bias override requested, but policy {policy_id!r} effective {mode} scale is {value}, "
                        f"expected {expected}"
                    )
        report[policy_id] = {
            "requested": True,
            "requested_scale": expected,
            "before_learner": before["learner"],
            "before_actor": before["actor"],
            "effective_learner": after["learner"],
            "effective_actor": after["actor"],
        }
    return report
```

## Relevant code excerpt: scoring mode and greedy override

```python
class _MatrixSimulatorEvalRunner(SimulatorEvalRunner):
    def __init__(
        self,
        *args: Any,
        scoring_mode: str,
        greedy_policy_ids: Sequence[str] = (),
        action_rng_salt_mode: str = "shared",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._matrix_scoring_mode = str(scoring_mode)
        self._greedy_policy_ids = frozenset(str(policy_id) for policy_id in greedy_policy_ids)
        self._action_rng_salt_mode = str(action_rng_salt_mode)
        self._matrix_counters = {
            "model_decisions": 0,
            "heuristic_decisions": 0,
            "random_legal_decisions": 0,
            "sample_decisions": 0,
            "greedy_override_decisions": 0,
            "fallback_to_parent_decisions": 0,
        }

    def _select_action(self, **kwargs: Any) -> tuple[int, torch.Tensor | None]:
        current_policy_id = str(kwargs.get("current_policy_id"))
        policy = self.policies.get(current_policy_id)
        if policy is None:
            raise RuntimeError(f"Missing resolved eval policy for {current_policy_id!r}")
        if policy.heuristic_policy is not None:
            self._matrix_counters["heuristic_decisions"] += 1
            action = policy.heuristic_policy.choose_action(
                np.asarray(kwargs["batch"].obs[0], dtype=np.float32),
                np.asarray(kwargs["legal_ids"], dtype=np.uint32),
            )
            return int(action), kwargs.get("seat_hidden")
        if policy.model is None:
            self._matrix_counters["random_legal_decisions"] += 1
            self._matrix_counters["sample_decisions"] += 1
            action, _logp = sample_action_pinned(
                self._baseline_logits,
                np.asarray(kwargs["legal_ids"], dtype=np.uint32),
                rng=kwargs["rng"],
            )
            return int(action), kwargs.get("seat_hidden")
        seat_hidden = kwargs.get("seat_hidden")
        if seat_hidden is None:
            self._matrix_counters["fallback_to_parent_decisions"] += 1
            return super()._select_action(**kwargs)
        batch = kwargs["batch"]
        current_seat = int(kwargs["current_seat"])
        legal_ids = np.asarray(kwargs["legal_ids"], dtype=np.uint32)
        self._matrix_counters["model_decisions"] += 1
        with torch.inference_mode():
            logits_tensor, _value_tensor, next_seat_hidden = policy.model.forward_seat_aware(
                torch.as_tensor(np.asarray(batch.obs, dtype=np.float32), device=self._device),
                torch.as_tensor([current_seat], device=self._device, dtype=torch.long),
                seat_hidden,
                scoring_mode=self._matrix_scoring_mode,
            )
        logits = logits_tensor[0].detach().cpu().numpy().astype(np.float32, copy=False)
        if current_policy_id in self._greedy_policy_ids:
            self._matrix_counters["greedy_override_decisions"] += 1
            legal_logits = logits[legal_ids.astype(np.int64, copy=False)]
            return int(legal_ids[int(np.argmax(legal_logits))]), next_seat_hidden
        self._matrix_counters["sample_decisions"] += 1
        action, _logp = sample_action_pinned(logits, legal_ids, rng=kwargs["rng"])
        return int(action), next_seat_hidden
```

## Validation

Compiled:

```text
uv run python -m py_compile python/scripts/b1_artifact_matrix.py
```

Control smoke:

```text
uv run python python/scripts/b1_artifact_matrix.py \
  --stack-config configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_b1init_devtargetsharp_actorparity3_shift450_vlowlr_frontierweighted_seedpool_lag10_evalgated_lowlr_evalguard_localpromo.yaml \
  --run-dir runs/league_typedpools_seedquarantine_u500_to_u540_20260427 \
  --b1-baseline-run-dir runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425 \
  --checkpoint-policy u480=runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/training/checkpoints/checkpoint_480.pt \
  --checkpoint-policy u540=runs/league_typedpools_seedquarantine_u500_to_u540_20260427/training/checkpoints/checkpoint_540.pt \
  --pairs 1 \
  --artifact-dir-name b1_artifact_matrix_controls_smoke_p1_20260427 \
  --device cuda:0 \
  --disable-public-heuristic-bias \
  --both-greedy \
  --matchup u540=B1
```

Smoke result:

```text
u540 vs B1 NoLeague baseline:
  mean 0.5
  wins 1
  losses 1
  pair_classes 1-1: 1
  greedy_override_decisions 342
  sample_decisions 0
  model_decisions 342
  effective learner bias 0.0 for B1/u480/u540
  effective actor bias 0.0 for B1/u480/u540
```

So the controls are wired.

## Diagnostic run A: official learner mode with contrast policies

Command shape:

```text
--include-builtin B0_RandomLegal
--include-builtin B2_HeuristicPublic
--include-builtin B3_HeuristicAggro
--include-builtin B4_HeuristicControl
--matchup B1=B0
--matchup B1=B2
--matchup B1=B3
--matchup B1=B4
--matchup u540=B0
--matchup u540=B2
--matchup u540=B3
--matchup u540=B4
--pairs 8
--scoring-mode learner
```

Artifact:

```text
runs/league_typedpools_seedquarantine_u500_to_u540_20260427/eval/b1_artifact_matrix_builtincontrast_p8_20260427
```

Results:

```text
B1 vs B0 RandomLegal:
  mean 1.0, wins 16, losses 0, pair classes 2-0: 8

B1 vs B2 HeuristicPublic:
  mean 1.0, wins 16, losses 0, pair classes 2-0: 8

B1 vs B3 HeuristicPublicAggro:
  mean 0.8125, wins 13, losses 3, pair classes 2-0: 5, 1-1: 3

B1 vs B4 HeuristicPublicControl:
  mean 1.0, wins 16, losses 0, pair classes 2-0: 8

u540 vs B0 RandomLegal:
  mean 1.0, wins 16, losses 0, pair classes 2-0: 8

u540 vs B2 HeuristicPublic:
  mean 1.0, wins 16, losses 0, pair classes 2-0: 8

u540 vs B3 HeuristicPublicAggro:
  mean 1.0, wins 16, losses 0, pair classes 2-0: 8

u540 vs B4 HeuristicPublicControl:
  mean 1.0, wins 16, losses 0, pair classes 2-0: 8
```

Interpretation:

- The eval/simulator/action path is policy-sensitive in general.
- The model-vs-model exact `0.50` is not because every policy is ignored.
- u540 looks better than B1 against B3 on this small contrast surface.
- But this does not solve B1 because B1-family model-vs-model remains locked at official bias.

## Diagnostic run B: no public heuristic bias, common scale 0.0

Command shape:

```text
--disable-public-heuristic-bias
--pairs 8
--scoring-mode learner
```

Artifact:

```text
runs/league_typedpools_seedquarantine_u500_to_u540_20260427/eval/b1_artifact_matrix_nobias_p8_20260427
```

Manifest proved effective bias:

```text
B1 learner before 3.0, actor before 1.0, effective learner 0.0, effective actor 0.0
u480 learner before 3.0, actor before 3.0, effective learner 0.0, effective actor 0.0
u540 learner before 3.0, actor before 3.0, effective learner 0.0, effective actor 0.0
```

Results:

```text
B1 vs u480:
  mean 0.8125, wins 13, losses 3, pair classes 2-0: 5, 1-1: 3

B1 vs u540:
  mean 0.9375, wins 15, losses 1, pair classes 2-0: 7, 1-1: 1

u480 vs B1:
  mean 0.1875, wins 3, losses 13, pair classes 0-2: 5, 1-1: 3

u540 vs B1:
  mean 0.25, wins 4, losses 12, pair classes 0-2: 4, 1-1: 4

u480 vs u540:
  mean 0.5, wins 8, losses 8, pair classes 2-0: 2, 1-1: 4, 0-2: 2

u540 vs u480:
  mean 0.5625, wins 9, losses 7, pair classes 2-0: 3, 1-1: 3, 0-2: 2
```

Interpretation:

- With the hand-coded public heuristic bias removed, B1 beats u480/u540 strongly.
- Therefore u480/u540 are not stronger learned networks underneath the wrapper.
- Official bias scale `3.0` seems to be doing most of the useful play, and it also masks the weakness of u480/u540 against B1.

## Diagnostic run C: both-greedy official learner mode

Command shape:

```text
--both-greedy
--pairs 8
--scoring-mode learner
```

Artifact:

```text
runs/league_typedpools_seedquarantine_u500_to_u540_20260427/eval/b1_artifact_matrix_bothgreedy_p8_20260427
```

Results:

```text
B1 vs u480:
  mean 0.5, wins 8, losses 8, pair classes 1-1: 8
  model_decisions 2128, greedy_override_decisions 2128, sample_decisions 0

B1 vs u540:
  mean 0.5, wins 8, losses 8, pair classes 1-1: 8
  model_decisions 2128, greedy_override_decisions 2128, sample_decisions 0

u480 vs B1:
  mean 0.5, wins 8, losses 8, pair classes 1-1: 8
  model_decisions 2128, greedy_override_decisions 2128, sample_decisions 0

u540 vs B1:
  mean 0.5, wins 8, losses 8, pair classes 1-1: 8
  model_decisions 2128, greedy_override_decisions 2128, sample_decisions 0

u480 vs u540:
  mean 0.5, wins 8, losses 8, pair classes 1-1: 8
  model_decisions 2128, greedy_override_decisions 2128, sample_decisions 0

u540 vs u480:
  mean 0.5, wins 8, losses 8, pair classes 1-1: 8
  model_decisions 2128, greedy_override_decisions 2128, sample_decisions 0
```

Interpretation:

- Common action RNG / stochastic sampling is not the main reason official B1-family matchups lock to `0.50`.
- With official learner scoring and bias `3.0`, even deterministic argmax policies stay in the same physical-seat winner pattern.

## Diagnostic run D: actor scoring mode, no override

Command shape:

```text
--scoring-mode actor
--pairs 8
```

Artifact:

```text
runs/league_typedpools_seedquarantine_u500_to_u540_20260427/eval/b1_artifact_matrix_actor_p8_20260427
```

Important manifest detail:

```text
B1 actor bias scale: 1.0
u480 actor bias scale: 3.0
u540 actor bias scale: 3.0
```

Results:

```text
B1 vs u480:
  mean 0.125, wins 2, losses 14, pair classes 0-2: 6, 1-1: 2

B1 vs u540:
  mean 0.125, wins 2, losses 14, pair classes 0-2: 6, 1-1: 2

u480 vs B1:
  mean 0.9375, wins 15, losses 1, pair classes 2-0: 7, 1-1: 1

u540 vs B1:
  mean 0.9375, wins 15, losses 1, pair classes 2-0: 7, 1-1: 1

u480 vs u540:
  mean 0.5, wins 8, losses 8, pair classes 1-1: 8

u540 vs u480:
  mean 0.5, wins 8, losses 8, pair classes 1-1: 8
```

Interpretation:

- Actor mode gives a huge apparent win for u480/u540 over B1.
- But this is probably not a clean "learned agent beats B1" result because the effective wrappers are unequal: B1 actor bias is `1.0`, u480/u540 actor bias is `3.0`.
- It may still reveal a serious training/eval mismatch: the B1 opponent during actor-side sampling may not match the B1 anchor used in official learner eval.

## Diagnostic run E: common public bias scale 1.0

Command shape:

```text
--public-heuristic-bias-scale 1.0
--pairs 8
--scoring-mode learner
```

Artifact:

```text
runs/league_typedpools_seedquarantine_u500_to_u540_20260427/eval/b1_artifact_matrix_bias1_p8_20260427
```

Manifest proved effective bias:

```text
B1 learner before 3.0, actor before 1.0, effective learner 1.0, effective actor 1.0
u480 learner before 3.0, actor before 3.0, effective learner 1.0, effective actor 1.0
u540 learner before 3.0, actor before 3.0, effective learner 1.0, effective actor 1.0
```

Results:

```text
B1 vs u480:
  mean 0.5, wins 8, losses 8, pair classes 2-0: 1, 1-1: 6, 0-2: 1

B1 vs u540:
  mean 0.625, wins 10, losses 6, pair classes 2-0: 2, 1-1: 6

u480 vs B1:
  mean 0.5, wins 8, losses 8, pair classes 1-1: 8

u540 vs B1:
  mean 0.4375, wins 7, losses 9, pair classes 2-0: 1, 1-1: 5, 0-2: 2

u480 vs u540:
  mean 0.8125, wins 13, losses 3, pair classes 2-0: 5, 1-1: 3

u540 vs u480:
  mean 0.3125, wins 5, losses 11, pair classes 1-1: 5, 0-2: 3
```

Interpretation:

- Common scale `1.0` does not produce a clear learned-policy win over B1.
- It does break exact `1-1:8` invariance in several matchups.
- u480 looks better than u540 at common scale `1.0`.
- Results are small/noisy and not symmetric exactly because stochastic sampling and policy-id RNG salts still vary, but the key point is that the official bias scale affects the dynamics dramatically.

## Consolidated interpretation so far

What seems mostly ruled out:

```text
1. The eval runner is globally ignoring policies.
   Built-in contrasts are policy-sensitive.

2. Current/B1 are the exact same loaded checkpoint.
   Hashes and paths differ.

3. Greedy override was silently doing nothing.
   New counters prove thousands of greedy-overridden decisions.

4. Stochastic sampling alone explains official 0.50.
   Both-greedy official mode is still exact 0.50.
```

What now seems likely:

```text
1. Official learner-mode public bias scale 3.0 collapses B1/u480/u540 into a near-identical effective policy.

2. The learned network body of u480/u540 is not clearly stronger than B1.
   With bias removed, B1 crushes them.

3. The league may be "improving" mainly by preserving or exploiting the hand-coded public heuristic wrapper, not by learning a robust better policy.

4. The actor/learner scoring mismatch is dangerous.
   B1 actor bias is 1.0; u480/u540 actor bias is 3.0; official learner eval gives all three learner scale 3.0.

5. Training opponents and eval opponents may not be the same effective policies.
   This could explain why B1 remains 0.50 in official eval no matter how much league pressure we add.

6. More PFSP with official bias 3.0 likely continues clone self-play inside the wrapper basin.
```

## What we need from you

Please answer as a debugging/research advisor. We need concrete next implementation steps, not just general theory.

Questions:

1. Is the strongest conclusion now that official B1 eval at public bias `3.0` is not measuring learned improvement well, because the bias wrapper dominates B1/u480/u540?

2. Should the thesis define the evaluated agent as:

   ```text
   neural network only
   neural network plus fixed public heuristic wrapper
   neural network plus annealed wrapper
   actor-mode policy
   learner-mode policy
   ```

   Which of these is scientifically defensible for a self-play/league thesis?

3. Is it valid to keep the public heuristic wrapper if it is constant across policies, or does the no-bias result mean our "learning" claim is too weak?

4. What should the next training branch be?

   Options we are considering:

   ```text
   A. Train/evaluate with bias scale annealed down from 3.0 to 1.0 or 0.0.
   B. Keep bias scale 3.0 for early stability but add a no-bias/low-bias auxiliary eval gate.
   C. Train a B1 exploiter under common bias 1.0 or no-bias, then import it as a hard negative.
   D. Define B1 anchor and league policies with identical actor/learner bias scale, then rerun from u480.
   E. Treat B1 actor-scale 1.0 as the actual opponent because that is what actors saw, and stop using learner-scale 3.0 B1 as the main gate.
   F. Remove or strongly reduce reference/teacher/public heuristic bias after warm-start and accept short-term B3/B4 regression while searching for a real best response.
   G. Do not train yet: implement action trace / logit probe / forced-action counterfactual first.
   ```

5. Should promotion gates be changed to include:

   ```text
   official bias B1 score
   common-scale-1 B1 score
   no-bias B1 score
   B3/B4 official score
   action/logit divergence from B1
   pair class counts
   ```

   If yes, what exact gates would you use locally versus on the L40 server?

6. How should we interpret actor-mode result where u480/u540 beat B1 `0.9375`, given that B1 actor bias is `1.0` and u480/u540 actor bias is `3.0`?

   Is this:

   ```text
   a real improvement because actor mode matches actor-side deployment,
   a wrapper mismatch artifact,
   or evidence that the B1 training/eval anchor is inconsistent?
   ```

7. Do we need to rebuild B1 as a baseline with learner and actor bias parity, e.g. both `3.0` or both `1.0`, before continuing league work?

8. Should we now implement the action trace/logit probe and forced-action counterfactual, or is the bias evidence already enough to redesign training/eval first?

## Candidate next code changes

Please critique or improve this plan.

### Patch 1: add a bias sweep confirm script or mode

Use `b1_artifact_matrix.py` to run common bias scales:

```text
0.0
0.5
1.0
2.0
3.0
```

For each scale:

```text
B1 vs u480
u480 vs B1
B1 vs u540
u540 vs B1
u480 vs u540
u540 vs u480
```

Record:

```text
overall score
seat0/seat1 split
pair classes
decision counters
effective bias scales
```

Goal:

```text
map exactly where B1-family policies become indistinguishable.
```

### Patch 2: add action trace/logit probe

For a few invariant seeds, record per-decision:

```json
{
  "episode_seed": 338610598310627562,
  "pair_index": 0,
  "swap_index": 0,
  "decision_index": 37,
  "actor_seat": 1,
  "policy_id": "u540",
  "scoring_mode": "learner",
  "public_bias_scale": 3.0,
  "legal_action_count": 12,
  "selected_action_id": 1234,
  "selected_action_family": "main_play_character",
  "legal_fingerprint": "...",
  "topk_action_ids": [1234, 99, 104],
  "topk_action_families": ["main_play_character", "main_move", "pass"],
  "topk_logits": [4.2, 3.1, -0.5],
  "public_heuristic_top_action": 1234,
  "public_heuristic_top_family": "main_play_character",
  "pre_state_digest": "...",
  "post_state_digest": "..."
}
```

Run under:

```text
official bias 3.0
common bias 1.0
no bias 0.0
both-greedy
```

Goal:

```text
prove whether u480/u540 differ strategically, or only by tiny logits/within-family noise.
```

### Patch 3: forced-action counterfactual

Before training an exploiter, implement a small replay-from-seed intervention:

```text
take pair 0 where physical seat 0 wins under B1-family policies
force winning seat to make bad actions at high-impact decisions
verify the winner can flip
then force losing seat to try plausible alternatives
```

Goal:

```text
determine whether legal deviations can flip B1-family physical-seat outcomes at all.
```

### Patch 4: new training branch

If you think the bias evidence is already enough, propose a concrete new config:

```text
branch from u480
common actor/learner bias scale maybe 1.0 or anneal 3.0 -> 1.0 -> 0.0
low LR
no reference exact-action BC or reduced BC
B1 direct lane
B3/B4 sanity lanes
promotion gate includes common-bias and no-bias diagnostics
candidate admitted only if official score does not collapse and low-bias B1 does not degrade
```

Please be specific about:

```text
which scale to train at
which scale to evaluate at
whether to keep B3/B4 anchors
how to keep the policy from becoming weak when the wrapper is reduced
how to avoid claiming wrapper-driven progress as learned progress
```

## Requested answer format

Please give:

1. A ranked diagnosis after these new bias/scoring results.
2. The most likely root cause of the flat B1 `0.50`.
3. Whether the official eval should be changed, supplemented, or kept.
4. The exact next 3 experiments, with stop/continue criteria.
5. Any code/config changes you recommend before more training.
6. Any thesis framing warnings about hand-coded heuristic wrappers.
7. What additional code blocks or artifacts you need next.

## Extra code/artifacts you may ask for next

If you need more context, tell us exactly what to paste. The most useful files or snippets are probably:

```text
python/weiss_rl/model.py
  PolicyValueModel.forward_seat_aware
  set_public_heuristic_logit_bias_scale
  get_public_heuristic_logit_bias_scale
  _public_heuristic_logit_bias_scale_for
  public heuristic logit bias application code

python/weiss_rl/eval/simulator_runner.py
  resolve_eval_policies
  _restore_public_heuristic_bias_schedule
  SimulatorEvalRunner._select_action
  SimulatorEvalRunner._rng_seed

python/weiss_rl/runtime.py
  actor-side model forward/scoring_mode
  opponent B1 baseline load/restore
  guidance schedule restore on resume

python/scripts/train.py
  checkpoint eval model load
  resume/import guidance schedule logic
  promotion gate config and current B1 gate

configs/presets/...current B1-init league config...
  model public heuristic bias fields
  actor/learner guidance schedule fields
  training loss coefficients
  league sampling lanes

Artifact rows:
  one full policy_load_manifest.json for official, no-bias, actor, and bias1 matrices
  pair_table.jsonl rows for u540 vs B1 under official/no-bias/actor/bias1
  first 20 action trace rows once implemented
```

