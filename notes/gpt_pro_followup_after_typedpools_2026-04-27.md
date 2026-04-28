# GPT Pro Follow-Up: League Topology Fixed, But B1 Still Flat And Quality Did Not Improve

You are reviewing a Weiss Schwarz reinforcement-learning thesis project after a concrete repair pass inspired by your previous feedback.

Assume this is a continuation of the same GPT Pro session: you have already seen the earlier long prompt and your own previous output. This follow-up is not asking you to repeat the old diagnosis. It gives you more code-shaped context and post-fix evidence because you cannot open the repository. Please use the previous context plus the concrete snippets below to reason like a senior RL systems debugger.

Assume you cannot inspect the repository. This prompt is self-contained and focuses on the new evidence after implementing typed league-pool / seed-history quarantine fixes.

Workspace:

```text
C:\Users\Bruger\Desktop\this one\weiss_schwarz_rl
```

Date:

```text
2026-04-27
```

## One-Sentence Status

We fixed the bookkeeping/topology bug enough that PFSP now samples true local league recents instead of imported B1 seed history, and a real local champion was promoted, but the model still stays exactly `0.50` against B1 and the 32-pair quality confirm is worse than the previous u480 best.

## Context From The Previous Review

Your previous diagnosis was:

1. The old league was not a healthy evolving league ladder.
2. `Previous recent snapshot` often resolved to imported B1 seed history, not a learned league opponent.
3. Seed imports could become active champions, which could pollute champion aliases/PFSP.
4. Raw `SnapshotRegistry.latest_ids(...)` was too ambiguous for semantic league pools.
5. The B1 `0.50` problem is likely a separate best-response / B1-clone issue.

We implemented the topology/bookkeeping repair first.

## Code Changes Implemented

### Registry Typed Selectors

File:

```text
python/weiss_rl/league/registry.py
```

Implemented selectors:

```python
snapshot_by_policy_id()
latest_seed_history_ids(...)
latest_local_candidate_ids(...)
latest_active_champion_ids(...)
latest_eligible_ids(...)
```

Behavior:

```text
seed_import snapshots are seed history only;
baseline_anchor snapshots are not active champions;
active champion selectors exclude seed history and baseline anchors;
local candidate selectors include source_kind local and league_import;
raw latest_ids remains, but PFSP/eval alias code now uses typed selectors.
```

`normalize()` now strips seed imports and baseline anchors out of active `champion_snapshots`.

### Seed Import Quarantine

File:

```text
python/scripts/train.py
```

Old behavior:

```python
if source_snapshot.policy_id in source_champions:
    registry.add_champion(imported_policy_id)
```

New behavior:

```text
Do not add seed imports to active champions.
Write metadata source_was_champion instead.
```

So imported B1 history can no longer silently become the current run's champion ladder.

### Explicit Aliases

File:

```text
python/scripts/train.py
```

Added / routed aliases:

```text
Latest local candidate snapshot
Previous local candidate snapshot
Latest imported seed history snapshot
Previous imported seed history snapshot
Latest promoted champion snapshot
Previous promoted champion snapshot
```

Existing aliases:

```text
Latest recent snapshot
Previous recent snapshot
Latest champion snapshot
Previous champion snapshot
```

now resolve through seed-safe selectors rather than raw registry ordering.

Important remaining semantic wrinkle:

```text
When promotion_gate_enabled=True, old "Previous recent snapshot" still behaves champion-like for compatibility.
The new explicit local-candidate aliases are clearer and should probably replace it in future configs/eval.
```

### Resume Import Hardened

File:

```text
python/scripts/train.py
```

`_import_resume_league_snapshot_pool(...)` now validates an existing same-policy ID collision via metadata instead of silently accepting it.

It still preserves source policy IDs like:

```text
policy_000011
policy_000012
```

when importing true league history from the resumed run.

### Runtime PFSP Pool

File:

```text
python/weiss_rl/runtime.py
```

PFSP now uses typed selectors:

```text
champion pool -> latest_active_champion_ids(...)
recent pool   -> latest_local_candidate_ids(...), excluding active champions
```

Seed history now has a separate warmup lane:

```text
_opponent_warmup_snapshot_ids
pfsp_warmup_snapshot_pool_size
```

Before PFSP handoff:

```text
seed history may be sampled via warmup_snapshot_mix_fraction;
it is no longer called recent/champion.
```

After PFSP handoff:

```text
pfsp_warmup_snapshot_envs = 0
pfsp_sampling_weight_warmup_snapshot = 0
true local/league_import recents drive PFSP recent lane.
```

## Tests Passed

Compile:

```text
uv run python -m py_compile python/weiss_rl/league/registry.py python/scripts/train.py python/weiss_rl/runtime.py
```

Focused registry/train tests:

```text
7 passed
```

Included:

```text
test_import_seed_snapshot_pool_imports_external_snapshots_as_seed_history_not_champions
test_import_seed_snapshot_pool_respects_max_update_for_resume_continuation
test_import_resume_league_snapshot_pool_preserves_local_recents_and_champions
test_snapshot_registry_typed_selectors_keep_seed_history_out_of_active_champions
test_periodic_dev_eval_previous_recent_ignores_seed_imports_and_rejections
test_periodic_dev_eval_previous_recent_requires_champion_when_promotion_gated
test_symbolic_snapshot_aliases_keep_seed_history_explicit
```

Focused runtime tests:

```text
6 passed
```

Included:

```text
test_refresh_opponent_pool_can_exclude_seed_imports_after_pfsp_handoff
test_refresh_opponent_pool_keeps_seed_imports_before_pfsp_handoff
test_refresh_opponent_pool_never_treats_seed_history_as_active_champion_or_recent
test_refresh_opponent_pool_keeps_small_recent_reservoir_when_promotion_gate_enabled
test_refresh_opponent_pool_uses_probationary_recent_pool_before_first_champion
test_refresh_opponent_pool_keeps_champions_out_of_recent_lane
```

Broader repaired-sampler regression:

```text
6 passed
```

Included:

```text
test_assign_episode_roles_uses_weighted_sampler_on_diverse_warmup_lane
test_sample_opponent_policy_ids_can_force_noleague_baseline_bucket_before_pfsp_ready
test_sample_opponent_policy_ids_can_force_warmup_snapshot_bucket_before_pfsp_ready
test_guidance_schedule_applies_configured_actor_bias_after_resume
```

## Runs After The Fix

### Smoke: u480 To u481

Run:

```text
runs/league_typedpools_seedquarantine_smoke_u480_to_u481_20260427
```

Command shape:

```text
resume from current best u480 checkpoint
reset optimizer
max-updates 481
seed B1 history from b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425
```

Startup:

```text
Imported resume league snapshot pool: count=1
Imported seeded snapshot pool: count=7
```

Registry:

```text
champion_snapshots: []
seed_315d5e55ce_policy_000003  update 150  seed_import
seed_315d5e55ce_policy_000004  update 200  seed_import
seed_315d5e55ce_policy_000005  update 250  seed_import
seed_315d5e55ce_policy_000006  update 300  seed_import
seed_315d5e55ce_policy_000007  update 350  seed_import
seed_315d5e55ce_policy_000008  update 400  seed_import
b1_noleague_baseline           update 450  baseline_anchor
seed_315d5e55ce_policy_000009  update 450  seed_import
policy_000011                  update 480  league_import
policy_000012                  update 481  local
```

Runtime:

```text
pfsp_pool_size=1
pfsp_recent_pool_size=1
pfsp_champion_pool_size=0
pfsp_warmup_snapshot_pool_size=7
pfsp_sampling_ready=0
pfsp_sampling_weight_warmup_snapshot=0.35
```

Interpretation:

```text
Topology smoke passed. True league history is recent; seed history is warmup seed history.
```

### Short Run: u480 To u500

Run:

```text
runs/league_typedpools_seedquarantine_u480_to_u500_20260427
```

Result:

```text
Periodic dev eval u500 aggregate: 0.8000
B0 RandomLegal:       1.0
B1 NoLeague baseline: 0.50
B2 HeuristicPublic:   1.0
B3 HeuristicAggro:    1.0
B4 HeuristicControl:  1.0
```

Registry:

```text
champion_snapshots: []
rejected_snapshots: []
policy_000011 update 480 source_kind league_import
policy_000012 update 500 source_kind local
```

Runtime tail before PFSP:

```text
league_effective_update=490
pfsp_sampling_ready=0
pfsp_recent_pool_size=1
pfsp_warmup_snapshot_pool_size=7
pfsp_sampling_weight_warmup_snapshot=0.35
pfsp_warmup_snapshot_envs > 0
pfsp_recent_envs=0
```

Interpretation:

```text
Looks good locally, but it is still pre-PFSP due effective-update lag.
```

### PFSP Handoff Run: u500 To u540

Run:

```text
runs/league_typedpools_seedquarantine_u500_to_u540_20260427
```

Startup:

```text
Imported resume league snapshot pool: count=2
Imported seeded snapshot pool: count=7
```

At u520:

```text
Periodic dev eval aggregate: 0.7750
B1: 0.50
B3: 0.9375
Checkpoint guard rollback: score_drop
policy_000013 rejected
```

At u540:

```text
Promotion gate passed: candidate=policy_000014
Periodic dev eval aggregate: 0.8000
B1: 0.50
B3: 1.0
B4: 1.0
```

Final registry:

```text
champion_snapshots: ['policy_000014']
rejected_snapshots: ['policy_000013']

seed_315d5e55ce_policy_000003  update 150  seed_import
seed_315d5e55ce_policy_000004  update 200  seed_import
seed_315d5e55ce_policy_000005  update 250  seed_import
seed_315d5e55ce_policy_000006  update 300  seed_import
seed_315d5e55ce_policy_000007  update 350  seed_import
seed_315d5e55ce_policy_000008  update 400  seed_import
b1_noleague_baseline           update 450  baseline_anchor
seed_315d5e55ce_policy_000009  update 450  seed_import
policy_000011                  update 480  league_import
policy_000012                  update 500  league_import
policy_000013                  update 520  local
policy_000014                  update 540  local
```

PFSP handoff runtime evidence after effective update crossed warmup threshold:

```text
league_effective_update=530
pfsp_sampling_ready=1
pfsp_pool_size=2
pfsp_recent_pool_size=2
pfsp_champion_pool_size=0
pfsp_warmup_snapshot_pool_size=7
pfsp_sampling_weight_recent=0.40
pfsp_sampling_weight_warmup_snapshot=0.0
pfsp_recent_envs > 0
pfsp_warmup_snapshot_envs=0
```

Important interpretation:

```text
PFSP is now mechanically and semantically using true local/league-import recent policies.
Seed imports are present in registry but not sampled as PFSP recent/champion after handoff.
This is the first good evidence that the topology bug is fixed.
```

## 32-Pair Confirmatory Eval At u540

Command:

```text
uv run python python/scripts/manual_dev_eval_confirm.py \
  --stack-config configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_b1init_devtargetsharp_actorparity3_shift450_vlowlr_frontierweighted_seedpool_lag10_evalgated_lowlr_evalguard_localpromo.yaml \
  --run-dir runs/league_typedpools_seedquarantine_u500_to_u540_20260427 \
  --checkpoint runs/league_typedpools_seedquarantine_u500_to_u540_20260427/training/checkpoints/checkpoint_540.pt \
  --summary runs/league_typedpools_seedquarantine_u500_to_u540_20260427/eval/dev_eval/update_540/summary.json \
  --update 540 \
  --pairs 32 \
  --workers 6 \
  --artifact-dir-name dev_eval_trueanchors_manual32_20260427
```

Result:

```text
aggregate_score:       0.768750
unweighted_aggregate:  0.884375
B0 RandomLegal:        1.0
B1 NoLeague baseline:  0.50
B2 HeuristicPublic:    1.0
B3 HeuristicAggro:     0.921875
B4 HeuristicControl:   1.0
```

Seat diagnostics for B1:

```text
train_u540_p14 vs B1:
  total wins: 32 / 64
  wins as seat0: 14 / 32
  wins as seat1: 18 / 32

B1 baseline:
  total wins: 32 / 64
  wins as seat0: 14 / 32
  wins as seat1: 18 / 32
```

This is exactly symmetric in total and seat split.

Comparison to previous best u480 true-anchor 32-pair confirm:

```text
u480 previous best:
  weighted aggregate:   0.787500
  unweighted aggregate: 0.893750
  B1:                   0.50
  B3:                   0.96875
  B4:                   1.0

u540 after topology fix:
  weighted aggregate:   0.768750
  unweighted aggregate: 0.884375
  B1:                   0.50
  B3:                   0.921875
  B4:                   1.0
```

So the topology repair worked, but the quality did not improve. It is slightly worse than u480 on 32-pair confirm, mainly from B3 cooling.

## What We Need From You Now

Please focus on the next structural learning step, not the already-fixed bookkeeping bug.

Key facts:

```text
1. True local recent PFSP now works.
2. Seed imports are quarantined from active champion/recent semantics.
3. A real local champion was promoted at u540.
4. B1 is still exactly 0.50.
5. The u540 32-pair confirm is worse than u480, not better.
6. B1 seat diagnostics are exactly symmetric between current and B1.
7. The model is still likely B1-family cloned.
```

Questions:

1. Is this enough evidence that the old topology bug is fixed?
2. Given that fixed PFSP still does not improve quality by u540, should we:
   - stop extending this branch;
   - continue longer because the first true PFSP/champion only appeared at u540;
   - modify promotion/admission tiers before continuing;
   - immediately switch to a dedicated B1 exploiter/counterfactual branch?
3. Is the u540 champion actually useful as a league opponent even though it is not better than u480?
4. Should the champion gate have promoted u540 when B1 is 0.50 and B3 confirmed lower than u480 on larger eval?
5. Should the next code change be:
   - candidate/admitted-recent tier separate from champion;
   - B1 exploiter role;
   - counterfactual rollout teacher from B1 seat/loss states;
   - seat-specific B1 objective;
   - policy diversity objective;
   - or eval redesign first?
6. The B1 32-pair seat diagnostics show both current and B1 have identical total/seat wins.
   - Does this suggest the policies are functionally identical on the paired eval surface?
   - What diagnostic would prove whether there is any real strategic divergence?
7. What exact next experiment should a coding agent run locally before any L40 server work?

## Desired Output

Please produce:

1. A diagnosis of the post-fix state.
2. A verdict on whether to continue or stop this fixed-topology branch.
3. A ranked next-step plan for achieving real B1 improvement.
4. Concrete code/config changes to make next.
5. A minimal local experiment matrix with stop/continue criteria.
6. Any additional artifact/logging requirements before server scaling.

---

# Continuation Addendum: More Code Context For Debugging

This addendum is deliberately more detailed than normal. You should assume the Codex agent can implement your suggestions, but you cannot inspect files yourself. We need your help finding the structural reason the league/self-play system is not trending upward after the topology repair.

The user's simple mental model is:

```text
If B1 no-league anchor is already useful, then a sound league/self-play system should eventually create stronger policies.
With enough compute, it should not stay flat forever against the same B1 anchor.
The thesis does not need a perfect solved game, but it does need a believable upward-learning system.
```

The observed reality:

```text
1. B1-initialized league is much better than fresh league.
2. Registry/alias/PFSP topology bugs existed and are now materially fixed.
3. After topology fix, true local recents are sampled and a local champion appears.
4. Quality still does not improve against B1.
5. The B1 score is not just "roughly" 0.50. It repeatedly lands exactly 0.50 on paired eval.
6. Larger confirm shows u540 after the repair is worse than u480 before continuing.
```

Please treat this as a possible learning-signal / evaluation / clone-equilibrium issue, not just a config-tuning problem.

## Important Current Code Snippets

These are shortened but faithful excerpts from the current repaired code.

### Snapshot Registry Typed Selectors

File:

```text
python/weiss_rl/league/registry.py
```

Current selector shape:

```python
@dataclass(slots=True)
class SnapshotRegistry:
    recent_size: int = 24
    champion_size: int = 4
    snapshots: list[SnapshotMeta] = field(default_factory=list)
    champion_snapshots: list[str] = field(default_factory=list)
    pinned_snapshots: list[str] = field(default_factory=list)
    rejected_snapshots: list[str] = field(default_factory=list)

    def latest_ids(self, n: int = 1) -> list[str]:
        return [snapshot.policy_id for snapshot in self.latest(n)]

    def snapshot_by_policy_id(self) -> dict[str, SnapshotMeta]:
        self.normalize()
        return {snapshot.policy_id: snapshot for snapshot in self.snapshots}

    def latest_seed_history_ids(
        self,
        n: int = 1,
        *,
        exclude_rejected: bool = True,
        exclude_policy_ids: Iterable[str] = (),
    ) -> list[str]:
        return self.latest_eligible_ids(
            n,
            source_kinds={"seed_import"},
            include_seed_history=True,
            exclude_rejected=exclude_rejected,
            exclude_policy_ids=exclude_policy_ids,
        )

    def latest_local_candidate_ids(
        self,
        n: int = 1,
        *,
        include_league_import: bool = True,
        exclude_rejected: bool = True,
        exclude_policy_ids: Iterable[str] = (),
    ) -> list[str]:
        source_kinds = {"local", "legacy"}
        if include_league_import:
            source_kinds.add("league_import")
        return self.latest_eligible_ids(
            n,
            source_kinds=source_kinds,
            include_seed_history=False,
            exclude_rejected=exclude_rejected,
            exclude_policy_ids=exclude_policy_ids,
        )

    def latest_active_champion_ids(
        self,
        n: int = 1,
        *,
        current_update: int | None = None,
        max_age_updates: int | None = None,
        exclude_policy_ids: Iterable[str] = (),
    ) -> list[str]:
        self.normalize()
        snapshots_by_id = self.snapshot_by_policy_id()
        champion_ids = self.latest_champions(
            len(self.champion_snapshots),
            current_update=current_update,
            max_age_updates=max_age_updates,
        )
        active_ids = [
            snapshot_id
            for snapshot_id in champion_ids
            if snapshot_id not in excluded_ids
            and not self._snapshot_is_seed_history(snapshots_by_id.get(snapshot_id), policy_id=snapshot_id)
            and not self._snapshot_is_baseline_anchor(snapshots_by_id.get(snapshot_id))
        ]
        return active_ids[-n:]

    def latest_eligible_ids(
        self,
        n: int = 1,
        *,
        source_kinds: Iterable[str] | None = None,
        include_seed_history: bool = False,
        include_baseline_anchors: bool = False,
        exclude_rejected: bool = True,
        exclude_policy_ids: Iterable[str] = (),
    ) -> list[str]:
        self.normalize()
        for snapshot in reversed(self.snapshots):
            if not include_seed_history and self._snapshot_is_seed_history(snapshot, policy_id=policy_id):
                continue
            if not include_baseline_anchors and self._snapshot_is_baseline_anchor(snapshot):
                continue
            if source_kind_filter is not None and source_kind not in source_kind_filter:
                continue
            eligible_ids.append(policy_id)
        return list(reversed(eligible_ids))

    def normalize(self) -> None:
        self.snapshots = self._normalized_snapshots()
        existing_snapshot_ids = {snapshot.policy_id for snapshot in self.snapshots}
        active_champion_ids = {
            snapshot.policy_id
            for snapshot in self.snapshots
            if not self._snapshot_is_seed_history(snapshot, policy_id=snapshot.policy_id)
            and not self._snapshot_is_baseline_anchor(snapshot)
        }
        self.champion_snapshots = self._normalized_refs(
            self.champion_snapshots,
            existing_snapshot_ids=existing_snapshot_ids & active_champion_ids,
            limit=self.champion_size,
        )
```

Question for you:

```text
Does this selector design look sufficient, or is there still a semantic flaw because status/pool_role is not explicit enough?
Should local candidates, admitted recents, frontier champions, and B1 exploiters be first-class status/pool roles rather than inferred from source_kind + champion_snapshots + rejected_snapshots?
```

### Symbolic Eval Alias Resolution

File:

```text
python/scripts/train.py
```

Current alias resolver:

```python
def _resolve_symbolic_promotion_anchor_policy_id(
    anchor_name: str,
    *,
    registry: SnapshotRegistry,
    promotion_gate_enabled: bool = False,
) -> str | None:
    if anchor_name in {"Latest champion snapshot", "Latest promoted champion snapshot"}:
        champion_ids = registry.latest_active_champion_ids(
            1,
            exclude_policy_ids=_FIXED_OPPONENT_EXCLUSIONS,
        )
        return None if not champion_ids else str(champion_ids[-1])

    if anchor_name in {"Previous champion snapshot", "Previous promoted champion snapshot"}:
        champion_ids = registry.latest_active_champion_ids(
            2,
            exclude_policy_ids=_FIXED_OPPONENT_EXCLUSIONS,
        )
        return None if len(champion_ids) < 2 else str(champion_ids[-2])

    if anchor_name in {"Latest recent snapshot", "Latest local candidate snapshot"}:
        recent_ids = _true_local_recent_snapshot_ids(registry, promotion_gate_enabled=promotion_gate_enabled)
        return None if not recent_ids else str(recent_ids[-1])

    if anchor_name in {"Previous recent snapshot", "Previous local candidate snapshot"}:
        recent_ids = _true_local_recent_snapshot_ids(registry, promotion_gate_enabled=promotion_gate_enabled)
        return None if len(recent_ids) < 2 else str(recent_ids[-2])

    if anchor_name == "Latest imported seed history snapshot":
        seed_ids = registry.latest_seed_history_ids(
            1,
            exclude_rejected=True,
            exclude_policy_ids=_FIXED_OPPONENT_EXCLUSIONS,
        )
        return None if not seed_ids else str(seed_ids[-1])

    if anchor_name == "Previous imported seed history snapshot":
        seed_ids = registry.latest_seed_history_ids(
            2,
            exclude_rejected=True,
            exclude_policy_ids=_FIXED_OPPONENT_EXCLUSIONS,
        )
        return None if len(seed_ids) < 2 else str(seed_ids[-2])

    return None


def _true_local_recent_snapshot_ids(
    registry: SnapshotRegistry,
    *,
    promotion_gate_enabled: bool = False,
) -> tuple[str, ...]:
    if promotion_gate_enabled:
        return tuple(
            registry.latest_active_champion_ids(
                len(getattr(registry, "champion_snapshots", ())),
                exclude_policy_ids=_FIXED_OPPONENT_EXCLUSIONS,
            )
        )
    return tuple(
        registry.latest_local_candidate_ids(
            len(getattr(registry, "snapshots", ())),
            include_league_import=True,
            exclude_rejected=True,
            exclude_policy_ids=_FIXED_OPPONENT_EXCLUSIONS,
        )
    )
```

Potential concern:

```text
When promotion_gate_enabled=True, "Previous recent snapshot" still behaves champion-like for compatibility.
The runtime PFSP recent pool can contain local candidates even before champion promotion, but eval's symbolic "Previous recent snapshot" may still resolve only active champions when promotion-gated.
This may mean training PFSP and eval aliases are not perfectly aligned.
```

Please assess:

```text
Should "Previous recent snapshot" be fully deprecated now?
Should periodic eval/promotion gates use explicit aliases only?
Should promotion-gated eval still look at previous local/admitted recent, not only champions?
```

### Seed Import Quarantine

File:

```text
python/scripts/train.py
```

Current seed import behavior:

```python
def _import_seed_snapshot_pool(...):
    source_registry = SnapshotRegistry.load(source_registry_path)
    source_champions = set(source_registry.champion_snapshots)

    for source_snapshot in source_snapshots:
        imported_policy_id = _seed_snapshot_policy_id(
            source_run_dir=source_run_dir,
            source_policy_id=source_snapshot.policy_id,
        )

        imported_payload = dict(payload)
        imported_payload["policy_id"] = imported_policy_id
        imported_payload["imported_from_run_dir"] = source_run_dir.as_posix()
        imported_payload["imported_from_policy_id"] = source_snapshot.policy_id
        imported_payload["imported_from_snapshot_path"] = source_snapshot.path
        imported_payload["seeded_from_external_registry"] = True
        imported_payload["source_was_champion"] = source_snapshot.policy_id in source_champions
        torch.save(imported_payload, weights_path)

        registry.add_snapshot(
            policy_id=imported_policy_id,
            update=int(source_snapshot.update),
            weights_sha256=weights_sha256,
            path=weights_path.relative_to(run_dir).as_posix(),
            source_kind="seed_import",
        )
```

Old behavior removed:

```python
if source_snapshot.policy_id in source_champions:
    registry.add_champion(imported_policy_id)
```

Question:

```text
Is source_kind alone enough to keep seed snapshots from contaminating future champion/recent semantics?
Should seed history be in a different registry or pool file entirely?
```

### Resume League Import

File:

```text
python/scripts/train.py
```

Current resume import filter:

```python
def _source_snapshot_is_resume_league_snapshot(snapshot: SnapshotMeta, *, rejected_policy_ids: set[str]) -> bool:
    policy_id = str(snapshot.policy_id).strip()
    if (
        not policy_id
        or policy_id in rejected_policy_ids
        or policy_id in _FIXED_OPPONENT_EXCLUSIONS
        or policy_id.startswith("seed_")
    ):
        return False
    source_kind = str(getattr(snapshot, "source_kind", "local") or "local").strip().lower()
    return source_kind not in {"seed_import", "baseline_anchor"}


def _import_resume_league_snapshot_pool(...):
    source_run_dir = _infer_run_dir_from_checkpoint_path(resume_checkpoint_path)
    source_registry = SnapshotRegistry.load(source_registry_path)
    rejected_policy_ids = set(getattr(source_registry, "rejected_snapshots", ()))
    source_snapshots = [
        snapshot
        for snapshot in source_registry.snapshots
        if int(snapshot.update) <= int(max_update)
        and _source_snapshot_is_resume_league_snapshot(snapshot, rejected_policy_ids=rejected_policy_ids)
    ]
```

Current effect:

```text
When starting u500->u540 from the fixed u500 run, both policy_000011 and policy_000012 were imported as league_import.
Rejected policy_000013 did not become a future import target.
Seed imports did not become league imports.
```

Question:

```text
Should rejected candidates be omitted from future league imports, or preserved as diagnostic/hard-negative candidates?
Could omitting rejected candidates remove useful diversity that a league needs?
```

### Runtime PFSP Pool Construction

File:

```text
python/weiss_rl/runtime.py
```

Current repaired pool refresh:

```python
registry = SnapshotRegistry.load(self._registry_path)

admitted_champion_ids = tuple(
    registry.latest_active_champion_ids(
        int(self._league_config.snapshot_pool_champion_size),
        current_update=current_update,
        max_age_updates=max_age_updates,
        exclude_policy_ids=_FIXED_OPPONENT_EXCLUSIONS,
    )
)

recent_size = int(self._league_config.snapshot_pool_recent_size)
if bool(self._league_config.promotion_gate_enabled):
    recent_size = self._promotion_gated_recent_reservoir_size(
        base_recent_size=recent_size,
        champion_size=int(self._league_config.snapshot_pool_champion_size),
        admitted_champion_ids=admitted_champion_ids,
    )

warmup_snapshot_ids = (
    ()
    if self._exclude_seed_import_snapshots_from_pfsp_active(current_update=current_update)
    else tuple(
        registry.latest_seed_history_ids(
            seed_history_query_size,
            exclude_rejected=bool(self._league_config.promotion_gate_enabled),
            exclude_policy_ids=_FIXED_OPPONENT_EXCLUSIONS,
        )
    )
)

recent_ids = tuple(
    registry.latest_local_candidate_ids(
        int(recent_size),
        include_league_import=True,
        exclude_rejected=bool(self._league_config.promotion_gate_enabled),
        exclude_policy_ids=tuple(dict.fromkeys([*_FIXED_OPPONENT_EXCLUSIONS, *champion_ids])),
    )
)

candidate_ids = tuple(dict.fromkeys([*champion_ids, *recent_ids]))
hard_negative_ids = self._select_hard_negative_ids(candidate_ids)
candidate_ids = tuple(dict.fromkeys([*hard_negative_ids, *champion_ids, *recent_ids]))

self._opponent_candidate_ids = candidate_ids
self._opponent_champion_ids = champion_ids
self._opponent_recent_ids = recent_ids
self._opponent_warmup_snapshot_ids = warmup_snapshot_ids
self._pfsp_champion_pool_size = len(champion_ids)
self._pfsp_recent_pool_size = len(recent_ids)
self._pfsp_warmup_snapshot_pool_size = len(warmup_snapshot_ids)
```

Current PFSP sampling groups:

```python
pfsp_ready = self._pfsp_sampling_ready()

heuristic_public_weight = self._active_heuristic_public_mix_fraction()
heuristic_public_variant_weight = self._active_heuristic_public_variant_mix_fraction()
noleague_baseline_weight = self._active_noleague_baseline_mix_fraction()
warmup_snapshot_weight = self._active_warmup_snapshot_mix_fraction()
mirror_weight = max(0.0, float(getattr(sampling_cfg, "mirror_mix_fraction", 0.0)))
champion_weight = max(0.0, float(getattr(sampling_cfg, "champion_mix_fraction", 0.35)))
hard_negative_weight = max(0.0, float(getattr(sampling_cfg, "hard_negative_mix_fraction", 0.2)))

if noleague_baseline_weight > 0.0 and _NOLEAGUE_BASELINE_POLICY_ID in self._opponent_models:
    groups.append(("noleague_baseline", (_NOLEAGUE_BASELINE_POLICY_ID,), noleague_baseline_weight))
    non_recent_weight += noleague_baseline_weight

if not pfsp_ready and warmup_snapshot_weight > 0.0 and warmup_snapshot_ids:
    groups.append(("warmup_snapshot", warmup_snapshot_ids, warmup_snapshot_weight))
    non_recent_weight += warmup_snapshot_weight

if pfsp_ready and self._opponent_hard_negative_ids:
    groups.append(("hard_negative", self._opponent_hard_negative_ids, hard_negative_weight))
    non_recent_weight += hard_negative_weight

if pfsp_ready and self._opponent_champion_ids:
    groups.append(("champion", self._opponent_champion_ids, champion_weight))
    non_recent_weight += champion_weight

if pfsp_ready and self._opponent_recent_ids:
    recent_weight = max(0.0, 1.0 - non_recent_weight)
    groups.append(("recent", self._opponent_recent_ids, recent_weight))

if not pfsp_ready:
    mirror_weight = max(0.0, 1.0 - non_recent_weight)
    groups.append(("mirror", (_MIRROR_OPPONENT_POLICY_ID,), mirror_weight))
```

Runtime evidence after handoff:

```text
league_effective_update=530
pfsp_sampling_ready=1
pfsp_pool_size=2
pfsp_recent_pool_size=2
pfsp_champion_pool_size=0
pfsp_warmup_snapshot_pool_size=7
pfsp_sampling_weight_recent=0.40
pfsp_sampling_weight_warmup_snapshot=0.0
pfsp_recent_envs > 0
pfsp_warmup_snapshot_envs=0
```

Important nuance:

```text
The metric pfsp_warmup_snapshot_pool_size remains 7 because the registry can still see seed history,
but sampling weight/envs are zero after handoff.
So the warmup pool size metric alone is not evidence of active sampling.
```

Question:

```text
Does this PFSP sampling schedule make strategic sense?
After PFSP opens, recent gets whatever remains after B1/heuristic/variant/champion/hard-negative weights.
Could true recent pressure still be too weak, too clone-like, or too abrupt?
Should recent/champion pressure ramp, be opponent-family balanced, or be outcome/PFSP-score based differently?
```

## Main Config Surface

Best/current fixed-topology branch config:

```text
configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_b1init_devtargetsharp_actorparity3_shift450_vlowlr_frontierweighted_seedpool_lag10_evalgated_lowlr_evalguard_localpromo.yaml
```

This config extends the B1-initialized bridge and changes LR to:

```yaml
training:
  optimizer:
    learning_rate: 0.000005
league:
  sampling:
    exclude_seed_snapshots_from_pfsp: true
```

Parent B1-init config has:

```yaml
training:
  reference_policy_top_action_bc_coef: 0.20
  reference_policy_top_action_bc_final_coef: 0.08
  reference_policy_top_action_bc_start_updates: 450
  reference_policy_top_action_bc_end_updates: 750
  reference_policy_top_action_family_bc_coef: 0.30
  reference_policy_top_action_family_bc_final_coef: 0.12
  reference_policy_top_action_family_bc_start_updates: 450
  reference_policy_top_action_family_bc_end_updates: 750
  structured_aux:
    teacher_public_heuristic_coef: 0.20
    teacher_public_heuristic_final_coef: 0.08
    teacher_public_heuristic_start_updates: 450
    teacher_public_heuristic_end_updates: 750
    teacher_public_heuristic_temperature: 8.0
    teacher_public_main_move_coef: 0.12
league:
  sampling:
    noleague_baseline_mix_end_updates: 750
  warmup:
    first_updates: 520
    eval_gate_enabled: true
    eval_gate_min_aggregate_score: 0.60
    eval_gate_min_anchor_scores:
      B1 NoLeague baseline: 0.50
      B3 HeuristicPublicAggro: 0.65
      Previous recent snapshot: 0.50
```

Earlier parent configs include:

```yaml
model:
  public_heuristic_logit_bias_scale: 3.0
  public_heuristic_actor_logit_bias_scale: 3.0
  public_heuristic_logit_bias_final_scale: 3.0
training:
  exploration:
    entropy_coef: 0.05
    entropy_anneal_to: 0.025
  structured_aux:
    teacher_public_heuristic_coef: 0.20
    teacher_public_heuristic_final_coef: 0.08
    teacher_public_heuristic_temperature: 8.0
    teacher_public_main_move_coef: 0.12
league:
  pool:
    recent_size: 32
    champion_size: 8
  sampling:
    heuristic_public_mix_fraction: 0.30
    heuristic_public_final_mix_fraction: 0.20
    heuristic_public_variant_mix_fraction: 0.20
    heuristic_public_variant_final_mix_fraction: 0.20
    noleague_baseline_mix_fraction: 0.20
    warmup_snapshot_mix_fraction: 0.35
```

Questions:

```text
1. Does reference BC + public heuristic bias likely keep the student too close to B1/teacher?
2. If yes, why did previous ref-off / anti-clone / stronger B1 pressure attempts still fail?
3. Should the main policy and B1 exploiter have different objective configs?
4. Is very low LR (5e-6) stabilizing the B1 clone rather than enabling strategic search?
5. Is the promotion gate too easy because aggregate is dominated by B2/B4 and not B1 progress?
6. Does the old `Previous recent snapshot` warmup gate still point at the wrong semantic thing under promotion-gated alias rules?
```

## Manual Eval Helper Limitations

The 32-pair confirm helper:

```text
python/scripts/manual_dev_eval_confirm.py
```

Important behavior:

```python
paired_seeds = train_script._expand_periodic_dev_eval_paired_seeds(
    base_seeds,
    requested_pairs=int(args.pairs),
    seed_file_sha256=seed_sha,
    update_count=int(args.update),
    policy_version=int(summary["policy_version"]),
    scope=str(args.artifact_dir_name),
)

result = train_script._run_periodic_dev_eval_for_checkpoint(
    stack=stack,
    contract=contract,
    run_dir=args.run_dir,
    checkpoint_path=args.checkpoint,
    focal_policy_id=focal_policy_id,
    update_count=int(args.update),
    policy_version=int(summary["policy_version"]),
    paired_seeds_override=paired_seeds,
    parallel_workers_override=int(args.workers),
    batched_inference_override=False,
    opponent_specs=opponent_specs,
)
```

It can optionally force greedy focal action selection:

```python
if args.focal_action_mode == "greedy":
    # monkeypatch SimulatorEvalRunner._select_action for the focal policy only
```

But right now it does not yet produce a full pair-level B1 sweep table in the summary, unless the underlying periodic eval summary already includes enough seat/opponent diagnostic fields. The current seat diagnostics were inspected from the resulting summary.

Please suggest exact eval artifacts to add:

```text
pair-level B1 table:
  seed_id
  focal seat0 result
  focal seat1 result
  pair class: 2-0 / 1-1 / 0-2
  terminal reason
  turns
  damage / clock / level / stock summaries if available
  current policy action trace digest
  B1 action trace digest

alias resolution table:
  display_name
  policy_id
  source_kind
  registry_status
  update
  weights_sha256
  path
```

## B1 Disagreement Evidence From Earlier Work

Previous B1 audit on the current-best area suggested:

```text
top-family match rate against B1: about 1.0
top-action match rate against B1: about 0.87
mean probability on B1 top action: about 0.71
wins and losses had nearly identical B1-action agreement
```

The user summary also said:

```text
Look specifically at pass when non-pass available,
main_play_character vs pass,
main_move vs pass,
clock/mulligan/attack/level-up choices,
early board development,
climax/event usage,
whether learner copies B1 too exactly or diverges only in losing states.
```

Question:

```text
Given this audit plus exact 0.50 paired B1 results, what specific state/action diagnostic would most quickly distinguish:
A. true B1 clone / same policy;
B. stochastic same-family but tactically equivalent policy;
C. seat-pair eval artifact hiding real progress;
D. stale loading bug where B1/current accidentally load same weights;
E. B1 is a local equilibrium for this small model/surface;
F. credit assignment cannot discover rare B1-beating deviations.
```

## Same-Surface Results To Compare

Please reason from these as relative local evidence, not final throughput claims.

### Previous Best Before Typed-Pool Continuation

Run:

```text
runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426
```

Checkpoint:

```text
runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/training/checkpoints/checkpoint_480.pt
```

32-pair true-anchor confirm:

```text
weighted aggregate:   0.787500
unweighted aggregate: 0.893750
B1:                   0.50
B3:                   0.96875
B4:                   1.0
```

Older manual confirm on a different opponent set had:

```text
weighted aggregate: 0.722039
unweighted:         0.825521
B1:                 0.50
B2:                 1.0
B3:                 0.953125
B4:                 1.0
Previous recent:    0.50
```

Important caveat:

```text
The old Previous recent was often B1-like seed history, so the old aggregate double-counted B1-style pressure.
The true-anchor comparison is more relevant after the repair.
```

### Fixed-Topology u500

Run:

```text
runs/league_typedpools_seedquarantine_u480_to_u500_20260427
```

Periodic 8-pair eval:

```text
aggregate: 0.8000
B1:        0.50
B2:        1.0
B3:        1.0
B4:        1.0
```

But:

```text
league_effective_update=490
pfsp_sampling_ready=0
```

So this does not prove PFSP helps.

### Fixed-Topology u540

Run:

```text
runs/league_typedpools_seedquarantine_u500_to_u540_20260427
```

Periodic 8-pair eval:

```text
aggregate: 0.8000
B1:        0.50
B2:        1.0
B3:        1.0
B4:        1.0
champion:  policy_000014 promoted
```

32-pair confirm:

```text
weighted aggregate:   0.768750
unweighted aggregate: 0.884375
B1:                   0.50
B2:                   1.0
B3:                   0.921875
B4:                   1.0
```

B1 seat diagnostics:

```text
current train_u540_p14:
  total wins: 32/64
  wins as seat0: 14/32
  wins as seat1: 18/32

B1 baseline:
  total wins: 32/64
  wins as seat0: 14/32
  wins as seat1: 18/32
```

Interpretation we currently hold:

```text
The topology repair was real.
The quality improvement was not real.
B1 remains exactly mirrored.
u540 champion is a real local champion only under the noisy small eval/gate, not under larger thesis-grade evidence.
```

## Things Already Tried That Did Not Break B1 0.50

Across earlier pass-3 attempts, the following did not produce reliable B1 improvement:

```text
normal LR continuation after warmup sampler fix:
  worse aggregate, B3 dropped

stronger direct B1 opponent pressure:
  B1 stayed 0.50, B3/B4 hurt

very-low LR from later u480 point:
  looked okay on 8-pair, cooled on 32-pair

PFSP handoff:
  active mechanically, hurt B3, did not improve B1

PFSP with seed quarantine / true local recents:
  active semantically, still B1 0.50, u540 worse than u480

ref-off / anti-clone / no-seed warmup / higher entropy variants:
  did not reliably move B1 off 0.50

B1 seat1 reward / positive advantage / reward shaping variants:
  did not produce a convincing B1 breakthrough
```

Question:

```text
Given this negative set, what class of intervention remains most promising?
Please avoid recommending another tiny coefficient tweak unless it is paired with a diagnostic that can falsify a hypothesis quickly.
```

## Suspected Design Issue: Champion vs Recent vs Candidate

The current system still has these concepts:

```text
snapshots: append-only-ish registry entries
champion_snapshots: active champion IDs
rejected_snapshots: rejected IDs
source_kind: local, league_import, seed_import, baseline_anchor, legacy
```

It does not yet have explicit statuses:

```text
candidate
admitted_recent
frontier_champion
b1_exploiter_champion
seed_history
baseline
diagnostic_only
hard_negative
```

Potential problem:

```text
The checkpoint guard / promotion gate can reject exploratory candidates,
but the league might need useful-but-not-main-champion policies as hard negatives.
The main best checkpoint and the live opponent pool are still too coupled.
```

Please assess this architecture:

```text
Should a candidate that preserves B1 parity and offers some diversity be admitted to recent even if it is not a champion?
Should a candidate that hurts aggregate but beats B1 from seat1 be admitted as a B1 exploiter hard negative?
Should champion promotion be thesis-claim strict, while recent admission is looser?
Should rejected-but-diverse policies remain available for diagnostics/hard-negative lanes?
```

## Suspected Design Issue: Exact 0.50 Against B1

The exact symmetry is so persistent that we need to rule out implementation artifacts.

Please propose checks for:

```text
1. accidental same checkpoint loaded for current and B1 baseline;
2. B1 baseline alias resolving to current policy after resume;
3. eval runner policy cache sharing model object/state between focal and B1;
4. paired seeds forcing deterministic one-win-one-loss splits;
5. seat assignment or focal/opponent labeling bug in summary;
6. action selection mode making both policies effectively same due public heuristic logit bias;
7. stochastic sampling plus paired seed schedule mathematically expected to average exactly 0.50 for near-clones;
8. reward/value target not distinguishing B1-beating decisions;
9. legal-action/action-index mismatch that makes within-family deviations meaningless;
10. hidden state reset / recurrent state issue that makes policy adaptation impossible or identical across opponents.
```

We are especially interested in low-cost local tests:

```text
hash current and B1 loaded model weights in eval runner;
log object ids / policy ids / snapshot paths;
run current-vs-current, B1-vs-B1, current-vs-B1, B1-vs-current on identical seed schedule;
run unpaired/shuffled-seat B1 eval;
force greedy focal action mode;
evaluate against B1 u300/u400/u450 separately;
evaluate current checkpoint as both focal and opponent under swapped labels;
compare action traces on the same public states;
compare logits KL and top-k families per decision boundary;
```

## What We Need You To Output Now

Please be direct and actionable. We do not need reassurance; we need leverage.

Please answer these in order:

1. **Post-fix topology verdict**
   - Is the repaired topology probably correct enough to move on?
   - What remaining topology/code concern would you inspect first?

2. **B1 0.50 artifact triage**
   - Rank the most likely artifact explanations.
   - Give the fastest tests to eliminate them.

3. **B1 best-response plan**
   - If artifacts are ruled out, design the next implementation.
   - Prefer a concrete B1 exploiter/counterfactual/action-search plan over coefficient tweaking.

4. **League redesign plan**
   - How should candidate/admitted-recent/champion/hard-negative roles be separated?
   - What selectors/registry fields should be added next?

5. **Promotion/eval redesign**
   - What should be in the promotion gate?
   - Should 8-pair eval ever promote champions?
   - What B1 seat/pair diagnostics should be required before a champion is thesis-relevant?

6. **Minimal local experiment matrix**
   - Give 3 to 5 short local experiments with commands conceptually described, expected evidence, and stop criteria.
   - Include at least one artifact-check experiment and one actual learning experiment.

7. **Server scaling gate**
   - What exact evidence must exist before running a serious L40 multi-GPU pilot?

8. **If you think B1 may be a true local equilibrium**
   - Say what evidence would convince you.
   - Say what change would be required to break it: larger model, search/planning, counterfactual labels, richer reward, or simulator/eval changes.

The most useful answer would tell Codex exactly what to implement next and how to know within a short run whether it is promising.
