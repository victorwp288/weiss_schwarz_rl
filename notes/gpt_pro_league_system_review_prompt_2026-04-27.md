# GPT Pro Review Prompt: Weiss Schwarz RL League System Not Escaping B1 Parity

You are reviewing a multi-day rescue/optimization effort for a Weiss Schwarz reinforcement-learning thesis project.

Assume you cannot inspect the repository. This prompt is intentionally long and includes code/config excerpts, run names, observed metrics, and suspected failure modes. Please reason from the material here as if it were the only evidence available. If a conclusion depends on seeing missing code, say exactly what code path or artifact you would request next.

Repository/workspace:

```text
C:\Users\Bruger\Desktop\this one\weiss_schwarz_rl
```

Current date:

```text
2026-04-27
```

## What We Hoped The League Would Do

The simple thesis intuition was:

1. Build a useful no-league B1 anchor.
2. Initialize or train a league/self-play model from that anchor.
3. Let league/self-play run with enough compute.
4. The system should keep producing stronger opponents, exploiters, champions, and recents.
5. Given enough compute, this should trend toward a very strong or potentially superhuman Weiss Schwarz agent.

We do not need a superhuman proof locally. But we do need the league system to visibly learn, trend upward, and become clearly better than the strong B1 anchor.

That has not happened yet.

## High-Level Problem

The current best league model is strong against heuristic anchors, but remains pinned at exactly `0.50` against the B1 no-league anchor across many experiments.

This is suspicious. It may mean:

- the model is truly at a B1 local equilibrium;
- the eval surface is hiding progress through paired-seat symmetry;
- the model is cloned/regularized too tightly to B1;
- the league opponent pool is not actually evolving;
- recent/champion aliases are not resolving to meaningful league opponents;
- promotion/PFSP is not sampling the opponents we think it is;
- there is a deeper simulator/eval/action pathology.

Recent investigation found a serious league-history issue: snapshots were being written, but continuation runs were mostly rebuilding the opponent pool from the imported B1 seed history instead of carrying forward true local league recents/champions. So "Previous recent snapshot" was often an imported B1-like seed snapshot, not a newly learned league opponent.

## Current Best Confirmed Model

Current best local small-model league checkpoint:

```text
runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/training/checkpoints/checkpoint_480.pt
```

Config:

```text
configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_b1init_devtargetsharp_actorparity3_shift450_vlowlr_frontierweighted_seedpool_lag10_evalgated_lowlr_evalguard_localpromo.yaml
```

Best same-surface 32-pair true-anchor confirm:

```text
runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/eval/dev_eval_trueanchors_manual32_20260427/update_480/summary.json
```

Metrics:

```text
weighted aggregate:   0.787500
unweighted aggregate: 0.893750
B0 RandomLegal:       1.0
B1 NoLeague baseline: 0.50
B2 HeuristicPublic:   1.0
B3 HeuristicAggro:    0.96875
B4 HeuristicControl:  1.0
```

This is much better than older/fresh league attempts, but it is not thesis-satisfying because B1 remains exactly `0.50`.

## Important Earlier Baselines

Strong B1 no-league anchor source:

```text
runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425
runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425/training/checkpoints/checkpoint_450.pt
```

Fresh-student league best-ish prior:

```text
runs/league_seedpool_lag10_evalgated_devtargetsharp_u200_to_u220_pfsp_evalguard_20260426
```

Large confirm was only around:

```text
weighted:   ~0.6266
unweighted: ~0.7734
B1:         ~0.4688
B3:         ~0.7031
B4:         1.0
recent:     ~0.4688
```

B1-initialized bridge was a major improvement over fresh-student league, but still did not break B1 parity.

## Key Code Areas

Snapshot registry:

```text
python/weiss_rl/league/registry.py
```

Training driver, snapshot import, promotion, periodic dev eval:

```text
python/scripts/train.py
```

Relevant functions in `train.py`:

```text
_resolve_symbolic_promotion_anchor_policy_id
_true_local_recent_snapshot_ids
_resolve_promotion_anchor_policy_ids
_resolve_periodic_dev_eval_opponent_specs
_import_seed_snapshot_pool
_import_resume_league_snapshot_pool
_process_completed_promotion_gate
```

Runtime/PFSP opponent pool:

```text
python/weiss_rl/runtime.py
```

Relevant runtime functions:

```text
refresh_opponent_pool
_exclude_seed_import_snapshots_from_pfsp_active
_promotion_gated_recent_reservoir_size
```

Eval policy resolution:

```text
python/weiss_rl/eval/simulator_runner.py
python/weiss_rl/eval/policy_set.py
```

Manual larger scalar eval helper:

```text
python/scripts/manual_dev_eval_confirm.py
```

B1/B2 disagreement audit helper:

```text
python/scripts/b2_disagreement_audit.py
```

## Self-Contained Code Packet

This section includes the main code surfaces that may be causing the league to stagnate. Please audit them as if you were reviewing a design document plus code excerpt.

### Snapshot Registry Data Model

File:

```text
python/weiss_rl/league/registry.py
```

Relevant excerpt:

```python
@dataclass(frozen=True, slots=True)
class SnapshotMeta:
    policy_id: str
    update: int
    weights_sha256: str
    path: str
    created_utc: str = field(default_factory=_now_utc_iso)
    source_kind: str = "local"

    def sort_key(self) -> tuple[int, str]:
        return (int(self.update), str(self.policy_id))


@dataclass(slots=True)
class SnapshotRegistry:
    """Durable snapshot registry with stable ordering and champion tracking."""

    recent_size: int = 24
    champion_size: int = 4
    snapshots: list[SnapshotMeta] = field(default_factory=list)
    champion_snapshots: list[str] = field(default_factory=list)
    pinned_snapshots: list[str] = field(default_factory=list)
    rejected_snapshots: list[str] = field(default_factory=list)

    def latest(self, n: int = 1) -> list[SnapshotMeta]:
        n = int(n)
        if n <= 0:
            return []
        self.normalize()
        return self.snapshots[-n:]

    def latest_ids(self, n: int = 1) -> list[str]:
        return [snapshot.policy_id for snapshot in self.latest(n)]

    def latest_champions(
        self,
        n: int = 1,
        *,
        current_update: int | None = None,
        max_age_updates: int | None = None,
    ) -> list[str]:
        n = int(n)
        if n <= 0:
            return []
        self.normalize()
        champion_ids = list(self.champion_snapshots)
        if current_update is not None and max_age_updates is not None and int(max_age_updates) > 0:
            updates_by_policy = self._updates_by_policy()
            current_update_i = int(current_update)
            max_age_updates_i = int(max_age_updates)
            champion_ids = [
                snapshot_id
                for snapshot_id in champion_ids
                if (current_update_i - int(updates_by_policy.get(snapshot_id, current_update_i))) <= max_age_updates_i
            ]
        return champion_ids[-n:]

    def add_champion(self, snapshot_id: str) -> None:
        normalized_snapshot_id = self._require_existing_snapshot_id(snapshot_id)
        self.rejected_snapshots = [
            existing for existing in self.rejected_snapshots if existing != normalized_snapshot_id
        ]
        self.champion_snapshots = self._move_ref_to_end(self.champion_snapshots, normalized_snapshot_id)
        self.normalize()

    def reject_snapshot(self, snapshot_id: str) -> None:
        normalized_snapshot_id = self._require_existing_snapshot_id(snapshot_id)
        self.rejected_snapshots = self._move_ref_to_end(self.rejected_snapshots, normalized_snapshot_id)
        self.normalize()

    def add_snapshot(
        self,
        *,
        policy_id: str,
        update: int,
        weights_sha256: str,
        path: str,
        source_kind: str = "local",
    ) -> None:
        ...
```

Important design detail:

```text
SnapshotRegistry.latest_ids(n) is purely the latest n snapshots by update/policy_id after normalize().
It does not intrinsically know whether a snapshot is:
  - imported B1 seed history;
  - true local league history;
  - a baseline anchor;
  - rejected;
  - champion.
Callers must filter by policy_id/source_kind/rejected/champion state.
```

Review question:

```text
Is this one-list registry too ambiguous for a thesis-grade league? Should "recent", "seed history", "baseline anchors", "champions", "challengers", "exploiters", and "rejected candidates" be separate typed pools instead of one sorted snapshot list plus filters?
```

### Symbolic Recent/Champion Eval Resolution

File:

```text
python/scripts/train.py
```

Relevant excerpt:

```python
def _resolve_symbolic_promotion_anchor_policy_id(
    anchor_name: str,
    *,
    registry: SnapshotRegistry,
    promotion_gate_enabled: bool = False,
) -> str | None:
    if anchor_name == "Latest champion snapshot":
        champion_ids = registry.latest_champions(1)
        return None if not champion_ids else str(champion_ids[-1])
    if anchor_name == "Previous champion snapshot":
        champion_ids = registry.latest_champions(2)
        return None if len(champion_ids) < 2 else str(champion_ids[-2])
    if anchor_name == "Latest recent snapshot":
        recent_ids = _true_local_recent_snapshot_ids(registry, promotion_gate_enabled=promotion_gate_enabled)
        return None if not recent_ids else str(recent_ids[-1])
    if anchor_name == "Previous recent snapshot":
        recent_ids = _true_local_recent_snapshot_ids(registry, promotion_gate_enabled=promotion_gate_enabled)
        return None if len(recent_ids) < 2 else str(recent_ids[-2])
    return None


def _true_local_recent_snapshot_ids(
    registry: SnapshotRegistry,
    *,
    promotion_gate_enabled: bool = False,
) -> tuple[str, ...]:
    registry.normalize()
    rejected_ids = set(getattr(registry, "rejected_snapshots", ()))
    champion_ids = set(getattr(registry, "champion_snapshots", ())) if promotion_gate_enabled else set()
    recent_ids: list[str] = []
    for snapshot in registry.snapshots:
        policy_id = str(snapshot.policy_id).strip()
        source_kind = str(getattr(snapshot, "source_kind", "local")).strip().lower()
        if (
            not policy_id
            or policy_id in rejected_ids
            or policy_id in _FIXED_OPPONENT_EXCLUSIONS
            or policy_id.startswith("seed_")
            or source_kind in {"seed_import", "baseline_anchor"}
        ):
            continue
        if promotion_gate_enabled and policy_id not in champion_ids:
            continue
        recent_ids.append(policy_id)
    return tuple(recent_ids)
```

Key concern:

```text
When promotion_gate_enabled=True, "Previous recent snapshot" only sees champion snapshots.
This avoids treating failed candidates as official recents, but it may also make recent/champion eval disappear before the first promotion.
It may starve diagnostics during exactly the phase where we need to know whether true local snapshots are improving.
```

Please critique:

```text
Should "previous recent" during warmup mean the previous true local snapshot even if not champion?
Should "previous champion" remain champion-only?
Should eval report both:
  - Previous local candidate/recent;
  - Previous promoted champion;
  - Previous imported seed/history snapshot;
instead of overloading one alias?
```

### Seed Snapshot Import

File:

```text
python/scripts/train.py
```

Relevant excerpt:

```python
def _seed_snapshot_policy_id(*, source_run_dir: Path, source_policy_id: str) -> str:
    source_hash = hashlib.sha1(source_run_dir.as_posix().encode("utf-8")).hexdigest()[:10]
    safe_policy_id = str(source_policy_id).replace("/", "_").replace("\\", "_").strip()
    return f"seed_{source_hash}_{safe_policy_id}"


def _import_seed_snapshot_pool(
    *,
    stack: StackConfig,
    training_paths: TrainingPaths,
    run_dir: Path,
    seed_snapshot_run_dir: Path,
    max_update: int | None = None,
    exclude_source_policy_ids: Sequence[str] = (),
    expected_model_state_dict: dict[str, Any],
    expected_config_canonical: dict[str, Any] | None,
    expected_spec_hash256: str | None,
) -> list[str]:
    source_run_dir = Path(seed_snapshot_run_dir).resolve()
    source_layout = ArtifactLayout.from_run_dir(source_run_dir)
    source_registry_path = source_layout.training_snapshots_dir / REGISTRY_FILENAME
    source_registry = SnapshotRegistry.load(source_registry_path)
    excluded_source_policy_ids = {str(policy_id).strip() for policy_id in exclude_source_policy_ids}
    source_snapshots = [
        snapshot
        for snapshot in source_registry.snapshots
        if snapshot.policy_id not in _promotion_anchor_policy_id_candidates(_PROMOTION_GATE_NOLEAGUE_BASELINE_NAME)
        and snapshot.policy_id not in excluded_source_policy_ids
        and (max_update is None or int(snapshot.update) <= int(max_update))
    ]
    ...
    source_champions = set(source_registry.champion_snapshots)
    ...
    for source_snapshot in source_snapshots:
        imported_policy_id = _seed_snapshot_policy_id(
            source_run_dir=source_run_dir,
            source_policy_id=source_snapshot.policy_id,
        )
        ...
        imported_payload["policy_id"] = imported_policy_id
        imported_payload["imported_from_run_dir"] = source_run_dir.as_posix()
        imported_payload["imported_from_policy_id"] = source_snapshot.policy_id
        imported_payload["imported_from_snapshot_path"] = source_snapshot.path
        imported_payload["seeded_from_external_registry"] = True
        ...
        registry.add_snapshot(
            policy_id=imported_policy_id,
            update=int(source_snapshot.update),
            weights_sha256=weights_sha256,
            path=weights_path.relative_to(run_dir).as_posix(),
            source_kind="seed_import",
        )
        if source_snapshot.policy_id in source_champions:
            registry.add_champion(imported_policy_id)
```

Important:

```text
Seed imports get prefixed IDs like seed_315d5e55ce_policy_000009.
They are marked source_kind="seed_import".
If the source snapshot was a champion in its source run, this import code can add the seed import to champion_snapshots.
```

Review questions:

```text
Should imported seed snapshots ever become champions in the new run, or should imported champion status be stored separately?
Could seed-import champions pollute "Previous champion snapshot" and PFSP champion sampling?
If seed imports are only curriculum ballast, should they be excluded from champion_snapshots entirely and represented as seed/history lane metadata?
```

### Resume League Snapshot Import

This was added after discovering continuation runs were losing true local league history.

File:

```text
python/scripts/train.py
```

Relevant excerpt:

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


def _import_resume_league_snapshot_pool(
    *,
    stack: StackConfig,
    training_paths: TrainingPaths,
    run_dir: Path,
    resume_checkpoint_path: Path,
    max_update: int,
    expected_model_state_dict: dict[str, Any],
) -> list[str]:
    source_run_dir = _infer_run_dir_from_checkpoint_path(resume_checkpoint_path)
    if source_run_dir is None or source_run_dir.resolve() == Path(run_dir).resolve():
        return []
    source_layout = ArtifactLayout.from_run_dir(source_run_dir)
    source_registry_path = source_layout.training_snapshots_dir / REGISTRY_FILENAME
    if not source_registry_path.is_file():
        return []
    source_registry = SnapshotRegistry.load(source_registry_path)
    rejected_policy_ids = set(getattr(source_registry, "rejected_snapshots", ()))
    source_snapshots = [
        snapshot
        for snapshot in source_registry.snapshots
        if int(snapshot.update) <= int(max_update)
        and _source_snapshot_is_resume_league_snapshot(snapshot, rejected_policy_ids=rejected_policy_ids)
    ]
    if not source_snapshots:
        return []

    registry_path = training_paths.snapshots_dir / REGISTRY_FILENAME
    registry = SnapshotRegistry.load(registry_path)
    _sync_snapshot_registry_retention(stack, registry)
    existing_policy_ids = {snapshot.policy_id for snapshot in registry.snapshots}
    source_champions = set(source_registry.champion_snapshots)
    imported_policy_ids: list[str] = []
    for source_snapshot in source_snapshots:
        policy_id = str(source_snapshot.policy_id)
        if policy_id in existing_policy_ids:
            imported_policy_ids.append(policy_id)
            continue
        source_weights_path = source_run_dir / source_snapshot.path
        payload = torch.load(source_weights_path, map_location="cpu", weights_only=True)
        _validate_snapshot_tensor_contract(
            label="Imported resume league snapshot",
            source_path=source_weights_path,
            payload=payload,
            expected_model_state_dict=expected_model_state_dict,
        )
        snapshot_dir = training_paths.snapshots_dir / policy_id
        weights_path = snapshot_dir / SNAPSHOT_WEIGHTS_FILENAME
        imported_payload = dict(payload)
        imported_payload["policy_id"] = policy_id
        imported_payload["imported_from_run_dir"] = source_run_dir.as_posix()
        imported_payload["imported_from_policy_id"] = source_snapshot.policy_id
        imported_payload["imported_from_snapshot_path"] = source_snapshot.path
        imported_payload["resumed_from_league_registry"] = True
        torch.save(imported_payload, weights_path)
        ...
        registry.add_snapshot(
            policy_id=policy_id,
            update=int(source_snapshot.update),
            weights_sha256=weights_sha256,
            path=weights_path.relative_to(run_dir).as_posix(),
            source_kind="league_import",
        )
        if source_snapshot.policy_id in source_champions:
            registry.add_champion(policy_id)
```

Intent:

```text
Resumed continuation runs should preserve true learned league recents/champions from the source run.
These are not seed-prefixed because they are considered the actual continuation history.
They are source_kind="league_import".
```

Review questions:

```text
Is preserving policy IDs like policy_000011 across runs safe, or should the imported IDs be namespaced?
Should a continuation instead copy the entire registry exactly and append new run-local snapshots, rather than rebuilding/importing selected entries?
Could preserving policy IDs collide with future policy_000011 if the snapshot counter is local to the new run?
Should rejected snapshots from the source run remain rejected, be imported as rejected, or be omitted?
Should source champion membership be preserved across continuation if the new run changes eval/promotion criteria?
```

### Training Startup Order

File:

```text
python/scripts/train.py
```

Relevant excerpt:

```python
_ensure_noleague_baseline_snapshot(...)
distributed_barrier(ddp_context)

_attach_reference_policy_model_if_configured(...)

imported_resume_league_policy_ids: tuple[str, ...] = ()
if rank0 and resume_state is not None:
    imported_resume_league_policy_ids = tuple(
        _import_resume_league_snapshot_pool(
            stack=stack,
            training_paths=training_paths,
            run_dir=artifacts.run_dir,
            resume_checkpoint_path=resume_state.checkpoint_path,
            max_update=int(resume_state.update_count),
            expected_model_state_dict=learner.model.state_dict(),
        )
    )

seed_snapshot_max_update = _seed_snapshot_import_max_update(...)
if rank0 and seed_snapshot_run_dir is not None:
    _import_seed_snapshot_pool(
        stack=stack,
        training_paths=training_paths,
        run_dir=artifacts.run_dir,
        seed_snapshot_run_dir=seed_snapshot_run_dir,
        max_update=seed_snapshot_max_update,
        exclude_source_policy_ids=imported_resume_league_policy_ids,
        expected_model_state_dict=learner.model.state_dict(),
        expected_config_canonical=canonical_config_dict(stack),
        expected_spec_hash256=spec_hash256,
    )

if seed_snapshot_run_dir is not None or resume_state is not None:
    distributed_barrier(ddp_context)

runtime_config = build_runtime_config(...)
```

Intent:

```text
1. Ensure B1 baseline anchor exists.
2. Attach frozen/reference policy if configured.
3. Import true resume league history.
4. Import seed snapshot history, excluding any source IDs already imported as true league history.
5. Barrier before runtime reads registry on distributed workers.
6. Build runtime/PFSP pool.
```

Review question:

```text
Is this order correct for multi-GPU DDP? Are there hidden rank/race/artifact-cleanliness risks? Should reference attach occur before or after snapshot import? Should seed import contract compare config hashes if the source run is intentionally from B1 no-league?
```

### Runtime PFSP Pool Refresh

File:

```text
python/weiss_rl/runtime.py
```

Relevant excerpt:

```python
def _exclude_seed_import_snapshots_from_pfsp_active(self, *, current_update: int | None = None) -> bool:
    if self._league_config is None:
        return False
    sampling_cfg = getattr(self._league_config, "sampling", self._league_config)
    if not bool(getattr(sampling_cfg, "exclude_seed_snapshots_from_pfsp", False)):
        return False
    warmup_cfg = getattr(self._league_config, "warmup", None)
    warmup_updates = int(getattr(warmup_cfg, "first_updates", getattr(self._league_config, "warmup_first_updates", 0)))
    reference_update = self._league_reference_update() if current_update is None else int(current_update)
    if reference_update < warmup_updates:
        return False
    return not self._league_eval_warmup_gate_blocks_pfsp()


def refresh_opponent_pool(self) -> None:
    ...
    current_update = int(self._league_reference_update())
    registry = SnapshotRegistry.load(self._registry_path)
    snapshots_by_id = {snapshot.policy_id: snapshot for snapshot in registry.snapshots}
    exclude_seed_imports_from_pfsp = self._exclude_seed_import_snapshots_from_pfsp_active(
        current_update=current_update
    )
    admitted_champion_ids = tuple(
        registry.latest_champions(
            int(self._league_config.snapshot_pool_champion_size),
            current_update=current_update,
            max_age_updates=max_age_updates,
        )
    )
    recent_size = int(self._league_config.snapshot_pool_recent_size)
    if bool(self._league_config.promotion_gate_enabled):
        recent_size = self._promotion_gated_recent_reservoir_size(
            base_recent_size=recent_size,
            champion_size=int(self._league_config.snapshot_pool_champion_size),
            admitted_champion_ids=admitted_champion_ids,
        )
    sampler = OpponentPoolSampler(...)
    champion_ids = tuple(
        policy_id
        for policy_id in admitted_champion_ids
        if policy_id not in _FIXED_OPPONENT_EXCLUSIONS
        and not (
            exclude_seed_imports_from_pfsp
            and _snapshot_meta_is_seed_import(snapshots_by_id.get(policy_id), policy_id=policy_id)
        )
    )
    rejected_ids = set(getattr(registry, "rejected_snapshots", ())) if bool(self._league_config.promotion_gate_enabled) else set()
    recent_query_size = int(recent_size)
    if recent_query_size > 0:
        recent_query_size = max(
            recent_query_size,
            int(recent_size) + len(champion_ids) + len(rejected_ids) + len(_FIXED_OPPONENT_EXCLUSIONS),
        )
    recent_ids = tuple(
        policy_id
        for policy_id in registry.latest_ids(recent_query_size)
        if policy_id not in _FIXED_OPPONENT_EXCLUSIONS and policy_id not in rejected_ids
        and not (
            exclude_seed_imports_from_pfsp
            and _snapshot_meta_is_seed_import(snapshots_by_id.get(policy_id), policy_id=policy_id)
        )
    )
    candidate_ids = tuple(dict.fromkeys([*champion_ids, *recent_ids]))
    filtered_candidate_ids = self._filter_timeout_heavy_opponents(candidate_ids)
    candidate_ids, quarantined_count = self._apply_opponent_pool_diversity_floor(...)
    hard_negative_ids = self._select_hard_negative_ids(candidate_ids)
    hard_negative_set = set(hard_negative_ids)
    champion_ids = tuple(
        policy_id for policy_id in champion_ids if policy_id in candidate_ids and policy_id not in hard_negative_set
    )
    champion_set = set(champion_ids)
    eligible_recent_ids = tuple(
        policy_id
        for policy_id in recent_ids
        if policy_id in candidate_ids and policy_id not in hard_negative_set
    )
    non_champion_recent_ids = tuple(policy_id for policy_id in eligible_recent_ids if policy_id not in champion_set)
    champion_recent_backfill_ids = tuple(policy_id for policy_id in eligible_recent_ids if policy_id in champion_set)
    if int(recent_size) <= 0:
        recent_ids = ()
    elif len(non_champion_recent_ids) >= int(recent_size) or not champion_recent_backfill_ids:
        recent_ids = non_champion_recent_ids[-int(recent_size) :]
    else:
        missing_recent_ids = max(0, int(recent_size) - len(non_champion_recent_ids))
        recent_ids = tuple(
            dict.fromkeys(
                [
                    *non_champion_recent_ids,
                    *champion_recent_backfill_ids[-missing_recent_ids:],
                ]
            )
        )
    candidate_ids = tuple(dict.fromkeys([*hard_negative_ids, *champion_ids, *recent_ids]))
    self._opponent_candidate_ids = candidate_ids
    self._pfsp_pool_size = len(candidate_ids)
    self._opponent_champion_ids = champion_ids
    self._opponent_recent_ids = recent_ids
    self._opponent_hard_negative_ids = hard_negative_ids
    self._pfsp_champion_pool_size = len(champion_ids)
    self._pfsp_recent_pool_size = len(recent_ids)
    self._pfsp_hard_negative_pool_size = len(hard_negative_ids)
```

Important behavior:

```text
exclude_seed_snapshots_from_pfsp only activates after:
  - config league.sampling.exclude_seed_snapshots_from_pfsp is true;
  - reference update >= league.warmup.first_updates;
  - eval warmup gate no longer blocks PFSP.

Before that, imported seed snapshots can still appear in the PFSP-style/recent candidate set.
After that, seed imports should be excluded from champion_ids and recent_ids.
```

Known historical issue:

```text
Before this fix, PFSP recent pool in the PFSP500 handoff was dominated by imported B1 seed snapshots.
Metrics showed pfsp_sampling_ready=1, recent weight=0.40, pfsp_recent_pool_size=7, but those 7 recents were seed_import B1-history.
```

Review questions:

```text
Is it correct to keep seed imports in warmup and exclude them only after PFSP opens?
Should seed imports be excluded from PFSP/recent even during warmup once a local league snapshot exists?
Does recent_query_size look robust enough when many excluded/rejected/fixed snapshots are interleaved?
If promotion_gate_enabled shrinks recent_size before any champion exists, does the league get enough true local history pressure?
Should hard_negative sampling include B1/baseline anchors, seed imports, local recents, or only true league snapshots?
```

## Self-Contained Config Packet

The config hierarchy is long. These are the parts most relevant to the B1-init league bridge.

### Current Best B1-Initialized Very-Low-LR Config

File:

```text
configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_b1init_devtargetsharp_actorparity3_shift450_vlowlr_frontierweighted_seedpool_lag10_evalgated_lowlr_evalguard_localpromo.yaml
```

Excerpt:

```yaml
schema_version: 2
description: B1-initialized league bridge with the fixed warmup sampler and very-low learning rate for post-anchor polishing
extends: structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_b1init_devtargetsharp_actorparity3_shift450_frontierweighted_seedpool_lag10_evalgated_lowlr_evalguard_localpromo.yaml
training:
  optimizer:
    learning_rate: 0.000005
league:
  sampling:
    exclude_seed_snapshots_from_pfsp: true
```

Meaning:

```text
This is the current best branch.
It uses LR=5e-6 after initializing/resuming from the strong B1 checkpoint.
It now excludes seed imports from true PFSP after warmup/eval gate.
```

### Parent Shifted B1-Init Config

File:

```text
configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_b1init_devtargetsharp_actorparity3_shift450_frontierweighted_seedpool_lag10_evalgated_lowlr_evalguard_localpromo.yaml
```

Excerpt:

```yaml
schema_version: 2
description: league bridge initialized from the strong B1 checkpoint with guidance schedules shifted to the B1 resume update
extends: structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_freshstudent_refb1_devtargetsharp_actorparity3_frontierweighted_seedpool_lag10_evalgated_lowlr_evalguard_localpromo.yaml
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

Key concern:

```text
This keeps reference BC and heuristic teacher guidance active after the B1 resume update.
The model may be regularized into a B1-family clone.
However, turning off/refading variants so far did not break B1 parity.
```

### Base League/PFSP Defaults From Locked Thesis Preset

File:

```text
configs/presets/typed_thesis_locked.yaml
```

Excerpt:

```yaml
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
```

Local benchmark configs override seed counts and warmup lengths heavily.

### Local Promotion Config

File:

```text
configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_localpromo.yaml
```

Excerpt:

```yaml
schema_version: 2
description: reduced-size B1-anchored league preset with internally consistent local promotion seeds
extends: structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark.yaml
seed_sets:
  dev_eval: configs/seeds/local_dev_eval_seeds.txt
  promotion_gate: configs/seeds/local_promotion_eval_seeds.txt
  report_eval: configs/seeds/report_eval_seeds.txt
league:
  promotion:
    paired_seeds: 8
    seed_file: configs/seeds/local_promotion_eval_seeds.txt
    gate:
      parallel_workers: 6
evaluation:
  seed_files:
    dev_eval: configs/seeds/local_dev_eval_seeds.txt
    promotion_gate: configs/seeds/local_promotion_eval_seeds.txt
    report_eval: configs/seeds/report_eval_seeds.txt
  periodic_dev_eval_interval_updates: 0
  periodic_dev_eval_paired_seeds: 8
```

Concern:

```text
Local promotion uses only 8 paired seeds unless manually confirmed.
That is too noisy for thesis claims, but useful for quick pass/fail.
The B1 score staying exactly 0.50 across 32/64 paired games is more suspicious than a single noisy 8-pair result.
```

### Recent B1-Exploit Gate Variant

File:

```text
configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_b1init_devtargetsharp_actorparity3_shift450_vlowlr_noseedpfsp_b1exploitgate_frontierweighted_seedpool_lag10_evalgated_lowlr_evalguard_localpromo.yaml
```

Excerpt:

```yaml
schema_version: 2
description: very-low-LR B1-init bridge that blocks true recent/PFSP and promotion unless scalar dev eval shows a real B1 exploit
extends: structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_b1init_devtargetsharp_actorparity3_shift450_vlowlr_noseedpfsp_frontierweighted_seedpool_lag10_evalgated_lowlr_evalguard_localpromo.yaml
league:
  warmup:
    eval_gate_min_anchor_scores:
      B1 NoLeague baseline: 0.5625
      B3 HeuristicPublicAggro: 0.90
  promotion:
    gate:
      target_min_anchor_scores:
        B1 NoLeague baseline: 0.5625
        B3 HeuristicPublicAggro: 0.90
```

Concern:

```text
This gate makes intuitive sense if we only want true exploiters to become champions.
But if B1 is locked at 0.50 because of paired-seat symmetry or eval design, this may prevent any champion ladder forever.
```

### No Seed Warmup / Higher Entropy Variant

File:

```text
configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_b1init_devtargetsharp_actorparity3_shift450_vlowlr_noseedpfsp_b1exploitgate_noseedwarm_entropy08_frontierweighted_seedpool_lag10_evalgated_lowlr_evalguard_localpromo.yaml
```

Excerpt:

```yaml
schema_version: 2
description: B1-init exploit-gated bridge with imported seed snapshots removed from warmup pressure and higher entropy for B1-deviation search
extends: structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_b1init_devtargetsharp_actorparity3_shift450_vlowlr_noseedpfsp_b1exploitgate_frontierweighted_seedpool_lag10_evalgated_lowlr_evalguard_localpromo.yaml
training:
  exploration:
    entropy_coef: 0.08
    entropy_anneal_to: 0.04
    entropy_anneal_steps_updates: 300000
league:
  sampling:
    warmup_snapshot_mix_fraction: 0.0
```

Question:

```text
Would this kind of branch be more likely to discover deviations, or is it likely to destroy the useful B1/heuristic skill without a better best-response signal?
```

## Observed Artifact Packet

### Current Best Registry Before Resume-Carry Fix

Run:

```text
runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426
```

Registry:

```text
champion_snapshots: []
snapshots:
  seed_315d5e55ce_policy_000003  update 150  source_kind seed_import
  seed_315d5e55ce_policy_000004  update 200  source_kind seed_import
  seed_315d5e55ce_policy_000005  update 250  source_kind seed_import
  seed_315d5e55ce_policy_000006  update 300  source_kind seed_import
  seed_315d5e55ce_policy_000007  update 350  source_kind seed_import
  seed_315d5e55ce_policy_000008  update 400  source_kind seed_import
  b1_noleague_baseline           update 450  source_kind baseline_anchor
  seed_315d5e55ce_policy_000009  update 450  source_kind seed_import
  policy_000011                  update 480  source_kind local
```

Small dev eval:

```text
Previous recent snapshot resolved to seed_315d5e55ce_policy_000009.
Previous recent score was 0.50.
B1 NoLeague baseline score was 0.50.
```

Interpretation:

```text
"Previous recent" was not a true learned league recent. It was imported B1-history.
This likely double-counted B1-like opponents in aggregate metrics.
```

### u480 To u500 Stable Continuation

Run:

```text
runs/league_b1init_b1warmfix_vlowlr_u480best_to_u500_20260426
```

Metrics:

```text
u500 automatic 32-pair confirm weighted aggregate: 0.722039
B1: 0.50
B3: 0.953125
B4: 1.0
Previous recent: 0.50
```

Registry:

```text
champion_snapshots: []
policy_000012 was saved but rejected by checkpoint guard.
Previous recent again resolved to seed_315d5e55ce_policy_000009.
```

Interpretation:

```text
This proved stability of the B1-init bridge but did not prove an improving league ladder.
```

### PFSP500 Handoff

Run:

```text
runs/league_b1init_b1warmfix_vlowlr_pfsp500_u500_to_u520_20260426
```

Runtime showed:

```text
pfsp_sampling_ready: 1.0
pfsp_sampling_weight_recent: 0.40
pfsp_sampling_weight_noleague_baseline: 0.20
pfsp_sampling_weight_heuristic_public: 0.20
pfsp_sampling_weight_heuristic_variants: 0.20
pfsp_recent_pool_size: 7
pfsp_champion_pool_size: 0
pfsp_recent_envs > 0
pfsp_champion_envs: 0
```

Metric:

```text
u520 weighted: 0.677632
B1: 0.50
B3: 0.8125
```

Interpretation:

```text
PFSP was technically active but likely sampled seed_import B1-history as recent.
The handoff eroded heuristic strength without producing a real B1 exploit.
```

### Resume-Carry Smoke After Fix

Run:

```text
runs/league_resume_registrycarry_smoke_u480_to_u481_20260427
```

Startup:

```text
Imported resume league snapshot pool: count=1 source_run_dir=.../league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426
Imported seeded snapshot pool: count=7 source_run_dir=.../b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425
```

Registry after smoke:

```text
seed_315d5e55ce_policy_000003  update 150  source_kind seed_import
seed_315d5e55ce_policy_000004  update 200  source_kind seed_import
seed_315d5e55ce_policy_000005  update 250  source_kind seed_import
seed_315d5e55ce_policy_000006  update 300  source_kind seed_import
seed_315d5e55ce_policy_000007  update 350  source_kind seed_import
seed_315d5e55ce_policy_000008  update 400  source_kind seed_import
b1_noleague_baseline           update 450  source_kind baseline_anchor
seed_315d5e55ce_policy_000009  update 450  source_kind seed_import
policy_000011                  update 480  source_kind league_import
policy_000012                  update 481  source_kind local
```

This seems much healthier, but it is only a smoke/correctness check, not proof of learning.

### B1 Disagreement Audit

Run:

```text
runs/league_b1init_b1warmfix_vlowlr_u480_b1_audit_allowhash_20260427/audit/summary_outcome_conditioned.json
```

Summary:

```text
64 games vs B1:
  wins: 32
  losses: 32

Seat split:
  focal seat 0: 21 wins / 11 losses
  focal seat 1: 11 wins / 21 losses

Seat1 losses:
  top-action match rate vs B1: ~0.8742
  top-family match rate vs B1: 1.0
  mean probability on B1 top action: ~0.7104

Seat1 wins:
  top-action match rate vs B1: ~0.8730
  top-family match rate vs B1: 1.0
  mean probability on B1 top action: ~0.7107

Mismatch-only counters:
  Mostly within-family differences, e.g.
    encore_decline(slot=0) vs encore_decline(slot=1)
    main_play_character(hand_index=1/2/3/5, stage_slot=...) vs hand_index=0 same stage_slot
  top_mismatched_family_pairs empty
```

Interpretation:

```text
The model is a very close B1-family clone.
The 50/50 wall looks more seat-shaped than action-family-shaped.
There was no obvious pass/main_move/family-level pathology in this audit.
```

## Suspected Failure Modes To Consider

Please evaluate these, rank them, and say what evidence would distinguish them.

```text
1. League-history bug:
   Continuation runs were not carrying true local recents/champions.
   Recent/champion lanes were empty or filled with B1 seed history.

2. Eval alias bug/design flaw:
   "Previous recent snapshot" often pointed at imported B1 history.
   Aggregate metrics were double-counting B1-like opponents.

3. Promotion-gate starvation:
   The gate rejects local candidates unless they already beat B1/B3.
   If B1 is stuck at paired 0.50, no champion ladder starts.

4. Reference/teacher cap:
   B1-init + reference BC + family BC + heuristic teacher keeps the policy inside B1's action family.
   Local RL signal may be too weak to discover a useful deviation.

5. Paired-seat symmetry:
   Against B1, the policy wins as seat 0 and loses as seat 1 in a way that averages exactly 0.50.
   This may be a game/eval property rather than no improvement.

6. B1 local equilibrium:
   B1 may be a very strong local fixed point for this model/eval surface.
   Small model/local compute may not discover exploit deviations.

7. Missing best-response mechanism:
   League self-play only reinforces same-family behavior.
   Need explicit exploiters/search/counterfactual action evaluation/MCTS/rollout teacher.

8. Reward/credit assignment issue:
   Sparse outcome reward may not assign enough signal to the decisions that beat B1.
   B1-specific reward shaping and positive advantage attempts did not break parity.

9. Snapshot/policy update bug:
   The learner may be writing snapshots, but actors/eval may be loading stale or wrong policies.
   Smoke shows registry is now better, but we still need artifact-level proof over a real continuation.

10. Model capacity / server scaling:
   Local small model may be at capacity.
   But before burning server compute, we need to know the league topology is actually sampling/evaluating true recents/champions.
```

## What We Need From You

Please do not give generic RL advice like "train longer" or "tune learning rate" unless you tie it to a concrete bug/mechanism above.

We need a structural diagnosis and a pragmatic repair plan.

Specifically, please answer:

```text
1. What is the most likely reason the league has not shown an upward trend?
2. Is the new resume-league-import + seed-PFSP-exclusion fix conceptually right?
3. What is still wrong or risky in the current registry/recent/champion design?
4. How should previous/recent/champion aliases be redesigned so they cannot silently point at B1 seed history?
5. How should promotion work if B1 remains exactly 0.50 but the policy improves against B3/B4?
6. Should B1 > 0.50 be mandatory for champion promotion, or should we promote frontier policies that preserve B1 parity and improve other anchors?
7. How do we create real best-response pressure against B1 instead of cloning B1?
8. What exact metrics/artifacts must be logged every update to prove the league pool is alive?
9. What is the best next 3-experiment sequence before any large L40 server run?
10. What code changes should be made immediately?
```

## Recent Fix: Carry Forward True League History On Resume

Problem found:

- When resuming from a prior league checkpoint, the new run imported the B1 seed snapshot pool.
- It did not reliably carry forward the prior league run's own local snapshots/champions.
- That meant a continuation run could have:
  - imported B1 seed snapshots;
  - B1 baseline anchor;
  - one newly written local candidate;
  - no meaningful prior local recent/champion ladder.

This made "recent" misleading.

New fix added:

```text
python/scripts/train.py
```

New helper:

```text
_import_resume_league_snapshot_pool(...)
```

Behavior:

- Infer source run dir from `--resume-from .../training/checkpoints/checkpoint_N.pt`.
- Load source run registry.
- Import only true league snapshots:
  - exclude fixed anchors;
  - exclude `seed_...`;
  - exclude `source_kind in {"seed_import", "baseline_anchor"}`;
  - exclude rejected snapshots;
  - respect `max_update <= resume update`.
- Preserve source policy IDs, for example `policy_000011`.
- Mark imported source kind as:

```text
source_kind: league_import
```

- Preserve champion membership when the source snapshot was a champion.
- Avoid re-importing those same source policy IDs as seed-prefixed snapshots if the seed snapshot run is auto-inferred from the same resume source.

Smoke validation run:

```text
runs/league_resume_registrycarry_smoke_u480_to_u481_20260427
```

Command shape:

```text
uv run python python/scripts/train.py \
  --stack-config configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_b1init_devtargetsharp_actorparity3_shift450_vlowlr_frontierweighted_seedpool_lag10_evalgated_lowlr_evalguard_localpromo.yaml \
  --run-label league_resume_registrycarry_smoke_u480_to_u481_20260427 \
  --runtime-mode train_async_fast \
  --autoscale \
  --hardware-profile local \
  --resume-from runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/training/checkpoints/checkpoint_480.pt \
  --resume-allow-config-mismatch \
  --resume-reset-optimizer \
  --seed-snapshot-run-dir runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425 \
  --b1-baseline-run-dir runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425 \
  --max-updates 481 \
  --checkpoint-interval-updates 1 \
  --profile-timers
```

Startup printed:

```text
Imported resume league snapshot pool: count=1 source_run_dir=.../runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426
Imported seeded snapshot pool: count=7 source_run_dir=.../runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425
```

Registry after smoke:

```text
seed_315d5e55ce_policy_000003  update 150  source_kind seed_import
seed_315d5e55ce_policy_000004  update 200  source_kind seed_import
seed_315d5e55ce_policy_000005  update 250  source_kind seed_import
seed_315d5e55ce_policy_000006  update 300  source_kind seed_import
seed_315d5e55ce_policy_000007  update 350  source_kind seed_import
seed_315d5e55ce_policy_000008  update 400  source_kind seed_import
b1_noleague_baseline           update 450  source_kind baseline_anchor
seed_315d5e55ce_policy_000009  update 450  source_kind seed_import
policy_000011                  update 480  source_kind league_import
policy_000012                  update 481  source_kind local
```

This is the desired separation.

## Recent Config Fix: Exclude Seed Snapshots From True PFSP

The main B1-init very-low-LR config now sets:

```yaml
league:
  sampling:
    exclude_seed_snapshots_from_pfsp: true
```

File:

```text
configs/presets/structured_acceptance_thesis_model_server_train_auto_gpu_b1anchored_league_benchmark_b1init_devtargetsharp_actorparity3_shift450_vlowlr_frontierweighted_seedpool_lag10_evalgated_lowlr_evalguard_localpromo.yaml
```

Intent:

- Keep imported seed snapshots available as warmup ballast before true PFSP handoff.
- After eval-gated PFSP opens, do not let imported B1-like seed snapshots dominate the recent lane.
- Let the recent/champion pool become true local league history.

Validation:

```text
uv run python -c "from weiss_rl.config import load_stack_config; s=load_stack_config('<config>'); print(s.config.league.sampling.exclude_seed_snapshots_from_pfsp)"
```

Printed:

```text
True
```

Runtime tests:

```text
uv run pytest -q \
  python/weiss_rl/tests/test_runtime.py::test_refresh_opponent_pool_can_exclude_seed_imports_after_pfsp_handoff \
  python/weiss_rl/tests/test_runtime.py::test_refresh_opponent_pool_keeps_seed_imports_before_pfsp_handoff \
  --tb=short
```

Result:

```text
2 passed
```

## Why The Previous/Recent Diagnosis Matters

Current best u480 registry before the fix:

```text
runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/training/snapshots/registry.json
```

Had:

```text
champion_snapshots: []
snapshots:
  seed_315d5e55ce_policy_000003 update 150
  seed_315d5e55ce_policy_000004 update 200
  seed_315d5e55ce_policy_000005 update 250
  seed_315d5e55ce_policy_000006 update 300
  seed_315d5e55ce_policy_000007 update 350
  seed_315d5e55ce_policy_000008 update 400
  b1_noleague_baseline update 450
  seed_315d5e55ce_policy_000009 update 450
  policy_000011 update 480
```

u480 small dev eval:

```text
runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/eval/dev_eval/update_480/summary.json
```

Had:

```text
Previous recent snapshot -> seed_315d5e55ce_policy_000009
Previous recent snapshot score -> 0.50
B1 NoLeague baseline score -> 0.50
```

So previous recent was not real league progress. It was imported B1-history.

PFSP500 handoff run:

```text
runs/league_b1init_b1warmfix_vlowlr_pfsp500_u500_to_u520_20260426
```

Tail runtime metrics showed:

```text
pfsp_sampling_ready: 1.0
pfsp_sampling_weight_recent: 0.40
pfsp_recent_pool_size: 7
pfsp_champion_pool_size: 0
pfsp_recent_envs > 0
pfsp_champion_envs: 0
```

Those 7 recents were imported seed snapshots, not true local league recents.

## Experiments That Did Not Break B1 Parity

Many probes failed to move B1 above `0.50`.

### Normal LR after sampler fix

```text
runs/league_b1init_b1warmfix_u480_to_u500_20260426
```

Result:

```text
weighted: 0.651316
B1:      0.50
B3:      0.75
```

Verdict:

```text
Too much drift; harms heuristic edge.
```

### Stronger B1 pressure

```text
runs/league_b1init_b1warmfix_strongb1_u480_to_u500_20260426
```

Result:

```text
weighted: 0.664474
B1:      0.50
B3:      0.8125
B4:      0.875
```

Verdict:

```text
More direct B1 volume did not break parity and hurt other anchors.
```

### Very-low LR u480-to-u500

```text
runs/league_b1init_b1warmfix_vlowlr_u480_to_u500_20260426
```

32-pair cooled to:

```text
weighted: ~0.685855
B1:       0.50
```

### PFSP handoff

Normal:

```text
runs/league_b1init_b1warmfix_vlowlr_u500_to_u520_20260426
```

Earlier PFSP500:

```text
runs/league_b1init_b1warmfix_vlowlr_pfsp500_u500_to_u520_20260426
```

PFSP active but result:

```text
weighted: 0.677632
B1:      0.50
B3:      0.8125
```

Now suspected cause:

```text
PFSP recent lane was dominated by imported B1 seed snapshots, not true local league recents/champions.
```

### Reset-optimizer hard B1 seat-1/ref-off

```text
runs/league_b1init_b1seat1_reward5_refbcoff_lr2e5_resetopt_u480_to_u500_20260427
```

32-pair:

```text
weighted: 0.756250
B1:      0.50
B3:      0.890625
B4:      1.0
```

### B1-seat1 positive advantage objective

Code:

```text
training.b1_second_seat_positive_advantage_policy_coef
```

Runs:

```text
runs/league_b1init_b1seat1_posadv_lr2e5_resetopt_u480_to_u500_20260427
runs/league_b1init_b1seat1_posadv10_lr2e5_resetopt_u480_to_u500_20260427
```

Results:

```text
coef 1.0:  weighted 0.775, B1 0.50, B3 0.9375
coef 10.0: weighted 0.750, B1 0.50, B3 0.875
```

Verdict:

```text
Objective was live, but did not move B1 and stronger coefficient hurt B3.
```

### B1-seat1 anti-clone/reference avoidance objective

Code:

```text
training.b1_second_seat_reference_top_action_avoidance_coef
```

Run:

```text
runs/league_b1init_b1seat1_refavoid1_lr2e5_resetopt_u480_to_u500_20260427
```

8-pair:

```text
weighted: 0.800
B1:      0.50
B3:      1.0
B4:      1.0
```

32-pair confirm:

```text
weighted: 0.775
B1:      0.50
B3:      0.9375
B4:      1.0
```

Continuation:

```text
runs/league_b1init_b1seat1_refavoid1_lr2e5_u500_to_u520_20260427
```

u520:

```text
weighted: 0.750
B1:      0.50
B3:      0.875
```

Guard rolled back to u500.

Verdict:

```text
Exact-action anti-cloning is not sufficient.
```

## B1 Disagreement Audit Findings

Audit artifact:

```text
runs/league_b1init_b1warmfix_vlowlr_u480_b1_audit_allowhash_20260427/audit/summary_outcome_conditioned.json
```

Findings:

```text
64 games vs B1: W=32, L=32
focal seat 0: 21W / 11L
focal seat 1: 11W / 21L
```

Policy alignment:

```text
seat1 losses:
  top-action match rate ~0.8742
  top-family match rate 1.0
  mean probability on B1 top action ~0.7104

seat1 wins:
  top-action match rate ~0.8730
  top-family match rate 1.0
  mean probability on B1 top action ~0.7107
```

Mismatch-only counters:

```text
Mostly within-family differences:
  encore_decline(slot=0) vs encore_decline(slot=1)
  main_play_character(hand_index=1/2/3/5, stage_slot=...) vs hand_index=0 same stage_slot
top_mismatched_family_pairs empty
```

Interpretation:

```text
The B1 wall is not explained by obvious pass/main_move/family-level pathology.
The model is a close B1-family clone.
The win/loss split is more seat-shaped than action-family-shaped.
```

## Important Commands

Train from current best:

```text
uv run python python/scripts/train.py \
  --stack-config <config> \
  --run-label <label> \
  --runtime-mode train_async_fast \
  --autoscale \
  --hardware-profile local \
  --resume-from runs/league_b1init_b1warmfix_vlowlr_u460_to_u480_20260426/training/checkpoints/checkpoint_480.pt \
  --resume-allow-config-mismatch \
  --resume-reset-optimizer \
  --seed-snapshot-run-dir runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425 \
  --b1-baseline-run-dir runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425 \
  --max-updates <N> \
  --checkpoint-interval-updates 20 \
  --profile-timers
```

Manual scalar confirm:

```text
uv run python python/scripts/manual_dev_eval_confirm.py \
  --stack-config <config> \
  --run-dir <run_dir> \
  --checkpoint <checkpoint.pt> \
  --summary <summary.json> \
  --update <update> \
  --pairs 32 \
  --workers 6 \
  --artifact-dir-name dev_eval_trueanchors_manual32_20260427
```

Focused tests after recent fix:

```text
uv run pytest -q \
  python/weiss_rl/tests/test_snapshot_registry.py::test_import_resume_league_snapshot_pool_preserves_local_recents_and_champions \
  python/weiss_rl/tests/test_train_stall_monitor.py::test_periodic_dev_eval_previous_recent_ignores_seed_imports_and_rejections \
  python/weiss_rl/tests/test_train_stall_monitor.py::test_periodic_dev_eval_previous_recent_requires_champion_when_promotion_gated \
  --tb=short
```

Result:

```text
3 passed
```

Runtime PFSP seed-exclusion tests:

```text
uv run pytest -q \
  python/weiss_rl/tests/test_runtime.py::test_refresh_opponent_pool_can_exclude_seed_imports_after_pfsp_handoff \
  python/weiss_rl/tests/test_runtime.py::test_refresh_opponent_pool_keeps_seed_imports_before_pfsp_handoff \
  --tb=short
```

Result:

```text
2 passed
```

## Server Context

Local machine:

```text
Windows single-GPU box
```

Final target:

```text
multi-GPU Linux L40 server
examples:
  uc1-l40-3: 72 AMD EPYC 9454 cores, 288 GB RAM, 3 NVIDIA L40 GPUs
  uc1-l40-4: 96 AMD EPYC 9454 cores, 384 GB RAM, 4 NVIDIA L40 GPUs
```

Local runs are for:

```text
correctness
artifact validation
relative comparison
learning-direction checks
```

Do not make final throughput claims from local Windows timings.

Before serious server runs:

```text
dry-run topology
1-2 update smoke
inspect artifacts
verify distributed.world_size
verify rank-0 artifact cleanliness
verify no NaNs / broken gradient sync
verify league/native rollout flags are live
```

## Questions For GPT Pro

Please review this as a structural league/self-play failure, not just a hyperparameter issue.

Key questions:

1. Is the new resume-registry carry-forward approach conceptually correct?
   - Should prior true league snapshots be imported as `league_import` and treated as local recents?
   - Should champion status be preserved across continuation runs?
   - Are there risks in preserving original policy IDs like `policy_000011`?

2. How should the league distinguish:
   - B1 baseline anchor;
   - imported B1 seed/history snapshots;
   - true local recent snapshots;
   - true promoted champions;
   - rejected candidates;
   - hard negatives?

3. Should PFSP sample seed imports at all after warmup?
   - Current proposed behavior: seed imports are warmup ballast only, then excluded from true PFSP once eval gate opens.
   - Is that right?

4. Why might B1 remain exactly `0.50` across so many interventions?
   - Is this likely paired-seat/eval symmetry?
   - A true B1 local equilibrium?
   - A policy clone issue?
   - A reward/credit-assignment limitation?
   - A missing best-response/counterfactual search signal?

5. What is the best next structural experiment?
   - Continue from u480 with fixed resume registry and seed exclusion?
   - Force a true local recent/champion ladder and re-test PFSP?
   - Implement counterfactual action evaluation from B1-seat1 losing states?
   - Add a best-response/search teacher?
   - Change promotion gates?
   - Change eval to report B1, imported seed recent, local recent, and champion separately?

6. What diagnostics would most quickly prove whether the league is now working?
   - Registry source-kind timelines?
   - PFSP pool composition over time?
   - Evaluation against true local previous recent?
   - Seat-specific B1 eval?
   - State/action counterfactual audit?

7. What should be avoided?
   - More tiny coefficient tweaks?
   - Overweighting B1?
   - Overtrusting 8-pair eval?
   - Letting imported B1 seed snapshots masquerade as recent league progress?

8. Please critique the exact code/config snippets above.
   - Is one sorted snapshot registry plus filters enough, or should the system have typed pools?
   - Is `promotion_gate_enabled` filtering "recent" down to champions during eval a good idea?
   - Should seed-import champion status be preserved or stripped?
   - Is the PFSP seed-exclusion activation condition correct?
   - Is preserving original policy IDs on resume safe?
   - Is there a hidden way the new local run can still overwrite/collide with imported `policy_000011` style IDs?

9. Please propose a better league algorithm if the current one is structurally weak.
   - Examples could include a main-player/exploiter league, fictitious self-play, prioritized best-response branches, local candidate ladder independent from champion ladder, or explicit B1 exploiters.
   - Please map the algorithm to this codebase's concepts: registry, source_kind, promotion gate, PFSP, periodic eval, and manual confirm.

10. Please propose an eval redesign.
   - It must separately report:
     - B1 no-league baseline;
     - imported B1 seed/history snapshots;
     - previous true local recent;
     - latest true local recent;
     - previous champion;
     - latest champion;
     - seat-specific B1 results;
     - unpaired or shuffled-seat B1 results if useful.
   - Please say which metrics should be used for promotion and which should only be diagnostics.

## Desired Output From GPT Pro

Please produce:

1. A diagnosis of the likely structural failure.
2. A critique of the recent `league_import` / seed-exclusion fix.
3. A prioritized implementation plan for making the league actually build stronger opponents.
4. Specific code/config changes to inspect or make.
5. A short experimental matrix with expected outcomes and stop/continue criteria.
6. Any red flags in the current metrics or interpretation.
7. A ranked list of likely bugs/design flaws, with confidence levels.
8. A minimal "do this next" patch plan that a coding agent can implement without needing more context.
