# Weiss Schwarz RL Rebuild Plan

Date: 2026-05-12

Status: executed through the 2026-05-17 B1 and main fixed-deck thesis passes.
The implemented standard surface uses `python -m weiss_rl.cli` with named
profiles instead of the earlier draft `--grade`/`--tier` flags. B1 is now a
locked selected artifact, and the current main fixed-deck selected artifact is
published as `main_league_selected` with final eval, metagame, figures, replay
verification, and passing paper readiness. Remaining plan work is cleanup,
additional ablations, and optional broader PFSP/ecology extensions, not the
core B1/main fixed-deck evidence path.

This plan rebuilds `weiss_schwarz_rl` into a lean, reproducible, thesis-grade
reinforcement learning system for Weiss Schwarz. The target simulator dependency
is `weiss-sim` / `weiss-schwarz-simulator` version `1.2.0` or newer.

The goal is not merely cleanup. The repo should become a reliable research
machine that can train, evaluate, document, and defend:

1. B0 RandomLegal
2. B1 NoLeague
3. B2 HeuristicPublic
4. B3 HeuristicPublicAggro
5. B4 HeuristicPublicControl
6. Main league thesis model
7. Baselines and ablations

Poor learning must not be treated as only a hyperparameter problem. The rebuild
must investigate structural causes: simulator integration, fixed deck selection,
observation encoding, legal action mapping, action metadata, reward perspective,
opponent loading, league sampling, evaluation correctness, artifacts, and stale
paths.

## Non-Negotiable Policies

### Fixed Deck Thesis Policy

Primary thesis comparisons use fixed decks:

1. Focal model, B0, B1, and B2 use `preset:main_deck_5hy_yotsuba_v1`.
2. B3 aggro uses `preset:aggro_deck_5hy_nino_v1`.
3. B4 control uses `preset:control_deck_jj_s66_v1`.
4. Multideck is exploratory/generalization only, not a primary thesis result.

Any thesis-grade train or eval route must fail if deck provenance is implicit,
missing, or mixed with exploratory multideck results.

### Artifact Preservation Policy

Do not modify historical run outputs unless the user explicitly asks for a
migration:

- `runs/`
- `run_logs/`
- `vast_artifacts/`
- `thesis_figures_final/`
- existing checkpoints, eval summaries, replay artifacts, and figure exports

New thesis-grade outputs should live in a canonical run layout and should be
traceable back to config, simulator contract, git state, seed files, and
snapshot registry entries.

### Workflow Separation Policy

The repo must keep three levels separate:

1. Smoke: tiny, fast checks that prove plumbing works.
2. Development/acceptance: moderate runs that prove the training/eval route is
   useful enough to continue.
3. Thesis-grade: saved artifacts, fixed seeds, confidence intervals, replay
   capture, and figure exports suitable for report claims.

Smoke/demo outputs must never be treated as thesis evidence.

## 1. Current Architecture Summary

### 1.1 Training Entry Points

Current public training surfaces:

- `python/scripts/thesis_run.py`
  - Convenience wrapper around train/eval/compare commands.
  - Currently exposes many knobs such as preset, device, environment count,
    unroll length, max updates, runtime mode, seed, B1 baseline directory, and
    passthrough train/eval args.
  - Good operator surface, but too flag-heavy for the final thesis workflow.
- `python/scripts/train.py`
  - Mostly a compatibility facade and script entrypoint.
  - Re-exports many internals for old tests.
  - Delegates real work into `weiss_rl.training`.
- `python/weiss_rl/training/cli.py`
  - Real train argument parser.
- `python/weiss_rl/training/train_entrypoint_main.py`
  - Loads stack config, applies overrides, verifies runtime prerequisites,
    writes manifest/environment summaries, and calls the training loop.
- `python/weiss_rl/training/minimal_loop.py`
  - Despite the name, this is effectively the canonical training loop.
  - Builds model/learner/runtime, runs warm start, collects batches, updates the
    learner, saves checkpoints/snapshots, runs promotion gates and dev evals.

Target direction:

- Keep `thesis_run.py` or console-script equivalents as the human-facing
  command surface.
- Move train semantics into package modules and keep scripts thin.
- Stop depending on script-internal helper re-exports in thesis-grade tests.

### 1.2 Simulator Integration

Current pieces:

- `pyproject.toml` exposes `weiss-sim>=1.2.0,<2` through the optional `sim` extra.
- `python/weiss_rl/core/simulator_contract.py` probes `weiss_sim`, exports the
  spec bundle, hashes the canonical JSON, and records simulator metadata.
- `python/weiss_rl/core/spec.py` parses action/observation contract data.
- `python/weiss_rl/training/startup.py` has stronger training startup gates for
  simulator version and stepping APIs.
- `python/weiss_rl/envs/pool_factory.py` creates simulator-backed pools and maps
  profiles:
  - `debug`: inspect profile, mask layout, `i32` action ids
  - `balanced`: fast profile, mask layout, `i16` action ids
  - `fast`: fast profile, packed `ids_offsets`, `i16` action ids
- `python/weiss_rl/envs/decision_env.py` wraps simulator reset/step APIs,
  legal-id layouts, fused stepping, metadata, and best-effort reset.

Problems:

- Simulator 1.2.0 validation is not one shared startup path for train, eval,
  replay, diagnostics, and figures.
- Some code still treats hard-coded action count/pass id facts as fallback
  defaults rather than checked simulator-1.1 contract facts.
- Baseline configs can leave deck selection implicit.

Target direction:

- Centralize simulator validation and require it for all non-demo workflows.
- Make the simulator 1.2.0 optimized `fast` packed legal-id path the standard
  route for thesis train/eval.
- Keep debug/mask paths only for diagnostics or legacy tests.

### 1.3 Environment, Pool, and Vectorization Paths

Current paths:

- Canonical thesis path:
  - `make_env_pool_from_config`
  - `DecisionBoundaryEnv`
  - `QueueRuntime`
  - fast packed legal ids
  - central/process collection where supported
- Compatibility or older paths:
  - `ActorWorker`
  - `MinimalRollout`
  - mask-only `build_training_env`
  - old batch builders and script-level helper exports

Runtime selection currently depends on mode, model kind, league state, collection
backend, actor count, process support, and structured model gates. This makes
performance claims hard to defend unless each lane is benchmarked separately.

Target direction:

- One blessed route per thesis lane:
  - B1 NoLeague route
  - Main league route
  - PPO-lite or other baseline route, if retained
- A small route selector object should explain why a route was selected and what
  optimized features are active.
- Runtime metrics should explicitly log mode, route, simulator profile,
  collection backend, actor count, env count, learner device, model kind, and
  fused-step support.

### 1.4 Model Code

Current pieces:

- `python/weiss_rl/model.py` remains a compatibility facade.
- `PolicyValueModel` supports dense/typed encoders and recurrent/feedforward
  cores.
- `StructuredLegalPolicyValueModel` uses simulator spec data, action catalog,
  structured observation contract, and structured legal action heads.
- Structured action scoring is split across packed/factorized/dense helpers.
- Public heuristic bias and teacher-related scoring exist.

Problems:

- Too many model paths are available for a thesis reader to understand quickly.
- Action metadata is trusted heavily in packed scoring.
- A concrete action-contract risk exists around `choice_prev_page` and
  `choice_next_page`: they decode as no-arg actions but can be classified as
  indexed action families in model action tables.

Target direction:

- Choose one canonical thesis model path:
  - structured legal policy
  - packed/factorized legal candidate scoring
  - GRU recurrent core
  - explicit simulator spec contract
- Keep no-GRU, PPO-lite, dense-action, and reward variants as named ablations.
- Add invariant tests that compare simulator candidate metadata against
  `ActionCatalog.decode(action_id)` for every packed legal action.

### 1.5 Rollout and Learner Code

Current pieces:

- `QueueRuntime.collect_update_batch` and runtime component mixins collect
  IMPALA/V-trace batches.
- Central collection supports structured packed rows, focal/mirror/heuristic
  actor rows, and opponent overwrites.
- `python/weiss_rl/training/batches.py` dispatches learner batch collection.
- `python/weiss_rl/training/learner_factory.py` constructs learners.
- PPO-lite currently shares significant infrastructure with the IMPALA learner.

Risks:

- Behavior policies can include deterministic heuristic rows, which makes V-trace
  rho/log-prob behavior important to inspect.
- Hidden state reset, seat perspective, reward sign, and truncation behavior are
  structural learning risks.
- Throughput and learning-quality claims can diverge; both must be measured.

Target direction:

- Make behavior policy provenance visible in every batch.
- Log rho percentiles, heuristic-row coverage, teacher support, truncation rate,
  reward scale, value target scale, and legal candidate stats.
- Either make PPO-lite an independent baseline or document clearly that it is a
  PPO objective on shared IMPALA infrastructure.

### 1.6 League and Self-Play Code

Current pieces:

- B1 import/aliasing before runtime creation.
- Snapshot registry and policy ids.
- Promotion gates.
- PFSP and champion/hard-negative sampling.
- Heuristic public opponents and variant profiles.

Risks:

- B1 path can be trainable but not pinned to the thesis main deck unless config
  enforces it.
- B2/B3/B4 final eval rows are not guaranteed strongly enough by the current
  selector.
- League sampling can hide failures if aggregate win rate improves while B2
  stays flat.

Target direction:

- Make B0-B4 anchors explicit and required in thesis configs.
- Require B1 baseline artifact provenance and deck contract before main league
  training starts.
- Promotion gates should check not just aggregate score, but anchor rows, seat
  balance, truncation rate, legality errors, and B2 movement.

### 1.7 Baselines and Heuristics

Current baselines:

- B0 RandomLegal
- B1 NoLeague baseline
- B2 HeuristicPublic
- B3 HeuristicPublicAggro
- B4 HeuristicPublicControl
- PPO-lite / no-recurrence / other ablations in configs

Current heuristic path:

- Public-only deterministic heuristic policy.
- Simulator-native heuristic pool parity tests exist or are partially present.

Target direction:

- Baselines are first-class thesis lanes, not helper modes.
- B1 has its own train/eval command and artifact contract.
- B2/B3/B4 are fixed policy ids with fixed deck mappings.
- Heuristic policies must be checked against simulator-native behavior and public
  observation constraints.

### 1.8 Config System

Current pieces:

- Strict grouped config dataclasses under `python/weiss_rl/config`.
- YAML `extends` support.
- Dotted `--override section.key=value` support.
- Many preset files under `configs/presets`, including dated rescue/experiment
  presets.

Problems:

- The standard preset extends a long experimental chain.
- Important thesis behavior is buried in inherited configs.
- Too many configs live in the public namespace.

Target direction:

Keep a small public thesis config set:

- `configs/thesis/base_fixed_deck_structured.yaml`
- `configs/thesis/b1_noleague.yaml`
- `configs/thesis/main_league.yaml`
- `configs/thesis/main_league_auto_gpu.yaml`
- `configs/thesis/final_eval.yaml`
- `configs/thesis/multideck_exploratory.yaml`
- `configs/thesis/ablations/no_gru.yaml`
- `configs/thesis/ablations/ppo_lite.yaml`
- `configs/thesis/ablations/terminal_only_reward.yaml`
- `configs/thesis/ablations/no_teacher.yaml`
- `configs/thesis/ablations/no_b2_exposure.yaml`

Move dated and rescue configs to `configs/archive/` after tests prove the new
surface covers the thesis workflow.

### 1.9 Evaluation, Export, and Figure Generation

Current pieces:

- `python/scripts/eval.py`
- `python/weiss_rl/eval/final_eval.py`
- `python/weiss_rl/eval/simulator_runner.py`
- `python/weiss_rl/eval/policy_set.py`
- `python/scripts/make_figures.py`
- `python/scripts/paper_readiness_check.py`
- replay and metagame helpers

Problems:

- `python/scripts/README.md` appears stale about non-demo eval behavior.
- Legacy figure generation under `scripts/` can diverge from canonical figure
  generation.
- Eval commands can update run directories, so historical outputs need a clear
  read-only policy.

Target direction:

- One standard final eval command that always includes B0-B4 unless explicitly
  running an ablation or diagnostic.
- One standard figure command that consumes canonical eval outputs.
- Figure data exports must include raw CSV/JSON plus rendered PNG/PDF/table
  outputs.

### 1.10 Tests and Docs

Current pieces:

- Broad test suite under `python/weiss_rl/tests`.
- `pyproject.toml` defines markers such as simulator/artifact/slow, but marker
  usage is not yet systematic.
- Docs cover architecture, training, evaluation, runtime modes, artifact
  contract, reproducibility, testing, and troubleshooting.

Problems:

- Test suite is broad but not layered around thesis confidence.
- Public-demo/scaffold tests are easy to confuse with thesis-grade checks.
- `docs/rebuild_log.md` is missing.

Target direction:

- Add test lanes:
  - unit
  - simulator
  - artifact
  - deterministic eval
  - public_demo
  - scaffold
  - legacy_compat
  - slow/thesis
- Add one high-signal `test_thesis_workflow_contract.py`.
- Keep `docs/rebuild_log.md` updated after every milestone.

## 2. Pain Points and Dead Paths

### 2.1 Unused or Overexposed Flags

Current wrapper flags expose too much implementation detail:

- runtime mode
- env count
- actor count indirectly through configs
- unroll length
- max updates
- device
- seed
- B1 baseline directory
- train/eval passthrough args

Target:

- Thesis commands should encode recommended defaults.
- Overrides remain possible but are not required for standard workflows.

### 2.2 Duplicate Config Routes

Problem:

- Multiple configs represent similar tiny32/packed/acceptance/standard/rescue
  variants.
- Long `extends` chains obscure what a run actually does.

Target:

- Public thesis configs are short and explicit.
- Historical configs are archived.
- Resolved config is written into every run artifact.

### 2.3 Stale Compatibility Layers

Quarantine candidates:

- `ActorWorker`
- `MinimalRollout`
- script helper re-export tests
- mask-only training environment builder
- legacy snapshot id parsing
- deprecated run-id compatibility
- legacy manifest training mode

Do not delete until replacement tests cover the thesis workflow.

### 2.4 Slow Paths

Slow or non-thesis paths:

- scalar Python opponent paths
- mask legality layout for training
- debug env profiles
- old actor worker path
- one-off scripts that rerun eval/figures without canonical artifact contract

Target:

- Use simulator 1.2.0 optimized packed legal-id path by default.
- Benchmark B1 and league routes separately.

### 2.5 Confusing Scripts

Scripts to consolidate, archive, or document:

- `structured_v2_campaign.py`
- `structured_v2_baseline.py`
- `launch_experiments.py`
- `sweep_experiments.py`
- `targeted_confirm_eval.py`
- `parallel_final_eval.py`
- `heuristic_sanity_scan.py`
- legacy `scripts/make_thesis_figures.py`

Target:

- Keep one clear operator command surface.
- Move one-off research helpers to `python/scripts/archive/` or document them as
  diagnostics.

### 2.6 Fragile Eval Behavior

Problems:

- B2/B3/B4 inclusion is not forced strongly enough.
- Eval can mutate run directories.
- Baseline policy resolution depends on registry/current-run behavior.
- Historical outputs can be accidentally refreshed with new simulator code.

Target:

- Explicit thesis policy set.
- Read-only mode for historical evaluation inspection.
- Clear `--write` behavior for eval outputs.
- Snapshot selection written to manifest.

### 2.7 Old Simulator Assumptions

Examples:

- hard-coded action count
- hard-coded pass action id
- fallback observation/action assumptions
- mask legality assumptions in training helpers
- action metadata trusted without decode validation

Target:

- Treat these as checked simulator-1.1 contract facts.
- Fail before training if the simulator contract drifts.

### 2.8 Structural Learning Risks

Potential structural causes of poor learning:

- wrong deck for B1/B2 comparison
- observation not consistently self-first
- private information leaking into public heuristic path
- legal ids or candidate ordering mismatch
- action metadata mismatch
- choice-pagination action family bug
- reward sign or reward perspective wrong
- hidden states not reset at episode boundaries
- seat swap or seed pairing mistakes
- stale opponent or B1 alias loaded
- heuristic behavior rows dominating learner updates
- V-trace rho/log-prob path wrong for heuristic behavior
- promotion gate accepts noisy aggregate improvement while B2 remains flat

These must be diagnosed before claiming hyperparameter failure.

## 3. Target Architecture

### 3.1 Minimal Module Layout

Keep package modules organized by thesis responsibility:

- `weiss_rl.config`: strict config loading, schema, resolved config export
- `weiss_rl.core`: simulator spec, action catalog, masking, reproducibility
- `weiss_rl.envs`: simulator pool factory and decision-boundary env wrapper
- `weiss_rl.models`: canonical structured model and ablation models
- `weiss_rl.learners`: IMPALA/V-trace and explicitly named baseline learners
- `weiss_rl.runtime`: route selection plus small runtime route modules
- `weiss_rl.training`: train orchestration, checkpointing, B1 anchor, gates
- `weiss_rl.league`: opponent pool, PFSP, promotion, snapshot registry
- `weiss_rl.eval`: policy sets, simulator eval runner, diagnostics, CIs
- `weiss_rl.figures`: figure data export and rendering
- `weiss_rl.artifacts`: canonical run layout and readiness checks

### 3.2 Standard Training Command Shape

Implemented commands:

```powershell
uv run --extra dev --extra sim python -m weiss_rl.cli train-b1 --run-label b1_<date> --profile thesis-local
uv run --extra dev --extra sim python -m weiss_rl.cli train-main --run-label main_<date> --profile thesis-local --b1-run runs/<b1_run>
```

Compatibility path:

```powershell
uv run --extra dev --extra sim python python/scripts/thesis_workflow.py train-b1 --run-label b1_<date> --profile thesis-local
uv run --extra dev --extra sim python python/scripts/thesis_workflow.py train-main --run-label main_<date> --profile thesis-local --b1-run runs/<b1_run>
```

The final shape can use either module commands or script wrappers, but the
operator should not need to know internal preset chains.

### 3.3 Standard Evaluation Command Shape

Implemented commands:

```powershell
uv run --extra dev --extra sim python -m weiss_rl.cli eval-final --run-dir runs/<main_run> --b1-run runs/<b1_run>
uv run --extra dev --extra sim python -m weiss_rl.cli b2-audit --run-dir runs/<main_run> --episodes-jsonl runs/<main_run>/eval/final_eval/episodes.jsonl --policy-id <focal_policy_id>
uv run --extra dev python -m weiss_rl.cli figures --run-dir runs/<main_run> --format png --format pdf
```

Final eval should always write the exact policy set, seed file, simulator
contract, snapshot selection, and deck mapping.

### 3.4 Config Naming Scheme

Use human names tied to thesis lanes:

- `base_fixed_deck_structured`
- `b1_noleague`
- `main_league`
- `main_league_auto_gpu`
- `final_eval`
- `multideck_exploratory`
- `ablations/<ablation_name>`

Avoid names based on old experiment chronology, rescue attempts, hidden tiny32
chains, or one-off dates.

### 3.5 Artifact Layout

Canonical run layout:

```text
runs/
  smoke/<run_label>/
  dev/<run_label>/
  thesis/<run_label>/
    manifest.json
    environment.json
    run_summary.json
    determinism_report.json
    paper_readiness_summary.json
    config/
      requested.yaml
      resolved.yaml
      resolved.json
    simulator/
      contract.json
      spec_bundle.json
      spec_hash.txt
    training/
      metrics.jsonl
      performance.jsonl
      checkpoints/
      snapshots/
        registry.json
    eval/
      final_eval/
      diagnostics/
      b2_disagreement/
      metagame/
    replays/
    figures/
      paper/
      data/
```

### 3.6 Docs Layout

Target docs:

- `README.md`: short quickstart and thesis workflow pointer
- `docs/getting_started.md`: install and environment setup
- `docs/thesis_workflow.md`: standard train/eval/figure workflow
- `docs/configuration.md`: small public config set
- `docs/training.md`: training internals and standard lanes
- `docs/evaluation.md`: final eval, CIs, seeds, replays
- `docs/artifact_contract.md`: canonical run tree
- `docs/simulator_compatibility.md`: simulator 1.2.0 contract
- `docs/testing.md`: test lanes and commands
- `docs/rebuild_log.md`: milestone log
- `docs/archive/`: historical notes and legacy result provenance

### 3.7 Test Layers

Test lanes:

- Unit: pure functions, config parser, action catalog, schedule math.
- Simulator: live `weiss_sim>=1.2.0` contract tests.
- Integration: short train/eval paths with tiny budgets.
- Artifact: manifest/readiness/figure layout.
- Deterministic eval: seed pairing, seat swaps, policy set.
- Public demo: demo-only outputs, clearly not thesis evidence.
- Legacy compatibility: kept only where migration risk exists.
- Slow/thesis: acceptance and final routes.

### 3.8 Smoke, Development, and Thesis Workflows

Smoke:

- tiny env count
- tiny update count
- may run on CPU
- proves no crash and artifact skeleton

Development/acceptance:

- enough envs/seeds to catch structural problems
- saved artifacts
- checks B0/B1/B2 movement and throughput

Thesis:

- fixed deck policy
- pinned seed files
- saved simulator contract
- B0-B4 matrix
- confidence intervals
- replay/debug artifacts
- figure exports
- readiness summary

## 4. Simulator 1.2.0 Integration Plan

### 4.1 Standard Routes Use Optimized Simulator Path

Tasks:

1. Make `weiss-sim>=1.2.0` part of the default thesis install path.
2. Keep `uv sync --extra dev --extra sim` as the explicit validation install.
3. Centralize simulator startup validation.
4. Use `profile=fast`, `legality=ids_offsets`, and packed `i16_legal_ids` for
   standard train/eval.
5. Log optimized path status into every run manifest.

### 4.2 Deck Selection Follows Fixed Policy

Tasks:

1. Add a central deck policy module or config validator.
2. Require B1 no-league configs to pin main deck.
3. Require eval policy set to map B0/B1/B2 to main deck, B3 to aggro, B4 to
   control.
4. Fail startup if deck pools are empty for thesis-grade workflows.
5. Label multideck results as exploratory in manifests and figures.

### 4.3 Remove or Quarantine Old Assumptions

Tasks:

1. Replace hard-coded action-space constants with simulator contract checks.
2. Quarantine mask-only training paths outside thesis commands.
3. Remove hidden fallbacks that silently switch simulator/profile/deck.
4. Keep debug profile for diagnostics only.

### 4.4 Identify Remaining Slow or Legacy Env Paths

Known candidates:

- mask legality training env
- scalar Python fixed opponent backend
- old actor worker
- old minimal rollout
- one-off script runners

Each path should be either:

- deleted,
- moved to archive/legacy,
- or documented as diagnostic-only.

### 4.5 Simulator Compatibility Tests

Required tests:

- simulator version is `>=1.2.0`
- required reset/step/fused APIs exist
- action space size and pass id match exported spec
- action ids decode round-trip through `ActionCatalog`
- candidate metadata matches decoded action id
- legal id ordering and offsets are valid
- fixed deck presets load
- public heuristic has no private-information access
- simulator-native heuristic matches Python oracle across live steps
- reward and curriculum JSON are accepted

## 5. Fixed Deck Policy Implementation

Tasks:

1. Add a central mapping:
   - `B0 RandomLegal -> preset:main_deck_5hy_yotsuba_v1`
   - `B1 NoLeague -> preset:main_deck_5hy_yotsuba_v1`
   - `B2 HeuristicPublic -> preset:main_deck_5hy_yotsuba_v1`
   - `B3 HeuristicPublicAggro -> preset:aggro_deck_5hy_nino_v1`
   - `B4 HeuristicPublicControl -> preset:control_deck_jj_s66_v1`
2. Validate train configs for focal/B1/B2 main deck use.
3. Validate eval schedules propagate policy decks to both seats.
4. Add tests that multideck configs are labeled exploratory.
5. Include deck policy in run manifest and figure data.

## 6. Training Plan

### 6.1 B1 NoLeague Baseline

Objective:

- Produce a trainable and evaluable no-league baseline on the main thesis deck.

Plan:

1. Use fixed main deck for focal and opponent.
2. Disable league.
3. Use optimized simulator fast packed path.
4. Use canonical structured model unless running named ablations.
5. Save B1 artifact with policy id `B1 NoLeague`.
6. Publish/record baseline alias in snapshot registry.
7. Evaluate against B0 and B2 before using as main league anchor.

Stop/go criteria:

- Go if B1 clearly beats B0 with saved artifacts and no legality/deck/eval
  failures.
- Stop if B1 does not beat B0 after acceptance budget; run structural diagnosis
  before tuning hyperparameters.

### 6.2 Main League Thesis Model

Objective:

- Train the main thesis model with B1 anchor, B0/B2 exposure, self-play, and
  promotion gates.

Plan:

1. Require B1 baseline run directory.
2. Load B1 as a fixed anchor with verified deck/config provenance.
3. Begin with B0/B1/B2 exposure and controlled teacher/curriculum support.
4. Add champion/PFSP/hard-negative sampling after basic anchor performance is
   stable.
5. Keep B3/B4 primarily for eval and labeled robustness diagnostics.
6. Promote snapshots automatically only when gates pass.

Stop/go criteria:

- Go if B0/B1 rows improve, B2 row moves, legality/reward diagnostics are clean,
  and throughput is acceptable.
- Stop if B2 stays flat after scheduled exposure; run B2 causal diagnosis.

### 6.3 Curriculum, Teacher, Hard Negative, and Self-Play Schedule

Suggested phases:

1. Structural sanity:
   - short runs
   - B0/B2/B1 checks
   - action metadata checks
   - reward and value scale logging
2. B1 acceptance:
   - no league
   - main deck only
   - B0/B2 eval
3. Main warm start:
   - B1 anchor
   - B0/B2/teacher exposure
   - low self-play pressure
4. League growth:
   - PFSP
   - champion snapshots
   - hard negatives
   - promotion gate
5. Thesis run:
   - fixed configs
   - long budget
   - saved seeds
   - final eval and figures

### 6.4 Reward Shaping Candidates Enabled by Simulator 1.2.0

Candidates:

- terminal win/loss `+1/-1`
- stock damage progress
- level progress
- reverse/direct attack incentives
- climax or card-advantage terms only if public/defensible
- truncation penalties or flags
- illegal-action should remain impossible, not silently repaired

Rules:

- Reward shaping must be config-driven.
- Every shaping term must be logged separately.
- Terminal-only remains an ablation.
- Shaping must not encode private or opponent-hidden information into public
  policies.

### 6.5 Throughput Targets

Report throughput by lane:

- B1 no-league fast path
- main league path
- PPO-lite or other baselines
- eval path

Metrics:

- samples/sec
- env step ms
- central forward ms
- batch pack ms
- learner update ms
- queue wait time
- actor idle time
- GPU utilization, where available

Prior memory/context indicates no-league optimized paths have reached roughly
22k samples/sec in earlier work, but that should not be generalized to league
mode without a fresh benchmark in this checkout.

### 6.6 Model and Batch Recommendations

Initial recommendations:

- model: structured legal policy
- core: GRU
- GRU hidden size: 128 for acceptance, 256 for thesis candidate
- unroll length: 32 for local/dev, 64 for thesis if stable
- learner batch: 128 to 256 unrolls, depending on memory and throughput
- local env count: 32 to 96
- larger machine env count: 192 to 384 after profiling
- ablations: no-GRU, PPO-lite, terminal-only reward, no-teacher, no-B2-exposure

### 6.7 Automatic Promotion and Gating

Promotion gate should require:

- no legality/spec/deck failures
- B0 row not regressed
- B1 row not regressed beyond tolerance
- B2 row improving or diagnosis flag set
- seat advantage within tolerance
- truncation rate within tolerance
- enough paired seeds for the gate tier
- confidence interval not obviously worse than previous champion

### 6.8 Short Acceptance vs Long Thesis Runs

Acceptance:

- moderate update count
- stage1 eval seeds
- replay capture on suspicious rows
- used to decide whether to continue

Thesis:

- long budget
- fixed seed files
- final B0-B4 matrix
- bootstrap confidence intervals
- figure export
- readiness check

## 7. Evaluation Plan

### 7.1 Focal Versus B0-B4

Final eval must include:

- focal vs B0 RandomLegal
- focal vs B1 NoLeague
- focal vs B2 HeuristicPublic
- focal vs B3 HeuristicPublicAggro
- focal vs B4 HeuristicPublicControl
- baseline-vs-baseline rows where useful for calibration

### 7.2 Per-Seat Evals

Every pairwise matchup must include:

- focal as seat 0
- focal as seat 1
- paired seeds
- same deck policy by policy id
- seat advantage diagnostics

### 7.3 Heavier Final Evals

Suggested seed tiers:

- smoke: 2 to 8 paired seeds
- acceptance: 32 to 128 paired seeds
- thesis final: 256 to 1024 paired seeds, depending on time budget

Do not overclaim from tiny local evals.

### 7.4 Snapshot Selection

Snapshot selection should be deterministic and saved:

- current/latest
- best gated champion
- spaced historical snapshots
- B1 baseline anchor
- explicitly selected ablation snapshots

The selected snapshot list must be written to final eval artifacts.

### 7.5 Ablations

Required or recommended ablations:

- B1 no-league
- no recurrence
- PPO-lite or alternate learner
- terminal-only reward
- no teacher
- no B2 exposure
- no league hard negatives
- multideck exploratory generalization

### 7.6 Confidence Intervals and Statistics

Report:

- win rate
- paired bootstrap confidence interval
- per-seat win rate
- number of games
- seed file/hash
- policy ids and decks
- effect sizes for main comparisons

### 7.7 Replay and Debug Artifacts

Capture replays for:

- B2 flatline
- unexpected B0/B1 regressions
- large seat advantage
- truncation spikes
- legality/spec mismatch
- policy disagreement examples

Replay artifacts should include:

- seed
- seat
- policy ids
- deck ids
- chosen legal id
- top-k policy actions
- action metadata
- decoded action
- public features used by heuristic

### 7.8 B2 HeuristicPublic Flatline Diagnosis

If B2 remains at or near zero win rate or does not improve:

1. Verify B2 deck is main deck.
2. Verify B2 policy loads the intended public heuristic.
3. Verify B2 uses no private information.
4. Compare learner and B2 on identical public states.
5. Log top-1 and top-k agreement.
6. Log action-family agreement.
7. Inspect pass/attack/play/cancel/choice behavior.
8. Compare candidate metadata to decoded action ids.
9. Check reward sign, value target scale, and advantage scale.
10. Check V-trace rho for heuristic behavior rows.
11. Check whether heuristic rows dominate focal learner updates.
12. Capture replay bundles for states where B2 clearly prefers a different
    tactical action.

This diagnostic should become a standard command, not an ad hoc script.

## 8. Figures and Thesis Artifacts

### 8.1 Required Plots

Training:

- episode return
- win rate vs anchors over time
- policy loss
- value loss
- entropy
- value target scale
- reward component scale
- samples/sec
- promotion history
- truncation rate

Evaluation:

- final B0-B4 win-rate matrix
- focal-vs-baseline bars with CIs
- per-seat comparison
- ablation comparison
- metagame heatmap
- B2 disagreement/family confusion

### 8.2 Data Export Locations

Use:

- `runs/<grade>/<run>/figures/data/*.csv`
- `runs/<grade>/<run>/figures/data/*.json`
- `runs/<grade>/<run>/figures/paper/*.png`
- `runs/<grade>/<run>/figures/paper/*.pdf`
- `runs/<grade>/<run>/figures/paper/*.tex` where useful

### 8.3 Regeneration Commands

Target:

```powershell
uv run --extra dev python -m weiss_rl.cli figures --run-dir runs/<run> --format png --format pdf
```

Compatibility:

```powershell
uv run python python/scripts/make_figures.py --run-dir runs/<run>
uv run python python/scripts/paper_readiness_check.py --run-dir runs/<run>
```

### 8.4 Smoke, Development, and Thesis Outputs

Smoke figures may be placeholders only if explicitly labeled.
Development figures may be exploratory.
Thesis figures must be backed by final eval artifacts and seed manifests.

### 8.5 Defensible Artifact Names

Names should include:

- run label
- workflow grade
- policy id
- baseline id
- eval tier
- seed file/hash
- simulator contract hash
- generation timestamp

## 9. Validation Strategy

### 9.1 Unit Tests

Must cover:

- config schema and resolved config
- fixed deck policy
- action catalog decode
- action family classification
- packed metadata validation
- reward config parsing
- seed pairing and seat swapping
- policy id mapping

### 9.2 Integration Tests

Must cover:

- B1 smoke training
- main smoke training with B1 anchor
- final eval smoke
- figure smoke
- readiness smoke

### 9.3 Simulator Compatibility Tests

See section 4.5. These tests must be explicitly marked and must not silently pass
when `weiss_sim` is absent.

### 9.4 Deterministic Eval Tests

Must cover:

- same seeds produce same schedule
- seat swaps are paired
- B0-B4 are included in thesis policy set
- deck mapping is correct for each policy id
- snapshot registry resolution is stable

### 9.5 Throughput Benchmark Commands

Target:

```powershell
uv run python python/scripts/profile_train_job.py --stack-config configs/thesis/b1_noleague.yaml --run-label profile_b1_fast
uv run python python/scripts/profile_train_job.py --stack-config configs/thesis/main_league.yaml --run-label profile_main_league
```

Record benchmark output in `docs/rebuild_log.md`.

### 9.6 Short Smoke Training

Target:

```powershell
uv run --extra dev --extra sim python -m weiss_rl.cli train-b1 --run-label smoke_b1 --profile smoke
uv run --extra dev --extra sim python -m weiss_rl.cli train-main --run-label smoke_main --profile smoke --b1-run runs/<b1>
```

### 9.7 Longer Acceptance Training

Target:

```powershell
uv run --extra dev --extra sim python -m weiss_rl.cli train-b1 --run-label accept_b1 --profile thesis-local
uv run --extra dev --extra sim python -m weiss_rl.cli train-main --run-label accept_main --profile thesis-local --b1-run runs/<b1>
uv run --extra dev --extra sim python -m weiss_rl.cli eval-final --run-dir runs/<main> --b1-run runs/<b1>
```

### 9.8 Final Thesis Evaluation Commands

Target:

```powershell
uv run --extra dev --extra sim python -m weiss_rl.cli eval-final --run-dir runs/<main> --b1-run runs/<b1>
uv run --extra dev --extra sim python -m weiss_rl.cli b2-audit --run-dir runs/<main> --episodes-jsonl runs/<main>/eval/final_eval/episodes.jsonl --policy-id <focal_policy_id>
uv run --extra dev python -m weiss_rl.cli figures --run-dir runs/<main> --format png --format pdf
```

## 10. Cleanup Plan

### 10.1 Files and Configs to Delete Eventually

Delete only after replacements pass:

- duplicate dated configs that are no longer referenced
- stale placeholder figure paths
- script helper re-export tests that test old compatibility surfaces only
- dead legacy wrappers with no documented thesis use

### 10.2 Files and Configs to Consolidate

Consolidate:

- standard config chain into `configs/thesis/`
- figure generation into one canonical command
- eval policy selection into explicit thesis policy set
- simulator validation into one shared startup module
- B2 diagnostic scripts into a standard diagnostic command

### 10.3 Paths to Keep for Thesis

Keep and strengthen:

- `python/weiss_rl/config`
- `python/weiss_rl/core`
- `python/weiss_rl/envs/pool_factory.py`
- `python/weiss_rl/envs/decision_env.py`
- canonical structured model modules
- `python/weiss_rl/training`
- `python/weiss_rl/eval`
- `python/weiss_rl/artifacts`
- `python/scripts/thesis_run.py` as compatibility operator wrapper
- `python/scripts/eval.py`
- `python/scripts/make_figures.py`
- `python/scripts/paper_readiness_check.py`

### 10.4 Paths to Quarantine as Legacy

Candidates:

- old actor worker path
- minimal rollout path
- mask-only training env builder
- old campaign/sweep/launch scripts
- hardcoded historical figure scripts
- public-demo placeholder tests
- legacy compatibility tests for deprecated flags

### 10.5 Historical Outputs Not to Modify

Do not touch without explicit migration:

- `runs/`
- `run_logs/`
- `vast_artifacts/`
- `thesis_figures_final/`
- historical checkpoints
- historical eval summaries
- historical replay files
- thesis artifacts already referenced by reports

## 11. Milestones

### Milestone 1: Rebuild Control Docs

Objective:

- Create `RL_REBUILD_PLAN.md` and `docs/rebuild_log.md`.

Expected code artifacts:

- none

Expected docs artifacts:

- root rebuild plan
- rebuild progress log

Validation commands:

```powershell
rg --files -g RL_REBUILD_PLAN.md -g docs/rebuild_log.md
```

Stop/go criteria:

- Go when both files exist and describe the rebuild workflow.

Risks:

- Plan may drift unless log is updated after each milestone.

### Milestone 2: Simulator and Deck Contract

Objective:

- Make simulator 1.2.0 and fixed deck policy a shared enforced contract.

Expected code artifacts:

- shared simulator startup validator
- fixed deck policy validator
- candidate metadata invariant checks

Expected docs artifacts:

- updated simulator compatibility docs
- updated thesis workflow docs

Validation commands:

```powershell
uv sync --extra dev --extra sim
make simulator-check
```

Stop/go criteria:

- Go when train/eval/diagnostics fail fast on simulator/deck/spec mismatch.

Risks:

- Current tests may silently skip simulator checks if markers are not fixed.

### Milestone 3: Command and Config Simplification

Objective:

- Replace flag/config maze with a small standard thesis surface.

Expected code artifacts:

- CLI or wrapper commands for train-b1, train-main, eval-final, figures, and
  b2-audit
- `configs/thesis/` small config set

Expected docs artifacts:

- updated README
- `docs/thesis_workflow.md`
- `docs/configuration.md`

Validation commands:

```powershell
uv run python -m pytest -q python/weiss_rl/tests/test_thesis_workflow_contract.py
```

Stop/go criteria:

- Go when a fresh reader can find one B1 command and one main command.

Risks:

- Old scripts may still be referenced by docs or tests.

### Milestone 4: Runtime Path Consolidation

Objective:

- Make one blessed optimized route per thesis lane.

Expected code artifacts:

- route selector / route manifest
- quarantined legacy env paths
- runtime metrics export

Expected docs artifacts:

- updated runtime modes doc
- benchmark notes in rebuild log

Validation commands:

```powershell
uv run python python/scripts/profile_train_job.py --stack-config configs/thesis/b1_noleague.yaml --run-label profile_b1_fast
uv run python python/scripts/profile_train_job.py --stack-config configs/thesis/main_league.yaml --run-label profile_main_league
```

Stop/go criteria:

- Go when B1 and main league routes are benchmarked separately.

Risks:

- Fast no-league results may not transfer to league mode.

### Milestone 5: B1 NoLeague Baseline

Objective:

- Produce a trainable/evaluable B1 baseline on the fixed main deck.

Expected code artifacts:

- B1 command
- B1 config
- B1 artifact/alias handling

Expected docs artifacts:

- B1 workflow docs
- rebuild log entry with metrics

Validation commands:

```powershell
uv run --extra dev --extra sim python -m weiss_rl.cli train-b1 --run-label accept_b1 --profile thesis-local
uv run --extra dev --extra sim python -m weiss_rl.cli eval-final --run-dir runs/<b1_run> --b1-run runs/<b1_run>
```

Stop/go criteria:

- Go when B1 beats B0 with saved artifacts and passes deck/spec checks.

Risks:

- If B1 fails B0, diagnose structural bugs before tuning.

### Milestone 6: Main League Training Path

Objective:

- Produce a trainable/evaluable main league model with B1 anchor.

Expected code artifacts:

- main league command
- main league config
- promotion gate checks
- hard-negative/self-play schedule

Expected docs artifacts:

- main workflow docs
- promotion/gating docs

Validation commands:

```powershell
uv run --extra dev --extra sim python -m weiss_rl.cli train-main --run-label accept_main --profile thesis-local --b1-run runs/<b1_run>
uv run --extra dev --extra sim python -m weiss_rl.cli eval-final --run-dir runs/<main_run> --b1-run runs/<b1_run>
```

Stop/go criteria:

- Go when B0/B1 rows improve and B2 shows movement or a diagnosis artifact is
  produced.

Risks:

- League can optimize aggregate results while hiding B2 failure.

### Milestone 7: B2 Flatline Diagnosis

Objective:

- Make B2 flatline causal diagnosis standard and repeatable.

Expected code artifacts:

- `b2-audit` command
- replay and disagreement artifact output
- rho/teacher/heuristic coverage logs

Expected docs artifacts:

- B2 diagnostic docs
- rebuild log entry with findings

Validation commands:

```powershell
uv run --extra dev --extra sim python -m weiss_rl.cli b2-audit --run-dir runs/<main_run> --episodes-jsonl runs/<main_run>/eval/final_eval/episodes.jsonl --policy-id <focal_policy_id>
```

Stop/go criteria:

- Go when disagreement artifacts explain whether the issue is tactical policy,
  action mapping, reward, eval, or opponent loading.

Risks:

- Without replay/state-level artifacts, B2 flatline remains ambiguous.

### Milestone 8: Final Eval, Figures, and Artifact Contract

Objective:

- Produce reproducible final eval and thesis figure artifacts.

Expected code artifacts:

- explicit B0-B4 policy-set selection
- confidence interval exports
- canonical figure data exports
- readiness checker updates

Expected docs artifacts:

- final eval docs
- figure regeneration docs

Validation commands:

```powershell
uv run --extra dev --extra sim python -m weiss_rl.cli eval-final --run-dir runs/<main_run> --b1-run runs/<b1_run>
uv run --extra dev python -m weiss_rl.cli figures --run-dir runs/<main_run> --format png --format pdf
```

Stop/go criteria:

- Go when final outputs are regenerable from saved run artifacts.

Risks:

- Legacy figure scripts can produce detached, hard-to-defend outputs.

### Milestone 9: Cleanup and Archive

Objective:

- Remove or quarantine stale paths after replacement coverage exists.

Expected code artifacts:

- archived configs/scripts
- removed dead compatibility layers where safe
- updated tests

Expected docs artifacts:

- archive index
- migration notes

Validation commands:

```powershell
uv run python -m pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy python
```

Stop/go criteria:

- Go when tests/lint/type checks pass or exceptions are documented.

Risks:

- Aggressive cleanup can break old artifact inspection unless archive/read-only
  workflows are preserved.

### Milestone 10: Thesis Acceptance and Final Run

Objective:

- Run the accepted thesis workflow end to end.

Expected code artifacts:

- no new code unless blockers appear

Expected docs artifacts:

- final rebuild log entry
- final report artifact manifest
- command transcript summary

Validation commands:

```powershell
uv sync --extra dev --extra sim
uv run python -m pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy python
uv run --extra dev --extra sim python -m weiss_rl.cli train-b1 --run-label thesis_b1_<date> --profile thesis-local
uv run --extra dev --extra sim python -m weiss_rl.cli train-main --run-label thesis_main_<date> --profile thesis-local --b1-run runs/<b1_run>
uv run --extra dev --extra sim python -m weiss_rl.cli eval-final --run-dir runs/<main_run> --b1-run runs/<b1_run>
uv run --extra dev --extra sim python -m weiss_rl.cli b2-audit --run-dir runs/<main_run> --episodes-jsonl runs/<main_run>/eval/final_eval/episodes.jsonl --policy-id <focal_policy_id>
uv run --extra dev python -m weiss_rl.cli figures --run-dir runs/<main_run> --format png --format pdf
```

Stop/go criteria:

- Done when a fresh reader can reproduce B1, main league, final eval, and
  figures from README/docs with saved artifacts and no hidden flag soup.

Risks:

- Long training may expose structural learning failures; those should trigger
  diagnosis rather than ad hoc hyperparameter chasing.
