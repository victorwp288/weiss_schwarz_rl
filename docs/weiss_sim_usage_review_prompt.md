# Expert Review Prompt: `weiss-sim` Usage in `weiss_schwarz_rl`

You are reviewing a thesis RL codebase that uses the published `weiss-sim` Python package as its simulator/runtime dependency.

Your job is not just to spot bugs. I want a deep architecture review of whether this repo is using `weiss-sim` in the right way:

- correctly
- idiomatically for the simulator's intended API surface
- robustly against contract drift
- efficiently on the hot path
- safely for deterministic thesis-grade train/eval artifacts

Please be opinionated. If the code is over-coupled to simulator internals, say so. If the coupling is justified for performance or determinism, say so. If the repo is reimplementing logic that should live in `weiss-sim`, call that out. If it is relying on private or unstable surfaces, identify them clearly.

## Scope

The codebase is `weiss_schwarz_rl`.

The simulator dependency is pinned in this repo to:

- `weiss-sim==0.8.2` in `pyproject.toml`

The repo explicitly presents `weiss-sim 0.8.2` as the canonical validation/runtime source for simulator-backed runs.

## Important framing

This repo does not use `weiss-sim` in only one way.

It uses it across four distinct layers:

1. As the source of truth for observation/action/spec contracts and provenance.
2. As the low-level batched stepping engine for training and evaluation.
3. As a provider of packed legality/action metadata used by structured policies and heuristics.
4. As a provider of auxiliary exported data such as card tables and replay/episode identity behavior.

Please review all four layers, not just whether the env steps.

## My current high-level read

The repo has a reasonably disciplined integration story:

- it pins the simulator version
- it records simulator provenance and spec bundles into artifacts
- it centralizes most stepping behind wrapper modules
- it has real simulator-backed contract tests
- it differentiates scaffold/demo paths from real simulator-backed runs

But it is also tightly coupled to specific low-level simulator APIs:

- concrete buffer classes
- concrete pool method names
- concrete layout names
- packed legal-id and legal-action-meta behavior
- specific trajectory types
- specific engine-error reset behavior
- parity with simulator episode-key mixing logic

So the main question is not "does it use `weiss-sim` at all?" It clearly does. The main question is whether this coupling is the right amount of coupling, and whether the boundaries are placed in the right modules.

## Upstream `weiss-sim 0.8.2` surfaces this repo depends on

I inspected the pinned package wheel for `weiss-sim 0.8.2`.

The repo depends on these public or quasi-public surfaces:

- package install/runtime presence
- `weiss_sim.__version__`
- `weiss_sim.__file__`
- `weiss_sim.build_info()`
- `weiss_sim.db_info()`
- `weiss_sim.export_spec_bundle()`
- `weiss_sim.spec_bundle()` as a compatibility fallback
- `weiss_sim.export_card_table()`
- `weiss_sim.PASS_ACTION_ID`
- `weiss_sim.fast(...)`
- `weiss_sim.inspect(...)`
- `weiss_sim.make(...)`
- `weiss_sim.make_pool(...)`
- `weiss_sim.rl.reset_rl(...)`
- `weiss_sim.rl.step_rl(...)`
- `weiss_sim.rl.step_rl_sample_from_logits(...)`
- `weiss_sim.rl.step_rl_sample_from_logits_with_logp(...)`
- `weiss_sim.BatchOutMinimal`
- `weiss_sim.BatchOutMinimalNoMask`
- `weiss_sim.BatchOutMinimalI16LegalIds`
- `weiss_sim.BatchOutTrajectoryI16LegalIds`
- `EnvPool` methods such as:
  - `reset_done_into`
  - `reset_done_into_i16_legal_ids`
  - `reset_indices_with_episode_seeds_into`
  - `reset_indices_with_episode_seeds_into_i16_legal_ids`
  - `legal_action_ids_into`
  - `legal_action_meta_into`
  - `auto_reset_on_error_codes_into`
  - `auto_reset_on_error_codes_into_nomask`
  - `timing_counters`
  - `reset_timing_counters`
  - `set_timing_enabled`
  - `rollout_heuristic_public_into_i16_legal_ids`
- `weiss_sim.decode_action_id(...)` as a fallback path in the interactive tool

The repo also mirrors or tests parity with `weiss_sim.runner` internals:

- `_mix_u64`
- `_episode_key`

That last part is especially important to evaluate from a maintenance perspective.

Important upstream context from the pinned package source:

- `weiss_sim._buffers.make_pool(...)` is described by the simulator itself as the recommended low-level API for high-throughput stepping.
- `weiss_sim.runner.WeissEnv` is a higher-level minimal wrapper around the pool.

So please do not assume that "anything below `weiss_sim.make()` is unsupported." The question is which low-level surfaces are genuinely public/stable enough to depend on this heavily.

## Where the repo uses `weiss-sim`

### 1. Dependency pinning, docs, and operator guidance

Files:

- `pyproject.toml`
- `README.md`
- `docs/getting_started.md`
- `docs/runtime_modes.md`
- `docs/artifact_contract.md`
- `docs/troubleshooting.md`
- `python/scripts/README.md`
- `init.sh`

What happens:

- The repo pins `weiss-sim==0.8.2`.
- Docs present simulator-backed runs as the canonical thesis path.
- Docs distinguish between scaffold/demo modes and real simulator-backed modes.
- Operator guidance assumes the simulator is either installed in the current env or discoverable through `WEISS_SIM_PYTHONPATH` and optionally `WEISS_SIM_PYTHON`.

Why it uses the simulator here:

- to define the canonical runtime
- to make run artifacts reproducible and reviewable
- to avoid confusing toy/demo artifacts with real thesis outputs

Review question:

- Is this pin-and-provenance story strong enough, or should the repo lock itself even more tightly to a known simulator build/config fingerprint?

### 2. Simulator provenance and spec-bundle capture

Files:

- `python/weiss_rl/simulator_contract.py`
- `python/weiss_rl/spec.py`
- `python/scripts/train.py`
- `python/scripts/eval.py`

What happens:

- The repo probes `weiss_sim` in a subprocess and records:
  - version
  - module file
  - `build_info()`
  - `db_info()`
  - `export_spec_bundle()`
- It parses the returned spec bundle and computes a canonical SHA-256 over the bundle JSON.
- Train/eval startup verify that the observed spec bundle matches the expected compatibility hash or bundle hash.
- The code supports a split setup where provenance may be collected via another interpreter / `PYTHONPATH`, but canonical runtime stepping still requires a simulator importable in the active interpreter.

Why it uses the simulator here:

- to freeze the observation/action contract
- to reject mixed simulator contracts
- to persist reproducibility-critical runtime metadata into artifacts

My read:

- This is one of the strongest parts of the integration.
- The repo correctly treats the spec bundle as a contract, not a convenience.

Review questions:

- Is the subprocess probing strategy appropriate, or does it create avoidable "probe env vs active runtime env" ambiguity?
- Should provenance capture and active-runtime capability checks be unified more tightly?

### 3. PASS action contract and legality assumptions

Files:

- `python/weiss_rl/masking.py`
- `python/weiss_rl/envs/decision_env.py`
- `python/scripts/train.py`
- `python/scripts/eval.py`

What happens:

- The repo hardcodes `PASS_ACTION_ID = 51` locally.
- It validates that the imported simulator exposes the same `PASS_ACTION_ID`.
- The training/eval stack also reads `pass_action_id` from the exported simulator spec bundle and threads it through runtime components.
- Empty-legal rows are handled by forcing PASS.

Why it uses the simulator here:

- because legal-action handling is a core RL contract
- because empty-legal fallback must match simulator semantics

My read:

- The validation is good.
- The duplication is still a compatibility risk because the repo has both a hardcoded constant and a spec-bundle-driven value.

Review question:

- Should the repo eliminate the local constant and derive PASS exclusively from the simulator contract everywhere?

### 4. Env construction and profile mapping

Files:

- `python/weiss_rl/envs/pool_factory.py`
- `python/scripts/train.py`
- `python/scripts/play_vs_model.py`
- `python/weiss_rl/eval/simulator_runner.py`

What happens:

- The repo builds simulator env configs from stack configs.
- It maps repo-level profiles to simulator entrypoints/layout choices.
- Current mapping in repo:
  - `debug` -> `inspect`, `mask_u8`, `i32`
  - `balanced` -> `fast`, `mask_u8`, `i16`
  - `fast` -> `fast`, `ids_u16`, `i16`
- It uses `make_env_pool_from_config(...)` to return a pool plus a layout name.
- It has compatibility shims for signature drift such as:
  - `curriculum_json` vs `curriculum`
  - calling `weiss_sim.make(...)` when entrypoints are wrapped

Why it uses the simulator here:

- to translate training/eval intent into concrete simulator pool construction
- to select hot-path legality representation
- to carry reward/curriculum/deck settings into the simulator

My read:

- This is a deliberate abstraction boundary.
- But it duplicates some upstream profile/layout policy rather than delegating it entirely to `weiss-sim`.

Important review point:

- Upstream `weiss-sim` itself already has profile concepts in its low-level helpers.
- This repo introduces its own profile mapping on top.
- Please assess whether this duplication is justified or whether the repo should lean harder on simulator-native profile semantics.

### 5. The main stepping boundary: `DecisionBoundaryEnv`

Files:

- `python/weiss_rl/envs/decision_env.py`
- tests in `python/weiss_rl/tests/test_decision_env.py`
- real-sim smoke in `python/weiss_rl/tests/test_rl_step_layout_contract_smoke.py`

What happens:

- This class is the core simulator wrapper used by the RL code.
- It wraps:
  - `weiss_sim.rl.reset_rl`
  - `weiss_sim.rl.step_rl`
  - `weiss_sim.rl.step_rl_sample_from_logits`
  - `weiss_sim.rl.step_rl_sample_from_logits_with_logp`
- It supports two legality modes:
  - dense mask
  - packed ids + offsets
- It converts simulator outputs into a stable local `DecisionBoundaryBatch`.
- It adds:
  - action validation
  - counter extraction
  - episode seed/key extraction
  - optional fallback episode-key derivation
  - best-effort engine-error reset
  - simulator timing collection

Why it uses the simulator here:

- to normalize the simulator into a learner-friendly batch API
- to keep training/eval code mostly independent from raw `EnvPool` method names

This is the single most important integration point to review.

Please evaluate in particular:

- Is `DecisionBoundaryEnv` the right abstraction boundary?
- Is it thin enough?
- Is it too simulator-specific to count as a reusable boundary?
- Is it correctly using public `weiss-sim` low-level APIs?

#### Specific coupling inside `DecisionBoundaryEnv`

The wrapper depends on exact simulator class/method names:

- output buffer classes:
  - `BatchOutMinimal`
  - `BatchOutMinimalI16LegalIds`
  - `BatchOutMinimalNoMask`
- pool methods:
  - `reset_done_into`
  - `reset_done_into_i16_legal_ids`
  - `reset_indices_with_episode_seeds_into`
  - `reset_indices_with_episode_seeds_into_i16_legal_ids`
  - `auto_reset_on_error_codes_into`
  - `auto_reset_on_error_codes_into_nomask`
  - `legal_action_ids_into`
  - `legal_action_meta_into`

This is not accidental. The repo is intentionally sitting on top of the low-level simulator API, not just the high-level `WeissEnv` API.

Review questions:

- Is this the correct choice for throughput-sensitive RL code?
- Which of these calls are stable/public enough to rely on?
- Which of them are too implementation-shaped?

#### Specific risk: custom best-effort reset for packed legality

For `ids_offsets`, the repo manually repairs faulted rows after a best-effort reset by:

1. calling the simulator resetter in a no-mask path
2. copying common fields back into the current step buffer
3. calling `legal_action_ids_into(...)`
4. optionally calling `legal_action_meta_into(...)`
5. rebuilding merged packed legality row-by-row

This is clever, but it is also subtle.

Please evaluate whether:

- this is the right way to recover packed-id rows after engine faults
- the simulator already provides a more canonical helper for this
- the merge logic is too risky to own in the RL repo

### 6. Queue runtime and hot-path collection

File:

- `python/weiss_rl/runtime.py`

What happens:

- The queue runtime is the repo's main single-node train collector/runtime.
- It uses `DecisionBoundaryEnv` and packed legality heavily.
- It assumes simulator-provided `legal_action_meta` exists and is meaningful.
- It records simulator timing counters.
- It contains a special native rollout path for heuristic-public play:
  - `pool.rollout_heuristic_public_into_i16_legal_ids(...)`
  - `weiss_sim.BatchOutTrajectoryI16LegalIds(...)`

Why it uses the simulator here:

- for throughput
- for minimizing Python overhead on actor collection
- for packed legal-id hot paths
- for heuristic-public teacher/opponent rollouts

My read:

- This is where the integration becomes deeply simulator-shaped.
- The runtime is not using a generic env API. It is using simulator-native rollout and packed legality semantics directly.

Please review whether this is:

- the right performance tradeoff
- a reasonable specialization for this thesis codebase
- too much logic in the RL repo rather than in the simulator package

#### Specific risk: native heuristic rollout dependency

The runtime directly requires:

- `rollout_heuristic_public_into_i16_legal_ids`
- `BatchOutTrajectoryI16LegalIds`

That means the collector is not just depending on general RL stepping, but on a specialized simulator rollout surface designed around one evaluation/teacher policy family.

Please assess whether this is an appropriate dependency boundary.

### 7. Spec-driven heuristics, action catalogs, and observation layout parsing

Files:

- `python/weiss_rl/action_catalog.py`
- `python/weiss_rl/observation_layout.py`
- `python/weiss_rl/eval/heuristic_public.py`
- `python/weiss_rl/eval/simulator_runner.py`
- `python/scripts/train.py`
- `python/scripts/eval.py`

What happens:

- The repo parses the simulator-exported spec bundle to build:
  - action catalogs
  - public observation layouts
  - heuristic-public policies
- It uses `action_meta_v1` and packed `legal_action_meta` to do fast heuristic action choice without fully decoding every action id.
- It uses the same spec data to build structured policy heads and replay/eval tools.

Why it uses the simulator here:

- to avoid hardcoding action-space semantics
- to make heuristics/spec consumers contract-aware
- to keep model/eval tooling aligned with simulator-exported schemas

My read:

- Conceptually this is good.
- It is better than scattering hardcoded action ids everywhere.
- The expert review should focus on whether the repo is parsing the right level of simulator schema, and whether it should trust `action_meta_v1` this deeply.

### 8. Model-side auxiliary use: exported card table

Files:

- `python/weiss_rl/card_table.py`
- `python/weiss_rl/model.py`

What happens:

- The model code opportunistically calls `weiss_sim.export_card_table()`.
- It turns that into dense static card features used by structured policy heads.

Why it uses the simulator here:

- to enrich card embeddings with simulator DB-derived metadata
- to avoid maintaining a separate feature-export pipeline

Review questions:

- Is this the right ownership boundary?
- Should card-table export be treated as part of the simulator contract more explicitly?
- Is it risky that model feature construction depends on runtime-exported DB content rather than a separately versioned feature artifact?

### 9. Interactive/dev surfaces

Files:

- `python/scripts/play_vs_model.py`
- `examples/run_loop_example.py`

What happens:

- `play_vs_model.py` uses the simulator-backed env and reads the simulator contract from the run.
- It falls back to `weiss_sim.decode_action_id(...)` if the local action catalog decode is insufficient.
- `examples/run_loop_example.py` demonstrates the high-level `weiss_sim.make(...)` / `WeissEnv` path.

Why it uses the simulator here:

- interactive debugging
- local sanity checking
- providing a minimal example of direct simulator usage

Important review point:

- The example path uses the high-level API.
- The production train/eval runtime uses low-level pool + RL wrappers.

Please assess whether that split is healthy documentation or confusing divergence.

### 10. Real simulator-backed tests

Files:

- `python/weiss_rl/tests/test_rl_step_layout_contract_smoke.py`
- `python/weiss_rl/tests/test_decision_env.py`
- `python/weiss_rl/tests/test_actor_worker.py`
- `python/weiss_rl/tests/test_pool_factory.py`
- `python/weiss_rl/tests/test_masking.py`
- `python/weiss_rl/tests/test_simulator_contract.py`
- `python/weiss_rl/tests/test_entrypoints.py`

What happens:

- The repo has real simulator-backed smoke tests for supported layouts.
- It checks mask and ids-offset stepping equivalence.
- It checks episode-key parity against simulator runner logic.
- It validates contract probing behavior.
- It also uses simulator stubs/fakes for unit-level boundary testing.

Why it uses the simulator here:

- to pin the low-level contract
- to catch layout drift
- to defend the wrapper behavior

My read:

- This is a strong sign that the authors know they are depending on subtle simulator behavior.
- The question is whether the tests cover the right invariants, not whether tests exist.

## Specific things I want you to judge

Please give a verdict on each item below.

### A. Is the code using the right simulator API level?

Should this repo be built around:

- high-level `weiss_sim.make()` / `WeissEnv`
- low-level `make_pool()` / `rl.*`
- direct `EnvPool` methods as it does now

If the answer is "mixed," explain the right split.

### B. Is the repo over-coupled to concrete class/method names?

Examples:

- `BatchOutMinimalI16LegalIds`
- `BatchOutTrajectoryI16LegalIds`
- `reset_done_into_i16_legal_ids`
- `rollout_heuristic_public_into_i16_legal_ids`
- `legal_action_meta_into`

I want a direct assessment of whether this is acceptable public-surface usage or fragile implementation coupling.

### C. Is the profile/layout abstraction well-designed?

The repo has its own `debug` / `balanced` / `fast` profile mapping in `envs/pool_factory.py`.

Please assess whether:

- this is a good local abstraction
- it is duplicating upstream logic in a risky way
- it would be better to delegate more directly to simulator-native profiles

### D. Is the spec-bundle contract story correct and sufficient?

Please assess:

- exported spec bundle verification
- bundle hash vs compatibility hash usage
- runtime provenance capture
- whether the split between probe interpreter and active stepping interpreter is acceptable

### E. Is the legality handling correct?

Please look closely at:

- PASS handling
- empty-legal fallback
- packed legal ids
- sorted legal-id assumptions
- `legal_action_meta`
- dense mask vs packed-ids parity

### F. Is the engine-fault/reset behavior sound?

Especially:

- `engine_status_policy`
- `best_effort_reset`
- manual packed-legality repair after reset

### G. Is the runtime putting simulator-specific logic in the right place?

For example:

- native heuristic rollout support
- episode-key derivation parity code
- timing counter plumbing
- packed metadata handling in the queue runtime

Should more of this logic live inside `weiss-sim` rather than the RL repo?

### H. Is the repo relying on private or unstable simulator internals?

Please call out anything you consider private, semi-private, or too implementation-shaped, especially:

- `weiss_sim.runner._mix_u64`
- `weiss_sim.runner._episode_key`
- assumptions about output buffer structure or lifetime

### I. Is the model-side use of `export_card_table()` a good idea?

If yes, explain why.

If no, explain what a better ownership/versioning boundary would look like.

## Preliminary strengths I see

You should challenge these if you disagree.

- The repo treats `weiss-sim` as a contract source, not just a black-box step function.
- It pins a concrete simulator version.
- It records simulator provenance into artifacts.
- It has a reasonably centralized wrapper layer instead of importing simulator APIs everywhere in random training code.
- It has real layout/contract smoke tests against the simulator.
- It uses spec-exported metadata to avoid unnecessary hardcoding in eval/heuristic/model code.

## Preliminary concerns I see

You should validate or refute these.

- The repo may be too coupled to low-level simulator method/class names for long-term maintainability.
- It duplicates some upstream profile/layout policy.
- It reimplements recovery logic that may belong in the simulator.
- It has both local and simulator-derived notions of PASS action identity.
- It mirrors simulator episode-key logic in local code and tests against simulator runner internals.
- It relies heavily on packed legality metadata, which may be correct but raises the maintenance bar.
- It uses the simulator for model-side feature export, which may blur runtime contract vs data artifact boundaries.

## Output format I want from you

Please answer in four sections:

1. `Overall Verdict`
   - Is the repo using `weiss-sim` in a sound way overall?
   - Short answer first.

2. `What Is Good`
   - Concrete integration choices that are correct or well-designed.

3. `What Is Risky or Wrong`
   - Concrete issues, ordered by severity.
   - Be explicit about whether each issue is:
     - correctness risk
     - maintenance risk
     - performance risk
     - reproducibility risk

4. `Recommended Refactors`
   - What should move into `weiss-sim`
   - What should stay in this repo
   - What abstractions should change
   - Which changes are urgent vs optional

If you think the current architecture is mostly right, say that. If you think it should be radically simplified around a different simulator API layer, say that too.

## Main files to inspect

- `pyproject.toml`
- `python/weiss_rl/simulator_contract.py`
- `python/weiss_rl/spec.py`
- `python/weiss_rl/masking.py`
- `python/weiss_rl/card_table.py`
- `python/weiss_rl/envs/pool_factory.py`
- `python/weiss_rl/envs/decision_env.py`
- `python/weiss_rl/runtime.py`
- `python/weiss_rl/action_catalog.py`
- `python/weiss_rl/observation_layout.py`
- `python/weiss_rl/eval/heuristic_public.py`
- `python/weiss_rl/eval/simulator_runner.py`
- `python/scripts/train.py`
- `python/scripts/eval.py`
- `python/scripts/play_vs_model.py`
- `examples/run_loop_example.py`
- `python/weiss_rl/tests/test_rl_step_layout_contract_smoke.py`
- `python/weiss_rl/tests/test_decision_env.py`
- `python/weiss_rl/tests/test_actor_worker.py`
- `python/weiss_rl/tests/test_masking.py`
- `python/weiss_rl/tests/test_simulator_contract.py`

## One-sentence summary

This repo is not merely "using a simulator"; it is building a thesis-grade RL runtime around `weiss-sim` as a versioned contract, low-level stepping engine, legality metadata provider, and provenance source, and I need you to judge whether that integration boundary is well-chosen.
