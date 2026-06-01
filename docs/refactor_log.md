# Refactor Log

This log records behavior-preserving refactor work. Historical run outputs, checkpoints, thesis figures, `vast_artifacts/`, and `run_logs/` are treated as read-only.

## 2026-05-10 - Baseline, Compatibility Fixes, and First CLI Extraction

### Scope

- Established a local validation baseline on Windows/PowerShell.
- Fixed pre-existing verification failures without changing training or evaluation semantics.
- Extracted the `train.py` argument parser into `weiss_rl.training.cli` while preserving the path-based public script.
- Added documentation and contributor guidance for the current architecture and safety contracts.

### Baseline Commands and Results

| Command | Result |
| --- | --- |
| `uv sync --extra dev` | Passed. Removed `weiss-sim==0.8.2` from the dev-only environment because the simulator extra was not requested. |
| `uv run python -c "import weiss_rl; print(weiss_rl.__all__)"` | Passed, printed `['load_stack_config', 'assert_spec_compatibility']`. |
| `uv run python python/scripts/verify_repo.py` | Initial failure in `ruff check`: unused import and unused local in `parallel_final_eval.py`, stale import formatting in `targeted_confirm_eval.py`. |
| `uv run python python/scripts/verify_repo.py` | Second failure in `ruff format --check`: `parallel_final_eval.py`, `heuristic_public.py`, and `runtime.py` needed formatting. |
| `uv run python python/scripts/verify_repo.py` | Third failure in pytest: 3 tests failed around partial `QueueRuntime` test doubles and fake eval models. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime.py::test_overwrite_central_outputs_with_opponents_only_touches_non_mirror_rows python/weiss_rl/tests/test_runtime.py::test_overwrite_central_outputs_with_batched_opponents_groups_rows_across_actors python/weiss_rl/tests/test_snapshot_registry.py::test_simulator_eval_runner_uses_learner_scoring_mode` | Passed after compatibility fixes. |
| `uv run python python/scripts/verify_repo.py` | Passed after the compatibility fixes: placeholder gate, ruff, ruff format check, mypy, vulture, 750 tests passed, 16 skipped, wrapper dry-runs passed. |
| `uv run python python/scripts/train.py --help` | Passed; inspected public train CLI surface. |
| `uv run python python/scripts/eval.py --help` | Passed; inspected public eval CLI surface. |
| `uv run python python/scripts/train.py --stack-config configs/stack_smoke.yaml --run-label refactor_stack_smoke_20260510 --num-envs 1 --unroll-length 1 --max-updates 1 --runtime-mode train_ordered --device cpu` | Passed. Wrote a new scaffold-only manifest under `runs/refactor_stack_smoke_20260510`. No learner rollout executed. |
| `uv run python python/scripts/train.py --stack-config configs/presets/structured_acceptance_standard.yaml --public-demo --run-label refactor_public_demo_20260510` | Passed. Wrote synthetic public-demo artifacts under `runs/refactor_public_demo_20260510`. |
| `uv run python python/scripts/eval.py --stack-config configs/presets/structured_acceptance_standard_thesis_eval.yaml --public-demo --run-dir runs/refactor_public_demo_20260510 --public-demo-paired-seeds 2 --public-demo-bootstrap-samples 20 --skip-metagame --skip-figures --skip-readiness` | Passed. Wrote demo-only `eval/final_eval/summary.json`. |
| `uv run python python/scripts/write_paper_readiness_fixture.py --run-dir runs/refactor_paper_readiness_fixture_20260510; uv run python python/scripts/paper_readiness_check.py --run-dir runs/refactor_paper_readiness_fixture_20260510` | Passed. Wrote a fresh readiness fixture and validated the paper-readiness contract. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_script_entrypoint_smokes.py::test_train_cli_parser_preserves_public_defaults_and_aliases python/weiss_rl/tests/test_script_entrypoint_smokes.py::test_train_entrypoint_applies_profile_flags_before_hashing` | Passed after CLI parser extraction. |
| `uv run python python/scripts/verify_repo.py` | Final checkpoint passed after the CLI extraction and docs updates: placeholder gate, ruff, ruff format check, mypy, vulture, 751 tests passed, 16 skipped, wrapper dry-runs passed. |

### Fixes Applied

- Removed an unused `PayoffFoldScheme` import and unused manifest read in `python/scripts/parallel_final_eval.py`.
- Ran `ruff format` on files that had formatting drift.
- Made `QueueRuntime._heuristic_opponent_policy` use `getattr(self, "_spec_bundle", None)` so characterization tests that construct partial runtime objects remain valid.
- Made `SimulatorEvalRunner` tolerate minimal fake model objects that do not implement `.to()` or `.eval()`, while preserving normal model device/eval preparation for real torch modules.
- Added `python/weiss_rl/training/cli.py` and `python/weiss_rl/training/__init__.py`.
- Replaced inline parser construction in `python/scripts/train.py` with `build_train_parser()`.
- Added `test_train_cli_parser_preserves_public_defaults_and_aliases` to pin public defaults and compatibility aliases.

### Behavior Changes

No intended training, evaluation, simulator, reward, legal-action, checkpoint, RNG, league, promotion, metric, or artifact semantics changed.

The only runtime-observable changes are compatibility broadening:

- partial `QueueRuntime` test doubles no longer need `_spec_bundle`;
- minimal eval fake models can be used in tests without `.to()` or `.eval()`.

### Artifacts Created

- `runs/refactor_stack_smoke_20260510`
- `runs/refactor_public_demo_20260510`
- `runs/refactor_paper_readiness_fixture_20260510`

These are new local validation artifacts and are not thesis result artifacts.

### Remaining Risks and Next Hypotheses

- `runtime.py`, `model.py`, `train.py`, and `impala_learner.py` remain large. The next safe code refactor should extract training persistence or checkpoint helpers behind compatibility wrappers.
- Simulator-extra validation was not rerun in this checkpoint after `uv sync --extra dev` removed `weiss-sim` from the environment. Run `uv sync --extra dev --extra sim` and the simulator contract tests before touching simulator boundaries.
- Full benchmark claims were not made. Any performance refactor still needs scoped measurements by runtime mode and league/no-league gate.
- The docs now describe the current architecture, but deeper API docs should be kept in sync as modules are extracted.

## 2026-05-10 - Simulator Gate and Checkpoint Helper Extraction

### Scope

- Closed the simulator-extra validation gap from the previous checkpoint.
- Extracted pure checkpoint tracker/path helper logic into `weiss_rl.training.checkpoints`.
- Preserved the existing `python/scripts/train.py` private helper names as compatibility wrappers.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv sync --extra dev --extra sim` | Passed. Installed `weiss-sim==0.8.1`. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_simulator_contract.py python/weiss_rl/tests/test_rl_step_layout_contract_smoke.py python/weiss_rl/tests/test_heuristic_public.py -k "simulator_native_heuristic_pool_matches_python_oracle_across_live_steps or simulator_contract or rl_step_layout"` | Passed: 10 passed, 26 deselected. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_snapshot_registry.py::test_resolve_resume_checkpoint_path_defaults_to_latest_alias python/weiss_rl/tests/test_snapshot_registry.py::test_checkpoint_aliases_track_latest_and_best_and_restore_resume_state python/weiss_rl/tests/test_snapshot_registry.py::test_train_snapshot_persistence_writes_artifact_bundle_and_registry_entry` | Passed: 3 passed. |
| `uv run python -m ruff check python tests examples python/scripts` | Passed after import sorting. |
| `uv run python -m ruff format --check python tests examples python/scripts` | Passed after formatting touched files. |

### Changes

- Added `python/weiss_rl/training/checkpoints.py` with:
  - checkpoint alias filenames and tracker format constants,
  - `relative_path_text`,
  - checkpoint tracker load/write helpers,
  - checkpoint guard log path helper,
  - checkpoint record builder,
  - resume checkpoint path resolver.
- Exported those helpers from `python/weiss_rl/training/__init__.py`.
- Updated `python/scripts/train.py` wrappers to delegate to the package helpers while preserving script-local names used by existing tests.
- Extended `test_resolve_resume_checkpoint_path_defaults_to_latest_alias` so the new package helper and legacy script wrapper are both pinned to the same behavior.

### Behavior Changes

No intended behavior changes. This is a pure extraction behind compatibility wrappers.

### Remaining Risks

- Checkpoint restore payload validation still lives in `python/scripts/train.py`; move it only after adding negative compatibility tests for unsupported format, missing `model_state_dict`, algorithm mismatch, config hash mismatch, and spec hash mismatch.
- Snapshot artifact import/publishing still lives in `python/scripts/train.py`; it is the next likely extraction seam after the checkpoint restore tests exist.

## 2026-05-10 - Checkpoint Restore Contract Tests and Validation Extraction

### Scope

- Added negative checkpoint restore characterization tests before moving restore validation logic.
- Extracted checkpoint payload contract validation into `weiss_rl.training.checkpoints`.
- Kept model, optimizer, grad-scaler, guidance, and counter restore side effects in `python/scripts/train.py`.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_snapshot_registry.py::test_restore_checkpoint_rejects_invalid_payload_contracts python/weiss_rl/tests/test_snapshot_registry.py::test_restore_checkpoint_allows_config_hash_mismatch_only_with_escape_hatch python/weiss_rl/tests/test_snapshot_registry.py::test_checkpoint_aliases_track_latest_and_best_and_restore_resume_state` | Passed: 8 passed. |
| `uv run python -m ruff check python tests examples python/scripts` | Passed after import sorting. |
| `uv run python -m ruff format --check python tests examples python/scripts` | Passed after formatting touched files. |

### Changes

- Added negative tests for:
  - non-dict checkpoint payloads,
  - unsupported checkpoint format,
  - config hash mismatch,
  - spec hash mismatch,
  - algorithm mismatch,
  - missing `model_state_dict`.
- Added a positive test for `WEISS_RL_ALLOW_RESUME_CONFIG_MISMATCH=1`.
- Added `CheckpointPayloadContract` and `validate_checkpoint_payload_contract()` to `python/weiss_rl/training/checkpoints.py`.
- Updated `_restore_learner_from_checkpoint()` to delegate validation to the package helper while preserving warning text and restore side effects.

### Behavior Changes

No intended behavior changes. The new tests pin the existing restore error messages and escape-hatch behavior before and after extraction.

### Remaining Risks

- Snapshot artifact writing/importing and B1/seed snapshot import logic still live in `python/scripts/train.py`.
- Checkpoint write payload construction still lives in `python/scripts/train.py`; extracting it should wait until payload compatibility tests are similarly explicit.

## 2026-05-10 - Snapshot Artifact and Registry Retention Extraction

### Scope

- Extracted snapshot artifact writing, snapshot JSON writing, snapshot hashing, registry retention, and pruned-artifact deletion helpers into `weiss_rl.training.snapshots`.
- Preserved all `python/scripts/train.py` helper names as wrappers.
- Left higher-level B1 baseline import and seed snapshot import orchestration in `python/scripts/train.py`.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_snapshot_registry.py::test_train_snapshot_persistence_writes_artifact_bundle_and_registry_entry python/weiss_rl/tests/test_snapshot_registry.py::test_train_snapshot_retention_prunes_old_snapshot_artifacts python/weiss_rl/tests/test_snapshot_registry.py::test_run_snapshot_promotion_gate_marks_passed_candidate_as_champion python/weiss_rl/tests/test_snapshot_registry.py::test_run_snapshot_promotion_gate_skips_during_warmup python/weiss_rl/tests/test_snapshot_registry.py::test_run_snapshot_promotion_gate_uses_effective_update_for_warmup` | Passed: 5 passed. |
| `uv run python -m ruff check python tests examples python/scripts` | Passed. |
| `uv run python -m ruff format --check python tests examples python/scripts` | Passed after formatting `train.py`. |

### Changes

- Added `python/weiss_rl/training/snapshots.py` with:
  - `sha256_file`,
  - `write_json_file`,
  - `write_snapshot_artifact`,
  - `sync_snapshot_registry_retention`,
  - `snapshot_artifact_dir_for_prune`,
  - `delete_pruned_snapshot_artifacts`,
  - `save_snapshot_registry_with_retention`,
  - `persist_snapshot_registry_entry`.
- Exported snapshot helpers from `python/weiss_rl/training/__init__.py`.
- Updated `python/scripts/train.py` wrappers to delegate to the package helpers.

### Behavior Changes

No intended behavior changes. Existing snapshot persistence, retention, and promotion-gate tests passed through the compatibility wrappers.

### Remaining Risks

- B1 no-league baseline import and seed snapshot import still contain repeated metadata-writing logic in `python/scripts/train.py`.
- Extracting those import paths should be done only after pinning their metadata payloads and mismatch failure cases.

## 2026-05-10 - Imported Snapshot Artifact Helper Extraction

### Scope

- Extracted the common imported-snapshot artifact writer used by B1 baseline import and seed snapshot import.
- Preserved the B1 metadata format `imported_train_snapshot_metadata_v1`.
- Preserved the seed metadata format `seeded_train_snapshot_metadata_v1` and `seeded_from_external_registry` payload marker.
- Left B1/seed import validation and registry orchestration in `python/scripts/train.py`.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_snapshot_registry.py::test_ensure_noleague_baseline_anchor_imports_frozen_snapshot_once python/weiss_rl/tests/test_snapshot_registry.py::test_ensure_noleague_baseline_anchor_rejects_non_b1_imported_run python/weiss_rl/tests/test_snapshot_registry.py::test_ensure_noleague_baseline_anchor_rejects_imported_environment_mismatch python/weiss_rl/tests/test_snapshot_registry.py::test_import_seed_snapshot_pool_imports_external_snapshots_and_champions python/weiss_rl/tests/test_snapshot_registry.py::test_import_seed_snapshot_pool_rejects_environment_mismatch` | Passed: 5 passed. |
| `uv run python -m ruff check python tests examples python/scripts` | Passed after removing now-unused imports. |
| `uv run python -m ruff format --check python tests examples python/scripts` | Passed after formatting `train.py`. |

### Changes

- Added `write_imported_snapshot_artifact()` to `python/weiss_rl/training/snapshots.py`.
- Exported the helper from `python/weiss_rl/training/__init__.py`.
- Replaced repeated imported snapshot payload/metadata writing in `_import_noleague_baseline_anchor()` and `_import_seed_snapshot_pool()` with the shared helper.

### Behavior Changes

No intended behavior changes. Existing B1 import, seed import, and mismatch tests passed after extraction.

### Remaining Risks

- B1 and seed import contract validation still lives in `python/scripts/train.py`.
- Checkpoint write payload construction still lives in `python/scripts/train.py`.

## 2026-05-10 - Checkpoint Payload Builder Extraction

### Scope

- Added a characterization test for the exact checkpoint payload key set and core values written by `_write_checkpoint()`.
- Extracted pure checkpoint payload construction into `weiss_rl.training.checkpoints`.
- Preserved checkpoint file writing and learner/model state collection in `python/scripts/train.py`.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_snapshot_registry.py::test_write_checkpoint_payload_shape_is_stable` | Passed before extraction. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_snapshot_registry.py::test_write_checkpoint_payload_shape_is_stable python/weiss_rl/tests/test_snapshot_registry.py::test_restore_checkpoint_rejects_invalid_payload_contracts python/weiss_rl/tests/test_snapshot_registry.py::test_checkpoint_aliases_track_latest_and_best_and_restore_resume_state` | Passed after extraction: 8 passed. |
| `uv run python -m ruff check python tests examples python/scripts` | Passed after import sorting. |
| `uv run python -m ruff format --check python tests examples python/scripts` | Passed after formatting touched files. |

### Changes

- Added `MINIMAL_TRAIN_CHECKPOINT_FORMAT` and `build_minimal_train_checkpoint_payload()` to `python/weiss_rl/training/checkpoints.py`.
- Reused `MINIMAL_TRAIN_CHECKPOINT_FORMAT` in checkpoint restore validation.
- Exported the builder and format constant from `python/weiss_rl/training/__init__.py`.
- Updated `_write_checkpoint()` to delegate payload construction to the package helper.

### Behavior Changes

No intended behavior changes. The characterization test pins the payload key set and representative values.

### Remaining Risks

- Higher-level training loop orchestration, dev-eval scheduling, and promotion gate wiring still live in `python/scripts/train.py`.
- Runtime and model god files remain large and should be decomposed only behind similarly focused tests.

## 2026-05-10 - Checkpoint Guard Scoring Extraction

### Scope

- Extracted pure dev-eval checkpoint guard scoring, confidence, ineligibility, candidate metric, and best-checkpoint promotion helpers into `weiss_rl.training.checkpoint_guard`.
- Preserved the old private `python/scripts/train.py` helper names as compatibility wrappers.
- Left confirmatory dev-eval request construction and training-loop alias publishing in `python/scripts/train.py`.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_train_stall_monitor.py` | Passed: 15 passed, 14 third-party warnings. |
| `uv run ruff check python/weiss_rl/training/checkpoint_guard.py python/weiss_rl/training/__init__.py python/scripts/train.py` | Passed after import sorting. |
| `uv run ruff format --check python/weiss_rl/training/checkpoint_guard.py python/weiss_rl/training/__init__.py python/scripts/train.py` | Passed after formatting `train.py`. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 773 pytest tests passed, 2 skipped, wrapper dry-runs passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 773 pytest tests passed, 2 skipped, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/training/checkpoint_guard.py` with:
  - `dev_eval_aggregate_score`,
  - `dev_eval_worst_truncation_rate`,
  - reason-rate helpers for no-progress and natural timeouts,
  - `dev_eval_confidence_stats`,
  - `dev_eval_ineligibility_reasons`,
  - `checkpoint_candidate_metric`,
  - `should_promote_best_checkpoint`.
- Exported the helpers from `python/weiss_rl/training/__init__.py`.
- Updated `python/scripts/train.py` wrappers to delegate to the package helpers.

### Behavior Changes

No intended behavior changes. Existing stall monitor and checkpoint promotion tests passed through the preserved compatibility wrappers.

### Remaining Risks

- Confirmatory dev-eval request construction still lives in `python/scripts/train.py`.
- The large training loop still mixes runtime orchestration, checkpoint publication, dev-eval scheduling, and metric logging.

## 2026-05-10 - Confirmatory Dev-Eval Request Extraction

### Scope

- Moved confirmatory periodic dev-eval request construction and deterministic paired-seed expansion into `weiss_rl.training.checkpoint_guard`.
- Preserved the private `python/scripts/train.py` wrapper names used by existing tests and training-loop call sites.
- Kept periodic dev-eval scheduling, opponent resolution, and summary persistence in `python/scripts/train.py`.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_train_stall_monitor.py -k "confirmatory_dev_eval_request or expand_periodic_dev_eval_paired_seeds or dev_eval_ineligibility"` | Passed: 4 passed, 11 deselected, 14 third-party warnings. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_train_stall_monitor.py` | Passed: 15 passed, 14 third-party warnings. |
| `uv run ruff check python/weiss_rl/training/checkpoint_guard.py python/weiss_rl/training/__init__.py python/scripts/train.py` | Passed after import sorting. |
| `uv run ruff format --check python/weiss_rl/training/checkpoint_guard.py python/weiss_rl/training/__init__.py python/scripts/train.py` | Passed after formatting `train.py`. |

### Changes

- Added `confirmatory_dev_eval_target_pairs()`, `expand_periodic_dev_eval_paired_seeds()`, and `confirmatory_dev_eval_request()` to `python/weiss_rl/training/checkpoint_guard.py`.
- Exported those helpers from `python/weiss_rl/training/__init__.py`.
- Updated `python/scripts/train.py` wrappers to delegate to the package helpers.

### Behavior Changes

No intended behavior changes. Existing tests still exercise the original `train.py` private helper names and passed after extraction.

### Remaining Risks

- The full training loop still owns when confirmatory evals run and how the returned request is consumed.
- Larger runtime/model decompositions remain open and need narrow characterization before edits.

## 2026-05-10 - Runtime Topology Helper Extraction

### Scope

- Extracted actor topology resolution and actor seed derivation into `weiss_rl.runtime_topology`.
- Preserved the private `weiss_rl.runtime._resolve_actor_topology()` and `_actor_seed()` wrappers used by existing tests and runtime call sites.
- Left CUDA actor device layout in `runtime.py` because its tests monkeypatch `weiss_rl.runtime.torch.cuda` directly.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime.py -k "build_runtime_config or resolve_actor_topology"` | Passed: 5 passed, 83 deselected. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_repro_ids.py` | Passed: 16 passed. |
| `uv run ruff check python/weiss_rl/runtime.py python/weiss_rl/runtime_topology.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/runtime.py python/weiss_rl/runtime_topology.py` | Passed after formatting `runtime.py`. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 773 pytest tests passed, 2 skipped, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/runtime_topology.py` with `QueueRuntimeMode`, `resolve_actor_topology()`, and `actor_seed()`.
- Updated `python/weiss_rl/runtime.py` to delegate its existing private topology and seed helpers to the new module.

### Behavior Changes

No intended behavior changes. The existing runtime topology tests still exercise the original `runtime.py` private helper name.

### Remaining Risks

- `runtime.py` remains large; shared-memory collector transport, learner-batch array assembly, and CUDA device layout are still potential future extractions.
- Runtime collector loops and legal-action packing remain untouched because they carry higher behavior risk.

## 2026-05-10 - Runtime Batching Primitive Extraction

### Scope

- Extracted pure learner-batch array concatenation helpers and GAE advantage computation into `weiss_rl.runtime_batching`.
- Preserved the private `weiss_rl.runtime._concat_*` and `_gae_advantages()` wrapper names used by existing tests and runtime call sites.
- Deliberately left `_concatenate_legal_actions()` in `runtime.py` because it encodes packed legal-action ordering.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime.py -k "concat_optional_time_major_field or gae_advantages or build_learner_batch or build_ppo_batch"` | Passed: 7 passed, 81 deselected. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime.py -k "concatenate_legal_actions"` | Passed: 2 passed, 86 deselected. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime.py -k "concat_optional_time_major_field or gae_advantages or build_learner_batch or build_ppo_batch or concatenate_legal_actions"` | Passed: 9 passed, 79 deselected. |
| `uv run ruff check python/weiss_rl/runtime.py python/weiss_rl/runtime_batching.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/runtime.py python/weiss_rl/runtime_batching.py` | Passed after formatting `runtime.py`. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 773 pytest tests passed, 2 skipped, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/runtime_batching.py` with:
  - `concat_time_major_field`,
  - `concat_optional_time_major_field`,
  - `concat_batch_major_field`,
  - `gae_advantages`.
- Updated `python/weiss_rl/runtime.py` wrappers to delegate to the package helpers.

### Behavior Changes

No intended behavior changes. Existing learner-batch, PPO-batch, GAE, optional label fill, and packed legal-action ordering tests passed.

### Remaining Risks

- The runtime still contains shared-memory transport and collection loops.
- Packed legal-action concatenation remains in place and should only move with stronger ordering and metadata characterization.

## 2026-05-10 - Model State Helper Extraction

### Scope

- Extracted observation batch validation, hidden-state validation/conversion, acting-seat normalization, acting-seat hidden selection, and acting-seat hidden writes into `weiss_rl.model_state`.
- Preserved the existing private `PolicyValueModel` method names as wrappers.
- Left model forward passes, recurrent-step orchestration, and structured packed candidate scoring in `model.py`.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_contracts.py -k "seat_aware or hidden_state or shape_checks or write_acting_hidden"` | Passed: 14 passed, 36 deselected. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_vtrace.py -k "seat_aware or hidden_state"` | Passed: 5 passed, 12 deselected. |
| `uv run ruff check python/weiss_rl/model.py python/weiss_rl/model_state.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/model.py python/weiss_rl/model_state.py` | Passed after formatting `model.py`. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 773 pytest tests passed, 2 skipped, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/model_state.py` with:
  - `require_observation_batch`,
  - `prepare_hidden_state`,
  - `prepare_seat_hidden_state`,
  - `prepare_acting_seat`,
  - `select_acting_hidden`,
  - `write_acting_hidden`.
- Updated `PolicyValueModel` private methods to delegate to the new helpers.

### Behavior Changes

No intended behavior changes. Hidden-state shape checks, dtype preservation, acting-seat-only updates, and V-trace seat-aware hidden-state tests passed.

### Remaining Risks

- `model.py` remains large, especially structured packed scoring and public heuristic scoring.
- Candidate scoring helpers should only move with packed-candidate and public-heuristic contract tests before and after.

## 2026-05-10 - Shared Collector Transport Extraction

### Scope

- Extracted shared-memory collector slot classes, slot configuration, slot opening, shared metadata construction, slot writes, and slot reads into `weiss_rl.runtime_shared`.
- Preserved the private `weiss_rl.runtime` names used by existing tests and runtime call sites.
- Kept process collector orchestration and rollout collection in `runtime.py`.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime.py -k "shared_collector_slot_round_trip or shared_pending_unroll or fill_pending_unrolls"` | Passed: 5 passed, 83 deselected. |
| `uv run ruff check python/weiss_rl/runtime.py python/weiss_rl/runtime_shared.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/runtime.py python/weiss_rl/runtime_shared.py` | Passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 773 pytest tests passed, 2 skipped, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/runtime_shared.py` with `SharedCollectorSlot`, `SharedPendingUnroll`, shared segment/slot config helpers, shared slot open/read/write helpers, and shared unroll metadata construction.
- Updated `python/weiss_rl/runtime.py` to import the shared transport classes and delegate through compatibility wrappers.

### Behavior Changes

No intended behavior changes. Shared slot round-trip, shared pending unroll view preservation, and pending-unroll release tests passed.

### Remaining Risks

- Process collector orchestration and rollout collection still live in `runtime.py`.
- Legal-action concatenation and packed candidate scoring remain behavior-sensitive and should stay guarded by dedicated tests before any move.

## 2026-05-10 - Runtime Device Layout Extraction

### Scope

- Extracted CUDA auto-request parsing, available CUDA device listing, device-name normalization, learner-device normalization, and actor-device layout selection into `weiss_rl.runtime_devices`.
- Preserved `weiss_rl.runtime.resolve_actor_device_layout()` and private device helper names as wrappers.
- Left runtime actor construction and process-collector enablement decisions in `runtime.py`.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime.py -k "resolve_actor_device_layout or honors_non_cpu_actor_device"` | Passed: 2 passed, 86 deselected. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime.py -k "resolve_actor_device_layout or honors_non_cpu_actor_device or shared_collector_slot_round_trip or shared_pending_unroll or fill_pending_unrolls"` | Passed: 7 passed, 81 deselected. |
| `uv run ruff check python/weiss_rl/runtime.py python/weiss_rl/runtime_devices.py python/weiss_rl/runtime_shared.py` | Passed after import sorting. |
| `uv run ruff format --check python/weiss_rl/runtime.py python/weiss_rl/runtime_devices.py python/weiss_rl/runtime_shared.py` | Passed after formatting `runtime.py`. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 773 pytest tests passed, 2 skipped, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/runtime_devices.py` with:
  - `is_cuda_auto_request`,
  - `available_cuda_device_names`,
  - `normalize_device_name`,
  - `configured_learner_device_name`,
  - `resolve_actor_device_layout`.
- Updated `python/weiss_rl/runtime.py` wrappers to delegate to the new module.

### Behavior Changes

No intended behavior changes. Existing actor device layout and CUDA/non-CPU actor runtime tests passed through the preserved `runtime.py` surface.

### Remaining Risks

- Runtime process collector orchestration and rollout collection remain in `runtime.py`.
- CUDA behavior is still environment-dependent; current tests monkeypatch CUDA availability/device count rather than requiring real GPUs.

## 2026-05-10 - Training Run Metadata Extraction

### Scope

- Extracted git commit/dirty helpers, run nonce generation, hardware summary construction, evaluation pinning summary construction, manifest source path formatting, and JSON object loading into `weiss_rl.training.run_metadata`.
- Preserved the private `python/scripts/train.py` helper names as wrappers.
- Left actor-device manifest layout calculation in `train.py` because it depends on runtime config construction.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_script_entrypoint_smokes.py -k "train_metadata_helpers or train_entrypoint_applies_profile_flags or train_cli_parser"` | Passed: 3 passed, 11 deselected, 14 third-party warnings. |
| `uv run ruff check python/scripts/train.py python/weiss_rl/training/run_metadata.py python/weiss_rl/tests/test_script_entrypoint_smokes.py` | Passed. |
| `uv run ruff format --check python/scripts/train.py python/weiss_rl/training/run_metadata.py python/weiss_rl/tests/test_script_entrypoint_smokes.py` | Passed after formatting `train.py` and the smoke test file. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 774 pytest tests passed, 2 skipped, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/training/run_metadata.py`.
- Added `test_train_metadata_helpers_preserve_manifest_shapes()` to pin representative wrapper outputs.
- Updated `python/scripts/train.py` metadata wrappers to delegate to the new module.

### Behavior Changes

No intended behavior changes. Existing training entrypoint smoke coverage and the new metadata wrapper characterization test passed.

### Remaining Risks

- B1 import contract validation still lives in `train.py` because it checks artifact manifests, spec hashes, and tensor payload shapes.
- Training loop orchestration remains large and should be split only behind broader characterization.

## 2026-05-10 - No-League Import Contract Helper Extraction

### Scope

- Extracted B1 no-league role/config classification helpers and optional hash-file reading into `weiss_rl.training.import_contracts`.
- Preserved private wrapper names in `python/scripts/train.py`.
- Left full imported snapshot validation in `train.py` because it checks manifests, spec hashes, and tensor payload shape/dtype compatibility.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_script_entrypoint_smokes.py -k "noleague_import_contract or train_metadata_helpers"` | Passed: 2 passed, 13 deselected, 14 third-party warnings. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_snapshot_registry.py -k "noleague_baseline_anchor"` | Passed: 5 passed, 32 deselected, 14 third-party warnings. |
| `uv run ruff check python/scripts/train.py python/weiss_rl/training/import_contracts.py python/weiss_rl/tests/test_script_entrypoint_smokes.py` | Passed. |
| `uv run ruff format --check python/scripts/train.py python/weiss_rl/training/import_contracts.py python/weiss_rl/tests/test_script_entrypoint_smokes.py` | Passed after formatting `train.py` and the smoke test file. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 775 pytest tests passed, 2 skipped, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/training/import_contracts.py`.
- Added `test_train_noleague_import_contract_helpers_preserve_role_rules()`.
- Updated no-league role/config/hash-file wrappers in `python/scripts/train.py` to delegate to the package helper.

### Behavior Changes

No intended behavior changes. B1 no-league import tests and direct wrapper characterization passed.

### Remaining Risks

- The tensor/state-dict portion of imported snapshot validation remains in `train.py`.
- Training loop orchestration and promotion/dev-eval consumption remain large.

## 2026-05-10 - Imported Snapshot Contract Validation Extraction

### Scope

- Moved the full B1 imported snapshot contract validator into `weiss_rl.training.import_contracts`.
- Preserved the private `python/scripts/train.py` wrapper.
- Kept B1 import orchestration, artifact writing, and registry updates in `train.py`.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_snapshot_registry.py -k "noleague_baseline_anchor or import_seed_snapshot_pool"` | Passed: 7 passed, 30 deselected, 14 third-party warnings. |
| `uv run ruff check python/scripts/train.py python/weiss_rl/training/import_contracts.py python/weiss_rl/tests/test_script_entrypoint_smokes.py` | Passed after import sorting. |
| `uv run ruff format --check python/scripts/train.py python/weiss_rl/training/import_contracts.py python/weiss_rl/tests/test_script_entrypoint_smokes.py` | Passed after formatting `train.py`. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 775 pytest tests passed, 2 skipped, wrapper dry-runs passed. |

### Changes

- Added `validate_imported_snapshot_contract()` to `python/weiss_rl/training/import_contracts.py`.
- Updated `_validate_imported_snapshot_contract()` in `python/scripts/train.py` to delegate to the package helper.

### Behavior Changes

No intended behavior changes. Existing B1 baseline import and seed snapshot import tests passed through the wrapper.

### Remaining Risks

- B1 import orchestration and registry writes still live in `train.py`.
- Training loop orchestration and promotion/dev-eval consumption remain large.

## 2026-05-10 - Final Policy Selection Helper Extraction

### Scope

- Extracted final-policy-set input loading, dev-eval summary parsing, required-input checks, deterministic selector dispatch, and resolved/unresolved detail construction into `weiss_rl.training.policy_selection`.
- Preserved the private `python/scripts/train.py` helper names as wrappers.
- Kept manifest writing and training-run orchestration in `train.py`.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_entrypoints.py -k "policy_set_selection"` | Passed: 1 passed, 30 deselected. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_policy_set.py` | Passed: 12 passed. |
| `uv run ruff check python/scripts/train.py python/weiss_rl/training/policy_selection.py` | Passed. |
| `uv run ruff format --check python/scripts/train.py python/weiss_rl/training/policy_selection.py` | Passed after formatting `train.py`. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 775 pytest tests passed, 2 skipped, wrapper dry-runs passed. |
| `uv run python python/scripts/verify_repo.py` | Re-run after context handoff passed on the live tree: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 775 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/training/policy_selection.py`.
- Updated `_load_snapshot_registry()`, `_load_dev_eval_summaries()`, `_selection_requires_snapshot_registry()`, `_selection_requires_dev_eval_summaries()`, `_policy_set_selection()`, and `_resolve_policy_set_selection()` in `python/scripts/train.py` to delegate to the package helper.

### Behavior Changes

No intended behavior changes. Existing selector and train-entrypoint manifest policy-selection tests passed.

### Remaining Risks

- Final eval execution and reporting remain outside this helper.
- Training manifest writing and run orchestration remain in `train.py`.

## 2026-05-10 - Training Guidance Helper Extraction

### Scope

- Extracted entropy annealing, teacher public-heuristic coefficient annealing, public-heuristic logit-bias scheduling, and model guidance payload restore/serialization helpers into `weiss_rl.training.guidance`.
- Preserved the private `python/scripts/train.py` helper names as wrappers.
- Added fake learner/model characterization tests for guidance scheduling and checkpoint payload round-tripping behavior.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_training_guidance.py python/weiss_rl/tests/test_train_stall_monitor.py -k "entropy_coef or guidance"` | Passed: 4 passed, 14 deselected, 14 dependency warnings. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_snapshot_registry.py -k "checkpoint_payload or restore_checkpoint or snapshot_persistence"` | Passed: 9 passed, 28 deselected, 14 dependency warnings. |
| `uv run ruff check python/scripts/train.py python/weiss_rl/training/guidance.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_training_guidance.py` | Passed. |
| `uv run ruff format --check python/scripts/train.py python/weiss_rl/training/guidance.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_training_guidance.py` | Passed after formatting `train.py`. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 778 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/training/guidance.py`.
- Added `python/weiss_rl/tests/test_training_guidance.py`.
- Updated `_entropy_coef_for_next_update()`, `_teacher_public_heuristic_coef_for_next_update()`, `_public_heuristic_logit_bias_scale_for_next_update()`, `_apply_guidance_schedule_for_next_update()`, `_model_guidance_payload()`, and `_restore_model_guidance_from_payload()` in `python/scripts/train.py` to delegate to the package helper.

### Behavior Changes

No intended behavior changes. Existing entropy-schedule coverage and adjacent checkpoint payload/restore tests passed through the compatibility wrappers.

### Remaining Risks

- Guidance helpers still assume the learner/model expose the same public-heuristic accessors used by the current training path.
- Training loop orchestration, checkpoint restore orchestration, and eval-model construction remain in `train.py`.

## 2026-05-10 - Training Path Layout Helper Extraction

### Scope

- Extracted `TrainingPaths`, training artifact path construction, and existing-run artifact reconstruction into `weiss_rl.training.paths`.
- Preserved `_training_paths()` and `_run_artifacts_from_existing_run_dir()` in `python/scripts/train.py` as compatibility wrappers.
- Kept artifact writing, checkpoint persistence, and run orchestration in `train.py`.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_snapshot_registry.py -k "training_paths or checkpoint_aliases or snapshot_persistence or rollback"` | Passed: 2 passed, 35 deselected, 14 dependency warnings. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_train_stall_monitor.py -k "stall_monitor"` | Passed: 15 passed, 14 dependency warnings. |
| `uv run ruff check python/scripts/train.py python/weiss_rl/training/paths.py python/weiss_rl/training/__init__.py` | Passed after Ruff import sorting in `train.py`. |
| `uv run ruff format --check python/scripts/train.py python/weiss_rl/training/paths.py python/weiss_rl/training/__init__.py` | Passed after formatting `train.py`. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 778 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/training/paths.py`.
- Updated `python/scripts/train.py` to import `TrainingPaths` and delegate path helpers to the package module.

### Behavior Changes

No intended behavior changes. The extracted helpers only construct and ensure artifact-layout paths.

### Remaining Risks

- Several downstream training and eval helpers still accept `TrainingPaths`, so future extraction should preserve this stable shape.
- Run manifest writing and artifact mutation remain in `train.py`.

## 2026-05-10 - Training Startup Helper Extraction

### Scope

- Extracted runtime profile, learner device, seed resolution, manifest-only detection, simulator runtime prerequisite checks, and startup failure messaging into `weiss_rl.training.startup`.
- Preserved the private `python/scripts/train.py` helper names as wrappers.
- Added direct package-level tests for startup defaults and simulator prerequisite error messages.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_training_startup.py python/weiss_rl/tests/test_script_entrypoint_smokes.py -k "startup or train_entrypoint_resolves_cuda_auto"` | Passed: 5 passed, 14 deselected, 14 dependency warnings. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_entrypoints.py -k "missing_stepping or manifest"` | Passed: 6 passed, 25 deselected, 14 dependency warnings. |
| `uv run ruff check python/scripts/train.py python/weiss_rl/training/startup.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_training_startup.py` | Passed. |
| `uv run ruff format --check python/scripts/train.py python/weiss_rl/training/startup.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_training_startup.py` | Passed after formatting `train.py` and `test_training_startup.py`. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 782 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/training/startup.py`.
- Added `python/weiss_rl/tests/test_training_startup.py`.
- Updated `_resolve_runtime_profile()`, `_resolve_device()`, `_resolve_seed()`, `_manifest_scaffold_only_reason()`, `_runtime_training_prerequisite_failure()`, `_print_manifest_only_message()`, and `_raise_runtime_prerequisite_failure()` in `python/scripts/train.py` to delegate to the package helper.

### Behavior Changes

No intended behavior changes. Existing train-entrypoint CUDA auto behavior and manifest/prerequisite entrypoint tests passed through the wrappers.

### Remaining Risks

- Full training launch orchestration and manifest writing remain in `train.py`.
- Startup checks still intentionally validate only importability and required runtime attributes, not full simulator semantic compatibility.

## 2026-05-10 - Training Torch Thread Helper Extraction

### Scope

- Extracted learner torch thread configuration, scoped temporary torch thread overrides, and central-runtime actor thread selection into `weiss_rl.training.torch_threads`.
- Preserved `_configure_torch_threads()`, `_torch_num_threads_scope()`, and `_central_runtime_actor_torch_threads()` in `python/scripts/train.py` as compatibility wrappers.
- Added direct tests for thread scope restore behavior, invalid thread counts, and CPU central-runtime actor thread selection.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_training_torch_threads.py python/weiss_rl/tests/test_script_entrypoint_smokes.py -k "torch_threads or central_actor_torch_threads"` | Passed: 4 passed, 14 deselected, 14 dependency warnings. |
| `uv run ruff check python/scripts/train.py python/weiss_rl/training/torch_threads.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_training_torch_threads.py` | Passed. |
| `uv run ruff format --check python/scripts/train.py python/weiss_rl/training/torch_threads.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_training_torch_threads.py` | Passed after formatting `train.py` and `test_training_torch_threads.py`. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 785 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/training/torch_threads.py`.
- Added `python/weiss_rl/tests/test_training_torch_threads.py`.
- Updated the private torch-thread helpers in `python/scripts/train.py` to delegate to the package helper.

### Behavior Changes

No intended behavior changes. Existing central-runtime actor-thread wrapper behavior and direct thread-scope behavior are covered.

### Remaining Risks

- Global torch thread state remains process-wide; future tests that touch it should restore state carefully.
- Training collection and learner update scheduling still live in `train.py`.

## 2026-05-10 - Training Input Validation Helper Extraction

### Scope

- Extracted SHA-256 normalization, expected-hash parsing, hash matching, positive integer validation, spec-mismatch policy, and deprecated `--run-id` / `--run-label` reconciliation into `weiss_rl.training.inputs`.
- Preserved the private `python/scripts/train.py` helper names as wrappers.
- Added direct package-level tests for hash validation, run-label alias behavior, positive integer checks, and spec mismatch policy.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_training_inputs.py python/weiss_rl/tests/test_script_entrypoint_smokes.py -k "training_inputs or train_entrypoint_parser_defaults"` | Passed: 3 passed, 15 deselected. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_script_entrypoint_smokes.py -k "train_cli_parser_preserves_public_defaults"` | Passed: 1 passed, 14 deselected. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_entrypoints.py -k "deprecated_run_id_alias or config_hash_mismatch or runtime_spec_mismatch"` | Passed: 3 passed, 28 deselected. |
| `uv run ruff check python/scripts/train.py python/weiss_rl/training/inputs.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_training_inputs.py` | Passed. |
| `uv run ruff format --check python/scripts/train.py python/weiss_rl/training/inputs.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_training_inputs.py` | Passed after formatting `train.py`. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 788 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/training/inputs.py`.
- Added `python/weiss_rl/tests/test_training_inputs.py`.
- Updated `_normalize_sha256()`, `_expected_sha256()`, `_require_matching_hash()`, `_spec_mismatch_policy()`, `_resolve_run_label()`, and `_require_positive_int()` in `python/scripts/train.py` to delegate to the package helper.

### Behavior Changes

No intended behavior changes. Public CLI alias handling and relevant entrypoint failure paths still pass.

### Remaining Risks

- Higher-level train argument parsing remains in `weiss_rl.training.cli` and `train.py`; this helper only covers post-parse validation.
- Main training orchestration still lives in `train.py`.

## 2026-05-10 - Training Environment Builder Extraction

### Scope

- Extracted simulator spec dimension reading, env-pool config construction, mask-based training env construction, and ids-offset eval env construction into `weiss_rl.training.environments`.
- Preserved `_spec_dimensions()`, `_env_pool_config()`, `_build_env()`, and `_build_ids_eval_env()` in `python/scripts/train.py` as compatibility wrappers.
- Added fake boundary tests for mask legality, ids-offset legality, max-no-progress propagation, and wrong-layout rejection without requiring a live simulator.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_training_environments.py` | Passed: 4 passed. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_train_stall_monitor.py -k "build_learner_batch"` | Passed: 1 passed, 14 deselected, 14 dependency warnings. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_snapshot_registry.py -k "periodic_dev_eval or promotion_gate"` | Passed: 7 passed, 30 deselected, 14 dependency warnings. |
| `uv run ruff check python/scripts/train.py python/weiss_rl/training/environments.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_training_environments.py` | Passed after Ruff import sorting in `train.py`. |
| `uv run ruff format --check python/scripts/train.py python/weiss_rl/training/environments.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_training_environments.py` | Passed after formatting `train.py`. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 792 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/training/environments.py`.
- Added `python/weiss_rl/tests/test_training_environments.py`.
- Updated the private environment construction helpers in `python/scripts/train.py` to delegate to the package helper.

### Behavior Changes

No intended behavior changes. The training path still requires mask legality, and periodic dev eval still requires ids-based legality for the pinned eval protocol.

### Remaining Risks

- These tests pin constructor arguments with fake pools, but they do not replace simulator-backed contract tests.
- Rollout collection, bootstrap value computation, and learner batch assembly remain in `train.py`.

## 2026-05-10 - Minimal Training Batch Helper Extraction

### Scope

- Extracted `MinimalRollout`, bootstrap value computation, and minimal train learner-batch assembly into `weiss_rl.training.batches`.
- Preserved `_bootstrap_values()` and `_build_learner_batch()` in `python/scripts/train.py` as compatibility wrappers.
- Added direct package-level tests for truncation discount behavior, bootstrap actor-row filtering, and missing config block failure.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_training_batches.py python/weiss_rl/tests/test_train_stall_monitor.py -k "build_learner_batch or bootstrap_values"` | Passed after fixing the new test fixture shape: 4 passed, 14 deselected, 14 dependency warnings. |
| `uv run ruff check python/scripts/train.py python/weiss_rl/training/batches.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_training_batches.py` | Passed after removing a stale train import and sorting package exports. |
| `uv run ruff format --check python/scripts/train.py python/weiss_rl/training/batches.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_training_batches.py` | Passed after formatting `train.py` and `test_training_batches.py`. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 795 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/training/batches.py`.
- Added `python/weiss_rl/tests/test_training_batches.py`.
- Updated minimal training batch wrappers in `python/scripts/train.py` to delegate to the package helper.

### Behavior Changes

No intended behavior changes. The first focused test run exposed only a new test fixture shape mismatch, not a source behavior change; the fixture was corrected to match the one-env learner-batch shape.

### Remaining Risks

- This helper covers the minimal single-node training path only; the main `QueueRuntime` batch builders still live in `runtime.py`.
- V-trace, masking, and truncation semantics remain behavior-sensitive and should not be refactored further without stronger characterization.

## 2026-05-10 - Checkpoint and Scalar Writer Extraction

### Scope

- Moved scalar JSONL writing and minimal train checkpoint file writing into `weiss_rl.training.checkpoints`.
- Preserved `_write_scalars_record()` and `_write_checkpoint()` in `python/scripts/train.py` as compatibility wrappers.
- Added package-level tests for scalar JSON content, checkpoint payload shape, checkpoint save/load, and missing-model failure.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_training_checkpoint_writers.py python/weiss_rl/tests/test_snapshot_registry.py -k "checkpoint_payload_shape or checkpoint_aliases or write_minimal_train_checkpoint or write_scalars"` | Passed after restoring `json`/`time` imports still used by other train helpers: 5 passed, 35 deselected, 14 dependency warnings. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_entrypoints.py -k "periodic_dev_eval_writes_exact_current_checkpoint or public_demo"` | Passed: 5 passed, 26 deselected. |
| `uv run ruff check python/scripts/train.py python/weiss_rl/training/checkpoints.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_training_checkpoint_writers.py` | Passed after sorting imports and removing a stale train import. |
| `uv run ruff format --check python/scripts/train.py python/weiss_rl/training/checkpoints.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_training_checkpoint_writers.py` | Passed after formatting `train.py`. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 798 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `write_scalars_record()` and `write_minimal_train_checkpoint()` to `python/weiss_rl/training/checkpoints.py`.
- Added `python/weiss_rl/tests/test_training_checkpoint_writers.py`.
- Updated scalar/checkpoint private wrappers in `python/scripts/train.py` to delegate to the package helper.

### Behavior Changes

No intended behavior changes. Focused entrypoint and snapshot-registry tests confirmed checkpoint traceability and public-demo artifact paths still work.

### Remaining Risks

- Checkpoint restore orchestration remains in `train.py`.
- Alias publication and best-checkpoint promotion still live in training orchestration helpers and should remain guarded by snapshot-registry tests.

## 2026-05-10 - Checkpoint Restore Helper Extraction

### Scope

- Moved minimal checkpoint restore mechanics into `weiss_rl.training.checkpoints`.
- Moved `ResumeCheckpoint` into the checkpoint helper module and kept `python/scripts/train.py` importing it for compatibility.
- Preserved `_restore_learner_from_checkpoint()` in `train.py` as the public private-wrapper that still computes the current config hash and reads the config-mismatch escape hatch.
- Added a direct package-level restore test for model state loading, guidance payload handoff, optimizer restore, and `restore_counters=False`.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_training_checkpoint_writers.py python/weiss_rl/tests/test_snapshot_registry.py -k "restore_minimal_train_checkpoint or restore_checkpoint or checkpoint_aliases"` | Passed: 9 passed, 32 deselected, 14 dependency warnings. |
| `uv run ruff check python/scripts/train.py python/weiss_rl/training/checkpoints.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_training_checkpoint_writers.py` | Passed after Ruff import sorting. |
| `uv run ruff format --check python/scripts/train.py python/weiss_rl/training/checkpoints.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_training_checkpoint_writers.py` | Passed after formatting `train.py`. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 799 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `ResumeCheckpoint` and `restore_minimal_train_checkpoint()` to `python/weiss_rl/training/checkpoints.py`.
- Updated `_restore_learner_from_checkpoint()` in `python/scripts/train.py` to delegate restore mechanics to the package helper.
- Extended `python/weiss_rl/tests/test_training_checkpoint_writers.py`.

### Behavior Changes

No intended behavior changes. Existing invalid-payload, config-mismatch escape hatch, alias restore, and counter-preservation tests passed through the wrapper.

### Remaining Risks

- Checkpoint alias publication and best-checkpoint finalization remain in `train.py`.
- The helper preserves current checkpoint semantics; it does not broaden checkpoint compatibility.

## 2026-05-11 - Checkpoint Alias Publication Extraction

### Scope

- Moved latest/best checkpoint alias publication into `weiss_rl.training.checkpoints`.
- Preserved `_publish_checkpoint_aliases()` in `python/scripts/train.py` as a compatibility wrapper.
- Added direct package-level coverage for the alias contract: every checkpoint refreshes `latest`, while `best` only promotes when the candidate training-loss metric improves.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_training_checkpoint_writers.py python/weiss_rl/tests/test_snapshot_registry.py -k "publish_checkpoint_aliases or checkpoint_aliases"` | Passed after formatting: 2 passed, 40 deselected, 14 dependency warnings. |
| `uv run ruff check python/scripts/train.py python/weiss_rl/training/checkpoints.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_training_checkpoint_writers.py` | Passed. |
| `uv run ruff format --check python/scripts/train.py python/weiss_rl/training/checkpoints.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_training_checkpoint_writers.py` | Passed after formatting `train.py`. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 800 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `publish_checkpoint_aliases()` to `python/weiss_rl/training/checkpoints.py`.
- Exported the helper from `python/weiss_rl/training/__init__.py`.
- Updated `python/scripts/train.py` to delegate alias publication to the package helper.
- Extended `python/weiss_rl/tests/test_training_checkpoint_writers.py`.

### Behavior Changes

No intended behavior changes. Existing snapshot-registry alias tests still pass, and the new package-level test locks the training-loss promotion ordering before further checkpoint finalization work.

### Remaining Risks

- Final checkpoint publication and checkpoint-guard rollback/finalization orchestration still live in `train.py`.
- This helper only centralizes existing alias behavior; it does not change promotion metrics or broaden checkpoint compatibility.

## 2026-05-11 - Checkpoint Guard Event Helper Extraction

### Scope

- Moved checkpoint-guard JSONL event appending into `weiss_rl.training.checkpoints`.
- Moved best-checkpoint tracker lookup into `weiss_rl.training.checkpoints`.
- Preserved `_append_checkpoint_guard_event()` and `_best_checkpoint_record()` in `python/scripts/train.py` as compatibility wrappers.
- Added direct package-level coverage for sorted JSONL event appending and best-record lookup through the checkpoint tracker.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_training_checkpoint_writers.py -k "checkpoint_guard_event or publish_checkpoint_aliases"` | Passed: 2 passed, 4 deselected. |
| `uv run ruff check python/scripts/train.py python/weiss_rl/training/checkpoints.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_training_checkpoint_writers.py` | Passed. |
| `uv run ruff format --check python/scripts/train.py python/weiss_rl/training/checkpoints.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_training_checkpoint_writers.py` | Passed after formatting `train.py`. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 801 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `append_checkpoint_guard_event()` and `best_checkpoint_record()` to `python/weiss_rl/training/checkpoints.py`.
- Exported both helpers from `python/weiss_rl/training/__init__.py`.
- Updated the corresponding private wrappers in `python/scripts/train.py`.
- Extended `python/weiss_rl/tests/test_training_checkpoint_writers.py`.

### Behavior Changes

No intended behavior changes. JSONL guard events still use sorted-key JSON records and the same `logs/checkpoint_guard.jsonl` path.

### Remaining Risks

- Checkpoint-guard rollback/finalization decisions still belong to `train.py`.
- Further movement in this area should keep rollback event payloads and tracker updates under snapshot-registry coverage.

## 2026-05-11 - Snapshot Champion Demotion Helper Extraction

### Scope

- Moved registry-file champion demotion into `weiss_rl.training.snapshots`.
- Preserved `_demote_registry_champions_newer_than()` in `python/scripts/train.py` as a compatibility wrapper.
- Added direct package-level coverage that demotion updates the persisted snapshot registry file.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_snapshot_registry.py -k "demote_registry_champions_newer_than or finalize_from_best_checkpoint"` | Passed after formatting: 3 passed, 35 deselected, 14 dependency warnings. |
| `uv run ruff check python/scripts/train.py python/weiss_rl/training/snapshots.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_snapshot_registry.py` | Passed. |
| `uv run ruff format --check python/scripts/train.py python/weiss_rl/training/snapshots.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_snapshot_registry.py` | Passed after formatting `train.py` and `test_snapshot_registry.py`. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 802 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `demote_registry_champions_newer_than()` to `python/weiss_rl/training/snapshots.py`.
- Exported the helper from `python/weiss_rl/training/__init__.py`.
- Updated `python/scripts/train.py` to delegate the demotion wrapper.
- Extended `python/weiss_rl/tests/test_snapshot_registry.py`.

### Behavior Changes

No intended behavior changes. Missing registry files still produce an empty demotion list; existing registries are saved only when champion references are removed.

### Remaining Risks

- The rollback/finalization call sites still coordinate learner restore, runtime refresh, alias rewriting, and guard event payload construction in `train.py`.
- Further extraction should first pin rollback event payloads as tightly as the existing finalization test pins latest-alias rewriting.

## 2026-05-11 - Checkpoint Guard Rollback Characterization

### Scope

- Added a rollback-path characterization test before moving checkpoint-guard rollback control flow.
- Pinned the observable rollback behavior: best-checkpoint restore without counter rewind, forced runtime snapshot publication, runtime tracker reset/refresh, latest-alias rewrite, champion demotion, and checkpoint-guard JSONL event payload.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_snapshot_registry.py -k "rollback_to_best_checkpoint or finalize_from_best_checkpoint or demote_registry_champions_newer_than"` | Passed after formatting: 4 passed, 35 deselected, 14 dependency warnings. |
| `uv run ruff check python/weiss_rl/tests/test_snapshot_registry.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/tests/test_snapshot_registry.py` | Passed after formatting `test_snapshot_registry.py`. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 803 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Extended `python/weiss_rl/tests/test_snapshot_registry.py` with rollback-path characterization.

### Behavior Changes

No intended behavior changes. This checkpoint adds coverage only.

### Remaining Risks

- `_maybe_rollback_to_best_checkpoint()` and `_maybe_finalize_from_best_checkpoint()` still live in `train.py`.
- With rollback and finalization behavior now pinned, a later extraction can move checkpoint-guard orchestration more safely, but it should remain a separate checkpoint.

## 2026-05-11 - Checkpoint Guard Orchestration Extraction

### Scope

- Moved checkpoint-guard rollback and finalization orchestration into `weiss_rl.training.checkpoints`.
- Preserved `_maybe_rollback_to_best_checkpoint()` and `_maybe_finalize_from_best_checkpoint()` in `python/scripts/train.py` as compatibility wrappers.
- Kept behavior-sensitive checkpoint serialization and restore mechanics in `train.py` callbacks so config-hash validation, device handling, and checkpoint payload compatibility remain unchanged.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_snapshot_registry.py -k "rollback_to_best_checkpoint or finalize_from_best_checkpoint or demote_registry_champions_newer_than"` | Passed after formatting: 4 passed, 35 deselected, 14 dependency warnings. |
| `uv run ruff check python/scripts/train.py python/weiss_rl/training/checkpoints.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_snapshot_registry.py` | Passed. |
| `uv run ruff format --check python/scripts/train.py python/weiss_rl/training/checkpoints.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_snapshot_registry.py` | Passed after formatting `train.py`. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 803 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `maybe_rollback_to_best_checkpoint()` and `maybe_finalize_from_best_checkpoint()` to `python/weiss_rl/training/checkpoints.py`.
- Exported both helpers from `python/weiss_rl/training/__init__.py`.
- Updated the corresponding private wrappers in `python/scripts/train.py`.

### Behavior Changes

No intended behavior changes. The pinned rollback and finalization tests still verify latest-alias rewrites, guard event payloads, champion demotion, runtime reset/refresh, and forced rollback snapshot publication.

### Remaining Risks

- Periodic dev-eval scheduling, promotion-gate execution, and final policy-set orchestration still live in `train.py`.
- The checkpoint helper now imports snapshot champion demotion; keep future changes in these modules conscious of that package-level dependency.

## 2026-05-11 - Structured Main-Move Guard Extraction

### Scope

- Moved structured main-move checkpoint-guard warning construction into `weiss_rl.training.checkpoints`.
- Preserved `_maybe_log_structured_mainmove_guard()` and `_extract_structured_guard_b2_anchor_score()` in `python/scripts/train.py` as compatibility wrappers.
- Added direct package-level tests for warning emission, guard JSONL persistence, B2 anchor-score extraction, and healthy-B2 suppression.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_training_checkpoint_writers.py -k "structured_mainmove_guard or checkpoint_guard_event"` | Passed after formatting: 3 passed, 5 deselected. |
| `uv run ruff check python/scripts/train.py python/weiss_rl/training/checkpoints.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_training_checkpoint_writers.py` | Passed. |
| `uv run ruff format --check python/scripts/train.py python/weiss_rl/training/checkpoints.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_training_checkpoint_writers.py` | Passed after formatting `train.py`. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 805 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `maybe_log_structured_mainmove_guard()` and `extract_structured_guard_b2_anchor_score()` to `python/weiss_rl/training/checkpoints.py`.
- Exported both helpers from `python/weiss_rl/training/__init__.py`.
- Updated `python/scripts/train.py` wrappers.
- Extended `python/weiss_rl/tests/test_training_checkpoint_writers.py`.

### Behavior Changes

No intended behavior changes. The warning still writes a `structured_mainmove_warning_v1` checkpoint-guard event only when structured main-move metrics are suspicious and B2/aggregate dev-eval evidence does not suppress the warning.

### Remaining Risks

- The warning thresholds are now centralized in the checkpoint helper, but the learning metrics that feed them still originate from `ImpalaLearner`.
- Further checkpoint-area work should avoid changing metric names or guard thresholds without explicit bug-fix evidence.

## 2026-05-11 - Periodic Dev-Eval Support Extraction

### Scope

- Added `weiss_rl.training.dev_eval` for pure periodic-dev-eval support helpers.
- Moved evaluation contract checks, dev-eval seed-file resolution, schedule slicing/hash calculation, deterministic periodic/promotion RNG seeds, bootstrap seeds, JSON helper, and dev-eval/stall-monitor log paths behind `train.py` wrappers.
- Kept the periodic dev-eval runner, policy loading, promotion-gate execution, and simulator interaction in `python/scripts/train.py`.
- Added direct package-level tests for seed-source validation, mismatch failure, public contract error text, deterministic seed separation, interval decisions, and log paths.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_training_dev_eval.py python/weiss_rl/tests/test_snapshot_registry.py -k "training_dev_eval or periodic_dev_eval or promotion_gate_rng_seed or run_snapshot_promotion_gate"` | Passed after removing a stale `json` import and formatting: 10 passed, 34 deselected, 14 dependency warnings. |
| `uv run ruff check python/scripts/train.py python/weiss_rl/training/dev_eval.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_training_dev_eval.py` | Passed. |
| `uv run ruff format --check python/scripts/train.py python/weiss_rl/training/dev_eval.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_training_dev_eval.py` | Passed after formatting `train.py`. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 810 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/training/dev_eval.py`.
- Added `python/weiss_rl/tests/test_training_dev_eval.py`.
- Exported dev-eval support helpers from `python/weiss_rl/training/__init__.py`.
- Updated pure helper wrappers in `python/scripts/train.py`.

### Behavior Changes

No intended behavior changes. The extracted contract helper preserves the prior error messages for CPU, inference-mode, seat-swap, and pinned-CDF requirements.

### Remaining Risks

- The periodic dev-eval runner and promotion-gate orchestration remain behavior-sensitive and still live in `train.py`.
- Further movement should pin full artifact payloads and call order before moving simulator-backed eval loops.

## 2026-05-11 - Periodic Dev-Eval Persistence Extraction

### Scope

- Moved periodic dev-eval summary persistence into `weiss_rl.training.dev_eval`.
- Moved stall-monitor state update logic into `weiss_rl.training.dev_eval`.
- Preserved `_persist_periodic_dev_eval_summary()` and `_update_stall_monitor()` in `python/scripts/train.py` as compatibility wrappers.
- Added direct package-level tests for summary JSON persistence and consecutive stall-risk tracking.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_training_dev_eval.py python/weiss_rl/tests/test_train_stall_monitor.py -k "persist_periodic_dev_eval_summary or update_stall_monitor or periodic_dev_eval"` | Passed after import sorting and formatting: 12 passed, 10 deselected, 14 dependency warnings. |
| `uv run ruff check python/scripts/train.py python/weiss_rl/training/dev_eval.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_training_dev_eval.py python/weiss_rl/tests/test_train_stall_monitor.py` | Passed. |
| `uv run ruff format --check python/scripts/train.py python/weiss_rl/training/dev_eval.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_training_dev_eval.py python/weiss_rl/tests/test_train_stall_monitor.py` | Passed after formatting `train.py`. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 812 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `persist_periodic_dev_eval_summary()` and `update_stall_monitor()` to `python/weiss_rl/training/dev_eval.py`.
- Exported both helpers from `python/weiss_rl/training/__init__.py`.
- Updated `python/scripts/train.py` wrappers.
- Extended `python/weiss_rl/tests/test_training_dev_eval.py`.

### Behavior Changes

No intended behavior changes. Existing script-level stall-monitor tests still pass through the wrapper, while the new package-level tests pin the persisted JSON shapes.

### Remaining Risks

- The simulator-backed periodic dev-eval runner and promotion-gate execution remain in `train.py`.
- Moving those loops should be preceded by characterization of artifact payloads, checkpoint path usage, and runner call order.

## 2026-05-11 - Current Checkpoint Helper Extraction

### Scope

- Moved current focal policy ID construction, checkpoint path naming, and current-checkpoint ensure logic into `weiss_rl.training.checkpoints`.
- Preserved `_current_focal_policy_id()`, `_checkpoint_path_for_update()`, and `_ensure_current_checkpoint()` in `python/scripts/train.py` as compatibility wrappers.
- Kept actual checkpoint serialization in a `train.py` callback so run-specific config/spec/device behavior is unchanged.
- Added direct package-level coverage for reusing an existing checkpoint and writing a missing checkpoint exactly once.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_training_checkpoint_writers.py python/weiss_rl/tests/test_snapshot_registry.py -k "ensure_current_checkpoint or checkpoint_aliases or rollback_to_best_checkpoint or finalize_from_best_checkpoint"` | Passed after formatting: 5 passed, 43 deselected, 14 dependency warnings. |
| `uv run ruff check python/scripts/train.py python/weiss_rl/training/checkpoints.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_training_checkpoint_writers.py` | Passed. |
| `uv run ruff format --check python/scripts/train.py python/weiss_rl/training/checkpoints.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_training_checkpoint_writers.py` | Passed after formatting `train.py` and `test_training_checkpoint_writers.py`. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 813 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `current_focal_policy_id()`, `checkpoint_path_for_update()`, and `ensure_current_checkpoint()` to `python/weiss_rl/training/checkpoints.py`.
- Exported the helpers from `python/weiss_rl/training/__init__.py`.
- Updated `python/scripts/train.py` wrappers.
- Extended `python/weiss_rl/tests/test_training_checkpoint_writers.py`.

### Behavior Changes

No intended behavior changes. Checkpoint filenames remain `checkpoint_<update>.pt`, and missing checkpoint writes still go through the existing train-script writer.

### Remaining Risks

- The simulator-backed periodic dev-eval runner and promotion-gate execution still call these wrappers from `train.py`.
- Further extraction should keep checkpoint write callbacks explicit until full artifact/call-order coverage exists.

## 2026-05-11 - Promotion Anchor Resolver Extraction

### Scope

- Added `weiss_rl.training.promotion` for pure promotion-anchor name and policy-id resolution.
- Moved anchor slugging, legacy alias candidate generation, symbolic latest/previous champion/recent resolution, and required/optional anchor resolution behind `train.py` wrappers.
- Kept promotion-gate execution and simulator-backed runner logic in `python/scripts/train.py`.
- Added direct package-level tests for legacy aliases, symbolic registry windows, heuristic anchors, and missing required anchor reporting.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_training_promotion.py python/weiss_rl/tests/test_snapshot_registry.py -k "training_promotion or run_snapshot_promotion_gate"` | Passed after formatting: 6 passed, 36 deselected, 14 dependency warnings. |
| `uv run ruff check python/scripts/train.py python/weiss_rl/training/promotion.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_training_promotion.py` | Passed. |
| `uv run ruff format --check python/scripts/train.py python/weiss_rl/training/promotion.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_training_promotion.py` | Passed after formatting `train.py` and `test_training_promotion.py`. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 816 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/training/promotion.py`.
- Added `python/weiss_rl/tests/test_training_promotion.py`.
- Exported promotion resolver helpers from `python/weiss_rl/training/__init__.py`.
- Updated pure promotion-anchor wrappers in `python/scripts/train.py`.

### Behavior Changes

No intended behavior changes. The canonical B0/B1/B2 names, legacy B1 alias fallback, and symbolic latest/previous registry labels are preserved.

### Remaining Risks

- `_run_snapshot_promotion_gate()` still owns simulator-backed promotion execution and artifact handling.
- Further extraction should pin promotion-gate call arguments and artifact writes before moving that runner.

## 2026-05-11 - Algorithm Model Contract Extraction

### Scope

- Added `weiss_rl.training.algorithm_contracts` for algorithm/model compatibility validation.
- Preserved `_validate_algorithm_model_contract()` in `python/scripts/train.py` as a compatibility wrapper.
- Added direct package-level tests for accepted IMPALA/structured combinations and rejected recurrent-core/encoder mismatches.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_training_algorithm_contracts.py python/weiss_rl/tests/test_script_entrypoint_smokes.py -k "algorithm_model_contract or train_cli_parser_preserves_public_defaults"` | Passed after import sorting and formatting: 10 passed, 14 deselected. |
| `uv run ruff check python/scripts/train.py python/weiss_rl/training/algorithm_contracts.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_training_algorithm_contracts.py` | Passed. |
| `uv run ruff format --check python/scripts/train.py python/weiss_rl/training/algorithm_contracts.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_training_algorithm_contracts.py` | Passed after formatting `train.py`. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 825 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/training/algorithm_contracts.py`.
- Added `python/weiss_rl/tests/test_training_algorithm_contracts.py`.
- Exported `validate_algorithm_model_contract()` from `python/weiss_rl/training/__init__.py`.
- Updated the train-script compatibility wrapper.

### Behavior Changes

No intended behavior changes. Error text and allowed combinations are preserved.

### Remaining Risks

- Model construction and learner creation remain in `train.py`.
- Larger model/learner module movement still needs checkpoint-state-dict and action-scoring characterization before refactoring.

## 2026-05-11 - Promotion Support Helper Extraction

### Scope

- Moved heuristic-public policy construction into `weiss_rl.training.promotion`.
- Moved snapshot metadata indexing by policy id into `weiss_rl.training.promotion`.
- Preserved `_build_heuristic_public_policy()` and `_snapshot_meta_by_policy_id()` in `python/scripts/train.py` as compatibility wrappers.
- Added direct package-level tests for scoring-profile forwarding, legacy factory fallback, and snapshot metadata indexing.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_training_promotion.py python/weiss_rl/tests/test_snapshot_registry.py -k "training_promotion or heuristic_public_policy or snapshot_meta_by_policy_id or run_snapshot_promotion_gate"` | Passed after import sorting and formatting: 9 passed, 36 deselected, 14 dependency warnings. |
| `uv run ruff check python/scripts/train.py python/weiss_rl/training/promotion.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_training_promotion.py` | Passed. |
| `uv run ruff format --check python/scripts/train.py python/weiss_rl/training/promotion.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_training_promotion.py` | Passed after formatting `train.py`. |
| `uv run python python/scripts/verify_repo.py` | Initially failed in `test_periodic_dev_eval_opponents_include_optional_b2_when_available`; the extracted helper bypassed a script-level `HeuristicPublicPolicy` monkeypatch. Fixed by resolving the default policy class at call time and passing the script binding through the compatibility wrapper. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_train_stall_monitor.py::test_periodic_dev_eval_opponents_include_optional_b2_when_available python/weiss_rl/tests/test_training_promotion.py` | Passed after the compatibility fix: 7 passed, 14 dependency warnings. |
| `uv run python python/scripts/verify_repo.py` | Passed after the compatibility fix: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 828 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `build_heuristic_public_policy()` and `snapshot_meta_by_policy_id()` to `python/weiss_rl/training/promotion.py`.
- Exported both helpers from `python/weiss_rl/training/__init__.py`.
- Updated the train-script compatibility wrappers.
- Extended `python/weiss_rl/tests/test_training_promotion.py`.
- Preserved script-level monkeypatch compatibility by allowing the wrapper to pass its local `HeuristicPublicPolicy` binding into the extracted helper.

### Behavior Changes

No intended behavior changes. The heuristic-public factory still forwards `scoring_profile` only when the installed `HeuristicPublicPolicy.from_spec_bundle()` supports it, preserving compatibility with older factory signatures.

### Remaining Risks

- Simulator-backed promotion-gate execution still lives in `train.py`.
- Further promotion-gate extraction should pin opponent construction, artifact paths, and pass/fail registry updates before moving the runner.

## 2026-05-11 - Seed Snapshot Policy-ID Extraction

### Scope

- Moved seeded snapshot policy-id construction into `weiss_rl.training.snapshots`.
- Preserved `_seed_snapshot_policy_id()` in `python/scripts/train.py` as a compatibility wrapper.
- Added direct tests for exact hash input handling, source policy-id sanitization, and wrapper equivalence.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_snapshot_registry.py -k "seed_snapshot_policy_id or import_seed_snapshot_pool"` | Initially failed due an incorrect expected SHA-1 prefix in the new test fixture, then passed after correcting the expected value: 4 passed, 37 deselected, 14 dependency warnings. |
| `uv run ruff check python/scripts/train.py python/weiss_rl/training/snapshots.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_snapshot_registry.py` | Passed. |
| `uv run ruff format --check python/scripts/train.py python/weiss_rl/training/snapshots.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_snapshot_registry.py` | Passed after formatting `train.py` and `test_snapshot_registry.py`. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 830 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `seed_snapshot_policy_id()` to `python/weiss_rl/training/snapshots.py`.
- Exported the helper from `python/weiss_rl/training/__init__.py`.
- Updated the train-script compatibility wrapper.
- Extended `python/weiss_rl/tests/test_snapshot_registry.py`.

### Behavior Changes

No intended behavior changes. The helper still hashes `source_run_dir.as_posix()` exactly as passed by the caller and sanitizes only slashes/backslashes plus surrounding whitespace in the source policy id.

### Remaining Risks

- Seed snapshot pool import still combines artifact writes, registry mutation, champion preservation, and CLI output in `train.py`.
- Moving `_import_seed_snapshot_pool()` should wait until those side effects are characterized independently.

## 2026-05-11 - Dev-Eval Legal-ID Row Extraction

### Scope

- Moved packed legal-id row slicing into `weiss_rl.training.dev_eval`.
- Preserved `_legal_ids_for_env_row()` in `python/scripts/train.py` as a compatibility wrapper.
- Added direct tests for packed row slicing, missing `ids_offsets`, unsorted rows without enforcement, and sortedness enforcement.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_training_dev_eval.py python/weiss_rl/tests/test_snapshot_registry.py -k "legal_ids_for_env_row or periodic_dev_eval or promotion_gate_rng_seed or run_snapshot_promotion_gate"` | Initially failed because the new test expected `AssertionError`; corrected the characterization to the existing `ValueError("legal_ids must be strictly increasing")`, then passed: 13 passed, 38 deselected, 14 dependency warnings. |
| `uv run ruff check python/scripts/train.py python/weiss_rl/training/dev_eval.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_training_dev_eval.py` | Passed. |
| `uv run ruff format --check python/scripts/train.py python/weiss_rl/training/dev_eval.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_training_dev_eval.py` | Passed after retrying a transient Windows file-lock on `python/scripts/train.py`. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 833 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `legal_ids_for_env_row()` to `python/weiss_rl/training/dev_eval.py`.
- Exported the helper from `python/weiss_rl/training/__init__.py`.
- Updated the train-script compatibility wrapper.
- Extended `python/weiss_rl/tests/test_training_dev_eval.py`.

### Behavior Changes

No intended behavior changes. The helper still requires packed `ids_offsets`, slices by row offsets, returns `uint32` ids, and delegates sortedness validation to `assert_strictly_increasing_legal_ids()`.

### Remaining Risks

- Legal action ordering remains a high-risk contract; broader movement of eval loops should keep this helper under direct tests.
- `_run_periodic_dev_eval()` and `_run_snapshot_promotion_gate()` still own simulator-backed stepping and artifact creation in `train.py`.

## 2026-05-11 - B1 Baseline Snapshot Resolver Extraction

### Scope

- Moved B1 no-league baseline snapshot resolution into `weiss_rl.training.promotion`.
- Preserved `_find_noleague_baseline_snapshot()` in `python/scripts/train.py` as a compatibility wrapper.
- Added artifact-only tests for canonical policy id, legacy policy id, baseline-manifest fallback, and missing/non-baseline cases.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_training_promotion.py python/weiss_rl/tests/test_snapshot_registry.py -k "training_promotion or find_noleague_baseline_snapshot or import_noleague_baseline_anchor or run_snapshot_promotion_gate"` | Initially failed because the new fallback fixture expected the wrong highest-update policy id; corrected the fixture ordering, then passed: 12 passed, 38 deselected, 14 dependency warnings. |
| `uv run ruff check python/scripts/train.py python/weiss_rl/training/promotion.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_training_promotion.py` | Passed. |
| `uv run ruff format --check python/scripts/train.py python/weiss_rl/training/promotion.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_training_promotion.py` | Passed after formatting `train.py`. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 836 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `find_noleague_baseline_snapshot()` to `python/weiss_rl/training/promotion.py`.
- Exported the helper from `python/weiss_rl/training/__init__.py`.
- Updated the train-script compatibility wrapper.
- Extended `python/weiss_rl/tests/test_training_promotion.py`.

### Behavior Changes

No intended behavior changes. The resolver still prefers canonical and legacy B1 anchor ids before falling back to the highest-sort-key snapshot only when the run manifest is marked as a no-league baseline.

### Remaining Risks

- `_import_noleague_baseline_anchor()` still performs weight loading, contract validation, imported artifact writes, and registry mutation in `train.py`.
- Moving that flow should preserve the exact error messages and print behavior currently covered by script-level tests.

## 2026-05-11 - Training Flag Override Extraction

### Scope

- Moved profile-timer and torch-profiler CLI flag normalization into `weiss_rl.training.startup`.
- Preserved `_apply_training_flag_overrides()` in `python/scripts/train.py` as a compatibility wrapper.
- Added direct tests for missing training config, hash-changing flag application, and no-op behavior once requested flags are already set.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_training_startup.py python/weiss_rl/tests/test_script_entrypoint_smokes.py -k "apply_training_flag_overrides or profile_flags_before_hashing or startup"` | Passed: 8 passed, 14 deselected, 14 dependency warnings. |
| `uv run ruff check python/scripts/train.py python/weiss_rl/training/startup.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_training_startup.py` | Passed. |
| `uv run ruff format --check python/scripts/train.py python/weiss_rl/training/startup.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_training_startup.py` | Passed after formatting `train.py`. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 839 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `apply_training_flag_overrides()` to `python/weiss_rl/training/startup.py`.
- Exported the helper from `python/weiss_rl/training/__init__.py`.
- Updated the train-script compatibility wrapper.
- Extended `python/weiss_rl/tests/test_training_startup.py`.

### Behavior Changes

No intended behavior changes. CLI profiling flags still update stack config before config hashing, and already-enabled flags return the original stack unchanged.

### Remaining Risks

- This helper intentionally imports `apply_stack_overrides()` into startup; keep that dependency limited to this config-normalization boundary.
- Broader config override behavior remains owned by the config package and train-script CLI flow.

## 2026-05-11 - Completion Audit Checkpoint

### Scope

- Performed a prompt-to-artifact completion audit against the active refactor objective.
- Added `docs/refactor_completion_audit.md` with requirement-by-requirement evidence and gaps.
- Linked the audit from `docs/README.md`.
- Updated stale early-plan wording in `REFACTOR_PLAN.md` for the active `docs/refactor_log.md` path, current test-count direction, and the now-populated `weiss_rl.training` package.

### Commands and Results

| Command | Result |
| --- | --- |
| `rg --files -g "*.py" python ... Sort-Object Lines -Descending` | Confirmed remaining largest files: `runtime.py`, `model.py`, `impala_learner.py`, `train.py`, and `config/parse.py`. |
| `Get-Content python/scripts/verify_repo.py` | Confirmed verifier coverage: placeholder gate, Ruff, Ruff format, selected mypy, vulture, full pytest, and wrapper dry-runs. |
| Read-only subagent audits | Confirmed docs/GitHub deliverables are mostly present, architecture remains incomplete, and validation gaps remain for full-package mypy, dedicated train/eval smoke, repeated deterministic eval comparison, and historical checkpoint smoke. |

### Behavior Changes

No code behavior changes. This checkpoint updates docs/audit artifacts only.

### Remaining Risks

- The refactor objective is not complete: large god files remain and duplicate eval/training snapshot-resolution surfaces still exist.
- Passing `verify_repo.py` is strong evidence for covered behavior but does not cover every objective requirement.

## 2026-05-11 - Shared B1 Baseline Resolution

### Scope

- Added `weiss_rl.baselines` as a neutral home for no-league baseline role/mode detection and snapshot resolution.
- Preserved training import-contract helper names by delegating them to the shared baseline helpers.
- Updated training promotion and evaluation simulator B1 snapshot discovery to use the shared resolver with caller-specific alias ordering.
- Added evaluation coverage for the old eval alias-preference behavior when both display and canonical B1 snapshot ids are present.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_training_promotion.py python/weiss_rl/tests/test_heuristic_public.py python/weiss_rl/tests/test_script_entrypoint_smokes.py -k "find_noleague_baseline_snapshot or b1_display_id_preference or loads_b1_from_registry_source_run or uses_nested_manifest_role_for_latest_b1_snapshot or uses_legacy_manifest_training_mode_for_latest_b1_snapshot or noleague_import_contract"` | Passed: 8 passed, 44 deselected, 14 dependency warnings. |
| `uv run ruff check python/weiss_rl/baselines.py python/weiss_rl/training/import_contracts.py python/weiss_rl/training/promotion.py python/weiss_rl/eval/simulator_runner.py python/weiss_rl/tests/test_heuristic_public.py python/weiss_rl/tests/test_training_promotion.py` | Passed after import sorting. |
| `uv run ruff format --check python/weiss_rl/baselines.py python/weiss_rl/training/import_contracts.py python/weiss_rl/training/promotion.py python/weiss_rl/eval/simulator_runner.py python/weiss_rl/tests/test_heuristic_public.py python/weiss_rl/tests/test_training_promotion.py` | Passed after formatting `simulator_runner.py` and `test_heuristic_public.py`. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 840 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/baselines.py`.
- Updated `python/weiss_rl/training/import_contracts.py`, `python/weiss_rl/training/promotion.py`, and `python/weiss_rl/eval/simulator_runner.py`.
- Extended `python/weiss_rl/tests/test_heuristic_public.py`.

### Behavior Changes

No intended behavior changes. Training still prefers the canonical B1 policy id before the display name, while eval still prefers the display-name policy id before the canonical B1 id; both now share the same registry/manifest fallback implementation.

### Remaining Risks

- Snapshot eval model loading is still duplicated between training and evaluation.
- Full B1 import flows still combine validation, weight loading, artifact writes, registry mutation, and CLI behavior in `train.py`.

## 2026-05-11 - Shared Snapshot Eval Model Loading

### Scope

- Added `weiss_rl.model_loading` as a neutral helper module for snapshot-to-eval-model loading and public-heuristic guidance payload restoration.
- Updated `weiss_rl.training.guidance`, `python/scripts/train.py`, and `weiss_rl.eval.simulator_runner` to delegate to the shared helper while preserving compatibility wrappers.
- Added direct tests for CPU placement, torch-load options, state-dict loading, builder argument/spec forwarding, eval mode, guidance restoration, missing `model_state_dict`, and missing model config.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_model_loading.py python/weiss_rl/tests/test_training_guidance.py python/weiss_rl/tests/test_heuristic_public.py python/weiss_rl/tests/test_train_stall_monitor.py -k "model_loading or restore_model_guidance or resolve_eval_policies_loads_b1_from_registry_source_run or periodic_dev_eval_opponents_include_optional_b2_when_available"` | Passed: 6 passed, 44 deselected, 14 dependency warnings. |
| `uv run ruff check python/weiss_rl/model_loading.py python/weiss_rl/training/guidance.py python/scripts/train.py python/weiss_rl/eval/simulator_runner.py python/weiss_rl/tests/test_model_loading.py` | Passed after import sorting. |
| `uv run ruff format --check python/weiss_rl/model_loading.py python/weiss_rl/training/guidance.py python/scripts/train.py python/weiss_rl/eval/simulator_runner.py python/weiss_rl/tests/test_model_loading.py` | Passed after formatting `train.py` and `simulator_runner.py`. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 844 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/model_loading.py`.
- Added `python/weiss_rl/tests/test_model_loading.py`.
- Updated train/eval snapshot model loading wrappers and training guidance restoration.

### Behavior Changes

No intended behavior changes. Snapshot eval loading still uses `torch.load(..., map_location="cpu", weights_only=True)`, requires `model_state_dict`, builds the configured policy/value model, restores guidance payload fields, moves the eval model to CPU, loads weights, and sets eval mode.

### Remaining Risks

- The surrounding simulator-backed periodic dev-eval and promotion-gate runners still live in `train.py`.
- Future checkpoint payload schema changes should update both checkpoint tests and model-loading tests.

## 2026-05-11 - Runtime Legal Batching Extraction

### Scope

- Moved runtime legal-action concatenation, packed row slicing, structured legal-batch construction, batch legality concatenation, and packed-meta-width inference into `weiss_rl.runtime_batching`.
- Preserved the private helper names in `weiss_rl.runtime` as compatibility wrappers.
- Added direct tests for packed time-major reordering, meta slicing, structured packed batches, packed decision-batch concatenation, and missing packed-legality failure.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime_batching.py python/weiss_rl/tests/test_runtime.py -k "runtime_batching or concatenate_legal_actions or gae_advantages or concat_optional_time_major_field"` | Passed: 8 passed, 84 deselected. |
| `uv run ruff check python/weiss_rl/runtime_batching.py python/weiss_rl/runtime.py python/weiss_rl/tests/test_runtime_batching.py` | Passed after import sorting. |
| `uv run ruff format --check python/weiss_rl/runtime.py python/weiss_rl/runtime_batching.py python/weiss_rl/tests/test_runtime_batching.py` | Passed after sequential formatting; one earlier parallel check/format attempt raced on `runtime.py`, so final formatting was rerun sequentially. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 848 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Extended `python/weiss_rl/runtime_batching.py`.
- Updated wrapper helpers in `python/weiss_rl/runtime.py`.
- Added `python/weiss_rl/tests/test_runtime_batching.py`.

### Behavior Changes

No intended behavior changes. Runtime legal-action ordering, packed offsets, meta propagation, dense fallback, and GAE behavior remain covered by existing runtime tests plus the new direct module tests.

### Remaining Risks

- `QueueRuntime` still owns actor lifecycle, process collectors, opponent assignment, heuristic action selection, rollout collection, and runtime metrics.
- Legal-action ordering remains a danger zone for future runtime work.

## 2026-05-11 - Runtime Counter Helper Extraction

### Scope

- Moved collector counter templates, timeout-limit normalization, simulator timing counter drains, done-row timeout accounting, and packed step-output legality views into `weiss_rl.runtime_counters`.
- Preserved the private helper names in `weiss_rl.runtime` as compatibility wrappers.
- Added direct tests around counter independence, timeout classification, missing count defaults, simulator timing accumulation, and packed legality view trimming/casting.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime_counters.py python/weiss_rl/tests/test_runtime.py -k "runtime_counters or timeout_rows or runtime_metrics"` | Passed: 8 passed, 87 deselected. |
| `uv run ruff check python/weiss_rl/runtime_counters.py python/weiss_rl/runtime.py python/weiss_rl/tests/test_runtime_counters.py` | Passed; Ruff reported a cache write warning on one run, but the lint result was successful. |
| `uv run ruff format --check python/weiss_rl/runtime_counters.py python/weiss_rl/runtime.py python/weiss_rl/tests/test_runtime_counters.py` | Initially reported `runtime.py` needed formatting after import edits; passed after `uv run ruff format` on the touched files. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 855 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/runtime_counters.py`.
- Updated wrapper helpers in `python/weiss_rl/runtime.py`.
- Added `python/weiss_rl/tests/test_runtime_counters.py`.
- Updated `REFACTOR_PLAN.md`, `CHANGELOG.md`, and `docs/architecture.md`.

### Behavior Changes

No intended behavior changes. Timeout classification still delegates to `classify_episode_end_reason()`, simulator timing counters still drain with the same `simulator_` prefix, and packed step-output legality views still trim to the last offset with `uint32` legal ids/offsets and `uint16` metadata.

### Remaining Risks

- `QueueRuntime` still owns actor lifecycle, process collectors, opponent assignment, heuristic action selection, rollout collection, and runtime metrics.
- Runtime metric aggregation and opponent routing remain good candidates for future extraction, but should keep focused characterization tests around public metric names and PFSP lane semantics.

## 2026-05-11 - Runtime Metrics Extraction

### Scope

- Moved public runtime metric aggregation and collector-counter summing into `weiss_rl.runtime_metrics`.
- Kept `QueueRuntime._runtime_metrics()` as the compatibility method that owns mutable runtime state updates.
- Added direct tests for public metric names, PFSP counter overrides/fallbacks, timer conversions, occupancy and lag percentiles, empty selected batches, and cumulative env-step accounting.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime_metrics.py python/weiss_rl/tests/test_runtime.py -k "runtime_metrics or report_window_and_cumulative"` | Passed: 4 passed, 87 deselected. |
| `uv run ruff check python/weiss_rl/runtime_metrics.py python/weiss_rl/runtime.py python/weiss_rl/tests/test_runtime_metrics.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/runtime_metrics.py python/weiss_rl/runtime.py python/weiss_rl/tests/test_runtime_metrics.py` | Initially reported `runtime.py` and `runtime_metrics.py` needed formatting; passed after formatting the touched files. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 858 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/runtime_metrics.py`.
- Updated `QueueRuntime._runtime_metrics()` in `python/weiss_rl/runtime.py`.
- Added `python/weiss_rl/tests/test_runtime_metrics.py`.
- Updated `REFACTOR_PLAN.md`, `CHANGELOG.md`, and `docs/architecture.md`.

### Behavior Changes

No intended behavior changes. Public metric names, collector counter prefixes, simulator timer conversions, PFSP fallback counters, queue occupancy percentiles, policy lag percentiles, and cumulative env-step updates are preserved.

### Remaining Risks

- `QueueRuntime` still owns actor lifecycle, process collectors, opponent assignment, heuristic action selection, rollout collection, and PFSP lane selection.
- Runtime metric naming is now easier to audit, but downstream TensorBoard grouping and log consumers should still be treated as compatibility surfaces.

## 2026-05-11 - Runtime Hashing Extraction

### Scope

- Moved deterministic unroll hashing and state-dict fingerprinting into `weiss_rl.runtime_hashing`.
- Preserved `_hash_unroll()` and `_hash_state_dict()` in `weiss_rl.runtime` as compatibility wrappers.
- Added direct tests for byte ordering, non-contiguous unroll arrays, state-dict key-order independence, dtype/shape inclusion, and shape-sensitive hashes when raw bytes match.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime_hashing.py python/weiss_rl/tests/test_runtime.py -k "runtime_hashing or unroll_hash or shared_collector"` | Passed: 4 passed, 87 deselected. |
| `uv run ruff check python/weiss_rl/runtime_hashing.py python/weiss_rl/runtime.py python/weiss_rl/tests/test_runtime_hashing.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/runtime_hashing.py python/weiss_rl/runtime.py python/weiss_rl/tests/test_runtime_hashing.py` | Initially reported `runtime.py` needed formatting after import edits; passed after formatting the touched files. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 861 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/runtime_hashing.py`.
- Updated hash wrappers in `python/weiss_rl/runtime.py`.
- Added `python/weiss_rl/tests/test_runtime_hashing.py`.
- Updated `REFACTOR_PLAN.md`, `CHANGELOG.md`, and `docs/architecture.md`.

### Behavior Changes

No intended behavior changes. Unroll hashes still concatenate contiguous action, reward, and episode-seed bytes in the same order. State-dict fingerprints still sort keys, include dtype and shape metadata, and hash contiguous tensor/array bytes.

### Remaining Risks

- Hash wrappers are covered directly, but any future change to what fields participate in unroll hashes would be a reproducibility behavior change and must be separately justified.

## 2026-05-11 - Runtime IPC State-Dict Serialization

### Scope

- Moved process-collector state-dict serialization/deserialization helpers into `weiss_rl.runtime_ipc`.
- Preserved `_serialize_state_dict_for_ipc()` and `_deserialize_state_dict_from_ipc()` in `weiss_rl.runtime` as compatibility wrappers.
- Added direct tests for tensor-to-independent-NumPy conversion, NumPy-to-independent-tensor restoration, deep-copied metadata, and stringified keys.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime_ipc.py python/weiss_rl/tests/test_runtime.py -k "runtime_ipc or shared_collector or process_collector"` | Passed: 9 passed, 82 deselected. |
| `uv run ruff check python/weiss_rl/runtime_ipc.py python/weiss_rl/runtime.py python/weiss_rl/tests/test_runtime_ipc.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/runtime_ipc.py python/weiss_rl/runtime.py python/weiss_rl/tests/test_runtime_ipc.py` | Initially reported `runtime.py` needed formatting after import edits; passed after formatting the touched files. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 864 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/runtime_ipc.py`.
- Updated IPC wrapper helpers in `python/weiss_rl/runtime.py`.
- Added `python/weiss_rl/tests/test_runtime_ipc.py`.
- Updated `REFACTOR_PLAN.md`, `CHANGELOG.md`, and `docs/architecture.md`.

### Behavior Changes

No intended behavior changes. Tensor payloads still serialize to copied CPU NumPy arrays, NumPy payloads still deserialize to copied tensors, non-array metadata is deep-copied, and keys are still stringified.

### Remaining Risks

- Process collector transport is still owned by `QueueRuntime`; future movement must preserve payload shapes, slot lifetimes, and copy isolation.

## 2026-05-11 - Runtime Config Extraction

### Scope

- Moved `QueueRuntimeConfig` and runtime config construction into `weiss_rl.runtime_config`.
- Preserved `QueueRuntimeConfig`, `QueueRuntimeMode`, and `build_runtime_config()` through `weiss_rl.runtime` compatibility imports/wrappers.
- Added direct tests for `total_envs`, minimal-batch actor/unroll behavior, actor reload interval clamping, and missing system/training config errors.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime_config.py python/weiss_rl/tests/test_runtime.py -k "build_runtime_config or resolve_actor_topology or runtime_config"` | Passed: 8 passed, 83 deselected. |
| `uv run ruff check python/weiss_rl/runtime_config.py python/weiss_rl/runtime.py python/weiss_rl/tests/test_runtime_config.py` | Initially reported import ordering in `runtime.py`; passed after `uv run ruff check --fix` on the touched files. |
| `uv run ruff format --check python/weiss_rl/runtime_config.py python/weiss_rl/runtime.py python/weiss_rl/tests/test_runtime_config.py` | Initially reported `runtime.py` needed formatting; passed after formatting the touched files. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 867 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/runtime_config.py`.
- Updated `python/weiss_rl/runtime.py` to import the config dataclass and delegate `build_runtime_config()`.
- Added `python/weiss_rl/tests/test_runtime_config.py`.
- Updated `REFACTOR_PLAN.md`, `CHANGELOG.md`, and `docs/architecture.md`.

### Behavior Changes

No intended behavior changes. Runtime config construction still reads the same stack config fields, delegates actor topology to the existing topology helper, preserves minimal-batch queue clamping, and clamps actor reload intervals to at least one.

### Remaining Risks

- `QueueRuntime` construction still owns actor/process setup, environment construction, league state, and model wiring.
- Runtime config public import compatibility should be preserved in any later runtime package split.

### Current Size Snapshot

- `python/weiss_rl/runtime.py`: 6462 lines after the latest runtime extractions.
- Largest remaining production files: `runtime.py`, `model.py`, `learners/impala_learner.py`, `python/scripts/train.py`, and `config/parse.py`.

## 2026-05-11 - Runtime Opponent Bookkeeping Extraction

### Scope

- Moved active opponent mix fraction scheduling, actor-heuristic annealing, fixed-opponent slots, fixed-opponent active checks, promotion-gated recent reservoir sizing, timeout-heavy opponent filtering, and diversity-floor restoration into `weiss_rl.runtime_opponents`.
- Preserved the existing `QueueRuntime` private method names as compatibility wrappers/delegators.
- Added direct tests for anneal endpoints, no-league expiry, warmup snapshot gating, actor heuristic clamping/delayed start, fixed anchor slots, forced/scheduled fixed-opponent activation, recent reservoir sizing, timeout filtering, and diversity-floor restoration.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime_opponents.py python/weiss_rl/tests/test_runtime.py -k "runtime_opponents or active_heuristic_public or active_actor_heuristic or active_warmup or sample_opponent_policy_ids or fixed_opponent or refresh_opponent_pool"` | Initially found a compatibility regression where `_filter_timeout_heavy_opponents()` eagerly read `_outcomes`; after fixing the wrapper, passed: 28 passed, 66 deselected. |
| `uv run python python/scripts/verify_repo.py` | Initially failed in pytest because `_active_warmup_snapshot_mix_fraction()` eagerly read missing opponent-pool attrs in a partial runtime metrics test double; the wrapper was fixed with `getattr` defaults. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime_opponents.py python/weiss_rl/tests/test_runtime.py -k "runtime_opponents or runtime_metrics or active_heuristic_public or active_actor_heuristic or active_warmup or sample_opponent_policy_ids or fixed_opponent or refresh_opponent_pool"` | Passed after both lazy-access fixes: 29 passed, 65 deselected. |
| `uv run ruff check python/weiss_rl/runtime_opponents.py python/weiss_rl/runtime.py python/weiss_rl/tests/test_runtime_opponents.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/runtime_opponents.py python/weiss_rl/runtime.py python/weiss_rl/tests/test_runtime_opponents.py` | Passed after formatting touched files. |
| `uv run python python/scripts/verify_repo.py` | Passed after the lazy-access fixes: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 873 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/runtime_opponents.py`.
- Updated opponent bookkeeping wrappers in `python/weiss_rl/runtime.py`.
- Added `python/weiss_rl/tests/test_runtime_opponents.py`.
- Updated `REFACTOR_PLAN.md`, `CHANGELOG.md`, and `docs/architecture.md`.

### Behavior Changes

No intended behavior changes. The first focused test run caught and prevented an accidental eager `_outcomes` access in partial runtime test doubles; the wrapper now preserves the old short-circuit behavior when promotion gating is disabled.

### Remaining Risks

- `_sample_opponent_policy_ids()` still owns sampled group construction, RNG draws, PFSP snapshot sampling, and lane counters in `QueueRuntime`.
- Any future extraction of sampling itself should be paired with lane-count, fallback, RNG, and candidate-group tests.

### Current Size Snapshot

- `python/weiss_rl/runtime.py`: 6366 lines after the opponent bookkeeping extraction.
- Largest remaining production files: `runtime.py`, `model.py`, `learners/impala_learner.py`, `python/scripts/train.py`, and `config/parse.py`.

## 2026-05-11 - Model Tensor Ops Extraction

### Scope

- Moved pure tensor masking, pooling, optional embedding, packed-row math, packed row log-normalizer/CDF, deterministic seed mixing, masked log-softmax, and masked entropy helpers into `weiss_rl.model_tensor_ops`.
- Preserved the existing private helper names in `weiss_rl.model` as wrappers, including monkeypatch-sensitive sampling dependencies.
- Added direct tests for masked pooling, empty rows, optional embedding sentinel handling, negative fill values, packed row indices/log-z/CDF, deterministic uniform seed mapping, derived seeds, masked log-softmax, entropy, and shape mismatch errors.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_model_tensor_ops.py python/weiss_rl/tests/test_contracts.py python/weiss_rl/tests/test_runtime.py -k "model_tensor_ops or negative_logits_fill_value or sample_packed_action_scores"` | Passed: 9 passed, 136 deselected. |
| `uv run ruff check python/weiss_rl/model_tensor_ops.py python/weiss_rl/model.py python/weiss_rl/tests/test_model_tensor_ops.py` | Initially reported import sorting in the new test; passed after `ruff check --fix`. |
| `uv run ruff format --check python/weiss_rl/model_tensor_ops.py python/weiss_rl/model.py python/weiss_rl/tests/test_model_tensor_ops.py` | Initially reported `model.py` needed formatting; passed after formatting touched files. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 880 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/model_tensor_ops.py`.
- Updated helper wrappers in `python/weiss_rl/model.py`.
- Added `python/weiss_rl/tests/test_model_tensor_ops.py`.
- Updated `REFACTOR_PLAN.md`, `CHANGELOG.md`, and `docs/architecture.md`.

### Behavior Changes

No intended behavior changes. The private `weiss_rl.model` helper names remain available, and `_sample_packed_action_scores()` still calls the `model.py` wrapper names so existing monkeypatch-based characterization tests continue to exercise the same contract.

### Remaining Risks

- `model.py` still owns encoders, recurrent model behavior, structured candidate scoring, factorized scoring, public heuristic biasing, and builder logic.
- Future model extractions should preserve private wrapper names when tests or downstream scripts reach through `weiss_rl.model`.

### Current Size Snapshot

- `python/weiss_rl/model.py`: 5017 lines after the tensor-op extraction.
- Largest remaining production files: `runtime.py`, `model.py`, `learners/impala_learner.py`, `python/scripts/train.py`, and `config/parse.py`.

## 2026-05-11 - Model Observation Contract Extraction

### Scope

- Moved structured observation contract dataclass, card-vector slice names, slice/header lookup helpers, and structured observation contract construction into `weiss_rl.model_observation_contract`.
- Preserved the private `weiss_rl.model` helper names and type alias as wrappers/imports.
- Added direct tests for stage/card scalar index collection, sentinel values, choice header indices, missing lookup behavior, non-self-first rejection, and stage-width validation.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_model_observation_contract.py python/weiss_rl/tests/test_contracts.py -k "model_observation_contract or structured_v2 or observation_spec"` | Initially failed because the new characterization expected omitted `hand` vector indices; corrected to match current behavior, then passed: 7 passed, 47 deselected. |
| `uv run ruff check python/weiss_rl/model_observation_contract.py python/weiss_rl/model.py python/weiss_rl/tests/test_model_observation_contract.py` | Initially reported import sorting in `model.py`; passed after `ruff check --fix`. |
| `uv run ruff format --check python/weiss_rl/model_observation_contract.py python/weiss_rl/model.py python/weiss_rl/tests/test_model_observation_contract.py` | Passed after formatting touched files. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 884 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/model_observation_contract.py`.
- Updated observation-contract wrappers/imports in `python/weiss_rl/model.py`.
- Added `python/weiss_rl/tests/test_model_observation_contract.py`.
- Updated `REFACTOR_PLAN.md`, `CHANGELOG.md`, and `docs/architecture.md`.

### Behavior Changes

No intended behavior changes. The corrected direct test preserves the existing behavior that `card_scalar_indices` includes stage slot scalar positions and card-id vector slices such as `hand` and `waiting_room_top`.

### Remaining Risks

- `model.py` still owns typed encoders, recurrent model behavior, structured candidate scoring, factorized scoring, public heuristic biasing, and builder logic.
- Future observation-layout changes should update both observation-layout tests and model observation-contract tests.

### Current Size Snapshot

- `python/weiss_rl/model.py`: 4945 lines after the observation-contract extraction.
- Largest remaining production files: `runtime.py`, `model.py`, `learners/impala_learner.py`, `python/scripts/train.py`, and `config/parse.py`.

## 2026-05-11 - Model Factorized Tensor Helper Extension

### Scope

- Extended `weiss_rl.model_tensor_ops` with card-id bucketing, factorized row lookup, and factorized row-value scatter helpers.
- Preserved `_bucket_card_ids()`, `_factorized_local_row_indices()`, and `_scatter_factorized_row_values()` in `weiss_rl.model` as wrappers.
- Extended direct tests for nonpositive card IDs, vocabulary hashing, row lookup validation, empty selected rows, and scatter fill behavior.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_model_tensor_ops.py python/weiss_rl/tests/test_contracts.py python/weiss_rl/tests/test_runtime.py -k "model_tensor_ops or negative_logits_fill_value or sample_packed_action_scores or structured_v2"` | Passed: 13 passed, 134 deselected. |
| `uv run ruff check python/weiss_rl/model_tensor_ops.py python/weiss_rl/model.py python/weiss_rl/tests/test_model_tensor_ops.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/model_tensor_ops.py python/weiss_rl/model.py python/weiss_rl/tests/test_model_tensor_ops.py` | Initially reported `model.py` needed formatting; passed after formatting touched files. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 886 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Extended `python/weiss_rl/model_tensor_ops.py`.
- Updated wrappers in `python/weiss_rl/model.py`.
- Extended `python/weiss_rl/tests/test_model_tensor_ops.py`.
- Updated `REFACTOR_PLAN.md` and `docs/refactor_log.md`.

### Behavior Changes

No intended behavior changes. Card-id bucketing, factorized row lookup validation, and scatter fill behavior remain available through the original private `weiss_rl.model` helper names.

### Remaining Risks

- Larger packed/factorized scoring methods still live in `model.py`; moving them will require broader characterization around action IDs, log probabilities, and public heuristic biasing.

### Current Size Snapshot

- `python/weiss_rl/model.py`: 4930 lines after the factorized tensor-helper extension.
- Largest remaining production files: `runtime.py`, `model.py`, `learners/impala_learner.py`, `python/scripts/train.py`, and `config/parse.py`.

## 2026-05-11 - Learner Tensor Ops Extraction

### Scope

- Moved deterministic segment reductions, grouped sum, weighted mean, and nonfinite-index reporting into `weiss_rl.learners.tensor_ops`.
- Preserved the old private helper names in `weiss_rl.learners.impala_learner` as wrappers.
- Added direct tests for empty segments, stable grouped logsumexp, invalid group IDs, clamped zero-weight means, and NumPy/Torch nonfinite coordinate reporting.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_learner_tensor_ops.py python/weiss_rl/tests/test_impala_learner.py -k "learner_tensor_ops or structured_teacher_auxiliary or masked_action_logp"` | Passed: 18 passed, 29 deselected. |
| `uv run ruff check python/weiss_rl/learners/tensor_ops.py python/weiss_rl/learners/impala_learner.py python/weiss_rl/tests/test_learner_tensor_ops.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/learners/tensor_ops.py python/weiss_rl/learners/impala_learner.py python/weiss_rl/tests/test_learner_tensor_ops.py` | Initially reported `impala_learner.py` needed formatting; passed after formatting touched files. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 891 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/learners/tensor_ops.py`.
- Updated helper wrappers in `python/weiss_rl/learners/impala_learner.py`.
- Added `python/weiss_rl/tests/test_learner_tensor_ops.py`.
- Updated `REFACTOR_PLAN.md`, `CHANGELOG.md`, `docs/architecture.md`, and `docs/refactor_completion_audit.md`.

### Behavior Changes

No intended behavior changes. Segment reductions keep empty rows at `-inf`, grouped sums still ignore invalid group IDs, weighted means still clamp the denominator to at least one, and nonfinite diagnostics return the same coordinate payload shape.

### Remaining Risks

- `impala_learner.py` still owns update orchestration, tensor conversion, packed legality handling, diagnostics, optimization, and structured auxiliary losses.
- Future learner movement should preserve private wrappers until structured auxiliary and update-loop contracts are tested at narrower boundaries.

### Current Size Snapshot

- `python/weiss_rl/learners/impala_learner.py`: 4105 lines after the tensor-op extraction.
- Largest remaining production files: `runtime.py`, `model.py`, `learners/impala_learner.py`, `python/scripts/train.py`, and `config/parse.py`.

## 2026-05-11 - Learner Structured Auxiliary Metadata Extraction

### Scope

- Moved public-heuristic profile normalization, structured catalog metadata construction, and public-heuristic family-id resolution into `weiss_rl.learners.structured_auxiliary`.
- Preserved the old private helper and constant names in `weiss_rl.learners.impala_learner` as compatibility wrappers/aliases.
- Added direct tests for profile normalization, profile-mode validation, decoded structured catalog metadata, main-move pressure action id, and requested family-id resolution.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_learner_structured_auxiliary.py python/weiss_rl/tests/test_impala_learner.py -k "learner_structured_auxiliary or structured_teacher_auxiliary or public_heuristic"` | Initially failed because the new tiny-catalog expectations guessed decoded play/move indices incorrectly; after correcting the characterization, passed: 21 passed, 25 deselected. |
| `uv run ruff check python/weiss_rl/learners/structured_auxiliary.py python/weiss_rl/learners/impala_learner.py python/weiss_rl/tests/test_learner_structured_auxiliary.py` | Initially reported unused private-constant compatibility imports; passed after preserving constants via a module alias. |
| `uv run ruff format --check python/weiss_rl/learners/structured_auxiliary.py python/weiss_rl/learners/impala_learner.py python/weiss_rl/tests/test_learner_structured_auxiliary.py` | Initially reported `impala_learner.py` needed formatting; passed after formatting touched files. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 895 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/learners/structured_auxiliary.py`.
- Updated structured auxiliary wrappers/imports in `python/weiss_rl/learners/impala_learner.py`.
- Added `python/weiss_rl/tests/test_learner_structured_auxiliary.py`.
- Updated `REFACTOR_PLAN.md`, `CHANGELOG.md`, `docs/architecture.md`, and `docs/refactor_completion_audit.md`.

### Behavior Changes

No intended behavior changes. Public-heuristic profile normalization, mode validation, structured action metadata, and family-id resolution still produce the same values and errors through the original private learner helper names.

### Remaining Risks

- Packed structured legal-view construction and the structured auxiliary loss blocks still live in `impala_learner.py`.
- Future movement should preserve dense/packed parity and public-heuristic profile scheduling tests.

### Current Size Snapshot

- `python/weiss_rl/learners/impala_learner.py`: 4040 lines after the structured auxiliary metadata extraction.
- Largest remaining production files: `runtime.py`, `model.py`, `learners/impala_learner.py`, `python/scripts/train.py`, and `config/parse.py`.

## 2026-05-11 - Learner Packed Structured Legal View Extraction

### Scope

- Moved packed structured legal-view construction into `weiss_rl.learners.structured_auxiliary`.
- Preserved the old `_PackedStructuredLegalView` type name and `_packed_structured_legal_view()` helper in `weiss_rl.learners.impala_learner`.
- Extended direct tests for dense logits, flat packed logits, no-logit default scores, empty rows, metadata sentinel normalization, missing inputs, and shape validation errors.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_learner_structured_auxiliary.py python/weiss_rl/tests/test_impala_learner.py -k "learner_structured_auxiliary or packed_structured_legal_view or structured_teacher_auxiliary or public_heuristic"` | Passed: 24 passed, 25 deselected. |
| `uv run ruff check python/weiss_rl/learners/structured_auxiliary.py python/weiss_rl/learners/impala_learner.py python/weiss_rl/tests/test_learner_structured_auxiliary.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/learners/structured_auxiliary.py python/weiss_rl/learners/impala_learner.py python/weiss_rl/tests/test_learner_structured_auxiliary.py` | Initially reported `impala_learner.py` needed formatting; passed after formatting touched files. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 898 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Extended `python/weiss_rl/learners/structured_auxiliary.py`.
- Updated packed legal-view wrappers/imports in `python/weiss_rl/learners/impala_learner.py`.
- Extended `python/weiss_rl/tests/test_learner_structured_auxiliary.py`.
- Updated `REFACTOR_PLAN.md`, `CHANGELOG.md`, `docs/architecture.md`, and `docs/refactor_completion_audit.md`.

### Behavior Changes

No intended behavior changes. Packed row indices, selected logits, row log normalizers, empty-row flags, and metadata sentinel normalization remain available through the original private learner helper.

### Remaining Risks

- Packed group log-prob computation and structured auxiliary target/loss construction still live in `impala_learner.py`.
- Legal-action metadata ordering remains a danger zone; future changes should retain dense/packed metric parity tests in focused validation.

### Current Size Snapshot

- `python/weiss_rl/learners/impala_learner.py`: 3985 lines after the packed structured legal-view extraction.
- Largest remaining production files: `runtime.py`, `model.py`, `learners/impala_learner.py`, `python/scripts/train.py`, and `config/parse.py`.

## 2026-05-11 - Learner Packed Structured Probability Helpers

### Scope

- Moved packed group log-probability computation and packed soft-target cross-entropy/top-mass/entropy helpers into `weiss_rl.learners.structured_auxiliary`.
- Preserved the old `_packed_group_log_probs()` and `_packed_soft_target_cross_entropy()` names in `weiss_rl.learners.impala_learner`.
- Extended direct tests with manual row-level probability calculations, candidate-mask behavior, empty group-count behavior, public-heuristic temperature validation, and target-logit alignment errors.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_learner_structured_auxiliary.py python/weiss_rl/tests/test_impala_learner.py -k "learner_structured_auxiliary or packed_group_log_probs or packed_soft_target_cross_entropy or structured_teacher_auxiliary or public_heuristic"` | Passed: 26 passed, 25 deselected. |
| `uv run ruff check python/weiss_rl/learners/structured_auxiliary.py python/weiss_rl/learners/impala_learner.py python/weiss_rl/tests/test_learner_structured_auxiliary.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/learners/structured_auxiliary.py python/weiss_rl/learners/impala_learner.py python/weiss_rl/tests/test_learner_structured_auxiliary.py` | Initially reported `impala_learner.py` needed formatting; passed after formatting touched files. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 900 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Extended `python/weiss_rl/learners/structured_auxiliary.py`.
- Updated packed probability helper wrappers/imports in `python/weiss_rl/learners/impala_learner.py`.
- Extended `python/weiss_rl/tests/test_learner_structured_auxiliary.py`.
- Updated `REFACTOR_PLAN.md`, `CHANGELOG.md`, `docs/architecture.md`, and `docs/refactor_completion_audit.md`.

### Behavior Changes

No intended behavior changes. Packed group log-probs still normalize against either the full packed row or the candidate-mask subset, and public-heuristic soft target metrics still use the same temperature, target alignment, and student-top-mass semantics.

### Remaining Risks

- The full structured auxiliary loss assembly is still in `impala_learner.py` and should not be moved without a separate focused characterization plan.
- Public-heuristic soft targets remain behavior-sensitive because they affect auxiliary losses and metrics.

### Current Size Snapshot

- `python/weiss_rl/learners/impala_learner.py`: 3915 lines after the packed structured probability-helper extraction.
- Largest remaining production files: `runtime.py`, `model.py`, `learners/impala_learner.py`, `python/scripts/train.py`, and `config/parse.py`.

## 2026-05-11 - Learner Batch Field Validation Extraction

### Scope

- Moved learner batch field conversion and validation helpers into `weiss_rl.learners.batch_fields`.
- Preserved the old `ImpalaLearner` private helper names as wrappers.
- Added direct tests for dtype conversion, required-field errors, target shapes, integer seat/index fields, binary seat constraints, hidden-state shape checks, actor/to-play consistency, loss-mask clamping, and bool conversion.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_learner_batch_fields.py python/weiss_rl/tests/test_impala_learner.py -k "learner_batch_fields or optional_time_major or prepare_acting_seat or hidden_state or packed_legal_actions_match_dense_mask_loss"` | Passed: 7 passed, 41 deselected. |
| `uv run ruff check python/weiss_rl/learners/batch_fields.py python/weiss_rl/learners/impala_learner.py python/weiss_rl/tests/test_learner_batch_fields.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/learners/batch_fields.py python/weiss_rl/learners/impala_learner.py python/weiss_rl/tests/test_learner_batch_fields.py` | Initially reported `impala_learner.py` and the new test needed formatting; passed after formatting touched files. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 906 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/learners/batch_fields.py`.
- Updated batch field wrappers/imports in `python/weiss_rl/learners/impala_learner.py`.
- Added `python/weiss_rl/tests/test_learner_batch_fields.py`.
- Updated `REFACTOR_PLAN.md`, `CHANGELOG.md`, `docs/architecture.md`, and `docs/refactor_completion_audit.md`.

### Behavior Changes

No intended behavior changes. Batch conversion still uses the model parameter device, preserves dtype choices from the caller/reference tensor, keeps exact error messages for field shapes and integer/binary constraints, and keeps acting-seat conflict validation unchanged.

### Remaining Risks

- The learner update loop still owns tensor extraction flow, V-trace target handling, optimization, metrics, and numeric fault bundle wiring.
- Future movement around numeric fault bundles should preserve serialized context keys and nonfinite index payloads.

### Current Size Snapshot

- `python/weiss_rl/learners/impala_learner.py`: 3870 lines after the batch field validation extraction.
- Largest remaining production files: `runtime.py`, `model.py`, `learners/impala_learner.py`, `python/scripts/train.py`, and `config/parse.py`.

## 2026-05-11 - Full-Package Mypy Probe

### Scope

- Probed the audit gap around full-package type checking.
- Confirmed `verify_repo.py` still enforces selected-script mypy, not full-package mypy.
- Updated `docs/testing.md` so contributors do not mistake selected-script mypy for full package coverage.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run mypy python/weiss_rl` | Failed: 321 errors across 39 files while checking 205 source files. Representative classes include existing runtime/model/learner union narrowing, test doubles typed as concrete production classes, checkpoint protocol attribute gaps, and config/eval helper type annotations. |

### Behavior Changes

No code behavior changes. This checkpoint only records validation evidence and documentation.

### Remaining Risks

- Full-package mypy remains a real validation gap and should not be listed as passed.
- Any future move toward full-package type enforcement should start with scoped modules and avoid type-only rewrites that obscure behavior-preservation review.

## 2026-05-11 - Dedicated Smoke Validation Probe

### Scope

- Ran the scaffold training smoke with a fresh refactor validation run label.
- Ran the non-rollout evaluation contract check.
- Ran public-demo final evaluation twice from the same staged demo run and compared the two summary payloads after removing the expected output-directory field.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python python/scripts/train.py --stack-config configs/stack_smoke.yaml --run-label refactor_smoke_20260511` | Passed: wrote `runs/refactor_smoke_20260511/manifest.json` and exited as scaffold-only because environment/training/model blocks are absent. |
| `uv run python python/scripts/eval.py --stack-config configs/presets/structured_acceptance_standard_thesis_eval.yaml` | Passed: verified runtime spec bundle and completed the evaluation contract check without summarizing episodes. |
| `uv run python python/scripts/train.py --stack-config configs/presets/structured_acceptance_standard.yaml --public-demo --run-label refactor_public_demo_det_20260511` | Passed: staged synthetic public-demo catalog and policy bundle under `runs/refactor_public_demo_det_20260511/`. |
| `uv run python python/scripts/eval.py --stack-config configs/presets/structured_acceptance_standard_thesis_eval.yaml --public-demo --run-dir runs/refactor_public_demo_det_20260511 --final-eval-dir runs/refactor_public_demo_det_20260511/eval/final_eval_a` | Passed: wrote public-demo `summary.json` for 4 policies and 10 matchups. |
| `uv run python python/scripts/eval.py --stack-config configs/presets/structured_acceptance_standard_thesis_eval.yaml --public-demo --run-dir runs/refactor_public_demo_det_20260511 --final-eval-dir runs/refactor_public_demo_det_20260511/eval/final_eval_b` | Passed: wrote the second public-demo `summary.json` for the same 4 policies and 10 matchups. |
| Inline JSON comparison of `final_eval_a/summary.json` vs `final_eval_b/summary.json` | Exact equality after removing the expected `output_dir` field: `equal_without_output_dir: True`. |

### Behavior Changes

No code behavior changes. This checkpoint only records validation evidence.

### Remaining Risks

- The deterministic comparison is public-demo only. Canonical simulator-backed repeated final-eval comparison remains a heavier validation gap.
- The scaffold train smoke proves manifest/config/spec plumbing, not learner updates or rollout collection.

## 2026-05-11 - Learner Numeric Fault Helper Extraction

### Scope

- Moved learner numeric fault bundle helpers into `weiss_rl.learners.faults`.
- Preserved the old `ImpalaLearner` private helper names as wrappers.
- Added direct tests for batch-size precedence, fault directory precedence, snapshot field selection, payload schema, tensor nonfinite context, gradient collection, and gradient fault context.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_learner_faults.py python/weiss_rl/tests/test_impala_learner.py -k "learner_faults or nonfinite_forward_logits or nonfinite_gradients"` | Passed: 9 passed, 40 deselected. |
| `uv run ruff check python/weiss_rl/learners/faults.py python/weiss_rl/learners/impala_learner.py python/weiss_rl/tests/test_learner_faults.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/learners/faults.py python/weiss_rl/learners/impala_learner.py python/weiss_rl/tests/test_learner_faults.py` | Initially reported `impala_learner.py` needed formatting; passed after formatting touched files. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 913 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/learners/faults.py`.
- Updated numeric fault wrappers/imports in `python/weiss_rl/learners/impala_learner.py`.
- Added `python/weiss_rl/tests/test_learner_faults.py`.
- Updated `REFACTOR_PLAN.md`, `CHANGELOG.md`, `docs/architecture.md`, and `docs/refactor_completion_audit.md`.

### Behavior Changes

No intended behavior changes. Numeric fault bundles keep the same prefix, component, stage, update/policy counters, pass action id, batch snapshot keys, context keys, and nonfinite index payload shapes.

### Remaining Risks

- Checkpoint metadata writing and training metric logging remain in `impala_learner.py`.
- Numeric fault bundle serialization should remain a compatibility surface for debugging and replay inspection.

### Current Size Snapshot

- `python/weiss_rl/learners/impala_learner.py`: 3834 lines after the numeric fault helper extraction.
- Largest remaining production files: `runtime.py`, `model.py`, `learners/impala_learner.py`, `python/scripts/train.py`, and `config/parse.py`.

## 2026-05-11 - Learner Logging And Metadata Extraction

### Scope

- Moved learner checkpoint metadata payload/writer and training metric record construction into `weiss_rl.learners.logging`.
- Preserved the old `ImpalaLearner` private helper names and call order.
- Added direct tests for checkpoint metadata payloads, disabled checkpoint metadata writes, JSON newline behavior, custom metric availability flags, p95 propagation, entropy handling, and `TrainingMetrics` field assembly.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_learner_logging.py python/weiss_rl/tests/test_impala_learner.py python/weiss_rl/tests/test_training_logger.py -k "learner_logging or checkpoint_metadata or logs_masked_metrics or vtrace_batch_metrics_available"` | Passed: 6 passed, 62 deselected. |
| `uv run ruff check python/weiss_rl/learners/logging.py python/weiss_rl/learners/impala_learner.py python/weiss_rl/tests/test_learner_logging.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/learners/logging.py python/weiss_rl/learners/impala_learner.py python/weiss_rl/tests/test_learner_logging.py` | Initially reported `impala_learner.py` needed formatting; passed after formatting touched files. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 916 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/learners/logging.py`.
- Updated checkpoint/logging wrappers/imports in `python/weiss_rl/learners/impala_learner.py`.
- Added `python/weiss_rl/tests/test_learner_logging.py`.
- Updated `REFACTOR_PLAN.md`, `CHANGELOG.md`, `docs/architecture.md`, and `docs/refactor_completion_audit.md`.

### Behavior Changes

No intended behavior changes. Checkpoint metadata sidecars keep their file names, sorted JSON payload, newline terminator, and print behavior through the learner wrapper. Training metrics keep their existing fallback/override precedence and custom metric names.

### Remaining Risks

- The main learner update loop still owns loss computation, optimization, and timing decisions.
- Further learner decomposition should pause until a broader orchestration test plan is in place.

### Current Size Snapshot

- `python/weiss_rl/learners/impala_learner.py`: 3807 lines after the logging and metadata extraction.
- Largest remaining production files: `runtime.py`, `model.py`, `learners/impala_learner.py`, `python/scripts/train.py`, and `config/parse.py`.

## 2026-05-11 - Learner Legal Field Validation Extraction

### Scope

- Moved learner observation/action/legal-mask validators and packed legality resolution into `weiss_rl.learners.legal_fields`.
- Preserved the old `ImpalaLearner` private helper names as wrappers.
- Added direct tests for observation/action/legality shape errors, all supported legality representations, dense mask conversion from packed ids and `LegalActionBatch`, missing legality errors, packed offset shape errors, and structured metadata requirements.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_learner_legal_fields.py python/weiss_rl/tests/test_impala_learner.py -k "learner_legal_fields or packed_legal_actions_match_dense_mask_loss or structured_updates_require_packed"` | Passed: 5 passed, 41 deselected. |
| `uv run ruff check python/weiss_rl/learners/legal_fields.py python/weiss_rl/learners/impala_learner.py python/weiss_rl/tests/test_learner_legal_fields.py` | Initially reported import ordering in `impala_learner.py`; passed after `ruff check --fix`. |
| `uv run ruff format --check python/weiss_rl/learners/legal_fields.py python/weiss_rl/learners/impala_learner.py python/weiss_rl/tests/test_learner_legal_fields.py` | Initially reported formatting needed in all three touched files; passed after formatting touched files. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 920 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/learners/legal_fields.py`.
- Updated legal field wrappers/imports in `python/weiss_rl/learners/impala_learner.py`.
- Added `python/weiss_rl/tests/test_learner_legal_fields.py`.
- Updated `REFACTOR_PLAN.md`, `CHANGELOG.md`, `docs/architecture.md`, and `docs/refactor_completion_audit.md`.

### Behavior Changes

No intended behavior changes. Required observation/action/mask errors, dense mask conversion, packed offset validation, and structured metadata requirements remain available through the original private learner helper names.

### Remaining Risks

- Packed row slicing, candidate-position lookup, and scatter helpers still live in `impala_learner.py` and should not move without dedicated row-level tests.
- Legal action metadata remains a hard compatibility boundary.

### Current Size Snapshot

- `python/weiss_rl/learners/impala_learner.py`: 3759 lines after the legal field validation extraction.
- Largest remaining production files: `runtime.py`, `model.py`, `learners/impala_learner.py`, `python/scripts/train.py`, and `config/parse.py`.

## 2026-05-11 - Config Parsing Utility Extraction

### Scope

- Moved strict config document loading, repo-root/path resolution, scalar/list validators, unknown-key rejection, deep merge, and preset inheritance loading into `weiss_rl.config.parsing_utils`.
- Preserved the old private helper names in `weiss_rl.config.parse` as forwarding wrappers.
- Added direct tests for YAML/JSON mapping roots, strict bool/int/float/text/list validation, sorted choice and unknown-key errors, deep-merge behavior, preset extends/cycle handling, and repo path resolution.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_config_parsing_utils.py python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_config_overrides.py` | Passed: 46 passed. |
| `uv run ruff check python/weiss_rl/config/parsing_utils.py python/weiss_rl/config/parse.py python/weiss_rl/tests/test_config_parsing_utils.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/config/parsing_utils.py python/weiss_rl/config/parse.py python/weiss_rl/tests/test_config_parsing_utils.py` | Initially reported `parse.py` needed formatting; passed after formatting touched files. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 928 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/config/parsing_utils.py`.
- Updated `python/weiss_rl/config/parse.py` to use wrappers around the extracted helpers.
- Added `python/weiss_rl/tests/test_config_parsing_utils.py`.
- Updated `REFACTOR_PLAN.md`, `CHANGELOG.md`, `docs/architecture.md`, and `docs/refactor_completion_audit.md`.

### Behavior Changes

No intended behavior changes. The old private helper names remain importable from `parse.py`, error messages are preserved, preset `extends` behavior still skips overlay-side `extends` while preserving the existing deep-merge contract, and config loading/override tests pass unchanged.

### Remaining Risks

- `config/parse.py` is still a large section-parser module; the lower-level document helpers are now isolated, but field-specific section parsers remain in the same file.
- Further parser movement should pin canonical config hashes and representative preset JSON bytes before moving section parsers.

### Current Size Snapshot

- `python/weiss_rl/config/parse.py`: 1756 lines after the config parsing utility extraction.
- `python/weiss_rl/config/parsing_utils.py`: 143 lines.
- Largest remaining production files: `runtime.py`, `model.py`, `learners/impala_learner.py`, `python/scripts/train.py`, and `config/parse.py`.

## 2026-05-11 - Config Core Section Parser Extraction

### Scope

- Moved experiment-role parsing and system section parsing into `weiss_rl.config.sections_core`.
- Preserved `_parse_experiment_config()`, `_parse_system_config()`, and `_EXPERIMENT_ROLES` through `weiss_rl.config.parse`.
- Added direct tests for accepted experiment roles, sorted role validation, system `collection_backend` defaulting, explicit collection backend selection, nested profile unknown-key rejection, and existing minimum-value validation messages.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_config_sections_core.py python/weiss_rl/tests/test_config_parsing_utils.py python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_config_overrides.py` | Passed: 51 passed. |
| `uv run ruff check python/weiss_rl/config/sections_core.py python/weiss_rl/config/parse.py python/weiss_rl/tests/test_config_sections_core.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/config/sections_core.py python/weiss_rl/config/parse.py python/weiss_rl/tests/test_config_sections_core.py` | Initially reported `parse.py` needed formatting; passed after formatting touched files. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 933 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/config/sections_core.py`.
- Updated `python/weiss_rl/config/parse.py` to delegate experiment and system section parsing through wrappers.
- Added `python/weiss_rl/tests/test_config_sections_core.py`.
- Updated `REFACTOR_PLAN.md`, `CHANGELOG.md`, `docs/architecture.md`, and `docs/refactor_completion_audit.md`.

### Behavior Changes

No intended behavior changes. Experiment role choices, system unknown-key checks, `collection_backend` defaulting, and integer minimum checks remain available through the original `parse.py` private names.

### Remaining Risks

- Most field-specific section parsers still live in `config/parse.py`.
- Before moving model/training/evaluation sections, add stronger canonical config hash or representative JSON-byte tests for current public presets.

### Current Size Snapshot

- `python/weiss_rl/config/parse.py`: 1682 lines after the core section parser extraction.
- `python/weiss_rl/config/sections_core.py`: 92 lines.
- Largest remaining production files: `runtime.py`, `model.py`, `learners/impala_learner.py`, `python/scripts/train.py`, and `config/parse.py`.

## 2026-05-11 - Public Preset Canonical Hash Characterization

### Scope

- Added golden canonical hash and canonical JSON length coverage for the four public structured acceptance presets before moving larger config parser sections.
- Covered `structured_acceptance_standard.yaml`, `structured_acceptance_standard_auto_gpu.yaml`, `structured_acceptance_standard_thesis_eval.yaml`, and `structured_acceptance_standard_multideck.yaml`.

### Commands and Results

| Command | Result |
| --- | --- |
| Inline `uv run python -` hash probe for the four public structured acceptance presets | Captured current canonical hashes and canonical JSON lengths. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_config_loader.py -k "canonical_hash or structured_acceptance_public_preset"` | Passed: 4 passed, 34 deselected. |
| `uv run ruff check python/weiss_rl/tests/test_config_loader.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/tests/test_config_loader.py` | Initially reported formatting needed; passed after formatting the touched file. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 937 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Extended `python/weiss_rl/tests/test_config_loader.py` with pinned hashes and JSON lengths for public structured acceptance presets.
- Updated `docs/refactor_log.md`, `REFACTOR_PLAN.md`, and `docs/refactor_completion_audit.md`.

### Behavior Changes

No code behavior changes. This checkpoint adds characterization coverage only.

### Remaining Risks

- The golden hashes intentionally make behavior-preserving parser changes noisy if canonical config output changes; any future update must explain whether the difference is a confirmed bug fix or an unintended semantic change.

## 2026-05-11 - Config Model Section Parser Extraction

### Scope

- Moved model section parsing and model choice constants into `weiss_rl.config.sections_model`.
- Preserved `_parse_model_config()`, `_MODEL_ENCODER_KINDS`, `_STRUCTURED_POLICY_CONTRACTS`, and `_MODEL_RECURRENT_CORES` through `weiss_rl.config.parse`.
- Added direct tests for existing defaults, structured overrides, final-scale fallback, sorted choice errors, public-heuristic schedule validation, nonnegative final scale validation, unknown-key errors, nested dropout validation, and minimum-value validation.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_config_sections_model.py python/weiss_rl/tests/test_config_sections_core.py python/weiss_rl/tests/test_config_loader.py -k "sections_model or sections_core or structured_acceptance_public_preset or canonical_hash"` | Passed: 14 passed, 34 deselected. |
| `uv run ruff check python/weiss_rl/config/sections_model.py python/weiss_rl/config/parse.py python/weiss_rl/tests/test_config_sections_model.py` | Initially reported import ordering in `parse.py`; passed after `ruff check --fix`. |
| `uv run ruff format --check python/weiss_rl/config/sections_model.py python/weiss_rl/config/parse.py python/weiss_rl/tests/test_config_sections_model.py` | Initially reported `parse.py` needed formatting; passed after formatting touched files. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 942 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/config/sections_model.py`.
- Updated `python/weiss_rl/config/parse.py` to delegate model section parsing through a wrapper.
- Added `python/weiss_rl/tests/test_config_sections_model.py`.
- Updated `REFACTOR_PLAN.md`, `CHANGELOG.md`, `docs/architecture.md`, and `docs/refactor_completion_audit.md`.

### Behavior Changes

No intended behavior changes. Model defaults, choice sets, public-heuristic bias schedule checks, dropout parsing, and chunk-size validation remain available through the original `parse.py` private names.

### Remaining Risks

- `training`, `environment`, `rewards`, `curriculum`, `league`, `evaluation`, and `reproducibility` section parsers still live in `config/parse.py`.
- Larger sections are more behavior-sensitive and should keep public-preset hash tests in the focused validation set.

### Current Size Snapshot

- `python/weiss_rl/config/parse.py`: 1579 lines after the model section parser extraction.
- `python/weiss_rl/config/sections_model.py`: 132 lines.

## 2026-05-11 - Config Environment And Reward Section Extraction

### Scope

- Moved environment and reward section parsing into `weiss_rl.config.sections_environment`.
- Preserved `_parse_environment_config()` and `_parse_rewards_config()` through `weiss_rl.config.parse`.
- Added direct tests for environment deck-pool defaults, deck-pool normalization, environment unknown-key and nested deck-size validation, reward shaping defaults, optional shaping values, reward unknown-key validation, nested discount validation, and truncation boolean validation.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_config_sections_environment.py python/weiss_rl/tests/test_config_sections_model.py python/weiss_rl/tests/test_config_loader.py -k "sections_environment or sections_model or structured_acceptance_public_preset or canonical_hash"` | Passed: 15 passed, 34 deselected. |
| `uv run ruff check python/weiss_rl/config/sections_environment.py python/weiss_rl/config/parse.py python/weiss_rl/tests/test_config_sections_environment.py` | Initially reported import ordering in `sections_environment.py`; passed after `ruff check --fix`. |
| `uv run ruff format --check python/weiss_rl/config/sections_environment.py python/weiss_rl/config/parse.py python/weiss_rl/tests/test_config_sections_environment.py` | Initially reported `parse.py` and `sections_environment.py` needed formatting; passed after formatting touched files. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 948 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/config/sections_environment.py`.
- Updated `python/weiss_rl/config/parse.py` to delegate environment and reward section parsing through wrappers.
- Added `python/weiss_rl/tests/test_config_sections_environment.py`.
- Updated `REFACTOR_PLAN.md`, `CHANGELOG.md`, `docs/architecture.md`, and `docs/refactor_completion_audit.md`.

### Behavior Changes

No intended behavior changes. Environment visibility, truncation, max-decision, deck-pool, reward shaping, discount, and truncation reward parsing remain available through the original `parse.py` private names.

### Remaining Risks

- `training`, `curriculum`, `league`, `evaluation`, and `reproducibility` section parsers still live in `config/parse.py`.
- Training/league/evaluation sections are larger behavior-sensitive surfaces and should get focused tests before movement.

### Current Size Snapshot

- `python/weiss_rl/config/parse.py`: 1476 lines after the environment/reward section parser extraction.
- `python/weiss_rl/config/sections_environment.py`: 129 lines.

## 2026-05-11 - Config Curriculum Section Extraction

### Scope

- Moved curriculum section parsing and recursive simulator-payload normalization into `weiss_rl.config.sections_curriculum`.
- Preserved `_normalize_curriculum_payload()` and `_parse_curriculum_config()` through `weiss_rl.config.parse`.
- Added direct tests for supported nested simulator payloads, bad simulator payload keys/types, absent curriculum defaults, populated stall-monitor/checkpoint-guard parsing, unknown-key validation, and minimum-value validation.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_config_sections_curriculum.py python/weiss_rl/tests/test_config_sections_environment.py python/weiss_rl/tests/test_config_loader.py -k "sections_curriculum or sections_environment or structured_acceptance_public_preset or canonical_hash"` | Passed: 15 passed, 34 deselected. |
| `uv run ruff check python/weiss_rl/config/sections_curriculum.py python/weiss_rl/config/parse.py python/weiss_rl/tests/test_config_sections_curriculum.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/config/sections_curriculum.py python/weiss_rl/config/parse.py python/weiss_rl/tests/test_config_sections_curriculum.py` | Initially reported `parse.py` needed formatting; passed after formatting touched files. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 953 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/config/sections_curriculum.py`.
- Updated `python/weiss_rl/config/parse.py` to delegate curriculum normalization and parsing through wrappers.
- Added `python/weiss_rl/tests/test_config_sections_curriculum.py`.
- Updated `REFACTOR_PLAN.md`, `CHANGELOG.md`, `docs/architecture.md`, and `docs/refactor_completion_audit.md`.

### Behavior Changes

No intended behavior changes. Curriculum absence defaults, simulator payload normalization, stall-monitor defaults, checkpoint-guard defaults, and validation messages remain available through the original `parse.py` private names.

### Remaining Risks

- `training`, `league`, `evaluation`, and `reproducibility` section parsers still live in `config/parse.py`.
- The remaining parser sections are behavior-sensitive enough to move one at a time with broader focused tests.

### Current Size Snapshot

- `python/weiss_rl/config/parse.py`: 1386 lines after the curriculum section parser extraction.
- `python/weiss_rl/config/sections_curriculum.py`: 106 lines.

## 2026-05-11 - Config Reproducibility Section Extraction

### Scope

- Moved reproducibility section parsing into `weiss_rl.config.sections_reproducibility`.
- Preserved `_parse_reproducibility_config()` through `weiss_rl.config.parse`.
- Added direct tests for accepted fail-fast reproducibility config, fail-fast spec mismatch rejection, replay-eval mismatch policy rejection, unknown-key validation, boolean validation, seed minimum validation, and determinism requirement string-list validation.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_config_sections_reproducibility.py python/weiss_rl/tests/test_config.py python/weiss_rl/tests/test_config_loader.py -k "sections_reproducibility or fail_fast or structured_acceptance_public_preset or canonical_hash"` | Passed: 12 passed, 35 deselected. |
| `uv run ruff check python/weiss_rl/config/sections_reproducibility.py python/weiss_rl/config/parse.py python/weiss_rl/tests/test_config_sections_reproducibility.py` | Initially reported import ordering in `sections_reproducibility.py`; passed after `ruff check --fix`. |
| `uv run ruff format --check python/weiss_rl/config/sections_reproducibility.py python/weiss_rl/config/parse.py python/weiss_rl/tests/test_config_sections_reproducibility.py` | Initially reported formatting needed; passed after formatting touched files. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 957 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/config/sections_reproducibility.py`.
- Updated `python/weiss_rl/config/parse.py` to delegate reproducibility section parsing through a wrapper.
- Added `python/weiss_rl/tests/test_config_sections_reproducibility.py`.
- Updated `REFACTOR_PLAN.md`, `CHANGELOG.md`, `docs/architecture.md`, and `docs/refactor_completion_audit.md`.

### Behavior Changes

No intended behavior changes. Reproducibility spec-bundle policy parsing, ID fields, seed derivation, seed-file mapping, determinism requirements, legal-fingerprint fields, and hard-fail spec mismatch enforcement remain available through the original `parse.py` private name.

### Remaining Risks

- `training`, `league`, and `evaluation` section parsers still live in `config/parse.py`.
- Those remaining sections are large and behavior-sensitive; move only one per checkpoint with direct nested-default tests and public-preset hash coverage.

### Current Size Snapshot

- `python/weiss_rl/config/parse.py`: 1266 lines after the reproducibility section parser extraction.
- `python/weiss_rl/config/sections_reproducibility.py`: 142 lines.

## 2026-05-11 - Config Seed-Set Resolution Extraction

### Scope

- Moved seed-set path resolution and canonical run-artifact seed override parsing into `weiss_rl.config.seed_sets`.
- Preserved `_resolve_seed_sets()` and `_parse_seed_sets_override()` through `weiss_rl.config.parse`.
- Added direct tests for evaluation-over-league-over-reproducibility precedence, league promotion fallback when evaluation omits `promotion_gate`, blank league promotion seed fallback to reproducibility, override path resolution, absolute paths, and override key/value validation.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_config_seed_sets.py python/weiss_rl/tests/test_config_loader.py -k "config_seed_sets or reads_canonical_run_artifact_json or applies_extends or structured_acceptance_public_preset"` | Passed: 10 passed, 32 deselected. |
| `uv run ruff check python/weiss_rl/config/seed_sets.py python/weiss_rl/config/parse.py python/weiss_rl/tests/test_config_seed_sets.py` | Initially reported import ordering in `parse.py`; passed after `ruff check --fix`. |
| `uv run ruff format --check python/weiss_rl/config/seed_sets.py python/weiss_rl/config/parse.py python/weiss_rl/tests/test_config_seed_sets.py` | Initially reported `parse.py` needed formatting; passed after formatting touched files. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 961 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/config/seed_sets.py`.
- Updated `python/weiss_rl/config/parse.py` to delegate seed-set resolution and override parsing through wrappers.
- Added `python/weiss_rl/tests/test_config_seed_sets.py`.
- Updated `REFACTOR_PLAN.md`, `CHANGELOG.md`, `docs/architecture.md`, and `docs/refactor_completion_audit.md`.

### Behavior Changes

No intended behavior changes. Evaluation seed files still win first, league promotion seeds fill `promotion_gate` only if absent, reproducibility seed files fill missing names only, and canonical JSON seed-set overrides still replace resolved seed sets.

### Remaining Risks

- `training`, `league`, and `evaluation` section parsers still live in `config/parse.py`.
- Any future movement around those sections should continue carrying public-preset hash and config loader tests in the focused validation set.

### Current Size Snapshot

- `python/weiss_rl/config/parse.py`: 1252 lines after the seed-set helper extraction.
- `python/weiss_rl/config/seed_sets.py`: 39 lines.

## 2026-05-11 - Config League Section Extraction

### Scope

- Moved league pool, sampling, warmup, promotion, anchor-set, gate, and guardrail parsing into `weiss_rl.config.sections_league`.
- Preserved `_parse_league_config()` through `weiss_rl.config.parse`.
- Added direct tests for existing sampling defaults, mix-fraction final defaults, optional sampling values, nested unknown-key validation, `pfsp_stats_source` guard behavior, and minimum-value validation.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_config_sections_league.py python/weiss_rl/tests/test_config_seed_sets.py python/weiss_rl/tests/test_config_loader.py -k "sections_league or config_seed_sets or structured_acceptance_public_preset or canonical_hash"` | Passed: 12 passed, 34 deselected. |
| `uv run ruff check python/weiss_rl/config/sections_league.py python/weiss_rl/config/parse.py python/weiss_rl/tests/test_config_sections_league.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/config/sections_league.py python/weiss_rl/config/parse.py python/weiss_rl/tests/test_config_sections_league.py` | Initially reported `parse.py` and `sections_league.py` needed formatting; passed after formatting touched files. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 965 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/config/sections_league.py`.
- Updated `python/weiss_rl/config/parse.py` to delegate league section parsing through a wrapper.
- Added `python/weiss_rl/tests/test_config_sections_league.py`.
- Updated `REFACTOR_PLAN.md`, `CHANGELOG.md`, `docs/architecture.md`, and `docs/refactor_completion_audit.md`.

### Behavior Changes

No intended behavior changes. League pool defaults, PFSP sampling fields, heuristic-public mix defaults, warmup fields, promotion gate fields, anchor-set parsing, and the `online_outcomes`-only PFSP stats-source guard remain available through the original `parse.py` private name.

### Remaining Risks

- `training` and `evaluation` section parsers still live in `config/parse.py`.
- Evaluation parsing is behavior-sensitive because it guards seed files, policy selection, legal-fingerprint checks, and hard-fail mismatch behavior.
- Training parsing is the largest remaining parser surface and should get a dedicated focused suite before movement.

### Current Size Snapshot

- `python/weiss_rl/config/parse.py`: 1017 lines after the league section parser extraction.
- `python/weiss_rl/config/sections_league.py`: 256 lines.

## 2026-05-11 - Config Evaluation Section Extraction

### Scope

- Moved evaluation section parsing into `weiss_rl.config.sections_evaluation`.
- Preserved `_parse_evaluation_config()` through `weiss_rl.config.parse`.
- Added direct tests for seed files, stop rules, legal fingerprint hard-fail guard, decision-kind tagging, final policy-set selection, fixed anchor sets, unknown-key validation, minimum-value validation, and integer-list validation.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_config_sections_evaluation.py python/weiss_rl/tests/test_config.py python/weiss_rl/tests/test_config_loader.py -k "sections_evaluation or fail_fast or structured_acceptance_public_preset or canonical_hash"` | Passed: 12 passed, 35 deselected. |
| `uv run ruff check python/weiss_rl/config/sections_evaluation.py python/weiss_rl/config/parse.py python/weiss_rl/tests/test_config_sections_evaluation.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/config/sections_evaluation.py python/weiss_rl/config/parse.py python/weiss_rl/tests/test_config_sections_evaluation.py` | Initially reported `parse.py` needed formatting; passed after formatting touched files. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 969 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/config/sections_evaluation.py`.
- Updated `python/weiss_rl/config/parse.py` to delegate evaluation section parsing through a wrapper.
- Added `python/weiss_rl/tests/test_config_sections_evaluation.py`.
- Updated `REFACTOR_PLAN.md`, `CHANGELOG.md`, `docs/architecture.md`, and `docs/refactor_completion_audit.md`.

### Behavior Changes

No intended behavior changes. Evaluation seed files, stop rules, legal-fingerprint checks, hard-fail mismatch policy, decision-kind tagging, final policy-set selection, fixed anchors, folding, seat-swap, and tie-break parsing remain available through the original `parse.py` private name.

### Remaining Risks

- `training` is the only section parser still living in `config/parse.py`.
- Training parsing is the largest parser surface and includes algorithm/backend choices, actor heuristic schedules, structured auxiliary defaults, warmstart settings, PPO defaults, profiling flags, and precision flags; it should get a dedicated direct test module before movement.

### Current Size Snapshot

- `python/weiss_rl/config/parse.py`: 812 lines after the evaluation section parser extraction.
- `python/weiss_rl/config/sections_evaluation.py`: 229 lines.

## 2026-05-11 - Config Training Section Extraction

### Scope

- Moved training algorithm, PPO, profiling, backend, public-heuristic, diversity, structured-metrics, teacher-auxiliary, warmstart, and precision parsing into `weiss_rl.config.sections_training`.
- Preserved `_parse_training_config()` and private training choice constants through `weiss_rl.config.parse`.
- Added direct tests for existing defaults, nested overrides, public-heuristic option normalization, public-heuristic final-coefficient fallback, sorted choice errors, range validation, nested unknown-key validation, and minimum-value validation.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_config_sections_training.py python/weiss_rl/tests/test_config_loader.py -k "sections_training or structured_acceptance_public_preset or canonical_hash or supports_structured"` | Passed: 14 passed, 31 deselected. |
| `uv run ruff check --fix python/weiss_rl/config/parse.py python/weiss_rl/config/sections_training.py python/weiss_rl/tests/test_config_sections_training.py` | Passed; no additional edits were needed. |
| `uv run ruff format python/weiss_rl/config/parse.py python/weiss_rl/config/sections_training.py python/weiss_rl/tests/test_config_sections_training.py` | Passed; 3 files left unchanged. |
| `uv run ruff check python/weiss_rl/config/sections_training.py python/weiss_rl/config/parse.py python/weiss_rl/tests/test_config_sections_training.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/config/sections_training.py python/weiss_rl/config/parse.py python/weiss_rl/tests/test_config_sections_training.py` | Passed; 3 files already formatted. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 976 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/config/sections_training.py`.
- Updated `python/weiss_rl/config/parse.py` to delegate training section parsing through a wrapper.
- Added `python/weiss_rl/tests/test_config_sections_training.py`.
- Updated `REFACTOR_PLAN.md`, `CHANGELOG.md`, `docs/architecture.md`, and `docs/refactor_completion_audit.md`.

### Behavior Changes

No intended behavior changes. Training defaults, algorithm/backend choice validation, PPO parameters, actor heuristic schedules, public-heuristic structured options, diversity settings, structured metrics, teacher auxiliary settings, warmstart flags, and precision/profiling flags remain available through the original `parse.py` private names.

### Remaining Risks

- Config parser decomposition is now complete enough for the current architecture pass, and the full repository verifier passed after the extraction.
- The remaining largest behavior-sensitive files are runtime, model, learner, and training orchestration surfaces rather than config parsing.

### Current Size Snapshot

- `python/weiss_rl/config/parse.py`: 248 lines after the training section parser extraction.
- `python/weiss_rl/config/sections_training.py`: 519 lines.

## 2026-05-11 - Runtime Logging Helper Extraction

### Scope

- Moved runtime performance JSONL logging and process collector debug-file logging into `weiss_rl.runtime_logging`.
- Preserved `PerformanceLogger` imports and `_process_debug_log()` behavior through `weiss_rl.runtime`.
- Added direct tests for sorted JSONL output, parent-directory creation, environment-gated process debug logging, actor-id file formatting, timestamp payload shape, and `None` run-dir handling.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime_logging.py` | Passed: 3 passed. |
| `uv run ruff check python/weiss_rl/runtime_logging.py python/weiss_rl/runtime.py python/weiss_rl/tests/test_runtime_logging.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/runtime_logging.py python/weiss_rl/runtime.py python/weiss_rl/tests/test_runtime_logging.py` | Initially reported `runtime.py` needed formatting; passed after formatting touched files. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 979 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/runtime_logging.py`.
- Updated `python/weiss_rl/runtime.py` to import `PerformanceLogger` and delegate `_process_debug_log()`.
- Added `python/weiss_rl/tests/test_runtime_logging.py`.
- Updated `REFACTOR_PLAN.md`, `CHANGELOG.md`, `docs/architecture.md`, and `docs/refactor_completion_audit.md`.

### Behavior Changes

No intended behavior changes. Performance records are still sorted JSONL, process debug logs are still controlled by `WEISS_RL_PROCESS_DEBUG`, and collector debug paths still use `training/logs/collector_debug_actorXX.log`.

### Remaining Risks

- This is a small support-helper extraction only; it does not reduce the central collector loops or policy-row application code.
- The full repository verifier passed after this checkpoint.

### Current Size Snapshot

- `python/weiss_rl/runtime.py`: 6077 lines after the runtime logging helper extraction.
- `python/weiss_rl/runtime_logging.py`: 26 lines.

## 2026-05-11 - Runtime Collector Command Handler Extraction

### Scope

- Moved process-collector command handling into `weiss_rl.runtime_collector_commands`.
- Preserved `_handle_collector_commands()` through `weiss_rl.runtime`.
- Added direct tests for update/reload/refresh/stop command sequencing and fixed-opponent apply/restore behavior.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime_collector_commands.py python/weiss_rl/tests/test_runtime.py -k "handle_collector_commands"` | Passed: 3 passed, 87 deselected. |
| `uv run ruff check python/weiss_rl/runtime_collector_commands.py python/weiss_rl/runtime.py python/weiss_rl/tests/test_runtime_collector_commands.py` | Initially reported the still-needed `queue` import in `runtime.py`; passed after restoring it. |
| `uv run ruff format --check python/weiss_rl/runtime_collector_commands.py python/weiss_rl/runtime.py python/weiss_rl/tests/test_runtime_collector_commands.py` | Initially reported `runtime.py` and `runtime_collector_commands.py` needed formatting; passed after formatting touched files. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 981 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/runtime_collector_commands.py`.
- Updated `python/weiss_rl/runtime.py` to delegate `_handle_collector_commands()`.
- Added `python/weiss_rl/tests/test_runtime_collector_commands.py`.
- Updated `REFACTOR_PLAN.md`, `CHANGELOG.md`, `docs/architecture.md`, and `docs/refactor_completion_audit.md`.

### Behavior Changes

No intended behavior changes. Reload, set-update, refresh, stop, fixed-opponent, teacher-heuristic activation, no-league baseline override, and default restore commands still flow through the original `runtime.py` private wrapper.

### Remaining Risks

- The command handler remains coupled to runtime private fields by design, because it is preserving the existing process-collector protocol.
- The full repository verifier passed after this checkpoint.

### Current Size Snapshot

- `python/weiss_rl/runtime.py`: 5978 lines after the collector command handler extraction.
- `python/weiss_rl/runtime_collector_commands.py`: 133 lines.

## 2026-05-11 - Simulator-Backed Deterministic Eval Smoke

### Scope

- Probed the canonical simulator-backed train/eval path using the sibling `../weiss-schwarz-simulator/python` checkout.
- Kept all writes in new refactor smoke run directories under `runs/`; no historical result directories were intentionally modified.
- Verified repeated canonical eval determinism on a tiny no-league explicit-policy run by comparing regenerated artifacts.

### Commands and Results

| Command | Result |
| --- | --- |
| `PYTHONPATH=../weiss-schwarz-simulator/python;python uv run python python/scripts/train.py --stack-config configs/presets/structured_acceptance_standard.yaml --run-label refactor_sim_determinism_20260511 --device cpu --num-envs 2 --unroll-length 4 --max-updates 1 --runtime-mode train_ordered` | Failed after manifest creation: standard stack requires a completed B1 no-league baseline and `--b1-baseline-run-dir`. |
| `PYTHONPATH=../weiss-schwarz-simulator/python;python uv run python python/scripts/train.py --stack-config configs/presets/baselines/structured_acceptance_tiny32_fast_noleague.yaml --run-label refactor_sim_determinism_noleague_20260511 --device cpu --num-envs 2 --unroll-length 4 --max-updates 1 --runtime-mode train_ordered` | Failed after manifest creation: preset requested process collection, which was unsupported for the local runtime setup. |
| `PYTHONPATH=../weiss-schwarz-simulator/python;python uv run python python/scripts/train.py --stack-config configs/presets/baselines/structured_acceptance_tiny32_fast_noleague.yaml --override system.collection_backend=central --run-label refactor_sim_determinism_noleague_central_20260511 --device cpu --num-envs 2 --unroll-length 4 --max-updates 1 --runtime-mode train_ordered` | Failed after manifest creation: central collection requires a supported `train_async_fast` setup. |
| `PYTHONPATH=../weiss-schwarz-simulator/python;python uv run python python/scripts/train.py --stack-config configs/presets/baselines/structured_acceptance_tiny32_fast_noleague.yaml --override system.collection_backend=auto --run-label refactor_sim_determinism_noleague_async_20260511 --device cpu --num-envs 2 --unroll-length 4 --max-updates 1 --runtime-mode train_async_fast` | Passed: completed canonical single-node simulator-backed training run with one update and persisted `b1_noleague_baseline`. |
| `PYTHONPATH=../weiss-schwarz-simulator/python;python uv run python python/scripts/eval.py --stack-config configs/presets/baselines/structured_acceptance_tiny32_fast_noleague.yaml --run-dir runs/refactor_sim_determinism_noleague_async_20260511 --policy-id b1_noleague_baseline --snapshot-registry-json runs/refactor_sim_determinism_noleague_async_20260511/training/snapshots/registry.json --paired-seed-limit 1 --stage1-paired-seeds 1 --max-paired-seeds 1 --skip-metagame --skip-figures --skip-readiness --bootstrap-samples 16` | Passed twice. |
| SHA-256 comparison of first vs second eval outputs | Matched exactly: `summary.json` = `94849f99792022c4ebf9c07482c1034b730dea586093defc94e9373b9b9d1ea7`; `episodes.jsonl` = `320466f93582e3b3290e07020678d35692a211644458515eb54ba53a549ff842`; `replay_verification.json` = `23b1f6561d5d269a319eb859d79aacb27e494603a4542eb42f95096cffe06c4d`. |

### Behavior Changes

No code changes were made in this checkpoint. This was validation-only evidence for the canonical simulator-backed path.

### Remaining Risks

- The repeated eval comparison covers a tiny explicit-policy no-league run, not the full thesis final policy-set selection with all anchors.
- The failed standard-stack probe confirms the B1 baseline prerequisite is enforced; it was not bypassed.

## 2026-05-11 - Model MLP Layer Builder Extraction

### Scope

- Moved the shared MLP-stack builder into `weiss_rl.model_layers`.
- Preserved `weiss_rl.model._build_mlp_stack` as an alias to the extracted helper.
- Added direct tests for layer ordering, output shape, validation errors, and private wrapper preservation.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_model_layers.py python/weiss_rl/tests/test_model.py -k "mlp_stack or build_policy_value_model"` | Failed immediately because `python/weiss_rl/tests/test_model.py` does not exist. No code behavior was exercised by this mistaken command. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_model_layers.py python/weiss_rl/tests/test_model_tensor_ops.py python/weiss_rl/tests/test_model_observation_contract.py` | Passed: 19 passed. |
| `uv run ruff check python/weiss_rl/model_layers.py python/weiss_rl/model.py python/weiss_rl/tests/test_model_layers.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/model_layers.py python/weiss_rl/model.py python/weiss_rl/tests/test_model_layers.py` | Initially reported `model.py` needed formatting; passed after formatting touched files. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 987 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/model_layers.py`.
- Updated `python/weiss_rl/model.py` to import the builder as `_build_mlp_stack`.
- Added `python/weiss_rl/tests/test_model_layers.py`.
- Updated `REFACTOR_PLAN.md`, `CHANGELOG.md`, `docs/architecture.md`, and `docs/refactor_completion_audit.md`.

### Behavior Changes

No intended behavior changes. Existing model construction still calls `_build_mlp_stack`; the helper now lives in a focused module while model classes remain in `model.py`, avoiding class-path changes for checkpoint compatibility.

### Remaining Risks

- This only extracts a small shared layer builder; the structured legal-action head and public-heuristic scoring logic remain in `model.py`.
- The full repository verifier passed after this checkpoint.

### Current Size Snapshot

- `python/weiss_rl/model.py`: 4661 lines after the model layer helper extraction.
- `python/weiss_rl/model_layers.py`: 30 lines.

## 2026-05-11 - Model Typed Encoder Extraction

### Scope

- Moved typed observation encoder modules and segment helper functions into `weiss_rl.model_typed_encoder`.
- Preserved `weiss_rl.model` private aliases for `_TypedObservationEncoder`, `_TypedPlayerBlockEncoder`, `_TypedSegmentEncoder`, `_block_segments()`, and `_flatten_indices()`.
- Added direct tests for output shape, state-dict key names, error behavior, and alias preservation.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_model_typed_encoder.py python/weiss_rl/tests/test_model_layers.py python/weiss_rl/tests/test_model_observation_contract.py` | Passed: 14 passed. |
| `uv run ruff check python/weiss_rl/model_typed_encoder.py python/weiss_rl/model.py python/weiss_rl/tests/test_model_typed_encoder.py` | Initially reported import sorting and unused direct alias imports; passed after switching to explicit compatibility assignments and running `ruff check --fix`. |
| `uv run ruff format --check python/weiss_rl/model_typed_encoder.py python/weiss_rl/model.py python/weiss_rl/tests/test_model_typed_encoder.py` | Passed after formatting touched files. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 991 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/model_typed_encoder.py`.
- Updated `python/weiss_rl/model.py` to delegate typed encoder aliases and helpers.
- Added `python/weiss_rl/tests/test_model_typed_encoder.py`.
- Updated `REFACTOR_PLAN.md`, `CHANGELOG.md`, `docs/architecture.md`, and `docs/refactor_completion_audit.md`.

### Behavior Changes

No intended behavior changes. The typed encoder state-dict key names remain rooted at the same model attributes, and the private names in `model.py` still resolve for compatibility.

### Remaining Risks

- The typed encoder class `__module__` is now `weiss_rl.model_typed_encoder`; normal checkpoints store state dicts, but whole-object Python pickles of private model internals would not be a supported compatibility surface.
- The full repository verifier passed after this checkpoint.

### Current Size Snapshot

- `python/weiss_rl/model.py`: 4541 lines after the typed encoder extraction.
- `python/weiss_rl/model_typed_encoder.py`: 135 lines.

## 2026-05-11 - Model Sampling Helper Extraction

### Scope

- Moved deterministic masked and packed action sampling helpers into `weiss_rl.model_sampling`.
- Preserved `weiss_rl.model` private wrappers for `_sample_masked_log_probs()` and `_sample_packed_action_scores()`.
- Added direct tests for seeded outputs, selected log-probabilities, shape validation, and wrapper delegation.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_model_sampling.py python/weiss_rl/tests/test_model_tensor_ops.py` | Passed: 15 passed. |
| `uv run ruff check python/weiss_rl/model_sampling.py python/weiss_rl/model.py python/weiss_rl/tests/test_model_sampling.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/model_sampling.py python/weiss_rl/model.py python/weiss_rl/tests/test_model_sampling.py` | Initially reported `model.py` needed formatting; passed after formatting touched files. |
| `uv run python python/scripts/verify_repo.py` | Failed: 996 passed, 1 failed, 2 skipped. Existing `test_sample_packed_action_scores_falls_back_to_last_candidate_when_cdf_undershoots` monkeypatches `weiss_rl.model` private sampling dependencies; the first wrapper bypassed that compatibility surface. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_model_sampling.py python/weiss_rl/tests/test_runtime.py -k "sample_packed_action_scores or model_sampling"` | Passed after updating the wrapper to pass private dependency aliases into the extracted helper: 7 passed, 87 deselected. |
| `uv run ruff check python/weiss_rl/model_sampling.py python/weiss_rl/model.py python/weiss_rl/tests/test_model_sampling.py` | Passed after the compatibility fix. |
| `uv run ruff format --check python/weiss_rl/model_sampling.py python/weiss_rl/model.py python/weiss_rl/tests/test_model_sampling.py` | Passed after formatting touched `model.py`. |
| `uv run python python/scripts/verify_repo.py` | Passed after the compatibility fix: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 997 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/model_sampling.py`.
- Updated `python/weiss_rl/model.py` to delegate deterministic sampling wrappers.
- Added `python/weiss_rl/tests/test_model_sampling.py`.
- Updated `REFACTOR_PLAN.md`, `CHANGELOG.md`, `docs/architecture.md`, and `docs/refactor_completion_audit.md`.

### Behavior Changes

No intended behavior changes. Seeded masked and packed action sampling still flows through the original private wrapper names in `model.py`; the extracted helper tests pin representative action/log-probability outputs. The wrapper also preserves the existing private monkeypatch surface for `_uniform_from_seeds()` and packed CDF helpers.

### Remaining Risks

- This is action-selection-adjacent code, so future edits to the helper should continue to use direct seeded characterization tests.
- The full repository verifier passed after the compatibility fix.

### Current Size Snapshot

- `python/weiss_rl/model.py`: 4468 lines after the model sampling helper extraction.
- `python/weiss_rl/model_sampling.py`: 115 lines.

## 2026-05-11 - Model Action Plan Container Extraction

### Scope

- Moved structured/factorized action scoring container dataclasses into `weiss_rl.model_action_plans`.
- Preserved `weiss_rl.model` private aliases for `_PackedScoringPlan`, `_FactorizedEvaluationResult`, `_FactorizedFamilyPlan`, `_FactorizedConditionalLogProbs`, and `_FactorizedLegalityPlan`.
- Added direct tests for packed plan slicing, factorized container payload preservation, and alias preservation.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_model_action_plans.py python/weiss_rl/tests/test_model_sampling.py` | Passed: 9 passed. |
| `uv run ruff check python/weiss_rl/model_action_plans.py python/weiss_rl/model.py python/weiss_rl/tests/test_model_action_plans.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/model_action_plans.py python/weiss_rl/model.py python/weiss_rl/tests/test_model_action_plans.py` | Initially reported `model.py` needed formatting; passed after formatting touched files. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1000 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/model_action_plans.py`.
- Updated `python/weiss_rl/model.py` to use private aliases for extracted action-plan containers.
- Added `python/weiss_rl/tests/test_model_action_plans.py`.
- Updated `REFACTOR_PLAN.md`, `CHANGELOG.md`, `docs/architecture.md`, and `docs/refactor_completion_audit.md`.

### Behavior Changes

No intended behavior changes. These dataclasses are internal containers for existing model scoring flow; the private names in `model.py` still resolve for compatibility.

### Remaining Risks

- The full repository verifier passed after this checkpoint.
- The model file remains large because the structured legal-action head and public-heuristic scoring methods are still in one class.

### Current Size Snapshot

- `python/weiss_rl/model.py`: 4440 lines after the action-plan container extraction.
- `python/weiss_rl/model_action_plans.py`: 49 lines.

## 2026-05-11 - Learner Torch V-Trace Extraction

### Scope

- Moved the torch V-trace target computation into `weiss_rl.learners.vtrace_torch`.
- Preserved `_compute_vtrace_targets_torch()` through `weiss_rl.learners.impala_learner`.
- Added direct tests comparing torch outputs to the NumPy V-trace reference, checking extreme-rho capping, and verifying wrapper delegation.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_learner_vtrace_torch.py python/weiss_rl/tests/test_vtrace.py -k "vtrace_targets or vtrace_torch"` | Passed: 8 passed, 12 deselected. |
| `uv run ruff check python/weiss_rl/learners/vtrace_torch.py python/weiss_rl/learners/impala_learner.py python/weiss_rl/tests/test_learner_vtrace_torch.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/learners/vtrace_torch.py python/weiss_rl/learners/impala_learner.py python/weiss_rl/tests/test_learner_vtrace_torch.py` | Initially reported `impala_learner.py` needed formatting; passed after formatting touched files. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1003 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/learners/vtrace_torch.py`.
- Updated `python/weiss_rl/learners/impala_learner.py` to delegate `_compute_vtrace_targets_torch()`.
- Added `python/weiss_rl/tests/test_learner_vtrace_torch.py`.
- Updated `REFACTOR_PLAN.md`, `CHANGELOG.md`, `docs/architecture.md`, and `docs/refactor_completion_audit.md`.

### Behavior Changes

No intended behavior changes. Raw V-trace target computation still flows through the original private wrapper in `impala_learner.py`; the extracted helper is numerically checked against the existing NumPy reference.

### Remaining Risks

- The full repository verifier passed after this checkpoint.
- Other learner log-probability and packed legality helpers remain in `impala_learner.py`.

### Current Size Snapshot

- `python/weiss_rl/learners/impala_learner.py`: 3983 lines after the torch V-trace extraction.
- `python/weiss_rl/learners/vtrace_torch.py`: 39 lines.

## 2026-05-11 - Learner Action Log-Probability Extraction

### Scope

- Moved dense masked, packed dense, packed-score, and packed-subset action log-probability/entropy helpers into `weiss_rl.learners.action_logp`.
- Preserved `_masked_log_probs_and_entropy()`, `_masked_action_logp_and_entropy()`, `_packed_action_logp_and_entropy()`, `_packed_selected_action_logp()`, `_packed_subset_action_logp_and_top_action()`, and `_packed_scores_action_logp_and_entropy()` through `weiss_rl.learners.impala_learner`.
- Added direct tests comparing packed and dense log-probability paths, strict/non-strict unsupported-row behavior, packed-score entropy behavior, packed-subset top-action reporting, and private wrapper delegation.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_learner_action_logp.py python/weiss_rl/tests/test_impala_learner.py -k "learner_action_logp or masked_action_logp or packed_action_logp or packed_selected_action_logp"` | Passed: 5 passed, 42 deselected. |
| `uv run ruff check python/weiss_rl/learners/action_logp.py python/weiss_rl/learners/impala_learner.py python/weiss_rl/tests/test_learner_action_logp.py` | Initially reported import sorting in `impala_learner.py`; passed after `ruff check --fix`. |
| `uv run ruff format --check python/weiss_rl/learners/action_logp.py python/weiss_rl/learners/impala_learner.py python/weiss_rl/tests/test_learner_action_logp.py` | Initially reported `impala_learner.py` needed formatting; passed after formatting touched files. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_impala_learner.py python/weiss_rl/tests/test_ppo_lite_learner.py python/weiss_rl/tests/test_learner_action_logp.py` | Passed: 49 passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1008 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/learners/action_logp.py`.
- Updated `python/weiss_rl/learners/impala_learner.py` to delegate learner action log-probability helpers.
- Added `python/weiss_rl/tests/test_learner_action_logp.py`.
- Updated `REFACTOR_PLAN.md`, `CHANGELOG.md`, `docs/architecture.md`, and `docs/refactor_completion_audit.md`.

### Behavior Changes

No intended behavior changes. The learner still calls the same private helper names from `impala_learner.py`; the extracted module is covered against dense/packed parity and unsupported-action behavior.

### Remaining Risks

- This is action/log-probability-sensitive code, so future changes should keep dense-vs-packed parity tests and strict/non-strict unsupported-row tests close to the helper.
- The full repository verifier passed after this checkpoint.

### Current Size Snapshot

- `python/weiss_rl/learners/impala_learner.py`: 3693 lines after the learner action log-probability extraction.
- `python/weiss_rl/learners/action_logp.py`: 383 lines.

## 2026-05-11 - Learner Structured Policy Metrics Extraction

### Scope

- Moved structured policy metric summary reporting into `weiss_rl.learners.structured_policy_metrics`.
- Preserved `summarize_structured_policy_metrics()` through `weiss_rl.learners.impala_learner`.
- Added direct tests for the new helper, wrapper equivalence, and factorized-family metric reporting.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_learner_structured_policy_metrics.py python/weiss_rl/tests/test_impala_learner.py -k "structured_policy_metrics or summarize_structured_policy_metrics"` | Initially failed because the new direct test hard-coded the wrong `main_move 0->2` action id; passed after deriving the action id from the catalog: 4 passed, 40 deselected. |
| `uv run ruff check python/weiss_rl/learners/structured_policy_metrics.py python/weiss_rl/learners/impala_learner.py python/weiss_rl/tests/test_learner_structured_policy_metrics.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/learners/structured_policy_metrics.py python/weiss_rl/learners/impala_learner.py python/weiss_rl/tests/test_learner_structured_policy_metrics.py` | Initially reported `impala_learner.py` needed formatting; passed after formatting touched files. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_impala_learner.py python/weiss_rl/tests/test_ppo_lite_learner.py python/weiss_rl/tests/test_learner_structured_policy_metrics.py python/weiss_rl/tests/test_learner_action_logp.py` | Passed: 51 passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1010 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/learners/structured_policy_metrics.py`.
- Updated `python/weiss_rl/learners/impala_learner.py` to delegate structured policy metric summaries.
- Added `python/weiss_rl/tests/test_learner_structured_policy_metrics.py`.
- Updated `REFACTOR_PLAN.md`, `CHANGELOG.md`, `docs/architecture.md`, and `docs/refactor_completion_audit.md`.

### Behavior Changes

No intended behavior changes. Learner logging still calls the same public helper name from `impala_learner.py`; the extracted helper is covered against dense structured logits and factorized family log-probability inputs.

### Remaining Risks

- Structured policy metrics are reporting-only, but the action-catalog family mapping is still behavior-sensitive; tests should continue deriving action ids from the catalog instead of hard-coding them.
- The full repository verifier passed after this checkpoint.

### Current Size Snapshot

- `python/weiss_rl/learners/impala_learner.py`: 3564 lines after the structured policy metrics extraction.
- `python/weiss_rl/learners/structured_policy_metrics.py`: 169 lines.

## 2026-05-11 - Learner Logp and V-Trace Diagnostic Facade Extraction

### Scope

- Moved NumPy learner log-probability facade functions into `weiss_rl.learners.action_logp`.
- Moved V-trace diagnostic summary metrics into `weiss_rl.learners.vtrace_diagnostics`.
- Preserved `learner_logp_from_mask()`, `learner_logp_from_legal_ids()`, `summarize_vtrace_diagnostics()`, and `VTRACE_RHO_PERCENTILES` through `weiss_rl.learners.impala_learner`.
- Added direct tests for the extracted facades and wrapper equivalence.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_learner_action_logp.py python/weiss_rl/tests/test_masking.py -k "numpy_learner_logp or masking_core_is_reused"` | Passed: 2 passed, 25 deselected. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_learner_vtrace_diagnostics.py python/weiss_rl/tests/test_vtrace.py -k "vtrace_diagnostics or summarize_vtrace_diagnostics"` | Passed: 2 passed, 16 deselected. |
| `uv run ruff check python/weiss_rl/learners/action_logp.py python/weiss_rl/learners/vtrace_diagnostics.py python/weiss_rl/learners/impala_learner.py python/weiss_rl/tests/test_learner_action_logp.py python/weiss_rl/tests/test_learner_vtrace_diagnostics.py` | Initially reported import ordering and compatibility constant assignment cleanup; passed after `ruff check --fix` and explicit `VTRACE_RHO_PERCENTILES` aliasing. |
| `uv run ruff format --check python/weiss_rl/learners/action_logp.py python/weiss_rl/learners/vtrace_diagnostics.py python/weiss_rl/learners/impala_learner.py python/weiss_rl/tests/test_learner_action_logp.py python/weiss_rl/tests/test_learner_vtrace_diagnostics.py` | Initially reported `impala_learner.py` needed formatting; passed after formatting touched files. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_impala_learner.py python/weiss_rl/tests/test_ppo_lite_learner.py python/weiss_rl/tests/test_learner_action_logp.py python/weiss_rl/tests/test_learner_vtrace_diagnostics.py python/weiss_rl/tests/test_masking.py python/weiss_rl/tests/test_vtrace.py` | Passed: 89 passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1012 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/learners/vtrace_diagnostics.py`.
- Updated `python/weiss_rl/learners/action_logp.py` with NumPy learner logp facades.
- Updated `python/weiss_rl/learners/impala_learner.py` to delegate the compatibility facades.
- Added `python/weiss_rl/tests/test_learner_vtrace_diagnostics.py`.
- Extended `python/weiss_rl/tests/test_learner_action_logp.py`.
- Updated `REFACTOR_PLAN.md`, `CHANGELOG.md`, `docs/architecture.md`, and `docs/refactor_completion_audit.md`.

### Behavior Changes

No intended behavior changes. Existing imports from `impala_learner.py` continue to work, including the V-trace percentile constant.

### Remaining Risks

- These are small facades, but they sit on public test/import surfaces used by masking and V-trace tests. Keep wrapper-equivalence tests in place if the modules move again.
- The full repository verifier passed after this checkpoint.

### Current Size Snapshot

- `python/weiss_rl/learners/impala_learner.py`: 3559 lines after the logp/V-trace diagnostic facade extraction.
- `python/weiss_rl/learners/action_logp.py`: 412 lines.
- `python/weiss_rl/learners/vtrace_diagnostics.py`: 28 lines.

## 2026-05-11 - Runtime Actor Model Helper Extraction

### Scope

- Moved runtime actor model compile and inference-model selection helpers into `weiss_rl.runtime_actor_models`.
- Preserved `_maybe_compile_runtime_actor_model()` and `_actor_inference_model()` through `weiss_rl.runtime`.
- Added direct tests for compile helper behavior and compiled-model preference through the runtime wrappers.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime_actor_models.py python/weiss_rl/tests/test_runtime.py -k "actor_model or maybe_compile_runtime_actor_model"` | Passed: 3 passed, 87 deselected. |
| `uv run ruff check python/weiss_rl/runtime_actor_models.py python/weiss_rl/runtime.py python/weiss_rl/tests/test_runtime_actor_models.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/runtime_actor_models.py python/weiss_rl/runtime.py python/weiss_rl/tests/test_runtime_actor_models.py` | Initially reported `runtime.py` needed formatting; passed after formatting touched files. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_actor_models.py python/weiss_rl/tests/test_runtime_config.py python/weiss_rl/tests/test_runtime_batching.py python/weiss_rl/tests/test_runtime_logging.py python/weiss_rl/tests/test_runtime_metrics.py` | Passed: 103 passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1014 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/runtime_actor_models.py`.
- Updated `python/weiss_rl/runtime.py` to delegate actor compile/inference helpers.
- Added `python/weiss_rl/tests/test_runtime_actor_models.py`.
- Updated `REFACTOR_PLAN.md`, `CHANGELOG.md`, `docs/architecture.md`, and `docs/refactor_completion_audit.md`.

### Behavior Changes

No intended behavior changes. Runtime actor collection still calls the same private helper names from `runtime.py`; the extracted helpers preserve structured-trunk compile hook behavior and compiled-model preference.

### Remaining Risks

- This does not touch actor rollout loops. Future actor-runtime movement should still avoid collection loops until there is stronger characterization around central/process collector parity.
- The full repository verifier passed after this checkpoint.

### Current Size Snapshot

- `python/weiss_rl/runtime.py`: 6232 lines after the runtime actor model helper extraction.
- `python/weiss_rl/runtime_actor_models.py`: 29 lines.

## 2026-05-11 - Training Manifest Layout Extraction

### Scope

- Moved training manifest actor-device layout resolution into `weiss_rl.training.manifest_layout`.
- Preserved `_manifest_actor_device_layout()` through `python/scripts/train.py`.
- Added direct tests for manifest actor layout resolution and device-name normalization.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_training_manifest_layout.py python/weiss_rl/tests/test_script_entrypoint_smokes.py -k "training_manifest_layout or train_metadata_helpers"` | Passed: 3 passed, 14 deselected, 14 dependency warnings. |
| `uv run ruff check python/weiss_rl/training/manifest_layout.py python/scripts/train.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_training_manifest_layout.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/training/manifest_layout.py python/scripts/train.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_training_manifest_layout.py` | Initially reported `train.py` needed formatting; passed after formatting touched files. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_training_manifest_layout.py python/weiss_rl/tests/test_training_startup.py python/weiss_rl/tests/test_training_inputs.py python/weiss_rl/tests/test_script_entrypoint_smokes.py` | Passed: 27 passed, 14 dependency warnings. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1016 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/training/manifest_layout.py`.
- Updated `python/scripts/train.py` to delegate `_manifest_actor_device_layout()`.
- Updated `python/weiss_rl/training/__init__.py` exports.
- Added `python/weiss_rl/tests/test_training_manifest_layout.py`.
- Updated `REFACTOR_PLAN.md`, `CHANGELOG.md`, `docs/architecture.md`, and `docs/refactor_completion_audit.md`.

### Behavior Changes

No intended behavior changes. The training entrypoint still exposes the same private helper name for script-level compatibility, and manifest actor layout still flows through the same runtime config and actor-device resolution functions.

### Remaining Risks

- This is a manifest-shaping helper only; the training loop and runtime construction remain in `train.py`.
- The full repository verifier passed after this checkpoint.

### Current Size Snapshot

- `python/scripts/train.py`: 3306 lines after the training manifest layout extraction.
- `python/weiss_rl/training/manifest_layout.py`: 50 lines.

## 2026-05-11 - Runtime Thread Helper Extraction

### Scope

- Moved actor torch-thread configuration into `weiss_rl.runtime_threads`.
- Preserved `_configure_runtime_actor_torch_threads()` through `weiss_rl.runtime`.
- Added direct tests for wrapper behavior, invalid thread counts, and suppressed interop-thread `RuntimeError`.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime_threads.py python/weiss_rl/tests/test_runtime.py -k "runtime_threads or actor_torch_threads"` | Passed: 3 passed, 88 deselected. |
| `uv run ruff check python/weiss_rl/runtime_threads.py python/weiss_rl/runtime.py python/weiss_rl/tests/test_runtime_threads.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/runtime_threads.py python/weiss_rl/runtime.py python/weiss_rl/tests/test_runtime_threads.py` | Initially reported `runtime.py` needed formatting; passed after formatting touched files. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_threads.py python/weiss_rl/tests/test_runtime_actor_models.py python/weiss_rl/tests/test_runtime_config.py python/weiss_rl/tests/test_runtime_batching.py python/weiss_rl/tests/test_runtime_logging.py python/weiss_rl/tests/test_runtime_metrics.py` | Passed: 106 passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1019 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/runtime_threads.py`.
- Updated `python/weiss_rl/runtime.py` to delegate `_configure_runtime_actor_torch_threads()`.
- Added `python/weiss_rl/tests/test_runtime_threads.py`.
- Updated `REFACTOR_PLAN.md`, `CHANGELOG.md`, `docs/architecture.md`, and `docs/refactor_completion_audit.md`.

### Behavior Changes

No intended behavior changes. Actor process startup and central CPU runtime setup still call the same private runtime helper name.

### Remaining Risks

- This is deliberately limited to thread setup; collector loop behavior remains in `runtime.py`.
- The full repository verifier passed after this checkpoint.

### Current Size Snapshot

- `python/weiss_rl/runtime.py`: 6228 lines after the runtime thread helper extraction.
- `python/weiss_rl/runtime_threads.py`: 16 lines.

## 2026-05-11 - Production Type Cleanup Checkpoint

### Scope

- Tightened type contracts in recently extracted learner, runtime, evaluation, legal-action, checkpoint, TensorBoard, AlphaRank, and PPO-lite surfaces without changing runtime behavior.
- Made the checkpoint learner protocol declare the fields already saved/restored by checkpoint helpers.
- Changed AlphaRank `alpha` annotations from `int` to `float` to match existing config parsing and existing float-based fixation math.
- Kept the broader full-package mypy gap explicit instead of treating the selected verifier mypy target as complete package typing.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run mypy python/weiss_rl/learners/structured_policy_metrics.py python/weiss_rl/learners/action_logp.py python/weiss_rl/runtime_devices.py --show-error-codes --no-error-summary` | Passed. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_learner_structured_policy_metrics.py python/weiss_rl/tests/test_learner_action_logp.py python/weiss_rl/tests/test_runtime_config.py python/weiss_rl/tests/test_runtime.py -k "structured_policy_metrics or learner_action_logp or resolve_actor_device_layout"` | Passed: 9 passed, 90 deselected. |
| `uv run mypy python/weiss_rl/eval/paper_readiness_fixture.py python/weiss_rl/legal_actions.py python/weiss_rl/eval/heuristic_public.py python/weiss_rl/training/checkpoint_guard.py python/weiss_rl/training/dev_eval.py --show-error-codes --no-error-summary` | Passed after narrowing JSON payload casts, dense legal-mask action-space inference, dtype arguments, and checkpoint-guard/dev-eval locals. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_paper_readiness_fixture.py python/weiss_rl/tests/test_contracts.py python/weiss_rl/tests/test_heuristic_public.py python/weiss_rl/tests/test_training_dev_eval.py python/weiss_rl/tests/test_train_stall_monitor.py -k "paper_readiness_fixture or legal or heuristic_public or checkpoint_guard or stall_monitor"` | Passed: 62 passed, 42 deselected, 14 dependency warnings. |
| `uv run mypy python/weiss_rl/tensorboard_logger.py python/weiss_rl/metagame/alpharank.py python/weiss_rl/metagame/sensitivity.py python/weiss_rl/training/checkpoints.py --show-error-codes --no-error-summary` | Passed. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_tensorboard_logger.py python/weiss_rl/tests/test_metagame_alpharank.py python/weiss_rl/tests/test_training_checkpoint_writers.py python/weiss_rl/tests/test_ppo_lite_learner.py` | Passed: 24 passed, 14 dependency warnings. |
| `uv run ruff check python/weiss_rl/tensorboard_logger.py python/weiss_rl/metagame/alpharank.py python/weiss_rl/metagame/sensitivity.py python/weiss_rl/learners/impala_learner.py python/weiss_rl/learners/ppo_lite_learner.py python/weiss_rl/training/checkpoints.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/tensorboard_logger.py python/weiss_rl/metagame/alpharank.py python/weiss_rl/metagame/sensitivity.py python/weiss_rl/learners/impala_learner.py python/weiss_rl/learners/ppo_lite_learner.py python/weiss_rl/training/checkpoints.py` | Passed after formatting touched files. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1019 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Failed: 332 error lines in the broad package census before the follow-up config test fixture cleanup. Top errors were test fixture payload typing (`dict(object)` with unused ignores) plus older `impala_learner.py` optional/narrowing debt. |

### Changes

- Updated `python/weiss_rl/learners/structured_policy_metrics.py`, `python/weiss_rl/learners/action_logp.py`, `python/weiss_rl/runtime_devices.py`, `python/weiss_rl/eval/paper_readiness_fixture.py`, `python/weiss_rl/legal_actions.py`, `python/weiss_rl/eval/heuristic_public.py`, `python/weiss_rl/training/checkpoint_guard.py`, and `python/weiss_rl/training/dev_eval.py` to satisfy focused production mypy checks.
- Updated `python/weiss_rl/tensorboard_logger.py` so the optional TensorBoard import is represented as a runtime factory instead of assigning `None` to an imported class symbol.
- Updated `python/weiss_rl/metagame/alpharank.py` annotations so `alpha` accepts floats, matching config semantics.
- Updated `python/weiss_rl/learners/impala_learner.py` and `python/weiss_rl/learners/ppo_lite_learner.py` logging hook signatures so PPO-lite remains compatible with the base hook while preserving its context payload.
- Updated `python/weiss_rl/training/checkpoints.py` protocol fields to document checkpoint save/restore expectations.

### Behavior Changes

No intended behavior changes. The only semantic-looking annotation change is AlphaRank `alpha: float`, which matches existing config parsing and existing `float(alpha)` math; integer callers remain accepted.

### Remaining Risks

- Full-package mypy is still not clean and is explicitly tracked as a validation gap.
- The largest files remain `runtime.py` (6228 lines), `model.py` (4657), `impala_learner.py` (3561), and `train.py` (3306), so the repository-wide refactor is still incomplete.

## 2026-05-11 - Config Test Typing Cleanup Checkpoint

### Scope

- Removed fixture-copy `type: ignore` comments from config section tests by adding small typed section-copy helpers.
- Kept the tests behavior-preserving: invalid payload tests still pass intentionally bad values to parser validation paths.
- Reduced the broad package mypy census from 332 to 290 error lines.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run mypy python/weiss_rl/tests/test_config_sections_training.py python/weiss_rl/tests/test_config_sections_league.py python/weiss_rl/tests/test_config_sections_evaluation.py --show-error-codes --no-error-summary` | Passed. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_config_sections_training.py python/weiss_rl/tests/test_config_sections_league.py python/weiss_rl/tests/test_config_sections_evaluation.py` | Passed: 15 passed. |
| `uv run mypy python/weiss_rl/tests/test_config_sections_environment.py python/weiss_rl/tests/test_config_sections_reproducibility.py python/weiss_rl/tests/test_config_seed_sets.py --show-error-codes --no-error-summary` | Passed. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_config_sections_environment.py python/weiss_rl/tests/test_config_sections_reproducibility.py python/weiss_rl/tests/test_config_seed_sets.py` | Passed: 14 passed. |
| `uv run ruff check python/weiss_rl/tests/test_config_sections_training.py python/weiss_rl/tests/test_config_sections_league.py python/weiss_rl/tests/test_config_sections_evaluation.py python/weiss_rl/tests/test_config_sections_environment.py python/weiss_rl/tests/test_config_sections_reproducibility.py python/weiss_rl/tests/test_config_seed_sets.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/tests/test_config_sections_training.py python/weiss_rl/tests/test_config_sections_league.py python/weiss_rl/tests/test_config_sections_evaluation.py python/weiss_rl/tests/test_config_sections_environment.py python/weiss_rl/tests/test_config_sections_reproducibility.py python/weiss_rl/tests/test_config_seed_sets.py` | Passed. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Failed: 290 error lines in the broad package census. Current top errors include config-loader optional assertions, a small `envs.pool_factory` literal narrowing issue, replay-runner payload typing, and older model/learner/runtime debt. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1019 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Updated `python/weiss_rl/tests/test_config_sections_training.py`.
- Updated `python/weiss_rl/tests/test_config_sections_league.py`.
- Updated `python/weiss_rl/tests/test_config_sections_evaluation.py`.
- Updated `python/weiss_rl/tests/test_config_sections_environment.py`.
- Updated `python/weiss_rl/tests/test_config_sections_reproducibility.py`.
- Updated `python/weiss_rl/tests/test_config_seed_sets.py`.

### Behavior Changes

No behavior changes. These are test typing improvements only; parser validation expectations are unchanged.

### Remaining Risks

- Full-package mypy still fails with 290 error lines.
- The repository-wide refactor remains incomplete while the large-file audit is red.

## 2026-05-11 - Broad Mypy Typing Cleanup Checkpoint

### Scope

- Cleared the next bounded full-package mypy slice across config-loader tests, runtime config tests, runtime IPC tests, training batch tests, learner batch-field tests, actor/outcome tests, model-loading tests, checkpoint-writer tests, env-pool construction, replay runner setup, and simulator eval result construction.
- Kept production changes behavior-preserving: literal annotations now describe existing env-pool profile entrypoints, replay/env max counters use explicit locals, and simulator eval results are constructed with explicit `GameResult` fields instead of a dynamically typed payload dict.
- Reduced the broad package mypy census from 290 to 229 error lines.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run mypy python/weiss_rl/tests/test_config_loader.py python/weiss_rl/envs/pool_factory.py python/weiss_rl/replay/runner.py python/weiss_rl/tests/test_training_checkpoint_writers.py --show-error-codes --no-error-summary` | Passed after adding typed test helpers/protocol fields and production literal/payload narrowing. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_training_checkpoint_writers.py python/weiss_rl/tests/test_replay_bundles.py python/weiss_rl/tests/test_pool_factory.py` | Passed: 66 passed. |
| `uv run mypy python/weiss_rl/tests/test_runtime_config.py python/weiss_rl/tests/test_runtime_ipc.py python/weiss_rl/tests/test_training_batches.py python/weiss_rl/tests/test_learner_batch_fields.py --show-error-codes --no-error-summary` | Passed. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime_config.py python/weiss_rl/tests/test_runtime_ipc.py python/weiss_rl/tests/test_training_batches.py python/weiss_rl/tests/test_learner_batch_fields.py` | Passed: 15 passed. |
| `uv run mypy python/weiss_rl/tests/test_actor_worker.py python/weiss_rl/tests/test_actor_outcomes.py python/weiss_rl/tests/test_contracts.py python/weiss_rl/tests/test_model_loading.py --show-error-codes --no-error-summary` | Passed. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_actor_worker.py python/weiss_rl/tests/test_actor_outcomes.py python/weiss_rl/tests/test_contracts.py python/weiss_rl/tests/test_model_loading.py` | Passed: 69 passed. |
| `uv run mypy python/weiss_rl/eval/simulator_runner.py --show-error-codes --no-error-summary` | Passed. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_heuristic_public.py python/weiss_rl/tests/test_snapshot_registry.py -k "simulator or snapshot_registry or replay_capture"` | Passed: 42 passed, 27 deselected, 14 dependency warnings. |
| `uv run ruff check` on touched files | Passed after import sorting. |
| `uv run ruff format --check` on touched files | Passed after formatting touched files. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Failed: 229 error lines in the broad package census. Remaining top-level groups are now concentrated in `model.py`, `impala_learner.py`, `runtime.py`, and IMPALA/V-trace test kwargs. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1019 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Updated `python/weiss_rl/tests/test_config_loader.py`.
- Updated `python/weiss_rl/envs/pool_factory.py`.
- Updated `python/weiss_rl/replay/runner.py`.
- Updated `python/weiss_rl/tests/test_training_checkpoint_writers.py`.
- Updated `python/weiss_rl/tests/test_runtime_config.py`.
- Updated `python/weiss_rl/tests/test_runtime_ipc.py`.
- Updated `python/weiss_rl/tests/test_training_batches.py`.
- Updated `python/weiss_rl/tests/test_learner_batch_fields.py`.
- Updated `python/weiss_rl/tests/test_actor_worker.py`.
- Updated `python/weiss_rl/tests/test_actor_outcomes.py`.
- Updated `python/weiss_rl/tests/test_contracts.py`.
- Updated `python/weiss_rl/tests/test_model_loading.py`.
- Updated `python/weiss_rl/eval/simulator_runner.py`.

### Behavior Changes

No intended behavior changes. The production edits are type-contract clarifications and explicit construction of the same values.

### Remaining Risks

- Full-package mypy still fails with 229 error lines.
- Remaining broad-mypy errors are now concentrated in behavior-sensitive model, learner, and runtime files, so any further cleanup there should be separated from larger structural refactors and covered by focused characterization tests.
- The repository-wide refactor remains incomplete while the large-file audit is red.

## 2026-05-11 - Model and Test Harness Typing Cleanup Checkpoint

### Scope

- Cleared `python/weiss_rl/model.py` under focused mypy by narrowing packed candidate metadata construction, avoiding local-name redefinition in candidate projection, and documenting the intentional structured policy-head override with a cast.
- Cleaned IMPALA/V-trace/script/paper-readiness test kwargs and fake-object typing without changing tested behavior.
- Cleaned additional evaluation, replay, runtime-wrapper, promotion, environment, entrypoint, and stall-monitor test fakes so the broad package mypy census is now mostly concentrated in `impala_learner.py`, `runtime.py`, `test_runtime.py`, and a few snapshot-registry fixtures.
- Reduced the broad package mypy census from 229 to 98 error lines.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run mypy python/weiss_rl/model.py --show-error-codes --no-error-summary` | Passed. |
| `uv run ruff check python/weiss_rl/model.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/model.py` | Passed after formatting. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_contracts.py python/weiss_rl/tests/test_model_sampling.py python/weiss_rl/tests/test_model_tensor_ops.py python/weiss_rl/tests/test_model_typed_encoder.py python/weiss_rl/tests/test_model_layers.py python/weiss_rl/tests/test_model_action_plans.py` | Passed: 78 passed. |
| `uv run mypy python/weiss_rl/tests/test_script_entrypoint_smokes.py python/weiss_rl/tests/test_paper_readiness.py python/weiss_rl/tests/test_vtrace.py python/weiss_rl/tests/test_impala_learner.py --show-error-codes --no-error-summary` | Passed. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_script_entrypoint_smokes.py python/weiss_rl/tests/test_paper_readiness.py python/weiss_rl/tests/test_vtrace.py python/weiss_rl/tests/test_impala_learner.py` | Passed: 86 passed, 14 dependency warnings. |
| `uv run mypy python/weiss_rl/tests/test_play_vs_model.py python/weiss_rl/tests/test_runtime_actor_models.py python/weiss_rl/tests/test_replay_inspector.py python/weiss_rl/tests/test_heuristic_public.py --show-error-codes --no-error-summary` | Passed. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_play_vs_model.py python/weiss_rl/tests/test_runtime_actor_models.py python/weiss_rl/tests/test_replay_inspector.py python/weiss_rl/tests/test_heuristic_public.py` | Passed: 42 passed. |
| `uv run mypy python/weiss_rl/tests/test_training_promotion.py python/weiss_rl/tests/test_training_environments.py python/weiss_rl/tests/test_entrypoints.py python/weiss_rl/tests/test_train_stall_monitor.py --show-error-codes --no-error-summary` | Passed. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_training_promotion.py python/weiss_rl/tests/test_training_environments.py python/weiss_rl/tests/test_entrypoints.py python/weiss_rl/tests/test_train_stall_monitor.py` | Passed: 59 passed, 14 dependency warnings. |
| `uv run ruff check` on touched files | Passed. |
| `uv run ruff format --check` on touched files | Passed after formatting touched files. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Failed: 98 error lines in the broad package census. Remaining errors are concentrated in `impala_learner.py`, `runtime.py`, `test_runtime.py`, and a few snapshot-registry fixture casts/assertions. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1019 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Updated `python/weiss_rl/model.py`.
- Updated `python/weiss_rl/tests/test_impala_learner.py`.
- Updated `python/weiss_rl/tests/test_vtrace.py`.
- Updated `python/weiss_rl/tests/test_script_entrypoint_smokes.py`.
- Updated `python/weiss_rl/tests/test_paper_readiness.py`.
- Updated `python/weiss_rl/tests/test_play_vs_model.py`.
- Updated `python/weiss_rl/tests/test_runtime_actor_models.py`.
- Updated `python/weiss_rl/tests/test_replay_inspector.py`.
- Updated `python/weiss_rl/tests/test_heuristic_public.py`.
- Updated `python/weiss_rl/tests/test_training_promotion.py`.
- Updated `python/weiss_rl/tests/test_training_environments.py`.
- Updated `python/weiss_rl/tests/test_entrypoints.py`.
- Updated `python/weiss_rl/tests/test_train_stall_monitor.py`.

### Behavior Changes

No intended behavior changes. Production changes in `model.py` are typing/narrowing clarifications around existing packed metadata and structured-head behavior. Test changes add casts and explicit assertions around deliberately small fakes.

### Remaining Risks

- Full-package mypy still fails with 98 error lines.
- The remaining broad-mypy production errors are in `impala_learner.py` and `runtime.py`, both behavior-sensitive danger zones.
- The repository-wide refactor remains incomplete while `runtime.py`, `model.py`, `impala_learner.py`, and `train.py` remain large orchestration files.

## 2026-05-11 - Runtime and Snapshot Test Typing Cleanup Checkpoint

### Scope

- Cleaned the remaining broad-mypy errors in `python/weiss_rl/tests/test_runtime.py` and `python/weiss_rl/tests/test_snapshot_registry.py`.
- Kept changes test-only: casts document deliberate fake actors, fake models, optional fixture fields, and simplified stack/config objects used to exercise production compatibility wrappers.
- Reduced the broad package mypy census from 98 to 55 error lines. The remaining broad-mypy failures are now only in production `python/weiss_rl/learners/impala_learner.py` and `python/weiss_rl/runtime.py`.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run mypy python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_snapshot_registry.py --show-error-codes --no-error-summary` | Passed. |
| `uv run ruff check python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_snapshot_registry.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_snapshot_registry.py` | Passed after formatting. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_snapshot_registry.py` | Passed: 129 passed, 14 dependency warnings. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Failed: 55 error lines, all in `impala_learner.py` and `runtime.py`. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1019 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Updated `python/weiss_rl/tests/test_runtime.py`.
- Updated `python/weiss_rl/tests/test_snapshot_registry.py`.

### Behavior Changes

No behavior changes. The runtime and snapshot tests still exercise the same fake-object paths and production wrappers; the edits make those fakes explicit to mypy.

### Remaining Risks

- Full-package mypy still fails with 55 production error lines in `impala_learner.py` and `runtime.py`.
- Further progress now requires behavior-sensitive production narrowing or structural movement in the remaining danger-zone files.
- The repository-wide refactor remains incomplete while the large-file audit is red.

## 2026-05-11 - Production Mypy Closure Checkpoint

### Scope

- Cleaned the remaining broad-package mypy errors in `python/weiss_rl/learners/impala_learner.py` and `python/weiss_rl/runtime.py`.
- Kept edits behavior-preserving: branch-local context dictionaries now have distinct names, optional tensors are asserted only after existing runtime guards, factorized learner scatter/reshape code uses local helpers for non-optional narrowing, runtime shared-pending metrics accept the same pending-unroll objects already used at runtime, and debug `source_label` compatibility is represented in the mask-policy helper signature.
- Brought the broad package mypy census from 55 error lines to zero.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run mypy python/weiss_rl/learners/impala_learner.py --show-error-codes --no-error-summary` | Passed. |
| `uv run ruff check python/weiss_rl/learners/impala_learner.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/learners/impala_learner.py` | Passed after formatting. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_impala_learner.py python/weiss_rl/tests/test_vtrace.py python/weiss_rl/tests/test_learner_structured_auxiliary.py python/weiss_rl/tests/test_learner_structured_policy_metrics.py python/weiss_rl/tests/test_learner_action_logp.py python/weiss_rl/tests/test_learner_legal_fields.py python/weiss_rl/tests/test_learner_vtrace_torch.py python/weiss_rl/tests/test_learner_vtrace_diagnostics.py` | Passed: 84 passed. |
| `uv run mypy python/weiss_rl/runtime.py --show-error-codes --no-error-summary` | Passed. |
| `uv run ruff check python/weiss_rl/runtime.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/runtime.py` | Passed after formatting. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_config.py python/weiss_rl/tests/test_runtime_batching.py python/weiss_rl/tests/test_runtime_ipc.py python/weiss_rl/tests/test_runtime_metrics.py python/weiss_rl/tests/test_runtime_opponents.py python/weiss_rl/tests/test_runtime_threads.py python/weiss_rl/tests/test_runtime_actor_models.py python/weiss_rl/tests/test_runtime_collector_commands.py python/weiss_rl/tests/test_runtime_counters.py python/weiss_rl/tests/test_runtime_hashing.py python/weiss_rl/tests/test_runtime_logging.py` | Passed: 127 passed. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed: 0 error lines. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1019 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Updated `python/weiss_rl/learners/impala_learner.py`.
- Updated `python/weiss_rl/runtime.py`.

### Behavior Changes

No intended behavior changes. The edits are type-narrowing and compatibility-signature clarifications around already guarded paths.

### Remaining Risks

- Full-package mypy is now clean, but the repository-wide refactor is still not complete while the large-file audit remains red: current line counts are `runtime.py` 6253, `model.py` 4672, `impala_learner.py` 3552, and `python/scripts/train.py` 3306.
- Further structural movement in `runtime.py`, `model.py`, `impala_learner.py`, and `train.py` should still be split into behavior-preserving checkpoints with focused characterization tests.

## 2026-05-11 - Runtime Pending Selection Extraction Checkpoint

### Scope

- Extracted pending-unroll selection, diverse-lane counting, and diverse batch target calculation from `QueueRuntime` into `python/weiss_rl/runtime_pending.py`.
- Preserved `QueueRuntime` private compatibility wrappers: `_select_pending_unrolls`, `_actor_id_is_diverse_lane`, `_pending_unroll_is_diverse_lane`, `_pending_diverse_unroll_count`, and `_diverse_batch_target_count` still exist.
- Added direct characterization tests for the extracted helper behavior in `python/weiss_rl/tests/test_runtime_pending.py`.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run mypy python/weiss_rl/runtime_pending.py python/weiss_rl/runtime.py python/weiss_rl/tests/test_runtime_pending.py --show-error-codes --no-error-summary` | Passed. |
| `uv run ruff check python/weiss_rl/runtime_pending.py python/weiss_rl/runtime.py python/weiss_rl/tests/test_runtime_pending.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/runtime_pending.py python/weiss_rl/runtime.py python/weiss_rl/tests/test_runtime_pending.py` | Passed after formatting. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime_pending.py python/weiss_rl/tests/test_runtime.py -k "select_pending_unrolls or fill_pending_unrolls or diverse_lane"` | Passed: 8 passed, 84 deselected. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_pending.py python/weiss_rl/tests/test_runtime_config.py python/weiss_rl/tests/test_runtime_batching.py python/weiss_rl/tests/test_runtime_metrics.py python/weiss_rl/tests/test_runtime_opponents.py python/weiss_rl/tests/test_runtime_threads.py` | Passed: 111 passed. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1023 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/runtime_pending.py`.
- Added `python/weiss_rl/tests/test_runtime_pending.py`.
- Updated `python/weiss_rl/runtime.py`.

### Behavior Changes

No intended behavior changes. This is a behavior-preserving extraction of existing pending-unroll selection logic behind private compatibility wrappers.

### Remaining Risks

- `runtime.py` is still large at 6228 lines after the extraction.
- The large-file audit remains red: `model.py` is 4672 lines, `impala_learner.py` is 3552 lines, and `python/scripts/train.py` is 3306 lines.

## 2026-05-11 - Runtime Process Collector Extraction Checkpoint

### Scope

- Extracted the process-collector child-loop body from `runtime.py` into `python/weiss_rl/runtime_process.py`.
- Preserved the public/private multiprocessing target name `_collector_process_main` in `runtime.py`; it now delegates to `runtime_process.collector_process_main` with `QueueRuntime` passed explicitly to avoid import cycles.
- Left process startup ownership in `QueueRuntime._start_process_collectors`, where mutable runtime queues, slots, and process lists are managed.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run mypy python/weiss_rl/runtime_process.py python/weiss_rl/runtime.py --show-error-codes --no-error-summary` | Passed. |
| `uv run ruff check python/weiss_rl/runtime_process.py python/weiss_rl/runtime.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/runtime_process.py python/weiss_rl/runtime.py` | Passed after formatting. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_collector_commands.py python/weiss_rl/tests/test_runtime_ipc.py python/weiss_rl/tests/test_runtime_logging.py -k "process or collector or ipc or debug_log or shared"` | Passed: 17 passed, 79 deselected. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_collector_commands.py python/weiss_rl/tests/test_runtime_ipc.py python/weiss_rl/tests/test_runtime_logging.py python/weiss_rl/tests/test_runtime_pending.py python/weiss_rl/tests/test_runtime_config.py python/weiss_rl/tests/test_runtime_batching.py python/weiss_rl/tests/test_runtime_metrics.py python/weiss_rl/tests/test_runtime_opponents.py python/weiss_rl/tests/test_runtime_threads.py` | Passed: 119 passed. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1023 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/runtime_process.py`.
- Updated `python/weiss_rl/runtime.py`.

### Behavior Changes

No intended behavior changes. This is a process-collector child-loop extraction; the multiprocessing target wrapper, command handling, shared-slot handoff, debug logging, and runtime close behavior are preserved.

### Remaining Risks

- `runtime.py` is still large at 6090 lines.
- The large-file audit remains red: `model.py` is 4672 lines, `impala_learner.py` is 3552 lines, and `python/scripts/train.py` is 3306 lines.

## 2026-05-11 - Runtime Actor State Extraction Checkpoint

### Scope

- Extracted actor environment construction, actor-state initialization, and the actor seed formula from `runtime.py` into `python/weiss_rl/runtime_actor_state.py`.
- Preserved `QueueRuntime._build_env`, `QueueRuntime._build_actor_state`, and the module-level `_actor_seed` compatibility wrapper in `runtime.py`.
- Added direct characterization coverage for the extracted actor-state helper, including model eval/compile handoff, deterministic actor seed use, default mirror opponent slots, diverse-lane flags, fixed-opponent slots, and initial episode role assignment.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run mypy python/weiss_rl/runtime_actor_state.py python/weiss_rl/tests/test_runtime_actor_state.py python/weiss_rl/runtime.py --show-error-codes --no-error-summary` | Passed. |
| `uv run ruff check python/weiss_rl/runtime_actor_state.py python/weiss_rl/tests/test_runtime_actor_state.py python/weiss_rl/runtime.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/runtime_actor_state.py python/weiss_rl/tests/test_runtime_actor_state.py python/weiss_rl/runtime.py` | Passed. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime_actor_state.py python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_actor_models.py python/weiss_rl/tests/test_runtime_config.py -k "build_actor_state or actor_state or build_env or QueueRuntime or actor_seed"` | Passed: 3 passed, 92 deselected. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_actor_state.py python/weiss_rl/tests/test_runtime_actor_models.py python/weiss_rl/tests/test_runtime_config.py python/weiss_rl/tests/test_runtime_threads.py` | Passed: 98 passed. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1025 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/runtime_actor_state.py`.
- Added `python/weiss_rl/tests/test_runtime_actor_state.py`.
- Updated `python/weiss_rl/runtime.py`.

### Behavior Changes

No intended behavior changes. This is a behavior-preserving extraction of actor env/state setup behind existing runtime wrappers; seed formula, mirror opponent defaults, fixed-opponent slots, shared/non-shared actor-model handling, and initial role assignment are preserved.

### Remaining Risks

- `runtime.py` is still large at 6062 lines after the extraction.
- The large-file audit remains red: `model.py` is 4672 lines, `impala_learner.py` is 3552 lines, and `python/scripts/train.py` is 3306 lines.
- Central runtime collection loops, heuristic-public routing, PFSP/opponent sampling, learner-batch conversion, and reset/role utility glue remain behavior-sensitive future extraction areas.

## 2026-05-11 - Runtime Packed Debug Validation Extraction Checkpoint

### Scope

- Extracted debug-only packed legal-action validators from `runtime.py` into `python/weiss_rl/runtime_debug_validation.py`.
- Preserved the runtime feature-flag wrappers `_maybe_debug_validate_sampled_packed_actions` and `_maybe_debug_validate_env_step_packed_actions`.
- Added direct characterization tests for valid packed sampled actions, empty legal rows using the pass action, illegal sampled actions, matching env-step legality, and env-step legality mismatches.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run mypy python/weiss_rl/runtime_debug_validation.py python/weiss_rl/tests/test_runtime_debug_validation.py python/weiss_rl/runtime.py --show-error-codes --no-error-summary` | Passed. |
| `uv run ruff check python/weiss_rl/runtime_debug_validation.py python/weiss_rl/tests/test_runtime_debug_validation.py python/weiss_rl/runtime.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/runtime_debug_validation.py python/weiss_rl/tests/test_runtime_debug_validation.py python/weiss_rl/runtime.py` | Passed after formatting. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime_debug_validation.py python/weiss_rl/tests/test_runtime.py -k "debug_validate or packed_action or env_step or packed"` | Passed: 10 passed, 82 deselected. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_actor_state.py python/weiss_rl/tests/test_runtime_actor_models.py python/weiss_rl/tests/test_runtime_config.py python/weiss_rl/tests/test_runtime_threads.py python/weiss_rl/tests/test_runtime_debug_validation.py python/weiss_rl/tests/test_runtime_batching.py` | Passed: 106 passed. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1029 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/runtime_debug_validation.py`.
- Added `python/weiss_rl/tests/test_runtime_debug_validation.py`.
- Updated `python/weiss_rl/runtime.py`.

### Behavior Changes

No intended behavior changes. This is a behavior-preserving extraction of existing debug validators; the runtime still controls whether validation runs, and the validation error messages are preserved.

### Remaining Risks

- `runtime.py` is still large at 6024 lines after the extraction.
- The large-file audit remains red: `model.py` is 4672 lines, `impala_learner.py` is 3552 lines, and `python/scripts/train.py` is 3306 lines.
- Debug validation coverage is now direct, but the central runtime collection loops and policy/opponent routing remain future extraction targets.

## 2026-05-11 - Runtime Legal Metadata Extraction Checkpoint

### Scope

- Extracted packed legal-action metadata construction from `runtime.py` into `python/weiss_rl/runtime_legal_meta.py`.
- Preserved `QueueRuntime._legal_action_meta_from_ids` and `QueueRuntime._ensure_legal_action_meta` as compatibility wrappers.
- Added direct characterization tests for family/slot/attack metadata encoding, minimum and wider metadata widths, existing-meta casting, and no-catalog behavior.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run mypy python/weiss_rl/runtime_legal_meta.py python/weiss_rl/tests/test_runtime_legal_meta.py python/weiss_rl/runtime.py --show-error-codes --no-error-summary` | Passed. |
| `uv run ruff check python/weiss_rl/runtime_legal_meta.py python/weiss_rl/tests/test_runtime_legal_meta.py python/weiss_rl/runtime.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/runtime_legal_meta.py python/weiss_rl/tests/test_runtime_legal_meta.py python/weiss_rl/runtime.py` | Passed after formatting. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime_legal_meta.py python/weiss_rl/tests/test_runtime.py -k "legal_action_meta or legal_meta or packed"` | Passed: 9 passed, 83 deselected. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_legal_meta.py python/weiss_rl/tests/test_runtime_debug_validation.py python/weiss_rl/tests/test_runtime_actor_state.py python/weiss_rl/tests/test_runtime_actor_models.py python/weiss_rl/tests/test_runtime_config.py python/weiss_rl/tests/test_runtime_threads.py python/weiss_rl/tests/test_runtime_batching.py` | Passed: 110 passed. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1033 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/runtime_legal_meta.py`.
- Added `python/weiss_rl/tests/test_runtime_legal_meta.py`.
- Updated `python/weiss_rl/runtime.py`.

### Behavior Changes

No intended behavior changes. This is a behavior-preserving extraction of legal-action metadata construction behind the existing runtime wrappers; packed candidate ordering and decoded metadata field placement are unchanged.

### Remaining Risks

- `runtime.py` is still large at 6006 lines after the extraction.
- The large-file audit remains red: `model.py` is 4672 lines, `impala_learner.py` is 3552 lines, and `python/scripts/train.py` is 3306 lines.
- Legal-action metadata construction now has direct tests, but the broader policy-row execution and central collection loops remain behavior-sensitive future extraction targets.

## 2026-05-11 - Runtime Outcome Bookkeeping Extraction Checkpoint

### Scope

- Extracted opponent outcome bookkeeping from `runtime.py` into `python/weiss_rl/runtime_outcomes.py`.
- Preserved `QueueRuntime._update_outcomes` and `QueueRuntime._update_outcomes_from_transition_arrays` as compatibility wrappers.
- Added direct characterization tests for mirror-opponent skipping, terminal win/draw/timeout mapping, transition-array win/loss/draw/timeout mapping, reward-perspective winner derivation, and incomplete-row ignoring.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run mypy python/weiss_rl/runtime_outcomes.py python/weiss_rl/tests/test_runtime_outcomes.py python/weiss_rl/runtime.py --show-error-codes --no-error-summary` | Passed. |
| `uv run ruff check python/weiss_rl/runtime_outcomes.py python/weiss_rl/tests/test_runtime_outcomes.py python/weiss_rl/runtime.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/runtime_outcomes.py python/weiss_rl/tests/test_runtime_outcomes.py python/weiss_rl/runtime.py` | Passed after formatting. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime_outcomes.py python/weiss_rl/tests/test_runtime.py -k "update_outcomes or outcomes or transition_arrays"` | Passed: 3 passed, 88 deselected. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_outcomes.py python/weiss_rl/tests/test_runtime_legal_meta.py python/weiss_rl/tests/test_runtime_debug_validation.py python/weiss_rl/tests/test_runtime_actor_state.py python/weiss_rl/tests/test_runtime_actor_models.py python/weiss_rl/tests/test_runtime_config.py python/weiss_rl/tests/test_runtime_threads.py python/weiss_rl/tests/test_runtime_batching.py python/weiss_rl/tests/test_runtime_opponents.py` | Passed: 119 passed. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1036 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/runtime_outcomes.py`.
- Added `python/weiss_rl/tests/test_runtime_outcomes.py`.
- Updated `python/weiss_rl/runtime.py`.

### Behavior Changes

No intended behavior changes. This is a behavior-preserving extraction of opponent outcome updates; mirror opponents are still skipped, terminal batch outcomes still use `game_result_from_step`, and transition-array reward semantics are unchanged.

### Remaining Risks

- `runtime.py` is still large at 5984 lines after the extraction.
- The large-file audit remains red: `model.py` is 4672 lines, `impala_learner.py` is 3552 lines, and `python/scripts/train.py` is 3306 lines.
- Outcome bookkeeping now has direct tests, but PFSP sampling, role assignment, policy-row execution, and central collection loops remain behavior-sensitive future extraction targets.

## 2026-05-11 - Runtime Bootstrap Value Extraction Checkpoint

### Scope

- Extracted bootstrap value tensor computation from `runtime.py` into `python/weiss_rl/runtime_bootstrap.py`.
- Preserved `QueueRuntime._bootstrap_values` as the runtime-owned compatibility wrapper that still selects the actor or per-actor bootstrap model and device.
- Added direct characterization tests for the value-only model path, the `forward_seat_aware` fallback path, valid bootstrap actor row filtering, and all-invalid row handling.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run mypy python/weiss_rl/runtime_bootstrap.py python/weiss_rl/tests/test_runtime_bootstrap.py python/weiss_rl/runtime.py --show-error-codes --no-error-summary` | Passed. |
| `uv run ruff check python/weiss_rl/runtime_bootstrap.py python/weiss_rl/tests/test_runtime_bootstrap.py python/weiss_rl/runtime.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/runtime_bootstrap.py python/weiss_rl/tests/test_runtime_bootstrap.py python/weiss_rl/runtime.py` | Passed after formatting. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime_bootstrap.py python/weiss_rl/tests/test_runtime.py -k "bootstrap_values or bootstrap"` | Passed: 6 passed, 85 deselected. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_bootstrap.py python/weiss_rl/tests/test_runtime_outcomes.py python/weiss_rl/tests/test_runtime_legal_meta.py python/weiss_rl/tests/test_runtime_debug_validation.py python/weiss_rl/tests/test_runtime_actor_state.py python/weiss_rl/tests/test_runtime_actor_models.py python/weiss_rl/tests/test_runtime_config.py python/weiss_rl/tests/test_runtime_threads.py python/weiss_rl/tests/test_runtime_batching.py python/weiss_rl/tests/test_runtime_opponents.py` | Passed: 122 passed. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1039 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/runtime_bootstrap.py`.
- Added `python/weiss_rl/tests/test_runtime_bootstrap.py`.
- Updated `python/weiss_rl/runtime.py`.

### Behavior Changes

No intended behavior changes. This is a behavior-preserving extraction of bootstrap value computation; runtime model/device selection, valid bootstrap actor filtering, value-only preference, forward fallback, and CPU materialization are preserved.

### Remaining Risks

- `runtime.py` is still large at 5965 lines after the extraction.
- The large-file audit remains red: `model.py` is 4672 lines, `impala_learner.py` is 3552 lines, and `python/scripts/train.py` is 3306 lines.
- Bootstrap computation now has direct tests, but policy-row execution, central collection loops, PFSP sampling, and role/reset bookkeeping remain behavior-sensitive future extraction targets.

## 2026-05-11 - Model Candidate Component Extraction Checkpoint

### Scope

- Extracted structured candidate component resolution from `model.py` into `python/weiss_rl/model_candidate_components.py`.
- Preserved `_StructuredLegalActionHead._resolve_candidate_components` as the existing model method wrapper.
- Added direct characterization tests for action-id table lookup, packed metadata decoding by family, unused metadata sentinel handling, and missing-family behavior.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run mypy python/weiss_rl/model_candidate_components.py python/weiss_rl/tests/test_model_candidate_components.py python/weiss_rl/model.py --show-error-codes --no-error-summary` | Passed. |
| `uv run ruff check python/weiss_rl/model_candidate_components.py python/weiss_rl/tests/test_model_candidate_components.py python/weiss_rl/model.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/model_candidate_components.py python/weiss_rl/tests/test_model_candidate_components.py python/weiss_rl/model.py` | Passed after formatting. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_model_candidate_components.py python/weiss_rl/tests/test_contracts.py -k "candidate_components or packed_meta or packed_path or factorized_path or packed_legal"` | Passed: 7 passed, 46 deselected. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_model_candidate_components.py python/weiss_rl/tests/test_contracts.py python/weiss_rl/tests/test_model_sampling.py python/weiss_rl/tests/test_model_tensor_ops.py python/weiss_rl/tests/test_model_typed_encoder.py python/weiss_rl/tests/test_model_layers.py python/weiss_rl/tests/test_model_action_plans.py python/weiss_rl/tests/test_model_observation_contract.py` | Passed: 85 passed. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1042 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/model_candidate_components.py`.
- Added `python/weiss_rl/tests/test_model_candidate_components.py`.
- Updated `python/weiss_rl/model.py`.

### Behavior Changes

No intended behavior changes. This is a behavior-preserving extraction of the structured legal-head component resolver; action-id lookup, packed metadata decoding, sentinel handling, and family-specific component assignment are preserved.

### Remaining Risks

- `model.py` is still large at 4628 lines after the extraction.
- The large-file audit remains red: `runtime.py` is 5965 lines, `impala_learner.py` is 3552 lines, and `python/scripts/train.py` is 3306 lines.
- Candidate component resolution now has direct tests, but structured scoring, public-heuristic biasing, factorized distributions, and model forward paths remain behavior-sensitive future extraction targets.

## 2026-05-11 - Model Action Table Extraction Checkpoint

### Scope

- Extracted structured action component table construction and factorized lookup table construction from `_StructuredLegalActionHead.__init__` into `python/weiss_rl/model_action_tables.py`.
- Preserved all model buffer names and downstream model methods: `_family_ids`, `_action_arg0`, `_action_arg1`, component-index buffers, `_family_arg_kind`, argument-size buffers, no-arg/one-arg/two-arg action-id tables, and slot/index family id tuples.
- Added direct characterization tests for action catalog decoding into component arrays and factorized lookup contracts for no-arg, one-arg, and two-arg action families.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run mypy python/weiss_rl/model_action_tables.py python/weiss_rl/tests/test_model_action_tables.py python/weiss_rl/model.py --show-error-codes --no-error-summary` | Passed. |
| `uv run ruff check python/weiss_rl/model_action_tables.py python/weiss_rl/tests/test_model_action_tables.py python/weiss_rl/model.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/model_action_tables.py python/weiss_rl/tests/test_model_action_tables.py python/weiss_rl/model.py` | Passed after formatting. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_model_action_tables.py python/weiss_rl/tests/test_contracts.py -k "action_tables or factorized or packed_meta or packed_path or packed_legal"` | Passed: 9 passed, 43 deselected. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_model_action_tables.py python/weiss_rl/tests/test_model_candidate_components.py python/weiss_rl/tests/test_contracts.py python/weiss_rl/tests/test_model_sampling.py python/weiss_rl/tests/test_model_tensor_ops.py python/weiss_rl/tests/test_model_typed_encoder.py python/weiss_rl/tests/test_model_layers.py python/weiss_rl/tests/test_model_action_plans.py python/weiss_rl/tests/test_model_observation_contract.py` | Passed: 87 passed. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1044 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/model_action_tables.py`.
- Added `python/weiss_rl/tests/test_model_action_tables.py`.
- Updated `python/weiss_rl/model.py`.

### Behavior Changes

No intended behavior changes. This is a behavior-preserving extraction of constructor-time action lookup table construction; legal action ids, component tables, and factorized lookup tables retain the same ordering and sentinel conventions.

### Remaining Risks

- `model.py` is still large at 4571 lines after the extraction.
- The large-file audit remains red: `runtime.py` is 5965 lines, `impala_learner.py` is 3552 lines, and `python/scripts/train.py` is 3306 lines.
- Action table construction now has direct tests, but structured scoring, public-heuristic biasing, factorized distributions, and model forward paths remain behavior-sensitive future extraction targets.

## 2026-05-11 - Model Feature Gathering Extraction Checkpoint

### Scope

- Extracted structured stage feature gathering helpers from `_StructuredLegalActionHead` into `python/weiss_rl/model_feature_gathering.py`.
- Preserved `_StructuredLegalActionHead._gather_stage_features_for_rows`, `_gather_stage_features`, and `_slot_component` as existing model method wrappers.
- Added direct characterization tests for per-row slot gathering, batched stage gathering, shared table gathering, invalid-slot zeroing, and absent component offsets.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run mypy python/weiss_rl/model_feature_gathering.py python/weiss_rl/tests/test_model_feature_gathering.py python/weiss_rl/model.py --show-error-codes --no-error-summary` | Passed. |
| `uv run ruff check python/weiss_rl/model_feature_gathering.py python/weiss_rl/tests/test_model_feature_gathering.py python/weiss_rl/model.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/model_feature_gathering.py python/weiss_rl/tests/test_model_feature_gathering.py python/weiss_rl/model.py` | Passed after formatting `model.py`. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_model_feature_gathering.py python/weiss_rl/tests/test_contracts.py -k "stage_features or slot_component or packed_path or packed_legal"` | Passed: 6 passed, 48 deselected. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_model_feature_gathering.py python/weiss_rl/tests/test_model_action_tables.py python/weiss_rl/tests/test_model_candidate_components.py python/weiss_rl/tests/test_contracts.py python/weiss_rl/tests/test_model_sampling.py python/weiss_rl/tests/test_model_tensor_ops.py python/weiss_rl/tests/test_model_typed_encoder.py python/weiss_rl/tests/test_model_layers.py python/weiss_rl/tests/test_model_action_plans.py python/weiss_rl/tests/test_model_observation_contract.py` | Passed: 91 passed. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1048 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/model_feature_gathering.py`.
- Added `python/weiss_rl/tests/test_model_feature_gathering.py`.
- Updated `python/weiss_rl/model.py`.

### Behavior Changes

No intended behavior changes. This is a behavior-preserving extraction of tensor gathering helpers; valid-slot selection, invalid-slot zeroing, all-invalid row handling, shared table lookup, and missing-offset behavior are preserved behind the original model method names.

### Remaining Risks

- `model.py` is still large at 4553 lines after the extraction.
- The large-file audit remains red: `runtime.py` is 5965 lines, `impala_learner.py` is 3552 lines, and `python/scripts/train.py` is 3306 lines.
- Stage feature gathering now has direct tests, but structured scoring, public-heuristic biasing, factorized distributions, hand-card feature lookup, and model forward paths remain behavior-sensitive future extraction targets.

## 2026-05-11 - Model Public Heuristic Utility Extraction Checkpoint

### Scope

- Extracted small structured public-heuristic tensor utilities from `_StructuredLegalActionHead` into `python/weiss_rl/model_public_heuristics.py`.
- Preserved `_StructuredLegalActionHead._slot_preference_values`, `_public_prefer_lower`, `_public_slot_action_score`, `_combine_public_heuristic_scores`, and `_apply_public_heuristic_bias` as existing model method wrappers.
- Added direct characterization tests for invalid-slot zeroing, empty slot preferences, lower-index preference, slot power buckets, score combination weights, disabled bias, ungated bias, and family allow-list biasing.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run mypy python/weiss_rl/model_public_heuristics.py python/weiss_rl/tests/test_model_public_heuristics.py python/weiss_rl/model.py --show-error-codes --no-error-summary` | Passed. |
| `uv run ruff check python/weiss_rl/model_public_heuristics.py python/weiss_rl/tests/test_model_public_heuristics.py python/weiss_rl/model.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/model_public_heuristics.py python/weiss_rl/tests/test_model_public_heuristics.py python/weiss_rl/model.py` | Passed after formatting `model.py`. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_model_public_heuristics.py python/weiss_rl/tests/test_contracts.py -k "public_heuristic or packed_path or packed_legal"` | Passed: 7 passed, 48 deselected. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_model_public_heuristics.py python/weiss_rl/tests/test_model_feature_gathering.py python/weiss_rl/tests/test_model_action_tables.py python/weiss_rl/tests/test_model_candidate_components.py python/weiss_rl/tests/test_contracts.py python/weiss_rl/tests/test_model_sampling.py python/weiss_rl/tests/test_model_tensor_ops.py python/weiss_rl/tests/test_model_typed_encoder.py python/weiss_rl/tests/test_model_layers.py python/weiss_rl/tests/test_model_action_plans.py python/weiss_rl/tests/test_model_observation_contract.py` | Passed: 96 passed. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1053 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/model_public_heuristics.py`.
- Added `python/weiss_rl/tests/test_model_public_heuristics.py`.
- Updated `python/weiss_rl/model.py`.

### Behavior Changes

No intended behavior changes. This is a behavior-preserving extraction of public-heuristic tensor utilities; slot preference lookup, invalid-slot zeroing, power-bucket scoring, weighted combination, bias scale handling, and family allow-list gating are preserved behind the original model method names.

### Remaining Risks

- `model.py` is still large at 4556 lines after the extraction. This checkpoint improves isolation and testability but does not materially reduce the model god-file line count because compatibility wrappers and imports offset the moved helper bodies.
- The large-file audit remains red: `runtime.py` is 5965 lines, `impala_learner.py` is 3552 lines, and `python/scripts/train.py` is 3306 lines.
- The full public-heuristic scoring path remains behavior-sensitive; do not move larger score construction blocks without complete candidate-score parity coverage.

## 2026-05-11 - Model Candidate Partitioning Extraction Checkpoint

### Scope

- Extracted structured candidate-family partitioning from `_StructuredLegalActionHead` into `python/weiss_rl/model_candidate_partitioning.py`.
- Preserved `_StructuredLegalActionHead._partition_candidate_family_indices` as the existing model method wrapper.
- Added direct characterization tests for group order, candidate order inside each group, empty long tensor behavior, device preservation, and default-group fallback.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run mypy python/weiss_rl/model_candidate_partitioning.py python/weiss_rl/tests/test_model_candidate_partitioning.py python/weiss_rl/model.py --show-error-codes --no-error-summary` | Passed. |
| `uv run ruff check python/weiss_rl/model_candidate_partitioning.py python/weiss_rl/tests/test_model_candidate_partitioning.py python/weiss_rl/model.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/model_candidate_partitioning.py python/weiss_rl/tests/test_model_candidate_partitioning.py python/weiss_rl/model.py` | Passed after formatting `model.py`. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_model_candidate_partitioning.py python/weiss_rl/tests/test_contracts.py -k "partition_candidate or packed_path or packed_legal"` | Passed: 4 passed, 48 deselected. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_model_candidate_partitioning.py python/weiss_rl/tests/test_model_public_heuristics.py python/weiss_rl/tests/test_model_feature_gathering.py python/weiss_rl/tests/test_model_action_tables.py python/weiss_rl/tests/test_model_candidate_components.py python/weiss_rl/tests/test_contracts.py python/weiss_rl/tests/test_model_sampling.py python/weiss_rl/tests/test_model_tensor_ops.py python/weiss_rl/tests/test_model_typed_encoder.py python/weiss_rl/tests/test_model_layers.py python/weiss_rl/tests/test_model_action_plans.py python/weiss_rl/tests/test_model_observation_contract.py` | Passed: 98 passed. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1055 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/model_candidate_partitioning.py`.
- Added `python/weiss_rl/tests/test_model_candidate_partitioning.py`.
- Updated `python/weiss_rl/model.py`.

### Behavior Changes

No intended behavior changes. This is a behavior-preserving extraction of candidate-family partitioning; play, hand, move, attack, slot, index, and default group order plus within-group candidate ordering are preserved behind the original model method name.

### Remaining Risks

- `model.py` is still large at 4537 lines after the extraction.
- The large-file audit remains red: `runtime.py` is 5965 lines, `impala_learner.py` is 3552 lines, and `python/scripts/train.py` is 3306 lines.
- Candidate partitioning now has direct tests, but larger structured scoring and public-heuristic candidate score construction still need stronger parity tests before movement.

## 2026-05-11 - IMPALA Packed Row Helper Extraction Checkpoint

### Scope

- Extracted packed legal-action row slicing, flat candidate position construction, scatter-back, legal-action view construction, and observation-context row subsetting from `ImpalaLearner` into `python/weiss_rl/learners/packed_rows.py`.
- Preserved `ImpalaLearner._packed_legal_action_view`, `_slice_packed_legal_rows_with_meta`, `_packed_candidate_positions_for_rows`, `_scatter_packed_candidate_values`, and `_subset_observation_context_rows` as existing learner method wrappers.
- Added direct characterization tests for selected row order, zero-width rows, empty selections, metadata slicing, flat candidate positions, scatter fill values, and row-major context subsetting.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run mypy python/weiss_rl/learners/packed_rows.py python/weiss_rl/tests/test_learner_packed_rows.py python/weiss_rl/learners/impala_learner.py --show-error-codes --no-error-summary` | Passed after tightening an optional-metadata assertion in the new test. |
| `uv run ruff check python/weiss_rl/learners/packed_rows.py python/weiss_rl/tests/test_learner_packed_rows.py python/weiss_rl/learners/impala_learner.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/learners/packed_rows.py python/weiss_rl/tests/test_learner_packed_rows.py python/weiss_rl/learners/impala_learner.py` | Passed after formatting `impala_learner.py`. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_learner_packed_rows.py python/weiss_rl/tests/test_impala_learner.py -k "packed_row or packed_candidate or subset_observation_context or factorized or packed"` | Passed: 23 passed, 25 deselected. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_learner_packed_rows.py python/weiss_rl/tests/test_learner_action_logp.py python/weiss_rl/tests/test_learner_batch_fields.py python/weiss_rl/tests/test_learner_faults.py python/weiss_rl/tests/test_learner_legal_fields.py python/weiss_rl/tests/test_learner_logging.py python/weiss_rl/tests/test_learner_structured_auxiliary.py python/weiss_rl/tests/test_learner_structured_policy_metrics.py python/weiss_rl/tests/test_learner_tensor_ops.py python/weiss_rl/tests/test_learner_vtrace_diagnostics.py python/weiss_rl/tests/test_learner_vtrace_torch.py python/weiss_rl/tests/test_impala_learner.py` | Passed: 94 passed. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1061 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/learners/packed_rows.py`.
- Added `python/weiss_rl/tests/test_learner_packed_rows.py`.
- Updated `python/weiss_rl/learners/impala_learner.py`.

### Behavior Changes

No intended behavior changes. This is a behavior-preserving extraction of packed-row utilities; selected row order, zero-width row handling, empty selections, metadata slicing, candidate scatter positions, fill values, and observation-context row subsetting are preserved behind the original learner method names.

### Remaining Risks

- `impala_learner.py` is still large at 3508 lines after the extraction.
- The large-file audit remains red: `runtime.py` is 5965 lines, `model.py` is 4537 lines, and `python/scripts/train.py` is 3306 lines.
- Packed-row manipulation now has direct tests, but update orchestration, bootstrap value resolution, factorized teacher view construction, and learner forward sequencing remain behavior-sensitive future extraction targets.

## 2026-05-11 - IMPALA Bootstrap Helper Extraction Checkpoint

### Scope

- Extracted raw V-trace input gating, current-model bootstrap value evaluation, and stored bootstrap-value fallback from `ImpalaLearner` into `python/weiss_rl/learners/bootstrap.py`.
- Preserved `ImpalaLearner._has_raw_vtrace_inputs`, `_resolve_vtrace_bootstrap_value`, and `_current_model_bootstrap_value` as existing learner method wrappers.
- Added direct characterization tests for stored versus current bootstrap availability, `value_seat_aware` preference, `forward_seat_aware` fallback, missing fields, missing/unsupported model behavior, hidden-state shape handling, bootstrap shape validation, and current-model precedence over stored bootstrap values.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run mypy python/weiss_rl/learners/bootstrap.py python/weiss_rl/tests/test_learner_bootstrap.py python/weiss_rl/learners/impala_learner.py --show-error-codes --no-error-summary` | Passed. |
| `uv run ruff check python/weiss_rl/learners/bootstrap.py python/weiss_rl/tests/test_learner_bootstrap.py python/weiss_rl/learners/impala_learner.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/learners/bootstrap.py python/weiss_rl/tests/test_learner_bootstrap.py python/weiss_rl/learners/impala_learner.py` | Passed after formatting `bootstrap.py` and `impala_learner.py`. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_learner_bootstrap.py python/weiss_rl/tests/test_impala_learner.py -k "bootstrap or raw_vtrace"` | Passed: 10 passed, 39 deselected. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_learner_bootstrap.py python/weiss_rl/tests/test_learner_packed_rows.py python/weiss_rl/tests/test_learner_action_logp.py python/weiss_rl/tests/test_learner_batch_fields.py python/weiss_rl/tests/test_learner_faults.py python/weiss_rl/tests/test_learner_legal_fields.py python/weiss_rl/tests/test_learner_logging.py python/weiss_rl/tests/test_learner_structured_auxiliary.py python/weiss_rl/tests/test_learner_structured_policy_metrics.py python/weiss_rl/tests/test_learner_tensor_ops.py python/weiss_rl/tests/test_learner_vtrace_diagnostics.py python/weiss_rl/tests/test_learner_vtrace_torch.py python/weiss_rl/tests/test_impala_learner.py` | Passed: 101 passed. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1068 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/learners/bootstrap.py`.
- Added `python/weiss_rl/tests/test_learner_bootstrap.py`.
- Updated `python/weiss_rl/learners/impala_learner.py`.

### Behavior Changes

No intended behavior changes. This is a behavior-preserving extraction of raw V-trace bootstrap helpers; stored bootstrap availability, current-model bootstrap precedence, value-only path preference, forward fallback, shape validation, and dtype/device conversion are preserved behind the original learner method names.

### Remaining Risks

- `impala_learner.py` is still large at 3463 lines after the extraction.
- The large-file audit remains red: `runtime.py` is 5965 lines, `model.py` is 4537 lines, and `python/scripts/train.py` is 3306 lines.
- Bootstrap resolution now has direct tests, but learner update orchestration, factorized teacher view construction, and forward sequencing remain behavior-sensitive future extraction targets.

## 2026-05-11 - Training Report Payload Extraction Checkpoint

### Scope

- Extracted run summary, determinism report, environment manifest, and training-control payload augmentation from `python/scripts/train.py` into `python/weiss_rl/training/report_payloads.py`.
- Replaced duplicated payload mutation blocks in `main()` with helper calls while preserving the existing report keys.
- Added direct characterization tests for runtime mode handling, policy selection keys, training controls, B1 baseline path reporting, seed snapshot path reporting, resume path reporting, environment `cwd`/`argv`/hardware fields, and public-demo defaults.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run mypy python/weiss_rl/training/report_payloads.py python/weiss_rl/tests/test_training_report_payloads.py --show-error-codes --no-error-summary` | Passed. |
| `uv run mypy python/weiss_rl/training/report_payloads.py python/weiss_rl/tests/test_training_report_payloads.py python/scripts/train.py --show-error-codes --no-error-summary` | Failed on existing script-level typing debt in `python/scripts/train.py` around pre-existing protocol/cast annotations; no helper/test errors. |
| `uv run ruff check python/weiss_rl/training/report_payloads.py python/weiss_rl/tests/test_training_report_payloads.py python/scripts/train.py` | Passed after import sorting. |
| `uv run ruff format --check python/weiss_rl/training/report_payloads.py python/weiss_rl/tests/test_training_report_payloads.py python/scripts/train.py` | Passed after formatting `train.py`. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_training_report_payloads.py python/weiss_rl/tests/test_entrypoints.py -k "report_payloads or manifest_only or smoke or policy_set_selection_mode"` | Passed: 5 passed, 31 deselected. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_training_report_payloads.py python/weiss_rl/tests/test_training_algorithm_contracts.py python/weiss_rl/tests/test_training_batches.py python/weiss_rl/tests/test_training_checkpoint_writers.py python/weiss_rl/tests/test_training_dev_eval.py python/weiss_rl/tests/test_training_environments.py python/weiss_rl/tests/test_training_guidance.py python/weiss_rl/tests/test_training_inputs.py python/weiss_rl/tests/test_training_manifest_layout.py python/weiss_rl/tests/test_training_promotion.py python/weiss_rl/tests/test_training_startup.py python/weiss_rl/tests/test_training_torch_threads.py python/weiss_rl/tests/test_entrypoints.py -k "report_payloads or training or manifest_only or smoke or policy_set_selection_mode or resume"` | Passed: 67 passed, 31 deselected. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1073 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/training/report_payloads.py`.
- Added `python/weiss_rl/tests/test_training_report_payloads.py`.
- Updated `python/scripts/train.py`.

### Behavior Changes

No intended behavior changes. This is a behavior-preserving extraction of report payload augmentation; run summary and determinism report key names, training-control serialization, public-demo runtime naming, resume payloads, path serialization, and environment manifest fields are preserved.

### Remaining Risks

- `python/scripts/train.py` is still large at 3298 lines after the extraction.
- The large-file audit remains red: `runtime.py` is 5965 lines, `model.py` is 4537 lines, and `impala_learner.py` is 3463 lines.
- Direct script-level mypy for `python/scripts/train.py` still exposes pre-existing protocol/cast typing debt not covered by the package mypy gate; keep using the full verifier and targeted helper typing until that script debt is deliberately closed.

## 2026-05-11 - Training Run Identity Extraction Checkpoint

### Scope

- Extracted fresh-run id generation and resumed-run manifest identity/hash validation from `python/scripts/train.py` into `python/weiss_rl/training/run_identity.py`.
- Updated `main()` to use the helper for both new runs and resume runs while preserving run id, run directory, and mismatch error semantics.
- Added direct characterization tests for run id computation, default run directory naming, explicit run labels, resumed manifest id normalization, and spec/config hash mismatch errors.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run mypy python/weiss_rl/training/run_identity.py python/weiss_rl/tests/test_training_run_identity.py --show-error-codes --no-error-summary` | Passed. |
| `uv run ruff check python/weiss_rl/training/run_identity.py python/weiss_rl/tests/test_training_run_identity.py python/scripts/train.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/training/run_identity.py python/weiss_rl/tests/test_training_run_identity.py python/scripts/train.py` | Passed after formatting `train.py`. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_training_run_identity.py python/weiss_rl/tests/test_entrypoints.py -k "resume or run_identity or manifest_only"` | Passed: 5 passed, 30 deselected. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_training_run_identity.py python/weiss_rl/tests/test_training_report_payloads.py python/weiss_rl/tests/test_training_algorithm_contracts.py python/weiss_rl/tests/test_training_batches.py python/weiss_rl/tests/test_training_checkpoint_writers.py python/weiss_rl/tests/test_training_dev_eval.py python/weiss_rl/tests/test_training_environments.py python/weiss_rl/tests/test_training_guidance.py python/weiss_rl/tests/test_training_inputs.py python/weiss_rl/tests/test_training_manifest_layout.py python/weiss_rl/tests/test_training_promotion.py python/weiss_rl/tests/test_training_startup.py python/weiss_rl/tests/test_training_torch_threads.py python/weiss_rl/tests/test_entrypoints.py -k "run_identity or report_payloads or training or manifest_only or smoke or policy_set_selection_mode or resume"` | Passed: 72 passed, 30 deselected. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1077 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/training/run_identity.py`.
- Added `python/weiss_rl/tests/test_training_run_identity.py`.
- Updated `python/scripts/train.py`.

### Behavior Changes

No intended behavior changes. This is a behavior-preserving extraction of run identity handling; fresh run id computation, run-label/default-directory behavior, resumed manifest id normalization, and immutable spec/config hash validation are preserved.

### Remaining Risks

- `python/scripts/train.py` is still large at 3296 lines after the extraction.
- The large-file audit remains red: `runtime.py` is 5965 lines, `model.py` is 4537 lines, and `impala_learner.py` is 3463 lines.
- Resume identity is now directly tested, but the surrounding resume restore path and checkpoint state restoration remain behavior-sensitive future extraction targets.

## 2026-05-11 - Training Profiling Message Extraction Checkpoint

### Scope

- Extracted the structured profiling startup message from `python/scripts/train.py` into `python/weiss_rl/training/report_payloads.py`.
- Updated `main()` to print the helper-returned message when profiling is enabled.
- Added direct characterization coverage for the exact user-facing message and the no-profiling `None` path.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run mypy python/weiss_rl/training/report_payloads.py python/weiss_rl/tests/test_training_report_payloads.py --show-error-codes --no-error-summary` | Passed. |
| `uv run ruff check python/weiss_rl/training/report_payloads.py python/weiss_rl/tests/test_training_report_payloads.py python/scripts/train.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/training/report_payloads.py python/weiss_rl/tests/test_training_report_payloads.py python/scripts/train.py` | Passed after formatting `train.py` and the test. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_training_report_payloads.py python/weiss_rl/tests/test_entrypoints.py -k "report_payloads or profile_timers or torch_profiler"` | Passed: 9 passed, 28 deselected. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_training_report_payloads.py python/weiss_rl/tests/test_training_run_identity.py python/weiss_rl/tests/test_training_algorithm_contracts.py python/weiss_rl/tests/test_training_batches.py python/weiss_rl/tests/test_training_checkpoint_writers.py python/weiss_rl/tests/test_training_dev_eval.py python/weiss_rl/tests/test_training_environments.py python/weiss_rl/tests/test_training_guidance.py python/weiss_rl/tests/test_training_inputs.py python/weiss_rl/tests/test_training_manifest_layout.py python/weiss_rl/tests/test_training_promotion.py python/weiss_rl/tests/test_training_startup.py python/weiss_rl/tests/test_training_torch_threads.py python/weiss_rl/tests/test_entrypoints.py -k "run_identity or report_payloads or profile_timers or torch_profiler or training or manifest_only or smoke or policy_set_selection_mode or resume"` | Passed: 76 passed, 27 deselected. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1078 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Updated `python/weiss_rl/training/report_payloads.py`.
- Updated `python/weiss_rl/tests/test_training_report_payloads.py`.
- Updated `python/scripts/train.py`.

### Behavior Changes

No intended behavior changes. The structured profiling startup text and the condition for emitting it are preserved.

### Remaining Risks

- `python/scripts/train.py` is still large at 3291 lines after the extraction.
- The large-file audit remains red: `runtime.py` is 5965 lines, `model.py` is 4537 lines, and `impala_learner.py` is 3463 lines.
- The main train loop, promotion gate, and periodic dev-eval orchestration remain behavior-sensitive future extraction targets.

## 2026-05-11 - Training Execution Settings Extraction Checkpoint

### Scope

- Extracted checkpoint interval resolution, profiling flag resolution, and B1/seed snapshot directory normalization from `python/scripts/train.py` into `python/weiss_rl/training/execution.py`.
- Updated `main()` to pass a `TrainingExecutionSettings` result into `_run_minimal_training`.
- Added direct characterization tests for config defaults, CLI checkpoint interval override precedence, optional import directory normalization, and non-positive checkpoint interval rejection.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run mypy python/weiss_rl/training/execution.py python/weiss_rl/tests/test_training_execution.py --show-error-codes --no-error-summary` | Passed. |
| `uv run ruff check python/weiss_rl/training/execution.py python/weiss_rl/tests/test_training_execution.py python/scripts/train.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/training/execution.py python/weiss_rl/tests/test_training_execution.py python/scripts/train.py` | Passed after formatting `train.py`. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_training_execution.py python/weiss_rl/tests/test_entrypoints.py -k "training_execution or checkpoint_interval or profile_timers or torch_profiler or resume"` | Passed: 7 passed, 27 deselected. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_training_execution.py python/weiss_rl/tests/test_training_report_payloads.py python/weiss_rl/tests/test_training_run_identity.py python/weiss_rl/tests/test_training_algorithm_contracts.py python/weiss_rl/tests/test_training_batches.py python/weiss_rl/tests/test_training_checkpoint_writers.py python/weiss_rl/tests/test_training_dev_eval.py python/weiss_rl/tests/test_training_environments.py python/weiss_rl/tests/test_training_guidance.py python/weiss_rl/tests/test_training_inputs.py python/weiss_rl/tests/test_training_manifest_layout.py python/weiss_rl/tests/test_training_promotion.py python/weiss_rl/tests/test_training_startup.py python/weiss_rl/tests/test_training_torch_threads.py python/weiss_rl/tests/test_entrypoints.py -k "training_execution or run_identity or report_payloads or profile_timers or torch_profiler or training or manifest_only or smoke or policy_set_selection_mode or resume"` | Passed: 79 passed, 27 deselected. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1081 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/training/execution.py`.
- Added `python/weiss_rl/tests/test_training_execution.py`.
- Updated `python/scripts/train.py`.

### Behavior Changes

No intended behavior changes. This is a behavior-preserving extraction of training execution controls; checkpoint interval validation, profiling flags, and optional imported snapshot directory resolution are preserved.

### Remaining Risks

- `python/scripts/train.py` is still large at 3290 lines after the extraction.
- The large-file audit remains red: `runtime.py` is 5965 lines, `model.py` is 4537 lines, and `impala_learner.py` is 3463 lines.
- The main train loop, promotion gate, periodic dev-eval orchestration, and resume restore path remain behavior-sensitive future extraction targets.

## 2026-05-11 - Training Profiler Helper Extraction Checkpoint

### Scope

- Extracted the training profiler context manager and torch profiler construction from `python/scripts/train.py` into `python/weiss_rl/training/profiling.py`.
- Updated structured warmstart and minimal training loops to use `profile_block()` for named regions.
- Updated torch profiler setup to use `build_training_profiler()` while preserving disabled behavior, trace directory location, CPU activity, and CUDA activity selection.
- Added direct characterization tests for disabled profiling, named profiler regions, disabled no-directory behavior, CPU profiler construction, and CUDA device activity inclusion.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_training_profiling.py python/weiss_rl/tests/test_entrypoints.py -k "training_profiling or profile_timers or torch_profiler"` | Passed: 8 passed, 28 deselected. |
| `uv run mypy python/weiss_rl/training/profiling.py python/weiss_rl/tests/test_training_profiling.py --show-error-codes --no-error-summary` | Passed. |
| `uv run ruff check python/weiss_rl/training/profiling.py python/weiss_rl/tests/test_training_profiling.py python/scripts/train.py python/weiss_rl/training/__init__.py` | Passed after import cleanup. |
| `uv run ruff format --check python/weiss_rl/training/profiling.py python/weiss_rl/tests/test_training_profiling.py python/scripts/train.py python/weiss_rl/training/__init__.py` | Passed after formatting `train.py`. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_training_profiling.py python/weiss_rl/tests/test_training_execution.py python/weiss_rl/tests/test_training_report_payloads.py python/weiss_rl/tests/test_training_run_identity.py python/weiss_rl/tests/test_training_algorithm_contracts.py python/weiss_rl/tests/test_training_batches.py python/weiss_rl/tests/test_training_checkpoint_writers.py python/weiss_rl/tests/test_training_dev_eval.py python/weiss_rl/tests/test_training_environments.py python/weiss_rl/tests/test_training_guidance.py python/weiss_rl/tests/test_training_inputs.py python/weiss_rl/tests/test_training_manifest_layout.py python/weiss_rl/tests/test_training_promotion.py python/weiss_rl/tests/test_training_startup.py python/weiss_rl/tests/test_training_torch_threads.py python/weiss_rl/tests/test_entrypoints.py -k "training_profiling or training_execution or run_identity or report_payloads or profile_timers or torch_profiler or training or manifest_only or smoke or policy_set_selection_mode or resume"` | Passed: 84 passed, 27 deselected. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1086 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/training/profiling.py`.
- Added `python/weiss_rl/tests/test_training_profiling.py`.
- Updated `python/scripts/train.py`.
- Updated `python/weiss_rl/training/__init__.py`.

### Behavior Changes

No intended behavior changes. This is a behavior-preserving extraction of profiler plumbing; disabled profiling still avoids trace-directory creation, enabled torch profiling still writes under `profiling/torch_profiler`, CPU profiling remains the default activity, CUDA devices still add CUDA activity, and named timer blocks still use `torch.autograd.profiler.record_function()`.

### Remaining Risks

- `python/scripts/train.py` is still large at 3258 lines after the extraction.
- The large-file audit remains red: `runtime.py` is 5965 lines, `model.py` is 4537 lines, and `impala_learner.py` is 3463 lines.
- Promotion-gate and periodic-dev-eval orchestration remain in `train.py` and should wait for stronger parity tests before movement.

## 2026-05-11 - Training Batch Collection Extraction Checkpoint

### Scope

- Extracted the algorithm-specific runtime batch collection dispatcher from `python/scripts/train.py` into `python/weiss_rl/training/batches.py`.
- Promoted the IMPALA and PPO algorithm sets to `training.batches` constants and kept private aliases in `train.py` for compatibility with existing internal checks.
- Updated structured warmstart and minimal training loops to call `collect_training_batch()`.
- Added direct characterization tests for IMPALA/V-trace collection arguments, PPO/GAE collection arguments, and unsupported algorithm errors.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_training_batches.py python/weiss_rl/tests/test_entrypoints.py -k "collect_training_batch or training or smoke or profile_timers or torch_profiler"` | Passed: 9 passed, 28 deselected. |
| `uv run mypy python/weiss_rl/training/batches.py python/weiss_rl/tests/test_training_batches.py --show-error-codes --no-error-summary` | Passed. |
| `uv run ruff check python/weiss_rl/training/batches.py python/weiss_rl/tests/test_training_batches.py python/scripts/train.py python/weiss_rl/training/__init__.py` | Passed after import sorting. |
| `uv run ruff format --check python/weiss_rl/training/batches.py python/weiss_rl/tests/test_training_batches.py python/scripts/train.py python/weiss_rl/training/__init__.py` | Passed after formatting `train.py` and `training/batches.py`. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_training_batches.py python/weiss_rl/tests/test_training_profiling.py python/weiss_rl/tests/test_training_execution.py python/weiss_rl/tests/test_training_report_payloads.py python/weiss_rl/tests/test_training_run_identity.py python/weiss_rl/tests/test_training_algorithm_contracts.py python/weiss_rl/tests/test_training_checkpoint_writers.py python/weiss_rl/tests/test_training_dev_eval.py python/weiss_rl/tests/test_training_environments.py python/weiss_rl/tests/test_training_guidance.py python/weiss_rl/tests/test_training_inputs.py python/weiss_rl/tests/test_training_manifest_layout.py python/weiss_rl/tests/test_training_promotion.py python/weiss_rl/tests/test_training_startup.py python/weiss_rl/tests/test_training_torch_threads.py python/weiss_rl/tests/test_entrypoints.py -k "collect_training_batch or training_profiling or training_execution or run_identity or report_payloads or profile_timers or torch_profiler or training or manifest_only or smoke or policy_set_selection_mode or resume"` | Passed: 87 passed, 27 deselected. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1089 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Updated `python/weiss_rl/training/batches.py`.
- Updated `python/weiss_rl/tests/test_training_batches.py`.
- Updated `python/scripts/train.py`.
- Updated `python/weiss_rl/training/__init__.py`.

### Behavior Changes

No intended behavior changes. IMPALA algorithms still call `QueueRuntime.collect_update_batch()` with gamma, truncation reward/bootstrap, and V-trace rho/c arguments; PPO-lite still calls `QueueRuntime.collect_policy_batch()` with gamma, GAE lambda, and truncation settings; unsupported training algorithms still raise `RuntimeError` with the original message shape.

### Remaining Risks

- `python/scripts/train.py` is still large at 3234 lines after the extraction.
- The large-file audit remains red: `runtime.py` is 5965 lines, `model.py` is 4537 lines, and `impala_learner.py` is 3463 lines.
- The live training loop still owns learner update orchestration; avoid moving that loop without broader parity coverage.

## 2026-05-11 - Runtime Actor State Container Extraction Checkpoint

### Scope

- Moved the `_ActorState` dataclass from `python/weiss_rl/runtime.py` into `python/weiss_rl/runtime_actor_state.py`, colocating the container with the helper that builds actor state.
- Preserved the `runtime._ActorState` compatibility name by importing the moved class back into `runtime.py`.
- Added direct characterization coverage for actor-state defaults (`snapshot_version`, `next_unroll_seq`, fixed-opponent slots) and slots behavior.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime_actor_state.py python/weiss_rl/tests/test_runtime.py -k "actor_state or build_actor_state or next_unroll_seq"` | Passed: 4 passed, 87 deselected. |
| `uv run mypy python/weiss_rl/runtime_actor_state.py python/weiss_rl/tests/test_runtime_actor_state.py --show-error-codes --no-error-summary` | Passed. |
| `uv run ruff check python/weiss_rl/runtime.py python/weiss_rl/runtime_actor_state.py python/weiss_rl/tests/test_runtime_actor_state.py` | Passed after import sorting. |
| `uv run ruff format --check python/weiss_rl/runtime.py python/weiss_rl/runtime_actor_state.py python/weiss_rl/tests/test_runtime_actor_state.py` | Passed after formatting `runtime.py`. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime_actor_state.py python/weiss_rl/tests/test_runtime_actor_models.py python/weiss_rl/tests/test_runtime_batching.py python/weiss_rl/tests/test_runtime_bootstrap.py python/weiss_rl/tests/test_runtime_collector_commands.py python/weiss_rl/tests/test_runtime_config.py python/weiss_rl/tests/test_runtime_counters.py python/weiss_rl/tests/test_runtime_debug_validation.py python/weiss_rl/tests/test_runtime_hashing.py python/weiss_rl/tests/test_runtime_ipc.py python/weiss_rl/tests/test_runtime_legal_meta.py python/weiss_rl/tests/test_runtime_logging.py python/weiss_rl/tests/test_runtime_metrics.py python/weiss_rl/tests/test_runtime_opponents.py python/weiss_rl/tests/test_runtime_outcomes.py python/weiss_rl/tests/test_runtime_pending.py python/weiss_rl/tests/test_runtime_threads.py python/weiss_rl/tests/test_runtime.py -k "runtime or actor_state or build_actor_state or next_unroll_seq"` | Passed: 148 passed. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1090 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Updated `python/weiss_rl/runtime.py`.
- Updated `python/weiss_rl/runtime_actor_state.py`.
- Updated `python/weiss_rl/tests/test_runtime_actor_state.py`.

### Behavior Changes

No intended behavior changes. The actor state fields, default `snapshot_version`, default `next_unroll_seq`, fixed-opponent slot default, and slots behavior are preserved. Existing imports from `weiss_rl.runtime` still see `_ActorState`.

### Remaining Risks

- `python/weiss_rl/runtime.py` is still large at 5946 lines after the extraction.
- The large-file audit remains red: `model.py` is 4537 lines, `impala_learner.py` is 3463 lines, and `train.py` is 3234 lines.
- Central runtime rollout collection, PFSP sampling, heuristic-public routing, and learner batch conversion remain behavior-sensitive future extraction targets.

## 2026-05-11 - Runtime Type Container Extraction Checkpoint

### Scope

- Moved immutable runtime data containers `RuntimeUnroll` and `RuntimeBatch` from `python/weiss_rl/runtime.py` into `python/weiss_rl/runtime_types.py`.
- Added the `PendingUnroll` type alias to `runtime_types.py`.
- Preserved `weiss_rl.runtime.RuntimeUnroll`, `weiss_rl.runtime.RuntimeBatch`, and `weiss_rl.runtime.PendingUnroll` import compatibility by importing the moved names back into `runtime.py`.
- Added direct characterization tests for optional default fields, frozen dataclass behavior, slots behavior, and runtime import compatibility.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime_types.py python/weiss_rl/tests/test_runtime.py -k "runtime_unroll or runtime_batch or shared_unroll"` | Passed: 2 passed, 88 deselected. |
| `uv run mypy python/weiss_rl/runtime_types.py python/weiss_rl/tests/test_runtime_types.py --show-error-codes --no-error-summary` | Passed. |
| `uv run ruff check python/weiss_rl/runtime.py python/weiss_rl/runtime_types.py python/weiss_rl/tests/test_runtime_types.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/runtime.py python/weiss_rl/runtime_types.py python/weiss_rl/tests/test_runtime_types.py` | Passed after formatting `runtime.py`. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime_types.py python/weiss_rl/tests/test_runtime_actor_state.py python/weiss_rl/tests/test_runtime_actor_models.py python/weiss_rl/tests/test_runtime_batching.py python/weiss_rl/tests/test_runtime_bootstrap.py python/weiss_rl/tests/test_runtime_collector_commands.py python/weiss_rl/tests/test_runtime_config.py python/weiss_rl/tests/test_runtime_counters.py python/weiss_rl/tests/test_runtime_debug_validation.py python/weiss_rl/tests/test_runtime_hashing.py python/weiss_rl/tests/test_runtime_ipc.py python/weiss_rl/tests/test_runtime_legal_meta.py python/weiss_rl/tests/test_runtime_logging.py python/weiss_rl/tests/test_runtime_metrics.py python/weiss_rl/tests/test_runtime_opponents.py python/weiss_rl/tests/test_runtime_outcomes.py python/weiss_rl/tests/test_runtime_pending.py python/weiss_rl/tests/test_runtime_threads.py python/weiss_rl/tests/test_runtime.py -k "runtime or actor_state or runtime_unroll or runtime_batch or shared_unroll"` | Passed: 150 passed. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1092 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/runtime_types.py`.
- Added `python/weiss_rl/tests/test_runtime_types.py`.
- Updated `python/weiss_rl/runtime.py`.

### Behavior Changes

No intended behavior changes. The runtime unroll/batch fields, optional defaults, frozen dataclass behavior, slots behavior, and import path compatibility through `weiss_rl.runtime` are preserved.

### Remaining Risks

- `python/weiss_rl/runtime.py` is still large at 5906 lines after the extraction.
- The large-file audit remains red: `model.py` is 4537 lines, `impala_learner.py` is 3463 lines, and `train.py` is 3234 lines.
- Moving data containers does not yet reduce the core complexity of central rollout collection, PFSP sampling, heuristic-public routing, or learner batch conversion.

## 2026-05-11 - Runtime Actor Scheduling Extraction Checkpoint

### Scope

- Extracted round-robin actor batch selection from `python/weiss_rl/runtime.py` into `python/weiss_rl/runtime_actor_scheduling.py`.
- Preserved `QueueRuntime._next_actor_batch()` as the compatibility wrapper that updates `_next_actor_index`.
- Added direct characterization tests for non-positive counts, wraparound, empty actor lists, and runtime wrapper cursor updates.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime_actor_scheduling.py python/weiss_rl/tests/test_runtime.py -k "next_actor_batch or fill_pending_unrolls_uses_parallel_executor"` | Passed: 5 passed, 87 deselected. |
| `uv run mypy python/weiss_rl/runtime_actor_scheduling.py python/weiss_rl/tests/test_runtime_actor_scheduling.py --show-error-codes --no-error-summary` | Passed. |
| `uv run ruff check python/weiss_rl/runtime.py python/weiss_rl/runtime_actor_scheduling.py python/weiss_rl/tests/test_runtime_actor_scheduling.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/runtime.py python/weiss_rl/runtime_actor_scheduling.py python/weiss_rl/tests/test_runtime_actor_scheduling.py` | Passed after formatting `runtime.py`. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime_actor_scheduling.py python/weiss_rl/tests/test_runtime_types.py python/weiss_rl/tests/test_runtime_actor_state.py python/weiss_rl/tests/test_runtime_actor_models.py python/weiss_rl/tests/test_runtime_batching.py python/weiss_rl/tests/test_runtime_bootstrap.py python/weiss_rl/tests/test_runtime_collector_commands.py python/weiss_rl/tests/test_runtime_config.py python/weiss_rl/tests/test_runtime_counters.py python/weiss_rl/tests/test_runtime_debug_validation.py python/weiss_rl/tests/test_runtime_hashing.py python/weiss_rl/tests/test_runtime_ipc.py python/weiss_rl/tests/test_runtime_legal_meta.py python/weiss_rl/tests/test_runtime_logging.py python/weiss_rl/tests/test_runtime_metrics.py python/weiss_rl/tests/test_runtime_opponents.py python/weiss_rl/tests/test_runtime_outcomes.py python/weiss_rl/tests/test_runtime_pending.py python/weiss_rl/tests/test_runtime_threads.py python/weiss_rl/tests/test_runtime.py -k "runtime or actor_scheduling or next_actor_batch or actor_state or runtime_unroll or runtime_batch or shared_unroll"` | Passed: 154 passed. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1096 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/runtime_actor_scheduling.py`.
- Added `python/weiss_rl/tests/test_runtime_actor_scheduling.py`.
- Updated `python/weiss_rl/runtime.py`.

### Behavior Changes

No intended behavior changes. The round-robin actor selection still caps each selection to at most the actor count, wraps the cursor modulo actor count, leaves the cursor unchanged for non-positive counts, and the runtime wrapper still stores the updated `_next_actor_index`.

### Remaining Risks

- `python/weiss_rl/runtime.py` is still large at 5905 lines after the extraction.
- The large-file audit remains red: `model.py` is 4537 lines, `impala_learner.py` is 3463 lines, and `train.py` is 3234 lines.
- Central rollout collection, PFSP sampling, heuristic-public routing, and learner batch conversion remain behavior-sensitive future extraction targets.

## 2026-05-11 - Runtime Teacher Label Extraction Checkpoint

### Scope

- Extracted teacher-guidance activation rules, default teacher-label array construction, and chosen-action label decoding from `python/weiss_rl/runtime.py` into `python/weiss_rl/runtime_teacher_labels.py`.
- Preserved `QueueRuntime._teacher_guidance_active_for_collection()`, `_teacher_label_arrays()`, and `_teacher_labels_from_actions()` as runtime wrappers.
- Added direct characterization tests for aux-mode activation rules, sentinel/default label arrays, family-specific decoded labels, and inactive sentinel behavior.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime_teacher_labels.py python/weiss_rl/tests/test_runtime.py -k "teacher_labels or teacher_guidance or concat_optional_time_major_field"` | Passed: 10 passed, 82 deselected. |
| `uv run mypy python/weiss_rl/runtime_teacher_labels.py python/weiss_rl/tests/test_runtime_teacher_labels.py --show-error-codes --no-error-summary` | Passed. |
| `uv run ruff check python/weiss_rl/runtime.py python/weiss_rl/runtime_teacher_labels.py python/weiss_rl/tests/test_runtime_teacher_labels.py` | Passed after import sorting. |
| `uv run ruff format --check python/weiss_rl/runtime.py python/weiss_rl/runtime_teacher_labels.py python/weiss_rl/tests/test_runtime_teacher_labels.py` | Passed after formatting `runtime.py`. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime_teacher_labels.py python/weiss_rl/tests/test_runtime_actor_scheduling.py python/weiss_rl/tests/test_runtime_types.py python/weiss_rl/tests/test_runtime_actor_state.py python/weiss_rl/tests/test_runtime_actor_models.py python/weiss_rl/tests/test_runtime_batching.py python/weiss_rl/tests/test_runtime_bootstrap.py python/weiss_rl/tests/test_runtime_collector_commands.py python/weiss_rl/tests/test_runtime_config.py python/weiss_rl/tests/test_runtime_counters.py python/weiss_rl/tests/test_runtime_debug_validation.py python/weiss_rl/tests/test_runtime_hashing.py python/weiss_rl/tests/test_runtime_ipc.py python/weiss_rl/tests/test_runtime_legal_meta.py python/weiss_rl/tests/test_runtime_logging.py python/weiss_rl/tests/test_runtime_metrics.py python/weiss_rl/tests/test_runtime_opponents.py python/weiss_rl/tests/test_runtime_outcomes.py python/weiss_rl/tests/test_runtime_pending.py python/weiss_rl/tests/test_runtime_threads.py python/weiss_rl/tests/test_runtime.py -k "runtime or teacher_labels or teacher_guidance or actor_scheduling or next_actor_batch or actor_state or runtime_unroll or runtime_batch or shared_unroll"` | Passed: 158 passed. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1100 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/runtime_teacher_labels.py`.
- Added `python/weiss_rl/tests/test_runtime_teacher_labels.py`.
- Updated `python/weiss_rl/runtime.py`.

### Behavior Changes

No intended behavior changes. Teacher-guidance activation still honors `enabled`, `off`, `warmstart_only`, warmstart limits, and current update; label arrays still use `-1` sentinels and `False` validity defaults; decoded labels still preserve family, slot, move-source, attack-type, and action-id semantics.

### Remaining Risks

- `python/weiss_rl/runtime.py` is still large at 5875 lines after the extraction.
- The large-file audit remains red: `model.py` is 4537 lines, `impala_learner.py` is 3463 lines, and `train.py` is 3234 lines.
- `_teacher_labels_from_ids()` and `_teacher_labels_from_mask()` still live in `runtime.py` because they depend on heuristic-public action selection; moving them should wait for stronger direct parity tests around that routing.

## 2026-05-11 - Runtime Deterministic Logit Extraction Checkpoint

### Scope

- Extracted deterministic logit writing for unpacked legal ids and packed legal ids from `python/weiss_rl/runtime.py` into `python/weiss_rl/runtime_deterministic_logits.py`.
- Preserved `QueueRuntime._write_deterministic_logits()` and `_write_deterministic_logits_from_packed()` as runtime wrappers.
- Added direct characterization tests for unselected-row preservation, legal-action sentinel logits, chosen-action zero logits, packed row-offset behavior, and `None` output-buffer no-ops.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime_deterministic_logits.py python/weiss_rl/tests/test_runtime.py -k "deterministic_logits or logits_out or heuristic_actor_rows"` | Passed: 4 passed, 88 deselected. |
| `uv run mypy python/weiss_rl/runtime_deterministic_logits.py python/weiss_rl/tests/test_runtime_deterministic_logits.py --show-error-codes --no-error-summary` | Passed. |
| `uv run ruff check python/weiss_rl/runtime.py python/weiss_rl/runtime_deterministic_logits.py python/weiss_rl/tests/test_runtime_deterministic_logits.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/runtime.py python/weiss_rl/runtime_deterministic_logits.py python/weiss_rl/tests/test_runtime_deterministic_logits.py` | Passed after formatting `runtime.py`. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime_deterministic_logits.py python/weiss_rl/tests/test_runtime_teacher_labels.py python/weiss_rl/tests/test_runtime_actor_scheduling.py python/weiss_rl/tests/test_runtime_types.py python/weiss_rl/tests/test_runtime_actor_state.py python/weiss_rl/tests/test_runtime_actor_models.py python/weiss_rl/tests/test_runtime_batching.py python/weiss_rl/tests/test_runtime_bootstrap.py python/weiss_rl/tests/test_runtime_collector_commands.py python/weiss_rl/tests/test_runtime_config.py python/weiss_rl/tests/test_runtime_counters.py python/weiss_rl/tests/test_runtime_debug_validation.py python/weiss_rl/tests/test_runtime_hashing.py python/weiss_rl/tests/test_runtime_ipc.py python/weiss_rl/tests/test_runtime_legal_meta.py python/weiss_rl/tests/test_runtime_logging.py python/weiss_rl/tests/test_runtime_metrics.py python/weiss_rl/tests/test_runtime_opponents.py python/weiss_rl/tests/test_runtime_outcomes.py python/weiss_rl/tests/test_runtime_pending.py python/weiss_rl/tests/test_runtime_threads.py python/weiss_rl/tests/test_runtime.py -k "runtime or deterministic_logits or logits_out or teacher_labels or teacher_guidance or actor_scheduling or next_actor_batch or actor_state or runtime_unroll or runtime_batch or shared_unroll"` | Passed: 162 passed. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1104 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/runtime_deterministic_logits.py`.
- Added `python/weiss_rl/tests/test_runtime_deterministic_logits.py`.
- Updated `python/weiss_rl/runtime.py`.

### Behavior Changes

No intended behavior changes. Deterministic logits still no-op when no output buffer is provided, preserve unselected rows, write `-1.0e9` for illegal actions, write `-100.0` for non-chosen legal actions, and write `0.0` for the chosen action. Packed legal-id offsets are still interpreted by row index.

### Remaining Risks

- `python/weiss_rl/runtime.py` is still large at 5867 lines after the extraction.
- The large-file audit remains red: `model.py` is 4537 lines, `impala_learner.py` is 3463 lines, and `train.py` is 3234 lines.
- The central policy-output application paths still live in `runtime.py`; moving them should wait for broader parity tests around model/heuristic row routing.

## 2026-05-11 - Runtime Actor Routing Extraction Checkpoint

### Scope

- Extracted focal actor row splitting between model-policy and heuristic-policy paths from `python/weiss_rl/runtime.py` into `python/weiss_rl/runtime_actor_routing.py`.
- Extracted actor policy-train-mask rules into `runtime_actor_routing.py`.
- Preserved `QueueRuntime._split_focal_actor_rows()` and `_policy_train_mask_for_actor()` as runtime wrappers.
- Added direct characterization tests for missing teacher policy errors, forced model-policy lanes, heuristic fraction endpoints, pure heuristic lane masking, and non-heuristic masking cases.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime_actor_routing.py python/weiss_rl/tests/test_runtime.py -k "split_focal_actor_rows or policy_train_mask or actor_heuristic_fraction"` | Passed: 10 passed, 83 deselected. |
| `uv run mypy python/weiss_rl/runtime_actor_routing.py python/weiss_rl/tests/test_runtime_actor_routing.py --show-error-codes --no-error-summary` | Passed. |
| `uv run ruff check python/weiss_rl/runtime.py python/weiss_rl/runtime_actor_routing.py python/weiss_rl/tests/test_runtime_actor_routing.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/runtime.py python/weiss_rl/runtime_actor_routing.py python/weiss_rl/tests/test_runtime_actor_routing.py` | Passed after formatting `runtime.py`. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime_actor_routing.py python/weiss_rl/tests/test_runtime_deterministic_logits.py python/weiss_rl/tests/test_runtime_teacher_labels.py python/weiss_rl/tests/test_runtime_actor_scheduling.py python/weiss_rl/tests/test_runtime_types.py python/weiss_rl/tests/test_runtime_actor_state.py python/weiss_rl/tests/test_runtime_actor_models.py python/weiss_rl/tests/test_runtime_batching.py python/weiss_rl/tests/test_runtime_bootstrap.py python/weiss_rl/tests/test_runtime_collector_commands.py python/weiss_rl/tests/test_runtime_config.py python/weiss_rl/tests/test_runtime_counters.py python/weiss_rl/tests/test_runtime_debug_validation.py python/weiss_rl/tests/test_runtime_hashing.py python/weiss_rl/tests/test_runtime_ipc.py python/weiss_rl/tests/test_runtime_legal_meta.py python/weiss_rl/tests/test_runtime_logging.py python/weiss_rl/tests/test_runtime_metrics.py python/weiss_rl/tests/test_runtime_opponents.py python/weiss_rl/tests/test_runtime_outcomes.py python/weiss_rl/tests/test_runtime_pending.py python/weiss_rl/tests/test_runtime_threads.py python/weiss_rl/tests/test_runtime.py -k "runtime or actor_routing or split_focal_actor_rows or policy_train_mask or actor_heuristic_fraction or deterministic_logits or logits_out or teacher_labels or teacher_guidance or actor_scheduling or next_actor_batch or actor_state or runtime_unroll or runtime_batch or shared_unroll"` | Passed: 167 passed. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1109 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/runtime_actor_routing.py`.
- Added `python/weiss_rl/tests/test_runtime_actor_routing.py`.
- Updated `python/weiss_rl/runtime.py`.

### Behavior Changes

No intended behavior changes. Focal actor rows still bypass heuristic routing outside `heuristic_public`, require the teacher policy for heuristic routing, respect forced model lanes, route all rows at heuristic fraction 1.0, route no rows at fraction 0.0, and preserve the policy-train-mask exclusion for pure heuristic lanes when `train_on_heuristic_actor_rows` is false.

### Remaining Risks

- `python/weiss_rl/runtime.py` is still large at 5861 lines after the extraction.
- The large-file audit remains red: `model.py` is 4537 lines, `impala_learner.py` is 3463 lines, and `train.py` is 3234 lines.
- The central policy-output application paths still live in `runtime.py`; moving them should wait for broader parity tests around model/heuristic row routing.

## 2026-05-11 - Runtime Heuristic Actor Output Extraction Checkpoint

### Scope

- Extracted heuristic actor output scattering for mask-backed and packed legal-action layouts from `python/weiss_rl/runtime.py` into `python/weiss_rl/runtime_heuristic_actor_outputs.py`.
- Kept heuristic-public action selection, behavior-value computation, hidden-state advancement, and packed-action debug validation inside `QueueRuntime`.
- Reused the packed-layout helper from both the single-actor heuristic path and the central packed heuristic path.
- Added direct characterization tests for mask-row legal-id extraction, deterministic heuristic logits, chosen-action scatter, zero behavior log-probability scatter, row-order preservation, dtype preservation, and optional output-buffer no-ops.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime_heuristic_actor_outputs.py python/weiss_rl/tests/test_runtime_deterministic_logits.py python/weiss_rl/tests/test_runtime.py -k "heuristic_actor_outputs or deterministic_logits or heuristic_actor_rows or central_sample_policy_rows_ids_heuristic"` | Passed: 8 passed, 88 deselected. |
| `uv run ruff check python/weiss_rl/runtime_heuristic_actor_outputs.py python/weiss_rl/tests/test_runtime_heuristic_actor_outputs.py python/weiss_rl/runtime.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/runtime_heuristic_actor_outputs.py python/weiss_rl/tests/test_runtime_heuristic_actor_outputs.py python/weiss_rl/runtime.py` | Passed after formatting `runtime.py`. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime_heuristic_actor_outputs.py python/weiss_rl/tests/test_runtime_deterministic_logits.py python/weiss_rl/tests/test_runtime_teacher_labels.py python/weiss_rl/tests/test_runtime_actor_routing.py python/weiss_rl/tests/test_runtime_actor_scheduling.py python/weiss_rl/tests/test_runtime_types.py python/weiss_rl/tests/test_runtime_actor_state.py python/weiss_rl/tests/test_runtime_actor_models.py python/weiss_rl/tests/test_runtime_batching.py python/weiss_rl/tests/test_runtime_bootstrap.py python/weiss_rl/tests/test_runtime_collector_commands.py python/weiss_rl/tests/test_runtime_config.py python/weiss_rl/tests/test_runtime_counters.py python/weiss_rl/tests/test_runtime_debug_validation.py python/weiss_rl/tests/test_runtime_hashing.py python/weiss_rl/tests/test_runtime_ipc.py python/weiss_rl/tests/test_runtime_legal_meta.py python/weiss_rl/tests/test_runtime_logging.py python/weiss_rl/tests/test_runtime_metrics.py python/weiss_rl/tests/test_runtime_opponents.py python/weiss_rl/tests/test_runtime_outcomes.py python/weiss_rl/tests/test_runtime_pending.py python/weiss_rl/tests/test_runtime_threads.py python/weiss_rl/tests/test_runtime.py -k "runtime or heuristic_actor_outputs or deterministic_logits or logits_out or heuristic_actor_rows or teacher_labels or teacher_guidance or actor_routing or split_focal_actor_rows or policy_train_mask or actor_scheduling or next_actor_batch or actor_state or runtime_unroll or runtime_batch or shared_unroll"` | Passed: 171 passed. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1113 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/runtime_heuristic_actor_outputs.py`.
- Added `python/weiss_rl/tests/test_runtime_heuristic_actor_outputs.py`.
- Updated `python/weiss_rl/runtime.py`.

### Behavior Changes

No intended behavior changes. Heuristic actor rows still write deterministic logits with `-1.0e9` illegal-action sentinels, `-100.0` non-chosen legal-action sentinels, `0.0` chosen-action logits, selected action ids, and zero behavior log-probabilities. Packed legal-id offsets are still interpreted by row index, and `None` output buffers still no-op.

### Remaining Risks

- `python/weiss_rl/runtime.py` is still large at 5863 lines after the extraction.
- The large-file audit remains red: `model.py` is 4537 lines, `impala_learner.py` is 3463 lines, and `train.py` is 3234 lines.
- The remaining runtime central collection loops are more coupled than the extracted row helpers; further movement should wait for stronger parity coverage around model/heuristic row routing and opponent overwrite behavior.

## 2026-05-11 - Model Public-Heuristic Profile Extraction Checkpoint

### Scope

- Moved public-heuristic slot preference table construction into `python/weiss_rl/model_public_heuristics.py`.
- Moved front-row public attack profile computation into `model_public_heuristics.py` and reused it from both the factorized raw heuristic path and packed public-heuristic scoring plan.
- Kept scoring priorities, action-family routing, structured candidate projection, and public-heuristic bias behavior in `python/weiss_rl/model.py`.
- Added direct characterization tests for slot preference table construction and ready-attacker/front-defender profile counts.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_model_public_heuristics.py python/weiss_rl/tests/test_model_action_plans.py python/weiss_rl/tests/test_model_action_tables.py python/weiss_rl/tests/test_model_candidate_components.py python/weiss_rl/tests/test_model_candidate_partitioning.py python/weiss_rl/tests/test_model_feature_gathering.py python/weiss_rl/tests/test_model_layers.py python/weiss_rl/tests/test_model_loading.py python/weiss_rl/tests/test_model_observation_contract.py python/weiss_rl/tests/test_model_sampling.py python/weiss_rl/tests/test_model_tensor_ops.py python/weiss_rl/tests/test_model_typed_encoder.py python/weiss_rl/tests/test_play_vs_model.py -k "model or public_heuristic or structured or factorized"` | Passed: 61 passed. |
| `uv run ruff check python/weiss_rl/model_public_heuristics.py python/weiss_rl/tests/test_model_public_heuristics.py python/weiss_rl/model.py` | Passed after import sorting. |
| `uv run ruff format --check python/weiss_rl/model_public_heuristics.py python/weiss_rl/tests/test_model_public_heuristics.py python/weiss_rl/model.py` | Passed after formatting `model.py`. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1115 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Updated `python/weiss_rl/model_public_heuristics.py`.
- Updated `python/weiss_rl/tests/test_model_public_heuristics.py`.
- Updated `python/weiss_rl/model.py`.

### Behavior Changes

No intended behavior changes. Public-heuristic slot preferences still use the same per-slot weights, unknown stage slots still receive zero preference, front-row attacker counts still require occupied and not-rested self front slots, and front defender counts still use occupied opposing front slots.

### Remaining Risks

- `python/weiss_rl/model.py` is still large at 4520 lines after the extraction.
- The large-file audit remains red: `runtime.py` is 5863 lines, `impala_learner.py` is 3463 lines, and `train.py` is 3234 lines.
- Packed public-heuristic scoring still lives mostly in `model.py`; moving it further should wait for broader parity coverage around candidate family routing and score assembly.

## 2026-05-11 - IMPALA Learner Update Bookkeeping Extraction Checkpoint

### Scope

- Extracted learner update bookkeeping helpers from `python/weiss_rl/learners/impala_learner.py` into `python/weiss_rl/learners/update_bookkeeping.py`.
- Covered timer accumulation, teacher auxiliary activation, structured metric emission cadence, throughput metric calculation, and AMP acceleration-state resolution.
- Preserved the `ImpalaLearner` methods as wrappers around the extracted helpers for behavior-sensitive call sites.
- Stabilized an existing random-initialized auxiliary-update test by seeding the tiny teacher model before asserting exact teacher-action accuracy; this matches the seed pattern already used by adjacent IMPALA tests.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_learner_update_bookkeeping.py python/weiss_rl/tests/test_impala_learner.py -k "update_bookkeeping or timing or structured_metrics or teacher_aux or mixed_precision or update"` | Passed after seeding the existing auxiliary-update exact-accuracy test: 43 passed, 15 deselected. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_learner_update_bookkeeping.py python/weiss_rl/tests/test_learner_action_logp.py python/weiss_rl/tests/test_learner_batch_fields.py python/weiss_rl/tests/test_learner_bootstrap.py python/weiss_rl/tests/test_learner_faults.py python/weiss_rl/tests/test_learner_legal_fields.py python/weiss_rl/tests/test_learner_logging.py python/weiss_rl/tests/test_learner_packed_rows.py python/weiss_rl/tests/test_learner_structured_auxiliary.py python/weiss_rl/tests/test_learner_structured_policy_metrics.py python/weiss_rl/tests/test_learner_tensor_ops.py python/weiss_rl/tests/test_learner_vtrace_diagnostics.py python/weiss_rl/tests/test_learner_vtrace_torch.py python/weiss_rl/tests/test_impala_learner.py -k "learner or impala or vtrace or structured or update_bookkeeping"` | Passed: 116 passed. |
| `uv run ruff check python/weiss_rl/learners/update_bookkeeping.py python/weiss_rl/tests/test_learner_update_bookkeeping.py python/weiss_rl/learners/impala_learner.py python/weiss_rl/tests/test_impala_learner.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/learners/update_bookkeeping.py python/weiss_rl/tests/test_learner_update_bookkeeping.py python/weiss_rl/learners/impala_learner.py python/weiss_rl/tests/test_impala_learner.py` | Passed after formatting `impala_learner.py` and `test_impala_learner.py`. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1130 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/learners/update_bookkeeping.py`.
- Added `python/weiss_rl/tests/test_learner_update_bookkeeping.py`.
- Updated `python/weiss_rl/learners/impala_learner.py`.
- Updated `python/weiss_rl/tests/test_impala_learner.py`.

### Behavior Changes

No intended production behavior changes. Timer metrics still no-op unless profiling is active and a timing dictionary exists, teacher auxiliary mode still honors `off`, `warmstart_only`, and `always`, sampled structured metrics still emit only every 10 non-auxiliary updates, throughput still floors elapsed time at `1e-6`, and mixed precision remains disabled without a CUDA-backed model parameter.

### Remaining Risks

- `python/weiss_rl/learners/impala_learner.py` is still large at 3461 lines after the extraction.
- The large-file audit remains red: `runtime.py` is 5863 lines, `model.py` is 4520 lines, and `train.py` is 3234 lines.
- The remaining learner body contains optimizer, forward, V-trace, and structured auxiliary loss flow; further movement should be guarded by parity tests around numeric outputs and metrics.

## 2026-05-11 - Training Eval-Model Clone Extraction Checkpoint

### Scope

- Extracted CPU eval-model cloning for periodic dev eval and promotion gates from `python/scripts/train.py` into `python/weiss_rl/training/dev_eval.py`.
- Preserved the `_clone_cpu_eval_model()` compatibility wrapper in `train.py`.
- Added direct characterization tests for copied CPU weights, independent parameter storage, eval mode, guidance payload/restore handoff, and missing model-config failure.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_training_dev_eval.py python/weiss_rl/tests/test_entrypoints.py -k "clone_cpu_eval_model or periodic_dev_eval or training_dev_eval"` | Passed after correcting the test to assert guidance payload/restore handoff for the base model: 14 passed, 29 deselected. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_training_dev_eval.py python/weiss_rl/tests/test_training_batches.py python/weiss_rl/tests/test_training_checkpoint_writers.py python/weiss_rl/tests/test_training_execution.py python/weiss_rl/tests/test_training_guidance.py python/weiss_rl/tests/test_training_inputs.py python/weiss_rl/tests/test_training_manifest_layout.py python/weiss_rl/tests/test_training_profiling.py python/weiss_rl/tests/test_training_promotion.py python/weiss_rl/tests/test_training_report_payloads.py python/weiss_rl/tests/test_training_run_identity.py python/weiss_rl/tests/test_training_startup.py python/weiss_rl/tests/test_training_torch_threads.py python/weiss_rl/tests/test_entrypoints.py -k "training or clone_cpu_eval_model or periodic_dev_eval or smoke or manifest_only or policy_set_selection_mode or resume"` | Passed: 74 passed, 29 deselected. |
| `uv run ruff check python/weiss_rl/training/dev_eval.py python/weiss_rl/tests/test_training_dev_eval.py python/scripts/train.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/training/dev_eval.py python/weiss_rl/tests/test_training_dev_eval.py python/scripts/train.py` | Passed after formatting `train.py`. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1132 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Updated `python/weiss_rl/training/dev_eval.py`.
- Updated `python/weiss_rl/tests/test_training_dev_eval.py`.
- Updated `python/scripts/train.py`.

### Behavior Changes

No intended behavior changes. Eval models are still rebuilt from the locked model config, moved to CPU, loaded from cloned learner weights, restored with the learner guidance payload, and switched to eval mode before use.

### Remaining Risks

- `python/scripts/train.py` is still large at 3228 lines after the extraction.
- The large-file audit remains red: `runtime.py` is 5863 lines, `model.py` is 4520 lines, and `impala_learner.py` is 3461 lines.
- Periodic dev-eval and promotion-gate runner classes still live in `train.py`; moving them should wait for parity tests around deterministic seed use, legal-id fallback, and artifact paths.

## 2026-05-11 - Periodic Dev-Eval Runner Extraction Checkpoint

### Scope

- Extracted the periodic dev-eval runner from `python/scripts/train.py` into `python/weiss_rl/training/dev_eval_runner.py`.
- Preserved `train.py`'s `_PeriodicDevEvalRunner` compatibility class as a thin subclass that injects the existing `_build_ids_eval_env` wrapper, so script-level monkeypatch tests still exercise the public training surface.
- Left deterministic seed helpers, legal-id row slicing, CPU eval-model cloning, and artifact persistence in their existing training helper modules.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_snapshot_registry.py -k "periodic_dev_eval_runner" python/weiss_rl/tests/test_entrypoints.py -k "periodic_dev_eval"` | Passed: 4 passed, 68 deselected, 14 dependency warnings. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_training_dev_eval.py python/weiss_rl/tests/test_training_batches.py python/weiss_rl/tests/test_training_checkpoint_writers.py python/weiss_rl/tests/test_training_execution.py python/weiss_rl/tests/test_training_guidance.py python/weiss_rl/tests/test_training_inputs.py python/weiss_rl/tests/test_training_manifest_layout.py python/weiss_rl/tests/test_training_profiling.py python/weiss_rl/tests/test_training_promotion.py python/weiss_rl/tests/test_training_report_payloads.py python/weiss_rl/tests/test_training_run_identity.py python/weiss_rl/tests/test_training_startup.py python/weiss_rl/tests/test_training_torch_threads.py python/weiss_rl/tests/test_snapshot_registry.py python/weiss_rl/tests/test_entrypoints.py -k "training or periodic_dev_eval_runner or periodic_dev_eval or clone_cpu_eval_model or smoke or manifest_only or policy_set_selection_mode or resume"` | Passed: 80 passed, 64 deselected, 14 dependency warnings. |
| `uv run ruff check python/weiss_rl/training/dev_eval_runner.py python/scripts/train.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/training/dev_eval_runner.py python/scripts/train.py` | Passed after formatting `train.py`. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1132 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/training/dev_eval_runner.py`.
- Updated `python/scripts/train.py`.

### Behavior Changes

No intended behavior changes. Periodic dev eval still resets with the scheduled episode seed, uses the deterministic per-seat PCG seed helper, samples model actions with `scoring_mode="learner"`, falls back through heuristic/opponent/random-legal paths as before, aborts on engine faults, and closes the environment in a `finally` block.

### Remaining Risks

- `python/scripts/train.py` is still large at 3099 lines after the extraction.
- The large-file audit remains red: `runtime.py` is 5863 lines, `model.py` is 4520 lines, and `impala_learner.py` is 3461 lines.
- The promotion-gate runner still lives in `train.py`; it should be moved only with direct parity coverage around deterministic reset/action behavior and artifact path handling.

## 2026-05-11 - Promotion-Gate Runner Extraction Checkpoint

### Scope

- Extracted the promotion-gate runner from `python/scripts/train.py` into `python/weiss_rl/training/promotion_gate_runner.py`.
- Preserved `train.py`'s `_PromotionGateRunner` compatibility class as a thin subclass that injects the existing `_build_ids_eval_env` wrapper and random-legal policy id.
- Kept promotion anchor resolution, policy loading, promotion-gate orchestration, and checkpoint/snapshot publishing unchanged in `train.py`.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_snapshot_registry.py -k "promotion_gate_runner"` | Passed: 2 passed, 39 deselected, 14 dependency warnings. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_training_dev_eval.py python/weiss_rl/tests/test_training_batches.py python/weiss_rl/tests/test_training_checkpoint_writers.py python/weiss_rl/tests/test_training_execution.py python/weiss_rl/tests/test_training_guidance.py python/weiss_rl/tests/test_training_inputs.py python/weiss_rl/tests/test_training_manifest_layout.py python/weiss_rl/tests/test_training_profiling.py python/weiss_rl/tests/test_training_promotion.py python/weiss_rl/tests/test_training_report_payloads.py python/weiss_rl/tests/test_training_run_identity.py python/weiss_rl/tests/test_training_startup.py python/weiss_rl/tests/test_training_torch_threads.py python/weiss_rl/tests/test_snapshot_registry.py python/weiss_rl/tests/test_promotion_gate.py python/weiss_rl/tests/test_entrypoints.py -k "training or promotion_gate_runner or promotion_gate or periodic_dev_eval_runner or periodic_dev_eval or clone_cpu_eval_model or smoke or manifest_only or policy_set_selection_mode or resume"` | Passed: 91 passed, 59 deselected, 14 dependency warnings. |
| `uv run ruff check python/weiss_rl/training/promotion_gate_runner.py python/scripts/train.py` | Passed after removing imports made unused by the extraction. |
| `uv run ruff format --check python/weiss_rl/training/promotion_gate_runner.py python/scripts/train.py` | Passed after formatting `train.py`. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1132 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/training/promotion_gate_runner.py`.
- Updated `python/scripts/train.py`.

### Behavior Changes

No intended behavior changes. Promotion-gate games still reset with the scheduled episode seed, use deterministic per-seat PCG seeds, sample model actions with `scoring_mode="learner"`, support heuristic anchors, fall back only for the random-legal anchor, abort on engine faults, and close the environment in a `finally` block.

### Remaining Risks

- `python/scripts/train.py` is still large at 2988 lines after the extraction, though it is now below 3000 lines.
- The large-file audit remains red: `runtime.py` is 5863 lines, `model.py` is 4520 lines, and `impala_learner.py` is 3461 lines.
- The main training loop, promotion orchestration, and checkpoint/snapshot publishing still live in `train.py`; further movement should be guarded by parity tests around checkpoint publishing and finalization.

## 2026-05-11 - Periodic Dev-Eval Opponent Resolution Extraction Checkpoint

### Scope

- Extracted periodic dev-eval opponent resolution from `python/scripts/train.py` into `python/weiss_rl/training/dev_eval_opponents.py`.
- Preserved the `_periodic_dev_eval_opponents()` compatibility wrapper in `train.py`.
- Kept snapshot model loading and heuristic-public policy construction as explicit dependencies passed through the wrapper, preserving existing script-level monkeypatch tests.
- Left periodic dev-eval execution, seed usage payloads, matchup artifacts, summary persistence, and stall-monitor updates unchanged in `train.py`.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run mypy python/weiss_rl/training/dev_eval_opponents.py --show-error-codes --no-error-summary` | Passed after replacing the temporary lambda with a typed heuristic-policy builder protocol. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_train_stall_monitor.py -k "periodic_dev_eval_opponents"` | Passed: 3 passed, 12 deselected, 14 dependency warnings. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_train_stall_monitor.py python/weiss_rl/tests/test_training_dev_eval.py python/weiss_rl/tests/test_snapshot_registry.py -k "periodic_dev_eval_opponents or periodic_dev_eval or promotion_gate_runner or run_snapshot_promotion_gate"` | Passed: 16 passed, 52 deselected, 14 dependency warnings. |
| `uv run ruff check python/weiss_rl/training/dev_eval_opponents.py python/scripts/train.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/training/dev_eval_opponents.py python/scripts/train.py` | Passed after formatting `train.py`. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1132 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/training/dev_eval_opponents.py`.
- Updated `python/scripts/train.py`.

### Behavior Changes

No intended behavior changes. Periodic dev eval still loads the snapshot registry from the run artifact layout, resolves required/optional/symbolic promotion anchors through the same promotion helpers, uses random-legal and heuristic-public anchors as before, loads snapshot eval models with the same stack/spec arguments, skips unavailable optional anchors, and raises for missing required anchors.

### Remaining Risks

- `python/scripts/train.py` is still large at 2930 lines after the extraction.
- The large-file audit remains red: `runtime.py` is 5863 lines, `model.py` is 4520 lines, and `impala_learner.py` is 3461 lines.
- Periodic dev-eval execution and promotion-gate orchestration still live in `train.py`; future movement should preserve seed usage payloads, artifact paths, summary aggregation, checkpoint publishing, and monkeypatchable script dependencies.

## 2026-05-11 - Runtime Hard-Negative Opponent Selection Extraction Checkpoint

### Scope

- Extracted hard-negative opponent selection from `QueueRuntime._select_hard_negative_ids()` into `python/weiss_rl/runtime_opponents.py`.
- Preserved the `QueueRuntime._select_hard_negative_ids()` compatibility wrapper.
- Added direct characterization tests for sample-count filtering, max-win-rate filtering, registry-update tie breaking, and empty/missing prerequisite behavior.
- Left PFSP opponent sampling, opponent-pool refresh, outcome tracking, actor role assignment, and rollout collection unchanged.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime_opponents.py -k "hard_negative or opponent"` | Passed: 8 passed. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime_opponents.py python/weiss_rl/tests/test_runtime.py -k "hard_negative or opponent or pfsp or refresh_opponent_pool"` | Passed: 33 passed, 63 deselected. |
| `uv run ruff check python/weiss_rl/runtime_opponents.py python/weiss_rl/runtime.py python/weiss_rl/tests/test_runtime_opponents.py` | Passed after import sorting in `test_runtime_opponents.py`. |
| `uv run ruff format --check python/weiss_rl/runtime_opponents.py python/weiss_rl/runtime.py python/weiss_rl/tests/test_runtime_opponents.py` | Passed after formatting `test_runtime_opponents.py`. |
| `uv run mypy python/weiss_rl/runtime_opponents.py python/weiss_rl/tests/test_runtime_opponents.py --show-error-codes --no-error-summary` | Passed. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1134 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Updated `python/weiss_rl/runtime_opponents.py`.
- Updated `python/weiss_rl/runtime.py`.
- Updated `python/weiss_rl/tests/test_runtime_opponents.py`.

### Behavior Changes

No intended behavior changes. Hard-negative selection still returns no policies without candidate ids, league config, or outcomes; still uses `league.sampling.hard_negative_min_samples` and `hard_negative_max_win_rate`; still loads registry update numbers only when a registry path exists; still sorts by win rate, newer snapshot update, then policy id.

### Remaining Risks

- `python/weiss_rl/runtime.py` is still large at 5850 lines after the extraction.
- The large-file audit remains red: `model.py` is 4520 lines, `impala_learner.py` is 3461 lines, and `train.py` is 2930 lines.
- Follow-up candidates from read-only audits are promising but should stay separate checkpoints: structured-warmstart source mix in runtime, factorized legality plan construction in model, and forward-time-major dispatch in the IMPALA learner.

## 2026-05-11 - Model Factorized Legality Plan Extraction Checkpoint

### Scope

- Extracted factorized structured-action legality plan construction from `_StructuredLegalActionHead._build_factorized_legality_plan()` into `python/weiss_rl/model_action_plans.py`.
- Preserved `_StructuredLegalActionHead._build_factorized_legality_plan()` as a compatibility method that delegates with the existing family and argument lookup buffers.
- Added direct characterization for packed row-order preservation with family-interleaved, unsorted action ids.
- Left factorized log-probability computation, packed candidate scoring, public-heuristic biasing, recurrent forward paths, and sampling unchanged.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_model_action_plans.py` | Passed: 4 passed. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_model_action_plans.py python/weiss_rl/tests/test_contracts.py -k "factorized or structured_legal_policy_value_model or action_plan"` | Passed: 18 passed, 36 deselected. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_impala_learner.py -k "factorized or packed"` | Passed: 17 passed, 25 deselected. |
| `uv run ruff check python/weiss_rl/model_action_plans.py python/weiss_rl/model.py python/weiss_rl/tests/test_model_action_plans.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/model_action_plans.py python/weiss_rl/model.py python/weiss_rl/tests/test_model_action_plans.py` | Passed after formatting `model.py`. |
| `uv run mypy python/weiss_rl/model_action_plans.py python/weiss_rl/tests/test_model_action_plans.py --show-error-codes --no-error-summary` | Passed. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1135 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Updated `python/weiss_rl/model_action_plans.py`.
- Updated `python/weiss_rl/model.py`.
- Updated `python/weiss_rl/tests/test_model_action_plans.py`.

### Behavior Changes

No intended behavior changes. Factorized legality planning still requires packed legal ids and offsets, derives row ids from packed offsets, keeps action ids as the canonical source for family/argument lookup, uses `torch.unique_consecutive()` for row grouping, and does not sort or normalize candidate order.

### Remaining Risks

- `python/weiss_rl/model.py` is still large at 4472 lines after the extraction.
- The large-file audit remains red: `runtime.py` is 5850 lines, `impala_learner.py` is 3461 lines, and `train.py` is 2930 lines.
- Avoid moving packed candidate scoring or public-heuristic scoring plans wholesale until tests can pin candidate-order restoration and actor/learner scoring-mode equivalence.

## 2026-05-11 - Runtime Structured-Warmstart Source-Mix Extraction Checkpoint

### Scope

- Extracted structured-warmstart fixed-opponent source-mix handling from `QueueRuntime` into `python/weiss_rl/runtime_structured_warmstart.py`.
- Preserved `QueueRuntime._set_process_collector_fixed_opponents()`, `_restore_process_collector_fixed_opponents()`, and `structured_warmstart_source_mix()` as compatibility wrappers.
- Added characterization that actor fixed slots, forced policy ids, and an inserted teacher heuristic are restored when the context body raises.
- Left central/runtime collection, policy-output overwrites, fixed-opponent routing, process collector command handling, and model state IPC format unchanged.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime.py -k "structured_warmstart_source_mix or disable_mirror_policy_fusion"` | Passed after fixing the IPC helper import name: 4 passed, 85 deselected. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_ipc.py -k "structured_warmstart_source_mix or disable_mirror_policy_fusion or ipc or state_dict"` | Passed: 7 passed, 85 deselected. |
| `uv run ruff check python/weiss_rl/runtime.py python/weiss_rl/runtime_structured_warmstart.py python/weiss_rl/tests/test_runtime.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/runtime.py python/weiss_rl/runtime_structured_warmstart.py python/weiss_rl/tests/test_runtime.py` | Passed after formatting `runtime.py` and `test_runtime.py`. |
| `uv run mypy python/weiss_rl/runtime_structured_warmstart.py --show-error-codes --no-error-summary` | Passed. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed after adding an explicit non-None assertion in the new test. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1136 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/runtime_structured_warmstart.py`.
- Updated `python/weiss_rl/runtime.py`.
- Updated `python/weiss_rl/tests/test_runtime.py`.

### Behavior Changes

No intended behavior changes. Structured warmstart still inserts the teacher heuristic only when absent, balances self-play/B1/B2 source slots with the same ceiling allocation, sends the same process-collector fixed-opponent payloads including serialized B1 state when present, restores process collectors through the same `restore_defaults` command, restores actor fixed slots and forced policy ids in `finally`, and removes only the heuristic policy it inserted.

### Remaining Risks

- `python/weiss_rl/runtime.py` is still large at 5770 lines after the extraction.
- The large-file audit remains red: `model.py` is 4472 lines, `impala_learner.py` is 3461 lines, and `train.py` is 2930 lines.
- The remaining runtime central collection and fixed-opponent overwrite paths are much more coupled to rollout semantics and should not be moved without broader parity tests.

## 2026-05-11 - IMPALA Forward-Time-Major Extraction Checkpoint

### Scope

- Extracted IMPALA time-major model-forward dispatch from `ImpalaLearner._forward_time_major()` into `python/weiss_rl/learners/forward_time_major.py`.
- Preserved `_ForwardTimeMajorResult` as a compatibility alias and kept `ImpalaLearner._forward_time_major()` as the wrapper that supplies learner-specific callbacks.
- Moved timestep legality slicing into the new module and kept `_time_step_legal_actions()` as a compatibility wrapper.
- Added direct characterization that the legacy no-seat recurrent path matches manual per-step model rollout.
- Left optimizer stepping, V-trace, policy/value losses, teacher auxiliary losses, factorized learner evaluation, and metric aggregation unchanged.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_impala_learner.py -k "forward_time_major or restricts_packed_policy_scoring_to_train_rows or uses_compiled_forward_model_when_provided"` | Passed after restoring still-needed `time`/`dataclass` imports: 6 passed, 37 deselected. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_impala_learner.py python/weiss_rl/tests/test_vtrace.py -k "forward_time_major or update_reduces_loss_on_seat_aware_batches or factorized or packed"` | Passed: 22 passed, 38 deselected. |
| `uv run ruff check python/weiss_rl/learners/forward_time_major.py python/weiss_rl/learners/impala_learner.py python/weiss_rl/tests/test_impala_learner.py` | Passed after import sorting. |
| `uv run ruff format --check python/weiss_rl/learners/forward_time_major.py python/weiss_rl/learners/impala_learner.py python/weiss_rl/tests/test_impala_learner.py` | Passed after formatting the touched files. |
| `uv run mypy python/weiss_rl/learners/forward_time_major.py python/weiss_rl/learners/impala_learner.py --show-error-codes --no-error-summary` | Passed. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1137 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/learners/forward_time_major.py`.
- Updated `python/weiss_rl/learners/impala_learner.py`.
- Updated `python/weiss_rl/tests/test_impala_learner.py`.

### Behavior Changes

No intended behavior changes. The helper still selects the compiled model when available, validates 3D time-major observations, prepares acting-seat and hidden-state tensors through the learner callbacks, requires packed ids/offsets/meta for structured candidate scoring, preserves sequence/trunk fast paths, restricts packed scoring to train rows when requested, records the same timing and packed-candidate metrics, and falls back to per-step legacy or seat-aware rollout when fast paths are unavailable.

### Remaining Risks

- `python/weiss_rl/learners/impala_learner.py` is still large at 3254 lines after the extraction.
- The large-file audit remains red: `runtime.py` is 5770 lines, `model.py` is 4472 lines, and `train.py` is 2930 lines.
- The remaining learner body now concentrates loss orchestration, V-trace integration, structured auxiliary losses, numeric fault handling, and logging; further movement should start with characterization rather than broad code motion.

## 2026-05-11 - Runtime Heuristic Fast-Path Predicate Extraction Checkpoint

### Scope

- Extracted simulator-native fixed-opponent availability, fixed-opponent heuristic-only checks, and all-heuristic IDs fast/native-rollout gating from `QueueRuntime` into `python/weiss_rl/runtime_heuristic_fast_path.py`.
- Preserved `QueueRuntime._simulator_native_fixed_opponent_available()`, `_actor_fixed_opponents_all_heuristic_public()`, `_can_collect_all_heuristic_ids_fast()`, and `_can_collect_all_heuristic_ids_native_rollout()` as compatibility wrappers.
- Added direct predicate characterization for backend/pool hook requirements, inactive fixed-opponent slots, all fast-path gate conditions, non-heuristic opponent rejection, and stateless value-free native rollout requirements.
- Left heuristic action selection, native simulator rollout execution, hidden-state updates, teacher labels, fixed-opponent assignment, and rollout construction unchanged.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime_heuristic_fast_path.py python/weiss_rl/tests/test_runtime.py -k "heuristic_ids_fast or heuristic_ids_native_rollout or heuristic_fast_path"` | Passed: 8 passed, 85 deselected. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime_heuristic_fast_path.py python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_heuristic_actor_outputs.py -k "heuristic_ids_fast or heuristic_ids_native_rollout or heuristic_fast_path or heuristic_actor"` | Passed: 15 passed, 82 deselected. |
| `uv run ruff check python/weiss_rl/runtime.py python/weiss_rl/runtime_heuristic_fast_path.py python/weiss_rl/tests/test_runtime_heuristic_fast_path.py` | Passed after sorting imports in `runtime.py`. |
| `uv run ruff format --check python/weiss_rl/runtime.py python/weiss_rl/runtime_heuristic_fast_path.py python/weiss_rl/tests/test_runtime_heuristic_fast_path.py` | Passed. |
| `uv run mypy python/weiss_rl/runtime.py python/weiss_rl/runtime_heuristic_fast_path.py python/weiss_rl/tests/test_runtime_heuristic_fast_path.py --show-error-codes --no-error-summary` | Passed. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1141 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/runtime_heuristic_fast_path.py`.
- Added `python/weiss_rl/tests/test_runtime_heuristic_fast_path.py`.
- Updated `python/weiss_rl/runtime.py`.

### Behavior Changes

No intended behavior changes. The fast-path predicates still require the `i16_legal_ids` layout, `heuristic_public` actor backend, full actor heuristic fraction, simulator-native fixed-opponent backend with the pool hook, an active teacher policy, league config, full heuristic-public mix, all active fixed slots resolving to `heuristic_public`, and all assigned opponent ids equal to `heuristic_public`. The native rollout predicate still additionally requires native rollout enablement, no actor behavior-value requirement, no heuristic hidden-state tracking, and both simulator rollout/reset hooks.

### Remaining Risks

- `python/weiss_rl/runtime.py` is still large at 5757 lines after the extraction.
- The large-file audit remains red: `model.py` is 4472 lines, `impala_learner.py` is 3254 lines, and `train.py` is 2930 lines.
- The remaining runtime central collection, native rollout, and policy-output paths are much more coupled to rollout semantics and should not be moved without broader parity tests.

## 2026-05-11 - Model Public-Heuristic Raw Scoring Extraction Checkpoint

### Scope

- Extracted structured-model play, move, and attack public-heuristic raw scoring formulas from `_StructuredLegalActionHead` into `python/weiss_rl/model_public_heuristics.py`.
- Preserved `_StructuredLegalActionHead._play_public_heuristic_raw()`, `_move_public_heuristic_raw()`, and `_attack_public_heuristic_raw()` as compatibility wrappers that pass the existing slot-preference tensor.
- Added direct formula characterization for open/occupied play slots, move validity and slot-improvement bonuses, and attack-type/power/soul/occupancy scoring.
- Left packed candidate grouping, public-heuristic bias application, candidate-order restoration, scoring-mode resolution, and model public API behavior unchanged.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_model_public_heuristics.py python/weiss_rl/tests/test_heuristic_public.py -k "public_heuristic_raw or public_heuristic or heuristic_public"` | Passed: 38 passed. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_model_public_heuristics.py python/weiss_rl/tests/test_heuristic_public.py python/weiss_rl/tests/test_contracts.py -k "public_heuristic or structured_legal_policy_value_model or action_plan"` | Passed: 25 passed, 63 deselected. |
| `uv run ruff check python/weiss_rl/model.py python/weiss_rl/model_public_heuristics.py python/weiss_rl/tests/test_model_public_heuristics.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/model.py python/weiss_rl/model_public_heuristics.py python/weiss_rl/tests/test_model_public_heuristics.py` | Passed after formatting `model.py`. |
| `uv run mypy python/weiss_rl/model_public_heuristics.py python/weiss_rl/model.py python/weiss_rl/tests/test_model_public_heuristics.py --show-error-codes --no-error-summary` | Passed. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1144 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Updated `python/weiss_rl/model_public_heuristics.py`.
- Updated `python/weiss_rl/model.py`.
- Updated `python/weiss_rl/tests/test_model_public_heuristics.py`.

### Behavior Changes

No intended behavior changes. Play raw scores still reward open preferred front/back stage slots and reject occupied targets. Move raw scores still require an occupied source and open target, preserve slot-preference improvement, back-to-front bonus, and center bonus. Attack raw scores still require an occupied attacker and preserve direct/frontal/side attack-type bonuses, slot preference, power term, and effective-soul term.

### Remaining Risks

- `python/weiss_rl/model.py` is still large at 4426 lines after the extraction.
- The large-file audit remains red: `runtime.py` is 5757 lines, `impala_learner.py` is 3254 lines, and `train.py` is 2930 lines.
- The remaining structured model packed scoring plans and candidate grouping paths are behavior-sensitive and should not be moved without broader parity tests around candidate ordering and scoring-mode equivalence.

## 2026-05-11 - Model Family-Gated Public-Heuristic Raw Scoring Extraction Checkpoint

### Scope

- Extracted structured-model slot-family, hand-index, generic-index, and default public-heuristic raw scoring formulas into `python/weiss_rl/model_public_heuristics.py`.
- Preserved `_StructuredLegalActionHead._slot_family_public_heuristic_raw()`, `_hand_public_heuristic_raw()`, `_index_public_heuristic_raw()`, and `_default_public_heuristic_raw()` as compatibility wrappers that pass existing family ids and slot-preference tensors.
- Added direct formula characterization for encore slot families, hand tactical families, index/paging families, and default confirm/pass families.
- Left candidate component resolution, packed candidate grouping, public-heuristic bias application, candidate-order restoration, and scoring-mode resolution unchanged.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_model_public_heuristics.py python/weiss_rl/tests/test_heuristic_public.py -k "public_heuristic_raw or public_heuristic or heuristic_public"` | Passed after adding a missed wrapper import: 42 passed. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_model_public_heuristics.py python/weiss_rl/tests/test_heuristic_public.py python/weiss_rl/tests/test_contracts.py -k "public_heuristic or structured_legal_policy_value_model or action_plan"` | Passed: 29 passed, 63 deselected. |
| `uv run ruff check python/weiss_rl/model.py python/weiss_rl/model_public_heuristics.py python/weiss_rl/tests/test_model_public_heuristics.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/model.py python/weiss_rl/model_public_heuristics.py python/weiss_rl/tests/test_model_public_heuristics.py` | Passed after formatting `model.py`. |
| `uv run mypy python/weiss_rl/model_public_heuristics.py python/weiss_rl/model.py python/weiss_rl/tests/test_model_public_heuristics.py --show-error-codes --no-error-summary` | Passed after adding the missed wrapper import. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1148 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Updated `python/weiss_rl/model_public_heuristics.py`.
- Updated `python/weiss_rl/model.py`.
- Updated `python/weiss_rl/tests/test_model_public_heuristics.py`.

### Behavior Changes

No intended behavior changes. Slot-family raw scores still gate on encore pay/decline family ids and preserve slot preference plus power terms. Hand raw scores still preserve climax, clock-from-hand, main-event, and mulligan-selection family formulas. Index raw scores still preserve choice, level-up, trigger-order, next-page, and previous-page formulas. Default raw scores still preserve mulligan-confirm and pass family scores.

### Remaining Risks

- `python/weiss_rl/model.py` is still large at 4365 lines after the extraction.
- The large-file audit remains red: `runtime.py` is 5757 lines, `impala_learner.py` is 3254 lines, and `train.py` is 2930 lines.
- The remaining structured model public-heuristic plan code is now more tightly coupled to candidate grouping and row restoration; further movement should start with parity characterization rather than broad extraction.

## 2026-05-11 - Training Learner Compile Extraction Checkpoint

### Scope

- Extracted learner compile selection from `python/scripts/train.py` into `python/weiss_rl/training/learner_compile.py`.
- Preserved `_maybe_compile_learner_model()` in `train.py` as a compatibility wrapper.
- Added branch characterization for disabled compile, non-CUDA skip messaging, structured trunk compile success, structured trunk compile failure, structured no-hook skip messaging, and plain `torch.compile` dispatch.
- Left model construction, learner construction, checkpoint loading, runtime construction, and training-loop behavior unchanged.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_training_learner_compile.py` | Passed: 6 passed. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_training_learner_compile.py python/weiss_rl/tests/test_training_guidance.py python/weiss_rl/tests/test_training_execution.py python/weiss_rl/tests/test_entrypoints.py -k "compile_learner or guidance or execution_settings or train_cli_defaults"` | Passed: 12 passed, 31 deselected. |
| `uv run ruff check python/weiss_rl/training/learner_compile.py python/weiss_rl/tests/test_training_learner_compile.py python/scripts/train.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/training/learner_compile.py python/weiss_rl/tests/test_training_learner_compile.py python/scripts/train.py` | Passed after formatting `train.py`, the helper, and tests. |
| `uv run mypy python/weiss_rl/training/learner_compile.py python/weiss_rl/tests/test_training_learner_compile.py --show-error-codes --no-error-summary` | Passed. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1154 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `python/weiss_rl/training/learner_compile.py`.
- Added `python/weiss_rl/tests/test_training_learner_compile.py`.
- Updated `python/scripts/train.py`.

### Behavior Changes

No intended behavior changes. Learner compilation still returns `None` when disabled, skips with the same note on non-CUDA devices, prefers structured trunk compilation when the model supports legal candidate scoring and exposes the hook, reports structured hook failures without falling back to full `torch.compile`, skips structured legal scoring models without a trunk hook, and uses `torch.compile(..., mode="reduce-overhead")` for plain models.

### Remaining Risks

- `python/scripts/train.py` is still large at 2908 lines after the extraction.
- Direct `mypy python/scripts/train.py` still reports known script-level protocol/cast debt outside the package mypy gate; this checkpoint validated the new helper directly and kept the full package mypy gate clean.
- The remaining train script body still owns substantial orchestration around warmstart, snapshot import, promotion, periodic dev eval, and the main training loop.

## 2026-05-11 - IMPALA Public-Heuristic Profile Selection Extraction Checkpoint

### Scope

- Extracted active/selected public-heuristic teacher profile selection from `ImpalaLearner` into `python/weiss_rl/learners/structured_auxiliary.py`.
- Extracted packed public-heuristic profile-logit mixing from `ImpalaLearner` into `python/weiss_rl/learners/structured_auxiliary.py`.
- Preserved `ImpalaLearner._active_teacher_public_heuristic_profiles()` as a compatibility wrapper and kept model scoring calls in `ImpalaLearner`.
- Added direct tests for default profiles, end-update fallback before cycle selection, cycle/mixture mode selection, row-wise mixture normalization over uneven packed offsets, and empty/single-profile cases.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_learner_structured_auxiliary.py python/weiss_rl/tests/test_impala_learner.py -k "public_heuristic_profiles or public_heuristic_profile or mix_public_heuristic or public_heuristic"` | Passed: 14 passed, 42 deselected. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_learner_structured_auxiliary.py python/weiss_rl/tests/test_impala_learner.py python/weiss_rl/tests/test_vtrace.py -k "public_heuristic or factorized or packed or update_reduces_loss"` | Passed: 36 passed, 37 deselected. |
| `uv run ruff check python/weiss_rl/learners/structured_auxiliary.py python/weiss_rl/learners/impala_learner.py python/weiss_rl/tests/test_learner_structured_auxiliary.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/learners/structured_auxiliary.py python/weiss_rl/learners/impala_learner.py python/weiss_rl/tests/test_learner_structured_auxiliary.py` | Passed after formatting `impala_learner.py` and the structured-auxiliary tests. |
| `uv run mypy python/weiss_rl/learners/structured_auxiliary.py python/weiss_rl/learners/impala_learner.py python/weiss_rl/tests/test_learner_structured_auxiliary.py --show-error-codes --no-error-summary` | Passed. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1158 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Updated `python/weiss_rl/learners/structured_auxiliary.py`.
- Updated `python/weiss_rl/learners/impala_learner.py`.
- Updated `python/weiss_rl/tests/test_learner_structured_auxiliary.py`.

### Behavior Changes

No intended behavior changes. Empty configured profiles still default to `("base",)`. End-update fallback still returns the first configured profile before cycle-mode selection. Cycle mode still selects `profiles[update_count % len(profiles)]` while mixture mode keeps all active profiles. Packed profile mixing still normalizes each profile per packed row, averages profile log-probabilities, and rescales by the configured temperature.

### Remaining Risks

- `python/weiss_rl/learners/impala_learner.py` is still large at 3238 lines after the extraction.
- The runtime PFSP/opponent sampling candidate remains useful but RNG-sensitive; it should start with direct parity tests for draw order and PFSP counter accounting.
- The remaining learner body is mostly loss orchestration, V-trace integration, numeric fault handling, and logging; further movement should start with characterization rather than broad extraction.

## 2026-05-11 - Runtime PFSP Opponent Sampling Extraction Checkpoint

### Scope

- Extracted PFSP/opponent policy-id sampling from `QueueRuntime._sample_opponent_policy_ids()` into `python/weiss_rl/runtime_opponents.py`.
- Extracted warmup snapshot sampling from `QueueRuntime._sample_warmup_snapshot_policy_ids()` into `python/weiss_rl/runtime_opponents.py`.
- Added `OpponentSamplingResult` so sampled policy ids and every PFSP counter bucket are returned explicitly.
- Preserved the runtime methods as state adapters that copy result counters back into the existing `_pfsp_last_*` fields.
- Added seeded direct tests for no-league/empty cases, PFSP-ready hard/champion/recent bucket draw order, pre-PFSP mixed heuristic/variant/B1/warmup/mirror draw order, and warmup-snapshot counter accounting.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime_opponents.py python/weiss_rl/tests/test_runtime.py -k "sample_runtime_opponent or sample_warmup_snapshot or sample_opponent_policy_ids"` | Passed after fixing a wrapper adapter issue around partially mocked runtime attributes: 11 passed, 90 deselected. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime_opponents.py python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_metrics.py -k "opponent or pfsp or sample_opponent_policy_ids or runtime_metrics"` | Passed: 41 passed, 63 deselected. |
| `uv run ruff check python/weiss_rl/runtime_opponents.py python/weiss_rl/runtime.py python/weiss_rl/tests/test_runtime_opponents.py` | Passed after import sorting. |
| `uv run ruff format --check python/weiss_rl/runtime_opponents.py python/weiss_rl/runtime.py python/weiss_rl/tests/test_runtime_opponents.py` | Passed after formatting `runtime.py` and `runtime_opponents.py`. |
| `uv run mypy python/weiss_rl/runtime_opponents.py python/weiss_rl/runtime.py python/weiss_rl/tests/test_runtime_opponents.py --show-error-codes --no-error-summary` | Passed after annotating direct-test defaults. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1162 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Updated `python/weiss_rl/runtime_opponents.py`.
- Updated `python/weiss_rl/runtime.py`.
- Updated `python/weiss_rl/tests/test_runtime_opponents.py`.

### Behavior Changes

No intended behavior changes. The helper preserves the old group construction order, the first `rng.choice(...)` over groups, per-group draw order, direct fixed-policy replication for mirror/B2/B1 groups, heuristic-variant random draw behavior, PFSP snapshot sampling calls, and the `_pfsp_last_*` counter definitions. The runtime wrapper also preserves the old behavior for partially initialized test/runtime objects by not forcing unrelated opponent-id attributes to exist.

### Remaining Risks

- `python/weiss_rl/runtime.py` is still large at 5659 lines after the extraction.
- The remaining central collection and native rollout paths are behavior-sensitive and should not be moved without much stronger parity coverage.
- Seeded helper tests now pin PFSP draw order for representative mixed buckets, but they are not a full stochastic proof for every configuration combination.

## 2026-05-11 - Runtime Opponent Policy-ID Bookkeeping Extraction Checkpoint

### Scope

- Extracted active assigned opponent policy-id collection into `python/weiss_rl/runtime_opponents.py`.
- Extracted configured fixed/resident opponent policy-id bookkeeping into `python/weiss_rl/runtime_opponents.py`.
- Preserved `QueueRuntime._active_assigned_opponent_policy_ids()`, `_configured_fixed_opponent_policy_ids()`, and `_configured_resident_opponent_policy_ids()` as compatibility wrappers.
- Added direct tests for mirror/empty filtering, first-seen deduplication, fixed-opponent ordering, heuristic-public availability gating, variant inclusion, and B1 deduplication.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime_opponents.py -k "active_assigned or configured_fixed or configured_resident or active_mix or fixed_opponent"` | Passed: 5 passed, 98 deselected. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime_opponents.py python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_actor_state.py -k "opponent or fixed_opponent or resident or active_assigned or refresh_opponent_pool or actor_state"` | Passed: 43 passed, 63 deselected. |
| `uv run ruff check python/weiss_rl/runtime_opponents.py python/weiss_rl/runtime.py python/weiss_rl/tests/test_runtime_opponents.py` | Passed after import sorting. |
| `uv run ruff format --check python/weiss_rl/runtime_opponents.py python/weiss_rl/runtime.py python/weiss_rl/tests/test_runtime_opponents.py` | Passed after formatting. |
| `uv run mypy python/weiss_rl/runtime_opponents.py python/weiss_rl/runtime.py python/weiss_rl/tests/test_runtime_opponents.py --show-error-codes --no-error-summary` | Passed. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1164 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Updated `python/weiss_rl/runtime_opponents.py`.
- Updated `python/weiss_rl/runtime.py`.
- Updated `python/weiss_rl/tests/test_runtime_opponents.py`.

### Behavior Changes

No intended behavior changes. Active assigned policy ids still skip missing actor arrays, empty ids, and mirror ids while preserving first-seen order. Fixed opponent ids still include B2 only when reserved and available as a heuristic policy and include B1 when reserved. Resident ids still start from fixed ids, then include configured heuristic variants when the active variant mix is positive, then include B1 when the active no-league mix is positive, with deduplication preserving order.

### Remaining Risks

- `python/weiss_rl/runtime.py` is still large at 5649 lines after the extraction.
- The remaining runtime central collection/native rollout paths are behavior-sensitive and should not be moved without much stronger parity coverage.

## 2026-05-11 - Big-Four Focused Extraction Checkpoint

### Scope

- Kept the checkpoint focused on the four largest files called out by the completion audit: `python/scripts/train.py`, `python/weiss_rl/learners/impala_learner.py`, `python/weiss_rl/model.py`, and `python/weiss_rl/runtime.py`.
- Extracted training learner construction from `train.py` into `python/weiss_rl/training/learner_factory.py`, preserving `_build_training_learner()` as the script compatibility wrapper.
- Extracted public-heuristic target-logit scoring from `ImpalaLearner` into `python/weiss_rl/learners/structured_auxiliary.py`, preserving the learner method as the state adapter.
- Extracted MLP/typed observation encoder construction from `PolicyValueModel` into `python/weiss_rl/model_typed_encoder.py`, preserving the model method as the class-local adapter.
- Added direct characterization tests for IMPALA/PPO learner kwargs, profile-target scorer calls/context passing, encoder builder output shapes, and old encoder error messages.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest python/weiss_rl/tests/test_training_learner_factory.py python/weiss_rl/tests/test_training_learner_compile.py python/weiss_rl/tests/test_script_entrypoint_smokes.py python/weiss_rl/tests/test_learner_structured_auxiliary.py python/weiss_rl/tests/test_impala_learner.py python/weiss_rl/tests/test_model_typed_encoder.py -q` | Passed: 87 passed, 14 dependency warnings. |
| `uv run ruff check python/scripts/train.py python/weiss_rl/training/learner_factory.py python/weiss_rl/training/__init__.py python/weiss_rl/learners/structured_auxiliary.py python/weiss_rl/learners/impala_learner.py python/weiss_rl/model_typed_encoder.py python/weiss_rl/model.py python/weiss_rl/tests/test_training_learner_factory.py python/weiss_rl/tests/test_learner_structured_auxiliary.py python/weiss_rl/tests/test_model_typed_encoder.py` | Passed. |
| `uv run ruff format --check python/scripts/train.py python/weiss_rl/training/learner_factory.py python/weiss_rl/training/__init__.py python/weiss_rl/learners/structured_auxiliary.py python/weiss_rl/learners/impala_learner.py python/weiss_rl/model_typed_encoder.py python/weiss_rl/model.py python/weiss_rl/tests/test_training_learner_factory.py python/weiss_rl/tests/test_learner_structured_auxiliary.py python/weiss_rl/tests/test_model_typed_encoder.py` | Passed. |
| `uv run mypy python/weiss_rl/training/learner_factory.py python/weiss_rl/learners/structured_auxiliary.py python/weiss_rl/model_typed_encoder.py python/weiss_rl/tests/test_training_learner_factory.py python/weiss_rl/tests/test_learner_structured_auxiliary.py python/weiss_rl/tests/test_model_typed_encoder.py --show-error-codes --no-error-summary` | Passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1170 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |
| `git diff --check` | Passed with pre-existing CRLF normalization warnings for `README.md`, `configs/README.md`, `docs/README.md`, `test_contracts.py`, `test_play_vs_model.py`, and `test_vtrace.py`. |

### Current Big-File Line Counts

- `python/weiss_rl/runtime.py`: 5649 lines.
- `python/weiss_rl/model.py`: 4344 lines.
- `python/weiss_rl/learners/impala_learner.py`: 3224 lines.
- `python/scripts/train.py`: 2873 lines.

### Behavior Changes

No intended behavior changes. Learner construction still selects the same IMPALA and PPO-lite classes for the same algorithm families and passes the same constructor kwargs. Public-heuristic target scoring still selects profiles with the same end-update and cycle/mixture rules, calls the model scorer with the same observation rows, legal-action view, observation context, and scoring profile, then mixes packed profile logits with the same row-wise normalization. Observation encoder construction still accepts the same encoder kinds and raises the same user-facing errors for missing specs, mismatched observation lengths, and unsupported encoders.

### Remaining Risks

- The four target files are still large; this checkpoint deliberately moved only low-risk adapter/helper code.
- `runtime.py` did not shrink in this checkpoint because the remaining runtime candidates are mostly central rollout/native-collection code and need stronger parity tests before movement.
- Direct `mypy python/scripts/train.py` still has known script-level protocol/cast debt outside the selected verifier gate; the new helper and package-level helper surfaces were typed directly.

## 2026-05-11 - Structured Warmstart Training Extraction Checkpoint

### Scope

- Extracted structured warmstart execution from `python/scripts/train.py` into `python/weiss_rl/training/warmstart.py`.
- Preserved `_run_structured_warmstart()` as the script compatibility wrapper.
- Added direct fake-runtime tests for disabled/zero-update skips, IMPALA-only validation, temporary teacher-coefficient override, coefficient restoration, warmstart source metrics, runtime metrics, scalar logging, TensorBoard logging, and latest-metric return values.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest python/weiss_rl/tests/test_training_warmstart.py python/weiss_rl/tests/test_script_entrypoint_smokes.py -q` | Passed: 18 passed, 14 dependency warnings. |
| `uv run ruff check python/weiss_rl/training/warmstart.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_training_warmstart.py python/scripts/train.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/training/warmstart.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_training_warmstart.py python/scripts/train.py` | Passed. |
| `uv run mypy python/weiss_rl/training/warmstart.py python/weiss_rl/tests/test_training_warmstart.py --show-error-codes --no-error-summary` | Passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1173 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |
| `git diff --check` | Passed with pre-existing CRLF normalization warnings for `README.md`, `configs/README.md`, `docs/README.md`, `test_contracts.py`, `test_play_vs_model.py`, and `test_vtrace.py`. |

### Behavior Changes

No intended behavior changes. Structured warmstart still skips when disabled or configured for zero updates, still rejects non-IMPALA algorithms, still overrides teacher auxiliary coefficients only for the warmstart block, still restores the previous coefficient/profile settings in a `finally` path, still runs collection and auxiliary learner updates under the same profiling/thread scopes, and still writes scalar/TensorBoard records with warmstart and source metrics.

### Remaining Risks

- `python/scripts/train.py` is still large at 2797 lines after this extraction.
- The helper is covered with fake-runtime characterization; it does not replace a simulator-backed warmstart smoke.

## 2026-05-11 - Periodic Dev-Eval Seed Payload Extraction Checkpoint

### Scope

- Extracted periodic dev-eval seed-usage payload construction from `python/scripts/train.py` into `python/weiss_rl/training/dev_eval.py`.
- Added direct artifact-contract-style coverage for seed file metadata, seed-schedule expansion fields, protocol fields, focal checkpoint path relativization, and opponent policy metadata.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest python/weiss_rl/tests/test_training_dev_eval.py python/weiss_rl/tests/test_script_entrypoint_smokes.py -q` | Passed: 28 passed, 14 dependency warnings. |
| `uv run ruff check python/weiss_rl/training/dev_eval.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_training_dev_eval.py python/scripts/train.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/training/dev_eval.py python/weiss_rl/training/__init__.py python/weiss_rl/tests/test_training_dev_eval.py python/scripts/train.py` | Passed. |
| `uv run mypy python/weiss_rl/training/dev_eval.py python/weiss_rl/tests/test_training_dev_eval.py --show-error-codes --no-error-summary` | Passed. |

### Behavior Changes

No intended behavior changes. The helper preserves the existing `seed_usage.json` payload shape, including dev-eval seed file metadata, validated source names, configured/requested seed counts, expansion marker, paired seed list, deterministic eval protocol fields, focal checkpoint path relativization against the run dir, and opponent policy metadata.

### Remaining Risks

- This only extracts payload construction; the periodic dev-eval loop itself remains in `train.py`.

## 2026-05-11 - Runtime Teacher-Label ID/Mask Extraction Checkpoint

### Scope

- Moved the public teacher decision-kind contract into `python/weiss_rl/runtime_teacher_labels.py`.
- Extracted teacher-label routing from packed legal IDs and dense legal masks into `python/weiss_rl/runtime_teacher_labels.py`.
- Preserved `QueueRuntime._teacher_labels_from_ids()` and `_teacher_labels_from_mask()` as state adapters.
- Added direct tests for public decision-kind coverage, selector callback inputs, teacher counter mutation, dense-mask routing, and inactive/missing-policy sentinel returns.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime_teacher_labels.py python/weiss_rl/tests/test_runtime.py -k "teacher_labels or build_learner_batch_preserves_teacher_labels"` | Passed: 11 passed, 85 deselected. |
| `uv run ruff check python/weiss_rl/runtime.py python/weiss_rl/runtime_teacher_labels.py python/weiss_rl/tests/test_runtime_teacher_labels.py python/weiss_rl/tests/test_runtime.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/runtime.py python/weiss_rl/runtime_teacher_labels.py python/weiss_rl/tests/test_runtime_teacher_labels.py python/weiss_rl/tests/test_runtime.py` | Passed. |
| `uv run mypy python/weiss_rl/runtime_teacher_labels.py python/weiss_rl/runtime.py python/weiss_rl/tests/test_runtime_teacher_labels.py --show-error-codes --no-error-summary` | Passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1177 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |
| `git diff --check` | Passed with pre-existing CRLF normalization warnings for `README.md`, `configs/README.md`, `docs/README.md`, `test_contracts.py`, `test_play_vs_model.py`, and `test_vtrace.py`. |

### Behavior Changes

No intended behavior changes. The helper preserves the public decision-kind set `{1,2,3,4,5,6,7,8}`, returns sentinel arrays when guidance is inactive or no teacher policy is present, increments `teacher_tactical_row_count` before selector dispatch, passes through the same mutable counters object, and decodes selected teacher actions through the existing action-catalog helper.

### Remaining Risks

- `runtime.py` is still large at 5624 lines after the extraction.
- The central collection/native rollout bodies still need stronger parity coverage before movement.

## 2026-05-11 - IMPALA Structured Dense Group Helper Extraction Checkpoint

### Scope

- Extracted dense structured group lookup and dense group log-probability helpers from `python/weiss_rl/learners/impala_learner.py` into `python/weiss_rl/learners/structured_auxiliary.py`.
- Kept the larger `compute_structured_teacher_auxiliary_metrics()` body in place for now; this checkpoint only moved small pure helpers from that cluster.
- Added direct tests for dense action-table construction and manual group log-sum-exp parity.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_learner_structured_auxiliary.py python/weiss_rl/tests/test_impala_learner.py -k "structured_group_lookup or dense_group_log_probs or structured_teacher_auxiliary or compute_structured_teacher_auxiliary_metrics"` | Passed: 15 passed, 44 deselected. |
| `uv run ruff check python/weiss_rl/learners/structured_auxiliary.py python/weiss_rl/learners/impala_learner.py python/weiss_rl/tests/test_learner_structured_auxiliary.py python/weiss_rl/tests/test_impala_learner.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/learners/structured_auxiliary.py python/weiss_rl/learners/impala_learner.py python/weiss_rl/tests/test_learner_structured_auxiliary.py python/weiss_rl/tests/test_impala_learner.py` | Passed. |
| `uv run mypy python/weiss_rl/learners/structured_auxiliary.py python/weiss_rl/learners/impala_learner.py python/weiss_rl/tests/test_learner_structured_auxiliary.py --show-error-codes --no-error-summary` | Passed. |

### Behavior Changes

No intended behavior changes. The lookup helper still builds the same family, play-slot, move-target, attack-slot, and attack-type tensors on the requested device. Dense group log probabilities still use the same `-1.0e9` empty-group sentinel and row-wise `logsumexp` normalization.

### Remaining Risks

- `impala_learner.py` is still large at 3188 lines after this extraction.
- The full structured teacher-auxiliary metrics body remains in `impala_learner.py`; moving it later should preserve wrapper imports and broad metric-key coverage.

## 2026-05-11 - Live Baseline, Model Bug Fix, and Runtime Policy-ID Cleanup

### Scope

- Re-established a live validation baseline after the existing large uncommitted refactor draft.
- Fixed a behavior-preservation bug found during read-only code audit: choice pagination families were not included in the extracted factorized/index family partition, so `choice_next_page` and `choice_prev_page` could fall through to default scoring instead of the intended index-family path.
- Fixed a second behavior-preservation bug in the extracted public attack heuristic helper: direct/frontal/side attack-type ids were hard-coded instead of using the action catalog ids already stored by `_StructuredLegalActionHead`.
- Centralized runtime mirror/B1 policy-id constants in `runtime_policy_ids.py` and `baselines.py` while preserving the existing private runtime facade names.
- Corrected stale documentation around full-package mypy, training-log scope, the refactor-log filename, live verifier counts, and package smoke evidence.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv sync --extra dev` | Passed. Removed the simulator extra from the dev-only environment for the first live baseline. |
| `uv run python -c "import weiss_rl; print(weiss_rl.__all__)"` | Passed: `['load_stack_config', 'assert_spec_compatibility']`. |
| `uv run python python/scripts/verify_repo.py` | Passed before the latest fixes: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1165 pytest tests passed, 16 skipped, 14 dependency warnings, wrapper dry-runs passed. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed; full-package mypy remains clean. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_model_action_tables.py::test_build_factorized_action_lookup_tables_preserves_family_argument_contracts` | Failed before the fix, demonstrating that page families were not marked with index argument kind `6`. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_model_public_heuristics.py::test_attack_public_heuristic_raw_uses_catalog_attack_type_ids` | Failed before the fix because `attack_public_heuristic_raw()` did not accept catalog-derived attack-type ids. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_model_action_tables.py python/weiss_rl/tests/test_model_public_heuristics.py python/weiss_rl/tests/test_model_candidate_partitioning.py` | Passed after the fixes: 19 passed. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_model_action_plans.py python/weiss_rl/tests/test_model_action_tables.py python/weiss_rl/tests/test_model_candidate_components.py python/weiss_rl/tests/test_model_candidate_partitioning.py python/weiss_rl/tests/test_model_feature_gathering.py python/weiss_rl/tests/test_model_layers.py python/weiss_rl/tests/test_model_loading.py python/weiss_rl/tests/test_model_observation_contract.py python/weiss_rl/tests/test_model_public_heuristics.py python/weiss_rl/tests/test_model_sampling.py python/weiss_rl/tests/test_model_tensor_ops.py python/weiss_rl/tests/test_model_typed_encoder.py` | Passed: 65 passed. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime_actor_state.py python/weiss_rl/tests/test_runtime_outcomes.py python/weiss_rl/tests/test_runtime_collector_commands.py python/weiss_rl/tests/test_runtime.py -k "actor_state or outcome or collector_command or refresh_opponent_pool or fixed_opponent or noleague or mirror"` | Passed after policy-id constant centralization: 24 passed, 73 deselected. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_model_action_tables.py python/weiss_rl/tests/test_model_public_heuristics.py python/weiss_rl/tests/test_runtime_actor_state.py python/weiss_rl/tests/test_runtime_outcomes.py python/weiss_rl/tests/test_runtime_collector_commands.py` | Passed: 25 passed. |
| `uv sync --extra dev --extra sim` | Passed. Installed `weiss-sim==0.8.1`. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_simulator_contract.py python/weiss_rl/tests/test_rl_step_layout_contract_smoke.py python/weiss_rl/tests/test_heuristic_public.py -k "simulator_native_heuristic_pool_matches_python_oracle_across_live_steps or simulator_contract or rl_step_layout"` | Passed: 10 passed, 27 deselected. |
| `uv run python python/scripts/write_paper_readiness_fixture.py --run-dir runs/refactor_paper_readiness_fixture_20260511_live; uv run python python/scripts/paper_readiness_check.py --run-dir runs/refactor_paper_readiness_fixture_20260511_live` | Passed. Wrote a fresh local fixture and validated the paper-readiness contract. |
| `uv run python python/scripts/train.py --stack-config configs/stack_smoke.yaml --run-label refactor_stack_smoke_20260511_live --num-envs 1 --unroll-length 1 --max-updates 1 --runtime-mode train_ordered --device cpu` | Passed. Wrote a scaffold-only manifest; no learner training or rollout collection executed because required config blocks are absent. |
| `uv run python python/scripts/eval.py --stack-config configs/presets/structured_acceptance_standard_thesis_eval.yaml` | Passed. Evaluation contract check completed without episode summaries. |
| `uv run python python/scripts/verify_repo.py` | Passed after the fixes with simulator extra installed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1180 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed after the fixes. |
| `uv run python -m build` | Passed. Built sdist and wheel. |
| Temp-venv wheel import smoke with `numpy` and `PyYAML` installed, then `pip install --no-deps dist/weiss_schwarz_rl-0.1.0-py3-none-any.whl` and `python -c "import weiss_rl; print(weiss_rl.__all__)"` | Passed: `['load_stack_config', 'assert_spec_compatibility']`. A prior no-deps temp import failed as expected because the temp venv lacked `numpy`. |
| `git diff --check` | Passed with pre-existing CRLF normalization warnings for `README.md`, `configs/README.md`, `docs/README.md`, `docs/training_logs.md`, `test_contracts.py`, `test_play_vs_model.py`, and `test_vtrace.py`. |

### Changes

- Added pagination coverage to `python/weiss_rl/tests/test_model_action_tables.py`.
- Added catalog-id coverage to `python/weiss_rl/tests/test_model_public_heuristics.py`.
- Updated `python/weiss_rl/model_action_tables.py` so `choice_next_page` and `choice_prev_page` are treated as index families.
- Updated `python/weiss_rl/model_public_heuristics.py` so attack scoring accepts direct/frontal/side attack-type ids and gracefully skips unavailable ids.
- Updated `python/weiss_rl/model.py` so `_StructuredLegalActionHead._attack_public_heuristic_raw()` passes catalog-derived attack-type ids to the extracted helper.
- Added `python/weiss_rl/runtime_policy_ids.py`.
- Updated `python/weiss_rl/runtime.py`, `runtime_actor_state.py`, `runtime_outcomes.py`, `runtime_process.py`, and `runtime_collector_commands.py` to share runtime policy ids instead of duplicating literals.
- Updated `docs/testing.md`, `docs/training_logs.md`, `docs/refactor_completion_audit.md`, `REFACTOR_PLAN.md`, and `AGENTS.md` for current validation state and path consistency.

### Behavior Changes

Two confirmed behavior-preserving bug fixes were made:

- Choice pagination actions now stay in the same index-family scoring path as other choice/level/trigger indexed actions. This restores the documented model scoring intent for `choice_next_page` and `choice_prev_page`; old results that depended on the extracted refactor draft before this fix should be treated as suspect for choice-paging model scores.
- Public attack heuristic scoring now respects the simulator/action-catalog attack-type encoding instead of assuming fixed direct/frontal/side ids. This restores the pre-extraction dynamic-id behavior; old results from the original pre-refactor code are not affected by this extraction-regression fix.

The runtime policy-id cleanup has no intended behavior change. It only removes duplicated literals while keeping existing private runtime constants and public policy ids stable.

### Current Big-File Line Counts

- `python/weiss_rl/runtime.py`: 5626 lines.
- `python/weiss_rl/model.py`: 4347 lines.
- `python/weiss_rl/learners/impala_learner.py`: 3188 lines.
- `python/scripts/train.py`: 2780 lines.
- `python/weiss_rl/config/parse.py`: 319 lines.

### Failed Ideas

- The first temp-venv wheel import smoke used `pip install --no-deps` without installing import-time dependencies. It failed with `ModuleNotFoundError: No module named 'numpy'`, so the useful package smoke was rerun with minimal import-time dependencies installed before the no-deps wheel install.

### Remaining Risks and Next Hypotheses

- The refactor is still not complete: `runtime.py`, `model.py`, `impala_learner.py`, and `train.py` remain large enough to require more behavior-characterized extraction.
- Compatibility wrappers are useful but now form a hidden private API. Future checkpoints should migrate helper-focused tests toward split modules while preserving only necessary facade imports.
- The next high-value extraction candidate is still the large structured teacher-auxiliary metrics region in `impala_learner.py` or a self-contained runtime central-collection helper with stronger parity tests.

## 2026-05-11 - Structured Teacher-Auxiliary Loss Module Extraction

### Scope

- Extracted `compute_structured_teacher_auxiliary_metrics()` from `python/weiss_rl/learners/impala_learner.py` into `python/weiss_rl/learners/structured_teacher_auxiliary.py`.
- Preserved `weiss_rl.learners.impala_learner.compute_structured_teacher_auxiliary_metrics` as a compatibility export.
- Added a compatibility test proving the old `impala_learner.py` symbol is the new implementation function.
- Kept the existing structured-teacher auxiliary tests as the behavior harness for dense, packed, factorized, public-heuristic, same-family, move-source, and metric-key behavior.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_impala_learner.py -k "compute_structured_teacher_auxiliary_metrics"` | Passed after the extraction: 13 passed, 30 deselected. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_impala_learner.py python/weiss_rl/tests/test_learner_structured_auxiliary.py` | Passed after adding the facade-export test: 60 passed. |
| `uv run ruff check python/weiss_rl/learners/impala_learner.py python/weiss_rl/learners/structured_teacher_auxiliary.py python/weiss_rl/tests/test_impala_learner.py python/weiss_rl/tests/test_learner_structured_auxiliary.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/learners/impala_learner.py python/weiss_rl/learners/structured_teacher_auxiliary.py python/weiss_rl/tests/test_impala_learner.py python/weiss_rl/tests/test_learner_structured_auxiliary.py` | Passed after formatting the touched files. |
| `uv run mypy python/weiss_rl/learners/impala_learner.py python/weiss_rl/learners/structured_teacher_auxiliary.py python/weiss_rl/tests/test_impala_learner.py python/weiss_rl/tests/test_learner_structured_auxiliary.py --show-error-codes --no-error-summary` | Passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1181 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `git diff --check` | Passed with pre-existing CRLF normalization warnings for `README.md`, `configs/README.md`, `docs/README.md`, `docs/training_logs.md`, `test_contracts.py`, `test_play_vs_model.py`, and `test_vtrace.py`. |

### Changes

- Added `python/weiss_rl/learners/structured_teacher_auxiliary.py`.
- Updated `python/weiss_rl/learners/impala_learner.py` to import and re-export `compute_structured_teacher_auxiliary_metrics`.
- Updated `python/weiss_rl/tests/test_impala_learner.py` with a facade-export compatibility assertion.
- Updated `CHANGELOG.md`, `REFACTOR_PLAN.md`, and `docs/refactor_completion_audit.md`.

### Behavior Changes

No intended behavior changes. This is a function-level extraction of existing dense, packed, and factorized structured-teacher auxiliary loss behavior behind the old import path. The existing tests still call through `impala_learner.py`, and the new compatibility assertion pins the facade relationship.

### Current Big-File Line Counts

- `python/weiss_rl/runtime.py`: 5626 lines.
- `python/weiss_rl/model.py`: 4347 lines.
- `python/scripts/train.py`: 2780 lines.
- `python/weiss_rl/learners/impala_learner.py`: 2033 lines.
- `python/weiss_rl/learners/structured_teacher_auxiliary.py`: 1177 lines.
- `python/weiss_rl/config/parse.py`: 319 lines.

### Failed Ideas

- The first mechanical identifier replacement in the new file was too broad and produced local names such as `selecteddense_group_log_probs` and `packeddense_group_log_probs`. This was corrected before validation; the focused structured-teacher tests passed afterward.

### Remaining Risks and Next Hypotheses

- The refactor is still not complete: `runtime.py`, `model.py`, and `train.py` remain large enough to require more behavior-characterized extraction.
- `structured_teacher_auxiliary.py` is intentionally a large cohesive module for the moved loss computation. Splitting its dense/packed/factorized branches further should wait until the new module has direct branch-level tests.
- The next highest-value checkpoint is likely a runtime central-collection helper with parity tests, or a model structured-head helper that reduces `model.py` without touching legal-action ordering.

## 2026-05-11 - Model Candidate Projection Helper Extraction

### Scope

- Extracted structured candidate projection and joint candidate-group scoring from `python/weiss_rl/model.py` into `python/weiss_rl/model_candidate_projection.py`.
- Preserved `_StructuredLegalActionHead._project_candidate_sections()` and `_score_candidate_group()` as model-local compatibility wrappers.
- Added direct actor/learner parity tests for candidate projection and joint group scoring, including constant numeric-one columns and the existing error text for invalid projections or empty inputs.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_model_candidate_projection.py python/weiss_rl/tests/test_model_candidate_components.py python/weiss_rl/tests/test_model_feature_gathering.py python/weiss_rl/tests/test_model_public_heuristics.py python/weiss_rl/tests/test_model_typed_encoder.py` | Passed: 31 passed. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_model_candidate_projection.py python/weiss_rl/tests/test_model_action_plans.py python/weiss_rl/tests/test_model_action_tables.py python/weiss_rl/tests/test_model_candidate_components.py python/weiss_rl/tests/test_model_candidate_partitioning.py python/weiss_rl/tests/test_model_feature_gathering.py python/weiss_rl/tests/test_model_layers.py python/weiss_rl/tests/test_model_loading.py python/weiss_rl/tests/test_model_observation_contract.py python/weiss_rl/tests/test_model_public_heuristics.py python/weiss_rl/tests/test_model_sampling.py python/weiss_rl/tests/test_model_tensor_ops.py python/weiss_rl/tests/test_model_typed_encoder.py` | Passed: 68 passed. |
| `uv run ruff check python/weiss_rl/model.py python/weiss_rl/model_candidate_projection.py python/weiss_rl/tests/test_model_candidate_projection.py` | Passed after removing a now-unused `torch.nn.functional` import from `model.py`. |
| `uv run ruff format --check python/weiss_rl/model.py python/weiss_rl/model_candidate_projection.py python/weiss_rl/tests/test_model_candidate_projection.py` | Passed after formatting the touched files. |
| `uv run mypy python/weiss_rl/model.py python/weiss_rl/model_candidate_projection.py python/weiss_rl/tests/test_model_candidate_projection.py --show-error-codes --no-error-summary` | Passed after making the tests pass explicit keyword arguments instead of an untyped kwargs dict and avoiding a local-name redefinition in the helper. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1184 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `git diff --check` | Passed with pre-existing CRLF normalization warnings for `README.md`, `configs/README.md`, `docs/README.md`, `docs/training_logs.md`, `test_contracts.py`, `test_play_vs_model.py`, and `test_vtrace.py`. |

### Changes

- Added `python/weiss_rl/model_candidate_projection.py`.
- Added `python/weiss_rl/tests/test_model_candidate_projection.py`.
- Updated `python/weiss_rl/model.py` to delegate projection and group scoring to the new helper module.
- Updated `CHANGELOG.md` and `docs/refactor_completion_audit.md`.

### Behavior Changes

No intended behavior changes. The new helper preserves the existing actor-mode concatenated projection path, learner-mode decomposed projection path, constant numeric-one handling, post-projection modules, joint scorer decomposition, and error messages. The model keeps the old private methods as wrappers.

### Current Big-File Line Counts

- `python/weiss_rl/runtime.py`: 5626 lines.
- `python/weiss_rl/model.py`: 4274 lines.
- `python/scripts/train.py`: 2780 lines.
- `python/weiss_rl/learners/impala_learner.py`: 2033 lines.
- `python/weiss_rl/config/parse.py`: 319 lines.

### Failed Ideas

- The first typed helper/test draft used a shared untyped kwargs dictionary in tests and reused `projected` across branches, which made focused mypy noisy despite passing runtime tests. The helper and tests were tightened before validation.

### Remaining Risks and Next Hypotheses

- The refactor is still not complete: `runtime.py`, `model.py`, and `train.py` remain large enough to require more behavior-characterized extraction.
- `model.py` still contains large packed/factorized scoring and public-heuristic scoring plans. The next model-side move should target one similarly bounded projection/scoring helper or add branch-level tests before moving larger scoring-plan bodies.
- `runtime.py` remains the largest blocker and should be approached with parity tests around a narrow central-collection or actor-routing helper.

## 2026-05-11 - Periodic Dev-Eval Execution Orchestration Extraction

### Scope

- Extracted the periodic dev-eval execution loop from `python/scripts/train.py` into `python/weiss_rl/training/periodic_dev_eval_run.py`.
- Preserved `train.py`'s `_run_periodic_dev_eval()` as a compatibility wrapper that injects the existing script-local checkpoint writer, focal-policy id helper, eval runner subclass, opponent resolver, persistence hooks, and JSON writer.
- Kept periodic dev-eval seed scheduling, checkpoint creation, CPU eval-model cloning, opponent iteration, seat-swapped matchup execution, matchup summary/CSV/diagnostic artifact writes, aggregate summary persistence, and stall-monitor update behavior unchanged.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run ruff check python/scripts/train.py python/weiss_rl/training/periodic_dev_eval_run.py` | Passed after import sorting and removing one unused import. |
| `uv run mypy python/weiss_rl/training/periodic_dev_eval_run.py --show-error-codes --no-error-summary` | Passed after removing a redundant cast. |
| `uv run python -m pytest python/weiss_rl/tests/test_training_dev_eval.py python/weiss_rl/tests/test_snapshot_registry.py -q` | Passed: 54 passed, 14 dependency warnings. |
| `uv run mypy python/scripts/train.py --show-error-codes --no-error-summary` | Still reports existing script-level protocol/cast debt unrelated to this extraction; package-level mypy remains the authoritative tracked gate until `train.py` is typed directly. |
| `uv run python python/scripts/verify_repo.py` | Passed after formatting `python/scripts/train.py`: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1184 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `git diff --check` | Passed with pre-existing CRLF normalization warnings for `README.md`, `configs/README.md`, `docs/README.md`, `docs/training_logs.md`, `test_contracts.py`, `test_play_vs_model.py`, and `test_vtrace.py`. |

### Changes

- Added `python/weiss_rl/training/periodic_dev_eval_run.py`.
- Updated `python/scripts/train.py` to delegate `_run_periodic_dev_eval()` to the new module.
- Updated `CHANGELOG.md`, `REFACTOR_PLAN.md`, and `docs/refactor_completion_audit.md`.

### Behavior Changes

No intended behavior changes. This is an orchestration extraction with explicit dependency injection so the old `train.py` helper remains the public/script-private surface used by existing tests and monkeypatches.

### Current Big-File Line Counts

- `python/weiss_rl/runtime.py`: 5626 lines.
- `python/weiss_rl/model.py`: 4274 lines.
- `python/scripts/train.py`: 2637 lines.
- `python/weiss_rl/learners/impala_learner.py`: 2033 lines.
- `python/weiss_rl/config/parse.py`: 319 lines.

### Failed Ideas

- A direct focused mypy invocation including `python/scripts/train.py` was too broad for this checkpoint and surfaced the existing script-level protocol/cast debt. The new module itself type-checks cleanly.

### Remaining Risks and Next Hypotheses

- The refactor remains incomplete: `runtime.py`, `model.py`, and `train.py` are still large enough to block the final objective.
- The safest next `train.py` reduction is likely alias-cleaning the pure checkpoint-guard or startup/input forwarding helpers while preserving underscore names.
- The highest-value runtime move is likely an opponent/PFSP adapter mixin, but only after running the focused runtime opponent tests around sampling, active policy ids, and role assignment.

## 2026-05-11 - Checkpoint-Guard Compatibility Alias Cleanup

### Scope

- Removed pure forwarding definitions from `python/scripts/train.py` for checkpoint-guard/dev-eval metric helpers.
- Preserved the old underscore helper names as explicit module-level aliases to `weiss_rl.training.checkpoint_guard`.
- Left checkpoint alias publishing, checkpoint writing, promotion-gate execution, and training-loop control flow unchanged.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run ruff check python/scripts/train.py` | Passed after switching private re-export imports to explicit assignments. |
| `uv run python -m pytest python/weiss_rl/tests/test_train_stall_monitor.py -q` | Passed: 15 passed, 14 dependency warnings. |
| `uv run python python/scripts/verify_repo.py` | Passed after formatting `python/scripts/train.py`: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1184 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `git diff --check` | Passed with pre-existing CRLF normalization warnings for `README.md`, `configs/README.md`, `docs/README.md`, `docs/training_logs.md`, `test_contracts.py`, `test_play_vs_model.py`, and `test_vtrace.py`. |

### Changes

- Updated `python/scripts/train.py` to bind checkpoint-guard compatibility names directly from `training.checkpoint_guard`.
- Updated `CHANGELOG.md`, `REFACTOR_PLAN.md`, and `docs/refactor_completion_audit.md`.

### Behavior Changes

No intended behavior changes. The old private helper names still resolve from `train.py`; they now point directly at the shared helper implementations instead of one-line forwarding functions.

### Current Big-File Line Counts

- `python/weiss_rl/runtime.py`: 5626 lines.
- `python/weiss_rl/model.py`: 4274 lines.
- `python/scripts/train.py`: 2537 lines.
- `python/weiss_rl/learners/impala_learner.py`: 2033 lines.
- `python/weiss_rl/config/parse.py`: 319 lines.

### Failed Ideas

- A direct aliased import block tripped Ruff `F401` because the aliases are part of the script-private compatibility surface. Explicit assignment from a module import preserved the names without lint suppressions.

### Remaining Risks and Next Hypotheses

- The refactor remains incomplete: `runtime.py`, `model.py`, and `train.py` are still large enough to block the final objective.
- Another pure `train.py` wrapper cleanup is possible, but the next larger structural win is probably the runtime opponent/PFSP adapter mixin recommended by the explorer.
- Promotion-gate execution and the main training loop should stay in place until their side effects are characterized more directly.

## 2026-05-11 - Startup/Input Compatibility Alias Cleanup

### Scope

- Removed pure forwarding definitions from `python/scripts/train.py` for training startup and input-validation helpers.
- Preserved the old underscore helper names as explicit module-level aliases to `training.inputs` and `training.startup` helpers.
- Left run setup, manifest writing, training-loop control flow, and runtime prerequisite behavior unchanged.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run ruff check python/scripts/train.py` | Passed after removing the now-unused `argparse` import. |
| `uv run python -m pytest python/weiss_rl/tests/test_training_inputs.py python/weiss_rl/tests/test_training_startup.py python/weiss_rl/tests/test_script_entrypoint_smokes.py -q` | Passed: 25 passed, 14 dependency warnings. |
| `uv run python python/scripts/verify_repo.py` | Passed after formatting `python/scripts/train.py`: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1184 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `git diff --check` | Passed with pre-existing CRLF normalization warnings for `README.md`, `configs/README.md`, `docs/README.md`, `docs/training_logs.md`, `test_contracts.py`, `test_play_vs_model.py`, and `test_vtrace.py`. |

### Changes

- Updated `python/scripts/train.py` to bind startup/input compatibility names directly from extracted helper modules.
- Updated `CHANGELOG.md`, `REFACTOR_PLAN.md`, and `docs/refactor_completion_audit.md`.

### Behavior Changes

No intended behavior changes. The old private helper names still resolve from `train.py`; they now point directly at the shared helper implementations instead of one-line forwarding functions.

### Current Big-File Line Counts

- `python/weiss_rl/runtime.py`: 5626 lines.
- `python/weiss_rl/model.py`: 4274 lines.
- `python/scripts/train.py`: 2485 lines.
- `python/weiss_rl/learners/impala_learner.py`: 2033 lines.
- `python/weiss_rl/config/parse.py`: 319 lines.

### Failed Ideas

- None for this checkpoint. Formatting was required after the import and alias edits before the full verifier would pass.

### Remaining Risks and Next Hypotheses

- The refactor remains incomplete: `runtime.py`, `model.py`, and `train.py` are still large enough to block the final objective.
- The next meaningful architectural move should shift back to `runtime.py`, with the opponent/PFSP adapter cluster as the most promising low-risk target.
- Keep promotion-gate execution and the main training loop in place until side-effect-focused orchestration tests are stronger.

## 2026-05-11 - Runtime Opponent/PFSP Adapter Mixin Extraction

### Scope

- Added `python/weiss_rl/runtime_opponent_mixin.py` to hold QueueRuntime's opponent/PFSP adapter methods.
- Updated `QueueRuntime` to inherit `QueueRuntimeOpponentMixin`, preserving the old private method surface for runtime internals and tests.
- Removed the adapter method bodies from `python/weiss_rl/runtime.py`; the actual opponent/PFSP sampling algorithms remain in `runtime_opponents.py`.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest python/weiss_rl/tests/test_runtime_opponents.py python/weiss_rl/tests/test_runtime.py -k "active_heuristic_public_mix_fraction or active_warmup_snapshot_mix_fraction or active_actor_heuristic_fraction or sample_opponent_policy_ids or assign_episode_roles or pfsp_sampling_ready" -q` | Passed: 14 passed, 89 deselected. |
| `uv run python -m pytest python/weiss_rl/tests/test_runtime_opponents.py python/weiss_rl/tests/test_runtime.py -q` | Passed: 103 passed. |
| `uv run ruff check python/weiss_rl/runtime.py python/weiss_rl/runtime_opponent_mixin.py python/weiss_rl/tests/test_runtime_opponents.py python/weiss_rl/tests/test_runtime.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/runtime.py python/weiss_rl/runtime_opponent_mixin.py` | Passed after formatting `runtime.py`. |
| `uv run mypy python/weiss_rl/runtime_opponent_mixin.py python/weiss_rl/runtime.py --show-error-codes --no-error-summary` | Passed after adding the mixin attribute fallback. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1184 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `git diff --check` | Passed with pre-existing CRLF normalization warnings for `README.md`, `configs/README.md`, `docs/README.md`, `docs/training_logs.md`, `test_contracts.py`, `test_play_vs_model.py`, and `test_vtrace.py`. |

### Changes

- Added `python/weiss_rl/runtime_opponent_mixin.py`.
- Updated `python/weiss_rl/runtime.py` so `QueueRuntime` inherits the mixin.
- Updated `CHANGELOG.md`, `REFACTOR_PLAN.md`, and `docs/refactor_completion_audit.md`.

### Behavior Changes

No intended behavior changes. QueueRuntime still exposes the same private opponent/PFSP methods, and those methods still delegate to the previously extracted `runtime_opponents.py` helpers with the same constants and runtime state.

### Current Big-File Line Counts

- `python/weiss_rl/runtime.py`: 5397 lines.
- `python/weiss_rl/model.py`: 4274 lines.
- `python/scripts/train.py`: 2485 lines.
- `python/weiss_rl/learners/impala_learner.py`: 2033 lines.
- `python/weiss_rl/config/parse.py`: 319 lines.

### Failed Ideas

- A first typed mixin draft surfaced mypy `attr-defined` errors because the mixin reads QueueRuntime-owned attributes. Adding a small `__getattr__` fallback kept the mixin explicit while preserving runtime AttributeError behavior for truly missing attributes.

### Remaining Risks and Next Hypotheses

- The refactor remains incomplete: `runtime.py`, `model.py`, and `train.py` are still large enough to block the final objective.
- The next runtime candidate is the pending-unroll fill/release cluster, but shared-transport tests currently monkeypatch `weiss_rl.runtime._read_unroll_from_shared_slot`, so compatibility hooks must be handled carefully.
- Central rollout collection and policy-output row routing should stay in place until stronger parity tests exist.

## 2026-05-11 - Runtime Pending-Unroll Adapter Mixin Extraction

### Scope

- Added `python/weiss_rl/runtime_pending_mixin.py` for QueueRuntime pending-unroll selection, shared-slot release, actor round-robin, fill-loop, and diverse-lane adapter methods.
- Updated `QueueRuntime` to inherit `QueueRuntimePendingMixin`, preserving the private method surface used by runtime internals and tests.
- Kept a tiny `QueueRuntime._read_unroll_from_shared_slot()` hook in `runtime.py` so existing monkeypatches of `weiss_rl.runtime._read_unroll_from_shared_slot` still affect shared-slot spill behavior.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest python/weiss_rl/tests/test_runtime_pending.py python/weiss_rl/tests/test_runtime.py -k "select_pending_unrolls or fill_pending_unrolls or shared_pending_unroll or shared_collector_slot or next_actor_batch" -q` | Passed: 10 passed, 83 deselected. |
| `uv run python -m pytest python/weiss_rl/tests/test_runtime_pending.py python/weiss_rl/tests/test_runtime.py -q` | Passed: 93 passed. |
| `uv run ruff check python/weiss_rl/runtime.py python/weiss_rl/runtime_pending_mixin.py python/weiss_rl/tests/test_runtime_pending.py python/weiss_rl/tests/test_runtime.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/runtime.py python/weiss_rl/runtime_pending_mixin.py` | Passed after formatting `runtime.py`. |
| `uv run mypy python/weiss_rl/runtime_pending_mixin.py python/weiss_rl/runtime.py --show-error-codes --no-error-summary` | Passed after making the actor cursor read use `getattr()`. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1184 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `git diff --check` | Passed with pre-existing CRLF normalization warnings for `README.md`, `configs/README.md`, `docs/README.md`, `docs/training_logs.md`, `test_contracts.py`, `test_play_vs_model.py`, and `test_vtrace.py`. |

### Changes

- Added `python/weiss_rl/runtime_pending_mixin.py`.
- Updated `python/weiss_rl/runtime.py` so `QueueRuntime` inherits the pending mixin and keeps the shared-slot read compatibility hook.
- Updated `CHANGELOG.md`, `REFACTOR_PLAN.md`, and `docs/refactor_completion_audit.md`.

### Behavior Changes

No intended behavior changes. QueueRuntime still exposes the same private pending-unroll methods, and the shared-slot spill path still routes through the monkeypatchable `weiss_rl.runtime._read_unroll_from_shared_slot` function.

### Current Big-File Line Counts

- `python/weiss_rl/runtime.py`: 5275 lines.
- `python/weiss_rl/model.py`: 4274 lines.
- `python/scripts/train.py`: 2485 lines.
- `python/weiss_rl/learners/impala_learner.py`: 2033 lines.
- `python/weiss_rl/config/parse.py`: 319 lines.

### Failed Ideas

- Removing `queue` from `runtime.py` was too aggressive because `close()` still suppresses `queue.Full`. The import was restored before validation.

### Remaining Risks and Next Hypotheses

- The refactor remains incomplete: `runtime.py`, `model.py`, and `train.py` are still large enough to block the final objective.
- Shared collector transport facade helpers may be the next low-risk runtime target, but the module-level private helper names imported by tests must remain stable.
- Central rollout collection and policy-output row routing should stay in place until stronger parity tests exist.

## 2026-05-11 - Runtime Shared Collector Transport Facade Extraction

### Scope

- Added `python/weiss_rl/runtime_shared_transport.py` for runtime-specific shared collector slot configuration, open, metadata, write, and read helpers.
- Removed the one-line shared collector transport helper bodies from `python/weiss_rl/runtime.py`.
- Preserved the old private helper names in `weiss_rl.runtime` as explicit module-level aliases so existing tests/imports and the shared-slot monkeypatch hook keep working.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run ruff check python/weiss_rl/runtime.py python/weiss_rl/runtime_shared_transport.py python/weiss_rl/tests/test_runtime.py` | Passed after replacing unused private re-export imports with explicit aliases. |
| `uv run ruff format --check python/weiss_rl/runtime.py python/weiss_rl/runtime_shared_transport.py` | Passed after formatting `runtime.py`. |
| `uv run mypy python/weiss_rl/runtime_shared_transport.py python/weiss_rl/runtime.py --show-error-codes --no-error-summary` | Passed. |
| `uv run python -m pytest python/weiss_rl/tests/test_runtime.py -k "shared_collector_slot or shared_pending_unroll or fill_pending_unrolls_spills_shared_slots_when_target_exceeds_capacity" -q` | Passed: 3 passed, 86 deselected. |
| `uv run python -m pytest python/weiss_rl/tests/test_runtime.py -q` | Passed: 89 passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1184 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `git diff --check` | Passed with pre-existing CRLF normalization warnings for `README.md`, `configs/README.md`, `docs/README.md`, `docs/training_logs.md`, `test_contracts.py`, `test_play_vs_model.py`, and `test_vtrace.py`. |

### Changes

- Added `python/weiss_rl/runtime_shared_transport.py`.
- Updated `python/weiss_rl/runtime.py` to bind the private shared-transport helper names to the extracted facade module.
- Updated `CHANGELOG.md`, `REFACTOR_PLAN.md`, and `docs/refactor_completion_audit.md`.

### Behavior Changes

No intended behavior changes. The old private helper names still resolve from `weiss_rl.runtime`, and `QueueRuntime._read_unroll_from_shared_slot()` still routes through the monkeypatchable module-level alias.

### Current Big-File Line Counts

- `python/weiss_rl/runtime.py`: 5224 lines.
- `python/weiss_rl/model.py`: 4274 lines.
- `python/scripts/train.py`: 2485 lines.
- `python/weiss_rl/learners/impala_learner.py`: 2033 lines.
- `python/weiss_rl/config/parse.py`: 319 lines.

### Failed Ideas

- Direct private re-export imports worked at runtime but tripped Ruff `F401` for aliases that exist primarily as a compatibility surface. Explicit module-level alias assignments preserved those names without suppressions.

### Remaining Risks and Next Hypotheses

- The refactor remains incomplete: `runtime.py`, `model.py`, and `train.py` are still large enough to block the final objective.
- Further runtime movement is more coupled now; reassess `model.py` and `impala_learner.py` before another runtime extraction.
- Central rollout collection and policy-output row routing should stay in place until stronger parity tests exist.

## 2026-05-11 - Runtime Compatibility Alias Cleanup

### Scope

- Replaced many one-line private compatibility wrappers in `python/weiss_rl/runtime.py` with explicit aliases to already-extracted helper modules.
- Covered runtime batching, counters, hashing, IPC, topology, device, shared-transport, and config helper names.
- Kept `_configure_runtime_actor_torch_threads()` as a distinct wrapper because `test_runtime_threads.py` characterizes that compatibility shape.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run ruff check python/weiss_rl/runtime.py python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_batching.py python/weiss_rl/tests/test_runtime_counters.py python/weiss_rl/tests/test_runtime_hashing.py python/weiss_rl/tests/test_runtime_threads.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/runtime.py` | Passed after formatting `runtime.py`. |
| `uv run mypy python/weiss_rl/runtime.py --show-error-codes --no-error-summary` | Passed. |
| `uv run python -m pytest python/weiss_rl/tests/test_runtime_batching.py python/weiss_rl/tests/test_runtime_counters.py python/weiss_rl/tests/test_runtime_hashing.py python/weiss_rl/tests/test_runtime_threads.py python/weiss_rl/tests/test_runtime.py -k "concat_optional_time_major_field or gae_advantages or concatenate_legal_actions or runtime_counters or runtime_hashing or resolve_actor_device_layout or actor_topology or actor_seed or shared_collector_slot or shared_pending_unroll" -q` | Passed: 21 passed, 85 deselected. |
| `uv run python -m pytest python/weiss_rl/tests/test_runtime.py -q` | Passed: 89 passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1184 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `git diff --check` | Passed with pre-existing CRLF normalization warnings for `README.md`, `configs/README.md`, `docs/README.md`, `docs/training_logs.md`, `test_contracts.py`, `test_play_vs_model.py`, and `test_vtrace.py`. |

### Changes

- Updated `python/weiss_rl/runtime.py` to use explicit aliases for already-extracted helper implementations instead of duplicate one-line wrapper bodies.
- Updated `CHANGELOG.md`, `REFACTOR_PLAN.md`, and `docs/refactor_completion_audit.md`.

### Behavior Changes

No intended behavior changes. Private helper names still resolve from `weiss_rl.runtime`; this only removes duplicate forwarding bodies for helpers whose behavior already lives in focused modules.

### Current Big-File Line Counts

- `python/weiss_rl/runtime.py`: 5031 lines.
- `python/weiss_rl/model.py`: 4274 lines.
- `python/scripts/train.py`: 2485 lines.
- `python/weiss_rl/learners/impala_learner.py`: 2033 lines.
- `python/weiss_rl/config/parse.py`: 319 lines.

### Failed Ideas

- None. Formatting was required after moving aliases.

### Remaining Risks and Next Hypotheses

- The refactor remains incomplete: `runtime.py`, `model.py`, and `train.py` are still large enough to block the final objective.
- The next safest structural target is the structured model observation-context helper extraction; it keeps module parameters on the existing model head and avoids V-trace, recurrent state, legal ordering, checkpoint, and central rollout semantics.
- Keep the remaining runtime central collection and policy-output row routing in place until stronger parity tests exist.

## 2026-05-11 - Model Public-Heuristic Scoring Mixin Extraction

### Scope

- Moved structured policy-head public-heuristic adapter methods and packed public-heuristic scoring into `python/weiss_rl/model_public_heuristic_scoring.py`.
- Updated `_StructuredLegalActionHead` to inherit `StructuredPublicHeuristicScoringMixin`, preserving the old private method names on model instances.
- Left recurrent state handling, dense/packed legal-action ordering, factorized policy evaluation, checkpoint-sensitive module names, and builder behavior unchanged.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run ruff check python/weiss_rl/model.py python/weiss_rl/model_public_heuristic_scoring.py python/weiss_rl/tests/test_model_public_heuristics.py python/weiss_rl/tests/test_model_sampling.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/model.py python/weiss_rl/model_public_heuristic_scoring.py` | Passed after formatting the moved code. |
| `uv run mypy python/weiss_rl/model_public_heuristic_scoring.py python/weiss_rl/model.py --show-error-codes --no-error-summary` | Passed after using a file-local attr-defined suppression for the mixin's owner-provided attributes. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_model_public_heuristics.py python/weiss_rl/tests/test_model_sampling.py python/weiss_rl/tests/test_model_action_plans.py python/weiss_rl/tests/test_model_candidate_components.py python/weiss_rl/tests/test_model_candidate_projection.py python/weiss_rl/tests/test_model_loading.py python/weiss_rl/tests/test_model_typed_encoder.py` | Passed: 41 passed. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_heuristic_public.py python/weiss_rl/tests/test_impala_learner.py -k "public_heuristic or structured_model_public_bias or packed"` | Passed after removing an unsafe mixin `__getattr__` override: 17 passed, 55 deselected. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_model_feature_gathering.py python/weiss_rl/tests/test_model_candidate_partitioning.py python/weiss_rl/tests/test_model_observation_contract.py python/weiss_rl/tests/test_model_tensor_ops.py` | Passed: 19 passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1184 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `git diff --check` | Passed with the known pre-existing CRLF normalization warnings for `README.md`, `configs/README.md`, `docs/README.md`, `docs/training_logs.md`, `test_contracts.py`, `test_play_vs_model.py`, and `test_vtrace.py`. |

### Changes

- Added `python/weiss_rl/model_public_heuristic_scoring.py`.
- Removed the corresponding public-heuristic helper/scoring method bodies from `python/weiss_rl/model.py`.
- Kept the private method names available through `_StructuredLegalActionHead` inheritance, so existing model callers and tests still resolve the same method surface.
- Reduced `python/weiss_rl/model.py` from about 4014 lines to 3431 lines.

### Behavior Changes

No intended behavior changes. This is a structured extraction behind the existing model method surface.

### Failed Ideas

- The first mixin draft included a defensive `__getattr__`, which shadowed `torch.nn.Module.__getattr__` and broke submodule lookup for `hand_summary_projection`. The live heuristic-public model tests caught this; the override was removed before full validation.

### Current Big-File Line Counts

- `python/weiss_rl/runtime.py`: 4879 lines.
- `python/weiss_rl/model.py`: 3431 lines.
- `python/scripts/train.py`: 2222 lines.
- `python/weiss_rl/learners/impala_learner.py`: 1879 lines.
- `python/weiss_rl/config/parse.py`: 248 lines.

### Remaining Risks and Next Hypotheses

- `model.py` is materially smaller but still owns the core recurrent policy/value model plus structured candidate scoring. The next model-side candidates should be factorized scoring or dense/packed candidate scoring, but only behind parity tests.
- `runtime.py` remains the largest file; central collection and policy-output row routing should still wait for stronger same-seed parity tests.
- `train.py` and `impala_learner.py` remain large, but several apparent one-line wrappers are intentionally preserved because tests assert wrapper identity or compatibility behavior.

## 2026-05-11 - Train Compatibility Alias Cleanup

### Scope

- Replaced a group of pure one-line `python/scripts/train.py` compatibility wrappers with explicit module-level aliases to already-extracted helper implementations.
- Covered snapshot helper facades, import-contract helpers, policy-selection helpers, environment/batch helpers, checkpoint helper facades, and run-metadata helper facades.
- Preserved behavior-owning wrappers such as `_persist_snapshot_registry_entry()`, `_write_checkpoint()`, `_build_checkpoint_record()`, and `_publish_checkpoint_aliases()` because they still adapt script-local artifacts, config hashes, or model-guidance payloads.
- Preserved the public script-level `MinimalRollout` compatibility import after the full verifier caught tests importing it from `scripts.train`.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest python/weiss_rl/tests/test_snapshot_registry.py python/weiss_rl/tests/test_training_run_identity.py python/weiss_rl/tests/test_training_startup.py python/weiss_rl/tests/test_training_environments.py python/weiss_rl/tests/test_training_batches.py python/weiss_rl/tests/test_training_checkpoint_writers.py python/weiss_rl/tests/test_script_entrypoint_smokes.py -q` | Passed: 86 passed, 14 dependency warnings. |
| `uv run python -m pytest python/weiss_rl/tests/test_train_stall_monitor.py python/weiss_rl/tests/test_snapshot_registry.py python/weiss_rl/tests/test_training_batches.py python/weiss_rl/tests/test_script_entrypoint_smokes.py -q` | Passed after restoring `MinimalRollout`: 77 passed, 14 dependency warnings. |
| `uv run ruff check --fix python/scripts/train.py; uv run ruff format python/scripts/train.py` | Passed; Ruff sorted imports after alias edits. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1184 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `git diff --check` | Passed with the known pre-existing CRLF normalization warnings for `README.md`, `configs/README.md`, `docs/README.md`, `docs/training_logs.md`, `test_contracts.py`, `test_play_vs_model.py`, and `test_vtrace.py`. |

### Changes

- Updated `python/scripts/train.py` to bind many private compatibility names directly to package helpers instead of maintaining duplicate forwarding bodies.
- Kept `MinimalRollout = _TrainingMinimalRollout` so old tests and callers importing `MinimalRollout` from `scripts.train` continue to work.
- Reduced `python/scripts/train.py` from about 2222 lines to 2046 lines.

### Behavior Changes

No intended behavior changes. This is duplicate-wrapper cleanup behind the same private compatibility names.

### Failed Ideas

- Removing the `MinimalRollout` import entirely broke collection in `test_train_stall_monitor.py`; the compatibility alias was restored before the final verifier.

### Current Big-File Line Counts

- `python/weiss_rl/runtime.py`: 4879 lines.
- `python/weiss_rl/model.py`: 3431 lines.
- `python/scripts/train.py`: 2046 lines.
- `python/weiss_rl/learners/impala_learner.py`: 1879 lines.
- `python/weiss_rl/config/parse.py`: 248 lines.

### Remaining Risks and Next Hypotheses

- `train.py` is smaller, but the main training loop, B1/seed import orchestration, promotion-gate execution, checkpoint publication, and manifest mutation still make it a large behavior owner.
- The next `train.py` reductions should target another pure-wrapper cluster only after checking for compatibility imports, or extract a behavior-owning orchestration slice with stronger end-to-end train fixtures.
- `runtime.py` remains the biggest file, but central collection and policy-output row routing still need parity tests before movement.

## 2026-05-11 - IMPALA Compatibility Alias Cleanup

### Scope

- Collapsed pure private helper facades in `python/weiss_rl/learners/impala_learner.py` into explicit aliases to the extracted learner helper modules.
- Preserved real wrapper functions where characterization tests assert compatibility wrapper identity: NumPy logp facades, V-trace diagnostics, masked action logp/entropy, and packed selected-action logp.
- Removed a stale imported type and formatted the file after the previous crash left this checkpoint half-clean.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run ruff check python/weiss_rl/learners/impala_learner.py` | Passed after removing the stale `_StructuredCatalogMetadata` import. |
| `uv run ruff format --check python/weiss_rl/learners/impala_learner.py` | Passed after formatting `impala_learner.py`. |
| `uv run mypy python/weiss_rl/learners/impala_learner.py --show-error-codes --no-error-summary` | Passed. |
| `uv run python -m pytest python/weiss_rl/tests/test_learner_action_logp.py python/weiss_rl/tests/test_learner_vtrace_torch.py python/weiss_rl/tests/test_learner_vtrace_diagnostics.py python/weiss_rl/tests/test_learner_structured_auxiliary.py python/weiss_rl/tests/test_impala_learner.py -q` | Passed: 70 passed. |

### Changes

- Updated `python/weiss_rl/learners/impala_learner.py` to use aliases for already-extracted helpers including V-trace target computation, packed probability helpers, tensor segment helpers, public-heuristic profile helpers, and time-step legal action access.
- Reduced `python/weiss_rl/learners/impala_learner.py` from about 1879 lines to 1770 lines.

### Behavior Changes

No intended behavior changes. This is duplicate-wrapper cleanup behind the same private compatibility names.

### Failed Ideas

- The initial post-crash tree still had one unused imported type and needed formatting. Ruff caught both before this checkpoint was treated as clean.

### Current Big-File Line Counts

- `python/weiss_rl/runtime.py`: 4879 lines.
- `python/weiss_rl/model.py`: 3431 lines before the next model cleanup checkpoint.
- `python/scripts/train.py`: 2046 lines.
- `python/weiss_rl/learners/impala_learner.py`: 1770 lines.
- `python/weiss_rl/config/parse.py`: 248 lines.

### Remaining Risks and Next Hypotheses

- `impala_learner.py` is smaller, but update orchestration, optimizer stepping, packed legality flow, diagnostics, and compatibility wrappers still live together.
- The next IMPALA movement should be behavior-owning update-loop extraction only after additional characterization around optimizer/AMP state, recurrent hidden-state handling, and structured auxiliary loss aggregation.
- `runtime.py`, `model.py`, and `train.py` remain the main large-file targets.

## 2026-05-11 - Model Top-Level Compatibility Alias Cleanup

### Scope

- Replaced pure top-level `python/weiss_rl/model.py` tensor and observation helper facades with explicit aliases to the extracted helper modules.
- Preserved `_sample_masked_log_probs()` and `_sample_packed_action_scores()` as real wrappers because tests assert they are distinct from public helper functions and because their internals intentionally call monkeypatchable private RNG/CDF helpers.
- Left recurrent state handling, structured policy scoring, packed/factorized action ordering, checkpoint-sensitive module names, and builder behavior unchanged.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run ruff check python/weiss_rl/model.py python/weiss_rl/tests/test_model_typed_encoder.py python/weiss_rl/tests/test_model_tensor_ops.py python/weiss_rl/tests/test_model_sampling.py python/weiss_rl/tests/test_runtime.py` | Passed after removing unused observation-layout imports. |
| `uv run ruff format --check python/weiss_rl/model.py` | Passed after formatting `model.py`. |
| `uv run mypy python/weiss_rl/model.py --show-error-codes --no-error-summary` | Passed. |
| `uv run python -m pytest python/weiss_rl/tests/test_model_typed_encoder.py python/weiss_rl/tests/test_model_tensor_ops.py python/weiss_rl/tests/test_model_sampling.py python/weiss_rl/tests/test_model_observation_contract.py python/weiss_rl/tests/test_runtime.py::test_sample_packed_action_scores_falls_back_to_last_candidate_when_cdf_undershoots -q` | Passed: 26 passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1184 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed. |
| `git diff --check` | Passed with the known pre-existing CRLF normalization warnings for `README.md`, `configs/README.md`, `docs/README.md`, `docs/training_logs.md`, `test_contracts.py`, `test_play_vs_model.py`, and `test_vtrace.py`. |

### Changes

- Updated `python/weiss_rl/model.py` to alias pure compatibility names for typed-encoder helpers, observation-contract helpers, tensor ops, packed-row helpers, and deterministic seed helpers.
- Reduced `python/weiss_rl/model.py` from about 3431 lines to 3398 lines.

### Behavior Changes

No intended behavior changes. This is duplicate-wrapper cleanup behind the same private compatibility names. Sampling wrappers remain behavior-preserving wrappers.

### Failed Ideas

- `runtime.py` top-level actor-model/thread facades looked similar but tests intentionally assert their wrapper identity, so they were not collapsed in this checkpoint.

### Current Big-File Line Counts

- `python/weiss_rl/runtime.py`: 4879 lines.
- `python/weiss_rl/model.py`: 3398 lines.
- `python/scripts/train.py`: 2046 lines.
- `python/weiss_rl/learners/impala_learner.py`: 1770 lines.
- `python/weiss_rl/config/parse.py`: 248 lines.

### Remaining Risks and Next Hypotheses

- `runtime.py` remains the largest production file and still needs stronger parity tests before central collection or policy-output row routing is moved.
- `model.py` remains a large structured policy/value surface; the next safe model target is likely a behavior-owning factorized scoring slice with actor/learner parity tests.
- `train.py` remains a large orchestration file; further reductions should target either another identity-safe wrapper cluster or a small runner slice with end-to-end train smoke coverage.

## 2026-05-11 - Train Guidance and Dev-Eval Alias Cleanup

### Scope

- Replaced another bounded group of pure `python/scripts/train.py` compatibility wrappers with explicit aliases to extracted training helper modules.
- Covered manifest actor-device layout, guidance schedules, model-guidance payload helpers, learner compile selection, algorithm contract validation, dev-eval seed/path helpers, promotion-anchor helpers, and periodic dev-eval persistence/stall-monitor helpers.
- Preserved behavior-owning or dependency-injecting wrappers for checkpoint writing/publication, B1 and seed imports, snapshot eval model loading, periodic dev-eval opponent construction, promotion-gate orchestration, and warmstart execution.
- Restored compatibility surfaces that tests intentionally reach through `scripts.train`: `DecisionBoundaryBatch`, `ScheduledGame`, and the monkeypatch-sensitive `_build_heuristic_public_policy()` wrapper.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run ruff check python/scripts/train.py python/weiss_rl/tests/test_train_stall_monitor.py python/weiss_rl/tests/test_training_guidance.py python/weiss_rl/tests/test_training_dev_eval.py python/weiss_rl/tests/test_training_promotion.py python/weiss_rl/tests/test_training_learner_compile.py python/weiss_rl/tests/test_training_algorithm_contracts.py` | Passed after restoring compatibility aliases. |
| `uv run ruff format --check python/scripts/train.py` | Passed after formatting `train.py`. |
| `uv run python -m pytest python/weiss_rl/tests/test_train_stall_monitor.py python/weiss_rl/tests/test_training_guidance.py python/weiss_rl/tests/test_training_dev_eval.py python/weiss_rl/tests/test_training_promotion.py python/weiss_rl/tests/test_training_learner_compile.py python/weiss_rl/tests/test_training_algorithm_contracts.py python/weiss_rl/tests/test_script_entrypoint_smokes.py -q` | Passed: 70 passed, 14 dependency warnings. |
| `uv run python -m pytest python/weiss_rl/tests/test_snapshot_registry.py python/weiss_rl/tests/test_training_checkpoint_writers.py python/weiss_rl/tests/test_training_batches.py -q` | Passed after restoring `DecisionBoundaryBatch` and `ScheduledGame`: 56 passed, 14 dependency warnings. |
| `uv run python -m pytest python/weiss_rl/tests/test_train_stall_monitor.py python/weiss_rl/tests/test_snapshot_registry.py python/weiss_rl/tests/test_script_entrypoint_smokes.py python/weiss_rl/tests/test_training_manifest_layout.py -q` | Passed: 73 passed, 14 dependency warnings. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed after restoring `_update_stall_monitor()` as a type-narrowing adapter. |
| `uv run python -m pytest python/weiss_rl/tests/test_train_stall_monitor.py -q` | Passed after restoring `_update_stall_monitor()`: 15 passed, 14 dependency warnings. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1184 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |
| `uv run mypy python/weiss_rl --show-error-codes --no-error-summary` | Passed again after the final verifier. |

### Changes

- Updated `python/scripts/train.py` to bind another cluster of private compatibility names directly to package helpers.
- Kept `_build_heuristic_public_policy()` as a real wrapper because tests monkeypatch `scripts.train.HeuristicPublicPolicy` and expect periodic dev-eval opponent construction to observe it.
- Kept `DecisionBoundaryBatch` and `ScheduledGame` available from `scripts.train` as explicit aliases because snapshot tests construct runner fixtures through that script module.
- Reduced `python/scripts/train.py` from about 2046 lines to 1929 lines after the final alias cleanup pass.

### Behavior Changes

No intended behavior changes. This is duplicate-wrapper cleanup behind the same private compatibility names and restored script-level compatibility aliases.

### Failed Ideas

- Directly aliasing `_build_heuristic_public_policy` hid the monkeypatched `scripts.train.HeuristicPublicPolicy` used by `test_periodic_dev_eval_opponents_include_optional_b2_when_available`; the wrapper was restored.
- Removing unused-looking `DecisionBoundaryBatch` and `ScheduledGame` imports broke snapshot-registry runner fixture tests; both names were restored as explicit compatibility aliases.
- Directly aliasing `_update_stall_monitor` broke broad package mypy because the wrapper also narrows `TrainingPaths` for tests; the adapter wrapper was restored.
- `uv run mypy python/scripts/train.py --show-error-codes --no-error-summary` was tried during this checkpoint and still reports the known direct script-level protocol/cast debt. This remains outside the current package mypy gate and was not counted as passing validation.

### Current Big-File Line Counts

- `python/weiss_rl/runtime.py`: 4879 lines.
- `python/weiss_rl/model.py`: 3398 lines.
- `python/scripts/train.py`: 1929 lines.
- `python/weiss_rl/learners/impala_learner.py`: 1770 lines.
- `python/weiss_rl/config/parse.py`: 248 lines.

### Remaining Risks and Next Hypotheses

- `train.py` is still a large orchestration file. The remaining large chunks are behavior-owning checkpoint/snapshot publication, B1/seed import flows, promotion-gate execution, and the main training loop.
- Additional train reductions should target a real orchestration slice with focused end-to-end tests rather than only wrapper aliases.
- `runtime.py` and `model.py` remain the larger structural targets.

## 2026-05-11 - Runtime Learner-Batch Builder Extraction and Test Split

### Scope

- Extracted IMPALA and PPO learner-batch assembly from `QueueRuntime` into `python/weiss_rl/runtime_batching.py`.
- Kept `QueueRuntime._build_learner_batch()` and `QueueRuntime._build_ppo_batch()` as compatibility methods with the same signatures, including the currently unused `truncation_reward` parameter.
- Moved the matching GAE, legal-action concatenation, truncation, bootstrap, and teacher-label batch tests out of the monolithic `test_runtime.py` into `test_runtime_batching.py`.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run ruff check python/weiss_rl/runtime.py python/weiss_rl/runtime_batching.py python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_batching.py` | Passed after import sorting. |
| `uv run ruff format --check python/weiss_rl/runtime.py python/weiss_rl/runtime_batching.py python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_batching.py` | Passed after formatting edited runtime/test files. |
| `uv run mypy python/weiss_rl/runtime_batching.py python/weiss_rl/runtime.py --show-error-codes --no-error-summary` | Passed. |
| `uv run python -m pytest python/weiss_rl/tests/test_runtime_batching.py python/weiss_rl/tests/test_runtime.py::test_runtime_metrics_report_window_and_cumulative_env_step_rates -q` | Passed: 14 passed. |
| `uv run python -m pytest python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_batching.py -q` | Passed: 93 passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1184 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |
| `git diff --check` | Passed with the known pre-existing CRLF normalization warnings for `README.md`, `configs/README.md`, `docs/README.md`, `docs/training_logs.md`, `test_contracts.py`, `test_play_vs_model.py`, and `test_vtrace.py`. |

### Changes

- Added `build_impala_learner_batch()` and `build_ppo_learner_batch()` to `runtime_batching.py`.
- Updated `QueueRuntime` batch-building methods to delegate to the shared helpers while preserving the old private method names used by tests/call sites.
- Moved batch-construction characterization tests into the batching-focused runtime test file.

### Behavior Changes

No intended behavior changes. The extracted helpers build the same dictionaries, discounts, legal-action payloads, teacher-label fields, GAE advantages, returns, and timing callback as the previous `QueueRuntime` methods.

### Current Big-File Line Counts

- `python/weiss_rl/runtime.py`: 4710 lines after the subsequent process-collector startup extraction.
- `python/weiss_rl/tests/test_runtime.py`: 3637 lines.
- `python/weiss_rl/model.py`: 3398 lines.
- `python/scripts/train.py`: 1929 lines.
- `python/weiss_rl/learners/impala_learner.py`: 1770 lines.
- `python/weiss_rl/config/parse.py`: 248 lines.

### Remaining Risks and Next Hypotheses

- `runtime.py` is still the main source giant. The next lower-risk runtime code target is the small heuristic/teacher adapter mixins; central collection and policy row routing remain higher-risk.
- `test_runtime.py` still has large coherent opponent, topology, collection-mode, and shared-transport blocks that can be split without changing coverage.
- `model.py` likely needs a factorized-scoring mixin extraction next; keep public class/module identities stable for checkpoint and import compatibility.

## 2026-05-11 - Runtime Process-Collector Startup Extraction

### Scope

- Moved process-collector startup setup from `QueueRuntime._start_process_collectors()` into `python/weiss_rl/runtime_process.py`.
- Preserved `QueueRuntime._start_process_collectors()` as a compatibility hook because runtime process-mode tests monkeypatch it to verify startup routing/order.
- Kept `_collector_process_main()` in `runtime.py` as the top-level picklable process target that injects `QueueRuntime` into the extracted child loop.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run ruff check python/weiss_rl/runtime.py python/weiss_rl/runtime_process.py python/weiss_rl/tests/test_runtime.py` | Passed after import sorting. |
| `uv run ruff format --check python/weiss_rl/runtime.py python/weiss_rl/runtime_process.py python/weiss_rl/tests/test_runtime.py` | Passed after formatting `runtime.py` and `runtime_process.py`. |
| `uv run mypy python/weiss_rl/runtime.py python/weiss_rl/runtime_process.py --show-error-codes --no-error-summary` | Passed. |
| `uv run python -m pytest python/weiss_rl/tests/test_runtime.py -k "process_collectors or shared_collector_slot or shared_pending_unroll or fill_pending_unrolls_spills_shared_slots_when_target_exceeds_capacity" -q` | Passed: 8 passed, 72 deselected. |
| `uv run python -m pytest python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_batching.py -q` | Passed: 93 passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1184 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |
| `git diff --check` | Passed with the known pre-existing CRLF normalization warnings for `README.md`, `configs/README.md`, `docs/README.md`, `docs/training_logs.md`, `test_contracts.py`, `test_play_vs_model.py`, and `test_vtrace.py`. |

### Changes

- Added `start_process_collectors()` to `runtime_process.py`.
- Updated `QueueRuntime._start_process_collectors()` to delegate to the extracted helper while passing the existing `_collector_process_main` target.
- Removed direct multiprocessing startup code from `runtime.py`.

### Behavior Changes

No intended behavior changes. Process context selection, result/control/free queues, shared-slot config/opening, serialized model state, child-process kwargs, daemon flag, process start order, and runtime bookkeeping lists are preserved.

### Current Big-File Line Counts

- `python/weiss_rl/runtime.py`: 4710 lines.
- `python/weiss_rl/tests/test_runtime.py`: 3637 lines.
- `python/weiss_rl/model.py`: 3398 lines.
- `python/scripts/train.py`: 1929 lines.
- `python/weiss_rl/learners/impala_learner.py`: 1770 lines.

### Remaining Risks and Next Hypotheses

- `runtime.py` still owns central collection and row-level policy/opponent application. Those should stay in place until their exact parity tests are stronger.
- The next lower-risk runtime source move is the small heuristic/teacher adapter method cluster.
- The next lower-risk test move is splitting `test_runtime.py` topology/collection-mode/shared-transport or opponent integration blocks into focused test files.

## 2026-05-11 - Runtime Teacher/Heuristic Adapter Mixin

### Scope

- Moved the small `QueueRuntime` adapter cluster for heuristic fast-path predicates and teacher-label routing into `python/weiss_rl/runtime_teacher_heuristic_mixin.py`.
- Preserved the private `QueueRuntime._...` method names through mixin inheritance.
- Kept the behavior-owning pure helpers in `runtime_heuristic_fast_path.py` and `runtime_teacher_labels.py`.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run ruff check python/weiss_rl/runtime.py python/weiss_rl/runtime_teacher_heuristic_mixin.py python/weiss_rl/runtime_heuristic_fast_path.py python/weiss_rl/runtime_teacher_labels.py python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_heuristic_fast_path.py python/weiss_rl/tests/test_runtime_teacher_labels.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/runtime.py python/weiss_rl/runtime_teacher_heuristic_mixin.py python/weiss_rl/runtime_heuristic_fast_path.py python/weiss_rl/runtime_teacher_labels.py python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_heuristic_fast_path.py python/weiss_rl/tests/test_runtime_teacher_labels.py` | Passed after formatting `runtime.py`. |
| `uv run mypy python/weiss_rl/runtime.py python/weiss_rl/runtime_teacher_heuristic_mixin.py --show-error-codes --no-error-summary` | Passed after annotating the mixin `self` surface as runtime-shaped. |
| `uv run python -m pytest python/weiss_rl/tests/test_runtime_heuristic_fast_path.py python/weiss_rl/tests/test_runtime_teacher_labels.py python/weiss_rl/tests/test_runtime.py -k "heuristic_ids_fast or heuristic_ids_native_rollout or teacher_labels or teacher_guidance" -q` | Passed: 17 passed, 74 deselected. |
| `uv run python -m pytest python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_batching.py python/weiss_rl/tests/test_runtime_heuristic_fast_path.py python/weiss_rl/tests/test_runtime_teacher_labels.py -q` | Passed: 104 passed. |

### Changes

- Added `QueueRuntimeTeacherHeuristicMixin`.
- Updated `QueueRuntime` to inherit the mixin.
- Removed the corresponding adapter method bodies and direct helper imports from `runtime.py`.
- Reduced `python/weiss_rl/runtime.py` from 4710 lines to 4585 lines.

### Behavior Changes

No intended behavior changes. This is a method-location change only; the same private methods still delegate to the same tested pure helper functions with the same runtime state inputs.

### Remaining Risks and Next Hypotheses

- `runtime.py` still owns the central collection loops and row-level policy/opponent application; these remain behavior sensitive and should stay put until parity tests are stronger.
- `test_runtime.py` remains 3637 lines and is now the lowest-risk big-file target because several coherent blocks can be moved without changing behavior.

## 2026-05-11 - Runtime Topology Test Split

### Scope

- Moved runtime topology and collection-backend selection tests from `python/weiss_rl/tests/test_runtime.py` into `python/weiss_rl/tests/test_runtime_topology.py`.
- Preserved the existing test bodies, monkeypatches, assertions, and runtime construction paths.
- Removed imports from `test_runtime.py` that were only needed by the moved topology/backend tests.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run ruff check python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_topology.py` | Passed after Ruff removed stale imports and sorted the new file. |
| `uv run ruff format --check python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_topology.py` | Passed after formatting both files. |
| `uv run mypy python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_topology.py --show-error-codes --no-error-summary` | Passed. |
| `uv run python -m pytest python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_topology.py -q` | Passed: 80 passed. |
| `uv run python -m pytest python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_topology.py python/weiss_rl/tests/test_runtime_batching.py python/weiss_rl/tests/test_runtime_heuristic_fast_path.py python/weiss_rl/tests/test_runtime_teacher_labels.py -q` | Passed: 104 passed. |

### Changes

- Added `test_runtime_topology.py` for `_resolve_actor_topology()`, CUDA actor-device layout, central batched collection selection, process-collector forcing, and startup-before-refresh ordering tests.
- Reduced `python/weiss_rl/tests/test_runtime.py` from 3637 lines to 3093 lines.

### Behavior Changes

No behavior changes. This is a test-only relocation.

### Remaining Risks and Next Hypotheses

- `test_runtime.py` remains large but is now closer to focused integration coverage. Pending-unroll/shared-transport tests are the next easiest mechanical split.
- `runtime.py` and `model.py` remain the largest production files; further source movement should be backed by direct parity tests.

## 2026-05-11 - Runtime Shared-Transport Test Split

### Scope

- Moved pending shared-slot spill, diverse-lane waiting, parallel actor executor, shared collector slot round-trip, and shared pending-unroll view tests from `python/weiss_rl/tests/test_runtime.py` into `python/weiss_rl/tests/test_runtime_shared_transport.py`.
- Kept the monkeypatch path for `weiss_rl.runtime._read_unroll_from_shared_slot` intact.
- Added a local `_make_runtime_unroll()` fixture helper in the new file so the moved tests remain self-contained.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run ruff check python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_shared_transport.py` | Passed after Ruff removed stale shared-slot imports from `test_runtime.py`. |
| `uv run ruff format --check python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_shared_transport.py` | Passed after formatting both files. |
| `uv run mypy python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_shared_transport.py --show-error-codes --no-error-summary` | Passed. |
| `uv run python -m pytest python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_shared_transport.py -q` | Passed: 67 passed. |
| `uv run python -m pytest python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_topology.py python/weiss_rl/tests/test_runtime_shared_transport.py python/weiss_rl/tests/test_runtime_batching.py python/weiss_rl/tests/test_runtime_heuristic_fast_path.py python/weiss_rl/tests/test_runtime_teacher_labels.py -q` | Passed: 104 passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1184 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |
| `git diff --check` | Passed with the known pre-existing CRLF normalization warnings for `README.md`, `configs/README.md`, `docs/README.md`, `docs/training_logs.md`, `test_contracts.py`, `test_play_vs_model.py`, and `test_vtrace.py`. |

### Changes

- Added `test_runtime_shared_transport.py`.
- Reduced `python/weiss_rl/tests/test_runtime.py` from 3093 lines to 2820 lines.

### Behavior Changes

No behavior changes. This is a test-only relocation.

### Remaining Risks and Next Hypotheses

- `test_runtime.py` still contains opponent refresh/sampling/application and heuristic actor integration blocks, but it is no longer one of the top production-code blockers.
- The next production source target should likely be `model.py`, using focused parity coverage around factorized and packed scoring.

## 2026-05-11 - Model Factorized Scoring Mixin

### Scope

- Moved deterministic factorized policy scoring/evaluation helpers from `_StructuredLegalActionHead` into `python/weiss_rl/model_factorized_scoring.py`.
- Kept `_StructuredLegalActionHead` as the public module identity for checkpoint/state-dict compatibility; only inherited helper methods moved.
- Left `sample_factorized_packed()` in `model.py` because it depends on the existing sampling wrapper globals that tests protect as monkeypatchable behavior.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run ruff check python/weiss_rl/model.py python/weiss_rl/model_factorized_scoring.py python/weiss_rl/tests/test_model_candidate_projection.py python/weiss_rl/tests/test_model_action_plans.py python/weiss_rl/tests/test_runtime.py` | Passed after removing stale imports and sorting imports. |
| `uv run ruff format --check python/weiss_rl/model.py python/weiss_rl/model_factorized_scoring.py` | Passed after formatting both files. |
| `uv run mypy python/weiss_rl/model.py python/weiss_rl/model_factorized_scoring.py --show-error-codes --no-error-summary` | Passed after typing the mixin `self` surface as runtime/model-shaped. |
| `uv run python -m pytest python/weiss_rl/tests/test_model_action_plans.py python/weiss_rl/tests/test_model_candidate_projection.py python/weiss_rl/tests/test_model_observation_contract.py python/weiss_rl/tests/test_model_sampling.py python/weiss_rl/tests/test_runtime.py::test_apply_policy_rows_ids_prefers_factorized_structured_sampler -q` | Passed: 18 passed. |
| `uv run python -m pytest python/weiss_rl/tests/test_model_action_plans.py python/weiss_rl/tests/test_model_action_tables.py python/weiss_rl/tests/test_model_candidate_components.py python/weiss_rl/tests/test_model_candidate_partitioning.py python/weiss_rl/tests/test_model_candidate_projection.py python/weiss_rl/tests/test_model_feature_gathering.py python/weiss_rl/tests/test_model_layers.py python/weiss_rl/tests/test_model_observation_contract.py python/weiss_rl/tests/test_model_public_heuristics.py python/weiss_rl/tests/test_model_sampling.py python/weiss_rl/tests/test_model_tensor_ops.py python/weiss_rl/tests/test_model_typed_encoder.py python/weiss_rl/tests/test_runtime.py::test_apply_policy_rows_ids_prefers_factorized_structured_sampler -q` | Passed: 65 passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1184 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |
| `git diff --check` | Passed with the known pre-existing CRLF normalization warnings for `README.md`, `configs/README.md`, `docs/README.md`, `docs/training_logs.md`, `test_contracts.py`, `test_play_vs_model.py`, and `test_vtrace.py`. |

### Changes

- Added `StructuredFactorizedScoringMixin`.
- Updated `_StructuredLegalActionHead` to inherit the mixin.
- Reduced `python/weiss_rl/model.py` from 3398 lines to 2759 lines.

### Behavior Changes

No intended behavior changes. The same helper bodies now resolve through mixin inheritance, and stochastic factorized sampling remains in `model.py`.

### Remaining Risks and Next Hypotheses

- The remaining large `model.py` region is mostly packed-candidate scoring and the `PolicyValueModel` facade. A packed-scoring mixin is plausible but should keep sampling compatibility wrappers untouched.
- Run a full verifier after this checkpoint because production model code moved.

## 2026-05-11 - Model Packed Scoring Mixin

### Scope

- Moved packed scoring-plan construction, packed candidate family partitioning, generic-index projection, packed chunking, and packed scoring-plan execution from `_StructuredLegalActionHead` into `python/weiss_rl/model_packed_scoring.py`.
- Kept `_StructuredLegalActionHead` in `model.py` for checkpoint/state-dict compatibility.
- Left `_score_candidates()` and stochastic sampling wrappers in `model.py` because they are separate compatibility surfaces and should not be mixed into the same refactor checkpoint.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run ruff check python/weiss_rl/model.py python/weiss_rl/model_packed_scoring.py python/weiss_rl/model_factorized_scoring.py python/weiss_rl/tests/test_model_candidate_projection.py python/weiss_rl/tests/test_model_action_plans.py python/weiss_rl/tests/test_runtime.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/model.py python/weiss_rl/model_packed_scoring.py` | Passed after formatting both files. |
| `uv run mypy python/weiss_rl/model.py python/weiss_rl/model_packed_scoring.py --show-error-codes --no-error-summary` | Passed after typing the mixin `self` surface as model-shaped. |
| `uv run python -m pytest python/weiss_rl/tests/test_model_action_plans.py python/weiss_rl/tests/test_model_candidate_projection.py python/weiss_rl/tests/test_model_candidate_partitioning.py python/weiss_rl/tests/test_model_observation_contract.py python/weiss_rl/tests/test_model_sampling.py python/weiss_rl/tests/test_runtime.py::test_apply_policy_rows_ids_prefers_factorized_structured_sampler -q` | Passed: 20 passed. |
| `uv run python -m pytest python/weiss_rl/tests/test_model_action_plans.py python/weiss_rl/tests/test_model_action_tables.py python/weiss_rl/tests/test_model_candidate_components.py python/weiss_rl/tests/test_model_candidate_partitioning.py python/weiss_rl/tests/test_model_candidate_projection.py python/weiss_rl/tests/test_model_feature_gathering.py python/weiss_rl/tests/test_model_layers.py python/weiss_rl/tests/test_model_observation_contract.py python/weiss_rl/tests/test_model_public_heuristics.py python/weiss_rl/tests/test_model_sampling.py python/weiss_rl/tests/test_model_tensor_ops.py python/weiss_rl/tests/test_model_typed_encoder.py python/weiss_rl/tests/test_runtime.py::test_apply_policy_rows_ids_prefers_factorized_structured_sampler -q` | Passed: 65 passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1184 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |
| `git diff --check` | Passed with the known pre-existing CRLF normalization warnings for `README.md`, `configs/README.md`, `docs/README.md`, `docs/training_logs.md`, `test_contracts.py`, `test_play_vs_model.py`, and `test_vtrace.py`. |

### Changes

- Added `StructuredPackedScoringMixin`.
- Updated `_StructuredLegalActionHead` to inherit the packed-scoring mixin.
- Reduced `python/weiss_rl/model.py` from 2759 lines to 2285 lines.

### Behavior Changes

No intended behavior changes. The same packed-scoring helper bodies now resolve through mixin inheritance, and stochastic sampling behavior remains in `model.py`.

### Remaining Risks and Next Hypotheses

- Run a full verifier after this checkpoint because production model scoring code moved.
- `runtime.py` central collection and remaining `test_runtime.py` opponent/application coverage are now the largest code/test hotspots.

## 2026-05-11 - Runtime Opponent-Pool Test Split

### Scope

- Moved opponent-pool refresh, B1 anchor residency, promotion-gated reservoir, probationary recent pool, inflight stale assignment, effective-update snapshot reuse, champion age, timeout quarantine, and stale champion demotion tests from `python/weiss_rl/tests/test_runtime.py` into `python/weiss_rl/tests/test_runtime_opponent_pool.py`.
- Preserved the test bodies and assertions.
- Removed stale imports from `test_runtime.py`.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run ruff check python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_opponent_pool.py` | Passed after Ruff removed stale imports and sorted imports in the new file. |
| `uv run ruff format --check python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_opponent_pool.py` | Passed after formatting both files. |
| `uv run mypy python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_opponent_pool.py --show-error-codes --no-error-summary` | Passed. |
| `uv run python -m pytest python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_opponent_pool.py -q` | Passed: 62 passed. |
| `uv run python -m pytest python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_opponent_pool.py python/weiss_rl/tests/test_runtime_topology.py python/weiss_rl/tests/test_runtime_shared_transport.py python/weiss_rl/tests/test_runtime_batching.py python/weiss_rl/tests/test_runtime_heuristic_fast_path.py python/weiss_rl/tests/test_runtime_teacher_labels.py -q` | Passed: 104 passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1184 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |
| `git diff --check` | Passed with the known pre-existing CRLF normalization warnings for `README.md`, `configs/README.md`, `docs/README.md`, `docs/training_logs.md`, `test_contracts.py`, `test_play_vs_model.py`, and `test_vtrace.py`. |

### Changes

- Added `test_runtime_opponent_pool.py`.
- Reduced `python/weiss_rl/tests/test_runtime.py` from 2820 lines to 2252 lines.

### Behavior Changes

No behavior changes. This is a test-only relocation.

### Remaining Risks and Next Hypotheses

- The generic `test_runtime.py` still contains mixed runtime coverage, but the largest opponent-pool and opponent-sampling/application blocks now live in focused files.
- The largest remaining production hotspot is `runtime.py`; central collection movement should wait for explicit parity fixtures.

## 2026-05-11 - Runtime Opponent Sampling/Application Test Split

### Scope

- Moved opponent sampling, heuristic-public mix annealing, warmup snapshot sampling, fixed-anchor role assignment, central opponent-output overwrite, batched opponent rows, and simulator-native heuristic-public application tests from `python/weiss_rl/tests/test_runtime.py` into `python/weiss_rl/tests/test_runtime_opponent_sampling.py`.
- Preserved the test bodies and assertions.
- Removed stale imports from `test_runtime.py` and formatted both touched test files.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run ruff check python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_opponent_sampling.py` | Passed after Ruff removed stale imports and sorted the new file. |
| `uv run ruff format --check python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_opponent_sampling.py` | Passed after formatting both files. |
| `uv run mypy python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_opponent_sampling.py --show-error-codes --no-error-summary` | Passed. |
| `uv run python -m pytest python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_opponent_sampling.py -q` | Passed: 51 passed. |
| `uv run python -m pytest python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_opponent_pool.py python/weiss_rl/tests/test_runtime_opponent_sampling.py python/weiss_rl/tests/test_runtime_topology.py python/weiss_rl/tests/test_runtime_shared_transport.py python/weiss_rl/tests/test_runtime_batching.py python/weiss_rl/tests/test_runtime_heuristic_fast_path.py python/weiss_rl/tests/test_runtime_teacher_labels.py -q` | Passed: 104 passed. |

### Changes

- Added `test_runtime_opponent_sampling.py`.
- Reduced `python/weiss_rl/tests/test_runtime.py` from 2252 lines to 1494 lines.

### Behavior Changes

No behavior changes. This is a test-only relocation.

### Remaining Risks and Next Hypotheses

- Remaining big production files are `runtime.py` (4724 lines), `model.py` (2402), `train.py` (2028), and `impala_learner.py` (1886).
- Next checkpoint should preferably move production code with characterization coverage, because the generic runtime test file is no longer a top blocker.

## 2026-05-11 - Runtime Policy-Row Mixin

### Scope

- Moved dense-mask and packed-ids policy-row application helpers from `QueueRuntime` into `python/weiss_rl/runtime_policy_rows.py`.
- Moved packed sampled-action debug validation, env-step debug validation, actor-batch sync from simulator step output, and legal-action metadata completion with the row helpers.
- Kept private method names available on `QueueRuntime` through `QueueRuntimePolicyRowsMixin`; no public runtime entry point or simulator contract changed.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run ruff check python/weiss_rl/runtime.py python/weiss_rl/runtime_policy_rows.py python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_opponent_sampling.py python/weiss_rl/tests/test_runtime_batching.py python/weiss_rl/tests/test_runtime_teacher_labels.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/runtime.py python/weiss_rl/runtime_policy_rows.py python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_opponent_sampling.py python/weiss_rl/tests/test_runtime_batching.py python/weiss_rl/tests/test_runtime_teacher_labels.py` | Passed after formatting `runtime.py` and the new mixin. |
| `uv run mypy python/weiss_rl/runtime.py python/weiss_rl/runtime_policy_rows.py --show-error-codes --no-error-summary` | Passed after declaring the runtime attributes used by the mixin for type checking. |
| `uv run python -m pytest python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_opponent_sampling.py python/weiss_rl/tests/test_runtime_batching.py python/weiss_rl/tests/test_runtime_teacher_labels.py -q` | Passed: 71 passed. |
| `uv run python -m pytest python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_opponent_pool.py python/weiss_rl/tests/test_runtime_opponent_sampling.py python/weiss_rl/tests/test_runtime_topology.py python/weiss_rl/tests/test_runtime_shared_transport.py python/weiss_rl/tests/test_runtime_batching.py python/weiss_rl/tests/test_runtime_heuristic_fast_path.py python/weiss_rl/tests/test_runtime_teacher_labels.py -q` | Passed: 104 passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1184 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `QueueRuntimePolicyRowsMixin`.
- Reduced `python/weiss_rl/runtime.py` from 4724 lines to 4458 lines.

### Behavior Changes

No intended behavior changes. The same private helper names still resolve on `QueueRuntime`; their bodies now live in the policy-row mixin.

### Remaining Risks and Next Hypotheses

- `runtime.py` remains the largest hotspot because central rollout collection, all-heuristic rollout paths, and actor unroll collection still live in the class.
- The safest next runtime move is another cohesive adapter/mixin extraction; central collection should wait for explicit parity fixtures around row ordering, masking, recurrent-state advancement, and behavior log probabilities.

## 2026-05-11 - Model Dense Scoring Mixin

### Scope

- Moved the dense `_score_candidates()` implementation from `_StructuredLegalActionHead` into `python/weiss_rl/model_dense_scoring.py`.
- Kept `_StructuredLegalActionHead`, `_resolve_candidate_components()`, checkpoint-sensitive module ownership, and stochastic sampling wrappers in `model.py`.
- Added `StructuredDenseScoringMixin` ahead of the packed/factorized/public-heuristic scoring mixins so the private method still resolves on the same action-head instance.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run ruff check python/weiss_rl/model.py python/weiss_rl/model_dense_scoring.py python/weiss_rl/model_packed_scoring.py python/weiss_rl/model_factorized_scoring.py python/weiss_rl/tests/test_model_candidate_projection.py python/weiss_rl/tests/test_model_action_plans.py python/weiss_rl/tests/test_runtime.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/model.py python/weiss_rl/model_dense_scoring.py python/weiss_rl/model_packed_scoring.py python/weiss_rl/model_factorized_scoring.py` | Passed after formatting `model.py` and the new mixin. |
| `uv run mypy python/weiss_rl/model.py python/weiss_rl/model_dense_scoring.py --show-error-codes --no-error-summary` | Passed. |
| `uv run python -m pytest python/weiss_rl/tests/test_model_action_plans.py python/weiss_rl/tests/test_model_candidate_projection.py python/weiss_rl/tests/test_model_candidate_partitioning.py python/weiss_rl/tests/test_model_observation_contract.py python/weiss_rl/tests/test_model_sampling.py python/weiss_rl/tests/test_runtime.py::test_apply_policy_rows_ids_prefers_factorized_structured_sampler -q` | Passed: 20 passed. |
| `uv run python -m pytest python/weiss_rl/tests/test_model_action_plans.py python/weiss_rl/tests/test_model_action_tables.py python/weiss_rl/tests/test_model_candidate_components.py python/weiss_rl/tests/test_model_candidate_partitioning.py python/weiss_rl/tests/test_model_candidate_projection.py python/weiss_rl/tests/test_model_feature_gathering.py python/weiss_rl/tests/test_model_layers.py python/weiss_rl/tests/test_model_observation_contract.py python/weiss_rl/tests/test_model_public_heuristics.py python/weiss_rl/tests/test_model_sampling.py python/weiss_rl/tests/test_model_tensor_ops.py python/weiss_rl/tests/test_model_typed_encoder.py python/weiss_rl/tests/test_runtime.py::test_apply_policy_rows_ids_prefers_factorized_structured_sampler -q` | Passed: 65 passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1184 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `StructuredDenseScoringMixin`.
- Reduced `python/weiss_rl/model.py` from 2402 lines to 2024 lines.

### Behavior Changes

No intended behavior changes. The same dense scoring method remains available on `_StructuredLegalActionHead`; only its implementation location changed.

### Remaining Risks and Next Hypotheses

- The remaining model hotspot is now mostly `StructuredLegalPolicyValueModel` plus checkpoint/public facade methods.
- The remaining very large files are `runtime.py` (4458 lines), `train.py` (2028), `model.py` (2024), and `impala_learner.py` (1886).

## 2026-05-11 - IMPALA Factorized Evaluation Mixin

### Scope

- Moved factorized public-heuristic teacher-view construction, factorized-policy capability checks, and factorized time-major evaluation from `ImpalaLearner` into `python/weiss_rl/learners/factorized_evaluation.py`.
- Kept the learner dataclass, public constructor, update loop, optimizer behavior, V-trace integration, and loss orchestration unchanged.
- Preserved helper names on `ImpalaLearner` through `ImpalaFactorizedEvaluationMixin`.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run ruff check python/weiss_rl/learners/impala_learner.py python/weiss_rl/learners/factorized_evaluation.py python/weiss_rl/tests/test_impala_learner.py python/weiss_rl/tests/test_learner_structured_auxiliary.py python/weiss_rl/tests/test_learner_packed_rows.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/learners/impala_learner.py python/weiss_rl/learners/factorized_evaluation.py` | Passed after formatting both files. |
| `uv run mypy python/weiss_rl/learners/impala_learner.py python/weiss_rl/learners/factorized_evaluation.py --show-error-codes --no-error-summary` | Passed. |
| `uv run python -m pytest python/weiss_rl/tests/test_impala_learner.py python/weiss_rl/tests/test_learner_structured_auxiliary.py python/weiss_rl/tests/test_learner_packed_rows.py python/weiss_rl/tests/test_learner_action_logp.py python/weiss_rl/tests/test_learner_legal_fields.py python/weiss_rl/tests/test_learner_vtrace_torch.py -q` | Passed: 79 passed. |
| `uv run python -m pytest python/weiss_rl/tests/test_impala_learner.py python/weiss_rl/tests/test_vtrace.py python/weiss_rl/tests/test_training_batches.py -q` | Passed: 67 passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1184 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `ImpalaFactorizedEvaluationMixin`.
- Reduced `python/weiss_rl/learners/impala_learner.py` from 1886 lines to 1578 lines.

### Behavior Changes

No intended behavior changes. The same private methods still resolve on `ImpalaLearner`; their bodies now live in the factorized-evaluation mixin.

### Remaining Risks and Next Hypotheses

- Remaining very large files are now `runtime.py` (4458 lines), `train.py` (2028), and `model.py` (2024).
- The next `train.py` move should be guarded by anchor import/current-run alias/promotion-gate registry tests because promotion orchestration is behavior sensitive.

## 2026-05-11 - Training NoLeague Anchor Helper

### Scope

- Moved B1 NoLeague baseline anchor orchestration from `python/scripts/train.py` into `python/weiss_rl/training/noleague_anchor.py`.
- Preserved the script-level `_ensure_noleague_baseline_anchor()` compatibility wrapper used by existing tests and training flow.
- Passed script-owned dependencies explicitly into the helper: checkpoint writing, imported-baseline copying, guidance payload extraction, snapshot artifact writing, and experiment-role lookup.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run ruff check python/scripts/train.py python/weiss_rl/training/noleague_anchor.py python/weiss_rl/tests/test_snapshot_registry.py` | Passed. |
| `uv run ruff format --check python/scripts/train.py python/weiss_rl/training/noleague_anchor.py` | Passed after formatting both files. |
| `uv run mypy python/weiss_rl/training/noleague_anchor.py --show-error-codes --no-error-summary` | Passed. Direct `python/scripts/train.py` mypy still has the known broader script-level protocol debt and was not used as a success gate. |
| `uv run python -m pytest python/weiss_rl/tests/test_snapshot_registry.py -k "ensure_noleague_baseline_anchor or run_minimal_training_bootstraps_noleague or run_snapshot_promotion_gate" -q` | Passed: 9 passed, 32 deselected. |
| `uv run python -m pytest python/weiss_rl/tests/test_training_promotion.py python/weiss_rl/tests/test_train_stall_monitor.py -q` | Passed: 24 passed. |
| `uv run python -m pytest python/weiss_rl/tests/test_snapshot_registry.py -q` | Passed: 41 passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1184 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `ensure_noleague_baseline_anchor()`.
- Reduced `python/scripts/train.py` from 2028 lines to 1917 lines.

### Behavior Changes

No intended behavior changes. The same script-level private wrapper remains in place, and the characterization tests around imported anchors, current-run aliases, promotion-gate interactions, and registry updates still pass.

### Remaining Risks and Next Hypotheses

- Remaining very large files are now `runtime.py` (4458 lines), `model.py` (2024), and `train.py` (1917).
- Promotion-gate execution is the next plausible `train.py` extraction, but it should keep `_run_snapshot_promotion_gate()` as a compatibility wrapper and preserve warmup skip, missing-anchor skip, heuristic-anchor handling, pass/fail reporting, and champion registry updates.

## 2026-05-11 - Training Promotion-Gate Execution Helper

### Scope

- Moved promotion-gate execution orchestration from `python/scripts/train.py` into `python/weiss_rl/training/promotion_gate_execution.py`.
- Preserved the script-level `_run_snapshot_promotion_gate()` compatibility wrapper used by existing tests and training flow.
- Passed monkeypatch-sensitive script dependencies explicitly into the helper: `run_promotion_gate`, `_PromotionGateRunner`, anchor resolution, snapshot eval-model loading, heuristic policy construction, CPU eval-model cloning, promotion-gate seed derivation, and registry saving.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run ruff check python/scripts/train.py python/weiss_rl/training/promotion_gate_execution.py python/weiss_rl/tests/test_snapshot_registry.py` | Passed. |
| `uv run ruff format --check python/scripts/train.py python/weiss_rl/training/promotion_gate_execution.py` | Passed after formatting both files. |
| `uv run mypy python/weiss_rl/training/promotion_gate_execution.py --show-error-codes --no-error-summary` | Passed. Direct `python/scripts/train.py` mypy still has the known broader script-level protocol debt and was not used as a success gate. |
| `uv run python -m pytest python/weiss_rl/tests/test_snapshot_registry.py -k "run_snapshot_promotion_gate or promotion_gate_runner" -q` | Passed: 5 passed, 36 deselected. |
| `uv run python -m pytest python/weiss_rl/tests/test_training_promotion.py python/weiss_rl/tests/test_train_stall_monitor.py -q` | Passed: 24 passed. |
| `uv run python -m pytest python/weiss_rl/tests/test_snapshot_registry.py -q` | Passed: 41 passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1184 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `run_snapshot_promotion_gate()`.
- Reduced `python/scripts/train.py` from 1917 lines to 1812 lines.

### Behavior Changes

No intended behavior changes. Warmup skips, missing-anchor skips, heuristic-anchor handling, pass/fail logging, champion registry updates, and the test monkeypatch points still pass through the script wrapper.

### Remaining Risks and Next Hypotheses

- Remaining very large files are now `runtime.py` (4458 lines), `model.py` (2024), and `train.py` (1812).
- Further `train.py` work should focus on seed snapshot import or the minimal training loop. The central runtime collection path remains the largest hotspot but still needs stronger parity fixtures before movement.

## 2026-05-11 - Training Seed Snapshot Import Helper

### Scope

- Moved seed snapshot import contract validation and external seeded snapshot-pool import from `python/scripts/train.py` into `python/weiss_rl/training/seed_snapshots.py`.
- Preserved the script-level `_validate_seed_snapshot_import_contract()`, `_seed_snapshot_policy_id()`, and `_import_seed_snapshot_pool()` compatibility wrappers.
- Kept seed snapshot policy-id generation delegated to the existing `training.snapshots.seed_snapshot_policy_id()` helper.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run ruff check python/scripts/train.py python/weiss_rl/training/seed_snapshots.py python/weiss_rl/tests/test_snapshot_registry.py` | Passed. |
| `uv run ruff format --check python/scripts/train.py python/weiss_rl/training/seed_snapshots.py` | Passed after formatting both files. |
| `uv run mypy python/weiss_rl/training/seed_snapshots.py --show-error-codes --no-error-summary` | Passed. Direct `python/scripts/train.py` mypy still has the known broader script-level protocol debt and was not used as a success gate. |
| `uv run python -m pytest python/weiss_rl/tests/test_snapshot_registry.py -k "seed_snapshot or import_seed_snapshot_pool" -q` | Passed: 4 passed, 37 deselected. |
| `uv run python -m pytest python/weiss_rl/tests/test_training_execution.py python/weiss_rl/tests/test_training_report_payloads.py -q` | Passed: 9 passed. |
| `uv run python -m pytest python/weiss_rl/tests/test_snapshot_registry.py -q` | Passed: 41 passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1184 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `import_seed_snapshot_pool()` and `validate_seed_snapshot_import_contract()`.
- Reduced `python/scripts/train.py` from 1812 lines to 1697 lines.

### Behavior Changes

No intended behavior changes. The same script-level wrappers remain, and seed snapshot import success, champion preservation, environment mismatch rejection, and seed snapshot policy-id compatibility tests still pass.

### Remaining Risks and Next Hypotheses

- Remaining very large files are now `runtime.py` (4458 lines), `model.py` (2024), `train.py` (1697), and `impala_learner.py` (1578).
- `train.py` is now a smaller orchestration file, though `_run_minimal_training()` and `main()` remain large. The dominant unresolved hotspot remains `runtime.py`.

### Follow-Up Note

- A proposed `StructuredLegalPolicyValueModel` facade-mixin extraction was attempted and backed out before checkpointing because the first mechanical marker matched the base `PolicyValueModel.encode()` block rather than the structured facade. The temporary file was removed, `model.py` was restored to 2024 lines, and the model-focused static/tests passed afterward. This should be retried only with stricter AST- or line-range-based extraction.

## 2026-05-11 - Runtime Opponent-Row Mixin

### Scope

- Moved `_apply_opponent_rows_mask()` and `_apply_opponent_rows_ids()` from `QueueRuntime` into `python/weiss_rl/runtime_opponent_rows.py`.
- Preserved private method names on `QueueRuntime` through `QueueRuntimeOpponentRowsMixin`.
- Kept actor inference routed lazily through `weiss_rl.runtime._actor_inference_model` so tests and downstream code retain the monkeypatchable compatibility hook.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run ruff check python/weiss_rl/runtime.py python/weiss_rl/runtime_opponent_rows.py python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_opponent_sampling.py python/weiss_rl/tests/test_runtime_batching.py python/weiss_rl/tests/test_runtime_teacher_labels.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/runtime.py python/weiss_rl/runtime_opponent_rows.py python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_opponent_sampling.py python/weiss_rl/tests/test_runtime_batching.py python/weiss_rl/tests/test_runtime_teacher_labels.py` | Passed after formatting the runtime files. |
| `uv run mypy python/weiss_rl/runtime.py python/weiss_rl/runtime_opponent_rows.py --show-error-codes --no-error-summary` | Passed after using the dynamic mixin adapter pattern already used by `runtime_opponent_mixin.py`. |
| `uv run python -m pytest python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_opponent_sampling.py python/weiss_rl/tests/test_runtime_batching.py python/weiss_rl/tests/test_runtime_teacher_labels.py -q` | Passed: 71 passed. |
| `uv run python -m pytest python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_opponent_pool.py python/weiss_rl/tests/test_runtime_opponent_sampling.py python/weiss_rl/tests/test_runtime_topology.py python/weiss_rl/tests/test_runtime_shared_transport.py python/weiss_rl/tests/test_runtime_batching.py python/weiss_rl/tests/test_runtime_heuristic_fast_path.py python/weiss_rl/tests/test_runtime_teacher_labels.py -q` | Passed: 104 passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1184 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `QueueRuntimeOpponentRowsMixin`.
- Reduced `python/weiss_rl/runtime.py` from 4458 lines to 4254 lines.

### Behavior Changes

No intended behavior changes. The same private methods still resolve on `QueueRuntime`, and focused tests around mirror rows, snapshot rows, heuristic-public rows, simulator-native heuristic handling, and fallback behavior still pass.

### Remaining Risks and Next Hypotheses

- Remaining very large files are now `runtime.py` (4254 lines), `model.py` (2024), `train.py` (1697), and `impala_learner.py` (1578).
- Runtime central collection, all-heuristic rollout paths, and actor unroll collection remain large. Moving them should wait for explicit parity fixtures around row ordering, masking, recurrent-state advancement, behavior log probabilities, and rollout tensors.

## 2026-05-11 - Runtime Policy-Output Fill Mixin

### Scope

- Moved `_fill_policy_outputs_mask()` and `_fill_policy_outputs_ids()` from `QueueRuntime` into `python/weiss_rl/runtime_policy_outputs.py`.
- Preserved private method names on `QueueRuntime` through `QueueRuntimePolicyOutputMixin`.
- Kept the adapter responsible only for splitting focal rows into model/heuristic paths and delegating opponent rows to the existing opponent-row helper.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run ruff check python/weiss_rl/runtime.py python/weiss_rl/runtime_policy_outputs.py python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_opponent_sampling.py python/weiss_rl/tests/test_runtime_heuristic_actor_outputs.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/runtime.py python/weiss_rl/runtime_policy_outputs.py python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_opponent_sampling.py python/weiss_rl/tests/test_runtime_heuristic_actor_outputs.py` | Passed after formatting the runtime files. |
| `uv run mypy python/weiss_rl/runtime.py python/weiss_rl/runtime_policy_outputs.py --show-error-codes --no-error-summary` | Passed. |
| `uv run python -m pytest python/weiss_rl/tests/test_runtime.py -k "fill_policy_outputs_ids" -q` | Passed: 3 passed, 27 deselected. |
| `uv run python -m pytest python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_opponent_sampling.py python/weiss_rl/tests/test_runtime_heuristic_actor_outputs.py python/weiss_rl/tests/test_runtime_batching.py python/weiss_rl/tests/test_runtime_teacher_labels.py -q` | Passed: 75 passed. |
| `uv run python -m pytest python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_opponent_pool.py python/weiss_rl/tests/test_runtime_opponent_sampling.py python/weiss_rl/tests/test_runtime_topology.py python/weiss_rl/tests/test_runtime_shared_transport.py python/weiss_rl/tests/test_runtime_batching.py python/weiss_rl/tests/test_runtime_heuristic_fast_path.py python/weiss_rl/tests/test_runtime_heuristic_actor_outputs.py python/weiss_rl/tests/test_runtime_teacher_labels.py -q` | Passed: 108 passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1184 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `QueueRuntimePolicyOutputMixin`.
- Reduced `python/weiss_rl/runtime.py` from 4254 lines to 4117 lines.

### Behavior Changes

No intended behavior changes. Focal model rows, focal heuristic-public rows, and opponent rows still route through the same lower-level helpers, and the heuristic actor output-fill characterization tests still pass.

### Remaining Risks and Next Hypotheses

- Remaining very large files are now `runtime.py` (4117 lines), `model.py` (2024), `train.py` (1697), and `impala_learner.py` (1578).
- Runtime central collection and actor unroll collection remain the next major blockers. Add explicit parity fixtures before extracting those loops.

## 2026-05-11 - Runtime Central-Row Mixin

### Scope

- Added `python/weiss_rl/tests/test_runtime_central_rows.py` before the production move to characterize sparse row ordering across actors, value-only inference, recurrent hidden-state advancement, and all-row forward scattering.
- Moved `_central_value_actor_rows()`, `_central_value_and_advance_actor_rows()`, `_central_advance_actor_rows()`, and `_central_forward_all_rows()` from `QueueRuntime` into `python/weiss_rl/runtime_central_rows.py`.
- Preserved private method names on `QueueRuntime` through `QueueRuntimeCentralRowsMixin`.
- Kept actor inference and concatenated legal-action construction routed lazily through `weiss_rl.runtime` so existing wrapper and monkeypatch contracts remain intact.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run ruff check python/weiss_rl/tests/test_runtime_central_rows.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/tests/test_runtime_central_rows.py` | Passed. |
| `uv run mypy python/weiss_rl/tests/test_runtime_central_rows.py --show-error-codes --no-error-summary` | Passed. |
| `uv run python -m pytest python/weiss_rl/tests/test_runtime_central_rows.py -q` | Passed before the production move: 4 passed. |
| `uv run ruff check python/weiss_rl/runtime.py python/weiss_rl/runtime_central_rows.py python/weiss_rl/tests/test_runtime_central_rows.py python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_opponent_sampling.py python/weiss_rl/tests/test_runtime_batching.py python/weiss_rl/tests/test_runtime_teacher_labels.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/runtime.py python/weiss_rl/runtime_central_rows.py python/weiss_rl/tests/test_runtime_central_rows.py python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_opponent_sampling.py python/weiss_rl/tests/test_runtime_batching.py python/weiss_rl/tests/test_runtime_teacher_labels.py` | Passed after formatting `runtime.py`. |
| `uv run mypy python/weiss_rl/runtime.py python/weiss_rl/runtime_central_rows.py python/weiss_rl/tests/test_runtime_central_rows.py --show-error-codes --no-error-summary` | Passed. |
| `uv run python -m pytest python/weiss_rl/tests/test_runtime_central_rows.py python/weiss_rl/tests/test_runtime.py -k "central_value or central_forward or central_advance or advance_hidden_only or bootstrap_values" -q` | Passed: 6 passed, 28 deselected. |
| `uv run python -m pytest python/weiss_rl/tests/test_runtime_central_rows.py python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_opponent_sampling.py python/weiss_rl/tests/test_runtime_batching.py python/weiss_rl/tests/test_runtime_teacher_labels.py -q` | Passed: 75 passed. |
| `uv run python -m pytest python/weiss_rl/tests/test_runtime_central_rows.py python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_opponent_pool.py python/weiss_rl/tests/test_runtime_opponent_sampling.py python/weiss_rl/tests/test_runtime_topology.py python/weiss_rl/tests/test_runtime_shared_transport.py python/weiss_rl/tests/test_runtime_batching.py python/weiss_rl/tests/test_runtime_heuristic_fast_path.py python/weiss_rl/tests/test_runtime_heuristic_actor_outputs.py python/weiss_rl/tests/test_runtime_teacher_labels.py -q` | Passed: 112 passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1188 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `QueueRuntimeCentralRowsMixin`.
- Added central-row characterization coverage.
- Reduced `python/weiss_rl/runtime.py` from 4117 lines to 3896 lines.

### Behavior Changes

No intended behavior changes. The same private methods still resolve on `QueueRuntime`, and the new tests pin row ordering, value scattering, hidden-state updates, and all-row forward outputs for the extracted methods.

### Remaining Risks and Next Hypotheses

- Remaining very large files are now `runtime.py` (3896 lines), `model.py` (2024), `train.py` (1697), and `impala_learner.py` (1578).
- Central rollout collection and actor unroll loops remain the largest runtime blockers. Do not move those loops without stronger same-seed rollout parity fixtures around masking, recurrent state, behavior log-probabilities, rewards, done/truncation handling, and legal-action ordering.

## 2026-05-11 - Runtime Central-Opponent Mixin

### Scope

- Moved `_overwrite_central_outputs_with_configured_opponents()`, `_overwrite_central_outputs_with_opponents()`, and `_overwrite_central_outputs_with_batched_opponents()` from `QueueRuntime` into `python/weiss_rl/runtime_central_opponents.py`.
- Preserved private method names on `QueueRuntime` through `QueueRuntimeCentralOpponentMixin`.
- Reused existing opponent-output tests covering snapshot model rows, batched heuristic-public rows, and simulator-native heuristic-public rows.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run ruff check python/weiss_rl/runtime.py python/weiss_rl/runtime_central_opponents.py python/weiss_rl/tests/test_runtime_opponent_sampling.py python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_central_rows.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/runtime.py python/weiss_rl/runtime_central_opponents.py python/weiss_rl/tests/test_runtime_opponent_sampling.py python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_central_rows.py` | Passed after formatting `runtime.py`. |
| `uv run mypy python/weiss_rl/runtime.py python/weiss_rl/runtime_central_opponents.py --show-error-codes --no-error-summary` | Passed. |
| `uv run python -m pytest python/weiss_rl/tests/test_runtime_opponent_sampling.py -k "overwrite_central_outputs or simulator_native" -q` | Passed: 5 passed, 16 deselected. |
| `uv run ruff check python/weiss_rl/runtime.py python/weiss_rl/runtime_central_opponents.py python/weiss_rl/tests/test_runtime_opponent_sampling.py python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_central_rows.py python/weiss_rl/tests/test_runtime_batching.py python/weiss_rl/tests/test_runtime_teacher_labels.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/runtime.py python/weiss_rl/runtime_central_opponents.py python/weiss_rl/tests/test_runtime_opponent_sampling.py python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_central_rows.py python/weiss_rl/tests/test_runtime_batching.py python/weiss_rl/tests/test_runtime_teacher_labels.py` | Passed. |
| `uv run mypy python/weiss_rl/runtime.py python/weiss_rl/runtime_central_opponents.py --show-error-codes --no-error-summary` | Passed. |
| `uv run python -m pytest python/weiss_rl/tests/test_runtime_central_rows.py python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_opponent_sampling.py python/weiss_rl/tests/test_runtime_batching.py python/weiss_rl/tests/test_runtime_teacher_labels.py -q` | Passed: 75 passed. |
| `uv run python -m pytest python/weiss_rl/tests/test_runtime_central_rows.py python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_opponent_pool.py python/weiss_rl/tests/test_runtime_opponent_sampling.py python/weiss_rl/tests/test_runtime_topology.py python/weiss_rl/tests/test_runtime_shared_transport.py python/weiss_rl/tests/test_runtime_batching.py python/weiss_rl/tests/test_runtime_heuristic_fast_path.py python/weiss_rl/tests/test_runtime_heuristic_actor_outputs.py python/weiss_rl/tests/test_runtime_teacher_labels.py -q` | Passed: 112 passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1188 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `QueueRuntimeCentralOpponentMixin`.
- Reduced `python/weiss_rl/runtime.py` from 3896 lines to 3616 lines.

### Behavior Changes

No intended behavior changes. The same private methods still resolve on `QueueRuntime`, and the moved logic is covered by existing central-opponent model, heuristic-public, and simulator-native runtime tests.

### Remaining Risks and Next Hypotheses

- Remaining very large files are now `runtime.py` (3616 lines), `model.py` (2024), `train.py` (1697), and `impala_learner.py` (1578).
- Runtime collection and actor-unroll loops remain behavior-sensitive. Add same-seed rollout parity before moving those larger loops.

## 2026-05-11 - Runtime Opponent-Pool Refresh Mixin

### Scope

- Moved `refresh_opponent_pool()` from `QueueRuntime` into `QueueRuntimeOpponentMixin`.
- Preserved the `QueueRuntime.refresh_opponent_pool()` method surface through mixin inheritance.
- Kept registry loading, champion demotion, promotion-gated recent reservoir sizing, timeout quarantine, hard-negative selection, resident model loading, and process-collector refresh notification behavior unchanged.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run ruff check python/weiss_rl/runtime.py python/weiss_rl/runtime_opponent_mixin.py python/weiss_rl/tests/test_runtime_opponent_pool.py python/weiss_rl/tests/test_runtime_opponents.py` | Passed after adding the moved registry import and removing the stale runtime import. |
| `uv run ruff format --check python/weiss_rl/runtime.py python/weiss_rl/runtime_opponent_mixin.py python/weiss_rl/tests/test_runtime_opponent_pool.py python/weiss_rl/tests/test_runtime_opponents.py` | Passed after formatting `runtime.py`. |
| `uv run mypy python/weiss_rl/runtime.py python/weiss_rl/runtime_opponent_mixin.py --show-error-codes --no-error-summary` | Passed after annotating empty policy-id tuples in the moved method. |
| `uv run python -m pytest python/weiss_rl/tests/test_runtime_opponent_pool.py python/weiss_rl/tests/test_runtime_opponents.py -q` | Passed: 25 passed. |
| `uv run python -m pytest python/weiss_rl/tests/test_runtime_central_rows.py python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_opponent_pool.py python/weiss_rl/tests/test_runtime_opponents.py python/weiss_rl/tests/test_runtime_opponent_sampling.py python/weiss_rl/tests/test_runtime_topology.py python/weiss_rl/tests/test_runtime_shared_transport.py python/weiss_rl/tests/test_runtime_batching.py python/weiss_rl/tests/test_runtime_heuristic_fast_path.py python/weiss_rl/tests/test_runtime_heuristic_actor_outputs.py python/weiss_rl/tests/test_runtime_teacher_labels.py -q` | Passed: 126 passed. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1188 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Moved opponent-pool refresh orchestration into the existing opponent mixin.
- Reduced `python/weiss_rl/runtime.py` from 3616 lines to 3504 lines.

### Behavior Changes

No intended behavior changes. The focused opponent-pool tests still cover fixed B1 anchor exclusion/residency, promotion-gated reservoirs, stale assignments, effective-update champion age, timeout quarantine, existing champion reservoirs, and stale champion demotion.

### Remaining Risks and Next Hypotheses

- Remaining very large files are now `runtime.py` (3504 lines), `model.py` (2024), `train.py` (1697), and `impala_learner.py` (1578).
- The remaining runtime bulk is dominated by rollout collection and heuristic rollout loops. Add same-seed rollout parity fixtures before moving those blocks, or pivot to `model.py`/`impala_learner.py` with direct characterization tests.

## 2026-05-11 - Structured Policy/Value Facade Mixin

### Scope

- Added `python/weiss_rl/model_policy_value_facade.py`.
- Moved the structured policy/value facade methods from `StructuredLegalPolicyValueModel` into `StructuredLegalPolicyValueFacadeMixin`.
- Kept `StructuredLegalPolicyValueModel`, its constructor, public import path, and state-dict/module layout anchored in `python/weiss_rl/model.py`.
- Kept packed-action sampling routed lazily through `weiss_rl.model._sample_packed_action_scores` so the existing compatibility wrapper remains monkeypatchable.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run ruff check python/weiss_rl/model.py python/weiss_rl/model_policy_value_facade.py python/weiss_rl/tests/test_contracts.py python/weiss_rl/tests/test_runtime.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/model.py python/weiss_rl/model_policy_value_facade.py python/weiss_rl/tests/test_contracts.py python/weiss_rl/tests/test_runtime.py` | Passed after formatting `model.py`. |
| `uv run mypy python/weiss_rl/model.py python/weiss_rl/model_policy_value_facade.py --show-error-codes --no-error-summary` | Passed after annotating the mixin's compiled trunk attributes. |
| `uv run python -m pytest python/weiss_rl/tests/test_contracts.py -k "structured_legal_policy_value_model or factorized or forward_trunk" python/weiss_rl/tests/test_runtime.py::test_apply_policy_rows_ids_prefers_factorized_structured_sampler python/weiss_rl/tests/test_runtime.py::test_sample_packed_action_scores_falls_back_to_last_candidate_when_cdf_undershoots -q` | Passed: 15 passed, 37 deselected. |
| `uv run python -m pytest python/weiss_rl/tests/test_model_action_plans.py python/weiss_rl/tests/test_model_action_tables.py python/weiss_rl/tests/test_model_candidate_components.py python/weiss_rl/tests/test_model_candidate_partitioning.py python/weiss_rl/tests/test_model_candidate_projection.py python/weiss_rl/tests/test_model_feature_gathering.py python/weiss_rl/tests/test_model_layers.py python/weiss_rl/tests/test_model_loading.py python/weiss_rl/tests/test_model_observation_contract.py python/weiss_rl/tests/test_model_public_heuristics.py python/weiss_rl/tests/test_model_sampling.py python/weiss_rl/tests/test_model_tensor_ops.py python/weiss_rl/tests/test_model_typed_encoder.py python/weiss_rl/tests/test_contracts.py -k "model or structured_legal_policy_value_model or factorized or structured_v2 or public_heuristic" python/weiss_rl/tests/test_runtime.py::test_apply_policy_rows_ids_prefers_factorized_structured_sampler -q` | Passed: 104 passed, 15 deselected. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1188 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Added `StructuredLegalPolicyValueFacadeMixin`.
- Reduced `python/weiss_rl/model.py` from 2024 lines to 1482 lines.

### Behavior Changes

No intended behavior changes. The focused structured model tests still cover factorized sampling/evaluation, canonical action-id behavior, chunking parity, packed trunk forwarding, public heuristic scoring, and runtime factorized sampler dispatch.

### Remaining Risks and Next Hypotheses

- Remaining very large files are now `runtime.py` (3504 lines), `train.py` (1697), `impala_learner.py` (1578), `test_runtime.py` (1494), and `model.py` (1482).
- `model.py` is now below the current top production hotspots. Continue with `impala_learner.py` or `train.py`, or add same-seed rollout parity fixtures before moving runtime collection loops.

## 2026-05-11 - Runtime Collection Split

### Scope

- Moved the central batched actor-unroll loop from `QueueRuntime` into `python/weiss_rl/runtime_central_collection.py`.
- Moved all-heuristic native/fast rollout paths into `python/weiss_rl/runtime_heuristic_rollouts.py`.
- Moved the generic single-actor rollout path into `python/weiss_rl/runtime_actor_unroll.py`.
- Preserved the same private method names on `QueueRuntime` through mixin inheritance.
- Kept actor inference resolved lazily through `weiss_rl.runtime._actor_inference_model` to preserve existing monkeypatch and wrapper identity tests.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run ruff check python/weiss_rl/runtime.py python/weiss_rl/runtime_central_collection.py python/weiss_rl/runtime_heuristic_rollouts.py python/weiss_rl/runtime_actor_unroll.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/runtime.py python/weiss_rl/runtime_central_collection.py python/weiss_rl/runtime_heuristic_rollouts.py python/weiss_rl/runtime_actor_unroll.py` | Passed. |
| `uv run mypy python/weiss_rl/runtime.py python/weiss_rl/runtime_central_collection.py python/weiss_rl/runtime_heuristic_rollouts.py python/weiss_rl/runtime_actor_unroll.py --show-error-codes --no-error-summary` | Passed. |
| `uv run python -m pytest python/weiss_rl/tests/test_runtime.py -k "collect_all_heuristic_ids or sync_actor_batch or runtime" python/weiss_rl/tests/test_runtime_heuristic_fast_path.py python/weiss_rl/tests/test_runtime_shared_transport.py -q` | Passed: 39 passed. |
| `uv run python -m pytest python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_central_rows.py python/weiss_rl/tests/test_runtime_opponent_pool.py python/weiss_rl/tests/test_runtime_opponents.py python/weiss_rl/tests/test_runtime_opponent_sampling.py python/weiss_rl/tests/test_runtime_topology.py python/weiss_rl/tests/test_runtime_shared_transport.py python/weiss_rl/tests/test_runtime_batching.py python/weiss_rl/tests/test_runtime_heuristic_fast_path.py python/weiss_rl/tests/test_runtime_heuristic_actor_outputs.py python/weiss_rl/tests/test_runtime_teacher_labels.py -q` | Passed: 126 passed. |
| `git diff --check` | Passed with the known CRLF warnings for touched documentation/test files. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1188 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Reduced `python/weiss_rl/runtime.py` from 3385 lines at the start of this resumed checkpoint to 1940 lines.
- Avoided leaving one new oversized collection module by splitting the moved rollout code into 583-line, 568-line, and 423-line modules.

### Behavior Changes

No intended behavior changes. This was a mechanical move of the existing collection methods behind mixins. Runtime tests covering central collection, heuristic fast-path gates, shared-transport dispatch, opponent routing, batching, teacher labels, and runtime integration still pass.

### Remaining Risks and Next Hypotheses

- Remaining largest production files are now `runtime.py` (1940), `train.py` (1609), `model.py` (1398), and `impala_learner.py` (1315).
- Runtime still owns construction, checkpoint/snapshot publication, actor role assignment, heuristic action routing, and learner-batch/reporting adapters.
- `train.py` is now the next best production target for a low-risk decomposition pass unless a narrower runtime constructor/helper extraction is chosen.

## 2026-05-11 - IMPALA Update-Loop Mixin

### Scope

- Added `python/weiss_rl/learners/impala_update_loop.py`.
- Moved `ImpalaLearner.update()` and `ImpalaLearner.auxiliary_update()` out of `python/weiss_rl/learners/impala_learner.py`.
- Kept `_loss_and_metrics_with_context()` and `_auxiliary_loss_and_metrics()` anchored in `impala_learner.py` because they contain behavior-sensitive V-trace, packed-legality, factorized-policy, and structured-teacher math.
- Kept `_batch_value()` and V-trace diagnostics resolved lazily through `weiss_rl.learners.impala_learner` so compatibility wrapper behavior stays intact.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run ruff check python/weiss_rl/learners/impala_learner.py python/weiss_rl/learners/impala_update_loop.py python/weiss_rl/tests/test_impala_learner.py python/weiss_rl/tests/test_learner_update_bookkeeping.py python/weiss_rl/tests/test_learner_faults.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/learners/impala_learner.py python/weiss_rl/learners/impala_update_loop.py python/weiss_rl/tests/test_impala_learner.py python/weiss_rl/tests/test_learner_update_bookkeeping.py python/weiss_rl/tests/test_learner_faults.py` | Passed. |
| `uv run mypy python/weiss_rl/learners/impala_learner.py python/weiss_rl/learners/impala_update_loop.py --show-error-codes --no-error-summary` | Passed after narrowing the optional active timing metrics field. |
| `uv run python -m pytest python/weiss_rl/tests/test_impala_learner.py -k "checkpoint_metadata or nonfinite or mixed_precision or auxiliary_update or amp_grad_overflow" python/weiss_rl/tests/test_learner_update_bookkeeping.py python/weiss_rl/tests/test_learner_faults.py -q` | Passed: 15 passed, 51 deselected. |
| `uv run python -m pytest python/weiss_rl/tests/test_impala_learner.py python/weiss_rl/tests/test_learner_action_logp.py python/weiss_rl/tests/test_learner_batch_fields.py python/weiss_rl/tests/test_learner_bootstrap.py python/weiss_rl/tests/test_learner_faults.py python/weiss_rl/tests/test_learner_legal_fields.py python/weiss_rl/tests/test_learner_logging.py python/weiss_rl/tests/test_learner_packed_rows.py python/weiss_rl/tests/test_learner_structured_auxiliary.py python/weiss_rl/tests/test_learner_structured_policy_metrics.py python/weiss_rl/tests/test_learner_tensor_ops.py python/weiss_rl/tests/test_learner_update_bookkeeping.py python/weiss_rl/tests/test_learner_vtrace_diagnostics.py python/weiss_rl/tests/test_learner_vtrace_torch.py -q` | Passed: 125 passed. |
| `uv run python python/scripts/verify_repo.py` | Passed after the later runtime collection split: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1188 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Reduced `python/weiss_rl/learners/impala_learner.py` to 1315 lines in the current live tree.
- Made update/auxiliary update orchestration easier to read without moving the high-risk loss bodies.

### Behavior Changes

No intended behavior changes. The public `ImpalaLearner` import path remains unchanged, and update-loop tests still cover checkpoint metadata, numeric faults, mixed precision, auxiliary updates, AMP overflow handling, update bookkeeping, and learner fault handling.

### Remaining Risks and Next Hypotheses

- `_loss_and_metrics_with_context()` remains the largest behavior-sensitive learner body. Move it only with narrower characterization across V-trace targets, masks, packed rows, factorized log-probs, and teacher auxiliary contexts.
- `train.py` and `runtime.py` now offer better next-step structural payoff than deeper IMPALA math movement.

## 2026-05-11 - Training Minimal Loop Extraction

### Scope

- Added `python/weiss_rl/training/minimal_loop.py`.
- Moved the canonical single-node training loop body out of `python/scripts/train.py`.
- Kept `scripts.train._run_minimal_training()` as the public compatibility wrapper for tests and existing scripts.
- Introduced `MinimalTrainingHooks` so script-level monkeypatch points remain effective for behavior-sensitive tests, especially `_ensure_noleague_baseline_anchor` and `QueueRuntime`.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run ruff check python/scripts/train.py python/weiss_rl/training/minimal_loop.py python/weiss_rl/tests/test_snapshot_registry.py python/weiss_rl/tests/test_train_stall_monitor.py python/weiss_rl/tests/test_script_entrypoint_smokes.py` | Passed. |
| `uv run ruff format --check python/scripts/train.py python/weiss_rl/training/minimal_loop.py python/weiss_rl/tests/test_snapshot_registry.py python/weiss_rl/tests/test_train_stall_monitor.py python/weiss_rl/tests/test_script_entrypoint_smokes.py` | Passed after formatting `train.py` and `minimal_loop.py`. |
| `uv run mypy python/weiss_rl/training/minimal_loop.py --show-error-codes --no-error-summary` | Passed after removing a redundant `spec_bundle` cast. |
| `uv run python -m pytest python/weiss_rl/tests/test_snapshot_registry.py::test_run_minimal_training_bootstraps_noleague_baseline_before_env_start python/weiss_rl/tests/test_train_stall_monitor.py python/weiss_rl/tests/test_script_entrypoint_smokes.py -q` | Passed: 31 passed, 14 dependency warnings. |
| `uv run python -m pytest python/weiss_rl/tests/test_snapshot_registry.py python/weiss_rl/tests/test_train_stall_monitor.py python/weiss_rl/tests/test_training_execution.py python/weiss_rl/tests/test_training_batches.py python/weiss_rl/tests/test_training_checkpoint_writers.py python/weiss_rl/tests/test_script_entrypoint_smokes.py -q` | Passed: 89 passed, 14 dependency warnings. |
| `git diff --check` | Passed with the known CRLF warnings for touched documentation/test files. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1188 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Reduced `python/scripts/train.py` from 1609 lines to 1248 lines.
- Added a 543-line focused training loop module instead of continuing to grow the script entrypoint.

### Behavior Changes

No intended behavior changes. The wrapper still constructs hooks from the current `scripts.train` globals, preserving existing tests that monkeypatch B1 bootstrap and runtime construction before invoking `_run_minimal_training()`.

### Remaining Risks and Next Hypotheses

- Run the full verifier after this checkpoint.
- Remaining `train.py` size is now mostly CLI/setup, compatibility wrappers, promotion/dev-eval wrappers, and manifest augmentation.
- Next train-side reduction should target setup/context construction or compatibility wrappers only after checking private imports in `test_snapshot_registry.py` and `test_train_stall_monitor.py`.

## 2026-05-11 - Runtime Lifecycle Mixin

### Scope

- Added `python/weiss_rl/runtime_lifecycle.py`.
- Moved `QueueRuntime.close()`, `QueueRuntime.maybe_publish_snapshot()`, `QueueRuntime.reset_outcome_tracker()`, and `QueueRuntime._league_reference_update()` out of `runtime.py`.
- Preserved method names through `QueueRuntimeLifecycleMixin`.
- Kept rollout, legal-action ordering, sampling, reward, done/truncation, and observation behavior untouched.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run ruff check python/weiss_rl/runtime.py python/weiss_rl/runtime_lifecycle.py python/weiss_rl/tests/test_runtime_opponent_pool.py python/weiss_rl/tests/test_runtime_topology.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/runtime.py python/weiss_rl/runtime_lifecycle.py python/weiss_rl/tests/test_runtime_opponent_pool.py python/weiss_rl/tests/test_runtime_topology.py` | Passed. |
| `uv run mypy python/weiss_rl/runtime.py python/weiss_rl/runtime_lifecycle.py --show-error-codes --no-error-summary` | Passed after annotating mixin `self` as dynamic. |
| `uv run python -m pytest python/weiss_rl/tests/test_runtime_opponent_pool.py::test_maybe_publish_snapshot_tracks_effective_update_for_reused_weights python/weiss_rl/tests/test_runtime_topology.py -q` | Passed: 14 passed. |
| `uv run python -m pytest python/weiss_rl/tests/test_runtime_opponent_pool.py python/weiss_rl/tests/test_runtime_topology.py python/weiss_rl/tests/test_runtime_opponent_sampling.py -q` | Passed: 45 passed. |
| `git diff --check` | Passed with the known CRLF warnings for touched documentation/test files. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1188 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Reduced `python/weiss_rl/runtime.py` from 1940 lines to 1825 lines.
- Added a 126-line lifecycle/snapshot publication mixin.

### Behavior Changes

No intended behavior changes. The snapshot effective-update characterization test still passes, and topology tests still cover runtime close/shutdown behavior.

### Remaining Risks and Next Hypotheses

- Run the full verifier after this checkpoint.
- Remaining `runtime.py` size is now mostly constructor/setup, collection entrypoints, actor state/env/model construction, actor role assignment, heuristic action routing, and compatibility wrappers.

## 2026-05-11 - Paper-Readiness Guardrail Split

### Scope

- Added `python/weiss_rl/eval/paper_readiness_fields.py`.
- Added `python/weiss_rl/eval/paper_readiness_guardrails.py`.
- Moved field-level JSON/object/manifest validators, companion-file comparisons, relative artifact path checks, final-eval guardrails, focal-policy inference, matchup normalization, diagnostics loading, truncation checks, seat-bias checks, and baseline win-rate checks out of `python/weiss_rl/eval/paper_readiness.py`.
- Kept the public readiness API unchanged: `build_paper_readiness_summary()` and `write_paper_readiness_json()`.
- Kept moved helpers imported back under the same private names used by the readiness orchestrator and artifact-contract auditor.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run ruff check python/weiss_rl/eval/paper_readiness.py python/weiss_rl/eval/paper_readiness_fields.py python/weiss_rl/eval/paper_readiness_guardrails.py python/weiss_rl/tests/test_paper_readiness.py python/weiss_rl/tests/test_paper_readiness_fixture.py python/weiss_rl/tests/test_entrypoints.py python/weiss_rl/tests/test_script_entrypoint_smokes.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/eval/paper_readiness.py python/weiss_rl/eval/paper_readiness_fields.py python/weiss_rl/eval/paper_readiness_guardrails.py python/weiss_rl/tests/test_paper_readiness.py python/weiss_rl/tests/test_paper_readiness_fixture.py python/weiss_rl/tests/test_entrypoints.py python/weiss_rl/tests/test_script_entrypoint_smokes.py` | Passed. |
| `uv run mypy python/weiss_rl/eval/paper_readiness.py python/weiss_rl/eval/paper_readiness_fields.py python/weiss_rl/eval/paper_readiness_guardrails.py --show-error-codes --no-error-summary` | Passed. |
| `uv run python -m pytest python/weiss_rl/tests/test_paper_readiness.py python/weiss_rl/tests/test_paper_readiness_fixture.py python/weiss_rl/tests/test_entrypoints.py -k "paper_readiness" python/weiss_rl/tests/test_script_entrypoint_smokes.py::test_write_paper_readiness_fixture_entrypoint_writes_ready_run -q` | Passed: 17 passed, 28 deselected. |
| `uv run python -m pytest python/weiss_rl/tests/test_paper_readiness.py python/weiss_rl/tests/test_paper_readiness_fixture.py python/weiss_rl/tests/test_entrypoints.py -k "paper_readiness" python/weiss_rl/tests/test_script_entrypoint_smokes.py::test_write_paper_readiness_fixture_entrypoint_writes_ready_run python/weiss_rl/tests/test_artifact_contract.py -q` | Failed before running tests because `python/weiss_rl/tests/test_artifact_contract.py` does not exist in this checkout. |
| `uv run python -m pytest python/weiss_rl/tests/test_artifact_hygiene.py python/weiss_rl/tests/test_paper_readiness.py python/weiss_rl/tests/test_paper_readiness_fixture.py -q` | Passed: 20 passed. |
| `git diff --check` | Passed with the known CRLF warnings for touched documentation/test files. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1188 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Reduced `python/weiss_rl/eval/paper_readiness.py` from 1337 lines in the pre-checkpoint audit to 737 lines.
- Added a 430-line final-eval guardrail module and a 208-line field-validator module.

### Behavior Changes

No intended behavior changes. The readiness summary payload shape, focal-policy inference behavior, guardrail thresholds, artifact contract checks, manifest field validation, and public CLI-facing API remain covered by the existing readiness and entrypoint tests.

### Remaining Risks and Next Hypotheses

- Remaining largest production files are `runtime.py` (1825), `model.py` (1398), `impala_learner.py` (1315), and `train.py` (1248).
- The next source checkpoint should target runtime batch-builder/method-adapter code or a carefully characterized learner/model slice; avoid changing eval definitions or payoff/reporting semantics.

## 2026-05-11 - Runtime Central Policy-Row Mixin

### Scope

- Moved central focal policy-row sampling from `QueueRuntime` into `QueueRuntimeCentralRowsMixin`.
- Kept `_central_sample_policy_rows_ids_model()`, `_central_sample_policy_rows_ids_heuristic()`, and `_central_sample_policy_rows_ids()` available on `QueueRuntime`.
- Kept compatibility-sensitive helper resolution lazy through `weiss_rl.runtime` for `_actor_inference_model`, `_require_ids_offsets`, `_optional_legal_action_meta`, and `_slice_packed_rows_with_meta`.
- Preserved legal-action ordering, packed metadata slicing, heuristic/model row splitting, RNG sampling, hidden-state updates, and central batch-timer labels.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run ruff check python/weiss_rl/runtime.py python/weiss_rl/runtime_central_rows.py python/weiss_rl/tests/test_runtime_central_rows.py python/weiss_rl/tests/test_runtime.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/runtime.py python/weiss_rl/runtime_central_rows.py python/weiss_rl/tests/test_runtime_central_rows.py python/weiss_rl/tests/test_runtime.py` | Passed. |
| `uv run mypy python/weiss_rl/runtime.py python/weiss_rl/runtime_central_rows.py --show-error-codes --no-error-summary` | Passed after typing the moved mixin methods with dynamic `self`, matching other runtime mixins. |
| `uv run python -m pytest python/weiss_rl/tests/test_runtime_central_rows.py python/weiss_rl/tests/test_runtime.py -k "central_sample_policy_rows_ids or central or heuristic_actor_backend or apply_policy_rows_ids" -q` | Passed: 8 passed, 26 deselected. |
| `uv run python -m pytest python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_central_rows.py python/weiss_rl/tests/test_runtime_heuristic_actor_outputs.py python/weiss_rl/tests/test_runtime_heuristic_fast_path.py python/weiss_rl/tests/test_runtime_shared_transport.py python/weiss_rl/tests/test_runtime_batching.py -q` | Passed: 60 passed. |
| `uv run python -m pytest python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_central_rows.py python/weiss_rl/tests/test_runtime_opponent_pool.py python/weiss_rl/tests/test_runtime_opponents.py python/weiss_rl/tests/test_runtime_opponent_sampling.py python/weiss_rl/tests/test_runtime_topology.py python/weiss_rl/tests/test_runtime_shared_transport.py python/weiss_rl/tests/test_runtime_batching.py python/weiss_rl/tests/test_runtime_heuristic_fast_path.py python/weiss_rl/tests/test_runtime_heuristic_actor_outputs.py python/weiss_rl/tests/test_runtime_teacher_labels.py -q` | Passed: 126 passed. |
| `git diff --check` | Passed with the known CRLF warnings for touched documentation/test files. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1188 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Reduced `python/weiss_rl/runtime.py` from 1825 lines to 1557 lines.
- Grew `python/weiss_rl/runtime_central_rows.py` from 241 lines to 531 lines, still focused on central actor/policy row operations.

### Behavior Changes

No intended behavior changes. This was a method move into an existing mixin, with the same private methods still present on `QueueRuntime`.

### Remaining Risks and Next Hypotheses

- Remaining largest production files are `runtime.py` (1557), `model.py` (1398), `impala_learner.py` (1315), and `train.py` (1248).
- Further runtime movement should target constructor/setup or small method adapters only after checking direct tests, because the remaining runtime body is more coupled.

## 2026-05-11 - Base Policy/Value Model Mixin

### Scope

- Added `python/weiss_rl/model_base.py`.
- Moved the base recurrent and seat-aware `PolicyValueModel` methods into `PolicyValueModelBaseMixin`.
- Kept `PolicyValueModel` and `StructuredLegalPolicyValueModel` defined in `python/weiss_rl/model.py`.
- Preserved hidden-state initialization, GRU/feedforward recurrent behavior, seat-aware recurrent updates, inplace seat-hidden updates, base dense forward paths, base packed-action unsupported errors, typed observation encoder construction, and no-op public heuristic methods.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run ruff check python/weiss_rl/model.py python/weiss_rl/model_base.py python/weiss_rl/tests/test_contracts.py python/weiss_rl/tests/test_model_loading.py python/weiss_rl/tests/test_model_typed_encoder.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/model.py python/weiss_rl/model_base.py python/weiss_rl/tests/test_contracts.py python/weiss_rl/tests/test_model_loading.py python/weiss_rl/tests/test_model_typed_encoder.py` | Passed. |
| `uv run mypy python/weiss_rl/model.py python/weiss_rl/model_base.py --show-error-codes --no-error-summary` | Passed. |
| `uv run python -m pytest python/weiss_rl/tests/test_contracts.py -k "PolicyValueModel or seat_aware or initial_hidden or forward_sequence or structured_legal_policy_value_model" python/weiss_rl/tests/test_model_typed_encoder.py python/weiss_rl/tests/test_model_loading.py -q` | Passed: 23 passed, 37 deselected. |
| `uv run python -m pytest python/weiss_rl/tests/test_model_action_plans.py python/weiss_rl/tests/test_model_action_tables.py python/weiss_rl/tests/test_model_candidate_components.py python/weiss_rl/tests/test_model_candidate_partitioning.py python/weiss_rl/tests/test_model_candidate_projection.py python/weiss_rl/tests/test_model_feature_gathering.py python/weiss_rl/tests/test_model_layers.py python/weiss_rl/tests/test_model_loading.py python/weiss_rl/tests/test_model_observation_contract.py python/weiss_rl/tests/test_model_public_heuristics.py python/weiss_rl/tests/test_model_sampling.py python/weiss_rl/tests/test_model_tensor_ops.py python/weiss_rl/tests/test_model_typed_encoder.py python/weiss_rl/tests/test_contracts.py -k "model or PolicyValueModel or structured_legal_policy_value_model or factorized or structured_v2 or public_heuristic or seat_aware" -q` | Passed: 104 passed, 14 deselected. |
| `git diff --check` | Passed with the known CRLF warnings for touched documentation/test files. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1188 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Reduced `python/weiss_rl/model.py` from 1398 lines to 1088 lines.
- Added a 337-line focused base recurrent policy/value mixin.

### Behavior Changes

No intended behavior changes. The public class definitions remain anchored in `weiss_rl.model`; only inherited method bodies moved.

### Remaining Risks and Next Hypotheses

- Remaining largest production files are `runtime.py` (1557), `impala_learner.py` (1315), and `train.py` (1248). `model.py` is now below the top three production hotspots.
- Future model movement should avoid moving public class definitions unless import identity, pickle/module paths, and checkpoint compatibility are deliberately audited.

## 2026-05-11 - IMPALA Auxiliary-Loss Mixin

### Scope

- Added `python/weiss_rl/learners/impala_auxiliary_loss.py`.
- Moved `ImpalaLearner._auxiliary_loss_and_metrics()` into `ImpalaAuxiliaryLossMixin`.
- Kept the public `ImpalaLearner` class in `python/weiss_rl/learners/impala_learner.py`.
- Left `_loss_and_metrics_with_context()` in place because it contains the behavior-sensitive IMPALA/V-trace policy/value loss path.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run ruff check python/weiss_rl/learners/impala_learner.py python/weiss_rl/learners/impala_auxiliary_loss.py python/weiss_rl/tests/test_impala_learner.py python/weiss_rl/tests/test_learner_structured_auxiliary.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/learners/impala_learner.py python/weiss_rl/learners/impala_auxiliary_loss.py python/weiss_rl/tests/test_impala_learner.py python/weiss_rl/tests/test_learner_structured_auxiliary.py` | Passed after formatting `impala_learner.py`. |
| `uv run mypy python/weiss_rl/learners/impala_learner.py python/weiss_rl/learners/impala_auxiliary_loss.py --show-error-codes --no-error-summary` | Passed. |
| `uv run python -m pytest python/weiss_rl/tests/test_impala_learner.py -k "auxiliary_update or teacher_aux or public_heuristic or factorized" python/weiss_rl/tests/test_learner_structured_auxiliary.py -q` | Passed: 33 passed, 27 deselected. |
| `uv run python -m pytest python/weiss_rl/tests/test_impala_learner.py python/weiss_rl/tests/test_learner_action_logp.py python/weiss_rl/tests/test_learner_batch_fields.py python/weiss_rl/tests/test_learner_bootstrap.py python/weiss_rl/tests/test_learner_faults.py python/weiss_rl/tests/test_learner_legal_fields.py python/weiss_rl/tests/test_learner_logging.py python/weiss_rl/tests/test_learner_packed_rows.py python/weiss_rl/tests/test_learner_structured_auxiliary.py python/weiss_rl/tests/test_learner_structured_policy_metrics.py python/weiss_rl/tests/test_learner_tensor_ops.py python/weiss_rl/tests/test_learner_update_bookkeeping.py python/weiss_rl/tests/test_learner_vtrace_diagnostics.py python/weiss_rl/tests/test_learner_vtrace_torch.py -q` | Passed: 125 passed. |
| `git diff --check` | Passed with the known CRLF warnings for touched documentation/test files. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1188 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Reduced `python/weiss_rl/learners/impala_learner.py` from 1315 lines to 1124 lines.
- Added a 219-line focused auxiliary loss mixin.

### Behavior Changes

No intended behavior changes. The public learner class remains unchanged; only the auxiliary loss method body moved behind inherited mixin behavior.

### Remaining Risks and Next Hypotheses

- Remaining largest production files are `runtime.py` (1557), `train.py` (1248), `structured_teacher_auxiliary.py` (1140), `impala_learner.py` (1124), and `model.py` (1088).
- The remaining `impala_learner.py` body is dominated by `_loss_and_metrics_with_context()`; move it only with stronger characterization across V-trace, masks, packed rows, factorized policy, and teacher auxiliary contexts.

## 2026-05-11 - Structured Teacher-Auxiliary Branch Split

### Scope

- Added `python/weiss_rl/learners/structured_teacher_common.py`.
- Added `python/weiss_rl/learners/structured_teacher_dense.py`.
- Added `python/weiss_rl/learners/structured_teacher_factorized.py`.
- Kept the public `compute_structured_teacher_auxiliary_metrics()` API and `ImpalaLearner` re-export behavior unchanged.
- Moved shared metric defaults and teacher-family coverage accounting into common helpers.
- Moved the dense-mask and factorized teacher-auxiliary branches into focused modules.
- Left the packed branch in `structured_teacher_auxiliary.py` because it still owns the most legal-action-ordering-sensitive path.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_impala_learner.py::test_compute_structured_teacher_auxiliary_metrics_infers_packed_move_source_from_action` before the bug fix | Failed with `NameError: name 'move_source_targets_by_action' is not defined`, confirming a latent packed move-source fallback bug. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_impala_learner.py::test_compute_structured_teacher_auxiliary_metrics_infers_packed_move_source_from_action` after the bug fix | Passed: 1 passed. |
| `uv run ruff check python/weiss_rl/learners/structured_teacher_auxiliary.py python/weiss_rl/learners/structured_teacher_common.py python/weiss_rl/learners/structured_teacher_dense.py python/weiss_rl/learners/structured_teacher_factorized.py python/weiss_rl/tests/test_impala_learner.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/learners/structured_teacher_auxiliary.py python/weiss_rl/learners/structured_teacher_common.py python/weiss_rl/learners/structured_teacher_dense.py python/weiss_rl/learners/structured_teacher_factorized.py python/weiss_rl/tests/test_impala_learner.py` | Passed after formatting. |
| `uv run mypy python/weiss_rl/learners/structured_teacher_auxiliary.py python/weiss_rl/learners/structured_teacher_common.py python/weiss_rl/learners/structured_teacher_dense.py python/weiss_rl/learners/structured_teacher_factorized.py --show-error-codes --no-error-summary` | Passed. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_learner_structured_auxiliary.py` | Passed: 16 passed. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_impala_learner.py -k "compute_structured_teacher_auxiliary_metrics"` | Passed: 14 passed, 47 deselected. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_impala_learner.py` | Passed: 45 passed. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_impala_learner.py python/weiss_rl/tests/test_learner_structured_auxiliary.py python/weiss_rl/tests/test_learner_structured_policy_metrics.py python/weiss_rl/tests/test_learner_packed_rows.py python/weiss_rl/tests/test_learner_action_logp.py python/weiss_rl/tests/test_learner_vtrace_torch.py python/weiss_rl/tests/test_learner_vtrace_diagnostics.py python/weiss_rl/tests/test_training_algorithm_contracts.py` | Passed: 88 passed. |
| `git diff --check` | Passed with the known CRLF warnings for pre-existing touched text/test files. |
| `uv run python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, Ruff format, selected mypy, vulture, 1189 pytest tests passed, 2 skipped, 14 dependency warnings, wrapper dry-runs passed. |

### Changes

- Reduced `python/weiss_rl/learners/structured_teacher_auxiliary.py` from 1140 lines to 522 lines.
- Added a 64-line common helper module.
- Added a 308-line dense branch module.
- Added a 376-line factorized branch module.
- Added a packed move-source characterization test to `python/weiss_rl/tests/test_impala_learner.py`.

### Bug Fixed

- The packed teacher-auxiliary branch inferred move-source labels from `teacher_action` when `teacher_move_source` was absent, but the action-id-to-source-slot lookup was only initialized inside the factorized branch. The new characterization test failed before the fix with `NameError`. The fix initializes the same `catalog_metadata.move_from_slots` lookup inside the packed branch, matching the factorized branch's intended fallback behavior.
- This bug affects only packed teacher move-source auxiliary supervision when `teacher_move_source` is absent and `move_source_coef` is nonzero. It should not affect old thesis results unless those runs used that exact packed auxiliary path without explicit move-source labels; no historical artifacts were modified.

### Behavior Changes

- Intended behavior is preserved except for the confirmed bug fix above.
- The public function name, return shape, metric names, context keys, and IMPALA re-export identity remain unchanged.

### Remaining Risks and Next Hypotheses

- Remaining largest production files are now `runtime.py` (1557), `train.py` (1248), `impala_learner.py` (1124), and `model.py` (1088).
- The remaining packed teacher-auxiliary branch is still legal-action-ordering-sensitive; move it only with dedicated packed parity tests around support fractions, pass fallback, and top-action tie behavior.

## 2026-05-11 - Runtime Heuristic Actor-Row Mixin

### Scope

- Added `python/weiss_rl/runtime_heuristic_actor_rows.py`.
- Moved hidden-state-only advancement, value-and-advance row evaluation, heuristic actor hidden-state tracking policy, and mask/packed heuristic actor-row application from `QueueRuntime` into `QueueRuntimeHeuristicActorRowsMixin`.
- Kept `_advance_hidden_only()`, `_value_and_advance_rows()`, `_should_track_heuristic_actor_hidden_state()`, `_apply_heuristic_actor_rows_mask()`, and `_apply_heuristic_actor_rows_ids()` available on `QueueRuntime`.
- Kept `_actor_inference_model()` resolution lazy through `weiss_rl.runtime` so private compatibility hooks remain monkeypatchable.
- Preserved deterministic heuristic logit writing, behavior value handling, optional hidden tracking, packed sampled-action debug validation, and teacher-policy error behavior.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run ruff check python/weiss_rl/runtime.py python/weiss_rl/runtime_heuristic_actor_rows.py python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_heuristic_actor_outputs.py python/weiss_rl/tests/test_runtime_heuristic_fast_path.py python/weiss_rl/tests/test_runtime_actor_state.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/runtime.py python/weiss_rl/runtime_heuristic_actor_rows.py python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_heuristic_actor_outputs.py python/weiss_rl/tests/test_runtime_heuristic_fast_path.py python/weiss_rl/tests/test_runtime_actor_state.py` | Passed after formatting `runtime.py`. |
| `uv run mypy python/weiss_rl/runtime.py python/weiss_rl/runtime_heuristic_actor_rows.py --show-error-codes --no-error-summary` | Passed after typing moved mixin methods with dynamic `self`, matching other runtime mixins. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_heuristic_actor_outputs.py python/weiss_rl/tests/test_runtime_heuristic_fast_path.py python/weiss_rl/tests/test_runtime_actor_state.py -k "heuristic_actor_backend or apply_heuristic_actor_rows or advance_hidden_only or reset_done_rows or collect_all_heuristic_ids"` | Passed: 11 passed, 30 deselected. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_heuristic_actor_outputs.py python/weiss_rl/tests/test_runtime_heuristic_fast_path.py python/weiss_rl/tests/test_runtime_actor_state.py python/weiss_rl/tests/test_runtime_actor_models.py python/weiss_rl/tests/test_runtime_central_rows.py python/weiss_rl/tests/test_runtime_opponent_pool.py python/weiss_rl/tests/test_runtime_opponents.py python/weiss_rl/tests/test_runtime_opponent_sampling.py python/weiss_rl/tests/test_runtime_topology.py python/weiss_rl/tests/test_runtime_shared_transport.py python/weiss_rl/tests/test_runtime_batching.py python/weiss_rl/tests/test_runtime_teacher_labels.py` | Passed: 135 passed. |

### Changes

- Reduced `python/weiss_rl/runtime.py` from 1557 lines to 1351 lines.
- Added a 225-line focused heuristic actor-row mixin.

### Behavior Changes

No intended behavior changes. This was a method move into a runtime mixin; private method names remain available on `QueueRuntime`.

### Remaining Risks and Next Hypotheses

- Remaining largest production files are now `runtime.py` (1351), `train.py` (1248), `impala_learner.py` (1124), and `model.py` (1088).
- Further runtime movement should prefer constructor/setup or clearly bounded adapters; the remaining runtime body still touches actor role assignment, process startup, metrics, and compatibility wrappers.

## 2026-05-11 - Model Factorized Sampling Move

### Scope

- Moved `_StructuredLegalActionHead._family_condition_input()` and `sample_factorized_packed()` into the existing `StructuredFactorizedScoringMixin` in `python/weiss_rl/model_factorized_scoring.py`.
- Kept `_StructuredLegalActionHead` and all public model classes defined in `python/weiss_rl/model.py`.
- Kept the private `_sample_masked_log_probs()` sampling wrapper resolved lazily through `weiss_rl.model` so tests and monkeypatchable seeded sampling hooks remain compatible.
- Preserved factorized family/arg sampling order, derived seed salts, pass-action fallback, canonical action-id lookup, and behavior-log-prob accumulation.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run ruff check python/weiss_rl/model.py python/weiss_rl/model_factorized_scoring.py python/weiss_rl/tests/test_contracts.py python/weiss_rl/tests/test_model_sampling.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/model.py python/weiss_rl/model_factorized_scoring.py python/weiss_rl/tests/test_contracts.py python/weiss_rl/tests/test_model_sampling.py` | Passed after formatting `model.py`. |
| `uv run mypy python/weiss_rl/model.py python/weiss_rl/model_factorized_scoring.py --show-error-codes --no-error-summary` | Passed. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_contracts.py -k "factorized or structured_legal_policy_value_model" python/weiss_rl/tests/test_model_sampling.py python/weiss_rl/tests/test_model_action_plans.py python/weiss_rl/tests/test_model_action_tables.py` | Passed: 17 passed, 45 deselected. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_model_action_plans.py python/weiss_rl/tests/test_model_action_tables.py python/weiss_rl/tests/test_model_candidate_components.py python/weiss_rl/tests/test_model_candidate_partitioning.py python/weiss_rl/tests/test_model_candidate_projection.py python/weiss_rl/tests/test_model_feature_gathering.py python/weiss_rl/tests/test_model_layers.py python/weiss_rl/tests/test_model_loading.py python/weiss_rl/tests/test_model_observation_contract.py python/weiss_rl/tests/test_model_public_heuristics.py python/weiss_rl/tests/test_model_sampling.py python/weiss_rl/tests/test_model_tensor_ops.py python/weiss_rl/tests/test_model_typed_encoder.py python/weiss_rl/tests/test_contracts.py -k "model or PolicyValueModel or structured_legal_policy_value_model or factorized or structured_v2 or public_heuristic or seat_aware"` | Passed: 104 passed, 14 deselected. |

### Changes

- Reduced `python/weiss_rl/model.py` from 1088 lines to 992 lines.
- Grew `python/weiss_rl/model_factorized_scoring.py` to 780 lines; it now owns factorized scoring and factorized packed sampling.

### Behavior Changes

No intended behavior changes. This was a method move into an existing factorized model mixin, with public classes and state-dict-bearing modules left in place.

### Remaining Risks and Next Hypotheses

- Remaining largest production files are now `runtime.py` (1351), `train.py` (1248), and `impala_learner.py` (1124). `model.py` is now below 1000 lines.
- Further model movement should split constructor/setup helpers or dense/packed facades only if the resulting modules stay clearer than the current mixin structure.

## 2026-05-11 - IMPALA Support-Method Mixin

### Scope

- Added `python/weiss_rl/learners/impala_support.py`.
- Moved optimizer creation, time-major forward dispatch, batch/legal-field adapters, packed-row view/slicing/scattering adapters, public-heuristic target-logit adapters, raw V-trace bootstrap adapters, tensor conversion helpers, numeric fault bundle helpers, training metric logging, checkpoint-metadata writing, and policy-version access from `ImpalaLearner` into `ImpalaSupportMixin`.
- Kept the moved private methods available on `ImpalaLearner` through inheritance.
- Left `_loss_and_metrics_with_context()` in `python/weiss_rl/learners/impala_learner.py` because it is the behavior-sensitive V-trace/action-logp/teacher-auxiliary core.
- Preserved the module-level `_batch_value()` compatibility hook by resolving the support mixin's batch reads lazily through `weiss_rl.learners.impala_learner`.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run ruff check python/weiss_rl/learners/impala_learner.py python/weiss_rl/learners/impala_support.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/learners/impala_learner.py python/weiss_rl/learners/impala_support.py` | Passed after formatting `impala_learner.py`. |
| `uv run mypy python/weiss_rl/learners/impala_learner.py python/weiss_rl/learners/impala_support.py --show-error-codes --no-error-summary` | Passed after keeping the optimizer assignment dynamically typed in the mixin. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_impala_learner.py python/weiss_rl/tests/test_vtrace.py python/weiss_rl/tests/test_learner_bootstrap.py python/weiss_rl/tests/test_learner_faults.py python/weiss_rl/tests/test_learner_logging.py python/weiss_rl/tests/test_learner_legal_fields.py` | Passed: 83 passed. |

### Changes

- Reduced `python/weiss_rl/learners/impala_learner.py` from 1124 lines to 671 lines.
- Added a 482-line focused support-method mixin.

### Behavior Changes

No intended behavior changes. This was a method move into an IMPALA mixin; public class identity, private method names, checkpoint state, optimizer behavior, forward dispatch, V-trace bootstrap lookup, fault-bundle payloads, and metric logging contracts are preserved.

### Remaining Risks and Next Hypotheses

- Remaining largest production files are now `runtime.py` (1351), `train.py` (1248), `actors/actor_worker.py` (935), `eval/final_eval.py` (915), `envs/decision_env.py` (859), `scripts/eval.py` (831), and `eval/simulator_runner.py` (830). `impala_learner.py` is now below the remaining production hotspots.
- The remaining IMPALA file is intentionally dominated by the core loss/metrics body and should only be split further with stronger parity tests around V-trace targets, packed action log-probs, factorized action paths, behavior log-prob handling, and structured teacher metrics.

## 2026-05-11 - Runtime Heuristic-Public Action Routing Mixin

### Scope

- Added `python/weiss_rl/runtime_heuristic_public_actions.py`.
- Moved heuristic-public opponent policy resolution and mask/packed/pool heuristic-public action selection from `QueueRuntime` into `QueueRuntimeHeuristicPublicActionsMixin`.
- Kept `_heuristic_opponent_policy()`, `_heuristic_public_actions_from_ids()`, `_heuristic_public_actions_from_mask()`, and `_heuristic_public_actions_from_pool()` available on `QueueRuntime`.
- Preserved simulator-native fixed-opponent routing, packed-candidate counter increments, batch `choose_actions_from_meta_batch()` behavior, fallback per-row `choose_action_from_meta()` behavior, dense-mask fallback action selection, and lazy variant-profile policy creation.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run ruff check python/weiss_rl/runtime.py python/weiss_rl/runtime_heuristic_public_actions.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/runtime.py python/weiss_rl/runtime_heuristic_public_actions.py` | Passed after formatting `runtime.py`. |
| `uv run mypy python/weiss_rl/runtime.py python/weiss_rl/runtime_heuristic_public_actions.py --show-error-codes --no-error-summary` | Passed. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_opponent_sampling.py python/weiss_rl/tests/test_runtime_heuristic_fast_path.py -k "heuristic_public_actions or heuristic_opponent_policy or simulator_native or collect_all_heuristic_ids"` | Passed: 10 passed, 45 deselected. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_opponent_sampling.py python/weiss_rl/tests/test_runtime_heuristic_fast_path.py python/weiss_rl/tests/test_runtime_opponents.py python/weiss_rl/tests/test_runtime_opponent_pool.py python/weiss_rl/tests/test_runtime_actor_state.py python/weiss_rl/tests/test_runtime_teacher_labels.py` | Passed: 90 passed. |
| `git diff --check` | Passed with the known CRLF normalization warnings for pre-existing touched text/test files. |
| `uv run python python/scripts/verify_repo.py` | Passed: 1189 passed, 2 skipped, 14 warnings; wrapper dry-runs passed. |

### Changes

- Reduced `python/weiss_rl/runtime.py` from 1351 lines to 1213 lines.
- Added a 148-line focused heuristic-public action-routing mixin.

### Behavior Changes

No intended behavior changes. This was a method move into a runtime mixin; the private method names remain available on `QueueRuntime` for existing tests and monkeypatch hooks.

### Remaining Risks and Next Hypotheses

- Remaining largest production files are now `train.py` (1248), `runtime.py` (1213), `actors/actor_worker.py` (935), `eval/final_eval.py` (915), `envs/decision_env.py` (859), `scripts/eval.py` (831), and `eval/simulator_runner.py` (830).
- `train.py` is now mostly compatibility wrappers plus real orchestration. Touch it only where wrappers can remain script-level and focused tests cover `test_snapshot_registry.py`, `test_entrypoints.py`, and train-stall behavior.

## 2026-05-11 - Runtime Support-Method Mixin

### Scope

- Added `python/weiss_rl/runtime_support.py`.
- Moved deterministic-logit wrappers, outcome update adapters, focal-row routing adapters, policy-train-mask adapter, IMPALA/PPO learner batch builders, bootstrap-value adapter, runtime metric assembly, reset-done fallback handling, and fixed-opponent actor reset handling from `QueueRuntime` into `QueueRuntimeSupportMixin`.
- Kept `_write_deterministic_logits()`, `_write_deterministic_logits_from_packed()`, `_update_outcomes()`, `_update_outcomes_from_transition_arrays()`, `_split_focal_actor_rows()`, `_policy_train_mask_for_actor()`, `_build_learner_batch()`, `_build_ppo_batch()`, `_bootstrap_values()`, `_runtime_metrics()`, `_reset_done_rows()`, and `_reset_actor_state_for_fixed_opponents()` available on `QueueRuntime`.
- Preserved deterministic logit writing, terminal outcome perspective mapping, actor heuristic split behavior, packed/dense batch-building contracts, bootstrap actor-model/device selection, runtime metric names and cumulative counters, and reset fallback role/hidden-state reinitialization.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run ruff check python/weiss_rl/runtime.py python/weiss_rl/runtime_support.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/runtime.py python/weiss_rl/runtime_support.py` | Passed after formatting `runtime.py`. |
| `uv run mypy python/weiss_rl/runtime.py python/weiss_rl/runtime_support.py --show-error-codes --no-error-summary` | Passed. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_actor_routing.py python/weiss_rl/tests/test_runtime_bootstrap.py python/weiss_rl/tests/test_runtime_batching.py python/weiss_rl/tests/test_runtime_deterministic_logits.py python/weiss_rl/tests/test_runtime_outcomes.py python/weiss_rl/tests/test_runtime_metrics.py` | Passed: 61 passed. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_actor_routing.py python/weiss_rl/tests/test_runtime_bootstrap.py python/weiss_rl/tests/test_runtime_batching.py python/weiss_rl/tests/test_runtime_deterministic_logits.py python/weiss_rl/tests/test_runtime_outcomes.py python/weiss_rl/tests/test_runtime_metrics.py python/weiss_rl/tests/test_runtime_actor_models.py python/weiss_rl/tests/test_runtime_actor_state.py python/weiss_rl/tests/test_runtime_central_rows.py python/weiss_rl/tests/test_runtime_heuristic_actor_outputs.py python/weiss_rl/tests/test_runtime_heuristic_fast_path.py python/weiss_rl/tests/test_runtime_opponent_pool.py python/weiss_rl/tests/test_runtime_opponent_sampling.py python/weiss_rl/tests/test_runtime_opponents.py python/weiss_rl/tests/test_runtime_shared_transport.py python/weiss_rl/tests/test_runtime_teacher_labels.py python/weiss_rl/tests/test_runtime_topology.py` | Passed: 149 passed. |

### Changes

- Reduced `python/weiss_rl/runtime.py` from 1213 lines to 986 lines.
- Added a 231-line focused runtime support mixin.

### Behavior Changes

No intended behavior changes. This was a method move into a runtime mixin; the private method names remain available on `QueueRuntime`.

### Remaining Risks and Next Hypotheses

- Remaining largest production file is now `train.py` (1248). `runtime.py`, `model.py`, and `impala_learner.py` are below 1000 lines after this pass.
- `train.py` should either receive a final carefully scoped entrypoint/run-context extraction or be explicitly justified as a compatibility-heavy script shell, with validation through snapshot-registry, entrypoint, train-stall, and wrapper dry-run tests.

## 2026-05-11 - Remaining Large-File Reduction Pass

### Scope

- Added `python/weiss_rl/training/train_entrypoint_main.py`, `python/weiss_rl/training/minimal_entrypoint_hooks.py`, and `python/weiss_rl/training/script_entrypoint_hooks.py`.
- Moved the path-based train script's `main()` body and several monkeypatch-sensitive train/dev-eval/promotion hook bodies behind explicit script-hook adapters that resolve dependencies through the executing `scripts.train` module.
- Added `python/weiss_rl/actors/actor_worker_helpers.py` and moved actor batch adapters, packed legal-id trimming, opponent-id refresh/coercion, checkpoint metadata filename parsing, behavior-logp adapter, and outcome-token mapping out of `actor_worker.py`.
- Added `python/weiss_rl/eval/final_eval_matrices.py` and moved final-eval matrix field definitions, reciprocal matrix-cell handling, posterior-sample cell reversal, and matrix-cell coverage helpers out of `final_eval.py`.
- Added `python/weiss_rl/model_structured_head.py` and moved the private structured legal-action head out of `model.py`, while preserving the `_StructuredLegalActionHead` binding used by `StructuredLegalPolicyValueModel`.
- Added `python/weiss_rl/runtime_episode_roles.py` and moved episode focal-seat/opponent-role assignment into `QueueRuntimeEpisodeRolesMixin`.
- Replaced the final `runtime.build_runtime_config()` forwarding function with a direct compatibility import from `runtime_config`, keeping `build_runtime_config` importable from `weiss_rl.runtime`.

### Commands and Results

| Command | Result |
| --- | --- |
| `uv run python -m pytest -q python/weiss_rl/tests/test_snapshot_registry.py python/weiss_rl/tests/test_train_stall_monitor.py python/weiss_rl/tests/test_entrypoints.py python/weiss_rl/tests/test_script_entrypoint_smokes.py python/weiss_rl/tests/test_training_dev_eval.py python/weiss_rl/tests/test_training_checkpoint_writers.py python/weiss_rl/tests/test_training_execution.py python/weiss_rl/tests/test_training_manifest_layout.py python/weiss_rl/tests/test_training_promotion.py python/weiss_rl/tests/test_training_run_identity.py` | Passed: 142 passed, 14 warnings. |
| `uv run ruff check python/weiss_rl/actors/actor_worker.py python/weiss_rl/actors/actor_worker_helpers.py` | Passed after Ruff sorted imports. |
| `uv run ruff format --check python/weiss_rl/actors/actor_worker.py python/weiss_rl/actors/actor_worker_helpers.py` | Passed after formatting. |
| `uv run mypy python/weiss_rl/actors/actor_worker.py python/weiss_rl/actors/actor_worker_helpers.py --show-error-codes --no-error-summary` | Passed. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_actor_worker.py python/weiss_rl/tests/test_actor_outcomes.py python/weiss_rl/tests/test_masking.py` | Passed: 36 passed. |
| `uv run ruff check python/weiss_rl/eval/final_eval.py python/weiss_rl/eval/final_eval_matrices.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/eval/final_eval.py python/weiss_rl/eval/final_eval_matrices.py` | Passed after formatting `final_eval.py`. |
| `uv run mypy python/weiss_rl/eval/final_eval.py python/weiss_rl/eval/final_eval_matrices.py --show-error-codes --no-error-summary` | Passed. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_final_eval.py python/weiss_rl/tests/test_metagame_sensitivity.py python/weiss_rl/tests/test_paper_readiness.py python/weiss_rl/tests/test_paper_readiness_fixture.py` | Passed: 29 passed. |
| `uv run ruff check python/weiss_rl/model.py python/weiss_rl/model_structured_head.py` | Passed after preserving `_build_mlp_stack` compatibility import and sorting imports. |
| `uv run ruff format --check python/weiss_rl/model.py python/weiss_rl/model_structured_head.py` | Passed. |
| `uv run mypy python/weiss_rl/model.py python/weiss_rl/model_structured_head.py --show-error-codes --no-error-summary` | Passed. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_model_action_plans.py python/weiss_rl/tests/test_model_action_tables.py python/weiss_rl/tests/test_model_candidate_components.py python/weiss_rl/tests/test_model_candidate_partitioning.py python/weiss_rl/tests/test_model_candidate_projection.py python/weiss_rl/tests/test_model_feature_gathering.py python/weiss_rl/tests/test_model_layers.py python/weiss_rl/tests/test_model_loading.py python/weiss_rl/tests/test_model_observation_contract.py python/weiss_rl/tests/test_model_public_heuristics.py python/weiss_rl/tests/test_model_sampling.py python/weiss_rl/tests/test_model_tensor_ops.py python/weiss_rl/tests/test_model_typed_encoder.py python/weiss_rl/tests/test_contracts.py` | Passed: 118 passed. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_actor_models.py python/weiss_rl/tests/test_training_learner_factory.py` | Passed: 35 passed. |
| `uv run ruff check python/weiss_rl/runtime.py python/weiss_rl/runtime_episode_roles.py` | Passed. |
| `uv run ruff format --check python/weiss_rl/runtime.py python/weiss_rl/runtime_episode_roles.py` | Passed after formatting `runtime.py`. |
| `uv run mypy python/weiss_rl/runtime.py python/weiss_rl/runtime_episode_roles.py --show-error-codes --no-error-summary` | Passed. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_runtime_actor_routing.py python/weiss_rl/tests/test_runtime_actor_state.py python/weiss_rl/tests/test_runtime_opponent_pool.py python/weiss_rl/tests/test_runtime_opponent_sampling.py python/weiss_rl/tests/test_runtime_opponents.py python/weiss_rl/tests/test_runtime_heuristic_fast_path.py python/weiss_rl/tests/test_runtime_config.py python/weiss_rl/tests/test_runtime_threads.py` | Passed: 94 passed. |
| `git diff --check` | Passed with known CRLF normalization warnings for touched text/Python files. |
| `uv run python python/scripts/verify_repo.py` | First run failed at Ruff import ordering in `python/scripts/train.py`; after `uv run ruff check --fix python/scripts/train.py`, rerun passed: 1189 passed, 2 skipped, 14 warnings; wrapper dry-runs passed. |

### Failures Found and Fixes

- A resumed train-cluster failure had previously reported missing `_update_stall_monitor_impl`; the current working tree already contained the alias, and the train cluster passed unchanged.
- Moving `_StructuredLegalActionHead` out of `model.py` initially dropped the private `_build_mlp_stack` compatibility import used by `test_model_layers.py`; restored it in `model.py` and included it in `__all__`.
- Two exploratory runtime test commands referenced non-existent test module names. These were command-selection mistakes, not code failures; reran the affected runtime slices with existing test modules.
- The first full verifier run after this pass failed on `python/scripts/train.py` import ordering; Ruff fixed the import block and the verifier passed on rerun.

### Changes

- `python/scripts/train.py`: 871 lines after hook/main extraction.
- `python/weiss_rl/actors/actor_worker.py`: 825 lines after helper extraction.
- `python/weiss_rl/eval/final_eval.py`: 791 lines after matrix extraction.
- `python/weiss_rl/model.py`: below the current top-25 production files after moving the 739-line structured head module out.
- `python/weiss_rl/runtime.py`: 886 lines after episode-role extraction and direct config-builder re-export.

### Behavior Changes

No intended behavior changes. These were method/function moves and compatibility-hook extractions. Script-level train hooks still resolve dependencies through `scripts.train` to preserve monkeypatch behavior; `actor_behavior_logp_from_legal_ids`, `build_runtime_config`, `QueueRuntimeMode`, and `_StructuredLegalActionHead` remain importable from their previous modules.

### Remaining Risks and Next Hypotheses

- The largest production files are now `python/scripts/train.py` (871), `python/weiss_rl/envs/decision_env.py` (859), `python/scripts/eval.py` (831), `python/weiss_rl/eval/simulator_runner.py` (830), `python/weiss_rl/actors/actor_worker.py` (825), `python/weiss_rl/eval/heuristic_public.py` (816), and `python/weiss_rl/eval/final_eval.py` (791).
- The large-file objective is substantially improved; remaining candidates should be split only where the extraction names a real domain concept and has focused characterization tests.
- Full verifier and `git diff --check` pass after this newest pass. Remaining work is final dirty-tree review and any last documentation consistency updates before declaring the overall refactor complete.

## 2026-05-11 - Final Refactor Completion Summary

### Completed Changes

- Reworked the repository into focused modules across config parsing, runtime collection, actor state/routing, league/opponent handling, learner helpers, model scoring, train orchestration, evaluation reporting, paper-readiness checks, replay handling, and GitHub-facing documentation.
- Preserved public import and script compatibility with explicit wrappers where tests or scripts rely on historical names, including `scripts.train`, `weiss_rl.runtime`, `weiss_rl.model`, actor-worker masking helpers, and evaluation entry points.
- Added and updated documentation: `README.md`, `REFACTOR_PLAN.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, GitHub issue/PR templates, and docs for architecture, training, evaluation, configuration, checkpoints, league behavior, reproducibility, testing, performance, and refactor history.
- Added characterization tests around the high-risk surfaces touched by the refactor: config sections, seed sets, runtime batching/routing/opponents/teacher labels, model tensor/scoring/action-plan helpers, learner action-logp/bootstrap/legal/packed/vtrace helpers, training startup/execution/dev-eval/promotion/checkpoint flows, final-eval matrices, and artifact/paper-readiness contracts.
- Reduced the prior obvious god files. The largest production files at completion are `runtime.py` (886), `python/scripts/train.py` (871), `decision_env.py` (859), `python/scripts/eval.py` (831), `simulator_runner.py` (830), `actor_worker.py` (825), `heuristic_public.py` (816), and `final_eval.py` (791). These remaining files are domain orchestration/adapters rather than unbounded catch-all modules.

### Final Validation Commands

| Command | Result |
| --- | --- |
| `uv sync --extra dev`; `uv sync --extra dev --extra sim` | Passed: dependency sync completed and final simulator-aware environment restored `weiss-sim==0.8.1`. |
| Windows package smoke: `uv run python -m build`, install newest `dist/*.whl` into a fresh temp venv, `python -c "import weiss_rl; print(weiss_rl.__all__)"` | Passed: built sdist/wheel, installed wheel, imported `weiss_rl`, printed `['load_stack_config', 'assert_spec_compatibility']`. |
| `uv run python python/scripts/verify_repo.py` | Passed after Ruff fixed `python/scripts/train.py` import ordering and again after restoring the simulator extra: 1189 passed, 2 skipped, 14 warnings; wrapper dry-runs passed. |
| `git diff --check` | Passed with known CRLF normalization warnings for touched files. |
| `git status --short -- runs dist build checkpoints outputs results artifacts` | No tracked/untracked historical run, checkpoint, result, or artifact changes reported; package smoke outputs are ignored/generated. |

### Tests Added

The refactor added targeted unit tests for decomposed modules rather than only updating existing tests. New coverage includes config parsing utilities and section parsers, learner helper modules, model action/scoring/tensor modules, runtime helper/mixin modules, training helper modules, final-eval/reporting helpers, and artifact-readiness guardrails. Existing tests were kept as behavioral contracts and expanded before risky movement where bugs or compatibility gaps were found.

### Bugs Fixed

- No intended algorithmic, simulator, reward, rollout, metric, evaluation, or league semantics were changed.
- Refactor-time defects were fixed only with tests/evidence, including dropped compatibility imports and helper-name regressions. These were implementation regressions introduced during movement, not thesis-result-changing algorithm changes.

### Performance Changes

- No semantic performance optimization was claimed as a result change in the final pass.
- The refactor preserves prior performance-sensitive paths while making hot-path modules easier to profile and modify. Performance-related documentation is in `docs/performance.md`.

### Compatibility Notes

- Checkpoint payload loading remains compatible through preserved model state-dict module names and explicit snapshot/checkpoint loaders.
- Config meanings, defaults, inheritance, and override behavior are covered by the config section and loader tests.
- Legal-action ordering, packed offsets, masking, deterministic sampling, seed derivation, seat-swapped evaluation, PFSP/opponent sampling, promotion gates, and eval matrix reversal remain covered by focused tests.
- `scripts.train` intentionally keeps compatibility wrappers and dynamic script-hook resolution for monkeypatch-sensitive tests and legacy script surfaces.

### Known Risks

- The final verifier does not run a full historical checkpoint smoke with real thesis checkpoints; fixture-level checkpoint compatibility is covered.
- Full thesis final policy-set evaluation with all anchors remains larger than the smoke validation; deterministic public-demo and tiny simulator-backed eval evidence are recorded earlier in this log.
- Direct `mypy python/scripts/train.py` still has script-level debt outside the configured verifier gate. Package-level helper modules and configured mypy gates pass.

### Suggested Future Work

- Add a small non-thesis historical checkpoint fixture if it can be safely included.
- Split `decision_env.py` packed-legality helpers or `simulator_runner.py` artifact/reporting helpers only if the extraction is behavior-preserving and covered by characterization tests.
- Keep future collector-loop or legal-action movement behind explicit row-ordering, masking, recurrent-state, and behavior-logp parity tests.

## 2026-05-11 - Root Package Grouping Follow-Up

### Changes

- Grouped the remaining crowded root-level modules under broad subsystem packages:
  - `weiss_rl.core`: action catalog, card table, legal actions, masking, observation layout, schedules, simulator/spec contracts, and termination reasons.
  - `weiss_rl.artifacts`: artifact layout, manifest writing, and reproducibility/hash helpers.
  - `weiss_rl.diagnostics`: action diagnostics, artifact hygiene, job telemetry, TensorBoard logging, training JSONL logging, and CLI banner helpers.
  - `weiss_rl.experiments`: no-league baseline helpers, launch plans, sweeps, structured acceptance helpers, and public-demo scaffolding.
  - `weiss_rl.models`: model internals and snapshot loading behind the `weiss_rl.model` facade.
  - `weiss_rl.runtime_components`: runtime internals behind the `weiss_rl.runtime` facade.
- Updated imports across source, tests, scripts, and examples.
- Updated `docs/architecture.md` and `docs/training_logs.md` for the grouped module paths.
- Updated `python/scripts/check_core_placeholders.py` to scan the new core/artifact locations.
- Applied behavior-neutral Ruff fixes to `scripts/make_thesis_figures.py` so repo-wide lint remains green.

### Validation

| Command | Result |
| --- | --- |
| `uv run ruff check .` | Passed. |
| `uv run ruff format --check .` | Passed: 406 files already formatted. |
| `uv run python -m pytest -q python/weiss_rl/tests/test_model_loading.py python/weiss_rl/tests/test_artifact_hygiene.py python/weiss_rl/tests/test_runtime_threads.py python/weiss_rl/tests/test_entrypoints.py python/weiss_rl/tests/test_manifest.py python/weiss_rl/tests/test_repro_ids.py` | Passed: 63 passed, 14 warnings. |
| `uv run python python/scripts/verify_repo.py` | Passed: 1189 passed, 2 skipped, 14 warnings; wrapper dry-runs passed. |
| `git diff --check` | Passed with known CRLF normalization warnings. |

### Behavior and Compatibility

- No intended behavior changes.
- `weiss_rl.model` and `weiss_rl.runtime` remain the public compatibility facades for model/runtime callers.
- This cleanup removes root clutter rather than changing simulator, training, evaluation, reward, RNG, rollout, metric, checkpoint, or league semantics.

## 2026-05-12 - Simulator 1.1 Phase 0 Compatibility Pass

### Changes

- Updated the simulator extra and lockfile from `weiss-sim==0.8.1` to `weiss-sim==1.1.0`.
- Added startup prerequisite checks for the 1.1 runtime surface: package version, `make_pool`, `EnvPoolBuffers`, spec constants, spec export, and fused sampled-logp RL helpers.
- Updated the standard multideck preset to use the published 1.1 deck preset names: `starter_deck_ws02_v1`, `main_deck_5hy_yotsuba_v1`, `aggro_deck_5hy_nino_v1`, and `control_deck_jj_s66_v1`.
- Added simulator compatibility documentation in `docs/simulator_compatibility.md`.
- Extended simulator smoke coverage to include `i16_legal_ids_nometa`, `EnvPoolBuffers.legal_action_context_v1(...)`, and `EnvPoolBuffers.step_sample_from_logits_with_logp(...)`.

### Validation

| Command | Result |
| --- | --- |
| `uv sync --extra dev --extra sim` | Passed. Replaced `weiss-sim==0.8.1` with `weiss-sim==1.1.0`. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_training_startup.py python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_pool_factory.py python/weiss_rl/tests/test_simulator_contract.py python/weiss_rl/tests/test_rl_step_layout_contract_smoke.py python/weiss_rl/tests/test_training_environments.py python/weiss_rl/tests/test_heuristic_public.py` | Passed: 95 passed. |
| `uv run --extra dev --extra sim python -m ruff check pyproject.toml python/weiss_rl/training/startup.py python/weiss_rl/tests/test_training_startup.py python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_pool_factory.py python/weiss_rl/tests/test_rl_step_layout_contract_smoke.py python/weiss_rl/tests/test_simulator_contract.py python/scripts/play_vs_model.py` | Passed. |
| `uv run --extra dev --extra sim python -m ruff format --check python/weiss_rl/training/startup.py python/weiss_rl/tests/test_training_startup.py python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_pool_factory.py python/weiss_rl/tests/test_rl_step_layout_contract_smoke.py python/weiss_rl/tests/test_simulator_contract.py python/scripts/play_vs_model.py` | Passed: 7 files already formatted. |
| `uv run --extra dev --extra sim python python/scripts/train.py --stack-config configs/stack_smoke.yaml --run-label phase0_sim11_smoke_20260512` | Passed. Wrote manifest scaffold with `weiss-sim` version `1.1.0`, compatibility hash `8590000130`, observation length `378`, and action space `527`; no learner training or rollout collection executed. |

### Findings

- The published `weiss-sim==1.1.0` package exposes the expected base contract and bundled presets.
- The old multideck aliases (`preset:starter_v1`, `preset:quints_*`) are not present in the published 1.1 package, so the RL multideck preset now uses the published preset names directly.
- A raw sibling simulator `python` source path is not enough for import parity unless the Rust extension is built and importable; use the published package or a built simulator environment for sibling-checkout parity.

### Next Hypotheses

- Phase 1 should make baseline bootstrap and guided/RL phases first-class instead of relying on inherited rescue presets.
- Phase 2 should benchmark a `structured_v2` path that can use the simulator 1.1 fused sampled-logp and nometa/legal-context surfaces where the learner does not need packed metadata.

## 2026-05-12 - Deck Decision Lock-In

### Changes

- Locked the canonical `standard`/`standard-thesis-eval` environment deck pools to `preset:main_deck_5hy_yotsuba_v1` for both seats.
- Added explicit eval deck mapping:
  - focal model snapshots/default policies, B0, B1, and B2 use `preset:main_deck_5hy_yotsuba_v1`.
  - B3 uses `preset:aggro_deck_5hy_nino_v1`.
  - B4 uses `preset:control_deck_jj_s66_v1`.
- Extended scheduled eval games and exported eval records with `seat0_deck` and `seat1_deck`, so seat-swapped B3/B4 rows preserve the correct profile decks in artifacts.
- Included explicit decks in fallback eval episode identity when simulator episode keys are unavailable.
- Carried deck/opponent-deck through replay rerun contracts so replay verification rebuilds the same deck setup.
- Updated deck-scoping docs in `README.md`, `docs/standard_recipe.md`, and `docs/simulator_compatibility.md`.

### Validation

| Command | Result |
| --- | --- |
| `uv run python -m pytest python/weiss_rl/tests/test_eval_harness.py python/weiss_rl/tests/test_eval_export.py python/weiss_rl/tests/test_policy_set.py python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_structured_dev_fast_and_acceptance_presets python/weiss_rl/tests/test_pool_factory.py python/weiss_rl/tests/test_replay_bundles.py::test_build_replay_env_uses_fast_pool_factory_for_default_reruns python/weiss_rl/tests/test_snapshot_registry.py::test_simulator_eval_runner_uses_learner_scoring_mode` | Passed: 49 passed. |
| `uv run python -m pytest python/weiss_rl/tests/test_final_eval.py python/weiss_rl/tests/test_eval_diagnostics.py python/weiss_rl/tests/test_eval_payoff_folding.py python/weiss_rl/tests/test_eval_stage2.py python/weiss_rl/tests/test_replay_bundles.py` | Passed: 42 passed. |
| `uv run python -m pytest python/weiss_rl/tests/test_eval_harness.py python/weiss_rl/tests/test_eval_export.py python/weiss_rl/tests/test_policy_set.py python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_final_eval.py python/weiss_rl/tests/test_replay_bundles.py python/weiss_rl/tests/test_snapshot_registry.py::test_simulator_eval_runner_uses_learner_scoring_mode` | Passed: 100 passed. |
| `uv run ruff check python/weiss_rl/eval/policy_set.py python/weiss_rl/eval/harness.py python/weiss_rl/eval/final_eval.py python/weiss_rl/eval/simulator_runner.py python/weiss_rl/eval/export.py python/weiss_rl/replay/bundles.py python/weiss_rl/replay/runner.py python/weiss_rl/tests/test_eval_harness.py python/weiss_rl/tests/test_eval_export.py python/weiss_rl/tests/test_policy_set.py python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_snapshot_registry.py` | Passed. |
| `uv run ruff check python/weiss_rl/eval/harness.py python/weiss_rl/tests/test_eval_harness.py` | Passed. |

### Notes

- This is a deck-scoping behavior change, not the Phase 1 simulator fast-lane pass.
- `standard-multideck` remains an exploratory generalization preset, separate from the primary same-main-deck comparison.

## 2026-05-28 - Public Workflow Cleanup and Smoke Contract Repair

### Changes

- Added pytest discovery defaults so `python -m pytest` only collects the repo test suite and ignores local scratch bundles under `temp/`.
- Archived tracked top-level progress/rescue/status notes under `docs/archive/top_level_notes/` and linked the archive from `docs/archive/README.md`.
- Added the public thesis ablation alias `configs/thesis/ablations/no_gru.yaml` plus a short ablation README.
- Split the package CLI workflow plumbing out of `weiss_rl.cli` into `weiss_rl.workflows.thesis`; `weiss_rl.cli` now mostly owns argument parsing and dispatch.
- Made the public `train-main` wrapper use `configs/thesis/main_league.yaml`, the packed medium64 stack that matches the standard B1 no-league checkpoint contract.
- Kept the selected factorized continuation behind explicit guided-bootstrap workflows.
- Made `smoke-eval` use `configs/thesis/main_league.yaml`; `eval-final` still uses `configs/thesis/final_eval.yaml`.
- Published the `b1_noleague_baseline` alias during baseline_noleague smoke training so the main league import path does not depend on a manual aliasing step.
- Updated README/docs around canonical surfaces, smoke vs final eval, top-level archive contents, and the public CLI commands.

### Commands Run

| Command | Result |
| --- | --- |
| `uv sync --extra dev --extra sim` | Passed. |
| `uv run --extra dev --extra sim python python/scripts/verify_repo.py` | Passed: placeholder gate, Ruff, format check, configured mypy/vulture, 1890 tests, wrapper dry-runs. |
| `uv run --extra dev --extra sim python -m pytest -q` | Passed: 1890 passed, 2 skipped, 14 warnings. |
| `uv run --extra dev --extra sim python -m ruff check .` | Passed. |
| `uv run --extra dev --extra sim python -m ruff format --check .` | Initially flagged `python/weiss_rl/workflows/thesis.py`; after `ruff format`, passed: 603 files already formatted. |
| `uv run --extra dev --extra sim python -m mypy python` | Failed before type checking with pre-existing duplicate module discovery: `python/scripts/eval.py` is seen as both `eval` and `scripts.eval`. |
| `make verify` | Not run: `make` is not installed in this Windows shell. |
| `make artifact-hygiene` | Not run: `make` is not installed in this Windows shell. |
| `uv run --extra dev --extra sim python -m weiss_rl.cli train-b1 --run-label refactor_b1_smoke_alias2_20260528 --profile smoke` | Passed; wrote a one-update B1 smoke run and persisted `b1_noleague_baseline`. |
| `uv run --extra dev --extra sim python -m weiss_rl.cli train-main --run-label refactor_main_smoke_plain_20260528 --b1-run runs/refactor_b1_smoke_alias2_20260528 --profile smoke` | Passed; initialized from the B1 checkpoint and imported the explicit B1 anchor. |
| `uv run --extra dev --extra sim python -m weiss_rl.cli smoke-eval --run-dir runs/refactor_main_smoke_plain_20260528 --b1-run runs/refactor_b1_smoke_alias2_20260528` | Passed; wrote the tiny B0-B4 smoke summary under `eval/final_eval/`. |
| `uv run --extra dev python -m weiss_rl.cli figures --run-dir runs/refactor_main_smoke_plain_20260528 --format png` | Passed; wrote four paper figure PNGs. |
| `git diff --check` | Passed with expected CRLF normalization warnings only. |

### Tests Added

- Added CLI coverage for the public `train-main` packed-main config selection.
- Added CLI coverage for smoke-profile fallback to a single unaliased snapshot.
- Added config-loader coverage for the public `configs/thesis/ablations/no_gru.yaml` alias.

### Failures Found

- Full pytest initially collected an incomplete local bundle under `temp/pro_context_bundle_20260521_074609`; pytest discovery now excludes scratch/artifact folders.
- The documented B1-to-main smoke chain was broken because `train-main` pointed at a factorized selected-bootstrap config while `train-b1` produces packed checkpoints.
- After that fix, `smoke-eval` still used factorized `final_eval.yaml` and could not load the packed B1 smoke checkpoint.
- Early B1 smoke runs did not publish the explicit `b1_noleague_baseline` alias; the main import path correctly refused to guess from `latest`.

### Fixes Applied

- Added focused pytest collection settings in `pyproject.toml`.
- Repointed public smoke train/eval wrappers to the packed canonical main stack.
- Preserved strict model/config contract checks rather than allowing packed/factorized checkpoint mixing.
- Published the explicit B1 alias from baseline_noleague training checkpoints.
- Updated tests and docs to make the smoke/final-eval contract split visible.

### Behavior Changes

- No simulator, reward, legal-action, rollout, learner, V-trace, checkpoint format, seed pairing, metric aggregation, or final-eval semantics were intentionally changed.
- Public `python -m weiss_rl.cli train-main` now maps to the plain packed `configs/thesis/main_league.yaml` workflow so it matches `train-b1`.
- Public `python -m weiss_rl.cli smoke-eval` now uses the packed smoke contract; `eval-final` remains the selected factorized final-eval contract.

### Files Moved or Added

- Moved top-level dated/progress notes into `docs/archive/top_level_notes/`.
- Added `configs/thesis/ablations/no_gru.yaml`.
- Added `configs/thesis/ablations/README.md`.
- Added `python/weiss_rl/workflows/__init__.py`.
- Added `python/weiss_rl/workflows/thesis.py`.

### Remaining Risks

- `python -m mypy python` still needs a package-layout fix for duplicate script module discovery.
- `make verify` and `make artifact-hygiene` were not executable on this machine because `make` is unavailable.
- Historical selected/factorized docs and rebuild reports remain as historical records; the current docs now label the public packed smoke path separately from selected factorized final reproduction.
- The archive moves are unstaged as deletes plus untracked archive files until Git can update the index; a pre-existing `.git/index.lock` blocked `git mv` during this pass.

### Next Action

- Fix the mypy duplicate-module invocation or package layout so `uv run --extra dev --extra sim python -m mypy python` can be used as a top-level validation command.
- Continue reducing docs/config surface area by moving dated result reports into archive/history once their references are audited.

## 2026-05-28 - Dated Report Archive and Focused Type Cleanup

### Changes

- Moved dated May 2026 report/lock/inventory docs out of the live docs root and into `docs/archive/reports/202605/`.
- Updated `docs/archive/README.md` so the archive structure explains both top-level notes and dated reports.
- Updated live thesis workflow references to point at the archived B1/main report paths.
- Made a small set of production-package type-narrowing fixes without changing runtime contracts:
  - explicit float accumulator for policy-alignment family mass totals
  - iterable-friendly opponent-context helper signatures
  - typed PFSP policy-env counters
  - registered-buffer casts for model typing
  - candidate-projection module iteration via `children()`
  - explicit hidden-state validation in the fallback time-major learner paths
  - narrow checkpoint restore typing for `init_schedule_offset_updates`

### Commands Run

| Command | Result |
| --- | --- |
| `uv run --extra dev --extra sim python -m ruff check <edited type-cleanup files>` | Passed. |
| `uv run --extra dev --extra sim python -m mypy <edited type-cleanup files> --show-error-codes --no-error-summary` | Passed. |
| `uv run --extra dev --extra sim python -m ruff format --check <edited type-cleanup files>` | Passed: 11 files already formatted. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_policy_alignment.py python/weiss_rl/tests/test_model_candidate_projection.py python/weiss_rl/tests/test_model_typed_encoder.py python/weiss_rl/tests/test_model_opponent_context.py python/weiss_rl/tests/test_training_checkpoint_writers.py python/weiss_rl/tests/test_impala_learner.py -k "forward_time_major or opponent_context or checkpoint_payload or restore_minimal or policy_alignment or candidate_projection or typed_encoder"` | Passed: 30 passed, 63 deselected. |
| `python scripts/check_docs_links.py` | Not configured in this checkout; no `scripts/check_docs_links.py` exists. |
| `python python/scripts/check_docs_links.py` | Not configured in this checkout; no `python/scripts/check_docs_links.py` exists. |
| `rg <dated-report references> README.md docs/README.md docs/getting_started.md docs/thesis_workflow.md docs/architecture.md docs/configuration.md docs/training.md docs/evaluation.md docs/artifact_contract.md docs/reproducibility.md docs/testing.md docs/troubleshooting.md docs/experiments.md docs/standard_recipe.md` | Passed: no stale live-doc references to the moved report paths. |

### Tests Added

- None. This was an archive move plus narrow type-cleanup pass covered by existing characterization tests.

### Failures Found

- The broad `python -m mypy python` path is not just a discovery problem. After excluding script collisions, it exposes substantial legacy typing debt across experiments, tests, and some production helper modules.
- There is no docs-link checker script in this checkout despite older command references.

### Fixes Applied

- Kept the broad mypy debt documented instead of hiding it behind a misleading passing command.
- Archived historical dated reports so the live docs root now contains only canonical docs plus logs.
- Applied focused type fixes only where they were local, readable, and behavior-preserving.

### Behavior Changes

- No intended simulator, legal-action, reward, training, evaluation, checkpoint-format, or artifact-schema changes.
- The fallback learner forward paths now raise a clear `ValueError` when required hidden state is absent, matching the existing helper contract that hidden state must be present for those paths.

### Remaining Risks

- Broad package/test/experiment mypy remains future work.
- Archived report files still contain some historical references to their original `docs/<file>` paths; these are provenance text inside archived docs, not current workflow links.

### Next Action

- Continue shrinking the live docs/config surface while leaving thesis artifacts and historical evidence untouched.

## 2026-05-28 - Dated Preset Config Archive

### Changes

- Moved 26 unreferenced May 2026 dated/probe preset configs from `configs/presets/` into `configs/archive/presets_20260506/`.
- Added `configs/archive/README.md` to explain that archived configs are historical and not part of the canonical thesis surface.
- Updated `configs/README.md` so current configs point readers toward `configs/thesis/` and historical dated probes toward `configs/archive/`.
- Rewrote archived config `extends:` paths so the moved configs still load from their new location.
- Updated `python/scripts/heuristic_sanity_scan.py` to point at the archived copy of `eval_gpu_exp031_fast_20260506.yaml`.

### Commands Run

| Command | Result |
| --- | --- |
| `rg <moved dated preset names> README.md docs python configs pyproject.toml Makefile` | Passed: no live references remain outside archived historical notes. |
| `uv run --extra dev --extra sim python -c "<load every configs/archive/presets_20260506/*.yaml>"` | Passed: loaded 26 archived configs. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_baselines_and_ablations python/weiss_rl/tests/test_cli_workflow.py::test_package_cli_train_b1_dry_run_uses_thesis_config python/weiss_rl/tests/test_cli_workflow.py::test_package_cli_train_main_requires_b1_and_uses_main_config` | Passed: 3 passed. |
| `uv run --extra dev --extra sim python -m ruff check python/scripts/heuristic_sanity_scan.py` | Passed. |

### Tests Added

- None. This was an archive/reference cleanup covered by config-loading and existing CLI/config workflow tests.

### Files Moved or Deleted

- Moved dated/probe preset configs into `configs/archive/presets_20260506/`.
- No thesis artifacts, runs, checkpoints, logs, figures, or simulator files were modified.

### Behavior Changes

- None intended. The archived configs continue to parse, and the one live diagnostic script that referenced a moved config now points at the archived copy of the same file.

### Remaining Risks

- Archived historical notes still mention the old `configs/presets/...` paths as provenance text.
- Additional dated configs under `configs/thesis/` may still be candidates for archive, but they were left in place until each can be audited against docs, tests, and workflows.

### Next Action

- Continue with package/source simplification in small slices, keeping characterization tests close to every behavior-sensitive move.

## 2026-05-28 - Live Docs Canonical Surface Cleanup

### Changes

- Updated `README.md`, `docs/configuration.md`, `docs/testing.md`, `docs/training.md`, `docs/architecture.md`, and `docs/standard_recipe.md` so the live docs consistently present `python -m weiss_rl.cli` and `configs/thesis/` as the canonical thesis surface.
- Reworded compatibility preset descriptions so `configs/presets/structured_acceptance_standard*.yaml` are no longer described as the public canonical recipe.
- Replaced stale full-package mypy guidance in `docs/testing.md` with the current status: selected-script mypy is the configured gate, while broad mypy is known debt for this checkout.
- Moved the stale May 11 completion audit from `docs/refactor_completion_audit.md` to `docs/archive/reports/202605/refactor_completion_audit_20260511.md`.
- Removed the completion-audit link from the live docs hub and recorded the archived audit in `docs/archive/README.md`.

### Commands Run

| Command | Result |
| --- | --- |
| `rg <stale canonical/mypy/audit phrases> README.md docs configs python/scripts/README.md` | Passed: only historical refactor-log provenance remains. |
| `rg refactor_completion_audit.md <live docs>` | Passed: no live docs link to the moved audit path. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_cli_workflow.py python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_baselines_and_ablations` | Passed: 20 passed. |

### Tests Added

- None. This was documentation cleanup with existing CLI/config tests as a behavior guard.

### Files Moved or Deleted

- Moved `docs/refactor_completion_audit.md` to `docs/archive/reports/202605/refactor_completion_audit_20260511.md`.

### Behavior Changes

- None. Documentation now matches the already-tested public CLI/config behavior.

### Remaining Risks

- `python/scripts/README.md` still documents lower-level script workflows and compatibility presets. It should be reviewed separately before deciding whether to archive or shorten it.
- Historical logs intentionally retain old wording and command records.

### Next Action

- Continue simplifying the public script/doc boundary, prioritizing live docs and thin wrappers before risky runtime/train internals.

## 2026-05-28 - Makefile and Script Entry-Point Surface Cleanup

### Changes

- Added package-CLI Make targets for the canonical smoke route:
  - `train-b1-smoke`
  - `train-main-smoke`
  - `eval-smoke`
  - `figures-smoke`
  - `thesis-smoke`
- Left `train-inline-smoke` as a lower-level compatibility target rather than removing it.
- Rewrote `python/scripts/README.md` into a short compatibility guide that points new users to `python -m weiss_rl.cli` first.
- Updated README and testing docs to mention the new Make smoke targets.
- Added a lightweight Makefile characterization test that locks the new smoke targets to the package CLI.

### Commands Run

| Command | Result |
| --- | --- |
| `rg train-inline-smoke README.md docs python/scripts configs Makefile` | Passed: live docs no longer route users through the old inline script target. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_makefile_figures.py python/weiss_rl/tests/test_cli_workflow.py` | Passed after formatting: 20 passed, 2 skipped. |
| `uv run --extra dev --extra sim python -m ruff check python/weiss_rl/tests/test_makefile_figures.py` | Passed. |
| `uv run --extra dev --extra sim python -m ruff format --check python/weiss_rl/tests/test_makefile_figures.py` | Passed after formatting the file. |

### Tests Added

- `test_makefile_thesis_smoke_targets_use_package_cli` in `python/weiss_rl/tests/test_makefile_figures.py`.

### Files Moved or Deleted

- None.

### Behavior Changes

- No RL/training/evaluation semantics changed.
- Make now exposes canonical package-CLI smoke wrappers; existing script targets remain available.

### Remaining Risks

- The old `train-inline-smoke` target still exists for compatibility and should stay clearly documented as lower-level if referenced again.
- `make` is unavailable in this Windows shell, so Make target execution itself remains covered by text characterization and existing skip-on-missing-make tests rather than a live Make invocation here.

### Next Action

- Continue reducing lower-level script prominence by auditing docs and wrappers that still present `thesis_run.py` or direct stack-config commands as first-choice workflows.

## 2026-05-28 - Direct Script Reference Triage

### Changes

- Reworded docs that still described `python/scripts/train.py` as owning public CLI behavior.
- Updated checkpoint, league, evaluation, and architecture docs to describe package helpers plus explicit compatibility-hook wiring.
- Updated `configs/README.md` so direct `train.py --override` examples are clearly lower-level and use a thesis config rather than a legacy typed preset.

### Commands Run

| Command | Result |
| --- | --- |
| `rg python/scripts/train.py README.md docs configs python/scripts/README.md Makefile python/weiss_rl/tests` | Audited. Remaining live references are scaffold, public-demo, lower-level override, Make compatibility target, tests, or historical logs. |
| `rg <stale train.py ownership phrases> README.md docs configs python/scripts/README.md` | Passed: no stale live-doc wording remains. |

### Tests Added

- None.

### Files Moved or Deleted

- None.

### Behavior Changes

- None. Documentation-only cleanup.

### Remaining Risks

- Direct script examples intentionally remain for scaffold smoke, public demo, and lower-level stack-config debugging.
- Historical rebuild/refactor logs still contain old script-first command records.

### Next Action

- Move back into source cleanup where safe: identify one compatibility-heavy module with a small removable duplication or wrapper cluster and cover it with focused tests.

## 2026-05-28 - Workflow Helper Public Names

### Changes

- Renamed the package workflow command builders in `weiss_rl.workflows.thesis` from underscore-prefixed helper names to explicit public names.
- Updated `weiss_rl.cli` to import those public workflow helpers.
- Added `__all__` to `weiss_rl.workflows.thesis` so the workflow module exposes a clear small API surface.

### Commands Run

| Command | Result |
| --- | --- |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_cli_workflow.py` | Passed: 19 passed. |
| `uv run --extra dev --extra sim python -m ruff check python/weiss_rl/cli.py python/weiss_rl/workflows/thesis.py` | Passed. |
| `uv run --extra dev --extra sim python -m ruff format --check python/weiss_rl/cli.py python/weiss_rl/workflows/thesis.py` | Passed: 2 files already formatted. |
| `uv run --extra dev --extra sim python -m mypy python/weiss_rl/cli.py python/weiss_rl/workflows/thesis.py --show-error-codes --no-error-summary` | Passed. |

### Tests Added

- None; existing package CLI workflow tests cover the command-building behavior.

### Files Moved or Deleted

- None.

### Behavior Changes

- None. This is a naming/API clarity cleanup inside the package CLI workflow layer.

### Remaining Risks

- `weiss_rl.cli` still has a long dispatch function. A future pass can split subcommand handlers once tests cover each branch well enough.

### Next Action

- Continue source cleanup by extracting or simplifying the next small CLI/script dispatch cluster without changing generated commands.

## 2026-05-29 - Package CLI Dispatch Split

### Changes

- Moved the long parsed-argument dispatch body out of `weiss_rl.cli` and into `weiss_rl.workflows.cli_dispatch`.
- Kept `weiss_rl.cli` focused on parser construction and a single dispatch call.
- Preserved generated command behavior through the existing package CLI workflow tests.

### Commands Run

| Command | Result |
| --- | --- |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_cli_workflow.py` | Passed: 19 passed. |
| `uv run --extra dev --extra sim python -m ruff check python/weiss_rl/cli.py python/weiss_rl/workflows/thesis.py python/weiss_rl/workflows/cli_dispatch.py` | Passed. |
| `uv run --extra dev --extra sim python -m ruff format --check python/weiss_rl/cli.py python/weiss_rl/workflows/thesis.py python/weiss_rl/workflows/cli_dispatch.py` | Passed: 3 files already formatted. |
| `uv run --extra dev --extra sim python -m mypy python/weiss_rl/cli.py python/weiss_rl/workflows/thesis.py python/weiss_rl/workflows/cli_dispatch.py --show-error-codes --no-error-summary` | Passed. |
| `Get-ChildItem python/weiss_rl/cli.py python/weiss_rl/workflows/cli_dispatch.py python/weiss_rl/workflows/thesis.py <line count>` | `cli.py`: 208 lines; `cli_dispatch.py`: 350 lines; `thesis.py`: 544 lines. |

### Tests Added

- None. Existing CLI workflow tests cover the generated-command contract.

### Files Moved or Deleted

- Added `python/weiss_rl/workflows/cli_dispatch.py`.

### Behavior Changes

- None intended. This is a source-organization change; CLI generated commands and validation semantics are preserved by focused tests.

### Remaining Risks

- `cli_dispatch.py` is still a substantial dispatch module. It is clearer than embedding this logic in the parser, but future work can split command families further if the public workflow surface grows.

### Next Action

- Continue improving package readability in small slices, preferably around configs or workflow command builders where tests already characterize behavior.

## 2026-05-29 - Workflow Snapshot Resolution Split

### Changes

- Extracted package-workflow snapshot checkpoint resolution helpers into `weiss_rl.workflows.snapshots`.
- Kept command builders and plan writing in `weiss_rl.workflows.thesis`.
- Updated CLI dispatch to import snapshot resolution from the new focused module.

### Commands Run

| Command | Result |
| --- | --- |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_cli_workflow.py` | First run failed because `thesis.py` still needed `json` for `_write_plan()` after the extraction; rerun passed: 19 passed. |
| `uv run --extra dev --extra sim python -m ruff check python/weiss_rl/cli.py python/weiss_rl/workflows/thesis.py python/weiss_rl/workflows/cli_dispatch.py python/weiss_rl/workflows/snapshots.py` | Passed after Ruff sorted the new dispatch imports. |
| `uv run --extra dev --extra sim python -m ruff format --check python/weiss_rl/cli.py python/weiss_rl/workflows/thesis.py python/weiss_rl/workflows/cli_dispatch.py python/weiss_rl/workflows/snapshots.py` | Passed: 4 files already formatted. |
| `uv run --extra dev --extra sim python -m mypy python/weiss_rl/cli.py python/weiss_rl/workflows/thesis.py python/weiss_rl/workflows/cli_dispatch.py python/weiss_rl/workflows/snapshots.py --show-error-codes --no-error-summary` | Passed. |
| `Get-ChildItem python/weiss_rl/cli.py python/weiss_rl/workflows/cli_dispatch.py python/weiss_rl/workflows/thesis.py python/weiss_rl/workflows/snapshots.py <line count>` | `cli.py`: 208; `cli_dispatch.py`: 352; `thesis.py`: 442; `snapshots.py`: 111. |

### Tests Added

- None. Existing CLI workflow tests cover B1 alias, single-snapshot smoke fallback, and guided seed checkpoint resolution.

### Files Moved or Deleted

- Added `python/weiss_rl/workflows/snapshots.py`.

### Behavior Changes

- None intended. Snapshot policy-id resolution, error messages, and checkpoint path construction were moved without semantic changes.

### Remaining Risks

- `weiss_rl.workflows.thesis` still combines profiles, plan writing, and command builders. It is smaller now, and future splits can target plan writing or command families.

### Next Action

- Continue workflow cleanup by separating plan writing or command builder families if tests remain focused and cheap.

## 2026-05-29 - Workflow Plan Helper Split

### Changes

- Extracted dry-run plan writing and command execution from `weiss_rl.workflows.thesis` into `weiss_rl.workflows.plans`.
- Updated `weiss_rl.workflows.cli_dispatch` to import `run_or_write_plan` from the new focused module.
- Left `weiss_rl.workflows.thesis` focused on profiles and command builders.

### Commands Run

| Command | Result |
| --- | --- |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_cli_workflow.py` | Passed: 19 passed. |
| `uv run --extra dev --extra sim python -m ruff check python/weiss_rl/cli.py python/weiss_rl/workflows/thesis.py python/weiss_rl/workflows/cli_dispatch.py python/weiss_rl/workflows/snapshots.py python/weiss_rl/workflows/plans.py` | Passed. |
| `uv run --extra dev --extra sim python -m ruff format --check python/weiss_rl/cli.py python/weiss_rl/workflows/thesis.py python/weiss_rl/workflows/cli_dispatch.py python/weiss_rl/workflows/snapshots.py python/weiss_rl/workflows/plans.py` | Passed: 5 files already formatted. |
| `uv run --extra dev --extra sim python -m mypy python/weiss_rl/cli.py python/weiss_rl/workflows/thesis.py python/weiss_rl/workflows/cli_dispatch.py python/weiss_rl/workflows/snapshots.py python/weiss_rl/workflows/plans.py --show-error-codes --no-error-summary` | Passed. |
| `Get-ChildItem python/weiss_rl/cli.py python/weiss_rl/workflows/cli_dispatch.py python/weiss_rl/workflows/thesis.py python/weiss_rl/workflows/snapshots.py python/weiss_rl/workflows/plans.py <line count>` | `cli.py`: 208; `cli_dispatch.py`: 352; `thesis.py`: 407; `snapshots.py`: 111; `plans.py`: 41. |

### Tests Added

- None. Existing CLI dry-run tests cover workflow plan payloads.

### Files Moved or Deleted

- Added `python/weiss_rl/workflows/plans.py`.

### Behavior Changes

- None intended. Dry-run output, saved plan shape, subprocess return handling, and generated commands are preserved.

### Remaining Risks

- `cli_dispatch.py` remains the largest workflow module and can be split by command family later.

### Next Action

- Shift to another small package cleanup surface, or run a broader validation checkpoint after the workflow-module series.

## 2026-05-29 - Workflow Split Focused Checkpoint

### Changes

- Updated `docs/architecture.md` to describe the cleaned `weiss_rl.workflows` package roles: profiles, dispatch, dry-run plans, snapshot resolution, and command builders.

### Commands Run

| Command | Result |
| --- | --- |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_cli_workflow.py python/weiss_rl/tests/test_makefile_figures.py python/weiss_rl/tests/test_script_entrypoint_smokes.py` | Passed: 35 passed, 2 skipped, 14 warnings. |

### Tests Added

- None.

### Files Moved or Deleted

- None.

### Behavior Changes

- None. This checkpoint validated the accumulated CLI/workflow split against package CLI, Makefile target, and script entrypoint smoke tests.

### Remaining Risks

- Warnings are from third-party matplotlib/pyparsing deprecations in the script-entrypoint smoke cluster.
- Broader full-suite validation has not been rerun after this workflow series.

### Next Action

- Choose the next cleanup surface or run a wider repo validation checkpoint before moving into behavior-sensitive runtime/model code.

## 2026-05-29 - CLI Parser Module Split

### Changes

- Extracted package CLI parser construction from `weiss_rl.cli` into `weiss_rl.workflows.cli_parser`.
- Added explicit `build_workflow_parser()` and `parse_workflow_args()` helpers so parser tests can target the workflow layer without making the public entrypoint grow again.
- Reduced `weiss_rl.cli` to the canonical front door: parse package CLI arguments, then dispatch them.
- Updated `docs/architecture.md` to include parser construction in the `weiss_rl.workflows` package role.

### Commands Run

| Command | Result |
| --- | --- |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_cli_workflow.py` | Passed: 19 passed. |
| `uv run --extra dev --extra sim python -m ruff check python/weiss_rl/cli.py python/weiss_rl/workflows/cli_parser.py python/weiss_rl/workflows/cli_dispatch.py python/weiss_rl/workflows/thesis.py python/weiss_rl/workflows/snapshots.py python/weiss_rl/workflows/plans.py` | Passed. |
| `uv run --extra dev --extra sim python -m ruff format --check python/weiss_rl/cli.py python/weiss_rl/workflows/cli_parser.py python/weiss_rl/workflows/cli_dispatch.py python/weiss_rl/workflows/thesis.py python/weiss_rl/workflows/snapshots.py python/weiss_rl/workflows/plans.py` | Passed: 6 files already formatted. |
| `uv run --extra dev --extra sim python -m mypy python/weiss_rl/cli.py python/weiss_rl/workflows/cli_parser.py python/weiss_rl/workflows/cli_dispatch.py python/weiss_rl/workflows/thesis.py python/weiss_rl/workflows/snapshots.py python/weiss_rl/workflows/plans.py --show-error-codes --no-error-summary` | Passed. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_cli_workflow.py python/weiss_rl/tests/test_makefile_figures.py python/weiss_rl/tests/test_script_entrypoint_smokes.py` | Passed: 35 passed, 2 skipped, 14 warnings. |
| `Get-Content <workflow files> <line count>` | `cli.py`: 12; `cli_parser.py`: 208; `cli_dispatch.py`: 352; `thesis.py`: 407; `snapshots.py`: 111; `plans.py`: 41. |

### Tests Added

- None. Existing package CLI workflow and script-entrypoint smoke tests cover this split.

### Files Moved or Deleted

- Added `python/weiss_rl/workflows/cli_parser.py`.
- Replaced `python/weiss_rl/cli.py` with a small parse-and-dispatch entrypoint.

### Behavior Changes

- None intended. Subcommands, flags, defaults, aliases, help strings, dry-run behavior, and dispatch behavior are preserved.

### Remaining Risks

- `weiss_rl.workflows.cli_dispatch` is still the largest workflow module.
- Third-party matplotlib/pyparsing warnings remain in the script-entrypoint smoke cluster.

### Next Action

- Continue with another visible public-surface cleanup slice, or run a wider validation checkpoint before touching runtime/model internals.

## 2026-05-29 - Workflow Dispatch Family Split

### Changes

- Split workflow command handlers out of the remaining large `weiss_rl.workflows.cli_dispatch` module.
- Added `weiss_rl.workflows.dispatch_training` for B1, guided B1 seed, main league, and guided-bootstrap training workflows.
- Added `weiss_rl.workflows.dispatch_evaluation` for smoke/final eval, figures, B2 audit, and guard-run workflows.
- Added `weiss_rl.workflows.dispatch_bootstrap` for segmented guided-bootstrap and guarded league bootstrap controllers.
- Reduced `cli_dispatch.py` to a small command router that resolves the repo root and Python executable, then delegates to the relevant command family.
- Updated `docs/architecture.md` to describe the workflow package as parser, router, handlers, plans, snapshots, and builders.

### Commands Run

| Command | Result |
| --- | --- |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_cli_workflow.py` | Passed: 19 passed. |
| `uv run --extra dev --extra sim python -m ruff check python/weiss_rl/cli.py python/weiss_rl/workflows` | Passed. |
| `uv run --extra dev --extra sim python -m ruff format --check python/weiss_rl/cli.py python/weiss_rl/workflows` | Passed: 10 files already formatted. |
| `uv run --extra dev --extra sim python -m mypy python/weiss_rl/cli.py python/weiss_rl/workflows --show-error-codes --no-error-summary` | Passed. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_cli_workflow.py python/weiss_rl/tests/test_makefile_figures.py python/weiss_rl/tests/test_script_entrypoint_smokes.py` | Passed: 35 passed, 2 skipped, 14 warnings. |
| `Get-Content <workflow files> <line count>` | `cli.py`: 12; `cli_parser.py`: 208; `cli_dispatch.py`: 64; `dispatch_training.py`: 145; `dispatch_evaluation.py`: 105; `dispatch_bootstrap.py`: 99; `thesis.py`: 407; `snapshots.py`: 111; `plans.py`: 41. |

### Tests Added

- None. Existing package CLI workflow and script-entrypoint smoke tests cover the dispatch split.

### Files Moved or Deleted

- Added `python/weiss_rl/workflows/dispatch_training.py`.
- Added `python/weiss_rl/workflows/dispatch_evaluation.py`.
- Added `python/weiss_rl/workflows/dispatch_bootstrap.py`.
- Replaced `python/weiss_rl/workflows/cli_dispatch.py` with a small router.

### Behavior Changes

- None intended. Generated commands, validation errors, dry-run payloads, and plan names are preserved.

### Remaining Risks

- `weiss_rl.workflows.thesis` still holds all command builders. It is readable, but a future split by builder family may make it calmer.
- Third-party matplotlib/pyparsing warnings remain in the script-entrypoint smoke cluster.

### Next Action

- Continue the public-surface cleanup by splitting workflow command builders or run a wider validation checkpoint before runtime/model internals.

## 2026-05-29 - Workflow Profiles and Command Builder Split

### Changes

- Split the remaining mixed `weiss_rl.workflows.thesis` module into focused modules.
- Added `weiss_rl.workflows.profiles` for standard train profiles, thesis config paths, repo-root resolution, and guided-bootstrap stack selection.
- Added `weiss_rl.workflows.command_builders` for deterministic subprocess command construction.
- Updated parser, router, and handler modules to import from the focused modules directly.
- Deleted `python/weiss_rl/workflows/thesis.py` instead of keeping a broad re-export shim.

### Commands Run

| Command | Result |
| --- | --- |
| `rg "workflows\\.thesis|from weiss_rl\\.workflows\\.thesis|import weiss_rl\\.workflows\\.thesis" python README.md docs Makefile` | No current code/import references; matches are historical log entries only. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_cli_workflow.py` | Passed: 19 passed. |
| `uv run --extra dev --extra sim python -m ruff check python/weiss_rl/cli.py python/weiss_rl/workflows` | Initially failed on import ordering in `dispatch_training.py`; passed after Ruff fixed the import order. |
| `uv run --extra dev --extra sim python -m ruff format --check python/weiss_rl/cli.py python/weiss_rl/workflows` | Passed: 11 files already formatted. |
| `uv run --extra dev --extra sim python -m mypy python/weiss_rl/cli.py python/weiss_rl/workflows --show-error-codes --no-error-summary` | Passed. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_cli_workflow.py python/weiss_rl/tests/test_makefile_figures.py python/weiss_rl/tests/test_script_entrypoint_smokes.py` | Passed: 35 passed, 2 skipped, 14 warnings. |
| `Get-Content <workflow files> <line count>` | `cli.py`: 12; `cli_parser.py`: 208; `cli_dispatch.py`: 64; `profiles.py`: 121; `command_builders.py`: 298; `dispatch_training.py`: 145; `dispatch_evaluation.py`: 105; `dispatch_bootstrap.py`: 99; `snapshots.py`: 111; `plans.py`: 41. |

### Tests Added

- None. Existing package CLI workflow and script-entrypoint smoke tests cover the import move and deleted module.

### Files Moved or Deleted

- Added `python/weiss_rl/workflows/profiles.py`.
- Added `python/weiss_rl/workflows/command_builders.py`.
- Deleted `python/weiss_rl/workflows/thesis.py`.

### Behavior Changes

- None intended. Profile values, config paths, generated commands, dry-run payloads, validation errors, and plan names are preserved.

### Remaining Risks

- `command_builders.py` is now the largest workflow module, but it is a single-purpose list of deterministic command builders rather than a mixed parser/dispatch/profile module.
- Third-party matplotlib/pyparsing warnings remain in the script-entrypoint smoke cluster.

### Next Action

- Run a wider validation checkpoint for the accumulated workflow cleanup, then move to the next thesis-facing public surface.

## 2026-05-29 - Workflow Cleanup Validation Checkpoint

### Changes

- Ran a wider verifier after the workflow package cleanup.
- Fixed artifact hygiene scanning so missing tracked paths in a dirty worktree are skipped instead of crashing. This matters while archived files are moved but not yet staged.
- Restored legacy and seat-aware learner time-major behavior where omitted `initial_hidden_state` lets the model initialize hidden state, matching existing model forward semantics.
- Added a regression test for dirty-worktree artifact hygiene scanning.

### Commands Run

| Command | Result |
| --- | --- |
| `uv run --extra dev --extra sim python python/scripts/verify_repo.py` | Initially failed: `heuristic_sanity_scan.py` needed formatting. |
| `uv run --extra dev --extra sim python -m ruff format python/scripts/heuristic_sanity_scan.py` | Reformatted 1 file. |
| `uv run --extra dev --extra sim python python/scripts/verify_repo.py` | Initially failed on missing tracked archive-move paths in artifact hygiene and optional hidden-state learner tests. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_artifact_hygiene.py` | Passed: 8 passed. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_impala_learner.py python/weiss_rl/tests/test_ppo_lite_learner.py python/weiss_rl/tests/test_vtrace.py` | Passed: 76 passed. |
| `uv run --extra dev --extra sim python -m ruff check python/weiss_rl/diagnostics/artifact_hygiene.py python/weiss_rl/tests/test_artifact_hygiene.py python/weiss_rl/learners/forward_time_major.py` | Passed. |
| `uv run --extra dev --extra sim python -m ruff format --check python/weiss_rl/diagnostics/artifact_hygiene.py python/weiss_rl/tests/test_artifact_hygiene.py python/weiss_rl/learners/forward_time_major.py` | Initially failed on `artifact_hygiene.py`; passed after formatting. |
| `uv run --extra dev --extra sim python python/scripts/verify_repo.py` | Passed: local verification completed; pytest 1892 passed, 2 skipped, 14 warnings. |
| `uv run --extra dev --extra sim python -m ruff check .` | Passed. |
| `uv run --extra dev --extra sim python -m ruff format --check .` | Passed: 611 files already formatted. |

### Tests Added

- Added `test_scan_tracked_repo_tree_skips_missing_tracked_paths_in_dirty_worktree`.

### Files Moved or Deleted

- None in this checkpoint.

### Behavior Changes

- No intended RL, training, evaluation, checkpoint, artifact-format, legal-action, or simulator-contract behavior changes.
- Artifact hygiene now skips missing tracked files during repo scans rather than raising `FileNotFoundError`.
- The learner time-major fallback again allows omitted hidden state when the model supports initialization from `None`.

### Remaining Risks

- `make verify` and `make artifact-hygiene` remain untested in this Windows shell if `make` is unavailable.
- The full verifier still reports third-party matplotlib/pyparsing deprecation warnings.

### Next Action

- Move to the next thesis-facing cleanup surface with the workflow package now verifier-clean.

## 2026-05-29 - Public Config Surface Docs Guard

### Changes

- Updated the reward-component probe in `docs/thesis_workflow.md` to use the canonical `configs/thesis/b1_noleague.yaml` stack instead of advertising the historical `configs/thesis/ablations/full_shaping_reward.yaml` probe config.
- Added a regression test that keeps the public README/docs surface limited to the canonical thesis ablation configs:
  - `configs/thesis/ablations/no_gru.yaml`
  - `configs/thesis/ablations/ppo_lite.yaml`
  - `configs/thesis/ablations/terminal_only_reward.yaml`

### Commands Run

| Command | Result |
| --- | --- |
| `python scripts/check_docs_links.py` | Failed: script is not present at that path in this checkout. |
| `python -m pytest -q python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed: 1 passed. |
| `python -m ruff check python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed. |
| `python -m ruff format --check python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed: 1 file already formatted. |

### Tests Added

- Added `test_public_docs_only_advertise_canonical_thesis_ablations`.

### Files Moved or Deleted

- None.

### Behavior Changes

- None intended. This is docs/test-only and does not alter training, evaluation, simulator contract, config parsing, artifacts, or checkpoint behavior.

### Remaining Risks

- Many historical probe configs still physically live in `configs/thesis/ablations/` for compatibility with existing characterization tests and historical command records. The public docs now guard against presenting those probe files as the standard thesis surface.
- The old `python scripts/check_docs_links.py` command referenced by prior memory/docs is absent in this checkout.

### Next Action

- Continue shrinking the thesis-facing config surface by moving or reclassifying historical ablation/probe configs once their characterization tests are either redirected to an archive path or replaced with narrower contract fixtures.

## 2026-05-29 - Reward Probe Config Archive

### Changes

- Moved the noncanonical May 13 reward-shaping probe configs out of `configs/thesis/ablations/` and into `configs/archive/thesis_reward_ablations_20260513/`.
- Kept `configs/thesis/ablations/terminal_only_reward.yaml` as the public terminal-only thesis ablation.
- Kept `configs/thesis/ablations/reward_ablation_base.yaml` in place for now because the canonical terminal-only config still extends it.
- Rewrote archived reward-probe `extends:` paths that depend on the shared reward base.
- Updated `public_teacher_tactical_mulliganguard_reward.yaml` so its historical guided-teacher config still loads through the archived full-shaping probe stack.
- Redirected characterization tests that intentionally cover the old reward probes to the archive path.
- Documented the archive bucket in `configs/archive/README.md`, `configs/archive/thesis_reward_ablations_20260513/README.md`, and `configs/thesis/ablations/README.md`.

### Commands Run

| Command | Result |
| --- | --- |
| `rg <moved reward-probe paths> README.md docs configs python/weiss_rl/tests python/scripts Makefile pyproject.toml` | Live references reduced to historical log entries and archive documentation. |
| `python -m pytest -q python/weiss_rl/tests/test_config_loader.py::test_thesis_reward_ablations_are_isolated_b1_routes python/weiss_rl/tests/test_public_config_surface_docs.py python/weiss_rl/tests/test_runtime.py::test_central_structured_unroll_snapshots_replay_behavior_logp` | Failed: plain Python cannot import `torch`, so `test_runtime.py` could not collect. |
| `python -m pytest -q python/weiss_rl/tests/test_config_loader.py::test_thesis_reward_ablations_are_isolated_b1_routes python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed: 2 passed. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_runtime.py::test_central_structured_unroll_snapshots_replay_behavior_logp` | Passed: 1 passed. |
| `python -m pytest -q python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_guided_b1_ablation` | Passed: 1 passed. |
| `python -m ruff check python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed. |
| `python -m ruff format --check python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed: 3 files already formatted. |

### Tests Added

- None. Existing config characterization tests now verify the archived reward probes, while the public-doc guard still verifies only canonical public ablations are advertised.

### Files Moved or Deleted

- Moved 19 historical reward-probe YAML files from `configs/thesis/ablations/` to `configs/archive/thesis_reward_ablations_20260513/`.
- Added `configs/archive/thesis_reward_ablations_20260513/README.md`.

### Behavior Changes

- None intended. The moved configs still load with the same effective settings through their archive paths.
- No simulator contract, observation/action interpretation, legal-action ordering, reward semantics for live configs, training loop, evaluation, checkpoint format, or artifact format was changed.

### Remaining Risks

- Historical rebuild-log commands still mention the old `configs/thesis/ablations/...` paths as provenance text.
- `reward_ablation_base.yaml` remains in `configs/thesis/ablations/` as an internal shared config for the canonical terminal-only reward ablation. A later slice can move it into a shared thesis support location after updating `terminal_only_reward.yaml`.
- Other noncanonical guided-teacher and main-league probe configs still live in `configs/thesis/ablations/` and should be archived in smaller dependency-aware clusters.

### Next Action

- Move the `reward_ablation_base.yaml` support file out of the public ablation directory, or archive the guided-teacher reward probe cluster that now depends on the archived reward stack.

## 2026-05-29 - Internal Reward Base Config

### Changes

- Moved `reward_ablation_base.yaml` out of the public `configs/thesis/ablations/` directory and into `configs/thesis/_shared/reward_ablation_base.yaml`.
- Added `configs/thesis/_shared/README.md` to make clear that shared thesis fragments are not launch targets.
- Updated `configs/thesis/ablations/terminal_only_reward.yaml` to extend the internal shared reward base.
- Updated the archived May 13 reward probes to extend the internal shared reward base.
- Updated `configs/README.md` to distinguish public thesis configs from internal shared fragments.

### Commands Run

| Command | Result |
| --- | --- |
| `python -m pytest -q python/weiss_rl/tests/test_config_loader.py::test_thesis_reward_ablations_are_isolated_b1_routes python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_guided_b1_ablation python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed: 3 passed. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_runtime.py::test_central_structured_unroll_snapshots_replay_behavior_logp` | Passed: 1 passed. |
| `python -m ruff check python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed. |
| `python -m ruff format --check python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_runtime.py python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed: 3 files already formatted. |
| `python -c "<load archived reward probes plus terminal_only_reward>"` | Failed: plain Python could not import `weiss_rl` in this checkout. |
| `uv run --extra dev --extra sim python -c "<load archived reward probes plus terminal_only_reward>"` | Passed: loaded 20 reward configs. |

### Tests Added

- None. Existing config and runtime characterization tests cover the moved base through the canonical terminal-only config, archived reward probes, and guided-teacher config that extends an archived reward probe.

### Files Moved or Deleted

- Moved `configs/thesis/ablations/reward_ablation_base.yaml` to `configs/thesis/_shared/reward_ablation_base.yaml`.
- Added `configs/thesis/_shared/README.md`.

### Behavior Changes

- None intended. Effective config values are preserved; only config file organization and `extends:` paths changed.
- No simulator contract, legal-action semantics, reward values, training loop, evaluation flow, checkpoint compatibility, or artifact format changed.

### Remaining Risks

- Historical rebuild/refactor log entries still mention the old support-file location as provenance text.
- Other noncanonical guided-teacher and main-league probe configs still live under `configs/thesis/ablations/` and should be archived in dependency-aware groups.

### Next Action

- Archive the guided-teacher reward probe cluster or split broader main-league probe configs out of the public ablation directory, preserving characterization tests for each moved cluster.

## 2026-05-29 - Internal Guided-Teacher Config Stack

### Changes

- Moved the `public_teacher_*reward.yaml` guided-teacher config stack out of `configs/thesis/ablations/` and into `configs/thesis/_shared/guided_teacher/`.
- Added `configs/thesis/_shared/guided_teacher/README.md` to mark the stack as internal support, not a public launch surface.
- Updated `configs/thesis/b1_guided_seed.yaml` to extend the internal guided-teacher stack.
- Updated `configs/thesis/ablations/main_league_guided_factorized_continuation_no_b1_anchor_probe.yaml` to extend the internal guided-teacher stack.
- Updated config characterization tests and stall-monitor tests to load guided-teacher configs from the internal shared path.
- Left historical rebuild-log command records unchanged as provenance text.

### Commands Run

| Command | Result |
| --- | --- |
| `rg <public-teacher references> configs README.md docs python/weiss_rl/tests python/scripts Makefile pyproject.toml` | Audited: current functional references were in config tests, stall-monitor tests, `b1_guided_seed.yaml`, and one main-league probe. |
| `rg 'configs/thesis/ablations/public_teacher' configs/thesis python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_train_stall_monitor.py` | Passed: no stale public-teacher ablation paths remain in live configs/tests. |
| `Get-ChildItem -Name configs/thesis/ablations | Where-Object { $_ -like 'public_teacher*reward.yaml' }` | Passed: no public-teacher reward files remain in the public ablation directory. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_train_stall_monitor.py::test_dev_eval_ineligibility_reasons_apply_checkpoint_confidence_when_stall_monitor_disabled python/weiss_rl/tests/test_train_stall_monitor.py::test_confirmatory_dev_eval_request_targets_multianchor_near_miss_candidate python/weiss_rl/tests/test_train_stall_monitor.py::test_confirmatory_dev_eval_request_rejects_multianchor_clear_anchor_failure python/weiss_rl/tests/test_cli_workflow.py::test_package_cli_train_b1_guided_seed_uses_guided_seed_config python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed: 209 passed, 14 third-party warnings. |
| `uv run --extra dev --extra sim python -c "<load all configs/thesis/_shared/guided_teacher/*.yaml plus b1_guided_seed and dependent main probe>"` | Passed: loaded 32 guided-teacher configs. |
| `python -m ruff check python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_train_stall_monitor.py python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed. |
| `python -m ruff format --check python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_train_stall_monitor.py python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed after formatting two touched test files. |

### Tests Added

- None. Existing characterization tests now point at the internal guided-teacher path.

### Files Moved or Deleted

- Moved 30 `public_teacher_*reward.yaml` files from `configs/thesis/ablations/` to `configs/thesis/_shared/guided_teacher/`.
- Added `configs/thesis/_shared/guided_teacher/README.md`.

### Behavior Changes

- None intended. Effective config composition is preserved through updated `extends:` paths.
- No simulator contract, observation/action interpretation, legal-action ordering, reward values, training runtime, evaluation flow, checkpoint compatibility, artifact format, or metric semantics changed.

### Remaining Risks

- Historical rebuild-log commands still mention old `configs/thesis/ablations/public_teacher_...` paths as provenance.
- Several noncanonical main-league probe configs still live in `configs/thesis/ablations/`; those should move in dependency-aware batches.

### Next Action

- Archive or internalize the next coherent probe cluster under `configs/thesis/ablations/`, likely the guided-factorized main-league continuation probes that now depend on the internal guided-teacher stack.

## 2026-05-29 - Internal Guided-Factorized Main Stack

### Changes

- Moved the `main_league_guided_factorized*.yaml` main-league continuation/probe stack out of `configs/thesis/ablations/` and into `configs/thesis/_shared/guided_factorized/`.
- Added `configs/thesis/_shared/guided_factorized/README.md`.
- Updated `configs/thesis/main_league_guided_bootstrap.yaml` to extend the internal guided-factorized stack.
- Updated the moved stack's cross-folder `extends:` paths for `main_league.yaml` and the internal guided-teacher stack.
- Updated `test_config_loader.py` characterization paths to the internal guided-factorized location.
- Updated `configs/thesis/_shared/README.md` to list the guided-teacher and guided-factorized internal stacks.

### Commands Run

| Command | Result |
| --- | --- |
| `rg main_league_guided_factorized configs README.md docs python/weiss_rl/tests python/scripts Makefile pyproject.toml` | Audited: current functional references were `main_league_guided_bootstrap.yaml` and config-loader characterization tests; historical rebuild-log command records remain unchanged. |
| `Get-ChildItem -Name configs/thesis/ablations | Where-Object { $_ -like 'main_league_guided_factorized*.yaml' }` | Passed: no guided-factorized files remain in the public ablation directory. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_cli_workflow.py::test_package_cli_train_main_guided_bootstrap_uses_seed_and_warmstart_without_strict_b1 python/weiss_rl/tests/test_cli_workflow.py::test_package_cli_train_main_guided_bootstrap_vtrace_uses_clamped_stack python/weiss_rl/tests/test_cli_workflow.py::test_package_cli_train_main_guided_bootstrap_seed_champions_uses_seedchampion_stack python/weiss_rl/tests/test_cli_workflow.py::test_package_cli_train_main_guided_bootstrap_selected_resolves_init_policy_id python/weiss_rl/tests/test_public_config_surface_docs.py` | Initially failed on one moved `extends: ../main_league.yaml`; passed after updating it to `../../main_league.yaml`: 209 passed. |
| `uv run --extra dev --extra sim python -c "<load all configs/thesis/_shared/guided_factorized/*.yaml plus main_league_guided_bootstrap>"` | Initially failed on the same moved `extends:` path; passed after the fix: loaded 21 guided-factorized configs. |
| `python -m ruff check python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed. |
| `python -m ruff format --check python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed after formatting `test_config_loader.py`. |

### Tests Added

- None. Existing config and CLI workflow characterization tests now point at the internal guided-factorized path.

### Files Moved or Deleted

- Moved 20 `main_league_guided_factorized*.yaml` files from `configs/thesis/ablations/` to `configs/thesis/_shared/guided_factorized/`.
- Added `configs/thesis/_shared/guided_factorized/README.md`.

### Behavior Changes

- None intended. Effective config composition is preserved through updated `extends:` paths.
- No simulator contract, observation/action interpretation, legal-action ordering, reward values, training runtime, evaluation flow, checkpoint compatibility, artifact format, or metric semantics changed.

### Remaining Risks

- Historical rebuild-log commands still mention the old `configs/thesis/ablations/main_league_guided_factorized...` paths as provenance.
- Other noncanonical main-league probe configs still live under `configs/thesis/ablations/`; they should move in similarly bounded clusters.

### Next Action

- Continue reducing `configs/thesis/ablations/` by moving the next coherent main-league probe family, likely the `main_b1only_p2_*` or `main_league_champion_hardneg_*` cluster depending on dependency size.

## 2026-05-29 - Internal B1-Only P2 Probe Stack

### Changes

- Moved the four `main_b1only_p2*.yaml` trust-region probe configs out of `configs/thesis/ablations/` and into `configs/thesis/_shared/main_b1only_p2/`.
- Added `configs/thesis/_shared/main_b1only_p2/README.md`.
- Updated the moved base config to extend `../../main_league_guided_bootstrap_selected_trajbc_direct_b2b3b4_anchor_nopublic.yaml` from its new location.
- Updated `configs/thesis/ablations/main_league_champion_hardneg_long_probe.yaml`, the one live config that extends the p2 base, to point at the internal shared stack.
- Updated config-loader characterization tests to load the p2 probes from the internal shared path.
- Updated `configs/thesis/_shared/README.md` to list the p2 probe stack.

### Commands Run

| Command | Result |
| --- | --- |
| `rg main_b1only_p2 configs README.md docs python/weiss_rl/tests python/scripts Makefile pyproject.toml` | Audited: current functional references were config-loader tests and `main_league_champion_hardneg_long_probe.yaml`; historical report/log references remain as provenance. |
| `Get-ChildItem -Name configs/thesis/ablations | Where-Object { $_ -like 'main_b1only_p2*.yaml' }` | Passed: no p2 probe configs remain in the public ablation directory. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_b1only_p2_trust_region_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_b1only_p2_trust_region_no_warmup_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_b1only_p2_trust_region_argmax_opp_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_b1only_p2_free_argmax_opp_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_champion_hardneg_long_probe python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed: 6 passed. |
| `uv run --extra dev --extra sim python -c "<load all configs/thesis/_shared/main_b1only_p2/*.yaml plus main_league_champion_hardneg_long_probe>"` | Passed: loaded 5 main-b1only-p2 configs. |
| `python -m ruff check python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed. |
| `python -m ruff format --check python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed after formatting `test_config_loader.py`. |

### Tests Added

- None. Existing config-loader characterization tests now target the internal shared path.

### Files Moved or Deleted

- Moved 4 `main_b1only_p2*.yaml` files from `configs/thesis/ablations/` to `configs/thesis/_shared/main_b1only_p2/`.
- Added `configs/thesis/_shared/main_b1only_p2/README.md`.

### Behavior Changes

- None intended. Effective config composition is preserved through updated `extends:` paths.
- No simulator contract, legal-action semantics, reward values, training runtime, evaluation flow, checkpoint compatibility, artifact format, or metric semantics changed.

### Remaining Risks

- Historical rebuild/report docs still mention old `configs/thesis/ablations/main_b1only_p2...` paths as provenance.
- `configs/thesis/ablations/` still contains many `main_league_champion_hardneg_*` investigation configs.

### Next Action

- Move the first coherent `main_league_champion_hardneg_*` probe family out of the public ablation directory, starting with the early long/rehearsal/consolidation/stable/polish variants that build on the internal p2 stack.

## 2026-05-29 - Internal Hard-Negative Core Probe Stack

### Changes

- Moved the early hard-negative main-league core probes out of `configs/thesis/ablations/` and into `configs/thesis/_shared/hardneg_core/`.
- Added `configs/thesis/_shared/hardneg_core/README.md`.
- Kept `configs/thesis/ablations/main_league_champion_hardneg_multiobjective_guard_probe.yaml` public for characterization, but updated it to extend the internal shared core.
- Updated config-loader characterization tests to load the moved long/rehearsal/consolidation/stable/polish configs from the shared path.
- Updated `configs/thesis/_shared/README.md` to list the hard-negative core stack.

### Commands Run

| Command | Result |
| --- | --- |
| `Get-ChildItem -Path configs/thesis/ablations -Filter 'main_league_champion_hardneg*.yaml'` | Audited the remaining hard-negative public probe surface. |
| `rg -n "main_league_champion_hardneg_(long|rehearsal|consolidation|stable_long|polish|multiobjective_guard)_probe" configs python docs` | Audited current references; functional references now point at `_shared/hardneg_core/`, while historical rebuild/report docs remain as provenance. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_champion_hardneg_long_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_champion_hardneg_rehearsal_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_champion_hardneg_consolidation_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_champion_hardneg_stable_long_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_champion_hardneg_polish_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_champion_hardneg_multiobjective_guard_probe python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed: 7 passed. |
| `uv run --extra dev --extra sim python -c "<load all configs/thesis/_shared/hardneg_core/*.yaml plus main_league_champion_hardneg_multiobjective_guard_probe>"` | Passed: loaded 6 hardneg-core configs. |
| `uv run --extra dev python -m ruff check python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed. |
| `uv run --extra dev python -m ruff format --check python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_public_config_surface_docs.py` | Initially found formatting drift in `test_config_loader.py`; passed after running `uv run --extra dev python -m ruff format python/weiss_rl/tests/test_config_loader.py`. |

### Tests Added

- None. Existing config-loader tests now characterize the internal shared hard-negative core path.

### Files Moved or Deleted

- Moved 5 `main_league_champion_hardneg_{long,rehearsal,consolidation,stable_long,polish}_probe.yaml` configs from `configs/thesis/ablations/` to `configs/thesis/_shared/hardneg_core/`.
- Added `configs/thesis/_shared/hardneg_core/README.md`.

### Behavior Changes

- None intended. Effective config composition is preserved through updated `extends:` paths.
- No simulator contract, legal-action semantics, reward values, training runtime, evaluation flow, checkpoint compatibility, artifact format, metric aggregation, or league promotion behavior changed.

### Remaining Risks

- Historical rebuild/report docs still mention the old hard-negative core paths as provenance.
- `configs/thesis/ablations/` still contains many later hard-negative investigation configs that should be archived or moved in smaller dependency-aware clusters.

### Next Action

- Continue shrinking the public ablation directory by moving the next hard-negative branch that depends on `main_league_champion_hardneg_multiobjective_guard_probe.yaml`, likely the multiobjective retention/replay-BC retention group.

## 2026-05-29 - Internal Hard-Negative Retention Probe Stack

### Changes

- Moved the hard-negative multiobjective retention base and replay-BC retention variants out of `configs/thesis/ablations/` and into `configs/thesis/_shared/hardneg_retention/`.
- Added `configs/thesis/_shared/hardneg_retention/README.md`.
- Kept `configs/thesis/ablations/main_league_champion_hardneg_selected_retention_b4guard_probe.yaml` public for the later selected/all-outcome branch, but updated it to extend the internal retention base.
- Updated config-loader characterization tests to load the moved retention configs from the shared path.
- Updated `configs/thesis/_shared/README.md` to list the hard-negative retention stack.

### Commands Run

| Command | Result |
| --- | --- |
| `Get-ChildItem -Path configs/thesis/ablations -Filter 'main_league_champion_hardneg*retention*.yaml'` | Audited the remaining hard-negative retention public surface before the move. |
| `rg -n "main_league_champion_hardneg_(multiobjective_retention|replaybc_retention|balanced_replaybc_retention|weighted_replaybc_win32_retention)_probe" configs python docs README.md` | Audited references after the move; current configs and tests now point at `_shared/hardneg_retention/`, while historical rebuild-log commands remain as provenance. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_champion_hardneg_multiobjective_retention_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_champion_hardneg_replaybc_retention_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_champion_hardneg_balanced_replaybc_retention_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_champion_hardneg_weighted_replaybc_win32_retention_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_champion_hardneg_selected_retention_b4guard_probe python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed: 6 passed. |
| `uv run --extra dev --extra sim python -c "<load all configs/thesis/_shared/hardneg_retention/*.yaml plus main_league_champion_hardneg_selected_retention_b4guard_probe>"` | Passed: loaded 5 hardneg-retention configs. |
| `uv run --extra dev python -m ruff check python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed. |
| `uv run --extra dev python -m ruff format --check python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed: 2 files already formatted. |

### Tests Added

- None. Existing config-loader tests now characterize the internal shared hard-negative retention path.

### Files Moved or Deleted

- Moved 4 `main_league_champion_hardneg_{multiobjective_retention,replaybc_retention,balanced_replaybc_retention,weighted_replaybc_win32_retention}_probe.yaml` configs from `configs/thesis/ablations/` to `configs/thesis/_shared/hardneg_retention/`.
- Added `configs/thesis/_shared/hardneg_retention/README.md`.

### Behavior Changes

- None intended. Effective config composition is preserved through updated `extends:` paths.
- No simulator contract, legal-action semantics, reward values, replay dataset paths, training runtime, evaluation flow, checkpoint compatibility, artifact format, metric aggregation, or league promotion behavior changed.

### Remaining Risks

- Historical rebuild-log commands still mention the old retention config paths as provenance.
- The selected/all-outcome hard-negative branch still lives under `configs/thesis/ablations/`; it now depends on the shared retention stack and should be moved in a separate dependency-aware pass.

### Next Action

- Move the next selected hard-negative branch that depends on `main_league_champion_hardneg_selected_retention_b4guard_probe.yaml`, starting with the selected replay-BC and conservative online guard configs if their downstream references are clean.

## 2026-05-29 - Internal Hard-Negative Selected Base Stack

### Changes

- Moved the selected-checkpoint hard-negative base configs out of `configs/thesis/ablations/` and into `configs/thesis/_shared/hardneg_selected/`.
- Added `configs/thesis/_shared/hardneg_selected/README.md`.
- Updated public selected all-outcome and conservative follow-up configs to extend the moved shared base configs.
- Updated config-loader characterization tests to load the moved selected-retention, selected all-outcome replay-BC, and selected conservative online guard configs from the shared path.
- Updated `configs/thesis/_shared/README.md` to list the selected hard-negative stack.

### Commands Run

| Command | Result |
| --- | --- |
| `Get-ChildItem -Path configs/thesis/ablations -Filter 'main_league_champion_hardneg_selected*.yaml'` | Audited the remaining selected hard-negative public config surface before the move. |
| `rg -n "extends:.*main_league_champion_hardneg_selected|main_league_champion_hardneg_selected_(retention_b4guard|alloutcome_replaybc_b4guard|conservative_online_guard)_probe" configs python docs README.md` | Audited references and direct dependents before and after the move; current configs and tests now point at `_shared/hardneg_selected/`, while historical logs remain as provenance. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_champion_hardneg_selected_retention_b4guard_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_champion_hardneg_selected_alloutcome_replaybc_b4guard_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_selected_conservative_online_guard_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_champion_hardneg_selected_alloutcome_b2repair_b4guard_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_lowpressure_pairloss_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_paired_swing_contrastive_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_paired_flipbc_conservative_probe python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed: 8 passed. |
| `uv run --extra dev --extra sim python -c "<load configs/thesis/_shared/hardneg_selected/*.yaml plus all configs/thesis/ablations/main_league_champion_hardneg_selected*.yaml>"` | Passed: loaded 36 selected hardneg configs. |
| `uv run --extra dev python -m ruff check python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed. |
| `uv run --extra dev python -m ruff format --check python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed. |

### Tests Added

- None. Existing config-loader tests now characterize the internal shared selected hard-negative base path.

### Files Moved or Deleted

- Moved 3 `main_league_champion_hardneg_selected_{retention_b4guard,alloutcome_replaybc_b4guard,conservative_online_guard}_probe.yaml` configs from `configs/thesis/ablations/` to `configs/thesis/_shared/hardneg_selected/`.
- Added `configs/thesis/_shared/hardneg_selected/README.md`.

### Behavior Changes

- None intended. Effective config composition is preserved through updated `extends:` paths.
- No simulator contract, legal-action semantics, reward values, replay dataset paths, training runtime, evaluation flow, checkpoint compatibility, artifact format, metric aggregation, or league promotion behavior changed.

### Remaining Risks

- Historical rebuild/refactor log entries still mention the old selected config paths as provenance.
- The larger selected all-outcome, paired, and outcome-contrastive hard-negative investigation branches still live under `configs/thesis/ablations/`.

### Next Action

- Move the next selected all-outcome branch rooted at `main_league_champion_hardneg_selected_alloutcome_b2repair_b4guard_probe.yaml`, updating public descendants to extend an internal shared selected-alloutcome stack.

## 2026-05-29 - Internal Hard-Negative Selected All-Outcome Stack

### Changes

- Moved the selected-checkpoint all-outcome repair lineage out of `configs/thesis/ablations/` and into `configs/thesis/_shared/hardneg_selected_alloutcome/`.
- Added `configs/thesis/_shared/hardneg_selected_alloutcome/README.md`.
- Updated the moved B2-repair root to extend the internal selected all-outcome replay-BC base.
- Kept the later stratified winner-repair branch public, but updated it to extend the moved shared winner-repair config.
- Updated config-loader characterization tests to load the moved all-outcome repair configs from the shared path.
- Updated `configs/thesis/_shared/README.md` to list the selected all-outcome stack.

### Commands Run

| Command | Result |
| --- | --- |
| `Get-ChildItem -Path configs/thesis/ablations -Filter 'main_league_champion_hardneg_selected_alloutcome*.yaml'` | Audited the remaining selected all-outcome public config surface before the move. |
| `rg -n "extends:.*main_league_champion_hardneg_selected_alloutcome|main_league_champion_hardneg_selected_alloutcome_(b2repair_b4guard|learnedfloor_b4b2guard|learnedpush_b4b2guard|focusoldhn_b4b2guard|focusoldhn_strong_b4b2guard|focusoldhn_b2retention_b4b2guard|winnerrepair_b4b2guard|swingrepair_b4b2guard|disjointrepair_b4b2guard)_probe" configs python docs README.md` | Audited dependencies and references; current configs and tests now point at `_shared/hardneg_selected_alloutcome/`, while historical logs and archive reports remain as provenance. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_champion_hardneg_selected_alloutcome_b2repair_b4guard_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_champion_hardneg_selected_alloutcome_learnedfloor_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_champion_hardneg_selected_alloutcome_learnedpush_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_champion_hardneg_selected_alloutcome_swingrepair_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_champion_hardneg_selected_alloutcome_disjointrepair_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_champion_hardneg_selected_alloutcome_focusoldhn_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_champion_hardneg_selected_alloutcome_focusoldhn_strong_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_champion_hardneg_selected_alloutcome_focusoldhn_b2retention_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_champion_hardneg_selected_alloutcome_extensionrepair_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_champion_hardneg_selected_alloutcome_winnerrepair_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_champion_hardneg_stratifiedwinnerrepair_probe python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed: 12 passed. |
| `uv run --extra dev --extra sim python -c "<load configs/thesis/_shared/hardneg_selected/*.yaml, configs/thesis/_shared/hardneg_selected_alloutcome/*.yaml, and all remaining configs/thesis/ablations/main_league_champion_hardneg_selected*.yaml>"` | Passed: loaded 36 selected hardneg configs. |
| `uv run --extra dev python -m ruff check python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed. |
| `uv run --extra dev python -m ruff format --check python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed: 2 files already formatted. |

### Tests Added

- None. Existing config-loader tests now characterize the internal shared selected all-outcome path.

### Files Moved or Deleted

- Moved 10 selected all-outcome repair configs from `configs/thesis/ablations/` to `configs/thesis/_shared/hardneg_selected_alloutcome/`: B2 repair, learned floor, learned push, focus-old-hard-negative, focus-old-hard-negative strong, B2-retention focus-old-hard-negative, extension repair, swing repair, disjoint repair, and winner repair.
- Added `configs/thesis/_shared/hardneg_selected_alloutcome/README.md`.

### Behavior Changes

- None intended. Effective config composition is preserved through updated `extends:` paths.
- No simulator contract, legal-action semantics, reward values, replay dataset paths, seed-set paths, training runtime, evaluation flow, checkpoint compatibility, artifact format, metric aggregation, or league promotion behavior changed.

### Remaining Risks

- Historical rebuild/refactor logs and archived reports still mention the old selected all-outcome config paths as provenance.
- The stratified all-outcome and overlap repair branch still lives under `configs/thesis/ablations/` and should move as the next dependency-aware cluster.

### Next Action

- Move the stratified selected all-outcome branch rooted at `main_league_champion_hardneg_selected_alloutcome_stratifiedwinnerrepair_b4b2guard_probe.yaml`, updating overlap descendants to extend an internal shared selected-stratified stack.

## 2026-05-29 - Internal Hard-Negative Selected Stratified Stack

### Changes

- Moved the selected-checkpoint stratified all-outcome repair branch out of `configs/thesis/ablations/` and into `configs/thesis/_shared/hardneg_selected_stratified/`.
- Added `configs/thesis/_shared/hardneg_selected_stratified/README.md`.
- Updated the moved stratified root to extend the internal selected all-outcome winner-repair base.
- Updated config-loader characterization tests to load the moved stratified and overlap repair configs from the shared path.
- Updated `configs/thesis/_shared/README.md` to list the selected stratified stack.

### Commands Run

| Command | Result |
| --- | --- |
| `Get-ChildItem -Path configs/thesis/ablations -Filter 'main_league_champion_hardneg_selected_alloutcome_stratified*.yaml'` | Audited the remaining stratified selected all-outcome public config surface before the move. |
| `rg -n "extends:.*main_league_champion_hardneg_selected_alloutcome_stratified|main_league_champion_hardneg_selected_alloutcome_stratified" configs python docs README.md` | Audited dependencies and references; current configs and tests now point at `_shared/hardneg_selected_stratified/`, while historical logs and archive reports remain as provenance. |
| `Get-ChildItem -Path configs/thesis/ablations -Filter 'main_league_champion_hardneg_selected_alloutcome*.yaml'` | Passed: no selected all-outcome configs remain in public ablations. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_champion_hardneg_stratifiedwinnerrepair_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_champion_hardneg_overlap_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_b1_loss_topaction_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_b1_hardneg_loss_topaction_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_b1_hardneg_preserved_winner_focus_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_b1_hardneg_preserved_winner_b1b3repair_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_grouped_b1b3_hardneg_repair_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_grouped_fixedwin_repair_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_grouped_b2split_fixedwin_repair_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_grouped_b2loss_fixedwin_repair_probe python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed: 11 passed. |
| `uv run --extra dev --extra sim python -c "<load configs/thesis/_shared/hardneg_selected*.yaml plus all remaining configs/thesis/ablations/main_league_champion_hardneg_selected*.yaml>"` | Passed: loaded 36 selected hardneg configs. |
| `uv run --extra dev python -m ruff check python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed. |
| `uv run --extra dev python -m ruff format --check python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed: 2 files already formatted. |

### Tests Added

- None. Existing config-loader tests now characterize the internal shared selected stratified path.

### Files Moved or Deleted

- Moved 10 selected stratified all-outcome repair configs from `configs/thesis/ablations/` to `configs/thesis/_shared/hardneg_selected_stratified/`.
- Added `configs/thesis/_shared/hardneg_selected_stratified/README.md`.

### Behavior Changes

- None intended. Effective config composition is preserved through updated `extends:` paths.
- No simulator contract, legal-action semantics, reward values, replay dataset paths, seed-set paths, training runtime, evaluation flow, checkpoint compatibility, artifact format, metric aggregation, or league promotion behavior changed.

### Remaining Risks

- Historical rebuild/refactor logs and archived reports still mention the old selected stratified config paths as provenance.
- Other selected hard-negative investigation branches remain in public ablations, especially paired flip-BC and outcome-contrastive probes.

### Next Action

- Move the next selected hard-negative branch under `configs/thesis/ablations/`, likely the paired flip-BC / paired swing branch that depends on the shared selected conservative base.

## 2026-05-29 - Internal Hard-Negative Selected Paired Stack

### Changes

- Moved the selected-checkpoint paired replay and low-pressure repair branch out of `configs/thesis/ablations/` and into `configs/thesis/_shared/hardneg_selected_paired/`.
- Added `configs/thesis/_shared/hardneg_selected_paired/README.md`.
- Updated moved configs that directly extend the selected conservative base to use the internal shared path.
- Kept the outcome-contrastive branch public for the next slice, but updated its root to extend the moved paired focus-old-hard-negative config.
- Updated config-loader characterization tests to load the moved paired configs from the shared path.
- Updated `configs/thesis/_shared/README.md` to list the selected paired stack.

### Commands Run

| Command | Result |
| --- | --- |
| `Get-ChildItem -Path configs/thesis/ablations -Filter 'main_league_champion_hardneg_selected*.yaml'` | Audited the remaining selected hard-negative public config surface before the move. |
| `rg -n "extends:.*main_league_champion_hardneg_selected_(paired|grouped128|lowpressure)|main_league_champion_hardneg_selected_(paired_flipbc_conservative|paired_flipbc_focusoldhn_conservative|paired_swing_contrastive|grouped128_paired_flipbc_focusoldhn|lowpressure_pairloss)_probe" configs python docs README.md` | Audited dependencies and references; current configs and tests now point at `_shared/hardneg_selected_paired/`, while historical rebuild logs remain as provenance. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_lowpressure_pairloss_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_paired_swing_contrastive_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_paired_flipbc_conservative_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_paired_flipbc_focusoldhn_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_grouped128_paired_flipbc_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_outcome_contrastive_focusoldhn_probe python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed: 7 passed. |
| `uv run --extra dev --extra sim python -c "<load configs/thesis/_shared/hardneg_selected*.yaml plus all remaining configs/thesis/ablations/main_league_champion_hardneg_selected*.yaml>"` | Passed: loaded 36 selected hardneg configs. |
| `uv run --extra dev python -m ruff check python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed. |
| `uv run --extra dev python -m ruff format --check python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed: 2 files already formatted. |

### Tests Added

- None. Existing config-loader tests now characterize the internal shared selected paired path.

### Files Moved or Deleted

- Moved 5 selected paired/low-pressure repair configs from `configs/thesis/ablations/` to `configs/thesis/_shared/hardneg_selected_paired/`.
- Added `configs/thesis/_shared/hardneg_selected_paired/README.md`.

### Behavior Changes

- None intended. Effective config composition is preserved through updated `extends:` paths.
- No simulator contract, legal-action semantics, reward values, replay dataset paths, paired-swing settings, seed-set paths, training runtime, evaluation flow, checkpoint compatibility, artifact format, metric aggregation, or league promotion behavior changed.

### Remaining Risks

- Historical rebuild logs still mention the old selected paired config paths as provenance.
- The remaining selected hard-negative public surface is now the outcome-contrastive branch under `configs/thesis/ablations/`.

### Next Action

- Move the remaining selected outcome-contrastive branch into an internal shared stack, updating any interpolation continuation configs that extend it.

## 2026-05-29 - Internal Hard-Negative Selected Outcome-Contrastive Stack

### Changes

- Moved the remaining selected-checkpoint outcome-contrastive branch out of `configs/thesis/ablations/` and into `configs/thesis/_shared/hardneg_selected_outcome_contrastive/`.
- Added `configs/thesis/_shared/hardneg_selected_outcome_contrastive/README.md`.
- Updated the moved outcome-contrastive root to extend the internal selected paired stack.
- Updated the public interpolation continuation root to extend the moved shared `rawext256_b2_oldhn` outcome-contrastive leaf.
- Updated config-loader characterization tests to load the moved outcome-contrastive configs from the shared path.
- Updated `configs/thesis/_shared/README.md` to list the selected outcome-contrastive stack.

### Commands Run

| Command | Result |
| --- | --- |
| `Get-ChildItem -Path configs/thesis/ablations -Filter 'main_league_champion_hardneg_selected*.yaml'` | Audited the remaining selected hard-negative public config surface before and after the move. |
| `rg -n "extends:.*main_league_champion_hardneg_selected_outcome_contrastive|main_league_champion_hardneg_selected_outcome_contrastive" configs python docs README.md` | Audited dependencies and references; current configs and tests now point at `_shared/hardneg_selected_outcome_contrastive/`, while historical rebuild logs remain as provenance. |
| `Get-ChildItem -Path configs/thesis/ablations -Filter 'main_league_champion_hardneg_selected*.yaml'` | Passed after the move: no selected hard-negative configs remain in public ablations. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_outcome_contrastive_focusoldhn_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_outcome_contrastive_full_focusoldhn_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_outcome_contrastive_edgehn_focus_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_outcome_contrastive_edgehn_b1b2focus_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_outcome_contrastive_extpreserve_a0375_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_outcome_contrastive_rawext256_allpreserve_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_outcome_contrastive_rawext256_b2_policy1_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_outcome_contrastive_rawext256_b2_oldhn_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_interp_a050_continue_probe python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed: 10 passed. |
| `uv run --extra dev --extra sim python -c "<load configs/thesis/_shared/hardneg_selected*.yaml plus configs/thesis/ablations/main_league_champion_hardneg_interp_a050*.yaml>"` | Passed: loaded 42 selected/interp-a050 configs. |
| `uv run --extra dev python -m ruff check python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed. |
| `uv run --extra dev python -m ruff format --check python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed: 2 files already formatted. |

### Tests Added

- None. Existing config-loader tests now characterize the internal shared selected outcome-contrastive path.

### Files Moved or Deleted

- Moved 8 selected outcome-contrastive configs from `configs/thesis/ablations/` to `configs/thesis/_shared/hardneg_selected_outcome_contrastive/`.
- Added `configs/thesis/_shared/hardneg_selected_outcome_contrastive/README.md`.

### Behavior Changes

- None intended. Effective config composition is preserved through updated `extends:` paths.
- No simulator contract, legal-action semantics, reward values, replay dataset paths, paired-swing settings, seed-set paths, training runtime, evaluation flow, checkpoint compatibility, artifact format, metric aggregation, or league promotion behavior changed.

### Remaining Risks

- Historical rebuild logs still mention the old selected outcome-contrastive config paths as provenance.
- Other non-selected hard-negative and interpolation continuation probes still live under `configs/thesis/ablations/`.

### Next Action

- Continue shrinking `configs/thesis/ablations/` by moving the `main_league_champion_hardneg_interp_a050*.yaml` interpolation continuation branch into `_shared`, now that its selected outcome-contrastive base is internal.

## 2026-05-29 - Internal Hard-Negative Interpolation a050 Stack

### Changes

- Moved the `main_league_champion_hardneg_interp_a050*.yaml` interpolation continuation branch out of `configs/thesis/ablations/` and into `configs/thesis/_shared/hardneg_interp_a050/`.
- Added `configs/thesis/_shared/hardneg_interp_a050/README.md`.
- Updated the moved root to extend the internal selected outcome-contrastive stack.
- Updated direct public a050/a075 follow-up configs to extend the moved shared leaves.
- Updated config-loader characterization tests to load the moved interp-a050 configs from the shared path.
- Updated `configs/thesis/_shared/README.md` to list the interp-a050 stack.

### Commands Run

| Command | Result |
| --- | --- |
| `Get-ChildItem -Path configs/thesis/ablations -Filter 'main_league_champion_hardneg_interp_a050*.yaml'` | Audited the interp-a050 public config surface before and after the move. |
| `rg -n "extends:.*main_league_champion_hardneg_interp_a050|main_league_champion_hardneg_interp_a050" configs python docs README.md` | Audited dependencies and references; current configs and tests now point at `_shared/hardneg_interp_a050/`, while historical rebuild/refactor logs remain as provenance. |
| `Get-ChildItem -Path configs/thesis/ablations -Filter 'main_league_champion_hardneg_interp_a050*.yaml'` | Passed after the move: no interp-a050 configs remain in public ablations. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_interp_a050_continue_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_interp_a050_u1_nowarm_b2guard_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_interp_a050_u1_nowarm_balanced_b2guard_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_interp_a050_p1p2_a025_b2exact_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_interp_a050_p1p2_a025_b2exact_learnedp16_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_interp_a050_p1p2_a025_b2pair70_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_a075_nonconflict_probe python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed: 8 passed. |
| `uv run --extra dev --extra sim python -c "<load configs/thesis/_shared/hardneg_interp_a050/*.yaml plus direct a050/a075 public dependents>"` | Passed: loaded 10 interp-a050 configs. |
| `uv run --extra dev python -m ruff check python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed. |
| `uv run --extra dev python -m ruff format --check python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed: 2 files already formatted. |

### Tests Added

- None. Existing config-loader tests now characterize the internal shared interp-a050 path.

### Files Moved or Deleted

- Moved 6 interp-a050 continuation configs from `configs/thesis/ablations/` to `configs/thesis/_shared/hardneg_interp_a050/`.
- Added `configs/thesis/_shared/hardneg_interp_a050/README.md`.

### Behavior Changes

- None intended. Effective config composition is preserved through updated `extends:` paths.
- No simulator contract, legal-action semantics, reward values, replay dataset paths, paired-swing settings, seed-set paths, training runtime, evaluation flow, checkpoint compatibility, artifact format, metric aggregation, or league promotion behavior changed.

### Remaining Risks

- Historical rebuild/refactor logs still mention the old interp-a050 config paths as provenance.
- Later a050/a075 hard-negative follow-up probes still live under `configs/thesis/ablations/`.

### Next Action

- Continue shrinking `configs/thesis/ablations/` by moving the direct a050/a075 follow-up branch rooted at `main_league_champion_hardneg_a050balanced_live_rowgate_probe.yaml` and `main_league_champion_hardneg_interp_a075_nonconflict_continue_probe.yaml`.

## 2026-05-29 - Internal Hard-Negative a050/a075 Follow-Up Stack

### Changes

- Moved the direct a050/a075 hard-negative follow-up layer out of `configs/thesis/ablations/` and into `configs/thesis/_shared/hardneg_a050_a075_followup/`.
- Added `configs/thesis/_shared/hardneg_a050_a075_followup/README.md`.
- Updated moved configs that directly extend the interp-a050 stack to use internal shared paths.
- Updated public p2-live and a075-context roots to extend the moved shared follow-up configs.
- Updated config-loader characterization tests to load the moved a050/a075 follow-up configs from the shared path.
- Updated `configs/thesis/_shared/README.md` to list the a050/a075 follow-up stack.

### Commands Run

| Command | Result |
| --- | --- |
| `Get-ChildItem -Path configs/thesis/ablations -Filter 'main_league_champion_hardneg_a0*.yaml'` | Audited the a050/a075 public config surface before the move. |
| `rg -n "extends:.*(main_league_champion_hardneg_a050balanced|main_league_champion_hardneg_a075_preference|main_league_champion_hardneg_interp_a075)|main_league_champion_hardneg_(a050balanced|a075_preference|interp_a075)" configs python docs README.md` | Audited dependencies and references; current configs and tests now point at `_shared/hardneg_a050_a075_followup/`, while historical rebuild/refactor logs remain as provenance. |
| `Get-ChildItem -Path configs/thesis/ablations -Filter 'main_league_champion_hardneg_interp_a075*.yaml'` | Passed after the move: no interp-a075 configs remain in public ablations. |
| `Get-ChildItem -Path configs/thesis/ablations -Filter 'main_league_champion_hardneg_a075_preference*.yaml'` | Passed after the move: no a075 preference follow-up configs remain in public ablations. |
| `Get-ChildItem -Path configs/thesis/ablations -Filter 'main_league_champion_hardneg_a050balanced*.yaml'` | Passed after the move: no a050balanced configs remain in public ablations. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_a075_nonconflict_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_a075_broad_conflictfilter_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_a075_episodepref_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_a075_preference_balanced_micro_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_a075_preference_groupbalanced_micro_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_a050balanced_live_rowgate_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_a075_context_episodepref_probe python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed: 8 passed. |
| `uv run --extra dev --extra sim python -c "<load configs/thesis/_shared/hardneg_a050_a075_followup/*.yaml plus representative p2/context public dependents>"` | Passed: loaded 11 a050-a075 follow-up configs. |
| `uv run --extra dev python -m ruff check python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed. |
| `uv run --extra dev python -m ruff format --check python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed: 2 files already formatted. |

### Tests Added

- None. Existing config-loader tests now characterize the internal shared a050/a075 follow-up path.

### Files Moved or Deleted

- Moved 7 a050/a075 follow-up configs from `configs/thesis/ablations/` to `configs/thesis/_shared/hardneg_a050_a075_followup/`.
- Added `configs/thesis/_shared/hardneg_a050_a075_followup/README.md`.

### Behavior Changes

- None intended. Effective config composition is preserved through updated `extends:` paths.
- No simulator contract, legal-action semantics, reward values, replay dataset paths, paired-swing/preference settings, seed-set paths, training runtime, evaluation flow, checkpoint compatibility, artifact format, metric aggregation, or league promotion behavior changed.

### Remaining Risks

- Historical rebuild/refactor logs still mention the old a050/a075 follow-up config paths as provenance.
- Later a050p2 and a075 context hard-negative probe branches still live under `configs/thesis/ablations/`.

### Next Action

- Continue shrinking `configs/thesis/ablations/` by moving either the a050p2 live branch or the a075 context branch into internal shared config stacks.

## 2026-05-29 - Internal Hard-Negative a050p2 Live Stack

### Changes

- Moved the a050p2 live hard-negative probe branch out of `configs/thesis/ablations/` and into `configs/thesis/_shared/hardneg_a050p2_live/`.
- Added `configs/thesis/_shared/hardneg_a050p2_live/README.md`.
- Updated the moved a050p2 root to extend the internal a050/a075 follow-up stack.
- Updated config-loader characterization tests to load the moved a050p2 live configs from the shared path.
- Updated `configs/thesis/_shared/README.md` to list the a050p2 live stack.

### Commands Run

| Command | Result |
| --- | --- |
| `rg -n "configs/thesis/ablations/main_league_champion_hardneg_a050p2|extends: main_league_champion_hardneg_a050p2|main_league_champion_hardneg_a050p2" configs python README.md docs` | Audited dependencies and references; current configs and tests now point at `_shared/hardneg_a050p2_live/`, while historical rebuild logs remain as provenance. |
| `Get-ChildItem -Path configs/thesis/_shared/hardneg_a050p2_live` | Passed after the move: the shared stack contains the four a050p2 live configs plus its README. |
| `Get-ChildItem -Path configs/thesis/ablations -Filter 'main_league_champion_hardneg_a050p2*.yaml'` | Passed after the move: no a050p2 live configs remain in public ablations. |
| `Get-Content -Path configs/thesis/_shared/hardneg_a050p2_live/main_league_champion_hardneg_a050p2_live_learnedpush_rowgate_probe.yaml -TotalCount 8` | Confirmed the moved root now extends `../hardneg_a050_a075_followup/main_league_champion_hardneg_a050balanced_live_rowgate_probe.yaml`. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_a050p2_live_learnedpush_rowgate_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_a050p2_live_rowdeficit_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_a050p2_live_unlocked_rowdeficit_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_a050p2_live_unlocked_learned_recovery_probe python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed: 5 passed. |
| `uv run --extra dev --extra sim python -c "<load configs/thesis/_shared/hardneg_a050p2_live/*.yaml>"` | Passed: loaded 4 a050p2-live configs. |
| `uv run --extra dev python -m ruff check python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed. |
| `uv run --extra dev python -m ruff format --check python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed: 2 files already formatted. |

### Tests Added

- None. Existing config-loader tests now characterize the internal shared a050p2 live path.

### Files Moved or Deleted

- Moved 4 a050p2 live configs from `configs/thesis/ablations/` to `configs/thesis/_shared/hardneg_a050p2_live/`.
- Added `configs/thesis/_shared/hardneg_a050p2_live/README.md`.

### Behavior Changes

- None intended. Effective config composition is preserved through updated `extends:` paths.
- No simulator contract, legal-action semantics, reward values, replay dataset paths, paired-swing/preference settings, seed-set paths, training runtime, evaluation flow, checkpoint compatibility, artifact format, metric aggregation, or league promotion behavior changed.

### Remaining Risks

- Historical rebuild logs still mention the old a050p2 live config paths as provenance.
- The later a075 context hard-negative probe branch still lives under `configs/thesis/ablations/`.

### Next Action

- Continue shrinking `configs/thesis/ablations/` by moving the a075 context branch into an internal shared config stack.

## 2026-05-29 - Internal Hard-Negative a075 Context Stack

### Changes

- Moved the a075 opponent-context hard-negative probe branch out of `configs/thesis/ablations/` and into `configs/thesis/_shared/hardneg_a075_context/`.
- Added `configs/thesis/_shared/hardneg_a075_context/README.md`.
- Updated the moved a075 context root to extend the internal a050/a075 follow-up stack.
- Updated the public a050 context preference root to extend the moved shared a075 context leaf.
- Added a small config-loader test helper for the moved a075 context stack and retargeted the a075 context characterization tests.
- Updated `configs/thesis/_shared/README.md` to list the a075 context stack.

### Commands Run

| Command | Result |
| --- | --- |
| `Get-ChildItem -Path configs/thesis/ablations -Filter '*a075*context*.yaml'` | Audited the a075 context public config surface before the move. |
| `rg -n -F "configs/thesis/ablations/main_league_champion_hardneg_a075_context" configs python README.md docs` | Audited stale public paths after the move; only historical rebuild-log provenance still references the old public paths. |
| `Get-ChildItem -Path configs/thesis/_shared/hardneg_a075_context -Filter '*.yaml'` | Passed after the move: the shared stack contains 28 a075 context configs. |
| `Get-ChildItem -Path configs/thesis/ablations -Filter 'main_league_champion_hardneg_a075_context*.yaml'` | Passed after the move: no a075 context configs remain in public ablations. |
| `Get-Content -Path configs/thesis/_shared/hardneg_a075_context/main_league_champion_hardneg_a075_context_probe.yaml -TotalCount 6` | Confirmed the moved root now extends `../hardneg_a050_a075_followup/main_league_champion_hardneg_interp_a075_nonconflict_continue_probe.yaml`. |
| `Get-Content -Path configs/thesis/ablations/main_league_champion_hardneg_a050_context_preference_width128_probe.yaml -TotalCount 8` | Confirmed the public a050 context root now extends `../_shared/hardneg_a075_context/main_league_champion_hardneg_a075_context_preference_groupbalanced_micro_probe.yaml`. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_config_loader.py -k "a075_context or a050_width128"` | Passed: 24 passed, 180 deselected. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed: 1 passed. |
| `uv run --extra dev --extra sim python -c "<load configs/thesis/_shared/hardneg_a075_context/*.yaml>"` | Passed: loaded 28 a075-context configs. |
| `uv run --extra dev python -m ruff check python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed. |
| `uv run --extra dev python -m ruff format --check python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed: 2 files already formatted. |

### Tests Added

- None. Existing config-loader tests now characterize the internal shared a075 context path.

### Files Moved or Deleted

- Moved 28 a075 context configs from `configs/thesis/ablations/` to `configs/thesis/_shared/hardneg_a075_context/`.
- Added `configs/thesis/_shared/hardneg_a075_context/README.md`.

### Behavior Changes

- None intended. Effective config composition is preserved through updated `extends:` paths.
- No simulator contract, legal-action semantics, observation/action interpretation, reward values, replay dataset paths, paired-swing/preference settings, seed-set paths, training runtime, evaluation flow, checkpoint compatibility, artifact format, metric aggregation, or league promotion behavior changed.

### Remaining Risks

- Historical rebuild logs still mention the old a075 context config paths as provenance.
- The a050 context preference width128 branch still lives under `configs/thesis/ablations/`, now extending this internal shared stack.

### Next Action

- Continue shrinking `configs/thesis/ablations/` by moving the a050 context preference width128 branch into an internal shared config stack.

## 2026-05-29 - Internal Hard-Negative a050 Context Width128 Stack

### Changes

- Moved the a050 opponent-context width128 preference branch out of `configs/thesis/ablations/` and into `configs/thesis/_shared/hardneg_a050_context_width128/`.
- Added `configs/thesis/_shared/hardneg_a050_context_width128/README.md`.
- Updated the moved a050 width128 root to extend the internal a075 context stack.
- Added a small config-loader test helper for the moved a050 width128 stack and retargeted the existing a050 width128 characterization tests.
- Added direct characterization for the `rich_exactaliases` leaf, which was previously covered only by historical commands and broad config loading.
- Updated `configs/thesis/_shared/README.md` to list the a050 context width128 stack.

### Commands Run

| Command | Result |
| --- | --- |
| `Get-ChildItem -Path configs/thesis/ablations -Filter 'main_league_champion_hardneg_a050_context_preference_width128*.yaml'` | Audited the a050 context width128 public config surface before and after the move. |
| `rg -n -F "configs/thesis/ablations/main_league_champion_hardneg_a050_context_preference_width128" configs python README.md docs` | Audited stale public paths after the move; only historical rebuild/refactor-log provenance still references the old public paths. |
| `Get-ChildItem -Path configs/thesis/_shared/hardneg_a050_context_width128 -Filter '*.yaml'` | Passed after the move: the shared stack contains 4 a050 context width128 configs. |
| `Get-Content -Path configs/thesis/_shared/hardneg_a050_context_width128/main_league_champion_hardneg_a050_context_preference_width128_rich_exactaliases_probe.yaml -TotalCount 40` | Inspected the exactaliases leaf before adding direct characterization. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_config_loader.py -k "a050_width128"` | Passed: 4 passed, 201 deselected. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed: 1 passed. |
| `uv run --extra dev --extra sim python -c "<load configs/thesis/_shared/hardneg_a050_context_width128/*.yaml>"` | Passed: loaded 4 a050-context-width128 configs. |
| `uv run --extra dev python -m ruff check python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed. |
| `uv run --extra dev python -m ruff format --check python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed: 2 files already formatted. |

### Tests Added

- Added `test_load_stack_config_supports_main_league_a050_width128_rich_exactaliases_probe` to pin the exact learned-alias context rows on the moved leaf.

### Files Moved or Deleted

- Moved 4 a050 context width128 configs from `configs/thesis/ablations/` to `configs/thesis/_shared/hardneg_a050_context_width128/`.
- Added `configs/thesis/_shared/hardneg_a050_context_width128/README.md`.

### Behavior Changes

- None intended. Effective config composition is preserved through updated `extends:` paths.
- No simulator contract, legal-action semantics, observation/action interpretation, reward values, replay dataset paths, paired-swing/preference settings, seed-set paths, training runtime, evaluation flow, checkpoint compatibility, artifact format, metric aggregation, or league promotion behavior changed.

### Remaining Risks

- Historical rebuild/refactor logs still mention the old a050 context width128 config paths as provenance.
- Other noncanonical historical hard-negative probes may still live under `configs/thesis/ablations/`; continue auditing by remaining public-surface families rather than individual files.

### Next Action

- Audit the now-smaller `configs/thesis/ablations/` surface and move the next coherent historical hard-negative probe family into `_shared` or archive, leaving only public thesis ablation launch targets.

## 2026-05-29 - Internal Hard-Negative Multiobjective Guard Root

### Changes

- Moved `main_league_champion_hardneg_multiobjective_guard_probe.yaml` out of public ablations and into `configs/thesis/_shared/hardneg_core/`.
- Updated the moved guard root to extend the local hard-negative core stable-long config.
- Updated the internal retention probe that depended on the guard root so it no longer reaches back into `configs/thesis/ablations/`.
- Updated the hard-negative core README to explain why the multiobjective guard root lives in the internal core stack.
- Retargeted the config-loader characterization test to the moved shared path.

### Commands Run

| Command | Result |
| --- | --- |
| `Get-ChildItem -Path configs/thesis/ablations -Filter '*.yaml'` | Audited the remaining public ablation surface before and after the move. |
| `rg -n "main_league_champion_hardneg_multiobjective_guard_probe|multiobjective_guard" configs python README.md docs` | Audited dependencies and references; current configs/tests now point at `_shared/hardneg_core/`, while historical rebuild/refactor logs remain as provenance. |
| `Get-Content -Path configs/thesis/_shared/hardneg_core/main_league_champion_hardneg_multiobjective_guard_probe.yaml -TotalCount 8` | Confirmed the moved root now extends `main_league_champion_hardneg_stable_long_probe.yaml`. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_champion_hardneg_multiobjective_guard_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_champion_hardneg_multiobjective_retention_probe python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed: 3 passed. |
| `uv run --extra dev --extra sim python -c "<load moved guard root plus dependent retention config>"` | Passed: loaded 2 multiobjective guard configs. |
| `uv run --extra dev python -m ruff check python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed. |
| `uv run --extra dev python -m ruff format python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_public_config_surface_docs.py` | Reformatted `test_config_loader.py`. |
| `uv run --extra dev python -m ruff format --check python/weiss_rl/tests/test_config_loader.py python/weiss_rl/tests/test_public_config_surface_docs.py` | Passed: 2 files already formatted. |

### Tests Added

- None. Existing config-loader tests now characterize the moved internal shared path.

### Files Moved or Deleted

- Moved 1 multiobjective guard config from `configs/thesis/ablations/` to `configs/thesis/_shared/hardneg_core/`.

### Behavior Changes

- None intended. Effective config composition is preserved through updated `extends:` paths.
- No simulator contract, legal-action semantics, observation/action interpretation, reward values, replay dataset paths, seed-set paths, training runtime, guarded-bootstrap logic, evaluation flow, checkpoint compatibility, artifact format, metric aggregation, or league promotion behavior changed.

### Remaining Risks

- Historical rebuild/refactor logs still mention the old public multiobjective guard path as provenance.
- Public ablations are now limited to baseline/model ablations plus final-eval variants; the next cleanup should move from config gardening into production Python module refactors.

### Next Action

- Start a production-code refactor slice, preferably in the public package CLI/workflow path, with characterization tests around existing commands.

## 2026-05-29 - Package CLI Figures Command Builder Extraction

### Changes

- Extracted figure workflow subprocess construction from `dispatch_figures` into `build_figures_command`.
- Kept `dispatch_figures` focused on resolving parsed arguments, naming the plan, and dispatching through `run_or_write_plan`.
- Added a CLI dry-run characterization test that pins the exact generated `python/scripts/make_figures.py` command, including `--fig-id` and repeated `--format` forwarding.

### Commands Run

| Command | Result |
| --- | --- |
| `Get-Content -Path python/weiss_rl/cli.py -TotalCount 260` | Confirmed the public `weiss_rl.cli` module is already a thin parser/dispatcher wrapper. |
| `Get-Content -Path python/weiss_rl/workflows/command_builders.py` | Inspected deterministic workflow command builders before adding the figures builder. |
| `Get-Content -Path python/weiss_rl/workflows/dispatch_evaluation.py` | Inspected evaluation workflow dispatch before extracting inline figures command construction. |
| `rg -n "package_cli|verify_repo|public_workflows|Package CLI" python/weiss_rl/tests/test_script_entrypoint_smokes.py` | Located the existing verify-repo package CLI smoke test after an initial stale pytest target. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_cli_workflow.py::test_package_cli_figures_dry_run_forwards_figure_options python/weiss_rl/tests/test_script_entrypoint_smokes.py::test_verify_repo_entrypoint_runs_release_verification_steps` | Passed: 2 passed. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_cli_workflow.py` | Passed: 20 passed. |
| `uv run --extra dev python -m ruff check python/weiss_rl/workflows/command_builders.py python/weiss_rl/workflows/dispatch_evaluation.py python/weiss_rl/tests/test_cli_workflow.py` | Passed. |
| `uv run --extra dev python -m ruff format python/weiss_rl/tests/test_cli_workflow.py` | Reformatted the new test after the first format check reported wrapping drift. |
| `uv run --extra dev python -m ruff format --check python/weiss_rl/workflows/command_builders.py python/weiss_rl/workflows/dispatch_evaluation.py python/weiss_rl/tests/test_cli_workflow.py` | Passed: 3 files already formatted. |

### Tests Added

- Added `test_package_cli_figures_dry_run_forwards_figure_options`.

### Files Moved or Deleted

- None.

### Behavior Changes

- None intended. The figure workflow still emits the same subprocess command and dry-run plan shape.
- No training, evaluation, simulator, config, checkpoint, artifact, figure-rendering, or CLI argument semantics changed.

### Remaining Risks

- Other package CLI dispatch modules still contain some argument normalization and command-building decisions in the same functions; continue extracting only where tests can pin command output.

### Next Action

- Continue production-code refactoring in the package CLI/workflow layer, likely by moving another inline workflow command into `command_builders` or by simplifying dispatch tables while preserving dry-run command output.

## 2026-05-29 - Package CLI Guard-Run Defaults Cleanup

### Changes

- Moved the default guard-run required anchor set from `dispatch_guard_run` into `command_builders.DEFAULT_GUARD_REQUIRED_ANCHORS`.
- Made `build_guard_run_command` resolve default B2/B3/B4 anchors when no explicit anchors are provided.
- Kept `dispatch_guard_run` focused on parsed-argument normalization and workflow-plan dispatch.
- Added CLI dry-run characterization for custom `--required-anchor` values replacing the defaults.

### Commands Run

| Command | Result |
| --- | --- |
| `Get-Content -Path python/weiss_rl/workflows/dispatch_bootstrap.py -TotalCount 280` | Inspected neighboring workflow dispatch style before choosing the guard-run cleanup. |
| `Get-Content -Path python/weiss_rl/workflows/plans.py -TotalCount 220` | Inspected plan writing helpers to avoid folding unrelated plan behavior into this slice. |
| `rg -n "B2 HeuristicPublic|guard-required-anchor|required_anchor|build_guard_run_command|dispatch_guard_run" python/weiss_rl docs README.md` | Audited the guard-run default anchor references before moving the defaults. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_cli_workflow.py::test_package_cli_guard_run_wraps_learning_progress_league_guard python/weiss_rl/tests/test_cli_workflow.py::test_package_cli_guard_run_custom_required_anchors_replace_defaults python/weiss_rl/tests/test_cli_workflow.py::test_package_cli_guard_run_failure_exits_without_traceback` | Passed: 3 passed. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_cli_workflow.py` | Passed: 21 passed. |
| `uv run --extra dev python -m ruff check python/weiss_rl/workflows/command_builders.py python/weiss_rl/workflows/dispatch_evaluation.py python/weiss_rl/tests/test_cli_workflow.py` | Passed. |
| `uv run --extra dev python -m ruff format --check python/weiss_rl/workflows/command_builders.py python/weiss_rl/workflows/dispatch_evaluation.py python/weiss_rl/tests/test_cli_workflow.py` | Passed: 3 files already formatted. |

### Tests Added

- Added `test_package_cli_guard_run_custom_required_anchors_replace_defaults`.

### Files Moved or Deleted

- None.

### Behavior Changes

- None intended. Default guard-run dry-run commands still include B2/B3/B4 required anchors, and explicit `--required-anchor` values still replace those defaults.
- No training, evaluation, simulator, config, checkpoint, artifact, figure-rendering, or CLI argument semantics changed.

### Remaining Risks

- Package CLI dispatch still contains some validation and payload-shaping logic; keep extracting only when command output and failure behavior are covered by focused tests.

### Next Action

- Continue production-code refactoring in the package CLI/workflow layer, likely by simplifying workflow dispatch selection or extracting repeated run-plan payload construction where behavior can be pinned.

## 2026-05-29 - Package CLI Dispatch Table

### Changes

- Replaced the long package CLI workflow `if` chain with an explicit typed command-to-handler table.
- Kept `dispatch_workflow_command` responsible for resolving the repo root and Python executable once, then invoking the selected handler.
- Added a parser/dispatcher consistency test so every parser subcommand must have a registered dispatch handler.

### Commands Run

| Command | Result |
| --- | --- |
| `Get-Content -Path python/weiss_rl/workflows/cli_dispatch.py -TotalCount 180` | Inspected the existing package CLI dispatch if-chain before replacing it. |
| `rg -n "dispatch_workflow_command|Unhandled workflow command|weiss_rl.cli" python/weiss_rl/tests python/scripts/verify_repo.py` | Audited existing package CLI dispatch and smoke-test coverage. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_cli_workflow.py::test_package_cli_parser_commands_are_all_dispatchable python/weiss_rl/tests/test_cli_workflow.py::test_package_cli_train_b1_dry_run_uses_thesis_config python/weiss_rl/tests/test_cli_workflow.py::test_package_cli_smoke_eval_uses_tiny_eval_budget python/weiss_rl/tests/test_cli_workflow.py::test_package_cli_guarded_league_bootstrap_wraps_controller` | Passed: 4 passed. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_cli_workflow.py` | Passed: 22 passed. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_script_entrypoint_smokes.py::test_verify_repo_entrypoint_runs_release_verification_steps` | Passed: 1 passed. |
| `uv run --extra dev python -m mypy python/weiss_rl/cli.py python/weiss_rl/workflows` | Passed: no issues found in 11 source files. |
| `uv run --extra dev python -m ruff check python/weiss_rl/workflows/cli_dispatch.py python/weiss_rl/tests/test_cli_workflow.py` | Passed. |
| `uv run --extra dev python -m ruff format --check python/weiss_rl/workflows/cli_dispatch.py python/weiss_rl/tests/test_cli_workflow.py` | Passed: 2 files already formatted. |

### Tests Added

- Added `test_package_cli_parser_commands_are_all_dispatchable`.

### Files Moved or Deleted

- None.

### Behavior Changes

- None intended. Existing package CLI commands still dispatch to the same workflow handlers, and the unhandled-command assertion is preserved.
- No training, evaluation, simulator, config, checkpoint, artifact, figure-rendering, or CLI argument semantics changed.

### Remaining Risks

- Workflow dispatch handlers still contain some repeated plan-name and payload-shaping patterns; those can be simplified in later production-code slices if tests continue pinning dry-run plans.

### Next Action

- Continue production-code refactoring in the package workflow layer, likely by extracting repeated dry-run plan payload construction or by simplifying training workflow dispatch while preserving command output.

## 2026-05-29 - Package CLI Simple Training Workflow Helper

### Changes

- Extracted the shared `train-b1` / `train-b1-guided-seed` profile, command-building, and dry-run plan flow into `_dispatch_simple_training_workflow`.
- Kept the public `dispatch_train_b1` and `dispatch_train_b1_guided_seed` functions as small named wrappers with explicit workflow names and stack configs.
- Preserved the existing generated commands and dry-run plan payloads.

### Commands Run

| Command | Result |
| --- | --- |
| `Get-Content -Path python/weiss_rl/workflows/dispatch_training.py -TotalCount 240` | Inspected the duplicated simple training dispatch flow before extraction. |
| `rg -n "train_b1|train-b1|guided-seed|train_main" python/weiss_rl/tests/test_cli_workflow.py python/weiss_rl/tests/test_script_entrypoint_smokes.py python/scripts/verify_repo.py` | Located focused B1 and guided-seed workflow coverage. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_cli_workflow.py::test_package_cli_train_b1_dry_run_uses_thesis_config python/weiss_rl/tests/test_cli_workflow.py::test_package_cli_train_b1_gpu_probe_uses_cuda_probe_shape python/weiss_rl/tests/test_cli_workflow.py::test_package_cli_train_b1_league_probe_uses_early_guard_shape python/weiss_rl/tests/test_cli_workflow.py::test_package_cli_train_b1_guided_seed_uses_guided_seed_config` | Passed: 4 passed. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_cli_workflow.py` | Passed: 22 passed. |
| `uv run --extra dev python -m mypy python/weiss_rl/cli.py python/weiss_rl/workflows` | Passed: no issues found in 11 source files. |
| `uv run --extra dev python -m ruff check python/weiss_rl/workflows/dispatch_training.py python/weiss_rl/tests/test_cli_workflow.py` | Passed. |
| `uv run --extra dev python -m ruff format --check python/weiss_rl/workflows/dispatch_training.py python/weiss_rl/tests/test_cli_workflow.py` | Passed: 2 files already formatted. |

### Tests Added

- None. Existing dry-run workflow tests cover both simple training wrappers.

### Files Moved or Deleted

- None.

### Behavior Changes

- None intended. `train-b1` and `train-b1-guided-seed` still emit the same commands and plan payload shapes.
- No training, evaluation, simulator, config, checkpoint, artifact, figure-rendering, or CLI argument semantics changed.

### Remaining Risks

- `dispatch_train_main` and `dispatch_train_main_guided_bootstrap` still contain workflow-specific checkpoint resolution and validation. Keep those explicit unless a later extraction can pin both success and error behavior.

### Next Action

- Continue production-code cleanup around workflow modules, likely by extracting a small typed plan-payload helper or by simplifying `dispatch_train_main_guided_bootstrap` validation while preserving error messages.

## 2026-05-29 - Package CLI Workflow Plan Helper

### Changes

- Added `run_or_write_workflow_plan` to centralize workflow dry-run payload construction.
- Updated training, evaluation, figure, guard, guided-loop, and guarded-league dispatchers to pass the workflow name separately from workflow-specific payload fields.
- Kept `run_or_write_plan` as the lower-level command execution and plan-writing primitive.
- Preserved existing dry-run JSON shape: workflow plans still include `workflow`, `command`, `cwd`, and `status` with the same workflow-specific fields.

### Commands Run

| Command | Result |
| --- | --- |
| `rg -n "workflow|cli_dispatch|dispatch_training|run_or_write_plan|command_builders" C:/Users/Bruger/.codex/memories/MEMORY.md` | Quick memory check found no workflow-specific prior constraints. |
| `Get-Content -Path python/weiss_rl/workflows/plans.py -TotalCount 220` | Inspected the lower-level plan writer before adding the workflow-specific wrapper. |
| `Get-Content -Path python/weiss_rl/workflows/dispatch_training.py -TotalCount 260` | Inspected training workflow payload repetition before updating dispatchers. |
| `Get-Content -Path python/weiss_rl/workflows/dispatch_evaluation.py -TotalCount 180` | Inspected evaluation, figures, B2-audit, and guard-run payload repetition before updating dispatchers. |
| `Get-Content -Path python/weiss_rl/workflows/dispatch_bootstrap.py -TotalCount 180` | Inspected guided-loop and guarded-league payload repetition before updating dispatchers. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_cli_workflow.py::test_package_cli_train_b1_dry_run_uses_thesis_config python/weiss_rl/tests/test_cli_workflow.py::test_package_cli_train_main_requires_b1_and_uses_main_config python/weiss_rl/tests/test_cli_workflow.py::test_package_cli_smoke_eval_uses_tiny_eval_budget python/weiss_rl/tests/test_cli_workflow.py::test_package_cli_figures_dry_run_forwards_figure_options python/weiss_rl/tests/test_cli_workflow.py::test_package_cli_guided_bootstrap_loop_wraps_segmented_controller python/weiss_rl/tests/test_cli_workflow.py::test_package_cli_guarded_league_bootstrap_wraps_controller` | Passed: 6 passed. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_cli_workflow.py` | Passed: 22 passed. |
| `uv run --extra dev python -m mypy python/weiss_rl/cli.py python/weiss_rl/workflows` | Passed: no issues found in 11 source files. |
| `uv run --extra dev python -m ruff check python/weiss_rl/workflows/plans.py python/weiss_rl/workflows/dispatch_training.py python/weiss_rl/workflows/dispatch_evaluation.py python/weiss_rl/workflows/dispatch_bootstrap.py python/weiss_rl/tests/test_cli_workflow.py` | Passed. |
| `uv run --extra dev python -m ruff format --check python/weiss_rl/workflows/plans.py python/weiss_rl/workflows/dispatch_training.py python/weiss_rl/workflows/dispatch_evaluation.py python/weiss_rl/workflows/dispatch_bootstrap.py python/weiss_rl/tests/test_cli_workflow.py` | Passed: 5 files already formatted. |

### Tests Added

- None. Existing dry-run workflow tests cover the preserved plan JSON shape across dispatch modules.

### Files Moved or Deleted

- None.

### Behavior Changes

- None intended. Workflow dry-run plans preserve the same public JSON fields and generated subprocess commands.
- No training, evaluation, simulator, config, checkpoint, artifact, figure-rendering, or CLI argument semantics changed.

### Remaining Risks

- `dispatch_train_main_guided_bootstrap` still mixes init-source validation with command dispatch; it should stay explicit until error-message tests cover any extraction.

### Next Action

- Add focused error-behavior coverage for `train-main-guided-bootstrap`, then consider extracting its init-source resolution into a small helper.

## 2026-05-29 - Guided Bootstrap Init Source Helper

### Changes

- Added focused CLI error coverage for `train-main-guided-bootstrap` when no init source is provided.
- Strengthened the ambiguous-init-source test to assert the CLI exits without a traceback.
- Extracted `_resolve_guided_bootstrap_init_checkpoint` from `dispatch_train_main_guided_bootstrap`.
- Kept `dispatch_train_main_guided_bootstrap` focused on profile selection, stack selection, command construction, and workflow plan dispatch.

### Commands Run

| Command | Result |
| --- | --- |
| `rg -n "train-main-guided-bootstrap|init-from-checkpoint|init-from-run-dir|init_policy_id" C:/Users/Bruger/.codex/memories/MEMORY.md` | Quick memory check found no guided-bootstrap init-source constraints. |
| `Get-Content -Path python/weiss_rl/workflows/dispatch_training.py -TotalCount 220` | Inspected the existing nested init-source validation and resolution before extraction. |
| `Get-Content -Path python/weiss_rl/tests/test_cli_workflow.py -Skip 300 -First 280` | Inspected existing guided-bootstrap success and ambiguous-source tests before adding missing-source coverage. |
| `rg -n "resolve_snapshot_checkpoint_path|resolve_b1_seed_checkpoint_path|resolve_single_snapshot_checkpoint_path" python/weiss_rl/workflows python/weiss_rl/tests` | Audited snapshot resolution helper usage before extracting the guided-bootstrap init helper. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_cli_workflow.py::test_package_cli_train_main_guided_bootstrap_uses_seed_and_warmstart_without_strict_b1 python/weiss_rl/tests/test_cli_workflow.py::test_package_cli_train_main_guided_bootstrap_selected_resolves_init_policy_id python/weiss_rl/tests/test_cli_workflow.py::test_package_cli_train_main_guided_bootstrap_rejects_ambiguous_init_sources python/weiss_rl/tests/test_cli_workflow.py::test_package_cli_train_main_guided_bootstrap_requires_init_source` | Passed: 4 passed. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_cli_workflow.py` | Passed: 23 passed. |
| `uv run --extra dev python -m mypy python/weiss_rl/cli.py python/weiss_rl/workflows` | Passed: no issues found in 11 source files. |
| `uv run --extra dev python -m ruff check python/weiss_rl/workflows/dispatch_training.py python/weiss_rl/tests/test_cli_workflow.py` | Passed. |
| `uv run --extra dev python -m ruff format --check python/weiss_rl/workflows/dispatch_training.py python/weiss_rl/tests/test_cli_workflow.py` | Passed: 2 files already formatted. |

### Tests Added

- Added `test_package_cli_train_main_guided_bootstrap_requires_init_source`.

### Files Moved or Deleted

- None.

### Behavior Changes

- None intended. `train-main-guided-bootstrap` keeps the same success commands and the same init-source error messages, now covered by focused tests.
- No training, evaluation, simulator, config, checkpoint, artifact, figure-rendering, or CLI argument semantics changed.

### Remaining Risks

- `dispatch_train_main` still has B1-specific checkpoint fallback logic for smoke runs. It should remain explicit until both strict and smoke fallback behavior are pinned tightly enough for extraction.

### Next Action

- Add focused coverage around `train-main` B1 checkpoint resolution and smoke fallback, then consider extracting its init checkpoint resolution helper.

## 2026-05-29 - Train-Main Init Checkpoint Helper

### Changes

- Added focused CLI coverage that `train-main --profile thesis-local` rejects a single unaliased B1 snapshot instead of using the smoke fallback.
- Preserved and reused the existing smoke-profile test that allows a single unaliased B1 snapshot.
- Extracted `_resolve_train_main_init_checkpoint` from `dispatch_train_main`.
- Kept `dispatch_train_main` focused on profile selection, command construction, and workflow plan dispatch.

### Commands Run

| Command | Result |
| --- | --- |
| `rg -n "train-main|b1_noleague_baseline|single_unaliased|resolve_b1_seed_checkpoint|smoke fallback|init_policy_id" C:/Users/Bruger/.codex/memories/MEMORY.md` | Refreshed prior B1 alias context; memory confirmed the canonical `b1_noleague_baseline` alias is important for downstream workflows. |
| `Get-Content -Path python/weiss_rl/workflows/dispatch_training.py -TotalCount 180` | Inspected train-main B1 init resolution before extracting a helper. |
| `Get-Content -Path python/weiss_rl/workflows/snapshots.py -TotalCount 150` | Inspected B1 alias and single-snapshot fallback helpers before reusing them from the extracted helper. |
| `Get-Content -Path python/weiss_rl/tests/test_cli_workflow.py -Skip 180 -First 125` | Inspected existing strict alias, smoke fallback, and seed-run train-main tests before adding strict-profile rejection coverage. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_cli_workflow.py::test_package_cli_train_main_requires_b1_and_uses_main_config python/weiss_rl/tests/test_cli_workflow.py::test_package_cli_train_main_smoke_accepts_single_unaliased_b1_snapshot python/weiss_rl/tests/test_cli_workflow.py::test_package_cli_train_main_strict_profile_rejects_unaliased_b1_snapshot python/weiss_rl/tests/test_cli_workflow.py::test_package_cli_train_main_accepts_guided_seed_run` | Initial run failed because the new strict test omitted `--profile thesis-local`, revealing that default `train-main` is the smoke profile. After correcting the test to use `--profile thesis-local`, passed: 4 passed. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_cli_workflow.py` | Passed: 24 passed. |
| `uv run --extra dev python -m mypy python/weiss_rl/cli.py python/weiss_rl/workflows` | Passed: no issues found in 11 source files. |
| `uv run --extra dev python -m ruff check python/weiss_rl/workflows/dispatch_training.py python/weiss_rl/tests/test_cli_workflow.py` | Passed. |
| `uv run --extra dev python -m ruff format --check python/weiss_rl/workflows/dispatch_training.py python/weiss_rl/tests/test_cli_workflow.py` | Passed: 2 files already formatted. |

### Tests Added

- Added `test_package_cli_train_main_strict_profile_rejects_unaliased_b1_snapshot`.

### Files Moved or Deleted

- None.

### Behavior Changes

- None intended. `train-main` still resolves canonical B1 aliases for normal profiles and still allows the single-snapshot fallback only for the smoke profile with automatic init policy selection.
- No training, evaluation, simulator, config, checkpoint, artifact, figure-rendering, or CLI argument semantics changed.

### Remaining Risks

- Training workflow dispatch is now fairly small, but `snapshots.py` still mixes registry parsing and policy-resolution policy. Future cleanup there should preserve exact error messages.

### Next Action

- Continue production-code cleanup in workflow support modules, likely by adding focused tests around `workflows.snapshots` error behavior before simplifying snapshot registry parsing.

## 2026-05-29 - Workflow Snapshot Registry Helper

### Changes

- Added focused tests for package workflow snapshot resolver failures:
  - missing registry for explicit `--init-from-run-dir` resolution;
  - malformed registry payloads without a `snapshots` list;
  - missing checkpoint files for a matching snapshot;
  - missing requested policy ids;
  - smoke fallback missing registry and multi-snapshot rejection.
- Extracted shared snapshot registry path, registry loading, and policy-id collection helpers in `weiss_rl.workflows.snapshots`.
- Preserved resolver-specific missing-registry messages so `train-main` and `train-main-guided-bootstrap` keep their existing user-facing failure modes.

### Commands Run

| Command | Result |
| --- | --- |
| `Get-Content -Path python/weiss_rl/workflows/snapshots.py` | Inspected duplicated registry parsing in explicit and single-snapshot resolvers. |
| `rg -n "resolve_snapshot_checkpoint_path|resolve_single_snapshot_checkpoint_path|resolve_b1_seed_checkpoint_path|snapshot registry" python/weiss_rl/tests python/weiss_rl/workflows` | Confirmed existing coverage only pinned snapshot helper usage indirectly plus one happy-path test in segmented bootstrap coverage. |
| `Get-Content -Path python/weiss_rl/tests/test_segmented_b1_guided_bootstrap.py` | Inspected the existing snapshot happy-path characterization before adding workflow-focused failure tests. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_workflow_snapshots.py python/weiss_rl/tests/test_cli_workflow.py::test_package_cli_train_main_requires_b1_and_uses_main_config python/weiss_rl/tests/test_cli_workflow.py::test_package_cli_train_main_smoke_accepts_single_unaliased_b1_snapshot python/weiss_rl/tests/test_cli_workflow.py::test_package_cli_train_main_strict_profile_rejects_unaliased_b1_snapshot python/weiss_rl/tests/test_cli_workflow.py::test_package_cli_train_main_guided_bootstrap_selected_resolves_init_policy_id` | Passed: 10 passed. |
| `uv run --extra dev python -m ruff check python/weiss_rl/workflows/snapshots.py python/weiss_rl/tests/test_workflow_snapshots.py python/weiss_rl/tests/test_cli_workflow.py` | Passed. |
| `uv run --extra dev python -m ruff format --check python/weiss_rl/workflows/snapshots.py python/weiss_rl/tests/test_workflow_snapshots.py python/weiss_rl/tests/test_cli_workflow.py` | Initially reported the new test file would be reformatted; after manual line wrapping, passed: 3 files already formatted. |
| `uv run --extra dev python -m mypy python/weiss_rl/cli.py python/weiss_rl/workflows` | Passed: no issues found in 11 source files. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_cli_workflow.py` | Passed: 24 passed. |

### Tests Added

- Added `python/weiss_rl/tests/test_workflow_snapshots.py` with six failure-mode tests around workflow snapshot resolution.

### Files Moved or Deleted

- None.

### Behavior Changes

- None intended. Snapshot checkpoint resolution still uses the same registry locations, policy-id normalization, update/update_count lookup, checkpoint path convention, and resolver-specific error text.
- No training, evaluation, simulator, observation, action, reward, checkpoint, artifact, figure-rendering, or CLI argument semantics changed.

### Remaining Risks

- `resolve_b1_seed_checkpoint_path` still owns B1 policy-id fallback ordering. That is behavior-sensitive and should only be simplified after adding direct tests for fallback ordering and final aggregate error text.

### Next Action

- Pin B1 seed checkpoint fallback ordering and aggregate error behavior, then simplify `resolve_b1_seed_checkpoint_path` without changing canonical B1 alias preference.

## 2026-05-29 - B1 Seed Checkpoint Resolver Policy

### Changes

- Added direct tests for `resolve_b1_seed_checkpoint_path`:
  - `auto` prefers the canonical `b1_noleague_baseline` alias even when it appears after other aliases in the registry;
  - empty init policy uses the same automatic B1 seed policy list and can fall back to the legacy `B1 NoLeague baseline` name;
  - aggregate auto-resolution errors report the exact policy ids tried in order and the final underlying snapshot error;
  - explicit init policy ids are stripped and tried alone rather than expanded into B1 fallbacks.
- Extracted the automatic B1 seed policy-id tuple, policy-id selection helper, and aggregate error formatter in `weiss_rl.workflows.snapshots`.
- Kept `resolve_b1_seed_checkpoint_path` as the public resolver while making the canonical alias policy visible and test-backed.

### Commands Run

| Command | Result |
| --- | --- |
| `rg -n "b1_noleague_baseline|NOLEAGUE_BASELINE_POLICY_ID|resolve_b1_seed_checkpoint|B1 seed" C:/Users/Bruger/.codex/memories/MEMORY.md` | Quick memory check confirmed prior work treats `b1_noleague_baseline` as the canonical B1 alias that downstream workflows depend on. |
| `Get-Content -Path python/weiss_rl/workflows/snapshots.py` | Inspected the inline B1 seed fallback policy before extracting named helpers. |
| `Get-Content -Path python/weiss_rl/tests/test_workflow_snapshots.py` | Inspected the focused snapshot resolver tests before extending them with direct B1 seed resolver coverage. |
| `rg -n "NOLEAGUE_BASELINE_NAME|NOLEAGUE_BASELINE_POLICY_ID|SELECTED_CANDIDATE_POLICY_ID" python/weiss_rl` | Audited constant usage and confirmed the workflow resolver imports the shared baseline constants. |
| `Get-Content -Path python/weiss_rl/experiments/baselines.py -TotalCount 40` | Confirmed canonical B1 alias, legacy B1 display name, and selected-candidate policy id constants. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_workflow_snapshots.py python/weiss_rl/tests/test_cli_workflow.py::test_package_cli_train_main_requires_b1_and_uses_main_config python/weiss_rl/tests/test_cli_workflow.py::test_package_cli_train_main_accepts_guided_seed_run python/weiss_rl/tests/test_cli_workflow.py::test_package_cli_train_main_guided_bootstrap_selected_resolves_init_policy_id` | Passed: 13 passed. |
| `uv run --extra dev python -m ruff check python/weiss_rl/workflows/snapshots.py python/weiss_rl/tests/test_workflow_snapshots.py python/weiss_rl/tests/test_cli_workflow.py` | Passed. |
| `uv run --extra dev python -m ruff format --check python/weiss_rl/workflows/snapshots.py python/weiss_rl/tests/test_workflow_snapshots.py python/weiss_rl/tests/test_cli_workflow.py` | Passed: 3 files already formatted. |
| `uv run --extra dev python -m mypy python/weiss_rl/cli.py python/weiss_rl/workflows` | Passed: no issues found in 11 source files. |
| `uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_cli_workflow.py` | Passed: 24 passed. |

### Tests Added

- Added four B1 seed checkpoint resolver tests to `python/weiss_rl/tests/test_workflow_snapshots.py`.

### Files Moved or Deleted

- None.

### Behavior Changes

- None intended. Automatic B1 seed resolution still tries `b1_noleague_baseline`, `B1 NoLeague baseline`, and `selected_candidate` in that order; explicit policy ids still bypass the automatic fallback list.
- No training, evaluation, simulator, observation, action, reward, checkpoint, artifact, figure-rendering, or CLI argument semantics changed.

### Remaining Risks

- `resolve_snapshot_checkpoint_path` still performs the snapshot scan inline. A later cleanup can extract a private snapshot lookup helper once update/update_count error behavior is pinned directly.

### Next Action

- Pin snapshot update-field handling, including `update_count` compatibility and missing/non-integer update errors, then extract a small snapshot lookup helper from `resolve_snapshot_checkpoint_path`.
