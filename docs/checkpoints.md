# Checkpoints

Checkpoint compatibility is a hard constraint.

## Canonical Files

- `training/checkpoints/checkpoint_<update>.pt`
- `training/checkpoints/latest.pt`
- `training/checkpoints/best.pt`
- `training/checkpoints/observed_best.pt`
- `training/checkpoints/checkpoint_metadata_<update>.json`
- `training/checkpoints/checkpoint_tracker.json`

Alias semantics are deliberately distinct:

- `latest.pt` is always the chronological latest checkpoint.
- `best.pt` is only the strict promotion-eligible best checkpoint.
- `observed_best.pt` is the highest periodic dev-eval checkpoint seen so far,
  even when the confidence guard refuses to promote `best.pt`.

## Snapshot Registry

League snapshots are tracked separately under `training/snapshots/registry.json`. Evaluation may resolve policies from a registry copied from another run, so relative paths must resolve from the registry source run, not from the consumer run.

## Resume

`train.py` supports:

- `--resume-run-dir runs/<run>`
- `--resume-from latest`
- `--resume-from best`
- `--resume-from observed_best`
- direct checkpoint paths

Resume must verify spec and config hashes unless an explicitly documented compatibility path says otherwise.

## Helper Modules

Reusable checkpoint tracker, resume-path, latest/best/observed-best alias, and checkpoint-guard event helpers live in `weiss_rl.training.checkpoints`. The public training script keeps compatibility wrappers for existing script-local helper names.

Checkpoint payload contract validation also lives in `weiss_rl.training.checkpoints`. The training script still performs the actual model, optimizer, scaler, guidance, and counter restoration.

Checkpoint payload construction is centralized in `build_minimal_train_checkpoint_payload()`. Minimal checkpoint file writing, restore mechanics, current-checkpoint path naming, and current-checkpoint ensure logic are also reusable helpers, but the training script still supplies the run-specific config hash, spec hash, device, model-guidance callback, and checkpoint write/restore callbacks used by higher-level orchestration.

Checkpoint alias publication and checkpoint-guard rollback/finalization orchestration live in `weiss_rl.training.checkpoints`. These helpers preserve the same latest/best tracker semantics, keep `observed_best` out of strict rollback/finalization decisions, maintain JSONL guard event payloads, and preserve champion-demotion behavior while leaving public CLI behavior in `python/scripts/train.py`.

Snapshot artifact writing, registry retention, and champion-demotion file helpers live in `weiss_rl.training.snapshots`. The higher-level B1 baseline and seed snapshot import flows remain in the training script until their metadata and failure cases are pinned more tightly.

Imported B1 and seeded snapshot artifact writing share `write_imported_snapshot_artifact()` in `weiss_rl.training.snapshots`; validation and registry orchestration still live in the training script.

## Refactor Rules

- Do not rename checkpoint keys casually.
- Do not change alias update semantics silently.
- Do not change model state dict compatibility without migration notes.
- Add negative tests before refactoring restore paths.
