# Local Artifacts Inventory - 2026-05-27

This inventory records the preservation boundary used during the pre-push
cleanup. The goal is to keep research artifacts on disk while keeping the GitHub
tree focused on source, configs, docs, compact summaries, and reproducible
commands.

## Local Artifact Roots

| Path | Approx size | Git policy | Notes |
| --- | ---: | --- | --- |
| `runs/` | 124605.86 MB | Keep local, ignore in Git except `runs/README.md` | Full training/eval runs, checkpoints, snapshots, replay bundles, tensorboard files. |
| `diagnostics/` | 663.71 MB | Keep local, ignore in Git | Raw audit outputs and replay bundles. Summaries should be promoted to docs or compact JSON artifacts before sharing. |
| `run_logs/` | 0.36 MB | Track selected logs | Existing historical logs are tracked; new logs are small enough to keep when they support thesis claims. |
| `vast_artifacts/` | 2.24 MB | Track | Compact thesis summaries from remote/Vast runs. |
| `thesis_figures_final/` | 1.41 MB | Track | Final figure exports and snippets. |
| `temp/` | 16.81 MB | Keep local, ignore in Git | Upload bundles and handoff zips. |
| `now/`, `now.zip` | 0.92 MB / 0.19 MB | Keep local, ignore in Git | Point-in-time handoff bundle copy. |
| `web/human-play/node_modules/`, `web/human-play/dist/` | generated | Keep local, ignore in Git | Recreated by npm install/build. |

## Cleanup Decisions

- Preserve all run and checkpoint files on disk.
- Remove tracked run payloads from the Git index while keeping `runs/README.md`
  tracked.
- Track source, docs, configs, tests, compact summaries, and the human-play web
  source.
- Ignore raw diagnostics, handoff bundles, local tool output, frontend build
  output, and dependency directories.
- Do not rewrite Git history in this cleanup pass. A history rewrite would be a
  separate, explicit GitHub maintenance step because the current `.git` object
  store is already large.

## Current Size Notes

- `runs/` is the dominant workspace payload at about 124.6 GB.
- `.git` is about 34.2 GB on disk, with `git count-objects -vH` reporting
  23.28 GiB loose objects and 10.13 GiB packed objects before cleanup.
- No currently tracked working-tree file above 50 MB was found in the pre-push
  scan.
