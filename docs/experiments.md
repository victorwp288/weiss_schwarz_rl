# Experiments

The thesis surface is intentionally small:

- `configs/thesis/b1_noleague.yaml`
- `configs/thesis/main_league.yaml`
- `configs/thesis/main_league_auto_gpu.yaml`
- `configs/thesis/final_eval.yaml`
- `configs/thesis/multideck_exploratory.yaml`
- `configs/thesis/ablations/no_gru.yaml`
- `configs/thesis/ablations/ppo_lite.yaml`
- `configs/thesis/ablations/terminal_only_reward.yaml`

B1 and main league are the primary training lanes. B0, B2, B3, and B4 are
evaluation policies. Multideck is exploratory only.

Older dated, rescue, campaign, sweep, and targeted-confirm scripts are
compatibility or investigation tools, not standard thesis commands. If used,
record the reason and output in `docs/rebuild_log.md`.

The standard B2 diagnostic command is:

```powershell
uv run --extra dev --extra sim python -m weiss_rl.cli b2-audit `
  --run-dir runs/<run_dir> `
  --episodes-jsonl runs/<run_dir>/eval/final_eval/episodes.jsonl `
  --policy-id <focal_policy_id>
```

## Claim Rules

- Training quality claims require saved run artifacts and eval summaries.
- Performance claims require benchmark or training metric artifacts.
- Smoke runs prove plumbing only.
- B2 flatline or suspicious B2 behavior requires causal diagnosis before
  adding more training time.
