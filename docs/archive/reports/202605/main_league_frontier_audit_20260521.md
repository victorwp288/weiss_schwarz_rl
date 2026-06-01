# Main League Frontier Audit

- Selected run: `runs/main_champion_hardneg_interp_u10_repair_a015_20260517`
- Selected policy id: `main_interp_repair_a015`
- Publishable successor exists: `false`
- Decision: `no_confirm256_publishable_successor`
- Candidate records: `80`
- Scorecard entries: `15`
- Gate artifacts: `113`

## Best Non-Publishable Signals

| candidate | panel | seeds | fixed delta | learned delta | decision | reason |
|---|---:|---:|---:|---:|---|---|
| a050_b2oldhn_u2_policy_000002 | full | 128 | 2 | 4 | stop | full_gate_failed |
| a050p2_live_learnedpush_u1 | full | 64 | 2 | 0 | stop | full_gate_failed |
| a050p2_live_rowdeficit_u1 | full | 64 | 2 | 0 | stop | full_gate_failed |
| a050p2_live_rowgate_u1 | full | 64 | 2 | 0 | stop | full_gate_failed |
| a050p2_live_unlocked_rowdeficit_u1 | full | 64 | 2 | 0 | stop | full_gate_failed |
| candidate | full | 64 | 2 | 0 | stop | full_gate_failed |
| a050p2_live_unlocked_learned_recovery_u1 | sentinel | 16 | 0 | 0 | stop | sentinel_gate_failed |
| interp_w025 | sentinel | 16 | 0 | 0 | stop | sentinel_gate_failed |
| interp_w050 | sentinel | 16 | 0 | 0 | stop | sentinel_gate_failed |
| interp_w075 | sentinel | 16 | 0 | 0 | stop | sentinel_gate_failed |

## Remaining Non-Stop Records

All candidate families are stopped or published.

## Interpretation

No model should replace the locked selected checkpoint unless the JSON audit reports a publishable successor. At this snapshot, the audit keeps selected locked unless a successor has actual publish-level evidence. Mechanistic-only and sentinel-only survivors are next-step candidates, not selected-model evidence.

