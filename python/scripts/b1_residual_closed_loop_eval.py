"""Run a tensor-trained B1 residual policy in the simulator closed loop.

This is a bridge probe between offline adoption and a real B1 exploiter. It
loads the residual head trained by ``b1_residual_adoption_probe.py``, wraps a
live frozen B1 model, and runs a small S1 eval with action traces.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import b1_artifact_matrix as matrix
import b1_counterfactual_labels as cf
import torch
from weiss_rl.residual_policy import (
    FrozenStoredLogitResidual,
    LiveFrozenB1Residual,
    load_frozen_stored_logit_residual,
)

_RESIDUAL_POLICY_ID = "B1 residual S1"


def _canonical_policy_id(raw: str) -> str:
    try:
        return matrix._canonical_builtin_policy_id(str(raw))
    except Exception:
        return str(raw)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class _ResidualClosedLoopRunner(matrix._MatrixSimulatorEvalRunner):  # type: ignore[name-defined]
    def __init__(
        self, *args: Any, rng_alias_policy_id: str = "", rng_alias_target_policy_id: str = "", **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._rng_alias_policy_id = str(rng_alias_policy_id or "")
        self._rng_alias_target_policy_id = str(rng_alias_target_policy_id or "")

    def _rng_seed(self, *, scheduled_game: matrix.ScheduledGame, seat: int) -> int:  # type: ignore[override]
        if self._rng_alias_policy_id and self._rng_alias_target_policy_id:
            seat0_policy_id = scheduled_game.seat0_policy_id
            seat1_policy_id = scheduled_game.seat1_policy_id
            if seat0_policy_id == self._rng_alias_policy_id:
                seat0_policy_id = self._rng_alias_target_policy_id
            if seat1_policy_id == self._rng_alias_policy_id:
                seat1_policy_id = self._rng_alias_target_policy_id
            if seat0_policy_id != scheduled_game.seat0_policy_id or seat1_policy_id != scheduled_game.seat1_policy_id:
                scheduled_game = replace(
                    scheduled_game,
                    seat0_policy_id=seat0_policy_id,
                    seat1_policy_id=seat1_policy_id,
                )
        return super()._rng_seed(scheduled_game=scheduled_game, seat=seat)


def _load_residual_probe(*, residual_state_path: Path, device: torch.device) -> FrozenStoredLogitResidual:
    return load_frozen_stored_logit_residual(residual_state_path, device=device)


def _load_first_label(label_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = _read_jsonl(label_dir / "counterfactual_labels.jsonl")
    if not rows:
        raise RuntimeError(f"no labels found in {label_dir / 'counterfactual_labels.jsonl'}")
    row = rows[0]
    ref = row.get("tensor_ref")
    tensor_record: dict[str, Any] = {}
    if isinstance(ref, str) and ref:
        loaded = torch.load(label_dir / ref, map_location="cpu", weights_only=False)
        if isinstance(loaded, dict):
            tensor_record = loaded
    return row, tensor_record


def _policy_manifest_entry(policy_id: str, policy: Any) -> dict[str, Any]:
    return {
        "policy_id": policy_id,
        "kind": getattr(policy, "kind", ""),
        "source_run_dir": getattr(policy, "source_run_dir", None),
        "snapshot_path": getattr(policy, "snapshot_path", None),
        "model_object_id": None if getattr(policy, "model", None) is None else id(policy.model),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack-config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--b1-baseline-run-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--residual-state", type=Path, required=True)
    parser.add_argument("--opponent-policy", default=matrix.NO_LEAGUE_POLICY_ID)
    parser.add_argument("--artifact-dir-name", default="b1_residual_closed_loop_eval")
    parser.add_argument("--pairs", type=int, default=1)
    parser.add_argument("--seed-scope", default="b1_residual_closed_loop_eval_fixed")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--public-heuristic-bias-scale", type=float, default=1.0)
    parser.add_argument("--trace-top-k", type=int, default=8)
    parser.add_argument("--trace-max-decisions-per-episode", type=int, default=220)
    parser.add_argument("--greedy-residual", action="store_true")
    parser.add_argument(
        "--action-rng-salt-mode",
        choices=("shared", "physical", "policy", "matchup"),
        default="shared",
    )
    parser.add_argument(
        "--gate-to-label-obs",
        action="store_true",
        help="Apply the residual only when the stored label observation/actor seat is reached.",
    )
    parser.add_argument(
        "--alias-residual-rng-to-b1",
        action="store_true",
        help="Seed the residual seat sampler as if it were the B1 policy, for exact B1 prefix replay.",
    )
    parser.add_argument(
        "--residual-as-opponent",
        action="store_true",
        help="Evaluate B1 as focal and the residual as opponent, so residual is seat 1 on swap 0.",
    )
    args = parser.parse_args()
    base_opponent_policy_id = _canonical_policy_id(str(args.opponent_policy))

    device_name = str(args.device)
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    ctx = cf._build_in_process_context(
        stack_config=args.stack_config,
        run_dir=args.run_dir,
        b1_baseline_run_dir=args.b1_baseline_run_dir,
        checkpoint_policies=(),
        pairs=int(args.pairs),
        seed_scope=str(args.seed_scope),
        device=str(device),
        public_heuristic_bias_scale=float(args.public_heuristic_bias_scale),
        include_builtin=() if base_opponent_policy_id == matrix.NO_LEAGUE_POLICY_ID else (base_opponent_policy_id,),
    )
    base_policy = ctx.policies[matrix.NO_LEAGUE_POLICY_ID]
    if base_policy.model is None:
        raise RuntimeError("B1 baseline resolved without a model")

    label_row_for_gate, tensor_record_for_gate = _load_first_label(args.label_dir)
    gate_obs_sha256 = ""
    gate_actor_seat: int | None = None
    if bool(args.gate_to_label_obs):
        gate_obs_sha256 = str(tensor_record_for_gate.get("obs_sha256") or "")
        actor_value = tensor_record_for_gate.get("actor_seat", label_row_for_gate.get("target_seat"))
        if torch.is_tensor(actor_value):
            gate_actor_seat = int(actor_value.item())
        elif actor_value is not None:
            gate_actor_seat = int(actor_value)
        if not gate_obs_sha256:
            raise RuntimeError("requested --gate-to-label-obs but the label tensor has no obs_sha256")

    residual_probe = _load_residual_probe(residual_state_path=args.residual_state, device=device)
    live_model = LiveFrozenB1Residual(
        base_model=base_policy.model,
        residual_probe=residual_probe,
        gate_obs_sha256=gate_obs_sha256,
        gate_actor_seat=gate_actor_seat,
    ).to(device)
    live_model.eval()
    policies = dict(ctx.policies)
    policies[_RESIDUAL_POLICY_ID] = matrix.ResolvedEvalPolicy(
        policy_id=_RESIDUAL_POLICY_ID,
        kind="live_frozen_b1_residual",
        source_run_dir=args.residual_state.parent.as_posix(),
        snapshot_path=args.residual_state.name,
        model=live_model,
    )

    matrix_dir = args.run_dir / "eval" / str(args.artifact_dir_name)
    if base_opponent_policy_id not in policies:
        raise RuntimeError(f"opponent policy {base_opponent_policy_id!r} was not resolved")
    focal_policy_id = base_opponent_policy_id if bool(args.residual_as_opponent) else _RESIDUAL_POLICY_ID
    opponent_policy_id = _RESIDUAL_POLICY_ID if bool(args.residual_as_opponent) else base_opponent_policy_id
    matchup_dir = matrix_dir / f"{matrix._safe_slug(focal_policy_id)}__vs__{matrix._safe_slug(opponent_policy_id)}"
    matchup_dir.mkdir(parents=True, exist_ok=True)
    trace_path = matchup_dir / "action_trace.jsonl"
    if trace_path.exists():
        trace_path.unlink()

    _write_json(
        matrix_dir / "policy_load_manifest.json",
        {
            "format": "b1_residual_closed_loop_policy_manifest_v1",
            "residual_state": args.residual_state.as_posix(),
            "label_dir": args.label_dir.as_posix(),
            "residual_policy_id": _RESIDUAL_POLICY_ID,
            "base_policy_id": matrix.NO_LEAGUE_POLICY_ID,
            "base_opponent_policy_id": base_opponent_policy_id,
            "pairs": int(args.pairs),
            "seed_scope": str(args.seed_scope),
            "greedy_residual": bool(args.greedy_residual),
            "action_rng_salt_mode": str(args.action_rng_salt_mode),
            "alias_residual_rng_to_b1": bool(args.alias_residual_rng_to_b1),
            "residual_as_opponent": bool(args.residual_as_opponent),
            "gate_to_label_obs": bool(args.gate_to_label_obs),
            "gate_obs_sha256": gate_obs_sha256,
            "gate_actor_seat": gate_actor_seat,
            "public_heuristic_bias_scale": float(args.public_heuristic_bias_scale),
            "policies": {
                policy_id: _policy_manifest_entry(policy_id, policy) for policy_id, policy in policies.items()
            },
        },
    )

    runner = _ResidualClosedLoopRunner(
        stack=ctx.stack,
        policies=policies,
        artifact_layout=ctx.artifact_layout,
        run_id256=matrix._manifest_value(ctx.manifest, "run_id256", "computed_run_id256"),
        spec_hash256=ctx.spec_hash,
        action_dim=ctx.action_dim,
        pass_action_id=ctx.pass_action_id,
        require_sorted_legal_ids=bool(ctx.evaluation.eval_assert_sorted_legal_ids),
        replay_capture_rate=0.0,
        regression_capture_count=0,
        eval_device=str(device),
        spec_bundle=ctx.spec_bundle,
        scoring_mode="learner",
        greedy_policy_ids=[_RESIDUAL_POLICY_ID] if bool(args.greedy_residual) else [],
        action_rng_salt_mode=str(args.action_rng_salt_mode),
        trace_path=trace_path,
        trace_top_k=int(args.trace_top_k),
        trace_max_decisions_per_episode=int(args.trace_max_decisions_per_episode),
        action_catalog=ctx.action_catalog,
        rng_alias_policy_id=_RESIDUAL_POLICY_ID if bool(args.alias_residual_rng_to_b1) else "",
        rng_alias_target_policy_id=base_opponent_policy_id if bool(args.alias_residual_rng_to_b1) else "",
    )
    try:
        result = matrix.run_seat_swapped_matchup(
            focal_policy_id=focal_policy_id,
            opponent_policy_id=opponent_policy_id,
            paired_seeds=ctx.paired_seeds[: int(args.pairs)],
            runner=runner,
            episodes_path=matchup_dir / "episodes.jsonl",
            run_id256=matrix._manifest_value(ctx.manifest, "run_id256", "computed_run_id256"),
            config_hash256=matrix._manifest_value(ctx.manifest, "config_hash256", "config_hash"),
            spec_hash256=ctx.spec_hash,
        )
        runner_counters = runner.matrix_counters()
    finally:
        runner.close()

    records = list(result.records)
    matchup_payload = matrix.build_matchup_export(
        records,
        stop_rules=ctx.evaluation.stop_rules,
        max_paired_seeds=int(args.pairs),
        scheme=ctx.evaluation.final_policy_set_selection.folding,
        sample_count=1000,
        seed=0,
    )
    pair_rows, pair_summary = matrix._pair_table(records, scheme=ctx.evaluation.final_policy_set_selection.folding)
    matchup_payload["pair_class_summary"] = pair_summary
    matchup_payload["seat_diagnostics"] = matrix.build_seat_advantage_diagnostics(records)
    matchup_payload["matrix_runner_counters"] = runner_counters
    matrix.write_matchup_summary_json(matchup_dir / "matchup_summary.json", matchup_payload)
    matrix._write_json(matchup_dir / "pair_class_summary.json", pair_summary)
    matrix._write_jsonl(matchup_dir / "pair_table.jsonl", pair_rows)

    label_row, tensor_record = _load_first_label(args.label_dir)
    trace_rows = _read_jsonl(trace_path)
    expected_pair = int(label_row.get("pair_index", -1))
    target_seat = int(label_row.get("target_seat", -1))
    if bool(args.residual_as_opponent):
        expected_swap = 0 if target_seat == 1 else 1
    else:
        expected_swap = 1 if target_seat == 1 else 0
    expected_decision = int(label_row.get("decision_index", -1))
    expected_positive = int(label_row.get("positive_action_id", tensor_record.get("positive_action_id", -1)))
    expected_fingerprint = label_row.get("legal_action_fingerprint") or tensor_record.get("legal_action_fingerprint")
    expected_fingerprint = str(expected_fingerprint) if expected_fingerprint is not None else ""
    index_matched_rows = [
        row
        for row in trace_rows
        if int(row.get("pair_index", -1)) == expected_pair
        and int(row.get("swap_index", -1)) == expected_swap
        and int(row.get("decision_index", -1)) == expected_decision
        and str(row.get("policy_id")) == _RESIDUAL_POLICY_ID
    ]
    fingerprint_rows = [
        row for row in trace_rows if expected_fingerprint and str(row.get("legal_ids_sha256")) == expected_fingerprint
    ]
    residual_fingerprint_rows = [row for row in fingerprint_rows if str(row.get("policy_id")) == _RESIDUAL_POLICY_ID]
    strict_rows = [
        row
        for row in index_matched_rows
        if not expected_fingerprint or str(row.get("legal_ids_sha256")) == expected_fingerprint
    ]
    label_decision: dict[str, Any] | None = strict_rows[0] if strict_rows else None
    index_only_decision: dict[str, Any] | None = index_matched_rows[0] if index_matched_rows else None
    selected_action = None
    if isinstance(label_decision, Mapping):
        selected = label_decision.get("selected_action")
        if isinstance(selected, Mapping) and selected.get("action_id") is not None:
            selected_action = int(selected["action_id"])

    drift_rows = [row for row in trace_rows if str(row.get("policy_id")) == _RESIDUAL_POLICY_ID]
    changed_top_count = 0
    changed_family_count = 0
    for row in drift_rows:
        top = row.get("final_topk")
        selected = row.get("selected_action")
        if not isinstance(top, list) or not top or not isinstance(selected, Mapping):
            continue
        top_action = top[0]
        if int(selected.get("action_id", -1)) != int(top_action.get("action_id", -1)):
            changed_top_count += 1
        if str(selected.get("family")) != str(top_action.get("family")):
            changed_family_count += 1

    report = {
        "format": "b1_residual_closed_loop_eval_report_v1",
        "matrix_dir": matrix_dir.relative_to(args.run_dir).as_posix(),
        "matchup_dir": matchup_dir.relative_to(args.run_dir).as_posix(),
        "trace_path": trace_path.relative_to(args.run_dir).as_posix(),
        "residual_policy_id": _RESIDUAL_POLICY_ID,
        "focal_policy_id": focal_policy_id,
        "opponent_policy_id": opponent_policy_id,
        "pairs": int(args.pairs),
        "greedy_residual": bool(args.greedy_residual),
        "action_rng_salt_mode": str(args.action_rng_salt_mode),
        "alias_residual_rng_to_b1": bool(args.alias_residual_rng_to_b1),
        "residual_as_opponent": bool(args.residual_as_opponent),
        "gate_to_label_obs": bool(args.gate_to_label_obs),
        "residual_gate_counters": live_model.residual_gate_counters(),
        "summary": matchup_payload["summary"],
        "uncertainty": matchup_payload["uncertainty"],
        "pair_class_summary": pair_summary,
        "runner_counters": runner_counters,
        "label_probe": {
            "label_id": label_row.get("label_id"),
            "expected_pair_index": expected_pair,
            "expected_swap_index_for_residual_seat": expected_swap,
            "expected_decision_index": expected_decision,
            "expected_legal_action_fingerprint": expected_fingerprint,
            "expected_positive_action_id": expected_positive,
            "matched": label_decision is not None,
            "index_only_matched": index_only_decision is not None,
            "index_only_legal_action_fingerprint": None
            if index_only_decision is None
            else index_only_decision.get("legal_ids_sha256"),
            "index_only_legal_action_count": None
            if index_only_decision is None
            else index_only_decision.get("legal_action_count"),
            "fingerprint_match_count": len(fingerprint_rows),
            "residual_fingerprint_match_count": len(residual_fingerprint_rows),
            "fingerprint_matches": [
                {
                    "pair_index": row.get("pair_index"),
                    "swap_index": row.get("swap_index"),
                    "decision_index": row.get("decision_index"),
                    "actor_seat": row.get("actor_seat"),
                    "policy_id": row.get("policy_id"),
                    "legal_action_count": row.get("legal_action_count"),
                    "selected_action": row.get("selected_action"),
                    "seat0_policy_id": row.get("seat0_policy_id"),
                    "seat1_policy_id": row.get("seat1_policy_id"),
                }
                for row in fingerprint_rows[:16]
            ],
            "selected_action_id": selected_action,
            "adopted_positive_action": selected_action == expected_positive,
            "trace_row": label_decision,
            "index_only_trace_row": index_only_decision,
        },
        "residual_trace_drift": {
            "residual_decision_rows": len(drift_rows),
            "selected_differs_from_final_top1_count": int(changed_top_count),
            "selected_differs_from_final_top1_rate": float(changed_top_count / max(len(drift_rows), 1)),
            "selected_family_differs_from_final_top1_count": int(changed_family_count),
            "selected_family_differs_from_final_top1_rate": float(changed_family_count / max(len(drift_rows), 1)),
        },
    }
    _write_json(matrix_dir / "closed_loop_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
