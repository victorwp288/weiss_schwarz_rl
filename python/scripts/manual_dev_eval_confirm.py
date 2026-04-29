from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import train as train_script
from weiss_rl.eval.simulator_runner import SimulatorEvalRunner


def _round_robin_cuda_worker_devices(worker_count: int) -> tuple[str, ...]:
    workers = max(1, int(worker_count))
    visible = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
    visible_count = len([part for part in visible.split(",") if part.strip()]) if visible else 0
    cuda_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    device_count = max(visible_count, cuda_count)
    if device_count <= 0:
        return ("cpu",) * workers
    return tuple(f"cuda:{index % device_count}" for index in range(workers))


def _manifest_value(manifest: dict[str, object], *keys: str) -> str:
    for key in keys:
        value = manifest.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an expanded scalar periodic dev-eval for one checkpoint.")
    parser.add_argument("--stack-config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--update", type=int, required=True)
    parser.add_argument("--pairs", type=int, default=32)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--artifact-dir-name", default="dev_eval_confirmatory_manual32")
    parser.add_argument(
        "--only-anchor",
        action="append",
        default=None,
        help="Restrict confirmatory eval to anchor display name or policy id; may be passed multiple times.",
    )
    parser.add_argument(
        "--extra-snapshot-anchor",
        action="append",
        default=None,
        metavar="POLICY_ID=DISPLAY_NAME",
        help=(
            "Add a snapshot from the run registry as an eval anchor. The value may be POLICY_ID or "
            "POLICY_ID=DISPLAY_NAME; may be passed multiple times."
        ),
    )
    parser.add_argument(
        "--focal-action-mode",
        choices=("sample", "greedy"),
        default="sample",
        help="Diagnostic-only action selection for the focal policy. Canonical eval uses sample.",
    )
    args = parser.parse_args()

    stack = train_script.load_stack_config(args.stack_config)
    manifest = json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    runtime_spec = manifest.get("runtime_spec")
    spec_hash = ""
    if isinstance(runtime_spec, dict):
        spec_hash = str(runtime_spec.get("sha256") or "")
    if not spec_hash:
        spec_hash = _manifest_value(manifest, "spec_hash256", "spec_hash")
    if not spec_hash:
        spec_hash_path = args.run_dir / "runtime_spec_hash.txt"
        spec_hash = spec_hash_path.read_text(encoding="utf-8").strip()
    contract = train_script.load_verified_simulator_contract(stack.root, expected_spec_hash=spec_hash)
    _seed_file, _validated, base_seeds, seed_sha = train_script._periodic_dev_eval_schedule(stack)
    paired_seeds = train_script._expand_periodic_dev_eval_paired_seeds(
        base_seeds,
        requested_pairs=int(args.pairs),
        seed_file_sha256=seed_sha,
        update_count=int(args.update),
        policy_version=int(summary["policy_version"]),
        scope=str(args.artifact_dir_name),
    )
    opponent_specs = None
    extra_specs = []
    if args.extra_snapshot_anchor:
        registry_path = args.run_dir / "training" / "snapshots" / train_script.REGISTRY_FILENAME
        if not registry_path.is_file():
            raise SystemExit(f"snapshot registry not found: {registry_path}")
        registry = train_script.SnapshotRegistry.load(registry_path)
        snapshots_by_id = {snapshot.policy_id: snapshot for snapshot in registry.snapshots}
        for raw_anchor in args.extra_snapshot_anchor:
            raw_text = str(raw_anchor).strip()
            if not raw_text:
                raise SystemExit("--extra-snapshot-anchor values must be non-empty")
            policy_id, sep, display_name = raw_text.partition("=")
            policy_id = policy_id.strip()
            display_name = display_name.strip() if sep else policy_id
            if not policy_id:
                raise SystemExit(f"invalid --extra-snapshot-anchor value: {raw_anchor!r}")
            snapshot = snapshots_by_id.get(policy_id)
            if snapshot is None:
                available = ", ".join(sorted(snapshots_by_id))
                raise SystemExit(f"snapshot policy id not found: {policy_id}. Available: {available}")
            extra_specs.append(
                train_script.PeriodicDevEvalOpponentSpec(
                    policy_id=policy_id,
                    display_name=display_name,
                    kind="snapshot",
                    snapshot_path=snapshot.path,
                )
            )
    if args.only_anchor:
        requested = {str(anchor).strip() for anchor in args.only_anchor if str(anchor).strip()}
        if not requested:
            raise SystemExit("--only-anchor values must be non-empty")
        resolved_specs, _pinned_snapshot_ids = train_script._resolve_periodic_dev_eval_opponent_specs(
            stack=stack,
            run_dir=args.run_dir,
        )
        opponent_specs = tuple(
            spec
            for spec in (*resolved_specs, *extra_specs)
            if str(spec.display_name) in requested or str(spec.policy_id) in requested
        )
        if not opponent_specs:
            available = ", ".join(f"{spec.display_name} ({spec.policy_id})" for spec in (*resolved_specs, *extra_specs))
            raise SystemExit(f"--only-anchor did not match any configured anchor. Available: {available}")
    elif extra_specs:
        opponent_specs = tuple(extra_specs)
    focal_policy_id = str(summary["policy_id"])
    if args.focal_action_mode == "greedy":
        original_select_action = SimulatorEvalRunner._select_action

        def _select_action_with_greedy_focal(self, **kwargs):
            if str(kwargs.get("current_policy_id")) != focal_policy_id:
                return original_select_action(self, **kwargs)
            policy = self.policies.get(focal_policy_id)
            if policy is None or policy.model is None:
                return original_select_action(self, **kwargs)
            seat_hidden = kwargs.get("seat_hidden")
            if seat_hidden is None:
                return original_select_action(self, **kwargs)
            batch = kwargs["batch"]
            current_seat = int(kwargs["current_seat"])
            legal_ids = np.asarray(kwargs["legal_ids"], dtype=np.uint32)
            if legal_ids.size == 0:
                return original_select_action(self, **kwargs)
            with torch.inference_mode():
                logits_tensor, _value_tensor, next_seat_hidden = policy.model.forward_seat_aware(
                    torch.as_tensor(np.asarray(batch.obs, dtype=np.float32), device=self._device),
                    torch.as_tensor([current_seat], device=self._device, dtype=torch.long),
                    seat_hidden,
                    scoring_mode="learner",
                )
            logits = logits_tensor[0].detach().cpu().numpy().astype(np.float32, copy=False)
            legal_logits = logits[legal_ids.astype(np.int64, copy=False)]
            return int(legal_ids[int(np.argmax(legal_logits))]), next_seat_hidden

        SimulatorEvalRunner._select_action = _select_action_with_greedy_focal
    try:
        result = train_script._run_periodic_dev_eval_for_checkpoint(
            stack=stack,
            contract=contract,
            run_dir=args.run_dir,
            checkpoint_path=args.checkpoint,
            focal_policy_id=focal_policy_id,
            update_count=int(args.update),
            policy_version=int(summary["policy_version"]),
            run_id256=_manifest_value(manifest, "run_id256", "computed_run_id256"),
            config_hash256=_manifest_value(manifest, "config_hash", "config_hash256"),
            spec_hash256=str(spec_hash),
            artifact_dir_name=str(args.artifact_dir_name),
            artifact_scope=str(args.artifact_dir_name),
            paired_seeds_override=paired_seeds,
            parallel_workers_override=int(args.workers),
            parallel_worker_devices_override=_round_robin_cuda_worker_devices(int(args.workers)),
            batched_inference_override=False,
            opponent_specs=opponent_specs,
        )
    finally:
        if args.focal_action_mode == "greedy":
            SimulatorEvalRunner._select_action = original_select_action
    print(
        "manual confirm done "
        f"run_dir={args.run_dir} update={int(args.update)} "
        f"pairs={len(paired_seeds)} aggregate={float(result['aggregate_score']):.6f} "
        f"unweighted={float(result.get('unweighted_aggregate_score', 0.0)):.6f} "
        f"focal_action_mode={args.focal_action_mode} "
        f"anchors={result.get('anchor_scores')}"
    )


if __name__ == "__main__":
    main()
