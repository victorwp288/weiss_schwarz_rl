from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.config import load_stack_config
from weiss_rl.eval.final_eval import _build_final_eval_payload, _run_matchup, _write_final_eval_artifacts
from weiss_rl.eval.payoff_folding import PayoffFoldScheme
from weiss_rl.eval.simulator_runner import SimulatorEvalRunner, resolve_eval_policies
from weiss_rl.repro import parse_seed_file
from weiss_rl.simulator_contract import load_verified_simulator_contract


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _worker(job: dict[str, Any]) -> dict[str, Any]:
    stack = load_stack_config(Path(job["stack_config"]))
    run_dir = Path(job["run_dir"])
    layout = ArtifactLayout.from_run_dir(run_dir)
    manifest = _load_json(layout.manifest_path)
    contract = load_verified_simulator_contract(
        stack.root,
        expected_spec_hash=str(manifest.get("spec_hash256", "")).strip(),
    )
    observation_dim = int(contract.spec_bundle["observation"]["obs_len"])
    action_dim = int(contract.spec_bundle["action"]["action_space_size"])
    pass_action_id = int(contract.spec_bundle["action"]["pass_action_id"])
    policy_pair = [str(job["focal_policy_id"]), str(job["opponent_policy_id"])]
    # Keep order unique on diagonal while still resolving the policy once.
    resolve_ids = list(dict.fromkeys(policy_pair))
    evaluation = stack.config.evaluation
    if evaluation is None:
        raise ValueError("stack config is missing evaluation settings")
    policies = resolve_eval_policies(
        stack=stack,
        policy_ids=resolve_ids,
        run_dir=run_dir,
        observation_dim=observation_dim,
        action_dim=action_dim,
        spec_bundle=contract.spec_bundle,
        snapshot_registry_path=Path(job["snapshot_registry_json"]) if job.get("snapshot_registry_json") else None,
        b1_baseline_run_dir=Path(job["b1_baseline_run_dir"]) if job.get("b1_baseline_run_dir") else None,
    )
    runner = SimulatorEvalRunner(
        stack=stack,
        policies=policies,
        artifact_layout=layout,
        run_id256=str(manifest["run_id256"]),
        spec_hash256=str(manifest["spec_hash256"]),
        action_dim=action_dim,
        pass_action_id=pass_action_id,
        require_sorted_legal_ids=bool(evaluation.eval_assert_sorted_legal_ids),
        replay_capture_rate=float(evaluation.replay_capture_rate_eval),
        regression_capture_count=int(evaluation.regression_capture_count),
    )
    return _run_matchup(
        output_dir=layout.final_eval_dir,
        focal_index=int(job["focal_index"]),
        opponent_index=int(job["opponent_index"]),
        focal_policy_id=str(job["focal_policy_id"]),
        opponent_policy_id=str(job["opponent_policy_id"]),
        paired_seeds=list(job["paired_seeds"]),
        stage1_paired_seeds=int(job["stage1_paired_seeds"]),
        max_paired_seeds=int(job["max_paired_seeds"]),
        stop_rules=evaluation.stop_rules,
        runner=runner,
        run_id256=str(manifest["run_id256"]),
        config_hash256=str(manifest["config_hash256"]),
        spec_hash256=str(manifest["spec_hash256"]),
        scheme=str(job["scheme"]),
        sample_count=int(job["bootstrap_samples"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel canonical final eval for independent matchups")
    parser.add_argument("--stack-config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--policy-id", action="append", default=[])
    parser.add_argument("--snapshot-registry-json", type=Path, default=None)
    parser.add_argument("--b1-baseline-run-dir", type=Path, default=None)
    parser.add_argument("--paired-seed-limit", type=int, default=16)
    parser.add_argument("--stage1-paired-seeds", type=int, default=16)
    parser.add_argument("--max-paired-seeds", type=int, default=16)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force-clear", action="store_true")
    args = parser.parse_args()

    policy_ids = [str(p).strip() for p in args.policy_id if str(p).strip()]
    if not policy_ids:
        raise SystemExit("provide at least one --policy-id")
    stack = load_stack_config(args.stack_config)
    evaluation = stack.config.evaluation
    if evaluation is None:
        raise ValueError("stack config is missing evaluation settings")
    layout = ArtifactLayout.from_run_dir(args.run_dir)
    layout.ensure_directories()
    if args.force_clear and layout.final_eval_dir.exists():
        shutil.rmtree(layout.final_eval_dir)
    layout.final_eval_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_json(layout.manifest_path)
    seed_file_path = stack.seed_sets["report_eval"]
    paired_seeds = parse_seed_file(seed_file_path)[: int(args.paired_seed_limit)]
    if len(paired_seeds) < int(args.max_paired_seeds):
        raise ValueError("not enough paired seeds")
    scheme = str(evaluation.final_policy_set_selection.folding)

    jobs: list[dict[str, Any]] = []
    for focal_index, focal_policy_id in enumerate(policy_ids):
        for opponent_index, opponent_policy_id in enumerate(policy_ids[focal_index:], start=focal_index):
            jobs.append(
                {
                    "stack_config": args.stack_config.as_posix(),
                    "run_dir": args.run_dir.as_posix(),
                    "snapshot_registry_json": None if args.snapshot_registry_json is None else args.snapshot_registry_json.as_posix(),
                    "b1_baseline_run_dir": None if args.b1_baseline_run_dir is None else args.b1_baseline_run_dir.as_posix(),
                    "paired_seeds": paired_seeds,
                    "stage1_paired_seeds": int(args.stage1_paired_seeds),
                    "max_paired_seeds": int(args.max_paired_seeds),
                    "bootstrap_samples": int(args.bootstrap_samples),
                    "scheme": scheme,
                    "focal_index": focal_index,
                    "opponent_index": opponent_index,
                    "focal_policy_id": focal_policy_id,
                    "opponent_policy_id": opponent_policy_id,
                }
            )

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
        futures = {executor.submit(_worker, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            result = future.result()
            results.append(result)
            print(
                f"completed {len(results)}/{len(jobs)}: "
                f"{job['focal_policy_id']} vs {job['opponent_policy_id']}",
                flush=True,
            )
    results.sort(key=lambda item: (int(item["focal_index"]), int(item["opponent_index"])))
    payload = _build_final_eval_payload(
        output_dir=layout.final_eval_dir,
        policy_ids=policy_ids,
        matchup_results=results,
        stage1_paired_seeds=int(args.stage1_paired_seeds),
        max_paired_seeds=int(args.max_paired_seeds),
        paired_seeds=paired_seeds,
        stop_rules=evaluation.stop_rules,
        scheme=scheme,  # type: ignore[arg-type]
        sample_count=int(args.bootstrap_samples),
        selection_payload={"mode": "explicit_parallel_cli", "policy_count": len(policy_ids), "workers": int(args.workers)},
        metadata={"pipeline": {"kind": "parallel_final_eval_v1", "workers": int(args.workers)}},
        seed_file_path=seed_file_path,
    )
    _write_final_eval_artifacts(output_dir=layout.final_eval_dir, payload=payload, matchup_results=results)
    print(f"Parallel final_eval summary JSON: {layout.final_eval_summary_json()}")


if __name__ == "__main__":
    main()
