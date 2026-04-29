"""Helpers for parallel canonical eval planning and execution."""

from __future__ import annotations

import multiprocessing
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast

import torch

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.config import load_stack_config
from weiss_rl.eval.final_eval import build_final_eval_matchups, finalize_final_eval, run_final_eval_matchup
from weiss_rl.eval.payoff_folding import PayoffFoldScheme


def shard_matchup_specs(
    *,
    matchup_specs: Sequence[Mapping[str, Any]],
    shard_count: int,
) -> list[list[dict[str, Any]]]:
    if shard_count < 1:
        raise ValueError("shard_count must be >= 1")
    shards: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    for index, matchup_spec in enumerate(matchup_specs):
        shards[index % shard_count].append(dict(matchup_spec))
    return [shard for shard in shards if shard]


def policy_ids_for_matchup_shard(matchup_specs: Sequence[Mapping[str, Any]]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for matchup_spec in matchup_specs:
        for key in ("focal_policy_id", "opponent_policy_id"):
            policy_id = str(matchup_spec[key])
            if policy_id not in seen:
                seen.add(policy_id)
                ordered.append(policy_id)
    return ordered


def resolved_parallel_worker_devices(
    *,
    parallel_workers: int,
    explicit_worker_devices: Sequence[str],
    eval_device: str,
) -> tuple[str, ...]:
    if parallel_workers < 1:
        raise ValueError("parallel_workers must be >= 1")

    normalized_explicit = tuple(device.strip() for device in explicit_worker_devices if device.strip())
    if normalized_explicit:
        device_pool = normalized_explicit
        validate_parallel_worker_device_pool(device_pool, source="parallel_worker_devices")
    else:
        normalized_eval_device = str(eval_device).strip().lower()
        if normalized_eval_device in {"auto", "cuda:auto"} and torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_pool = tuple(f"cuda:{index}" for index in range(device_count)) or ("cuda:auto",)
        elif normalized_eval_device in {"auto", "cuda:auto"}:
            device_pool = ("cpu",)
        else:
            device_pool = (str(eval_device).strip() or "cpu",)
            validate_parallel_worker_device_pool(device_pool, source="eval_device")
    return tuple(device_pool[index % len(device_pool)] for index in range(parallel_workers))


def validate_parallel_worker_device_pool(device_pool: Sequence[str], *, source: str) -> None:
    for device_text in device_pool:
        try:
            device = torch.device(str(device_text).strip())
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f"{source} contains invalid device {device_text!r}") from exc
        if device.type != "cuda":
            continue
        if not torch.cuda.is_available():
            raise ValueError(f"{source} requested CUDA device {device_text!r}, but CUDA is not available")
        device_count = int(torch.cuda.device_count())
        if device.index is not None and device.index >= device_count:
            raise ValueError(
                f"{source} requested CUDA device {device_text!r}, but only {device_count} CUDA device(s) are available"
            )


def run_final_eval_matchup_worker(
    *,
    stack_config_path: Path,
    run_dir: Path,
    output_dir: Path,
    policy_ids: Sequence[str],
    matchup_specs: Sequence[Mapping[str, Any]],
    paired_seeds: Sequence[int],
    stage1_paired_seeds: int,
    max_paired_seeds: int,
    stop_rules: Any,
    run_id256: str,
    config_hash256: str,
    spec_hash256: str,
    scheme: PayoffFoldScheme,
    sample_count: int,
    snapshot_registry_path: Path | None,
    b1_baseline_run_dir: Path | None,
    eval_device: str,
    replay_capture_rate: float,
    regression_capture_count: int,
) -> list[dict[str, Any]]:
    from weiss_rl.eval.simulator_runner import SimulatorEvalRunner, resolve_eval_policies
    from weiss_rl.simulator_contract import load_verified_simulator_contract

    stack = load_stack_config(stack_config_path)
    evaluation = stack.config.evaluation
    if evaluation is None:
        raise ValueError("stack config is missing evaluation settings")
    contract = load_verified_simulator_contract(stack.root, expected_spec_hash=spec_hash256)
    observation_dim = int(contract.spec_bundle["observation"]["obs_len"])
    action_dim = int(contract.spec_bundle["action"]["action_space_size"])
    pass_action_id = int(contract.spec_bundle["action"]["pass_action_id"])
    resolved_policies = resolve_eval_policies(
        stack=stack,
        policy_ids=list(policy_ids),
        run_dir=run_dir,
        observation_dim=observation_dim,
        action_dim=action_dim,
        spec_bundle=contract.spec_bundle,
        snapshot_registry_path=snapshot_registry_path,
        b1_baseline_run_dir=b1_baseline_run_dir,
        eval_device=eval_device,
    )
    layout = ArtifactLayout.from_run_dir(run_dir)
    runner = SimulatorEvalRunner(
        stack=stack,
        policies=resolved_policies,
        artifact_layout=layout,
        run_id256=run_id256,
        spec_hash256=spec_hash256,
        action_dim=action_dim,
        pass_action_id=pass_action_id,
        require_sorted_legal_ids=bool(evaluation.eval_assert_sorted_legal_ids),
        replay_capture_rate=float(replay_capture_rate),
        regression_capture_count=int(regression_capture_count),
        eval_device=eval_device,
        spec_bundle=contract.spec_bundle if isinstance(contract.spec_bundle, dict) else None,
    )
    try:
        return [
            run_final_eval_matchup(
                output_dir=output_dir,
                matchup_spec=matchup_spec,
                paired_seeds=paired_seeds,
                stage1_paired_seeds=stage1_paired_seeds,
                max_paired_seeds=max_paired_seeds,
                stop_rules=stop_rules,
                runner=runner,
                run_id256=run_id256,
                config_hash256=config_hash256,
                spec_hash256=spec_hash256,
                scheme=scheme,
                sample_count=sample_count,
            )
            for matchup_spec in matchup_specs
        ]
    finally:
        close_runner = getattr(runner, "close", None)
        if callable(close_runner):
            close_runner()


def run_parallel_final_eval(
    *,
    stack_config_path: Path,
    stack: Any,
    run_dir: Path,
    layout: ArtifactLayout,
    policy_ids: Sequence[str],
    paired_seeds: Sequence[int],
    stage1_paired_seeds: int,
    max_paired_seeds: int,
    stop_rules: Any,
    run_id256: str,
    config_hash256: str,
    spec_hash256: str,
    scheme: PayoffFoldScheme,
    sample_count: int,
    snapshot_registry_path: Path | None,
    b1_baseline_run_dir: Path | None,
    metadata: Mapping[str, Any],
    seed_file_path: Path | None,
    parallel_workers: int,
    parallel_worker_devices: Sequence[str],
    matchup_builder: Any = build_final_eval_matchups,
    matchup_worker: Any = run_final_eval_matchup_worker,
    finalizer: Any = finalize_final_eval,
) -> dict[str, Any]:
    evaluation = stack.config.evaluation
    if evaluation is None:
        raise ValueError("stack config is missing evaluation settings")

    matchup_specs = matchup_builder(policy_ids=policy_ids)
    metadata_payload = dict(metadata)
    pipeline_metadata = dict(cast(dict[str, Any], metadata_payload.get("pipeline", {})))
    worker_count = min(int(parallel_workers), len(matchup_specs))
    if worker_count < 2:
        pipeline_metadata["parallel_eval"] = {
            "enabled": False,
            "requested_worker_count": int(parallel_workers),
            "worker_count": int(max(1, worker_count)),
            "matchup_count": len(matchup_specs),
            "fallback_reason": "single_matchup",
            "replay_capture_mode": "serial_fallback_v1",
        }
        metadata_payload["pipeline"] = pipeline_metadata
        matchup_results = matchup_worker(
            stack_config_path=stack_config_path,
            run_dir=run_dir,
            output_dir=layout.final_eval_dir,
            policy_ids=policy_ids_for_matchup_shard(matchup_specs),
            matchup_specs=matchup_specs,
            paired_seeds=list(paired_seeds),
            stage1_paired_seeds=int(stage1_paired_seeds),
            max_paired_seeds=int(max_paired_seeds),
            stop_rules=stop_rules,
            run_id256=run_id256,
            config_hash256=config_hash256,
            spec_hash256=spec_hash256,
            scheme=scheme,
            sample_count=int(sample_count),
            snapshot_registry_path=snapshot_registry_path,
            b1_baseline_run_dir=b1_baseline_run_dir,
            eval_device=str(evaluation.eval_device),
            replay_capture_rate=float(evaluation.replay_capture_rate_eval),
            regression_capture_count=int(evaluation.regression_capture_count),
        )
        return finalizer(
            output_dir=layout.final_eval_dir,
            policy_ids=policy_ids,
            matchup_results=matchup_results,
            stage1_paired_seeds=stage1_paired_seeds,
            max_paired_seeds=max_paired_seeds,
            paired_seeds=paired_seeds,
            stop_rules=stop_rules,
            scheme=scheme,
            sample_count=sample_count,
            selection_payload={"mode": "explicit", "policy_count": len(policy_ids)},
            metadata=metadata_payload,
            seed_file_path=seed_file_path,
        )

    worker_devices = resolved_parallel_worker_devices(
        parallel_workers=worker_count,
        explicit_worker_devices=parallel_worker_devices,
        eval_device=str(evaluation.eval_device),
    )
    matchup_shards = shard_matchup_specs(matchup_specs=matchup_specs, shard_count=worker_count)
    pipeline_metadata["parallel_eval"] = {
        "enabled": True,
        "requested_worker_count": int(parallel_workers),
        "worker_count": len(matchup_shards),
        "matchup_count": len(matchup_specs),
        "worker_devices": list(worker_devices[: len(matchup_shards)]),
        "matchup_shard_sizes": [len(shard) for shard in matchup_shards],
        "replay_capture_mode": "disabled_parallel_v1",
    }
    metadata_payload["pipeline"] = pipeline_metadata

    futures = []
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(matchup_shards), mp_context=ctx) as executor:
        for shard_index, matchup_shard in enumerate(matchup_shards):
            futures.append(
                executor.submit(
                    matchup_worker,
                    stack_config_path=stack_config_path,
                    run_dir=run_dir,
                    output_dir=layout.final_eval_dir,
                    policy_ids=policy_ids_for_matchup_shard(matchup_shard),
                    matchup_specs=matchup_shard,
                    paired_seeds=list(paired_seeds),
                    stage1_paired_seeds=int(stage1_paired_seeds),
                    max_paired_seeds=int(max_paired_seeds),
                    stop_rules=stop_rules,
                    run_id256=run_id256,
                    config_hash256=config_hash256,
                    spec_hash256=spec_hash256,
                    scheme=scheme,
                    sample_count=int(sample_count),
                    snapshot_registry_path=snapshot_registry_path,
                    b1_baseline_run_dir=b1_baseline_run_dir,
                    eval_device=worker_devices[shard_index],
                    replay_capture_rate=0.0,
                    regression_capture_count=0,
                )
            )
        matchup_results: list[dict[str, Any]] = []
        for future in as_completed(futures):
            matchup_results.extend(future.result())

    return finalizer(
        output_dir=layout.final_eval_dir,
        policy_ids=policy_ids,
        matchup_results=matchup_results,
        stage1_paired_seeds=stage1_paired_seeds,
        max_paired_seeds=max_paired_seeds,
        paired_seeds=paired_seeds,
        stop_rules=stop_rules,
        scheme=scheme,
        sample_count=sample_count,
        selection_payload={"mode": "explicit", "policy_count": len(policy_ids)},
        metadata=metadata_payload,
        seed_file_path=seed_file_path,
    )
