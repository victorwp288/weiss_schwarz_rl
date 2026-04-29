"""Periodic dev-eval scheduling and league warmup-gate helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import torch

from weiss_rl.config import StackConfig
from weiss_rl.eval.harness import ScheduledGame
from weiss_rl.repro import hash_seed_file, parse_seed_file
from weiss_rl.training.dev_eval_metrics import dev_eval_aggregate_score


class RuntimeWarmupGate(Protocol):
    def set_league_eval_warmup_gate(self, *, open: bool) -> None: ...


@dataclass(frozen=True, slots=True)
class PeriodicDevEvalOpponentSpec:
    policy_id: str
    display_name: str
    kind: str
    snapshot_path: str | None = None
    heuristic_profile: str | None = None


@dataclass(frozen=True, slots=True)
class PeriodicDevEvalSeedBlockJob:
    opponent_index: int
    block_index: int
    opponent_spec: PeriodicDevEvalOpponentSpec
    paired_seed_items: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class PromotionGateSeedBlockJob:
    anchor_index: int
    block_index: int
    anchor_spec: PeriodicDevEvalOpponentSpec
    paired_seed_items: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class AsyncPeriodicDevEvalRequest:
    stack: StackConfig
    checkpoint_path: Path
    focal_policy_id: str
    update_count: int
    policy_version: int
    run_dir: Path
    run_id256: str
    config_hash256: str
    spec_hash256: str
    artifact_dir_name: str
    artifact_scope: str
    paired_seeds: tuple[int, ...]
    opponents: tuple[PeriodicDevEvalOpponentSpec, ...]
    eval_device_override: str | None
    parallel_workers: int
    parallel_worker_devices: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AsyncPromotionGateRequest:
    stack: StackConfig
    run_dir: Path
    candidate_policy_id: str
    candidate_snapshot_path: str
    update_count: int
    policy_version: int
    run_id256: str
    config_hash256: str
    spec_hash256: str
    anchor_policy_ids: dict[str, str]
    anchor_specs: tuple[PeriodicDevEvalOpponentSpec, ...]
    eval_device_override: str | None


def resolve_repo_path(root: Path, path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else root / path


def json_relative_path(path: Path, *, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def evaluation_config_or_raise(stack: StackConfig) -> Any:
    evaluation = stack.config.evaluation
    if evaluation is None:
        raise RuntimeError("The locked stack is missing the evaluation config block")
    return evaluation


def validate_periodic_dev_eval_contract(stack: StackConfig) -> Any:
    evaluation = evaluation_config_or_raise(stack)
    if not evaluation.seat_swap:
        raise RuntimeError("Periodic dev eval requires evaluation.seat_swap=true")
    if not evaluation.eval_inference_mode:
        raise RuntimeError("Periodic dev eval requires evaluation.eval_inference_mode=true")
    if evaluation.eval_sampling_algorithm != "pinned_cdf_pcg_v1":
        raise RuntimeError(
            "Periodic dev eval requires evaluation.eval_sampling_algorithm='pinned_cdf_pcg_v1', "
            f"got {evaluation.eval_sampling_algorithm!r}"
        )
    return evaluation


def resolve_periodic_dev_eval_seed_file(stack: StackConfig) -> tuple[Path, dict[str, str]]:
    evaluation = evaluation_config_or_raise(stack)
    reproducibility = stack.config.reproducibility
    resolved_paths: dict[str, Path] = {}
    if "dev_eval" in stack.seed_sets:
        resolved_paths["stack.seed_sets.dev_eval"] = stack.seed_sets["dev_eval"]
    if "dev_eval" in evaluation.seed_files:
        resolved_paths["evaluation.seed_files.dev_eval"] = resolve_repo_path(
            stack.root,
            evaluation.seed_files["dev_eval"],
        )
    if reproducibility is not None and "dev_eval" in reproducibility.seed_files:
        resolved_paths["reproducibility.seed_files.dev_eval"] = resolve_repo_path(
            stack.root,
            reproducibility.seed_files["dev_eval"],
        )
    if not resolved_paths:
        raise RuntimeError("Periodic dev eval requires a configured dev_eval seed file")

    unique_paths = {path.resolve() for path in resolved_paths.values()}
    if len(unique_paths) != 1:
        mismatch = {name: json_relative_path(path, root=stack.root) for name, path in resolved_paths.items()}
        raise RuntimeError(f"Periodic dev eval seed file mismatch: {mismatch}")

    seed_file = next(iter(resolved_paths.values()))
    return seed_file, {name: json_relative_path(path, root=stack.root) for name, path in resolved_paths.items()}


def periodic_dev_eval_schedule(stack: StackConfig) -> tuple[Path, dict[str, str], list[int], str]:
    evaluation = validate_periodic_dev_eval_contract(stack)
    seed_file, validated_sources = resolve_periodic_dev_eval_seed_file(stack)
    all_paired_seeds = parse_seed_file(seed_file)
    required_pairs = int(evaluation.periodic_dev_eval_paired_seeds)
    if len(all_paired_seeds) < required_pairs:
        raise RuntimeError(
            f"Periodic dev eval requires {required_pairs} paired seeds, found {len(all_paired_seeds)} in {seed_file}"
        )
    return seed_file, validated_sources, all_paired_seeds[:required_pairs], hash_seed_file(seed_file)


def periodic_dev_eval_anchor_weight_map(stack: StackConfig) -> dict[str, float]:
    evaluation = stack.config.evaluation
    if evaluation is None:
        return {}
    raw_weights = getattr(evaluation, "periodic_dev_eval_anchor_weights", {}) or {}
    if not isinstance(raw_weights, Mapping):
        return {}
    weights: dict[str, float] = {}
    for anchor_name, value in raw_weights.items():
        name = str(anchor_name).strip()
        if not name:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        weight = float(value)
        if not math.isfinite(weight) or weight < 0.0:
            continue
        weights[name] = weight
    return weights


def league_eval_warmup_gate_status(
    stack: StackConfig,
    dev_eval_summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    league = stack.config.league
    if league is None or not bool(league.enabled):
        return {"enabled": False, "open": True, "reasons": []}
    warmup = league.warmup
    if not bool(getattr(warmup, "eval_gate_enabled", False)):
        return {"enabled": False, "open": True, "reasons": []}
    reasons: list[str] = []
    if dev_eval_summary is None:
        return {"enabled": True, "open": False, "reasons": ["missing_dev_eval"]}
    current_score = dev_eval_aggregate_score(dev_eval_summary)
    min_aggregate_score = getattr(warmup, "eval_gate_min_aggregate_score", None)
    if min_aggregate_score is not None:
        if current_score is None or float(current_score) < float(min_aggregate_score):
            reasons.append("aggregate_score")
    anchor_scores = dev_eval_summary.get("anchor_scores")
    if not isinstance(anchor_scores, Mapping):
        anchor_scores = {}
    failed_anchors: dict[str, dict[str, float | None]] = {}
    for anchor_name, min_score in dict(getattr(warmup, "eval_gate_min_anchor_scores", {}) or {}).items():
        anchor_name_text = str(anchor_name)
        value = anchor_scores.get(anchor_name_text)
        if value is None and anchor_name_text in {"Latest recent snapshot", "Previous recent snapshot"}:
            continue
        score = float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None
        if score is None or score < float(min_score):
            failed_anchors[anchor_name_text] = {
                "score": score,
                "min_score": float(min_score),
            }
    if failed_anchors:
        reasons.append("anchor_scores")
    return {
        "enabled": True,
        "open": not reasons,
        "reasons": reasons,
        "failed_anchors": failed_anchors,
        "aggregate_score": current_score,
        "min_aggregate_score": min_aggregate_score,
    }


def sync_runtime_league_eval_warmup_gate(
    *,
    runtime: object,
    stack: StackConfig,
    dev_eval_summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    status = league_eval_warmup_gate_status(stack, dev_eval_summary)
    setter = getattr(runtime, "set_league_eval_warmup_gate", None)
    if callable(setter):
        setter(open=bool(status["open"]))
    return status


def format_league_eval_warmup_gate_message(status: Mapping[str, Any]) -> str:
    return (
        "League eval warmup gate: "
        f"open={bool(status['open'])} "
        f"reasons={','.join(str(reason) for reason in cast(Sequence[Any], status.get('reasons', [])))}"
    )


def should_run_periodic_dev_eval(stack: StackConfig, *, update_count: int) -> bool:
    evaluation = stack.config.evaluation
    if evaluation is None:
        return False
    interval = int(evaluation.periodic_dev_eval_interval_updates)
    return interval > 0 and update_count % interval == 0


def is_noleague_baseline_role(role: str) -> bool:
    normalized = str(role).strip()
    return normalized == "baseline_noleague" or normalized.startswith("baseline_noleague_")


def should_defer_noleague_baseline_alias_refresh(
    *,
    stack: StackConfig,
    experiment_role: str,
    update_count: int,
) -> bool:
    return is_noleague_baseline_role(experiment_role) and should_run_periodic_dev_eval(
        stack,
        update_count=update_count,
    )


def shard_periodic_dev_eval_opponents(
    *,
    opponent_specs: Sequence[PeriodicDevEvalOpponentSpec],
    shard_count: int,
) -> list[list[PeriodicDevEvalOpponentSpec]]:
    if shard_count < 1:
        raise ValueError("periodic dev eval shard_count must be >= 1")
    shards: list[list[PeriodicDevEvalOpponentSpec]] = [[] for _ in range(shard_count)]
    for index, opponent_spec in enumerate(opponent_specs):
        shards[index % shard_count].append(opponent_spec)
    return [shard for shard in shards if shard]


def periodic_dev_eval_duplicate_policy_ids(
    opponent_specs: Sequence[PeriodicDevEvalOpponentSpec],
) -> set[str]:
    counts: dict[str, int] = {}
    for spec in opponent_specs:
        counts[spec.policy_id] = counts.get(spec.policy_id, 0) + 1
    return {policy_id for policy_id, count in counts.items() if count > 1}


def split_periodic_dev_eval_seed_blocks(
    paired_seeds: Sequence[int],
    *,
    block_count: int,
) -> list[tuple[tuple[int, int], ...]]:
    if block_count < 1:
        raise ValueError("periodic dev eval seed block_count must be >= 1")
    indexed_seeds = tuple((index, int(seed)) for index, seed in enumerate(paired_seeds))
    if not indexed_seeds:
        return []
    effective_block_count = min(int(block_count), len(indexed_seeds))
    base_block_size, remainder = divmod(len(indexed_seeds), effective_block_count)
    blocks: list[tuple[tuple[int, int], ...]] = []
    start = 0
    for block_index in range(effective_block_count):
        block_size = base_block_size + (1 if block_index < remainder else 0)
        blocks.append(tuple(indexed_seeds[start : start + block_size]))
        start += block_size
    return blocks


def build_periodic_dev_eval_seed_block_jobs(
    *,
    opponent_specs: Sequence[PeriodicDevEvalOpponentSpec],
    paired_seeds: Sequence[int],
    configured_parallel_workers: int,
) -> list[PeriodicDevEvalSeedBlockJob]:
    if configured_parallel_workers < 1:
        raise ValueError("periodic dev eval configured_parallel_workers must be >= 1")
    if not opponent_specs:
        return []
    per_opponent_block_count = max(
        1,
        min(
            len(paired_seeds),
            int(math.ceil(configured_parallel_workers / max(1, len(opponent_specs)))),
        ),
    )
    seed_blocks = split_periodic_dev_eval_seed_blocks(
        paired_seeds,
        block_count=per_opponent_block_count,
    )
    jobs: list[PeriodicDevEvalSeedBlockJob] = []
    for opponent_index, opponent_spec in enumerate(opponent_specs):
        for block_index, paired_seed_items in enumerate(seed_blocks):
            jobs.append(
                PeriodicDevEvalSeedBlockJob(
                    opponent_index=opponent_index,
                    block_index=block_index,
                    opponent_spec=opponent_spec,
                    paired_seed_items=paired_seed_items,
                )
            )
    return jobs


def shard_periodic_dev_eval_seed_block_jobs(
    *,
    jobs: Sequence[PeriodicDevEvalSeedBlockJob],
    shard_count: int,
) -> list[list[PeriodicDevEvalSeedBlockJob]]:
    if shard_count < 1:
        raise ValueError("periodic dev eval seed-block shard_count must be >= 1")
    shards: list[list[PeriodicDevEvalSeedBlockJob]] = [[] for _ in range(shard_count)]
    for index, job in enumerate(jobs):
        shards[index % shard_count].append(job)
    return [shard for shard in shards if shard]


def periodic_dev_eval_schedule_for_seed_items(
    *,
    focal_policy_id: str,
    opponent_policy_id: str,
    paired_seed_items: Sequence[tuple[int, int]],
) -> list[ScheduledGame]:
    schedule: list[ScheduledGame] = []
    for pair_index, raw_seed in paired_seed_items:
        episode_seed = int(raw_seed)
        schedule.append(
            ScheduledGame(
                pair_index=int(pair_index),
                swap_index=0,
                episode_index=int(pair_index) * 2,
                episode_seed=episode_seed,
                focal_policy_id=focal_policy_id,
                opponent_policy_id=opponent_policy_id,
                seat0_policy_id=focal_policy_id,
                seat1_policy_id=opponent_policy_id,
                focal_seat=0,
            )
        )
        schedule.append(
            ScheduledGame(
                pair_index=int(pair_index),
                swap_index=1,
                episode_index=int(pair_index) * 2 + 1,
                episode_seed=episode_seed,
                focal_policy_id=focal_policy_id,
                opponent_policy_id=opponent_policy_id,
                seat0_policy_id=opponent_policy_id,
                seat1_policy_id=focal_policy_id,
                focal_seat=1,
            )
        )
    return schedule


def validate_parallel_worker_device_pool(device_pool: Sequence[str], *, source: str) -> None:
    for raw_device in device_pool:
        device_text = str(raw_device).strip()
        if not device_text:
            raise ValueError(f"{source} contains an empty device name")
        try:
            device = torch.device(device_text)
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


def resolved_promotion_gate_worker_devices(
    *,
    stack: StackConfig,
    parallel_workers: int,
    explicit_worker_devices: Sequence[str],
    eval_device: str,
) -> tuple[str, ...]:
    if parallel_workers < 1:
        raise ValueError("promotion gate parallel_workers must be >= 1")
    normalized_explicit = tuple(device.strip() for device in explicit_worker_devices if str(device).strip())
    if normalized_explicit:
        device_pool = normalized_explicit
        validate_parallel_worker_device_pool(device_pool, source="promotion_gate parallel_worker_devices")
    else:
        normalized_eval_device = str(eval_device).strip().lower()
        if normalized_eval_device in {"auto", "cuda:auto"} and torch.cuda.is_available():
            device_pool = tuple(f"cuda:{index}" for index in range(torch.cuda.device_count())) or ("cpu",)
        elif normalized_eval_device in {"auto", "cuda:auto"}:
            device_pool = ("cpu",)
        else:
            device_pool = (str(eval_device).strip() or "cpu",)
            validate_parallel_worker_device_pool(device_pool, source="promotion_gate eval_device")
    return tuple(device_pool[index % len(device_pool)] for index in range(parallel_workers))


def resolved_periodic_dev_eval_worker_devices(
    *,
    stack: StackConfig,
    parallel_workers: int,
    explicit_worker_devices: Sequence[str],
    eval_device: str,
    learner_device: torch.device | None = None,
    actor_device_layout_resolver: Any | None = None,
) -> tuple[str, ...]:
    if parallel_workers < 1:
        raise ValueError("periodic dev eval parallel_workers must be >= 1")

    normalized_explicit = tuple(device.strip() for device in explicit_worker_devices if str(device).strip())
    if normalized_explicit:
        device_pool = normalized_explicit
        validate_parallel_worker_device_pool(device_pool, source="periodic_dev_eval_parallel_worker_devices")
    else:
        normalized_eval_device = str(eval_device).strip().lower()
        if normalized_eval_device in {"auto", "cuda:auto"} and torch.cuda.is_available():
            if learner_device is not None and actor_device_layout_resolver is not None:
                actor_count = 1 if stack.config.system is None else int(stack.config.system.actor_process_count)
                actor_layout = actor_device_layout_resolver(
                    stack,
                    actor_count=actor_count,
                    learner_device=learner_device,
                    prefer_process_collectors=True,
                )
                actor_pool = tuple(
                    device_name
                    for device_name in dict.fromkeys(actor_layout)
                    if torch.device(device_name).type == "cuda"
                )
                if actor_pool:
                    device_pool = actor_pool
                else:
                    device_pool = tuple(f"cuda:{index}" for index in range(torch.cuda.device_count())) or ("cpu",)
            else:
                device_pool = tuple(f"cuda:{index}" for index in range(torch.cuda.device_count())) or ("cpu",)
        elif normalized_eval_device in {"auto", "cuda:auto"}:
            device_pool = ("cpu",)
        else:
            device_pool = (str(eval_device).strip() or "cpu",)
            validate_parallel_worker_device_pool(device_pool, source="periodic_dev_eval eval_device")
    return tuple(device_pool[index % len(device_pool)] for index in range(parallel_workers))


def shard_promotion_gate_anchor_specs(
    *,
    anchor_specs: Sequence[PeriodicDevEvalOpponentSpec],
    shard_count: int,
) -> list[list[tuple[int, PeriodicDevEvalOpponentSpec]]]:
    if shard_count < 1:
        raise ValueError("promotion gate shard_count must be >= 1")
    shards: list[list[tuple[int, PeriodicDevEvalOpponentSpec]]] = [[] for _ in range(shard_count)]
    for index, anchor_spec in enumerate(anchor_specs):
        shards[index % shard_count].append((index, anchor_spec))
    return [shard for shard in shards if shard]


def build_promotion_gate_seed_block_jobs(
    *,
    anchor_specs: Sequence[PeriodicDevEvalOpponentSpec],
    paired_seeds: Sequence[int],
    configured_parallel_workers: int,
) -> list[PromotionGateSeedBlockJob]:
    if configured_parallel_workers < 1:
        raise ValueError("promotion gate configured_parallel_workers must be >= 1")
    if not anchor_specs:
        return []
    per_anchor_block_count = max(
        1,
        min(
            len(paired_seeds),
            int(math.ceil(configured_parallel_workers / max(1, len(anchor_specs)))),
        ),
    )
    seed_blocks = split_periodic_dev_eval_seed_blocks(
        paired_seeds,
        block_count=per_anchor_block_count,
    )
    jobs: list[PromotionGateSeedBlockJob] = []
    for anchor_index, anchor_spec in enumerate(anchor_specs):
        for block_index, paired_seed_items in enumerate(seed_blocks):
            jobs.append(
                PromotionGateSeedBlockJob(
                    anchor_index=anchor_index,
                    block_index=block_index,
                    anchor_spec=anchor_spec,
                    paired_seed_items=paired_seed_items,
                )
            )
    return jobs


def shard_promotion_gate_seed_block_jobs(
    *,
    jobs: Sequence[PromotionGateSeedBlockJob],
    shard_count: int,
) -> list[list[PromotionGateSeedBlockJob]]:
    if shard_count < 1:
        raise ValueError("promotion gate seed-block shard_count must be >= 1")
    shards: list[list[PromotionGateSeedBlockJob]] = [[] for _ in range(shard_count)]
    for index, job in enumerate(jobs):
        shards[index % shard_count].append(job)
    return [shard for shard in shards if shard]


def build_async_periodic_dev_eval_request(
    *,
    stack: StackConfig,
    checkpoint_path: Path,
    focal_policy_id: str,
    update_count: int,
    policy_version: int,
    run_dir: Path,
    run_id256: str,
    config_hash256: str,
    spec_hash256: str,
    artifact_dir_name: str,
    artifact_scope: str,
    paired_seeds: Sequence[int],
    opponents: Sequence[PeriodicDevEvalOpponentSpec],
    eval_device_override: str | None,
    parallel_workers: int,
    parallel_worker_devices: Sequence[str],
) -> AsyncPeriodicDevEvalRequest:
    return AsyncPeriodicDevEvalRequest(
        stack=stack,
        checkpoint_path=checkpoint_path,
        focal_policy_id=str(focal_policy_id),
        update_count=int(update_count),
        policy_version=int(policy_version),
        run_dir=run_dir,
        run_id256=str(run_id256),
        config_hash256=str(config_hash256),
        spec_hash256=str(spec_hash256),
        artifact_dir_name=str(artifact_dir_name),
        artifact_scope=str(artifact_scope),
        paired_seeds=tuple(int(seed) for seed in paired_seeds),
        opponents=tuple(opponents),
        eval_device_override=eval_device_override,
        parallel_workers=max(1, int(parallel_workers)),
        parallel_worker_devices=tuple(str(device) for device in parallel_worker_devices),
    )


def build_async_promotion_gate_request(
    *,
    stack: StackConfig,
    run_dir: Path,
    candidate_policy_id: str,
    candidate_snapshot_path: str,
    update_count: int,
    policy_version: int,
    run_id256: str,
    config_hash256: str,
    spec_hash256: str,
    anchor_policy_ids: Mapping[str, str],
    anchor_specs: Sequence[PeriodicDevEvalOpponentSpec],
    eval_device_override: str | None,
) -> AsyncPromotionGateRequest:
    return AsyncPromotionGateRequest(
        stack=stack,
        run_dir=run_dir,
        candidate_policy_id=str(candidate_policy_id),
        candidate_snapshot_path=str(candidate_snapshot_path),
        update_count=int(update_count),
        policy_version=int(policy_version),
        run_id256=str(run_id256),
        config_hash256=str(config_hash256),
        spec_hash256=str(spec_hash256),
        anchor_policy_ids=dict(anchor_policy_ids),
        anchor_specs=tuple(anchor_specs),
        eval_device_override=eval_device_override,
    )
