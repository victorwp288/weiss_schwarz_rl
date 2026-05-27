"""Build targeted replay-BC data from champion and hard-negative matchups."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.artifacts.reproducibility import canonical_json_bytes, parse_seed_file, sha256_hex
from weiss_rl.config import StackConfig, compute_config_hash256
from weiss_rl.core.simulator_contract import SimulatorContract
from weiss_rl.eval.export import write_matchup_summary_json
from weiss_rl.eval.harness import (
    EvalGameRecord,
    build_seat_swapped_schedule,
    record_completed_game,
    write_episodes_jsonl,
)
from weiss_rl.eval.simulator_runner import SimulatorEvalRunner, resolve_eval_policies
from weiss_rl.league.registry import SnapshotRegistry
from weiss_rl.replay.trajectory_bc import (
    ReplayTrajectoryDataset,
    build_replay_trajectory_bc_dataset,
    merge_replay_trajectory_bc_datasets,
    save_replay_trajectory_bc_dataset,
)

_SCRIPT_KIND = "champion_hardneg_trajectory_bc_dataset_v1"


@dataclass(frozen=True, slots=True)
class OpponentDatasetResult:
    opponent_policy_id: str
    source_role: str
    dataset: ReplayTrajectoryDataset
    dataset_path: Path
    episodes_jsonl: Path
    bundle_paths: tuple[Path, ...]
    games: int
    wins: int
    losses: int
    draws: int
    truncations: int

    @property
    def mean(self) -> float | None:
        return None if self.games <= 0 else float(self.wins) / float(self.games)


def normalize_include_outcomes(values: Iterable[str] | None) -> tuple[str, ...]:
    outcomes = tuple(str(value).strip().upper() for value in (values or ("W",)) if str(value).strip())
    if any(value == "ALL" for value in outcomes):
        return ()
    invalid = sorted(set(outcomes) - {"W", "L", "D", "T"})
    if invalid:
        raise ValueError(f"include outcomes must be W/L/D/T or ALL, got: {', '.join(invalid)}")
    return outcomes or ("W",)


def normalize_explicit_paired_seeds(values: Iterable[int | str] | None) -> tuple[int, ...]:
    seeds: list[int] = []
    seen: set[int] = set()
    for value in values or ():
        try:
            seed = int(str(value).strip())
        except ValueError as exc:
            raise ValueError(f"paired seed must be an integer, got {value!r}") from exc
        if seed < 0:
            raise ValueError(f"paired seed must be non-negative, got {seed}")
        if seed not in seen:
            seeds.append(seed)
            seen.add(seed)
    return tuple(seeds)


def source_role_for_policy_id(
    policy_id: str,
    *,
    champion_ids: Iterable[str],
    hard_negative_ids: Iterable[str],
) -> str:
    normalized = str(policy_id).strip()
    hard_negative_set = {str(item).strip() for item in hard_negative_ids if str(item).strip()}
    champion_set = {str(item).strip() for item in champion_ids if str(item).strip()}
    if normalized in hard_negative_set:
        return "hard_negative"
    if normalized in champion_set:
        return "imported_champion"
    return "explicit_opponent"


def slug_policy_id(policy_id: str, *, max_length: int = 96) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(policy_id).strip())
    while "__" in slug:
        slug = slug.replace("__", "_")
    slug = slug.strip("_") or "policy"
    return slug[: int(max_length)].rstrip("_") or "policy"


def build_generation_metadata(
    *,
    focal_policy_id: str,
    stack_config: Path,
    run_dir: Path,
    snapshot_registry_json: Path,
    b1_baseline_run_dir: Path | None,
    paired_seeds: Sequence[int],
    include_outcomes: Sequence[str],
    champion_ids: Sequence[str],
    hard_negative_ids: Sequence[str],
    opponent_results: Sequence[OpponentDatasetResult],
) -> dict[str, Any]:
    return {
        "kind": _SCRIPT_KIND,
        "focal_policy_id": str(focal_policy_id),
        "stack_config": Path(stack_config).as_posix(),
        "run_dir": Path(run_dir).as_posix(),
        "snapshot_registry_json": Path(snapshot_registry_json).as_posix(),
        "b1_baseline_run_dir": None if b1_baseline_run_dir is None else Path(b1_baseline_run_dir).as_posix(),
        "paired_seed_count": len(tuple(paired_seeds)),
        "paired_seeds": [int(seed) for seed in paired_seeds],
        "include_outcomes": list(include_outcomes),
        "champion_snapshots": [str(policy_id) for policy_id in champion_ids],
        "hard_negative_policy_ids": [str(policy_id) for policy_id in hard_negative_ids],
        "opponents": [
            {
                "opponent_policy_id": result.opponent_policy_id,
                "source_role": result.source_role,
                "dataset_path": result.dataset_path.as_posix(),
                "episodes_jsonl": result.episodes_jsonl.as_posix(),
                "bundle_count": len(result.bundle_paths),
                "games": int(result.games),
                "wins": int(result.wins),
                "losses": int(result.losses),
                "draws": int(result.draws),
                "truncations": int(result.truncations),
                "mean": result.mean,
                "train_rows": int(result.dataset.metadata.get("train_rows", 0)),
            }
            for result in opponent_results
        ],
    }


def build_champion_hardneg_trajectory_bc_dataset(
    *,
    stack: StackConfig,
    contract: SimulatorContract,
    stack_config: Path,
    run_dir: Path,
    output_run_dir: Path,
    output_dataset: Path,
    snapshot_registry_json: Path,
    focal_policy_id: str,
    opponent_policy_ids: Sequence[str],
    paired_seed_count: int,
    include_outcomes: Sequence[str],
    b1_baseline_run_dir: Path | None = None,
    hard_negative_policy_ids: Sequence[str] = (),
    seed_set_name: str = "report_eval",
    explicit_paired_seeds: Sequence[int] | None = None,
) -> tuple[ReplayTrajectoryDataset, dict[str, Any]]:
    """Capture replay bundles and collate focal winning trajectories into one dataset."""

    if not opponent_policy_ids:
        raise ValueError("opponent_policy_ids must contain at least one policy id")

    paired_seeds = normalize_explicit_paired_seeds(explicit_paired_seeds)
    if not paired_seeds:
        if paired_seed_count <= 0:
            raise ValueError("paired_seed_count must be positive")
        if seed_set_name not in stack.seed_sets:
            raise KeyError(f"seed set not found in stack config: {seed_set_name}")
        paired_seeds = tuple(parse_seed_file(stack.seed_sets[seed_set_name])[: int(paired_seed_count)])
        if len(paired_seeds) < int(paired_seed_count):
            raise ValueError(f"requested {paired_seed_count} paired seeds, found {len(paired_seeds)}")

    output_layout = ArtifactLayout.from_run_dir(output_run_dir)
    output_layout.ensure_directories()
    audit_dir = output_run_dir / "audit"
    datasets_dir = output_run_dir / "datasets"
    audit_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)

    registry = SnapshotRegistry.load(snapshot_registry_json)
    champion_ids = tuple(str(policy_id) for policy_id in registry.champion_snapshots)
    policy_ids = tuple(dict.fromkeys([str(focal_policy_id), *[str(item) for item in opponent_policy_ids]]))
    policies = resolve_eval_policies(
        stack=stack,
        policy_ids=list(policy_ids),
        run_dir=run_dir,
        observation_dim=int(contract.spec_bundle["observation"]["obs_len"]),
        action_dim=int(contract.spec_bundle["action"]["action_space_size"]),
        spec_bundle=contract.spec_bundle,
        snapshot_registry_path=snapshot_registry_json,
        b1_baseline_run_dir=b1_baseline_run_dir,
    )
    runner = SimulatorEvalRunner(
        stack=stack,
        policies=policies,
        artifact_layout=output_layout,
        run_id256=_dataset_run_id256(
            focal_policy_id=focal_policy_id,
            opponent_policy_ids=opponent_policy_ids,
            paired_seeds=paired_seeds,
            output_run_dir=output_run_dir,
        ),
        spec_hash256=contract.spec_hash256,
        action_dim=int(contract.spec_bundle["action"]["action_space_size"]),
        pass_action_id=int(contract.spec_bundle["action"]["pass_action_id"]),
        require_sorted_legal_ids=bool(stack.config.evaluation.eval_assert_sorted_legal_ids),
        replay_capture_rate=1.0,
        regression_capture_count=max(1, len(opponent_policy_ids) * len(paired_seeds) * 2),
    )

    opponent_results: list[OpponentDatasetResult] = []
    config_hash256 = compute_config_hash256(stack)
    for opponent_policy_id in opponent_policy_ids:
        result = _build_one_opponent_dataset(
            stack=stack,
            runner=runner,
            contract=contract,
            stack_config=stack_config,
            run_dir=run_dir,
            output_run_dir=output_run_dir,
            audit_dir=audit_dir,
            datasets_dir=datasets_dir,
            focal_policy_id=str(focal_policy_id),
            opponent_policy_id=str(opponent_policy_id),
            paired_seeds=paired_seeds,
            include_outcomes=include_outcomes,
            config_hash256=config_hash256,
            champion_ids=champion_ids,
            hard_negative_policy_ids=tuple(hard_negative_policy_ids),
        )
        opponent_results.append(result)

    if len(opponent_results) == 1:
        merged = opponent_results[0].dataset
    else:
        merged = merge_replay_trajectory_bc_datasets(
            [result.dataset for result in opponent_results],
            source_labels=[result.opponent_policy_id for result in opponent_results],
        )
    generation_metadata = build_generation_metadata(
        focal_policy_id=focal_policy_id,
        stack_config=stack_config,
        run_dir=run_dir,
        snapshot_registry_json=snapshot_registry_json,
        b1_baseline_run_dir=b1_baseline_run_dir,
        paired_seeds=paired_seeds,
        include_outcomes=include_outcomes,
        champion_ids=champion_ids,
        hard_negative_ids=tuple(hard_negative_policy_ids),
        opponent_results=opponent_results,
    )
    merged.metadata["champion_hardneg_generation"] = generation_metadata
    merged.metadata["source_roles"] = [
        {"opponent_policy_id": result.opponent_policy_id, "source_role": result.source_role}
        for result in opponent_results
    ]
    output_dataset.parent.mkdir(parents=True, exist_ok=True)
    return merged, {"generation": generation_metadata, "dataset_path": output_dataset.as_posix()}


def _build_one_opponent_dataset(
    *,
    stack: StackConfig,
    runner: SimulatorEvalRunner,
    contract: SimulatorContract,
    stack_config: Path,
    run_dir: Path,
    output_run_dir: Path,
    audit_dir: Path,
    datasets_dir: Path,
    focal_policy_id: str,
    opponent_policy_id: str,
    paired_seeds: Sequence[int],
    include_outcomes: Sequence[str],
    config_hash256: str,
    champion_ids: Sequence[str],
    hard_negative_policy_ids: Sequence[str],
) -> OpponentDatasetResult:
    opponent_slug = slug_policy_id(opponent_policy_id)
    opponent_dir = audit_dir / opponent_slug
    bundle_dir = opponent_dir / "replay_bundles"
    opponent_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    schedule = build_seat_swapped_schedule(
        focal_policy_id=focal_policy_id,
        opponent_policy_id=opponent_policy_id,
        paired_seeds=paired_seeds,
    )
    records: list[EvalGameRecord] = []
    bundle_paths: list[Path] = []
    for scheduled_game in schedule:
        result = runner.run_game(scheduled_game)
        if result.replay_sample is None:
            raise RuntimeError(f"replay capture did not produce a bundle for {opponent_policy_id}")
        records.append(
            record_completed_game(
                scheduled_game=scheduled_game,
                result=result,
                run_id256=runner.run_id256_bytes,
                config_hash256=config_hash256,
                spec_hash256=contract.spec_hash256,
            )
        )
        bundle_paths.append(
            _copy_bundle_with_pair_swap(
                output_run_dir=output_run_dir,
                source_bundle_path=result.replay_sample.bundle_path,
                bundle_dir=bundle_dir,
                pair_index=int(scheduled_game.pair_index),
                swap_index=int(scheduled_game.swap_index),
            )
        )

    episodes_jsonl = opponent_dir / "episodes.jsonl"
    write_episodes_jsonl(episodes_jsonl, records)
    matchup_summary = _records_summary(records)
    write_matchup_summary_json(opponent_dir / "matchup_summary.json", matchup_summary)
    dataset = build_replay_trajectory_bc_dataset(
        bundle_paths=bundle_paths,
        run_dir=run_dir,
        stack=stack,
        episodes_jsonl=episodes_jsonl,
        include_outcomes=include_outcomes,
    )
    source_role = source_role_for_policy_id(
        opponent_policy_id,
        champion_ids=champion_ids,
        hard_negative_ids=hard_negative_policy_ids,
    )
    dataset.metadata["opponent_policy_id"] = opponent_policy_id
    dataset.metadata["source_role"] = source_role
    dataset.metadata["paired_seed_count"] = len(tuple(paired_seeds))
    dataset.metadata["episodes_jsonl"] = episodes_jsonl.as_posix()
    dataset.metadata["source_matchup_summary"] = matchup_summary
    dataset_path = datasets_dir / f"trajectory_bc_{opponent_slug}.npz"
    save_replay_trajectory_bc_dataset(dataset_path, dataset)
    return OpponentDatasetResult(
        opponent_policy_id=opponent_policy_id,
        source_role=source_role,
        dataset=dataset,
        dataset_path=dataset_path,
        episodes_jsonl=episodes_jsonl,
        bundle_paths=tuple(bundle_paths),
        games=int(matchup_summary["games"]),
        wins=int(matchup_summary["wins"]),
        losses=int(matchup_summary["losses"]),
        draws=int(matchup_summary["draws"]),
        truncations=int(matchup_summary["truncations"]),
    )


def _copy_bundle_with_pair_swap(
    *,
    output_run_dir: Path,
    source_bundle_path: str | Path,
    bundle_dir: Path,
    pair_index: int,
    swap_index: int,
) -> Path:
    source_path = Path(source_bundle_path)
    if not source_path.is_absolute():
        source_path = output_run_dir / source_path
    if not source_path.is_file():
        raise FileNotFoundError(f"replay bundle not found: {source_path}")
    copied_path = bundle_dir / f"{source_path.stem}_pair{int(pair_index):03d}_swap{int(swap_index)}{source_path.suffix}"
    shutil.copy2(source_path, copied_path)
    return copied_path


def _records_summary(records: Sequence[EvalGameRecord]) -> dict[str, Any]:
    wins = 0
    losses = 0
    draws = 0
    truncations = 0
    for record in records:
        outcome = str(record.outcome).strip().upper()
        if outcome == "T" or bool(record.truncated):
            truncations += 1
        elif outcome == "D":
            draws += 1
        elif outcome == "W":
            wins += 1
        else:
            losses += 1
    games = len(records)
    return {
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "truncations": truncations,
        "mean": None if games <= 0 else wins / games,
    }


def _dataset_run_id256(
    *,
    focal_policy_id: str,
    opponent_policy_ids: Sequence[str],
    paired_seeds: Sequence[int],
    output_run_dir: Path,
) -> str:
    return sha256_hex(
        canonical_json_bytes(
            {
                "kind": _SCRIPT_KIND,
                "focal_policy_id": str(focal_policy_id),
                "opponent_policy_ids": [str(policy_id) for policy_id in opponent_policy_ids],
                "paired_seeds": [int(seed) for seed in paired_seeds],
                "output_run_dir": Path(output_run_dir).resolve().as_posix(),
            }
        )
    )


def write_dataset_summary(path: Path, summary: Mapping[str, Any], dataset: ReplayTrajectoryDataset) -> None:
    payload = dict(summary)
    payload["dataset_metadata"] = dataset.metadata
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "OpponentDatasetResult",
    "build_champion_hardneg_trajectory_bc_dataset",
    "build_generation_metadata",
    "normalize_explicit_paired_seeds",
    "normalize_include_outcomes",
    "slug_policy_id",
    "source_role_for_policy_id",
    "write_dataset_summary",
]
