"""Source matchup and policy identity helpers for B2 disagreement audits."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from weiss_rl.artifacts.reproducibility import canonical_json_bytes, sha256_hex
from weiss_rl.eval.analysis.export import load_eval_game_records


@dataclass(frozen=True, slots=True)
class MatchupSource:
    focal_policy_id: str
    opponent_policy_id: str
    config_hash256: str
    spec_hash256: str
    paired_seeds: tuple[int, ...]


def load_matchup_source(path: Path) -> MatchupSource:
    records = load_eval_game_records(path)
    focal_policy_ids = {record.focal_policy_id for record in records}
    opponent_policy_ids = {record.opponent_policy_id for record in records}
    config_hashes = {record.config_hash256 for record in records}
    spec_hashes = {record.spec_hash256 for record in records}
    if len(focal_policy_ids) != 1 or len(opponent_policy_ids) != 1:
        raise ValueError("source episodes.jsonl must contain exactly one focal/opponent matchup")
    if len(config_hashes) != 1 or len(spec_hashes) != 1:
        raise ValueError("source episodes.jsonl must contain exactly one config/spec contract")

    pair_to_seed: dict[int, int] = {}
    pair_order: list[int] = []
    pair_swaps: dict[int, set[int]] = defaultdict(set)
    for record in records:
        pair_index = int(record.pair_index)
        swap_index = int(record.swap_index)
        if swap_index not in (0, 1):
            raise ValueError(f"source episodes.jsonl must be seat-swapped, got swap_index={swap_index}")
        pair_swaps[pair_index].add(swap_index)
        if pair_index not in pair_to_seed:
            pair_order.append(pair_index)
            pair_to_seed[pair_index] = int(record.episode_seed)
        elif pair_to_seed[pair_index] != int(record.episode_seed):
            raise ValueError(f"pair_index {pair_index} uses inconsistent episode_seed values")

    for pair_index, swaps in pair_swaps.items():
        if swaps != {0, 1}:
            raise ValueError(f"pair_index {pair_index} must contain swap_index 0 and 1")

    paired_seeds: list[int] = []
    seen_seeds: set[int] = set()
    for pair_index in pair_order:
        seed = pair_to_seed[pair_index]
        if seed in seen_seeds:
            continue
        seen_seeds.add(seed)
        paired_seeds.append(seed)

    if not paired_seeds:
        raise ValueError(f"no paired seeds found in {path}")

    return MatchupSource(
        focal_policy_id=next(iter(focal_policy_ids)),
        opponent_policy_id=next(iter(opponent_policy_ids)),
        config_hash256=next(iter(config_hashes)),
        spec_hash256=next(iter(spec_hashes)),
        paired_seeds=tuple(paired_seeds),
    )


def resolve_source_config_hash(
    *,
    source_config_hash256: str,
    stack_config_hash256: str,
    run_dir: Path,
) -> tuple[str, str | None]:
    run_manifest_config_hash256 = manifest_config_hash(run_dir)
    accepted_hashes = {stack_config_hash256}
    if run_manifest_config_hash256:
        accepted_hashes.add(run_manifest_config_hash256)
    if source_config_hash256 not in accepted_hashes:
        message = (
            "stack config hash does not match the source episodes.jsonl "
            f"(expected {source_config_hash256!r}, got {stack_config_hash256!r}"
        )
        if run_manifest_config_hash256:
            message += f", run manifest {run_manifest_config_hash256!r}"
        message += ")"
        raise ValueError(message)
    return source_config_hash256, run_manifest_config_hash256


def manifest_config_hash(run_dir: Path) -> str | None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    config_hash256 = payload.get("config_hash256")
    if isinstance(config_hash256, str) and config_hash256.strip():
        return config_hash256.strip()
    return None


def resolve_requested_policy_id(
    *,
    requested_policy_id: str,
    source_focal_policy_id: str,
    allow_mismatch: bool = False,
) -> str | None:
    normalized_requested = requested_policy_id.strip()
    if not normalized_requested:
        return None
    if allow_mismatch:
        return normalized_requested
    if normalized_requested == source_focal_policy_id:
        return normalized_requested
    for candidate in candidate_policy_aliases(source_focal_policy_id):
        if normalized_requested == candidate:
            return normalized_requested
    return None


def candidate_policy_aliases(source_focal_policy_id: str) -> tuple[str, ...]:
    normalized = source_focal_policy_id.strip()
    aliases: list[str] = [normalized]
    if "_p" in normalized:
        suffix = normalized.rsplit("_p", 1)[-1]
        if suffix.isdigit():
            aliases.append(f"policy_{int(suffix):06d}")
    return tuple(dict.fromkeys(alias for alias in aliases if alias))


def run_config_hashes(run_dir: Path | None) -> list[str]:
    if run_dir is None:
        return []
    hashes: list[str] = []
    hash_path = Path(run_dir) / "config_hash256.txt"
    if hash_path.is_file():
        value = hash_path.read_text(encoding="utf-8").strip()
        if value:
            hashes.append(value)
    manifest_path = Path(run_dir) / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        value = str(manifest.get("config_hash256", "")).strip() if isinstance(manifest, dict) else ""
        if value:
            hashes.append(value)
    return list(dict.fromkeys(hashes))


def audit_run_id256(
    *,
    policy_id: str,
    opponent_policy_id: str,
    episodes_jsonl: Path,
    output_run_dir: Path,
    paired_seeds: Sequence[int],
) -> str:
    payload = canonical_json_bytes(
        {
            "kind": "b2_disagreement_audit_v1",
            "policy_id": policy_id,
            "opponent_policy_id": opponent_policy_id,
            "episodes_jsonl": episodes_jsonl.resolve().as_posix(),
            "output_run_dir": output_run_dir.resolve().as_posix(),
            "paired_seeds": list(int(seed) for seed in paired_seeds),
        }
    )
    return sha256_hex(payload)
