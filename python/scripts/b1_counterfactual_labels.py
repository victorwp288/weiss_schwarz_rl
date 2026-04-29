"""Generate small S1 B1 counterfactual labels from replayed forced actions.

This script intentionally reuses ``b1_artifact_matrix.py`` as the execution
engine. It is a first proof tool, not a fast large-scale searcher.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import b1_artifact_matrix as matrix
import torch


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


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _parse_int_set(value: str) -> set[int]:
    result: set[int] = set()
    for item in str(value or "").split(","):
        stripped = item.strip()
        if stripped:
            result.add(int(stripped))
    return result


def _parse_str_set(value: str) -> set[str]:
    return {item.strip() for item in str(value or "").split(",") if item.strip()}


def _parse_action_pairs(value: str) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for item in str(value or "").split(","):
        stripped = item.strip()
        if not stripped:
            continue
        if "=" not in stripped:
            raise SystemExit(f"action pair exclusions must look like baseline=positive, got {stripped!r}")
        left, right = stripped.split("=", 1)
        pairs.add((int(left.strip()), int(right.strip())))
    return pairs


def _load_excluded_label_keys(paths: Sequence[Path]) -> tuple[set[int], set[str], set[tuple[int, int]]]:
    pair_indices: set[int] = set()
    fingerprints: set[str] = set()
    action_pairs: set[tuple[int, int]] = set()
    for path in paths:
        if not path.is_file():
            continue
        label_dir = path.parent
        for row in _read_jsonl(path):
            if row.get("pair_index") is not None:
                pair_indices.add(int(row["pair_index"]))
            fingerprint = row.get("legal_action_fingerprint") or row.get("legal_ids_sha256")
            tensor_record: Mapping[str, Any] = {}
            tensor_ref = row.get("tensor_ref")
            if isinstance(tensor_ref, str) and tensor_ref:
                tensor_path = label_dir / tensor_ref
                if tensor_path.is_file():
                    loaded = torch.load(tensor_path, map_location="cpu", weights_only=False)
                    if isinstance(loaded, Mapping):
                        tensor_record = loaded
            fingerprint = fingerprint or tensor_record.get("legal_action_fingerprint")
            if fingerprint:
                fingerprints.add(str(fingerprint))
            baseline_id = row.get("baseline_action_id", tensor_record.get("baseline_action_id"))
            positive_id = row.get("positive_action_id", tensor_record.get("positive_action_id"))
            try:
                if baseline_id is not None and positive_id is not None:
                    action_pairs.add(
                        (int(torch.as_tensor(baseline_id).item()), int(torch.as_tensor(positive_id).item()))
                    )
            except (TypeError, ValueError, RuntimeError):
                pass
    return pair_indices, fingerprints, action_pairs


def _sha256_tensor(tensor: torch.Tensor) -> str:
    arr = tensor.detach().cpu().contiguous().numpy()
    import hashlib

    return hashlib.sha256(arr.tobytes()).hexdigest()


def _matrix_script_path() -> Path:
    return Path(__file__).with_name("b1_artifact_matrix.py")


@dataclass(frozen=True)
class _InProcessMatrixContext:
    stack: Any
    run_dir: Path
    b1_baseline_run_dir: Path
    manifest: Mapping[str, Any]
    spec_hash: str
    spec_bundle: Mapping[str, Any]
    paired_seeds: Sequence[int]
    evaluation: Any
    action_dim: int
    pass_action_id: int
    artifact_layout: Any
    action_catalog: Any
    policies: Mapping[str, Any]
    source_paths: Mapping[str, Path | None]
    device: str
    surface_name: str
    public_heuristic_bias_scale: float
    checkpoint_policies: Sequence[str]
    include_builtin: Sequence[str] = ()


def _build_in_process_context(
    *,
    stack_config: Path,
    run_dir: Path,
    b1_baseline_run_dir: Path,
    checkpoint_policies: Sequence[str],
    pairs: int,
    seed_scope: str,
    device: str,
    public_heuristic_bias_scale: float,
    include_builtin: Sequence[str] = (),
) -> _InProcessMatrixContext:
    stack = matrix.train_script.load_stack_config(stack_config)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runtime_spec = manifest.get("runtime_spec")
    spec_hash = str(runtime_spec.get("sha256") or "") if isinstance(runtime_spec, dict) else ""
    spec_hash = spec_hash or matrix._manifest_value(manifest, "spec_hash256", "spec_hash")
    contract = matrix.train_script.load_verified_simulator_contract(stack.root, expected_spec_hash=spec_hash)
    seed_file, validated_sources, base_seeds, seed_sha = matrix.train_script._periodic_dev_eval_schedule(stack)
    paired_seeds = matrix.train_script._expand_periodic_dev_eval_paired_seeds(
        base_seeds,
        requested_pairs=int(pairs),
        seed_file_sha256=seed_sha,
        update_count=0,
        policy_version=0,
        scope=str(seed_scope),
    )
    matrix.train_script._validate_periodic_dev_eval_contract(stack)
    evaluation = matrix.train_script._evaluation_config_or_raise(stack)
    observation_dim, action_dim = matrix.train_script._spec_dimensions(contract)
    spec_bundle = contract.spec_bundle
    observation_spec = spec_bundle.get("observation")
    action_catalog = matrix.ActionCatalog.from_spec_bundle(spec_bundle)
    artifact_layout = matrix.ArtifactLayout.from_run_dir(run_dir)

    policies: dict[str, Any] = {}
    source_paths: dict[str, Path | None] = {}
    b1_resolved = matrix.resolve_eval_policies(
        stack=stack,
        policy_ids=[matrix.NO_LEAGUE_POLICY_ID],
        run_dir=run_dir,
        observation_dim=observation_dim,
        action_dim=action_dim,
        spec_bundle=spec_bundle,
        b1_baseline_run_dir=b1_baseline_run_dir,
        eval_device=device,
    )[matrix.NO_LEAGUE_POLICY_ID]
    policies[matrix.NO_LEAGUE_POLICY_ID] = b1_resolved
    source_paths[matrix.NO_LEAGUE_POLICY_ID] = (
        None
        if b1_resolved.source_run_dir is None or b1_resolved.snapshot_path is None
        else Path(b1_resolved.source_run_dir) / b1_resolved.snapshot_path
    )

    builtin_policy_ids: list[str] = []
    for policy_id in include_builtin:
        canonical = matrix._canonical_builtin_policy_id(str(policy_id))
        if canonical == matrix.NO_LEAGUE_POLICY_ID:
            continue
        if canonical not in builtin_policy_ids:
            builtin_policy_ids.append(canonical)
    if builtin_policy_ids:
        resolved_builtins = matrix.resolve_eval_policies(
            stack=stack,
            policy_ids=builtin_policy_ids,
            run_dir=run_dir,
            observation_dim=observation_dim,
            action_dim=action_dim,
            spec_bundle=spec_bundle,
            b1_baseline_run_dir=b1_baseline_run_dir,
            eval_device=device,
        )
        for policy_id in builtin_policy_ids:
            policies[policy_id] = resolved_builtins[policy_id]
            source_paths[policy_id] = None

    for item in checkpoint_policies:
        alias, sep, checkpoint = str(item).partition("=")
        if not sep:
            raise SystemExit(f"--checkpoint-policy must be alias=checkpoint.pt, got {item!r}")
        checkpoint_path = Path(checkpoint).resolve()
        model = matrix.train_script._load_checkpoint_eval_model(
            checkpoint_path=checkpoint_path,
            observation_dim=observation_dim,
            action_dim=action_dim,
            stack=stack,
            eval_device=device,
            observation_spec=observation_spec if isinstance(observation_spec, dict) else None,
            spec_bundle=spec_bundle if isinstance(spec_bundle, dict) else None,
        )
        policies[str(alias)] = matrix.ResolvedEvalPolicy(
            policy_id=str(alias),
            kind="checkpoint",
            source_run_dir=checkpoint_path.parent.parent.parent.as_posix(),
            snapshot_path=checkpoint_path.as_posix(),
            model=model,
        )
        source_paths[str(alias)] = checkpoint_path

    matrix._apply_public_heuristic_bias_override(policies, override_scale=float(public_heuristic_bias_scale))
    return _InProcessMatrixContext(
        stack=stack,
        run_dir=run_dir,
        b1_baseline_run_dir=b1_baseline_run_dir,
        manifest=manifest,
        spec_hash=spec_hash,
        spec_bundle=spec_bundle if isinstance(spec_bundle, Mapping) else {},
        paired_seeds=paired_seeds,
        evaluation=evaluation,
        action_dim=action_dim,
        pass_action_id=int(spec_bundle["action"]["pass_action_id"]),
        artifact_layout=artifact_layout,
        action_catalog=action_catalog,
        policies=policies,
        source_paths=source_paths,
        device=device,
        surface_name="lowbias_s1",
        public_heuristic_bias_scale=float(public_heuristic_bias_scale),
        checkpoint_policies=tuple(checkpoint_policies),
        include_builtin=tuple(include_builtin),
    )


def _write_in_process_load_manifest(
    ctx: _InProcessMatrixContext,
    *,
    matrix_dir: Path,
    pairs: int,
    artifact_dir_name: str,
    emit_action_traces: bool,
    trace_tensors: bool,
    force_action: Mapping[str, int] | None,
) -> None:
    payload = {
        "format": "b1_counterfactual_inprocess_load_manifest_v1",
        "run_dir": ctx.run_dir.as_posix(),
        "surface_name": ctx.surface_name,
        "pairs": int(pairs),
        "scoring_mode": "learner",
        "emit_action_traces": bool(emit_action_traces),
        "trace_tensors": bool(trace_tensors),
        "artifact_dir_name": artifact_dir_name,
        "public_heuristic_bias_override_requested": True,
        "public_heuristic_bias_override_scale": float(ctx.public_heuristic_bias_scale),
        "force_action": None if force_action is None else dict(force_action),
        "paired_seeds": [int(seed) for seed in ctx.paired_seeds[: int(pairs)]],
        "policies": {
            policy_id: matrix._policy_manifest_entry(
                policy_id=policy_id,
                policy=policy,
                source_path=ctx.source_paths.get(policy_id),
            )
            for policy_id, policy in ctx.policies.items()
        },
    }
    matrix._write_json(matrix_dir / "policy_load_manifest.json", payload)
    matrix._write_json(
        matrix_dir / "resolved_policies.json", {key: policy.to_manifest_dict() for key, policy in ctx.policies.items()}
    )


def _run_matrix_in_process(
    ctx: _InProcessMatrixContext,
    *,
    artifact_dir_name: str,
    pairs: int,
    emit_action_traces: bool,
    trace_top_k: int,
    trace_max_decisions_per_episode: int,
    trace_tensors: bool = False,
    force_action: Mapping[str, int] | None = None,
) -> Path:
    matrix_dir = ctx.run_dir / "eval" / artifact_dir_name
    matrix_dir.mkdir(parents=True, exist_ok=True)
    _write_in_process_load_manifest(
        ctx,
        matrix_dir=matrix_dir,
        pairs=int(pairs),
        artifact_dir_name=artifact_dir_name,
        emit_action_traces=emit_action_traces,
        trace_tensors=trace_tensors,
        force_action=force_action,
    )
    focal_policy_id = matrix.NO_LEAGUE_POLICY_ID
    opponent_policy_id = matrix.NO_LEAGUE_POLICY_ID
    matchup_dir = matrix_dir / f"{matrix._safe_slug(focal_policy_id)}__vs__{matrix._safe_slug(opponent_policy_id)}"
    matchup_dir.mkdir(parents=True, exist_ok=True)
    trace_path = matchup_dir / "action_trace.jsonl" if emit_action_traces else None
    if trace_path is not None and trace_path.exists():
        trace_path.unlink()
    runner = matrix._MatrixSimulatorEvalRunner(
        stack=ctx.stack,
        policies=ctx.policies,
        artifact_layout=ctx.artifact_layout,
        run_id256=matrix._manifest_value(ctx.manifest, "run_id256", "computed_run_id256"),
        spec_hash256=ctx.spec_hash,
        action_dim=ctx.action_dim,
        pass_action_id=ctx.pass_action_id,
        require_sorted_legal_ids=bool(ctx.evaluation.eval_assert_sorted_legal_ids),
        replay_capture_rate=0.0,
        regression_capture_count=0,
        eval_device=ctx.device,
        spec_bundle=ctx.spec_bundle,
        scoring_mode="learner",
        greedy_policy_ids=[],
        action_rng_salt_mode="shared",
        trace_path=trace_path,
        trace_top_k=int(trace_top_k),
        trace_max_decisions_per_episode=int(trace_max_decisions_per_episode),
        trace_tensors=bool(trace_tensors),
        action_catalog=ctx.action_catalog,
        force_action_seat=None if force_action is None else int(force_action["seat"]),
        force_action_decision_index=None if force_action is None else int(force_action["decision_index"]),
        force_action_id=None if force_action is None else int(force_action["action_id"]),
        force_action_pair_index=None if force_action is None else int(force_action["pair_index"]),
        force_action_swap_index=None if force_action is None else int(force_action["swap_index"]),
    )
    try:
        result = matrix.run_seat_swapped_matchup(
            focal_policy_id=focal_policy_id,
            opponent_policy_id=opponent_policy_id,
            paired_seeds=ctx.paired_seeds[: int(pairs)],
            runner=runner,
            episodes_path=matchup_dir / "episodes.jsonl",
            run_id256=matrix._manifest_value(ctx.manifest, "run_id256", "computed_run_id256"),
            config_hash256=matrix._manifest_value(ctx.manifest, "config_hash256", "config_hash"),
            spec_hash256=ctx.spec_hash,
        )
        runner_counters = runner.matrix_counters()
    finally:
        runner.close()
    matchup_payload = matrix.build_matchup_export(
        list(result.records),
        stop_rules=ctx.evaluation.stop_rules,
        max_paired_seeds=int(pairs),
        scheme=ctx.evaluation.final_policy_set_selection.folding,
        sample_count=1000,
        seed=0,
    )
    seat_diagnostics = matrix.build_seat_advantage_diagnostics(list(result.records))
    pair_rows, pair_summary = matrix._pair_table(
        list(result.records),
        scheme=ctx.evaluation.final_policy_set_selection.folding,
    )
    matchup_payload["seat_diagnostics"] = seat_diagnostics
    matchup_payload["pair_class_summary"] = pair_summary
    matchup_payload["matrix_runner_counters"] = runner_counters
    matchup_payload["evaluation_context"] = {
        "artifact_dir_name": artifact_dir_name,
        "surface_name": ctx.surface_name,
        "focal_policy_id": focal_policy_id,
        "opponent_policy_id": opponent_policy_id,
        "scoring_mode": "learner",
        "public_heuristic_bias_override_requested": True,
        "public_heuristic_bias_override_scale": float(ctx.public_heuristic_bias_scale),
        "force_action": None if force_action is None else dict(force_action),
        "episodes_path": (matchup_dir / "episodes.jsonl").relative_to(ctx.run_dir).as_posix(),
        "pair_table_path": (matchup_dir / "pair_table.jsonl").relative_to(ctx.run_dir).as_posix(),
        "action_trace_path": None if trace_path is None else trace_path.relative_to(ctx.run_dir).as_posix(),
    }
    matrix.write_matchup_summary_json(matchup_dir / "matchup_summary.json", matchup_payload)
    matrix._write_json(matchup_dir / "seat_diagnostics.json", seat_diagnostics)
    matrix._write_json(matchup_dir / "pair_class_summary.json", pair_summary)
    matrix._write_jsonl(matchup_dir / "pair_table.jsonl", pair_rows)
    matrix_summary = {
        "format": "b1_counterfactual_inprocess_matrix_summary_v1",
        "surface_name": ctx.surface_name,
        "scoring_mode": "learner",
        "public_heuristic_bias_override_requested": True,
        "public_heuristic_bias_override_scale": float(ctx.public_heuristic_bias_scale),
        "policy_ids": list(ctx.policies),
        "pairs": int(pairs),
        "matchups": {
            f"{focal_policy_id}__vs__{opponent_policy_id}": {
                "matchup_dir": matchup_dir.relative_to(ctx.run_dir).as_posix(),
                "mean": matchup_payload["uncertainty"]["mean"],
                "wins": matchup_payload["summary"]["wins"],
                "losses": matchup_payload["summary"]["losses"],
                "draws": matchup_payload["summary"]["draws"],
                "truncations": matchup_payload["summary"]["truncations"],
                "pair_class_summary": pair_summary,
                "seat_diagnostics": seat_diagnostics,
                "matrix_runner_counters": runner_counters,
            }
        },
        "complement_checks": [],
    }
    matrix._write_json(matrix_dir / "matrix_summary.json", matrix_summary)
    print(
        f"{focal_policy_id} vs {opponent_policy_id}: "
        f"mean={matchup_payload['uncertainty']['mean']} "
        f"wins={matchup_payload['summary']['wins']} losses={matchup_payload['summary']['losses']} "
        f"pair_classes={pair_summary['pair_class_counts']}"
    )
    return matrix_dir


def _run_forced_episode_in_process(
    ctx: _InProcessMatrixContext,
    *,
    force_action: Mapping[str, int],
    force_action_sequence: Sequence[Mapping[str, int]] = (),
    trace_path: Path | None = None,
    trace_top_k: int = 8,
    trace_max_decisions_per_episode: int = 0,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    pair_index = int(force_action["pair_index"])
    swap_index = int(force_action["swap_index"])
    focal_policy_id = matrix.NO_LEAGUE_POLICY_ID
    opponent_policy_id = matrix.NO_LEAGUE_POLICY_ID
    focal_seat = 0 if swap_index == 0 else 1
    scheduled_game = matrix.ScheduledGame(
        pair_index=pair_index,
        swap_index=swap_index,
        episode_index=pair_index * 2 + swap_index,
        episode_seed=int(ctx.paired_seeds[pair_index]),
        focal_policy_id=focal_policy_id,
        opponent_policy_id=opponent_policy_id,
        seat0_policy_id=focal_policy_id if focal_seat == 0 else opponent_policy_id,
        seat1_policy_id=opponent_policy_id if focal_seat == 0 else focal_policy_id,
        focal_seat=focal_seat,
    )
    runner = matrix._MatrixSimulatorEvalRunner(
        stack=ctx.stack,
        policies=ctx.policies,
        artifact_layout=ctx.artifact_layout,
        run_id256=matrix._manifest_value(ctx.manifest, "run_id256", "computed_run_id256"),
        spec_hash256=ctx.spec_hash,
        action_dim=ctx.action_dim,
        pass_action_id=ctx.pass_action_id,
        require_sorted_legal_ids=bool(ctx.evaluation.eval_assert_sorted_legal_ids),
        replay_capture_rate=0.0,
        regression_capture_count=0,
        eval_device=ctx.device,
        spec_bundle=ctx.spec_bundle,
        scoring_mode="learner",
        greedy_policy_ids=[],
        action_rng_salt_mode="shared",
        trace_path=trace_path,
        trace_top_k=int(trace_top_k),
        trace_max_decisions_per_episode=int(trace_max_decisions_per_episode),
        action_catalog=ctx.action_catalog,
        force_action_seat=None if force_action_sequence else int(force_action["seat"]),
        force_action_decision_index=None if force_action_sequence else int(force_action["decision_index"]),
        force_action_id=None if force_action_sequence else int(force_action["action_id"]),
        force_action_pair_index=None if force_action_sequence else pair_index,
        force_action_swap_index=None if force_action_sequence else swap_index,
        force_action_sequence=force_action_sequence,
    )
    try:
        result = runner.run_game(scheduled_game)
        counters = runner.matrix_counters()
    finally:
        runner.close()
    if result.winner_seat is None:
        outcome = "D"
    elif int(result.winner_seat) == int(focal_seat):
        outcome = "W"
    else:
        outcome = "L"
    episode = {
        "pair_index": pair_index,
        "swap_index": swap_index,
        "episode_seed": int(scheduled_game.episode_seed),
        "focal_seat": focal_seat,
        "outcome": outcome,
        "winner_seat": None if result.winner_seat is None else int(result.winner_seat),
        "seat0_policy_id": scheduled_game.seat0_policy_id,
        "seat1_policy_id": scheduled_game.seat1_policy_id,
        "terminated": bool(result.terminated),
        "truncated": bool(result.truncated),
        "engine_status": int(result.engine_status),
        "decision_count": int(result.decision_count),
        "tick_count": int(result.tick_count),
        "termination_reason": result.termination_reason,
        "terminal_summary": result.terminal_summary,
    }
    return episode, counters


def _run_matrix(
    *,
    stack_config: Path,
    run_dir: Path,
    b1_baseline_run_dir: Path,
    checkpoint_policies: Sequence[str],
    artifact_dir_name: str,
    pairs: int,
    seed_scope: str,
    device: str,
    public_heuristic_bias_scale: float,
    emit_action_traces: bool,
    trace_top_k: int,
    trace_max_decisions_per_episode: int,
    trace_tensors: bool = False,
    force_action: Mapping[str, int] | None = None,
) -> Path:
    cmd = [
        sys.executable,
        str(_matrix_script_path()),
        "--stack-config",
        str(stack_config),
        "--run-dir",
        str(run_dir),
        "--b1-baseline-run-dir",
        str(b1_baseline_run_dir),
        "--pairs",
        str(int(pairs)),
        "--artifact-dir-name",
        artifact_dir_name,
        "--surface-name",
        "lowbias_s1",
        "--device",
        device,
        "--scoring-mode",
        "learner",
        "--public-heuristic-bias-scale",
        str(float(public_heuristic_bias_scale)),
        "--seed-scope",
        seed_scope,
        "--matchup",
        "B1=B1",
    ]
    for item in checkpoint_policies:
        cmd.extend(["--checkpoint-policy", item])
    if emit_action_traces:
        cmd.append("--emit-action-traces")
        if trace_tensors:
            cmd.append("--emit-trace-tensors")
        cmd.extend(["--trace-top-k", str(int(trace_top_k))])
        cmd.extend(["--trace-max-decisions-per-episode", str(int(trace_max_decisions_per_episode))])
    if force_action is not None:
        cmd.extend(["--force-action-pair-index", str(int(force_action["pair_index"]))])
        cmd.extend(["--force-action-swap-index", str(int(force_action["swap_index"]))])
        cmd.extend(["--force-action-seat", str(int(force_action["seat"]))])
        cmd.extend(["--force-action-decision-index", str(int(force_action["decision_index"]))])
        cmd.extend(["--force-action-id", str(int(force_action["action_id"]))])
    subprocess.run(cmd, cwd=Path(__file__).resolve().parents[2], check=True)
    return run_dir / "eval" / artifact_dir_name


def _first_matchup_dir(matrix_dir: Path) -> Path:
    summary = _read_json(matrix_dir / "matrix_summary.json")
    matchups = summary.get("matchups")
    if not isinstance(matchups, dict) or not matchups:
        raise RuntimeError(f"matrix summary has no matchups: {matrix_dir}")
    first = next(iter(matchups.values()))
    matchup_dir = first.get("matchup_dir")
    if not isinstance(matchup_dir, str):
        raise RuntimeError(f"matrix summary matchup has no directory: {matrix_dir}")
    return matrix_dir.parents[1] / matchup_dir


def _episode_payload_for_swap(pair_row: Mapping[str, Any], swap_index: int) -> Mapping[str, Any]:
    key = "focal_as_seat0" if int(swap_index) == 0 else "focal_as_seat1"
    payload = pair_row.get(key)
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"pair row missing {key}")
    return payload


def _build_episode_baselines(pair_rows: Sequence[Mapping[str, Any]]) -> dict[tuple[int, int], Mapping[str, Any]]:
    by_episode: dict[tuple[int, int], Mapping[str, Any]] = {}
    for row in pair_rows:
        pair_index = int(row["pair_index"])
        for swap_index in (0, 1):
            by_episode[(pair_index, swap_index)] = _episode_payload_for_swap(row, swap_index)
    return by_episode


def _action_id(payload: Mapping[str, Any]) -> int | None:
    action_id = payload.get("action_id")
    return None if action_id is None else int(action_id)


def _candidate_rep_key(action_id: int, *, action_catalog: Any | None) -> tuple[Any, ...]:
    if action_catalog is None:
        return ("action", int(action_id))
    try:
        decoded = action_catalog.decode(int(action_id))
    except Exception:
        return ("action", int(action_id))
    if decoded.family == "main_play_character":
        return (decoded.family, decoded.stage_slot)
    if decoded.family == "main_move":
        return (decoded.family, decoded.from_slot, decoded.to_slot)
    if decoded.family == "attack":
        return (decoded.family, decoded.slot, decoded.attack_type)
    if decoded.family in {"encore_pay", "encore_decline"}:
        return (decoded.family, decoded.slot)
    return (decoded.family,)


def _candidate_family(action_id: int, *, action_catalog: Any | None) -> str:
    if action_catalog is None:
        return ""
    try:
        return str(action_catalog.decode(int(action_id)).family)
    except Exception:
        return ""


def _candidate_action_payload(action_id: int, *, action_catalog: Any | None) -> dict[str, Any]:
    return matrix._action_payload(int(action_id), action_catalog=action_catalog)


def _candidate_actions(
    trace_row: Mapping[str, Any],
    *,
    max_actions: int,
    pass_action_id: int | None,
    action_catalog: Any | None,
    family_representatives_per_family: int,
) -> list[dict[str, Any]]:
    selected = _action_id(
        trace_row.get("selected_action", {}) if isinstance(trace_row.get("selected_action"), Mapping) else {}
    )
    legal_ids = trace_row.get("legal_ids")
    legal_set = {int(item) for item in legal_ids} if isinstance(legal_ids, Sequence) else set()
    candidates: dict[int, dict[str, Any]] = {}

    def add_candidate(action_id: int, *, source: str, rank: int = 0, logit: float | None = None) -> None:
        action_id = int(action_id)
        if selected is not None and action_id == int(selected):
            return
        if legal_set and action_id not in legal_set:
            return
        existing = candidates.setdefault(
            action_id,
            {
                **_candidate_action_payload(action_id, action_catalog=action_catalog),
                "candidate_sources": [],
                "candidate_source_ranks": {},
                "candidate_logits": {},
            },
        )
        if source not in existing["candidate_sources"]:
            existing["candidate_sources"].append(source)
        existing["candidate_source_ranks"][source] = int(rank)
        if logit is not None:
            existing["candidate_logits"][source] = float(logit)

    if (
        pass_action_id is not None
        and isinstance(legal_ids, Sequence)
        and int(pass_action_id) in {int(item) for item in legal_ids}
    ):
        add_candidate(int(pass_action_id), source="pass_alternative")
    for key in ("raw_topk_no_public_bias", "final_topk"):
        items = trace_row.get(key)
        if not isinstance(items, Sequence):
            continue
        for rank, item in enumerate(items):
            if isinstance(item, Mapping):
                action_id = _action_id(item)
                if action_id is not None:
                    logit_value: float | None = None
                    try:
                        logit_value = float(item["logit"])
                    except (KeyError, TypeError, ValueError):
                        logit_value = None
                    add_candidate(action_id, source=key, rank=int(rank), logit=logit_value)
    if isinstance(legal_ids, Sequence):
        by_rep: dict[tuple[Any, ...], int] = {}
        family_counts: dict[str, int] = {}
        for action_id_raw in legal_ids:
            action_id = int(action_id_raw)
            if selected is not None and action_id == int(selected):
                continue
            rep_key = _candidate_rep_key(action_id, action_catalog=action_catalog)
            family = _candidate_family(action_id, action_catalog=action_catalog)
            if rep_key in by_rep:
                continue
            if family and family_counts.get(family, 0) >= int(family_representatives_per_family):
                continue
            by_rep[rep_key] = action_id
            family_counts[family] = family_counts.get(family, 0) + 1
            add_candidate(action_id, source="family_representative")

    source_priority = {
        "pass_alternative": 0,
        "final_topk": 1,
        "raw_topk_no_public_bias": 2,
        "family_representative": 3,
    }

    def sort_key(item: Mapping[str, Any]) -> tuple[int, int, int]:
        sources = item.get("candidate_sources", [])
        ranks = item.get("candidate_source_ranks", {})
        best_source_score = min(source_priority.get(str(source), 99) for source in sources) if sources else 99
        best_rank = (
            min(int(ranks.get(str(source), 999)) for source in sources)
            if isinstance(ranks, Mapping) and sources
            else 999
        )
        return (best_source_score, best_rank, int(item["action_id"]))

    unique = sorted(candidates.values(), key=sort_key)
    if int(max_actions) > 0:
        unique = unique[: int(max_actions)]
    return [dict(item) for item in unique]


def _candidate_action_ids(
    trace_row: Mapping[str, Any],
    *,
    max_actions: int,
    pass_action_id: int | None,
    action_catalog: Any | None,
    family_representatives_per_family: int,
) -> list[int]:
    return [
        int(item["action_id"])
        for item in _candidate_actions(
            trace_row,
            max_actions=max_actions,
            pass_action_id=pass_action_id,
            action_catalog=action_catalog,
            family_representatives_per_family=family_representatives_per_family,
        )
    ]


def _top_logit_margin(items: Any) -> float | None:
    if not isinstance(items, Sequence) or len(items) < 2:
        return None
    first, second = items[0], items[1]
    if not isinstance(first, Mapping) or not isinstance(second, Mapping):
        return None
    try:
        return float(first["logit"]) - float(second["logit"])
    except (KeyError, TypeError, ValueError):
        return None


def _target_rank_score(row: Mapping[str, Any], episode: Mapping[str, Any], selected_family: str) -> float:
    score = 0.0
    family_bonus = {
        "level_up": 35.0,
        "climax_play": 32.0,
        "main_play_event": 30.0,
        "attack": 28.0,
        "main_play_character": 24.0,
        "main_move": 20.0,
        "clock_from_hand": 16.0,
        "pass": 16.0,
        "encore_pay": 12.0,
        "encore_decline": 10.0,
    }
    score += family_bonus.get(selected_family, 0.0)

    decision_index = int(row.get("decision_index", 0))
    decision_count = int(episode.get("decision_count", 0) or 0)
    if decision_count > 0:
        progress = max(0.0, min(1.0, decision_index / decision_count))
        remaining = max(0, decision_count - decision_index)
        score += 20.0 * progress
        if remaining <= 50:
            score += float(50 - remaining) * 0.4

    if not bool(row.get("raw_top_family_matches_final", True)):
        score += 30.0
    elif not bool(row.get("raw_top_action_matches_final", True)):
        score += 15.0

    margin = _top_logit_margin(row.get("final_topk"))
    if margin is not None:
        if margin <= 0.25:
            score += 18.0
        elif margin <= 0.75:
            score += 12.0
        elif margin <= 1.50:
            score += 6.0

    legal_ids = row.get("legal_ids")
    selected_action = row.get("selected_action", {})
    selected_id = _action_id(selected_action if isinstance(selected_action, Mapping) else {})
    if isinstance(legal_ids, Sequence):
        legal_set = {int(item) for item in legal_ids}
        if selected_id is not None and int(selected_id) == 51 and len(legal_set) > 1:
            score += 16.0
        if 51 in legal_set:
            score += 3.0
    return float(score)


def _select_targets(
    *,
    trace_rows: Sequence[Mapping[str, Any]],
    episode_baselines: Mapping[tuple[int, int], Mapping[str, Any]],
    max_target_states: int,
    max_actions_per_state: int,
    allowed_families: set[str],
    pass_action_id: int | None,
    max_targets_per_pair: int,
    action_catalog: Any | None,
    family_representatives_per_family: int,
    excluded_pair_indices: set[int] | None = None,
    excluded_legal_fingerprints: set[str] | None = None,
    excluded_action_pairs: set[tuple[int, int]] | None = None,
    excluded_candidate_families: set[str] | None = None,
    excluded_candidate_action_ids: set[int] | None = None,
    require_pass_legal: bool = False,
    require_baseline_families: set[str] | None = None,
    randomize_target_order: bool = False,
    target_random_seed: int = 0,
) -> list[dict[str, Any]]:
    excluded_pair_indices = excluded_pair_indices or set()
    excluded_legal_fingerprints = excluded_legal_fingerprints or set()
    excluded_action_pairs = excluded_action_pairs or set()
    excluded_candidate_families = excluded_candidate_families or set()
    excluded_candidate_action_ids = excluded_candidate_action_ids or set()
    require_baseline_families = require_baseline_families or set()
    candidates_for_targets: list[dict[str, Any]] = []
    for row in trace_rows:
        pair_index = int(row["pair_index"])
        if pair_index in excluded_pair_indices:
            continue
        swap_index = int(row["swap_index"])
        legal_fingerprint = str(row.get("legal_ids_sha256") or "")
        if legal_fingerprint and legal_fingerprint in excluded_legal_fingerprints:
            continue
        episode = episode_baselines.get((pair_index, swap_index))
        if episode is None:
            continue
        baseline_winner = episode.get("winner_seat")
        if baseline_winner is None:
            continue
        target_seat = 1 - int(baseline_winner)
        if int(row.get("actor_seat", -1)) != int(target_seat):
            continue
        if int(row.get("legal_action_count", 0)) <= 1:
            continue
        legal_ids = row.get("legal_ids")
        legal_set = {int(item) for item in legal_ids} if isinstance(legal_ids, Sequence) else set()
        if bool(require_pass_legal) and (pass_action_id is None or int(pass_action_id) not in legal_set):
            continue
        selected_action = row.get("selected_action", {})
        selected_family = str(selected_action.get("family") or "") if isinstance(selected_action, Mapping) else ""
        if allowed_families and selected_family not in allowed_families:
            continue
        if require_baseline_families and selected_family not in require_baseline_families:
            continue
        selected_id = _action_id(selected_action if isinstance(selected_action, Mapping) else {})
        candidate_actions = _candidate_actions(
            row,
            max_actions=max_actions_per_state,
            pass_action_id=pass_action_id,
            action_catalog=action_catalog,
            family_representatives_per_family=int(family_representatives_per_family),
        )
        if selected_id is not None and excluded_action_pairs:
            candidate_actions = [
                item
                for item in candidate_actions
                if (int(selected_id), int(item["action_id"])) not in excluded_action_pairs
            ]
        if excluded_candidate_families or excluded_candidate_action_ids:
            candidate_actions = [
                item
                for item in candidate_actions
                if str(item.get("family") or "") not in excluded_candidate_families
                and int(item["action_id"]) not in excluded_candidate_action_ids
            ]
        candidates = [int(item["action_id"]) for item in candidate_actions]
        if not candidates:
            continue
        candidates_for_targets.append(
            {
                "pair_index": pair_index,
                "swap_index": swap_index,
                "episode_seed": int(row["episode_seed"]),
                "target_seat": int(target_seat),
                "baseline_winner_seat": int(baseline_winner),
                "decision_index": int(row["decision_index"]),
                "decision_id": int(row.get("decision_id", -1)),
                "legal_action_count": int(row.get("legal_action_count", 0)),
                "legal_ids_sha256": legal_fingerprint or row.get("legal_ids_sha256"),
                "selected_family": selected_family,
                "baseline_action": selected_action,
                "candidate_action_ids": candidates,
                "candidate_actions": candidate_actions,
                "target_rank_score": _target_rank_score(row, episode, selected_family),
                "trace_row": row,
            }
        )
    ranked = sorted(
        candidates_for_targets,
        key=lambda item: (
            -float(item["target_rank_score"]),
            int(item["pair_index"]),
            int(item["swap_index"]),
            int(item["decision_index"]),
        ),
    )
    if bool(randomize_target_order):
        rng = random.Random(int(target_random_seed))
        ranked = list(ranked)
        rng.shuffle(ranked)
    targets: list[dict[str, Any]] = []
    per_pair_counts: dict[int, int] = {}
    for target in ranked:
        pair_index = int(target["pair_index"])
        if max_targets_per_pair > 0 and per_pair_counts.get(pair_index, 0) >= int(max_targets_per_pair):
            continue
        targets.append(target)
        per_pair_counts[pair_index] = per_pair_counts.get(pair_index, 0) + 1
        if len(targets) >= int(max_target_states):
            break
    return targets


def _forced_result_for_target(
    matchup_dir: Path, *, pair_index: int, swap_index: int
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    pair_rows = _read_jsonl(matchup_dir / "pair_table.jsonl")
    target_pair = next((row for row in pair_rows if int(row["pair_index"]) == int(pair_index)), None)
    if target_pair is None:
        raise RuntimeError(f"forced replay missing pair_index={pair_index}: {matchup_dir}")
    episode = _episode_payload_for_swap(target_pair, int(swap_index))
    summary = _read_json(matchup_dir / "matchup_summary.json")
    counters = summary.get("matrix_runner_counters", {})
    return episode, counters if isinstance(counters, Mapping) else {}


def _nonnegative_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number < 0.0:
        return None
    return number


def _terminal_score(
    *,
    terminal_summary: Any,
    target_seat: int,
    winner_seat: Any,
) -> float | None:
    score = 0.0
    if winner_seat is not None:
        score += 1.0 if int(winner_seat) == int(target_seat) else -1.0
    if not isinstance(terminal_summary, Mapping):
        return score if winner_seat is not None else None

    target = terminal_summary.get(f"seat{int(target_seat)}")
    opponent = terminal_summary.get(f"seat{1 - int(target_seat)}")
    if not isinstance(target, Mapping) or not isinstance(opponent, Mapping):
        return score if winner_seat is not None else None

    target_level = _nonnegative_number(target.get("level"))
    opponent_level = _nonnegative_number(opponent.get("level"))
    if target_level is not None and opponent_level is not None:
        score += 0.15 * (opponent_level - target_level)

    target_clock = _nonnegative_number(target.get("clock"))
    opponent_clock = _nonnegative_number(opponent.get("clock"))
    if target_clock is not None and opponent_clock is not None:
        score += 0.02 * (opponent_clock - target_clock)

    target_stock = _nonnegative_number(target.get("stock"))
    opponent_stock = _nonnegative_number(opponent.get("stock"))
    if target_stock is not None and opponent_stock is not None:
        score += 0.01 * (target_stock - opponent_stock)

    target_hand = _nonnegative_number(target.get("hand_size"))
    opponent_hand = _nonnegative_number(opponent.get("hand_size"))
    if target_hand is not None and opponent_hand is not None:
        score += 0.005 * (target_hand - opponent_hand)

    target_stage = _nonnegative_number(target.get("stage_count"))
    opponent_stage = _nonnegative_number(opponent.get("stage_count"))
    if target_stage is not None and opponent_stage is not None:
        score += 0.005 * (target_stage - opponent_stage)

    return float(score)


def _label_weight(*, winner_flipped: bool, score_delta: float | None, margin_positive_threshold: float) -> float:
    if winner_flipped:
        return 1.0
    if score_delta is None:
        return 0.0
    if score_delta >= 0.30:
        return 0.75
    if score_delta >= float(margin_positive_threshold):
        return 0.4
    return 0.0


def _score_delta_stats(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(sum(values) / len(values)),
    }


def _label_tensor_ref(label_id: str) -> str:
    return f"states/{label_id}.pt"


def _trace_action_id(trace_row: Mapping[str, Any], key: str = "selected_action") -> int | None:
    payload = trace_row.get(key)
    if not isinstance(payload, Mapping):
        return None
    return _action_id(payload)


def _write_label_tensor_record(
    *,
    output_dir: Path,
    label_row: Mapping[str, Any],
    trace_row: Mapping[str, Any],
    positive_action_id: int,
    baseline_action: Mapping[str, Any] | None,
    target_seat: int,
    actor_seat: int | None = None,
    extra_trace_prefix: str | None = None,
) -> str:
    label_id = str(label_row["label_id"])
    obs_payload = trace_row.get("obs_float32")
    legal_payload = trace_row.get("legal_ids")
    final_legal_logits = trace_row.get("final_legal_logits")
    if not isinstance(obs_payload, Sequence) or isinstance(obs_payload, (str, bytes)):
        raise RuntimeError(
            f"cannot write tensor record for {label_id}: action trace row is missing obs_float32; "
            "rerun with trace tensor capture enabled"
        )
    if not isinstance(legal_payload, Sequence) or isinstance(legal_payload, (str, bytes)):
        raise RuntimeError(f"cannot write tensor record for {label_id}: action trace row is missing legal_ids")
    legal_ids = [int(item) for item in legal_payload]
    positive_id = int(positive_action_id)
    baseline_id = _action_id(baseline_action or {})
    if baseline_id is None:
        baseline_id = _trace_action_id(trace_row)
    if baseline_id is None:
        raise RuntimeError(f"cannot write tensor record for {label_id}: missing baseline action id")
    if positive_id not in set(legal_ids):
        raise RuntimeError(f"positive action {positive_id} is not legal for {label_id}")
    if int(baseline_id) not in set(legal_ids):
        raise RuntimeError(f"baseline action {baseline_id} is not legal for {label_id}")

    record_path = output_dir / _label_tensor_ref(label_id)
    obs = torch.as_tensor([float(item) for item in obs_payload], dtype=torch.float32)
    legal_tensor = torch.as_tensor(legal_ids, dtype=torch.long)
    payload: dict[str, Any] = {
        "format": "b1_counterfactual_state_tensor_v1",
        "label_id": label_id,
        "action_dim_hint": int(max(legal_ids) + 1) if legal_ids else 0,
        "surface": str(label_row.get("surface", "lowbias_s1")),
        "public_heuristic_bias_scale": torch.tensor(
            float(label_row.get("public_heuristic_bias_scale", 1.0)),
            dtype=torch.float32,
        ),
        "obs": obs,
        "obs_sha256": trace_row.get("obs_sha256") or _sha256_tensor(obs),
        "actor_seat": torch.tensor(
            int(trace_row.get("actor_seat", target_seat) if actor_seat is None else actor_seat),
            dtype=torch.long,
        ),
        "target_seat": torch.tensor(int(target_seat), dtype=torch.long),
        "legal_ids": legal_tensor,
        "legal_action_fingerprint": trace_row.get("legal_ids_sha256") or label_row.get("legal_ids_sha256"),
        "baseline_action_id": torch.tensor(int(baseline_id), dtype=torch.long),
        "positive_action_id": torch.tensor(positive_id, dtype=torch.long),
        "label_weight": torch.tensor(float(label_row.get("label_weight", 1.0)), dtype=torch.float32),
        "base_s1_legal_logits": torch.as_tensor(
            [float(item) for item in final_legal_logits],
            dtype=torch.float32,
        )
        if isinstance(final_legal_logits, Sequence) and not isinstance(final_legal_logits, (str, bytes))
        else torch.empty((0,), dtype=torch.float32),
        "base_s0_legal_logits": torch.as_tensor(
            [float(item) for item in trace_row.get("raw_legal_logits_no_public_bias", [])],
            dtype=torch.float32,
        ),
        "pair_index": torch.tensor(int(label_row.get("pair_index", -1)), dtype=torch.long),
        "swap_index": torch.tensor(int(label_row.get("swap_index", -1)), dtype=torch.long),
        "episode_seed": torch.tensor(int(label_row.get("episode_seed", -1)), dtype=torch.long),
        "decision_index": torch.tensor(int(label_row.get("decision_index", -1)), dtype=torch.long),
        "decision_id": torch.tensor(int(label_row.get("decision_id", -1)), dtype=torch.long),
        "selected_family": str(label_row.get("selected_family", "")),
        "positive_family": str((label_row.get("candidate_action") or {}).get("family", ""))
        if isinstance(label_row.get("candidate_action"), Mapping)
        else "",
        "action_family": str((label_row.get("candidate_action") or {}).get("family", ""))
        if isinstance(label_row.get("candidate_action"), Mapping)
        else "",
        "action_label": dict(label_row.get("candidate_action") or {})
        if isinstance(label_row.get("candidate_action"), Mapping)
        else {},
        "baseline_action": dict(baseline_action or {}),
        "score_delta": torch.tensor(
            float(label_row["score_delta"]) if label_row.get("score_delta") is not None else float("nan"),
            dtype=torch.float32,
        ),
        "winner_flipped": bool(label_row.get("winner_flipped", False)),
        "margin_positive": bool(label_row.get("margin_positive", False)),
        "baseline_terminal": label_row.get("baseline_terminal"),
        "forced_terminal": label_row.get("forced_terminal"),
    }
    if extra_trace_prefix:
        payload["extra_trace_prefix"] = str(extra_trace_prefix)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, record_path)
    return record_path.relative_to(output_dir).as_posix()


def _trace_row_for_trial(
    trace_rows: Sequence[Mapping[str, Any]],
    trial: Mapping[str, Any],
) -> Mapping[str, Any]:
    pair_index = int(trial["pair_index"])
    swap_index = int(trial["swap_index"])
    decision_index = int(trial["decision_index"])
    target_seat = int(trial["target_seat"])
    for row in trace_rows:
        if (
            int(row.get("pair_index", -1)) == pair_index
            and int(row.get("swap_index", -1)) == swap_index
            and int(row.get("decision_index", -1)) == decision_index
            and int(row.get("actor_seat", -1)) == target_seat
        ):
            return row
    raise RuntimeError(
        "could not find baseline trace row for "
        f"pair={pair_index} swap={swap_index} decision={decision_index} seat={target_seat}"
    )


def _second_step_candidates_from_trace(
    *,
    trace_rows: Sequence[Mapping[str, Any]],
    first_trial: Mapping[str, Any],
    allowed_families: set[str],
    pass_action_id: int | None,
    action_catalog: Any | None,
    max_actions: int,
    family_representatives_per_family: int,
    decision_window: int,
    excluded_candidate_families: set[str] | None = None,
    excluded_candidate_action_ids: set[int] | None = None,
) -> list[dict[str, Any]]:
    first_decision_index = int(first_trial["decision_index"])
    target_seat = int(first_trial["target_seat"])
    excluded_candidate_families = excluded_candidate_families or set()
    excluded_candidate_action_ids = excluded_candidate_action_ids or set()
    candidates: list[dict[str, Any]] = []
    for row in trace_rows:
        decision_index = int(row.get("decision_index", -1))
        if decision_index <= first_decision_index:
            continue
        if decision_index > first_decision_index + int(decision_window):
            break
        if int(row.get("actor_seat", -1)) != target_seat:
            continue
        if int(row.get("legal_action_count", 0)) <= 1:
            continue
        selected_action = row.get("selected_action", {})
        selected_family = str(selected_action.get("family") or "") if isinstance(selected_action, Mapping) else ""
        if allowed_families and selected_family not in allowed_families:
            continue
        actions = _candidate_actions(
            row,
            max_actions=max_actions,
            pass_action_id=pass_action_id,
            action_catalog=action_catalog,
            family_representatives_per_family=family_representatives_per_family,
        )
        if excluded_candidate_families or excluded_candidate_action_ids:
            actions = [
                item
                for item in actions
                if str(item.get("family") or "") not in excluded_candidate_families
                and int(item["action_id"]) not in excluded_candidate_action_ids
            ]
        if not actions:
            continue
        candidate = {
            "decision_index": decision_index,
            "decision_id": int(row.get("decision_id", -1)),
            "selected_family": selected_family,
            "baseline_action": selected_action,
            "legal_action_count": int(row.get("legal_action_count", 0)),
            "legal_ids_sha256": row.get("legal_ids_sha256"),
            "target_rank_score": _target_rank_score(row, first_trial.get("forced_episode", {}), selected_family),
            "trace_row": row,
            "candidate_actions": actions,
        }
        candidates.append(candidate)
    return sorted(candidates, key=lambda item: (-float(item["target_rank_score"]), int(item["decision_index"])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack-config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--b1-baseline-run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-policy", action="append", default=[])
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--artifact-dir-name", default="b1_counterfactual_labels_s1")
    parser.add_argument("--seed-scope", default="b1_counterfactual_labels_s1_fixed")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--public-heuristic-bias-scale", type=float, default=1.0)
    parser.add_argument("--trace-top-k", type=int, default=8)
    parser.add_argument("--trace-max-decisions-per-episode", type=int, default=80)
    parser.add_argument("--max-target-states", type=int, default=20)
    parser.add_argument("--max-targets-per-pair", type=int, default=2)
    parser.add_argument("--max-actions-per-state", type=int, default=8)
    parser.add_argument("--family-representatives-per-family", type=int, default=2)
    parser.add_argument("--max-forced-replays", type=int, default=64)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--stop-after-positive-labels", type=int, default=0)
    parser.add_argument("--exclude-labels", action="append", type=Path, default=[])
    parser.add_argument(
        "--ignore-excluded-label-pair-indices",
        action="store_true",
        help=(
            "When --exclude-labels is used with a fresh seed scope, keep fingerprint/action-pair "
            "exclusions but do not drop the same numeric pair indices."
        ),
    )
    parser.add_argument("--exclude-pair-index", default="")
    parser.add_argument("--exclude-legal-fingerprint", default="")
    parser.add_argument("--exclude-action-pair", default="")
    parser.add_argument(
        "--exclude-candidate-family",
        default="",
        help="Comma-separated candidate action families to skip, e.g. pass for non-pass mining.",
    )
    parser.add_argument(
        "--exclude-candidate-action-id",
        default="",
        help="Comma-separated candidate action ids to skip, e.g. 51 for non-pass mining.",
    )
    parser.add_argument("--require-pass-legal", action="store_true")
    parser.add_argument("--require-baseline-family", default="")
    parser.add_argument("--randomize-target-order", action="store_true")
    parser.add_argument("--target-random-seed", type=int, default=0)
    parser.add_argument("--margin-positive-threshold", type=float, default=0.15)
    parser.add_argument("--two-step-beam-targets", type=int, default=0)
    parser.add_argument("--two-step-min-first-delta", type=float, default=0.05)
    parser.add_argument("--two-step-window", type=int, default=40)
    parser.add_argument("--two-step-second-actions", type=int, default=4)
    parser.add_argument("--two-step-max-replays", type=int, default=64)
    parser.add_argument("--two-step-include-positive-first", action="store_true")
    parser.add_argument(
        "--two-step-target-families",
        default="main_play_character,main_move,main_play_event,climax_play,attack,pass",
        help="Comma-separated selected-action families for the second step; empty reuses --target-families.",
    )
    parser.add_argument(
        "--target-families",
        default="clock_from_hand,level_up,main_play_character,main_move,main_play_event,climax_play,attack,encore_pay,encore_decline,pass",
        help="Comma-separated selected-action families to target; empty means all.",
    )
    parser.add_argument("--pass-action-id", type=int, default=51)
    parser.add_argument("--execution-mode", choices=("in_process", "subprocess"), default="in_process")
    args = parser.parse_args()

    if not args.checkpoint_policy:
        raise SystemExit(
            "provide at least one --checkpoint-policy alias=checkpoint.pt for b1_artifact_matrix compatibility"
        )
    if int(args.pairs) <= 0:
        raise SystemExit("--pairs must be positive")

    run_dir = args.run_dir
    output_dir = run_dir / "eval" / args.artifact_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    ctx: _InProcessMatrixContext | None = None
    if args.execution_mode == "in_process":
        ctx = _build_in_process_context(
            stack_config=args.stack_config,
            run_dir=run_dir,
            b1_baseline_run_dir=args.b1_baseline_run_dir,
            checkpoint_policies=args.checkpoint_policy,
            pairs=int(args.pairs),
            seed_scope=str(args.seed_scope),
            device=str(args.device),
            public_heuristic_bias_scale=float(args.public_heuristic_bias_scale),
        )

    baseline_artifact = f"{args.artifact_dir_name}_baseline"
    if ctx is None:
        baseline_dir = _run_matrix(
            stack_config=args.stack_config,
            run_dir=run_dir,
            b1_baseline_run_dir=args.b1_baseline_run_dir,
            checkpoint_policies=args.checkpoint_policy,
            artifact_dir_name=baseline_artifact,
            pairs=int(args.pairs),
            seed_scope=str(args.seed_scope),
            device=str(args.device),
            public_heuristic_bias_scale=float(args.public_heuristic_bias_scale),
            emit_action_traces=True,
            trace_tensors=True,
            trace_top_k=int(args.trace_top_k),
            trace_max_decisions_per_episode=int(args.trace_max_decisions_per_episode),
        )
    else:
        baseline_dir = _run_matrix_in_process(
            ctx,
            artifact_dir_name=baseline_artifact,
            pairs=int(args.pairs),
            emit_action_traces=True,
            trace_tensors=True,
            trace_top_k=int(args.trace_top_k),
            trace_max_decisions_per_episode=int(args.trace_max_decisions_per_episode),
        )
    baseline_matchup_dir = _first_matchup_dir(baseline_dir)
    baseline_pairs = _read_jsonl(baseline_matchup_dir / "pair_table.jsonl")
    baseline_trace = _read_jsonl(baseline_matchup_dir / "action_trace.jsonl")
    episode_baselines = _build_episode_baselines(baseline_pairs)
    allowed_families = {item.strip() for item in str(args.target_families).split(",") if item.strip()}
    excluded_pairs_from_labels, excluded_fps_from_labels, excluded_action_pairs_from_labels = _load_excluded_label_keys(
        tuple(args.exclude_labels)
    )
    excluded_pair_indices = (
        set() if bool(args.ignore_excluded_label_pair_indices) else excluded_pairs_from_labels
    ) | _parse_int_set(str(args.exclude_pair_index))
    excluded_legal_fingerprints = excluded_fps_from_labels | _parse_str_set(str(args.exclude_legal_fingerprint))
    excluded_action_pairs = excluded_action_pairs_from_labels | _parse_action_pairs(str(args.exclude_action_pair))
    excluded_candidate_families = _parse_str_set(str(args.exclude_candidate_family))
    excluded_candidate_action_ids = _parse_int_set(str(args.exclude_candidate_action_id))
    require_baseline_families = _parse_str_set(str(args.require_baseline_family))
    two_step_families = {item.strip() for item in str(args.two_step_target_families).split(",") if item.strip()} or set(
        allowed_families
    )
    targets = _select_targets(
        trace_rows=baseline_trace,
        episode_baselines=episode_baselines,
        max_target_states=int(args.max_target_states),
        max_actions_per_state=int(args.max_actions_per_state),
        allowed_families=allowed_families,
        pass_action_id=None if args.pass_action_id is None else int(args.pass_action_id),
        max_targets_per_pair=int(args.max_targets_per_pair),
        action_catalog=None if ctx is None else ctx.action_catalog,
        family_representatives_per_family=int(args.family_representatives_per_family),
        excluded_pair_indices=excluded_pair_indices,
        excluded_legal_fingerprints=excluded_legal_fingerprints,
        excluded_action_pairs=excluded_action_pairs,
        excluded_candidate_families=excluded_candidate_families,
        excluded_candidate_action_ids=excluded_candidate_action_ids,
        require_pass_legal=bool(args.require_pass_legal),
        require_baseline_families=require_baseline_families,
        randomize_target_order=bool(args.randomize_target_order),
        target_random_seed=int(args.target_random_seed),
    )

    labels: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []
    attempted = 0
    forced_misses = 0
    two_step_attempted = 0
    two_step_forced_misses = 0

    positive_labels_path = output_dir / "counterfactual_labels.jsonl"
    trials_path = output_dir / "counterfactual_trials.jsonl"
    progress_path = output_dir / "counterfactual_progress.json"

    def _flush_progress(*, stage: str, target_index: int | None = None) -> None:
        _write_jsonl(positive_labels_path, labels)
        _write_jsonl(trials_path, trials)
        score_deltas = [float(row["score_delta"]) for row in trials if row.get("score_delta") is not None]
        _write_json(
            progress_path,
            {
                "format": "b1_counterfactual_labels_progress_v1",
                "stage": stage,
                "target_index": target_index,
                "pairs": int(args.pairs),
                "target_states": len(targets),
                "attempted_forced_replays": int(attempted),
                "forced_misses": int(forced_misses),
                "two_step_attempted_replays": int(two_step_attempted),
                "two_step_forced_misses": int(two_step_forced_misses),
                "positive_labels": len(labels),
                "winner_flip_labels": sum(1 for row in labels if bool(row.get("winner_flipped"))),
                "margin_positive_labels": sum(1 for row in labels if bool(row.get("margin_positive"))),
                "labels_path": positive_labels_path.relative_to(run_dir).as_posix(),
                "trials_path": trials_path.relative_to(run_dir).as_posix(),
                "score_delta_stats": _score_delta_stats(score_deltas),
                "max_forced_replays": int(args.max_forced_replays),
                "stop_after_positive_labels": int(args.stop_after_positive_labels),
                "exclude_labels": [str(path) for path in args.exclude_labels],
                "ignore_excluded_label_pair_indices": bool(args.ignore_excluded_label_pair_indices),
                "excluded_pair_indices": sorted(excluded_pair_indices),
                "excluded_legal_fingerprints": sorted(excluded_legal_fingerprints),
                "excluded_action_pairs": [[left, right] for left, right in sorted(excluded_action_pairs)],
                "excluded_candidate_families": sorted(excluded_candidate_families),
                "excluded_candidate_action_ids": sorted(excluded_candidate_action_ids),
                "require_pass_legal": bool(args.require_pass_legal),
                "require_baseline_families": sorted(require_baseline_families),
                "randomize_target_order": bool(args.randomize_target_order),
                "target_random_seed": int(args.target_random_seed),
            },
        )

    for target_index, target in enumerate(targets):
        if int(args.stop_after_positive_labels) > 0 and len(labels) >= int(args.stop_after_positive_labels):
            break
        for candidate_action in target["candidate_actions"]:
            action_id = int(candidate_action["action_id"])
            if attempted >= int(args.max_forced_replays):
                break
            if int(args.stop_after_positive_labels) > 0 and len(labels) >= int(args.stop_after_positive_labels):
                break
            attempted += 1
            forced_artifact = (
                f"{args.artifact_dir_name}_force_t{target_index:04d}"
                f"_p{int(target['pair_index'])}_s{int(target['swap_index'])}"
                f"_d{int(target['decision_index'])}_a{int(action_id)}"
            )
            force_action = {
                "pair_index": int(target["pair_index"]),
                "swap_index": int(target["swap_index"]),
                "seat": int(target["target_seat"]),
                "decision_index": int(target["decision_index"]),
                "action_id": int(action_id),
            }
            if ctx is None:
                forced_dir = _run_matrix(
                    stack_config=args.stack_config,
                    run_dir=run_dir,
                    b1_baseline_run_dir=args.b1_baseline_run_dir,
                    checkpoint_policies=args.checkpoint_policy,
                    artifact_dir_name=forced_artifact,
                    pairs=max(int(args.pairs), int(target["pair_index"]) + 1),
                    seed_scope=str(args.seed_scope),
                    device=str(args.device),
                    public_heuristic_bias_scale=float(args.public_heuristic_bias_scale),
                    emit_action_traces=False,
                    trace_tensors=False,
                    trace_top_k=int(args.trace_top_k),
                    trace_max_decisions_per_episode=int(args.trace_max_decisions_per_episode),
                    force_action=force_action,
                )
                forced_matchup_dir = _first_matchup_dir(forced_dir)
                forced_episode, counters = _forced_result_for_target(
                    forced_matchup_dir,
                    pair_index=int(target["pair_index"]),
                    swap_index=int(target["swap_index"]),
                )
                forced_artifact_ref: str | None = forced_dir.relative_to(run_dir).as_posix()
            else:
                forced_episode, counters = _run_forced_episode_in_process(
                    ctx,
                    force_action=force_action,
                )
                forced_artifact_ref = None
            baseline_episode = episode_baselines[(int(target["pair_index"]), int(target["swap_index"]))]
            if int(counters.get("forced_action_decisions", 0)) <= 0:
                forced_misses += 1
                trials.append(
                    {
                        "trial_id": f"trial_{len(trials):06d}",
                        "surface": "lowbias_s1",
                        "pair_index": int(target["pair_index"]),
                        "swap_index": int(target["swap_index"]),
                        "episode_seed": int(target["episode_seed"]),
                        "target_seat": int(target["target_seat"]),
                        "decision_index": int(target["decision_index"]),
                        "target_rank_score": float(target["target_rank_score"]),
                        "candidate_action": dict(candidate_action),
                        "positive_action_id": int(action_id),
                        "baseline_winner_seat": int(target["baseline_winner_seat"]),
                        "forced_winner_seat": forced_episode.get("winner_seat"),
                        "winner_flipped": False,
                        "margin_positive": False,
                        "positive_label": False,
                        "score_delta": None,
                        "label_weight": 0.0,
                        "baseline_episode": dict(baseline_episode),
                        "forced_episode": dict(forced_episode),
                        "forced_artifact_dir": forced_artifact_ref,
                        "trigger_matched": False,
                        "forced_action_was_legal": False,
                    }
                )
                continue
            forced_winner = forced_episode.get("winner_seat")
            winner_flipped = forced_winner is not None and int(forced_winner) == int(target["target_seat"])
            baseline_target_score = _terminal_score(
                terminal_summary=baseline_episode.get("terminal_summary"),
                target_seat=int(target["target_seat"]),
                winner_seat=target["baseline_winner_seat"],
            )
            forced_target_score = _terminal_score(
                terminal_summary=forced_episode.get("terminal_summary"),
                target_seat=int(target["target_seat"]),
                winner_seat=forced_winner,
            )
            score_delta = (
                None
                if baseline_target_score is None or forced_target_score is None
                else float(forced_target_score - baseline_target_score)
            )
            margin_positive = score_delta is not None and score_delta >= float(args.margin_positive_threshold)
            positive_label = bool(winner_flipped or margin_positive)
            label_weight = _label_weight(
                winner_flipped=bool(winner_flipped),
                score_delta=score_delta,
                margin_positive_threshold=float(args.margin_positive_threshold),
            )
            row = {
                "trial_id": f"trial_{len(trials):06d}",
                "surface": "lowbias_s1",
                "public_heuristic_bias_scale": float(args.public_heuristic_bias_scale),
                "pair_index": int(target["pair_index"]),
                "swap_index": int(target["swap_index"]),
                "episode_seed": int(target["episode_seed"]),
                "target_seat": int(target["target_seat"]),
                "decision_index": int(target["decision_index"]),
                "decision_id": int(target["decision_id"]),
                "legal_action_count": int(target["legal_action_count"]),
                "legal_ids_sha256": target["legal_ids_sha256"],
                "target_rank_score": float(target["target_rank_score"]),
                "selected_family": target["selected_family"],
                "baseline_action": target["baseline_action"],
                "candidate_action": dict(candidate_action),
                "positive_action_id": int(action_id),
                "baseline_winner_seat": int(target["baseline_winner_seat"]),
                "forced_winner_seat": None if forced_winner is None else int(forced_winner),
                "winner_flipped": bool(winner_flipped),
                "margin_positive": bool(margin_positive),
                "positive_label": bool(positive_label),
                "baseline_target_score": baseline_target_score,
                "forced_target_score": forced_target_score,
                "score_delta": score_delta,
                "label_weight": float(label_weight),
                "label_strength": "winner_flip"
                if winner_flipped
                else ("margin_positive" if margin_positive else "nonpositive"),
                "baseline_episode": dict(baseline_episode),
                "forced_episode": dict(forced_episode),
                "baseline_terminal": baseline_episode.get("terminal_summary"),
                "forced_terminal": forced_episode.get("terminal_summary"),
                "forced_artifact_dir": forced_artifact_ref,
                "trigger_matched": True,
                "forced_action_was_legal": True,
            }
            trials.append(row)
            if positive_label:
                label_row = dict(row)
                label_row["label_id"] = f"cf_{len(labels):06d}"
                tensor_ref = _write_label_tensor_record(
                    output_dir=output_dir,
                    label_row=label_row,
                    trace_row=target["trace_row"],
                    positive_action_id=int(action_id),
                    baseline_action=target["baseline_action"]
                    if isinstance(target.get("baseline_action"), Mapping)
                    else None,
                    target_seat=int(target["target_seat"]),
                )
                label_row["tensor_ref"] = tensor_ref
                row["tensor_ref"] = tensor_ref
                labels.append(label_row)
            if int(args.progress_every) > 0 and attempted % int(args.progress_every) == 0:
                _flush_progress(stage="one_step", target_index=target_index)
            if attempted >= int(args.max_forced_replays):
                break
            if int(args.stop_after_positive_labels) > 0 and len(labels) >= int(args.stop_after_positive_labels):
                break
        if int(args.progress_every) > 0:
            _flush_progress(stage="one_step", target_index=target_index)

    if ctx is not None and int(args.two_step_beam_targets) > 0:
        trace_dir = output_dir / "two_step_traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        first_step_pool = [
            row
            for row in trials
            if row.get("score_delta") is not None
            and float(row["score_delta"]) >= float(args.two_step_min_first_delta)
            and bool(row.get("trigger_matched", False))
            and (bool(args.two_step_include_positive_first) or not bool(row.get("positive_label", False)))
        ]
        first_step_pool = sorted(
            first_step_pool,
            key=lambda row: (
                -float(row["score_delta"]),
                int(row["pair_index"]),
                int(row["swap_index"]),
                int(row["decision_index"]),
            ),
        )[: int(args.two_step_beam_targets)]
        for first_index, first_trial in enumerate(first_step_pool):
            if two_step_attempted >= int(args.two_step_max_replays):
                break
            first_force = {
                "pair_index": int(first_trial["pair_index"]),
                "swap_index": int(first_trial["swap_index"]),
                "seat": int(first_trial["target_seat"]),
                "decision_index": int(first_trial["decision_index"]),
                "action_id": int(first_trial["positive_action_id"]),
            }
            first_trace_path = trace_dir / (
                f"first_{first_index:04d}_p{int(first_trial['pair_index'])}"
                f"_s{int(first_trial['swap_index'])}_d{int(first_trial['decision_index'])}.jsonl"
            )
            if first_trace_path.exists():
                first_trace_path.unlink()
            _run_forced_episode_in_process(
                ctx,
                force_action=first_force,
                force_action_sequence=[first_force],
                trace_path=first_trace_path,
                trace_top_k=int(args.trace_top_k),
                trace_max_decisions_per_episode=max(
                    int(args.trace_max_decisions_per_episode),
                    int(first_trial["decision_index"]) + int(args.two_step_window) + 10,
                ),
            )
            first_trace_rows = _read_jsonl(first_trace_path)
            second_targets = _second_step_candidates_from_trace(
                trace_rows=first_trace_rows,
                first_trial=first_trial,
                allowed_families=two_step_families,
                pass_action_id=None if args.pass_action_id is None else int(args.pass_action_id),
                action_catalog=ctx.action_catalog,
                max_actions=int(args.two_step_second_actions),
                family_representatives_per_family=int(args.family_representatives_per_family),
                decision_window=int(args.two_step_window),
                excluded_candidate_families=excluded_candidate_families,
                excluded_candidate_action_ids=excluded_candidate_action_ids,
            )
            if not second_targets:
                continue
            second_target = second_targets[0]
            for second_candidate in second_target["candidate_actions"][: int(args.two_step_second_actions)]:
                if two_step_attempted >= int(args.two_step_max_replays):
                    break
                two_step_attempted += 1
                second_force = {
                    "pair_index": int(first_trial["pair_index"]),
                    "swap_index": int(first_trial["swap_index"]),
                    "seat": int(first_trial["target_seat"]),
                    "decision_index": int(second_target["decision_index"]),
                    "action_id": int(second_candidate["action_id"]),
                }
                forced_episode, counters = _run_forced_episode_in_process(
                    ctx,
                    force_action=first_force,
                    force_action_sequence=[first_force, second_force],
                )
                baseline_episode = episode_baselines[(int(first_trial["pair_index"]), int(first_trial["swap_index"]))]
                forced_count = int(counters.get("forced_action_decisions", 0))
                if forced_count < 2:
                    two_step_forced_misses += 1
                forced_winner = forced_episode.get("winner_seat")
                winner_flipped = forced_winner is not None and int(forced_winner) == int(first_trial["target_seat"])
                baseline_target_score = _terminal_score(
                    terminal_summary=baseline_episode.get("terminal_summary"),
                    target_seat=int(first_trial["target_seat"]),
                    winner_seat=first_trial["baseline_winner_seat"],
                )
                forced_target_score = _terminal_score(
                    terminal_summary=forced_episode.get("terminal_summary"),
                    target_seat=int(first_trial["target_seat"]),
                    winner_seat=forced_winner,
                )
                score_delta = (
                    None
                    if baseline_target_score is None or forced_target_score is None
                    else float(forced_target_score - baseline_target_score)
                )
                margin_positive = score_delta is not None and score_delta >= float(args.margin_positive_threshold)
                positive_label = bool(winner_flipped or margin_positive)
                label_weight = _label_weight(
                    winner_flipped=bool(winner_flipped),
                    score_delta=score_delta,
                    margin_positive_threshold=float(args.margin_positive_threshold),
                )
                row = {
                    "trial_id": f"trial_{len(trials):06d}",
                    "label_type": "two_step",
                    "surface": "lowbias_s1",
                    "public_heuristic_bias_scale": float(args.public_heuristic_bias_scale),
                    "pair_index": int(first_trial["pair_index"]),
                    "swap_index": int(first_trial["swap_index"]),
                    "episode_seed": int(first_trial["episode_seed"]),
                    "target_seat": int(first_trial["target_seat"]),
                    "decision_index": int(first_trial["decision_index"]),
                    "second_decision_index": int(second_target["decision_index"]),
                    "decision_id": int(first_trial["decision_id"]),
                    "second_decision_id": int(second_target["decision_id"]),
                    "selected_family": first_trial["selected_family"],
                    "second_selected_family": second_target["selected_family"],
                    "baseline_action": first_trial["baseline_action"],
                    "second_baseline_action": second_target["baseline_action"],
                    "candidate_action": dict(first_trial["candidate_action"]),
                    "second_candidate_action": dict(second_candidate),
                    "positive_action_id": int(first_trial["positive_action_id"]),
                    "positive_action_sequence": [
                        int(first_trial["positive_action_id"]),
                        int(second_candidate["action_id"]),
                    ],
                    "baseline_winner_seat": int(first_trial["baseline_winner_seat"]),
                    "forced_winner_seat": None if forced_winner is None else int(forced_winner),
                    "winner_flipped": bool(winner_flipped),
                    "margin_positive": bool(margin_positive),
                    "positive_label": bool(positive_label),
                    "baseline_target_score": baseline_target_score,
                    "forced_target_score": forced_target_score,
                    "score_delta": score_delta,
                    "first_step_score_delta": first_trial.get("score_delta"),
                    "label_weight": float(label_weight),
                    "label_strength": "winner_flip"
                    if winner_flipped
                    else ("margin_positive" if margin_positive else "nonpositive"),
                    "baseline_episode": dict(baseline_episode),
                    "forced_episode": dict(forced_episode),
                    "baseline_terminal": baseline_episode.get("terminal_summary"),
                    "forced_terminal": forced_episode.get("terminal_summary"),
                    "first_trace_path": first_trace_path.relative_to(run_dir).as_posix(),
                    "trigger_matched": forced_count >= 2,
                    "forced_action_was_legal": forced_count >= 2,
                }
                trials.append(row)
                if positive_label:
                    label_row = dict(row)
                    label_row["label_id"] = f"cf_{len(labels):06d}"
                    first_trace_row = _trace_row_for_trial(baseline_trace, first_trial)
                    tensor_ref = _write_label_tensor_record(
                        output_dir=output_dir,
                        label_row=label_row,
                        trace_row=first_trace_row,
                        positive_action_id=int(first_trial["positive_action_id"]),
                        baseline_action=first_trial["baseline_action"]
                        if isinstance(first_trial.get("baseline_action"), Mapping)
                        else None,
                        target_seat=int(first_trial["target_seat"]),
                        extra_trace_prefix="first_step",
                    )
                    label_row["tensor_ref"] = tensor_ref
                    row["tensor_ref"] = tensor_ref
                    labels.append(label_row)
                if int(args.progress_every) > 0 and two_step_attempted % int(args.progress_every) == 0:
                    _flush_progress(stage="two_step", target_index=None)
                if int(args.stop_after_positive_labels) > 0 and len(labels) >= int(args.stop_after_positive_labels):
                    break
            if int(args.stop_after_positive_labels) > 0 and len(labels) >= int(args.stop_after_positive_labels):
                break

    _write_jsonl(positive_labels_path, labels)
    _write_jsonl(trials_path, trials)
    score_deltas = [float(row["score_delta"]) for row in trials if row.get("score_delta") is not None]
    margin_positive_labels = sum(1 for row in labels if bool(row.get("margin_positive")))
    winner_flip_labels = sum(1 for row in labels if bool(row.get("winner_flipped")))
    summary = {
        "format": "b1_counterfactual_labels_summary_v1",
        "baseline_artifact_dir": baseline_dir.relative_to(run_dir).as_posix(),
        "baseline_matchup_dir": baseline_matchup_dir.relative_to(run_dir).as_posix(),
        "pairs": int(args.pairs),
        "trace_rows": len(baseline_trace),
        "target_states": len(targets),
        "attempted_forced_replays": int(attempted),
        "forced_misses": int(forced_misses),
        "two_step_attempted_replays": int(two_step_attempted),
        "two_step_forced_misses": int(two_step_forced_misses),
        "two_step_beam_targets": int(args.two_step_beam_targets),
        "two_step_min_first_delta": float(args.two_step_min_first_delta),
        "two_step_window": int(args.two_step_window),
        "two_step_second_actions": int(args.two_step_second_actions),
        "two_step_max_replays": int(args.two_step_max_replays),
        "two_step_include_positive_first": bool(args.two_step_include_positive_first),
        "two_step_target_families": sorted(two_step_families),
        "winner_flip_labels": int(winner_flip_labels),
        "margin_positive_labels": int(margin_positive_labels),
        "positive_labels": len(labels),
        "labels_path": positive_labels_path.relative_to(run_dir).as_posix(),
        "trials_path": trials_path.relative_to(run_dir).as_posix(),
        "margin_positive_threshold": float(args.margin_positive_threshold),
        "score_delta_stats": _score_delta_stats(score_deltas),
        "max_target_states": int(args.max_target_states),
        "max_actions_per_state": int(args.max_actions_per_state),
        "family_representatives_per_family": int(args.family_representatives_per_family),
        "max_targets_per_pair": int(args.max_targets_per_pair),
        "max_forced_replays": int(args.max_forced_replays),
        "target_families": sorted(allowed_families),
        "exclude_labels": [str(path) for path in args.exclude_labels],
        "ignore_excluded_label_pair_indices": bool(args.ignore_excluded_label_pair_indices),
        "excluded_pair_indices": sorted(excluded_pair_indices),
        "excluded_legal_fingerprints": sorted(excluded_legal_fingerprints),
        "excluded_action_pairs": [[left, right] for left, right in sorted(excluded_action_pairs)],
        "excluded_candidate_families": sorted(excluded_candidate_families),
        "excluded_candidate_action_ids": sorted(excluded_candidate_action_ids),
        "require_pass_legal": bool(args.require_pass_legal),
        "require_baseline_families": sorted(require_baseline_families),
        "randomize_target_order": bool(args.randomize_target_order),
        "target_random_seed": int(args.target_random_seed),
        "execution_mode": str(args.execution_mode),
    }
    _write_json(output_dir / "counterfactual_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
