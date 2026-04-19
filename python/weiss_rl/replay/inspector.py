"""Replay inspection helpers for comparing policy distributions on a recorded replay."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.action_catalog import ActionCatalog
from weiss_rl.config import StackConfig, compute_config_hash256, load_stack_config
from weiss_rl.envs.decision_env import DecisionBoundaryBatch
from weiss_rl.eval.heuristic_public import HeuristicPublicPolicy
from weiss_rl.eval.policy_set import HEURISTIC_PUBLIC_POLICY_ID
from weiss_rl.league.registry import REGISTRY_FILENAME, SnapshotRegistry
from weiss_rl.masking import masked_log_softmax
from weiss_rl.model import GLOBAL_ACTION_SPACE_SIZE, PolicyValueModel, build_policy_value_model
from weiss_rl.replay.bundles import ReplayBundleMeta, ReplayStep, compute_legal_fingerprint64, load_replay_bundle
from weiss_rl.replay.runner import ReplayEnvFactory, build_replay_env, require_supported_rerun_contract


@dataclass(frozen=True, slots=True)
class LoadedReplayPolicy:
    spec: str
    label: str
    kind: str
    weights_path: Path | None
    model: PolicyValueModel | None = None
    heuristic_policy: HeuristicPublicPolicy | None = None


def inspect_replay_bundle(
    *,
    bundle_path: Path,
    stack: StackConfig | Path,
    policy_a: str,
    policy_b: str,
    run_dir: Path | None = None,
    snapshot_registry_path: Path | None = None,
    top_k: int = 10,
    top_actions: int = 5,
    env_factory: ReplayEnvFactory | None = None,
) -> dict[str, Any]:
    if top_k < 0:
        raise ValueError("top_k must be >= 0")
    if top_actions <= 0:
        raise ValueError("top_actions must be >= 1")

    bundle_path = Path(bundle_path).resolve()
    stack_config = load_stack_config(stack) if isinstance(stack, Path) else stack
    resolved_registry_path, resolved_run_dir, registry = _resolve_registry(
        run_dir=run_dir,
        snapshot_registry_path=snapshot_registry_path,
    )

    meta, steps, fault = load_replay_bundle(bundle_path)
    contract = require_supported_rerun_contract(meta)
    env = None
    compared_steps = 0
    run_spec_bundle = _load_run_spec_bundle(resolved_run_dir)
    action_catalog = _load_action_catalog(run_spec_bundle)

    try:
        env = build_replay_env(contract, env_factory=env_factory)
        current_batch = _require_single_env_batch(env.reset(seed=meta.episode_seed64), context="reset")
        _require_initial_identity(meta=meta, batch=current_batch)

        observation_dim = _observation_dim(current_batch)
        policy_a_loaded = _load_policy(
            spec=policy_a,
            stack=stack_config,
            observation_dim=observation_dim,
            action_dim=GLOBAL_ACTION_SPACE_SIZE,
            run_dir=resolved_run_dir,
            registry=registry,
            run_spec_bundle=run_spec_bundle,
        )
        policy_b_loaded = _load_policy(
            spec=policy_b,
            stack=stack_config,
            observation_dim=observation_dim,
            action_dim=GLOBAL_ACTION_SPACE_SIZE,
            run_dir=resolved_run_dir,
            registry=registry,
            run_spec_bundle=run_spec_bundle,
        )

        device = torch.device("cpu")
        policy_a_hidden = (
            None if policy_a_loaded.model is None else policy_a_loaded.model.initial_seat_hidden(1, device=device)
        )
        policy_b_hidden = (
            None if policy_b_loaded.model is None else policy_b_loaded.model.initial_seat_hidden(1, device=device)
        )
        spec_hash256 = bytes.fromhex(meta.spec_hash256)

        step_diffs: list[dict[str, Any]] = []
        for step_index, expected_step in enumerate(steps):
            _require_pre_step_match(
                step_index=step_index,
                expected_step=expected_step,
                current_batch=current_batch,
                spec_hash256=spec_hash256,
            )

            legal_ids = _legal_ids_for_env_row(current_batch)
            logits_a, policy_a_hidden = _forward_policy(
                policy=policy_a_loaded,
                batch=current_batch,
                seat_hidden=policy_a_hidden,
                legal_ids=legal_ids,
            )
            logits_b, policy_b_hidden = _forward_policy(
                policy=policy_b_loaded,
                batch=current_batch,
                seat_hidden=policy_b_hidden,
                legal_ids=legal_ids,
            )
            step_diffs.append(
                _build_step_diff(
                    step_index=step_index,
                    expected_step=expected_step,
                    legal_ids=legal_ids,
                    logits_a=logits_a,
                    logits_b=logits_b,
                    top_actions=top_actions,
                    action_catalog=action_catalog,
                )
            )

            next_batch = _require_single_env_batch(
                env.step(np.asarray([expected_step.action], dtype=np.uint32)),
                context=f"step[{step_index}]",
            )
            _require_post_step_match(step_index=step_index, expected_step=expected_step, next_batch=next_batch)

            compared_steps = step_index + 1
            if (expected_step.terminated or expected_step.truncated) and compared_steps != len(steps):
                raise RuntimeError("Recorded replay bundle contains additional steps after termination")
            current_batch = next_batch

        report = {
            "bundle_path": bundle_path.as_posix(),
            "policy_a": {
                "spec": policy_a_loaded.spec,
                "label": policy_a_loaded.label,
                "kind": policy_a_loaded.kind,
                "weights_path": None
                if policy_a_loaded.weights_path is None
                else policy_a_loaded.weights_path.as_posix(),
            },
            "policy_b": {
                "spec": policy_b_loaded.spec,
                "label": policy_b_loaded.label,
                "kind": policy_b_loaded.kind,
                "weights_path": None
                if policy_b_loaded.weights_path is None
                else policy_b_loaded.weights_path.as_posix(),
            },
            "run_dir": None if resolved_run_dir is None else resolved_run_dir.as_posix(),
            "snapshot_registry_path": None if resolved_registry_path is None else resolved_registry_path.as_posix(),
            "replay": {
                "replay_key64": f"{meta.replay_key64:016x}",
                "episode_key64": int(meta.episode_key64),
                "episode_seed64": int(meta.episode_seed64),
                "expected_steps": len(steps),
                "fault_present": fault is not None,
                "rerun_contract": None if meta.rerun_contract is None else asdict(meta.rerun_contract),
            },
            "summary": _summarize_step_diffs(step_diffs, top_k=top_k),
            "top_differences": _top_step_diffs(step_diffs, top_k=top_k),
            "compared_steps": compared_steps,
        }
        return report
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def format_replay_inspection_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    policy_a_source = _format_policy_source(report["policy_a"])
    policy_b_source = _format_policy_source(report["policy_b"])
    lines = [
        "Replay inspector",
        f"bundle: {report['bundle_path']}",
        f"policy_a: {report['policy_a']['label']} ({policy_a_source})",
        f"policy_b: {report['policy_b']['label']} ({policy_b_source})",
        f"compared_steps: {report['compared_steps']}",
        (
            "summary: "
            f"max_tv={summary['max_total_variation']:.6f} "
            f"mean_tv={summary['mean_total_variation']:.6f} "
            f"max_abs_prob_delta={summary['max_abs_probability_delta']:.6f}"
        ),
        (
            "alignment: "
            f"action_match_rate={summary['policy_a_matches_policy_b_top_action_rate']:.6f} "
            f"family_match_rate={summary['policy_a_matches_policy_b_top_action_family_rate']:.6f} "
            f"mean_p_on_b={summary['policy_a_mean_probability_on_policy_b_top_action']:.6f} "
            f"median_rank_of_b={summary['policy_a_median_rank_of_policy_b_top_action']:.2f}"
        ),
        "top_differences:",
    ]
    if summary["top_action_family_confusions"]:
        family_confusions = ", ".join(
            f"{item['policy_b_family']}->{item['policy_a_family']} x{item['count']}"
            for item in summary["top_action_family_confusions"]
        )
        lines.append(f"family_confusions: {family_confusions}")

    for index, diff in enumerate(report["top_differences"], start=1):
        lines.append(
            (
                f"{index}. step={diff['step_index']} decision_id={diff['decision_id']} actor={diff['actor']} "
                f"recorded_action={_format_action_descriptor(diff['recorded_action_detail'])} "
                f"tv={diff['total_variation']:.6f} "
                f"max_abs_prob_delta={diff['max_abs_probability_delta']:.6f}"
            )
        )
        lines.append(
            (
                f"   {report['policy_a']['label']}: top_action={_format_action_descriptor(diff['policy_a_top_action'])} "
                f"p={diff['policy_a_top_action']['probability']:.6f} "
                f"recorded_p={diff['policy_a_recorded_action_probability']:.6f}"
            )
        )
        lines.append(
            (
                f"   {report['policy_b']['label']}: top_action={_format_action_descriptor(diff['policy_b_top_action'])} "
                f"p={diff['policy_b_top_action']['probability']:.6f} "
                f"recorded_p={diff['policy_b_recorded_action_probability']:.6f}"
            )
        )
        lines.append(
            (
                f"   learner_on_b2: p={diff['policy_a_probability_on_policy_b_top_action']:.6f} "
                f"rank={diff['policy_a_rank_of_policy_b_top_action']} "
                f"action_match={diff['policy_a_matches_policy_b_top_action']} "
                f"family_match={diff['policy_a_matches_policy_b_top_action_family']}"
            )
        )
        action_delta_text = ", ".join(
            (
                f"{_format_action_descriptor(item)}:Δ={item['probability_delta_b_minus_a']:+.6f} "
                f"(A={item['probability_a']:.6f},B={item['probability_b']:.6f})"
            )
            for item in diff["top_action_deltas"]
        )
        lines.append(f"   deltas: {action_delta_text}")

    return "\n".join(lines) + "\n"


def write_replay_inspection_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_registry(
    *,
    run_dir: Path | None,
    snapshot_registry_path: Path | None,
) -> tuple[Path | None, Path | None, SnapshotRegistry | None]:
    resolved_registry_path = None if snapshot_registry_path is None else snapshot_registry_path.resolve()
    resolved_run_dir = None if run_dir is None else run_dir.resolve()

    if resolved_registry_path is None and resolved_run_dir is not None:
        candidate = resolved_run_dir / "training" / "snapshots" / REGISTRY_FILENAME
        if candidate.is_file():
            resolved_registry_path = candidate

    if resolved_run_dir is None and resolved_registry_path is not None:
        registry_path_parts = resolved_registry_path.parts[-3:]
        if registry_path_parts == ("training", "snapshots", REGISTRY_FILENAME):
            resolved_run_dir = resolved_registry_path.parents[2]

    registry = None if resolved_registry_path is None else SnapshotRegistry.load(resolved_registry_path)
    return resolved_registry_path, resolved_run_dir, registry


def _load_policy(
    *,
    spec: str,
    stack: StackConfig,
    observation_dim: int,
    action_dim: int,
    run_dir: Path | None,
    registry: SnapshotRegistry | None,
    run_spec_bundle: dict[str, Any] | None,
) -> LoadedReplayPolicy:
    normalized_spec = str(spec).strip()
    if normalized_spec == HEURISTIC_PUBLIC_POLICY_ID:
        if run_spec_bundle is None:
            raise RuntimeError(
                "Resolving B2 HeuristicPublic for replay inspection requires spec_bundle.json in run_dir"
            )
        return LoadedReplayPolicy(
            spec=normalized_spec,
            label=normalized_spec,
            kind="heuristic_public",
            weights_path=None,
            heuristic_policy=HeuristicPublicPolicy.from_spec_bundle(run_spec_bundle),
        )

    weights_path, label = _resolve_policy_weights_path(spec=spec, run_dir=run_dir, registry=registry)
    payload = torch.load(weights_path, map_location="cpu", weights_only=True)
    model_state_dict = payload.get("model_state_dict")
    if not isinstance(model_state_dict, dict):
        raise RuntimeError(f"Snapshot weights payload missing model_state_dict: {weights_path}")

    expected_config_hash256 = compute_config_hash256(stack)
    observed_config_hash256 = str(payload.get("config_hash256", "")).strip()
    if observed_config_hash256 and observed_config_hash256 != expected_config_hash256:
        raise RuntimeError(
            f"Snapshot config hash mismatch for {weights_path}: "
            f"expected {expected_config_hash256}, observed {observed_config_hash256}"
        )

    model_config = stack.config.model
    if model_config is None:
        raise RuntimeError("The locked stack is missing the model config block")

    model = build_policy_value_model(
        observation_dim=observation_dim,
        config=model_config,
        action_dim=action_dim,
        observation_spec=_load_run_observation_spec(run_spec_bundle),
        spec_bundle=run_spec_bundle,
    ).to(torch.device("cpu"))
    model.load_state_dict(model_state_dict)
    model.eval()
    return LoadedReplayPolicy(
        spec=str(spec),
        label=label,
        kind="model",
        weights_path=weights_path,
        model=model,
    )


def _load_run_spec_bundle(run_dir: Path | None) -> dict[str, Any] | None:
    if run_dir is None:
        return None
    layout = ArtifactLayout.from_run_dir(run_dir)
    if not layout.spec_bundle_path.is_file():
        return None
    payload = json.loads(layout.spec_bundle_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"spec_bundle.json must contain an object: {layout.spec_bundle_path}")
    return dict(payload)


def _load_run_observation_spec(spec_bundle: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if spec_bundle is None:
        return None
    observation = spec_bundle.get("observation")
    if observation is None:
        return None
    if not isinstance(observation, dict):
        raise RuntimeError("spec_bundle.json observation payload must be an object")
    return dict(observation)


def _load_action_catalog(spec_bundle: Mapping[str, Any] | None) -> ActionCatalog | None:
    if spec_bundle is None:
        return None
    try:
        return ActionCatalog.from_spec_bundle(spec_bundle)
    except (KeyError, TypeError, ValueError):
        return None


def _resolve_policy_weights_path(
    *,
    spec: str,
    run_dir: Path | None,
    registry: SnapshotRegistry | None,
) -> tuple[Path, str]:
    normalized_spec = str(spec).strip()
    if not normalized_spec:
        raise ValueError("policy spec must be non-empty")

    spec_path = Path(normalized_spec)
    direct_candidates: list[Path] = []
    if spec_path.is_absolute():
        direct_candidates.append(spec_path)
    else:
        if run_dir is not None:
            direct_candidates.append(run_dir / spec_path)
        direct_candidates.append(spec_path)
    for candidate in direct_candidates:
        if candidate.is_file():
            return candidate.resolve(), normalized_spec

    if registry is None:
        raise RuntimeError(
            "Could not resolve policy spec "
            f"{normalized_spec!r} as a weights path, and no snapshot registry is available"
        )
    if run_dir is None:
        raise RuntimeError(f"Cannot resolve policy id {normalized_spec!r} without a run_dir")

    snapshot_meta = next((snapshot for snapshot in registry.snapshots if snapshot.policy_id == normalized_spec), None)
    if snapshot_meta is None:
        raise RuntimeError(f"Unknown policy id: {normalized_spec!r}")

    resolved_path = (run_dir / snapshot_meta.path).resolve()
    if not resolved_path.is_file():
        raise RuntimeError(f"Resolved policy weights path does not exist: {resolved_path}")
    return resolved_path, snapshot_meta.policy_id


def _require_single_env_batch(batch: DecisionBoundaryBatch, *, context: str) -> DecisionBoundaryBatch:
    if batch.num_envs != 1:
        raise RuntimeError(f"Replay inspection expects a single-env batch from {context}, got {batch.num_envs}")
    return batch


def _observation_dim(batch: DecisionBoundaryBatch) -> int:
    obs = np.asarray(batch.obs)
    if obs.ndim != 2:
        raise RuntimeError(f"Replay inspection expects 2D observations, got shape {tuple(obs.shape)}")
    return int(obs.shape[1])


def _require_initial_identity(*, meta: ReplayBundleMeta, batch: DecisionBoundaryBatch) -> None:
    observed_seed = int(batch.episode_seed[0])
    if observed_seed != int(meta.episode_seed64):
        raise RuntimeError(
            f"Replay reset seed mismatch: expected episode_seed64={meta.episode_seed64}, got {observed_seed}"
        )
    if meta.simulator_episode_key_u64 is None:
        return

    observed_episode_key = int(batch.episode_key[0])
    if observed_episode_key != int(meta.simulator_episode_key_u64):
        raise RuntimeError(
            "Replay reset episode_key mismatch: "
            f"expected simulator episode key {meta.simulator_episode_key_u64}, got {observed_episode_key}"
        )


def _require_pre_step_match(
    *,
    step_index: int,
    expected_step: ReplayStep,
    current_batch: DecisionBoundaryBatch,
    spec_hash256: bytes,
) -> None:
    observed_t = step_index
    batch_t = getattr(current_batch, "t", None)
    if batch_t is not None:
        observed_t = int(np.asarray(batch_t).reshape(-1)[0])
    if observed_t != int(expected_step.t):
        raise RuntimeError(f"Replay step index mismatch at step {step_index}")

    actual_decision_id = int(current_batch.decision_id[0])
    if actual_decision_id != int(expected_step.decision_id):
        raise RuntimeError(f"Replay decision_id mismatch at step {step_index}")

    actual_actor = int(current_batch.actor[0])
    if actual_actor != int(expected_step.actor):
        raise RuntimeError(f"Replay actor mismatch at step {step_index}")

    legal_ids = _legal_ids_for_env_row(current_batch)
    actual_fingerprint = compute_legal_fingerprint64(
        spec_hash256=spec_hash256,
        decision_id=actual_decision_id,
        legal_ids=legal_ids,
    )
    if actual_fingerprint != int(expected_step.legal_fingerprint64):
        raise RuntimeError(f"Replay legal fingerprint mismatch at step {step_index}")


def _require_post_step_match(*, step_index: int, expected_step: ReplayStep, next_batch: DecisionBoundaryBatch) -> None:
    if _canonical_float(next_batch.reward[0]) != _canonical_float(expected_step.reward):
        raise RuntimeError(f"Replay reward mismatch at step {step_index}")
    if bool(next_batch.terminated[0]) != bool(expected_step.terminated):
        raise RuntimeError(f"Replay terminated mismatch at step {step_index}")
    if bool(next_batch.truncated[0]) != bool(expected_step.truncated):
        raise RuntimeError(f"Replay truncated mismatch at step {step_index}")
    if int(next_batch.engine_status[0]) != int(expected_step.engine_status):
        raise RuntimeError(f"Replay engine_status mismatch at step {step_index}")


def _legal_ids_for_env_row(batch: DecisionBoundaryBatch) -> np.ndarray:
    if batch.ids_offsets is None:
        raise RuntimeError("Replay inspection requires ids_offsets legality in the rerun environment")
    legal_ids, legal_offsets = batch.ids_offsets
    start = int(legal_offsets[0])
    end = int(legal_offsets[1])
    return np.asarray(legal_ids[start:end], dtype=np.uint32)


def _forward_policy(
    *,
    policy: LoadedReplayPolicy,
    batch: DecisionBoundaryBatch,
    seat_hidden: torch.Tensor | None,
    legal_ids: np.ndarray,
) -> tuple[np.ndarray, torch.Tensor | None]:
    if policy.heuristic_policy is not None:
        chosen_action = int(
            policy.heuristic_policy.choose_action(np.asarray(batch.obs[0], dtype=np.float32), np.asarray(legal_ids))
        )
        logits = np.full((GLOBAL_ACTION_SPACE_SIZE,), -1.0e9, dtype=np.float32)
        logits[chosen_action] = 0.0
        return logits, seat_hidden

    if policy.model is None or seat_hidden is None:
        raise RuntimeError(f"Replay policy {policy.label!r} is missing a model or hidden state")

    device = torch.device("cpu")
    acting_seat = int(batch.actor[0])
    with torch.inference_mode():
        logits_tensor, _value_tensor, next_seat_hidden = policy.model.forward_seat_aware(
            torch.as_tensor(np.asarray(batch.obs, dtype=np.float32), device=device),
            torch.as_tensor([acting_seat], device=device, dtype=torch.long),
            seat_hidden,
        )
    logits = logits_tensor[0].detach().cpu().numpy().astype(np.float32, copy=False)
    return logits, next_seat_hidden


def _build_step_diff(
    *,
    step_index: int,
    expected_step: ReplayStep,
    legal_ids: np.ndarray,
    logits_a: np.ndarray,
    logits_b: np.ndarray,
    top_actions: int,
    action_catalog: ActionCatalog | None,
) -> dict[str, Any]:
    legal_mask = np.zeros((logits_a.shape[0],), dtype=bool)
    legal_mask[np.asarray(legal_ids, dtype=np.int64)] = True
    stacked_logits = np.stack((logits_a, logits_b), axis=0)
    stacked_mask = np.stack((legal_mask, legal_mask), axis=0)
    log_probs = masked_log_softmax(stacked_logits, stacked_mask)
    safe_log_probs = np.where(stacked_mask, log_probs, 0.0)
    probs = np.exp(safe_log_probs.astype(np.float64, copy=False))

    kl_divergence_ab = float(np.sum(probs[0] * (safe_log_probs[0] - safe_log_probs[1]), dtype=np.float64))
    kl_divergence_ba = float(np.sum(probs[1] * (safe_log_probs[1] - safe_log_probs[0]), dtype=np.float64))
    probability_delta = probs[1] - probs[0]
    total_variation = float(0.5 * np.sum(np.abs(probability_delta), dtype=np.float64))
    abs_probability_delta = np.abs(probability_delta)
    legal_action_indices = np.flatnonzero(legal_mask)
    ranked_action_indices = legal_action_indices[np.argsort(abs_probability_delta[legal_action_indices])[::-1]]
    policy_a_top_action = _top_action_payload(
        probabilities=probs[0],
        legal_indices=legal_action_indices,
        action_catalog=action_catalog,
    )
    policy_b_top_action = _top_action_payload(
        probabilities=probs[1],
        legal_indices=legal_action_indices,
        action_catalog=action_catalog,
    )
    policy_b_top_action_id = int(policy_b_top_action["action"])

    return {
        "step_index": int(step_index),
        "decision_id": int(expected_step.decision_id),
        "actor": int(expected_step.actor),
        "recorded_action": int(expected_step.action),
        "recorded_action_detail": _action_descriptor(int(expected_step.action), action_catalog=action_catalog),
        "total_variation": total_variation,
        "kl_divergence_ab": kl_divergence_ab,
        "kl_divergence_ba": kl_divergence_ba,
        "max_abs_probability_delta": float(np.max(abs_probability_delta[legal_action_indices], initial=0.0)),
        "policy_a_recorded_action_probability": float(probs[0, int(expected_step.action)]),
        "policy_b_recorded_action_probability": float(probs[1, int(expected_step.action)]),
        "policy_a_probability_on_policy_b_top_action": float(probs[0, policy_b_top_action_id]),
        "policy_a_rank_of_policy_b_top_action": _rank_of_action(
            probabilities=probs[0],
            legal_indices=legal_action_indices,
            action_id=policy_b_top_action_id,
        ),
        "policy_a_matches_policy_b_top_action": int(policy_a_top_action["action"]) == policy_b_top_action_id,
        "policy_a_matches_policy_b_top_action_family": (
            policy_a_top_action.get("family") == policy_b_top_action.get("family")
            if "family" in policy_a_top_action and "family" in policy_b_top_action
            else False
        ),
        "policy_a_top_action": policy_a_top_action,
        "policy_b_top_action": policy_b_top_action,
        "top_action_deltas": [
            {
                **_action_descriptor(int(action_index), action_catalog=action_catalog),
                "probability_a": float(probs[0, action_index]),
                "probability_b": float(probs[1, action_index]),
                "probability_delta_b_minus_a": float(probability_delta[action_index]),
                "abs_probability_delta": float(abs_probability_delta[action_index]),
            }
            for action_index in ranked_action_indices.tolist()[:top_actions]
        ],
    }


def _top_action_payload(
    *,
    probabilities: np.ndarray,
    legal_indices: np.ndarray,
    action_catalog: ActionCatalog | None,
) -> dict[str, Any]:
    if legal_indices.size == 0:
        raise RuntimeError("Replay inspection requires at least one legal action per compared step")
    top_action = int(legal_indices[np.argmax(probabilities[legal_indices])])
    return {
        **_action_descriptor(top_action, action_catalog=action_catalog),
        "probability": float(probabilities[top_action]),
    }


def _rank_of_action(*, probabilities: np.ndarray, legal_indices: np.ndarray, action_id: int) -> int:
    legal_probabilities = probabilities[legal_indices]
    sorted_indices = legal_indices[np.argsort(legal_probabilities)[::-1]]
    positions = np.flatnonzero(sorted_indices == int(action_id))
    if positions.size == 0:
        raise RuntimeError(f"Expected action {action_id} to be legal in replay inspection")
    return int(positions[0]) + 1


def _action_descriptor(action_id: int, *, action_catalog: ActionCatalog | None) -> dict[str, Any]:
    payload: dict[str, Any] = {"action": int(action_id)}
    if action_catalog is None:
        return payload
    decoded = action_catalog.decode(int(action_id))
    payload["family"] = decoded.family
    if decoded.hand_index is not None:
        payload["hand_index"] = int(decoded.hand_index)
    if decoded.stage_slot is not None:
        payload["stage_slot"] = int(decoded.stage_slot)
    if decoded.from_slot is not None:
        payload["from_slot"] = int(decoded.from_slot)
    if decoded.to_slot is not None:
        payload["to_slot"] = int(decoded.to_slot)
    if decoded.slot is not None:
        payload["slot"] = int(decoded.slot)
    if decoded.attack_type is not None:
        payload["attack_type"] = str(decoded.attack_type)
    if decoded.index is not None:
        payload["index"] = int(decoded.index)
    return payload


def _summarize_step_diffs(step_diffs: Sequence[dict[str, Any]], *, top_k: int) -> dict[str, Any]:
    if not step_diffs:
        return {
            "compared_steps": 0,
            "top_k": int(top_k),
            "max_total_variation": 0.0,
            "mean_total_variation": 0.0,
            "median_total_variation": 0.0,
            "max_abs_probability_delta": 0.0,
            "policy_a_matches_policy_b_top_action_rate": 0.0,
            "policy_a_matches_policy_b_top_action_family_rate": 0.0,
            "policy_a_mean_probability_on_policy_b_top_action": 0.0,
            "policy_a_median_rank_of_policy_b_top_action": 0.0,
            "top_action_family_confusions": [],
        }

    total_variation = np.asarray([float(item["total_variation"]) for item in step_diffs], dtype=np.float64)
    max_abs_probability_delta = np.asarray(
        [float(item["max_abs_probability_delta"]) for item in step_diffs],
        dtype=np.float64,
    )
    policy_a_matches_policy_b_top_action = np.asarray(
        [bool(item["policy_a_matches_policy_b_top_action"]) for item in step_diffs],
        dtype=np.float64,
    )
    policy_a_matches_policy_b_top_action_family = np.asarray(
        [bool(item["policy_a_matches_policy_b_top_action_family"]) for item in step_diffs],
        dtype=np.float64,
    )
    policy_a_probability_on_policy_b_top_action = np.asarray(
        [float(item["policy_a_probability_on_policy_b_top_action"]) for item in step_diffs],
        dtype=np.float64,
    )
    policy_a_rank_of_policy_b_top_action = np.asarray(
        [float(item["policy_a_rank_of_policy_b_top_action"]) for item in step_diffs],
        dtype=np.float64,
    )
    confusion_counter: Counter[tuple[str, str]] = Counter()
    for item in step_diffs:
        policy_b_family = str(item["policy_b_top_action"].get("family", "unknown"))
        policy_a_family = str(item["policy_a_top_action"].get("family", "unknown"))
        confusion_counter[(policy_b_family, policy_a_family)] += 1
    return {
        "compared_steps": len(step_diffs),
        "top_k": int(top_k),
        "max_total_variation": float(np.max(total_variation)),
        "mean_total_variation": float(np.mean(total_variation)),
        "median_total_variation": float(np.median(total_variation)),
        "max_abs_probability_delta": float(np.max(max_abs_probability_delta)),
        "policy_a_matches_policy_b_top_action_rate": float(np.mean(policy_a_matches_policy_b_top_action)),
        "policy_a_matches_policy_b_top_action_family_rate": float(np.mean(policy_a_matches_policy_b_top_action_family)),
        "policy_a_mean_probability_on_policy_b_top_action": float(np.mean(policy_a_probability_on_policy_b_top_action)),
        "policy_a_median_rank_of_policy_b_top_action": float(np.median(policy_a_rank_of_policy_b_top_action)),
        "top_action_family_confusions": [
            {
                "policy_b_family": policy_b_family,
                "policy_a_family": policy_a_family,
                "count": int(count),
            }
            for (policy_b_family, policy_a_family), count in confusion_counter.most_common(5)
        ],
    }


def _top_step_diffs(step_diffs: Sequence[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    if top_k == 0:
        return []
    ranked = sorted(
        step_diffs,
        key=lambda item: (
            float(item["total_variation"]),
            float(item["max_abs_probability_delta"]),
            -int(item["step_index"]),
        ),
        reverse=True,
    )
    return list(ranked[:top_k])


def _canonical_float(value: Any) -> float:
    scalar = float(np.float32(value))
    return scalar if math.isfinite(scalar) else scalar


def _format_policy_source(policy_report: Mapping[str, Any]) -> str:
    weights_path = policy_report.get("weights_path")
    if isinstance(weights_path, str) and weights_path:
        return weights_path
    return str(policy_report.get("kind", "unknown"))


def _format_action_descriptor(payload: Mapping[str, Any]) -> str:
    parts = [f"a{int(payload['action'])}"]
    family = payload.get("family")
    if family is None:
        return "".join(parts)
    detail_parts: list[str] = [str(family)]
    for key in ("hand_index", "stage_slot", "from_slot", "to_slot", "slot", "attack_type", "index"):
        value = payload.get(key)
        if value is not None:
            detail_parts.append(f"{key}={value}")
    return f"{parts[0]}[{', '.join(detail_parts)}]"


__all__ = [
    "LoadedReplayPolicy",
    "format_replay_inspection_report",
    "inspect_replay_bundle",
    "write_replay_inspection_report",
]
