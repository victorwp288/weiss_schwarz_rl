"""Replay inspection helpers for comparing policy distributions on a recorded replay."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from weiss_rl.config import StackConfig, load_stack_config
from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.core.masking import masked_log_softmax
from weiss_rl.core.observation_layout import (
    ObservationLayout,
    ObservationPlayerBlock,
    ObservationSlice,
    parse_observation_layout,
)
from weiss_rl.envs.decision_env import DecisionBoundaryBatch
from weiss_rl.eval.model_sampling import model_eval_logits_for_legal_ids
from weiss_rl.model import GLOBAL_ACTION_SPACE_SIZE
from weiss_rl.models.observation_contract import header_field_index
from weiss_rl.replay.bundles import ReplayBundleMeta, ReplayStep, compute_legal_fingerprint64, load_replay_bundle
from weiss_rl.replay.inspection_policy_loading import (
    LoadedReplayPolicy,
)
from weiss_rl.replay.inspection_policy_loading import (
    load_action_catalog as _load_action_catalog,
)
from weiss_rl.replay.inspection_policy_loading import (
    load_policy as _load_policy,
)
from weiss_rl.replay.inspection_policy_loading import (
    load_run_spec_bundle as _load_run_spec_bundle,
)
from weiss_rl.replay.inspection_policy_loading import (
    normalize_config_hashes as _normalize_config_hashes,
)
from weiss_rl.replay.inspection_policy_loading import (
    opponent_context_index_for_policy as _opponent_context_index_for_policy,
)
from weiss_rl.replay.inspection_policy_loading import (
    resolve_registry as _resolve_registry,
)
from weiss_rl.replay.inspection_policy_loading import (
    snapshot_by_policy_id_or_imported_seed_suffix as _shared_snapshot_by_policy_id_or_imported_seed_suffix,
)
from weiss_rl.replay.inspection_summaries import (
    TRACKED_LEGAL_FAMILIES as _TRACKED_LEGAL_FAMILIES,
)
from weiss_rl.replay.inspection_summaries import (
    canonical_float as _canonical_float,
)
from weiss_rl.replay.inspection_summaries import (
    counter_items as _counter_items,
)
from weiss_rl.replay.inspection_summaries import (
    summarize_step_diffs as _summarize_step_diffs,
)
from weiss_rl.replay.inspection_summaries import (
    summarize_trajectory_records as _summarize_trajectory_records,
)
from weiss_rl.replay.inspection_summaries import (
    top_step_diffs as _top_step_diffs,
)
from weiss_rl.replay.inspector_report import format_replay_inspection_report, write_replay_inspection_report
from weiss_rl.replay.runner import ReplayEnvFactory, build_replay_env, require_supported_rerun_contract
from weiss_rl.runtime.components.action_surface import (
    filter_batch_main_move_only_rows_to_pass,
    filter_batch_mulligan_select_after_select,
    filter_batch_pass_when_attack_available,
)
from weiss_rl.runtime.components.legal_meta import action_catalog_indices


def _resolve_policy_weights_path(*, spec: str, run_dir: Path | None, registry: Any | None) -> tuple[Path, str]:
    from weiss_rl.replay.inspection_policy_loading import resolve_policy_weights_path

    return resolve_policy_weights_path(spec=spec, run_dir=run_dir, registry=registry)


def _snapshot_by_policy_id_or_imported_seed_suffix(*, registry: Any, policy_id: str) -> Any | None:
    return _shared_snapshot_by_policy_id_or_imported_seed_suffix(registry=registry, policy_id=policy_id)


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
    accepted_snapshot_config_hashes: Iterable[str] = (),
    opponent_context_policy_id: str | None = None,
    require_opponent_context_index: bool = False,
) -> dict[str, Any]:
    if top_k < 0:
        raise ValueError("top_k must be >= 0")
    if top_actions <= 0:
        raise ValueError("top_actions must be >= 1")

    bundle_path = Path(bundle_path).resolve()
    stack_config = load_stack_config(stack) if isinstance(stack, Path) else stack
    extra_accepted_hashes = _normalize_config_hashes(accepted_snapshot_config_hashes)
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
            extra_accepted_config_hashes=extra_accepted_hashes,
        )
        policy_b_loaded = _load_policy(
            spec=policy_b,
            stack=stack_config,
            observation_dim=observation_dim,
            action_dim=GLOBAL_ACTION_SPACE_SIZE,
            run_dir=resolved_run_dir,
            registry=registry,
            run_spec_bundle=run_spec_bundle,
            extra_accepted_config_hashes=extra_accepted_hashes,
        )

        device = torch.device("cpu")
        policy_a_hidden = (
            None if policy_a_loaded.model is None else policy_a_loaded.model.initial_seat_hidden(1, device=device)
        )
        policy_b_hidden = (
            None if policy_b_loaded.model is None else policy_b_loaded.model.initial_seat_hidden(1, device=device)
        )
        policy_a_opponent_context_index = _opponent_context_index_for_policy(
            policy=policy_a_loaded,
            opponent_context_policy_id=opponent_context_policy_id,
            require_nonzero=require_opponent_context_index,
        )
        policy_b_opponent_context_index = _opponent_context_index_for_policy(
            policy=policy_b_loaded,
            opponent_context_policy_id=opponent_context_policy_id,
            require_nonzero=require_opponent_context_index,
        )
        spec_hash256 = bytes.fromhex(meta.spec_hash256)

        step_diffs: list[dict[str, Any]] = []
        trajectory_records: list[dict[str, Any]] = []
        for step_index, expected_step in enumerate(steps):
            _require_pre_step_match(
                step_index=step_index,
                expected_step=expected_step,
                current_batch=current_batch,
                spec_hash256=spec_hash256,
            )

            raw_legal_ids = _legal_ids_for_env_row(current_batch)
            trajectory_records.append(
                _build_trajectory_record(
                    step_index=step_index,
                    expected_step=expected_step,
                    batch=current_batch,
                    raw_legal_ids=raw_legal_ids,
                    action_catalog=action_catalog,
                    spec_bundle=run_spec_bundle,
                )
            )
            policy_a_batch, policy_a_legal_ids = _policy_action_surface_batch_and_ids(
                policy=policy_a_loaded,
                stack=stack_config,
                batch=current_batch,
                legal_ids=raw_legal_ids,
                pass_action_id=_pass_action_id(run_spec_bundle),
            )
            policy_b_batch, policy_b_legal_ids = _policy_action_surface_batch_and_ids(
                policy=policy_b_loaded,
                stack=stack_config,
                batch=current_batch,
                legal_ids=raw_legal_ids,
                pass_action_id=_pass_action_id(run_spec_bundle),
            )
            logits_a, policy_a_hidden = _forward_policy(
                policy=policy_a_loaded,
                batch=policy_a_batch,
                seat_hidden=policy_a_hidden,
                legal_ids=policy_a_legal_ids,
                opponent_context_index=policy_a_opponent_context_index,
            )
            logits_b, policy_b_hidden = _forward_policy(
                policy=policy_b_loaded,
                batch=policy_b_batch,
                seat_hidden=policy_b_hidden,
                legal_ids=policy_b_legal_ids,
                opponent_context_index=policy_b_opponent_context_index,
            )
            step_diffs.append(
                _build_step_diff(
                    step_index=step_index,
                    expected_step=expected_step,
                    raw_legal_ids=raw_legal_ids,
                    legal_ids_a=policy_a_legal_ids,
                    legal_ids_b=policy_b_legal_ids,
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
            "trajectory_summary": _summarize_trajectory_records(trajectory_records),
            "top_differences": _top_step_diffs(step_diffs, top_k=top_k),
            "opponent_context": {
                "policy_id": None if opponent_context_policy_id is None else str(opponent_context_policy_id),
                "require_nonzero": bool(require_opponent_context_index),
                "policy_a_index": policy_a_opponent_context_index,
                "policy_b_index": policy_b_opponent_context_index,
            },
            "compared_steps": compared_steps,
        }
        return report
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


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


def _policy_action_surface_batch_and_ids(
    *,
    policy: LoadedReplayPolicy,
    stack: StackConfig,
    batch: DecisionBoundaryBatch,
    legal_ids: np.ndarray,
    pass_action_id: int,
) -> tuple[DecisionBoundaryBatch, np.ndarray]:
    """Mirror eval-time model-only action-surface guards for replay scoring."""

    if policy.model is None:
        return batch, legal_ids
    training_config = stack.config.training
    if training_config is None:
        return batch, legal_ids
    mulligan_guard = bool(getattr(training_config, "mulligan_force_confirm_after_select", False))
    main_move_guard = bool(getattr(training_config, "force_pass_over_main_move_only", False))
    attack_guard = bool(getattr(training_config, "force_attack_over_pass_when_attack_legal", False))
    if not mulligan_guard and not main_move_guard and not attack_guard:
        return batch, legal_ids
    action_catalog = getattr(policy.model, "action_catalog", None)
    if action_catalog is None:
        return batch, legal_ids

    filtered_batch = batch
    contract = getattr(policy.model, "_structured_observation_contract", None)
    layout = getattr(contract, "layout", None)
    field_index = None if layout is None else header_field_index(layout, "last_action_arg0")
    last_action_arg0_index = -1 if field_index is None else int(field_index)
    family_index, _attack_type_index = action_catalog_indices(action_catalog)
    if mulligan_guard:
        filtered_batch, _result = filter_batch_mulligan_select_after_select(
            filtered_batch,
            last_action_arg0_index=last_action_arg0_index,
            mulligan_select_family_id=int(family_index.get("mulligan_select", -1)),
            mulligan_confirm_family_id=int(family_index.get("mulligan_confirm", -1)),
        )
    if main_move_guard:
        filtered_batch, _result = filter_batch_main_move_only_rows_to_pass(
            filtered_batch,
            pass_action_id=int(pass_action_id),
            main_move_family_id=int(family_index.get("main_move", -1)),
        )
    if attack_guard:
        filtered_batch, _result = filter_batch_pass_when_attack_available(
            filtered_batch,
            pass_action_id=int(pass_action_id),
            attack_family_id=int(family_index.get("attack", -1)),
        )
    if filtered_batch.ids_offsets is None:
        return batch, legal_ids
    filtered_ids, filtered_offsets = filtered_batch.ids_offsets
    return (
        filtered_batch,
        np.asarray(filtered_ids[int(filtered_offsets[0]) : int(filtered_offsets[1])], dtype=np.uint32),
    )


def _pass_action_id(spec_bundle: Mapping[str, Any] | None) -> int:
    if spec_bundle is None:
        return 51
    action = spec_bundle.get("action")
    if not isinstance(action, Mapping):
        return 51
    return int(action.get("pass_action_id", 51))


def _build_trajectory_record(
    *,
    step_index: int,
    expected_step: ReplayStep,
    batch: DecisionBoundaryBatch,
    raw_legal_ids: np.ndarray,
    action_catalog: ActionCatalog | None,
    spec_bundle: Mapping[str, Any] | None,
) -> dict[str, Any]:
    obs = np.asarray(batch.obs[0])
    layout = _load_observation_layout(spec_bundle)
    recorded_action = _action_descriptor(int(expected_step.action), action_catalog=action_catalog)
    legal_family_counts = _legal_family_counts(raw_legal_ids=raw_legal_ids, action_catalog=action_catalog)
    payload: dict[str, Any] = {
        "step_index": int(step_index),
        "decision_id": int(expected_step.decision_id),
        "actor": int(expected_step.actor),
        "recorded_action": int(expected_step.action),
        "recorded_action_family": str(recorded_action.get("family", "unknown")),
        "raw_legal_action_count": int(np.asarray(raw_legal_ids).shape[0]),
        "has_nonpass_legal": bool(np.any(np.asarray(raw_legal_ids, dtype=np.int64) != _pass_action_id(spec_bundle))),
        "legal_family_counts": _counter_items(legal_family_counts, key_names=("family",)),
    }
    for family in _TRACKED_LEGAL_FAMILIES:
        payload[f"has_legal_{family}"] = int(legal_family_counts.get(family, 0)) > 0
    if layout is not None:
        for field_name in ("phase", "decision_kind", "active_player", "decision_player"):
            field_index = header_field_index(layout, field_name)
            if field_index is not None and field_index < obs.shape[0]:
                payload[field_name] = _safe_int(obs[field_index])
        if layout.player_blocks:
            payload.update(
                _player_trajectory_fields(
                    obs=obs,
                    block=layout.player_blocks[0],
                    prefix="self",
                    action_catalog=action_catalog,
                    spec_bundle=spec_bundle,
                )
            )
        if len(layout.player_blocks) > 1:
            payload.update(
                _player_trajectory_fields(
                    obs=obs,
                    block=layout.player_blocks[1],
                    prefix="opponent",
                    action_catalog=action_catalog,
                    spec_bundle=spec_bundle,
                )
            )
    return payload


def _load_observation_layout(spec_bundle: Mapping[str, Any] | None) -> ObservationLayout | None:
    if spec_bundle is None:
        return None
    observation = spec_bundle.get("observation")
    if not isinstance(observation, Mapping):
        return None
    try:
        return parse_observation_layout(observation)
    except (TypeError, ValueError):
        return None


def _player_trajectory_fields(
    *,
    obs: np.ndarray,
    block: ObservationPlayerBlock,
    prefix: str,
    action_catalog: ActionCatalog | None,
    spec_bundle: Mapping[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for name in (
        "level_count",
        "clock_count",
        "deck_count",
        "hand_count",
        "stock_count",
        "waiting_room_count",
        "memory_count",
        "climax_count",
    ):
        value = _player_scalar(obs=obs, block=block, slice_name=name)
        if value is not None:
            payload[f"{prefix}_{name}"] = value
    stage_slice = _player_slice(block, "stage")
    if stage_slice is not None:
        payload[f"{prefix}_stage_occupied_count"] = _stage_occupied_count(
            obs=obs,
            stage_start=stage_slice.start,
            stage_length=stage_slice.length,
            action_catalog=action_catalog,
            spec_bundle=spec_bundle,
        )
    return payload


def _player_scalar(*, obs: np.ndarray, block: ObservationPlayerBlock, slice_name: str) -> int | None:
    observation_slice = _player_slice(block, slice_name)
    if observation_slice is None or observation_slice.start >= obs.shape[0]:
        return None
    return _safe_int(obs[observation_slice.start])


def _player_slice(block: ObservationPlayerBlock, slice_name: str) -> ObservationSlice | None:
    return next((current for current in block.slices if current.name == slice_name), None)


def _stage_occupied_count(
    *,
    obs: np.ndarray,
    stage_start: int,
    stage_length: int,
    action_catalog: ActionCatalog | None,
    spec_bundle: Mapping[str, Any] | None,
) -> int:
    stage_slots = int(action_catalog.max_stage) if action_catalog is not None else 5
    if stage_slots <= 0:
        return 0
    slot_width = max(int(stage_length) // stage_slots, 1)
    observation = spec_bundle.get("observation") if isinstance(spec_bundle, Mapping) else None
    sentinel_empty = 0
    sentinel_hidden = -1
    if isinstance(observation, Mapping):
        sentinel_empty = int(observation.get("sentinel_empty_card", sentinel_empty))
        sentinel_hidden = int(observation.get("sentinel_hidden", sentinel_hidden))
    occupied = 0
    for slot_index in range(stage_slots):
        index = int(stage_start) + slot_index * slot_width
        if index >= obs.shape[0] or index >= int(stage_start) + int(stage_length):
            break
        card_value = _safe_int(obs[index])
        if card_value not in (sentinel_empty, sentinel_hidden):
            occupied += 1
    return occupied


def _legal_family_counts(*, raw_legal_ids: np.ndarray, action_catalog: ActionCatalog | None) -> Counter[str]:
    counter: Counter[str] = Counter()
    if action_catalog is None:
        return counter
    for action_id in np.asarray(raw_legal_ids, dtype=np.int64).tolist():
        try:
            family = action_catalog.decode(int(action_id)).family
        except ValueError:
            family = "unknown"
        counter[str(family)] += 1
    return counter


def _safe_int(value: Any) -> int:
    return int(np.asarray(value).reshape(()).item())


def _forward_policy(
    *,
    policy: LoadedReplayPolicy,
    batch: DecisionBoundaryBatch,
    seat_hidden: torch.Tensor | None,
    legal_ids: np.ndarray,
    opponent_context_index: int | None = None,
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
        logits, next_seat_hidden = model_eval_logits_for_legal_ids(
            model=policy.model,
            batch=batch,
            current_seat=acting_seat,
            seat_hidden=seat_hidden,
            legal_ids=legal_ids,
            action_dim=GLOBAL_ACTION_SPACE_SIZE,
            device=device,
            opponent_context_index=opponent_context_index,
        )
    return logits, next_seat_hidden


def _build_step_diff(
    *,
    step_index: int,
    expected_step: ReplayStep,
    raw_legal_ids: np.ndarray,
    legal_ids_a: np.ndarray,
    legal_ids_b: np.ndarray,
    logits_a: np.ndarray,
    logits_b: np.ndarray,
    top_actions: int,
    action_catalog: ActionCatalog | None,
) -> dict[str, Any]:
    legal_mask_a = _legal_mask_from_ids(logits_a.shape[0], legal_ids_a)
    legal_mask_b = _legal_mask_from_ids(logits_b.shape[0], legal_ids_b)
    union_mask = legal_mask_a | legal_mask_b
    stacked_logits = np.stack((logits_a, logits_b), axis=0)
    stacked_mask = np.stack((legal_mask_a, legal_mask_b), axis=0)
    log_probs = masked_log_softmax(stacked_logits, stacked_mask)
    probs = np.zeros_like(log_probs, dtype=np.float64)
    probs[stacked_mask] = np.exp(log_probs[stacked_mask].astype(np.float64, copy=False))

    kl_divergence_ab = _kl_divergence(probs[0], probs[1])
    kl_divergence_ba = _kl_divergence(probs[1], probs[0])
    probability_delta = probs[1] - probs[0]
    total_variation = float(0.5 * np.sum(np.abs(probability_delta[union_mask]), dtype=np.float64))
    abs_probability_delta = np.abs(probability_delta)
    legal_action_indices_a = np.flatnonzero(legal_mask_a)
    legal_action_indices_b = np.flatnonzero(legal_mask_b)
    union_action_indices = np.flatnonzero(union_mask)
    ranked_action_indices = union_action_indices[np.argsort(abs_probability_delta[union_action_indices])[::-1]]
    policy_a_top_action = _top_action_payload(
        probabilities=probs[0],
        legal_indices=legal_action_indices_a,
        action_catalog=action_catalog,
    )
    policy_b_top_action = _top_action_payload(
        probabilities=probs[1],
        legal_indices=legal_action_indices_b,
        action_catalog=action_catalog,
    )
    policy_a_top_action_id = int(policy_a_top_action["action"])
    policy_b_top_action_id = int(policy_b_top_action["action"])
    raw_legal_action_count = int(np.asarray(raw_legal_ids).shape[0])
    policy_a_legal_action_count = int(legal_action_indices_a.shape[0])
    policy_b_legal_action_count = int(legal_action_indices_b.shape[0])
    policy_a_top_logit_margin = _top_margin(values=logits_a, legal_indices=legal_action_indices_a)
    policy_a_top_probability_margin = _top_margin(values=probs[0], legal_indices=legal_action_indices_a)
    policy_a_b_top_action_logit_gap = _gap_from_top_to_action(
        values=logits_a,
        legal_indices=legal_action_indices_a,
        action_id=policy_b_top_action_id,
    )
    policy_a_b_top_action_same_family_logit_margin = _same_family_margin_to_action(
        values=logits_a,
        legal_indices=legal_action_indices_a,
        action_id=policy_b_top_action_id,
        action_catalog=action_catalog,
    )
    policy_a_family_masses = _family_probability_masses(
        probabilities=probs[0],
        legal_indices=legal_action_indices_a,
        action_catalog=action_catalog,
    )
    policy_b_family_masses = _family_probability_masses(
        probabilities=probs[1],
        legal_indices=legal_action_indices_b,
        action_catalog=action_catalog,
    )
    policy_a_top_family = str(policy_a_top_action.get("family", "unknown"))
    policy_b_top_family = str(policy_b_top_action.get("family", "unknown"))

    return {
        "step_index": int(step_index),
        "decision_id": int(expected_step.decision_id),
        "actor": int(expected_step.actor),
        "recorded_action": int(expected_step.action),
        "recorded_action_detail": _action_descriptor(int(expected_step.action), action_catalog=action_catalog),
        "raw_legal_action_count": raw_legal_action_count,
        "policy_a_legal_action_count": policy_a_legal_action_count,
        "policy_b_legal_action_count": policy_b_legal_action_count,
        "policy_a_legal_surface_removed_action_count": max(raw_legal_action_count - policy_a_legal_action_count, 0),
        "policy_b_legal_surface_removed_action_count": max(raw_legal_action_count - policy_b_legal_action_count, 0),
        "policy_a_legal_surface_is_filtered": policy_a_legal_action_count < raw_legal_action_count,
        "policy_b_legal_surface_is_filtered": policy_b_legal_action_count < raw_legal_action_count,
        "policy_b_top_action_legal_for_policy_a": bool(legal_mask_a[policy_b_top_action_id]),
        "policy_a_top_action_legal_for_policy_b": bool(legal_mask_b[policy_a_top_action_id]),
        "total_variation": total_variation,
        "kl_divergence_ab": kl_divergence_ab,
        "kl_divergence_ba": kl_divergence_ba,
        "max_abs_probability_delta": float(np.max(abs_probability_delta[union_action_indices], initial=0.0)),
        "policy_a_recorded_action_probability": float(probs[0, int(expected_step.action)]),
        "policy_b_recorded_action_probability": float(probs[1, int(expected_step.action)]),
        "policy_a_probability_on_policy_b_top_action": float(probs[0, policy_b_top_action_id]),
        "policy_a_probability_on_policy_b_top_action_family": float(
            policy_a_family_masses.get(policy_b_top_family, 0.0)
        ),
        "policy_a_top_logit_margin": policy_a_top_logit_margin,
        "policy_a_top_probability_margin": policy_a_top_probability_margin,
        "policy_a_gap_from_top_logit_to_policy_b_top_action": policy_a_b_top_action_logit_gap,
        "policy_a_policy_b_top_action_same_family_logit_margin": (policy_a_b_top_action_same_family_logit_margin),
        "policy_a_top_action_family_probability": float(policy_a_family_masses.get(policy_a_top_family, 0.0)),
        "policy_b_top_action_family_probability": float(policy_b_family_masses.get(policy_b_top_family, 0.0)),
        "policy_a_rank_of_policy_b_top_action": _rank_of_action(
            probabilities=probs[0],
            legal_indices=legal_action_indices_a,
            action_id=policy_b_top_action_id,
        ),
        "policy_a_matches_policy_b_top_action": policy_a_top_action_id == policy_b_top_action_id,
        "policy_a_matches_policy_b_top_action_family": (
            policy_a_top_action.get("family") == policy_b_top_action.get("family")
            if "family" in policy_a_top_action and "family" in policy_b_top_action
            else False
        ),
        "policy_a_top_action": policy_a_top_action,
        "policy_b_top_action": policy_b_top_action,
        "policy_a_family_probability_masses": policy_a_family_masses,
        "policy_b_family_probability_masses": policy_b_family_masses,
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


def _legal_mask_from_ids(action_dim: int, legal_ids: np.ndarray) -> np.ndarray:
    legal_mask = np.zeros((int(action_dim),), dtype=bool)
    legal_ids_array = np.asarray(legal_ids, dtype=np.int64)
    if legal_ids_array.size:
        legal_mask[legal_ids_array] = True
    return legal_mask


def _kl_divergence(probs_p: np.ndarray, probs_q: np.ndarray) -> float:
    support = probs_p > 0.0
    if not bool(np.any(support)):
        return 0.0
    q = np.maximum(probs_q[support], np.finfo(np.float64).tiny)
    p = probs_p[support]
    return float(np.sum(p * (np.log(p) - np.log(q)), dtype=np.float64))


def _family_probability_masses(
    *,
    probabilities: np.ndarray,
    legal_indices: np.ndarray,
    action_catalog: ActionCatalog | None,
) -> dict[str, float]:
    if action_catalog is None:
        return {}
    masses: dict[str, float] = {}
    for action_index in legal_indices.tolist():
        family = action_catalog.decode(int(action_index)).family
        masses[family] = masses.get(family, 0.0) + float(probabilities[int(action_index)])
    return dict(sorted(masses.items(), key=lambda item: (-item[1], item[0])))


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
        return int(legal_indices.shape[0]) + 1
    return int(positions[0]) + 1


def _top_margin(*, values: np.ndarray, legal_indices: np.ndarray) -> float | None:
    if legal_indices.size < 2:
        return None
    legal_values = np.asarray(values[legal_indices], dtype=np.float64)
    if not np.all(np.isfinite(legal_values)):
        return None
    top_two = np.sort(legal_values)[-2:]
    return float(top_two[-1] - top_two[-2])


def _gap_from_top_to_action(*, values: np.ndarray, legal_indices: np.ndarray, action_id: int) -> float | None:
    if legal_indices.size == 0 or not bool(np.any(legal_indices == int(action_id))):
        return None
    legal_values = np.asarray(values[legal_indices], dtype=np.float64)
    action_value = float(values[int(action_id)])
    if not np.all(np.isfinite(legal_values)) or not math.isfinite(action_value):
        return None
    return float(np.max(legal_values) - action_value)


def _same_family_margin_to_action(
    *,
    values: np.ndarray,
    legal_indices: np.ndarray,
    action_id: int,
    action_catalog: ActionCatalog | None,
) -> float | None:
    if action_catalog is None or legal_indices.size == 0 or not bool(np.any(legal_indices == int(action_id))):
        return None
    action_value = float(values[int(action_id)])
    if not math.isfinite(action_value):
        return None
    target_family = action_catalog.decode(int(action_id)).family
    same_family_legal_indices = np.asarray(
        [
            int(legal_id)
            for legal_id in legal_indices.tolist()
            if int(legal_id) != int(action_id) and action_catalog.decode(int(legal_id)).family == target_family
        ],
        dtype=np.int64,
    )
    if same_family_legal_indices.size == 0:
        return None
    competitor_values = np.asarray(values[same_family_legal_indices], dtype=np.float64)
    if not np.all(np.isfinite(competitor_values)):
        return None
    return float(action_value - np.max(competitor_values))


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


__all__ = [
    "LoadedReplayPolicy",
    "format_replay_inspection_report",
    "inspect_replay_bundle",
    "write_replay_inspection_report",
]
