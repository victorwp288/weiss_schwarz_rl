"""Replay inspection helpers for comparing policy distributions on a recorded replay."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from weiss_rl.config import StackConfig, load_stack_config
from weiss_rl.model import GLOBAL_ACTION_SPACE_SIZE
from weiss_rl.replay.bundles import load_replay_bundle
from weiss_rl.replay.inspection_policy_execution import (
    forward_policy as _forward_policy,
)
from weiss_rl.replay.inspection_policy_execution import (
    policy_action_surface_batch_and_ids as _policy_action_surface_batch_and_ids,
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
from weiss_rl.replay.inspection_step_diffs import build_step_diff as _build_step_diff
from weiss_rl.replay.inspection_summaries import (
    summarize_step_diffs as _summarize_step_diffs,
)
from weiss_rl.replay.inspection_summaries import (
    summarize_trajectory_records as _summarize_trajectory_records,
)
from weiss_rl.replay.inspection_summaries import (
    top_step_diffs as _top_step_diffs,
)
from weiss_rl.replay.inspection_trajectory_records import build_trajectory_record as _build_trajectory_record
from weiss_rl.replay.rerun_validation import legal_ids_for_env_row as _legal_ids_for_env_row
from weiss_rl.replay.rerun_validation import observation_dim as _observation_dim
from weiss_rl.replay.rerun_validation import pass_action_id_from_spec_bundle as _pass_action_id
from weiss_rl.replay.rerun_validation import require_initial_identity as _require_initial_identity
from weiss_rl.replay.rerun_validation import require_post_step_match as _require_post_step_match
from weiss_rl.replay.rerun_validation import require_pre_step_match as _require_pre_step_match
from weiss_rl.replay.rerun_validation import require_single_env_batch as _require_single_env_batch
from weiss_rl.replay.runner import ReplayEnvFactory, build_replay_env, require_supported_rerun_contract


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
        current_batch = _require_single_env_batch(
            env.reset(seed=meta.episode_seed64),
            context="reset",
            owner="Replay inspection",
        )
        _require_initial_identity(meta=meta, batch=current_batch)

        observation_dim = _observation_dim(current_batch, owner="Replay inspection")
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
                owner="Replay inspection",
            )

            raw_legal_ids = _legal_ids_for_env_row(current_batch, owner="Replay inspection")
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
                owner="Replay inspection",
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


__all__ = [
    "inspect_replay_bundle",
]
