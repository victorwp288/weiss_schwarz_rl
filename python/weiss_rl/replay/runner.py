"""Replay verification runner."""

from __future__ import annotations

import json
from dataclasses import asdict
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from weiss_rl.envs.decision_env import DecisionBoundaryBatch, DecisionBoundaryEnv
from weiss_rl.replay.bundles import (
    ReplayBundleMeta,
    ReplayRerunContract,
    ReplayStep,
    compute_legal_fingerprint64,
    load_replay_bundle,
)

ReplayEnvFactory = Callable[[ReplayRerunContract], Any]


def verify_replay_bundle(
    *,
    bundle_path: Path,
    report_path: Path | None = None,
    env_factory: ReplayEnvFactory | None = None,
) -> dict[str, Any]:
    meta, steps, fault = load_replay_bundle(bundle_path)
    resolved_report_path = _resolve_report_path(bundle_path, report_path)
    report = _base_report(
        bundle_path=bundle_path,
        report_path=resolved_report_path,
        meta=meta,
        steps=steps,
        fault=fault,
    )

    contract = meta.rerun_contract
    if not meta.rerun_supported or contract is None:
        message = meta.rerun_blocker or "Replay bundle is not rerunnable because the bundle has no rerun contract"
        report.update(
            {
                "status": "unsupported",
                "matched": False,
                "compared_steps": 0,
                "error": message,
            }
        )
        _write_report(resolved_report_path, report)
        raise RuntimeError(message)

    env = None
    compared_steps = 0
    try:
        env = _build_env(contract, env_factory=env_factory)
        current_batch = _require_single_env_batch(env.reset(seed=meta.episode_seed64), context="reset")
        _verify_initial_identity(meta=meta, batch=current_batch, report=report)

        spec_hash256 = bytes.fromhex(meta.spec_hash256)
        for step_index, expected_step in enumerate(steps):
            _verify_pre_step(
                step_index=step_index,
                expected_step=expected_step,
                current_batch=current_batch,
                spec_hash256=spec_hash256,
                report=report,
                report_path=resolved_report_path,
            )
            next_batch = _require_single_env_batch(
                env.step(np.array([expected_step.action], dtype=np.uint32)),
                context=f"step[{step_index}]",
            )
            _verify_post_step(
                step_index=step_index,
                expected_step=expected_step,
                next_batch=next_batch,
                report=report,
                report_path=resolved_report_path,
            )
            compared_steps = step_index + 1
            if (expected_step.terminated or expected_step.truncated) and compared_steps != len(steps):
                _fail_verification(
                    report=report,
                    report_path=resolved_report_path,
                    compared_steps=compared_steps,
                    field="episode_termination",
                    expected={"last_step_index": len(steps) - 1},
                    observed={"terminated_at_step_index": step_index},
                    message="Recorded replay bundle contains additional steps after termination",
                )
            current_batch = next_batch

        report.update(
            {
                "status": "success",
                "matched": True,
                "compared_steps": compared_steps,
            }
        )
        _write_report(resolved_report_path, report)
        return report
    except Exception as exc:
        if report.get("status") is None:
            report.update(
                {
                    "status": "error",
                    "matched": False,
                    "compared_steps": compared_steps,
                    "error": str(exc),
                }
            )
            _write_report(resolved_report_path, report)
        raise
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def _build_env(contract: ReplayRerunContract, *, env_factory: ReplayEnvFactory | None) -> Any:
    if env_factory is not None:
        return env_factory(contract)
    return DecisionBoundaryEnv.create(
        legality="ids_offsets",
        engine_status_policy="passthrough",
        num_envs=1,
        max_decisions=int(contract.max_decisions),
        max_ticks=int(contract.max_ticks),
        observation_visibility=contract.observation_visibility,
    )


def _require_single_env_batch(batch: DecisionBoundaryBatch, *, context: str) -> DecisionBoundaryBatch:
    if batch.num_envs != 1:
        raise RuntimeError(f"Replay verification expects a single-env batch from {context}, got {batch.num_envs}")
    return batch


def _verify_initial_identity(*, meta: ReplayBundleMeta, batch: DecisionBoundaryBatch, report: dict[str, Any]) -> None:
    observed_seed = int(batch.episode_seed[0])
    if observed_seed != meta.episode_seed64:
        raise RuntimeError(
            f"Replay reset seed mismatch: expected episode_seed64={meta.episode_seed64}, got {observed_seed}"
        )

    if meta.simulator_episode_key_u64 is None:
        return

    observed_episode_key = int(batch.episode_key[0])
    if observed_episode_key != meta.simulator_episode_key_u64:
        raise RuntimeError(
            "Replay reset episode_key mismatch: "
            f"expected simulator episode key {meta.simulator_episode_key_u64}, got {observed_episode_key}"
        )
    report["verified_simulator_episode_key_u64"] = observed_episode_key


def _verify_pre_step(
    *,
    step_index: int,
    expected_step: ReplayStep,
    current_batch: DecisionBoundaryBatch,
    spec_hash256: bytes,
    report: dict[str, Any],
    report_path: Path,
) -> None:
    actual_decision_id = int(current_batch.decision_id[0])
    _expect_equal(
        report=report,
        report_path=report_path,
        compared_steps=step_index,
        field="decision_id",
        expected=expected_step.decision_id,
        observed=actual_decision_id,
        message=f"Replay decision_id mismatch at step {step_index}",
    )

    actual_actor = int(current_batch.actor[0])
    _expect_equal(
        report=report,
        report_path=report_path,
        compared_steps=step_index,
        field="actor",
        expected=expected_step.actor,
        observed=actual_actor,
        message=f"Replay actor mismatch at step {step_index}",
    )

    if current_batch.ids_offsets is None:
        raise RuntimeError("Replay verification requires ids_offsets legality in the rerun environment")
    legal_ids, legal_offsets = current_batch.ids_offsets
    row_legal_ids = legal_ids[int(legal_offsets[0]) : int(legal_offsets[1])]
    actual_fingerprint = compute_legal_fingerprint64(
        spec_hash256=spec_hash256,
        decision_id=actual_decision_id,
        legal_ids=np.asarray(row_legal_ids),
    )
    _expect_equal(
        report=report,
        report_path=report_path,
        compared_steps=step_index,
        field="legal_fingerprint64",
        expected=expected_step.legal_fingerprint64,
        observed=actual_fingerprint,
        message=f"Replay legal fingerprint mismatch at step {step_index}",
    )


def _verify_post_step(
    *,
    step_index: int,
    expected_step: ReplayStep,
    next_batch: DecisionBoundaryBatch,
    report: dict[str, Any],
    report_path: Path,
) -> None:
    _expect_equal(
        report=report,
        report_path=report_path,
        compared_steps=step_index,
        field="reward",
        expected=_canonical_float(expected_step.reward),
        observed=_canonical_float(next_batch.reward[0]),
        message=f"Replay reward mismatch at step {step_index}",
    )
    _expect_equal(
        report=report,
        report_path=report_path,
        compared_steps=step_index,
        field="terminated",
        expected=bool(expected_step.terminated),
        observed=bool(next_batch.terminated[0]),
        message=f"Replay terminated mismatch at step {step_index}",
    )
    _expect_equal(
        report=report,
        report_path=report_path,
        compared_steps=step_index,
        field="truncated",
        expected=bool(expected_step.truncated),
        observed=bool(next_batch.truncated[0]),
        message=f"Replay truncated mismatch at step {step_index}",
    )
    _expect_equal(
        report=report,
        report_path=report_path,
        compared_steps=step_index,
        field="engine_status",
        expected=int(expected_step.engine_status),
        observed=int(next_batch.engine_status[0]),
        message=f"Replay engine_status mismatch at step {step_index}",
    )


def _expect_equal(
    *,
    report: dict[str, Any],
    report_path: Path,
    compared_steps: int,
    field: str,
    expected: Any,
    observed: Any,
    message: str,
) -> None:
    if expected == observed:
        return
    _fail_verification(
        report=report,
        report_path=report_path,
        compared_steps=compared_steps,
        field=field,
        expected=expected,
        observed=observed,
        message=message,
    )


def _fail_verification(
    *,
    report: dict[str, Any],
    report_path: Path,
    compared_steps: int,
    field: str,
    expected: Any,
    observed: Any,
    message: str,
) -> None:
    report.update(
        {
            "status": "mismatch",
            "matched": False,
            "compared_steps": compared_steps,
            "mismatch": {
                "field": field,
                "expected": expected,
                "observed": observed,
            },
            "error": message,
        }
    )
    _write_report(report_path, report)
    raise RuntimeError(message)


def _base_report(
    *,
    bundle_path: Path,
    report_path: Path,
    meta: ReplayBundleMeta,
    steps: list[ReplayStep],
    fault: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "bundle_path": str(bundle_path),
        "report_path": str(report_path),
        "replay_key64": f"{meta.replay_key64:016x}",
        "episode_key64": int(meta.episode_key64),
        "episode_seed64": int(meta.episode_seed64),
        "schema_version": int(meta.schema_version),
        "rerun_supported": bool(meta.rerun_supported),
        "rerun_contract": None if meta.rerun_contract is None else asdict(meta.rerun_contract),
        "expected_steps": len(steps),
        "fault_present": fault is not None,
    }


def _resolve_report_path(bundle_path: Path, report_path: Path | None) -> Path:
    if report_path is not None:
        return report_path
    return bundle_path.parent / "replay_verification.json"


def _write_report(report_path: Path, report: dict[str, Any]) -> None:
    payload = dict(report)
    payload["report_path"] = str(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _canonical_float(value: Any) -> float:
    scalar = float(np.float32(value))
    return scalar if math.isfinite(scalar) else scalar
