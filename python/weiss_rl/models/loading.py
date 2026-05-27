from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from weiss_rl.config import StackConfig
from weiss_rl.model import PolicyValueModel, build_policy_value_model
from weiss_rl.models.state_dict_compat import load_model_state_dict_with_context_compat


def restore_model_guidance_from_payload(
    model: Any | None,
    payload: Mapping[str, Any],
) -> None:
    if model is None:
        return
    set_bias_scale = getattr(model, "set_public_heuristic_logit_bias_scale", None)
    if not callable(set_bias_scale):
        return
    learner_scale = payload.get("public_heuristic_logit_bias_scale")
    actor_scale = payload.get("public_heuristic_actor_logit_bias_scale")
    if learner_scale is None and actor_scale is None:
        return
    resolved_learner_scale = None if learner_scale is None else float(learner_scale)
    resolved_actor_scale = None if actor_scale is None else float(actor_scale)
    if resolved_learner_scale is None and resolved_actor_scale is not None:
        current_learner_scale = getattr(model, "get_public_heuristic_logit_bias_scale", None)
        if callable(current_learner_scale):
            resolved_learner_scale = float(current_learner_scale(scoring_mode="learner"))
    if resolved_learner_scale is None:
        return
    set_bias_scale(resolved_learner_scale, actor_value=resolved_actor_scale)


def _structured_policy_contract_from_config_payload(payload: Mapping[str, Any]) -> str | None:
    config_payload = payload.get("config")
    if not isinstance(config_payload, Mapping):
        return None
    model_payload = config_payload.get("model")
    if not isinstance(model_payload, Mapping):
        return None
    value = model_payload.get("structured_policy_contract")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _read_json_object(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Could not parse snapshot source metadata JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"Snapshot source metadata must be a JSON object: {path}")
    return payload


def _source_structured_policy_contract(run_dir: Path, payload: Mapping[str, Any]) -> str | None:
    value = payload.get("structured_policy_contract")
    if isinstance(value, str) and value.strip():
        return value.strip()

    manifest = _read_json_object(run_dir / "manifest.json")
    if manifest is not None:
        config_canonical = manifest.get("config_canonical")
        if isinstance(config_canonical, Mapping):
            value = _structured_policy_contract_from_config_payload(config_canonical)
            if value is not None:
                return value

    config_canonical = _read_json_object(run_dir / "config_canonical.json")
    if config_canonical is not None:
        value = _structured_policy_contract_from_config_payload(config_canonical)
        if value is not None:
            return value
    return None


def _expected_structured_policy_contract(model_config: Any) -> str | None:
    value = getattr(model_config, "structured_policy_contract", None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def validate_snapshot_structured_policy_contract(
    *,
    run_dir: Path,
    snapshot_path: str,
    payload: Mapping[str, Any],
    model_config: Any,
) -> None:
    expected_contract = _expected_structured_policy_contract(model_config)
    if expected_contract is None:
        return
    source_contract = _source_structured_policy_contract(run_dir, payload)
    if source_contract is None:
        raise RuntimeError(
            "Snapshot structured_policy_contract is missing; refusing to infer policy surface for "
            f"{snapshot_path} from {run_dir}"
        )
    if source_contract != expected_contract:
        raise RuntimeError(
            "Snapshot structured_policy_contract mismatch for "
            f"{snapshot_path}: source run uses {source_contract!r}, "
            f"but the active eval stack uses {expected_contract!r} (run_dir={run_dir})"
        )


def load_snapshot_model_from_path(
    *,
    run_dir: Path,
    snapshot_path: str,
    stack: StackConfig,
    observation_dim: int,
    action_dim: int,
    observation_spec: Mapping[str, object] | None = None,
    spec_bundle: Mapping[str, object] | None = None,
    device: torch.device | str = "cpu",
) -> PolicyValueModel:
    payload = torch.load(run_dir / snapshot_path, map_location="cpu", weights_only=True)
    model_state_dict = payload.get("model_state_dict")
    if not isinstance(model_state_dict, dict):
        raise RuntimeError(f"Snapshot weights payload missing model_state_dict: {snapshot_path}")

    model_config = stack.config.model
    if model_config is None:
        raise RuntimeError("The locked stack is missing the model config block")
    validate_snapshot_structured_policy_contract(
        run_dir=run_dir,
        snapshot_path=snapshot_path,
        payload=payload,
        model_config=model_config,
    )

    eval_model = build_policy_value_model(
        observation_dim=observation_dim,
        config=model_config,
        action_dim=action_dim,
        observation_spec=observation_spec,
        spec_bundle=spec_bundle,
    ).to(torch.device(device))
    load_model_state_dict_with_context_compat(
        eval_model,
        model_state_dict,
        context=f"snapshot {snapshot_path}",
    )
    restore_model_guidance_from_payload(eval_model, payload)
    eval_model.eval()
    return eval_model


def load_snapshot_eval_model(
    *,
    run_dir: Path,
    snapshot_path: str,
    stack: StackConfig,
    observation_dim: int,
    action_dim: int,
    observation_spec: Mapping[str, object] | None = None,
    spec_bundle: Mapping[str, object] | None = None,
) -> PolicyValueModel:
    return load_snapshot_model_from_path(
        run_dir=run_dir,
        snapshot_path=snapshot_path,
        stack=stack,
        observation_dim=observation_dim,
        action_dim=action_dim,
        observation_spec=observation_spec,
        spec_bundle=spec_bundle,
        device=torch.device("cpu"),
    )
