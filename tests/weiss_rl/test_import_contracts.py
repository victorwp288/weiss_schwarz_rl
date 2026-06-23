from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from weiss_rl.training.import_contracts import validate_imported_snapshot_contract
from weiss_rl.training.seed_snapshots import validate_seed_snapshot_import_contract


def _write_manifest(run_dir: Path, *, role: str, model: dict[str, Any], environment: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "config_canonical": {
                    "config": {
                        "experiment": {"role": role},
                        "model": model,
                        "environment": environment,
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def _state_dict() -> dict[str, torch.Tensor]:
    return {"encoder.weight": torch.zeros((2, 3), dtype=torch.float32)}


def _state_dict_with_context_adapter() -> dict[str, torch.Tensor]:
    return {
        **_state_dict(),
        "opponent_context_hidden_adapter": torch.zeros((2, 128), dtype=torch.float32),
    }


def _payload() -> dict[str, Any]:
    return {"model_state_dict": _state_dict()}


def _base_model_section() -> dict[str, Any]:
    return {
        "gru_hidden_size": 128,
        "encoder_mlp_width": 256,
        "encoder_kind": "typed",
        "structured_policy_contract": "packed_v1",
    }


def _context_model_section() -> dict[str, Any]:
    return {
        **_base_model_section(),
        "opponent_context_policy_ids": ["B2 HeuristicPublic"],
        "opponent_context_hidden_scale": 0.75,
        "opponent_context_trainable_hidden_scale": 0.5,
        "opponent_context_trainable_recurrent_scale": 0.5,
        "opponent_context_trainable_action_bias_scale": 1.0,
        "opponent_context_candidate_residual_action_ids": [104, 124],
        "opponent_context_adapter_lr_multiplier": 10000.0,
        "opponent_context_eval_policy_ids": ["policy_000001"],
    }


def _expected_config(*, model: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "config": {
            "experiment": {"role": "main"},
            "model": _context_model_section() if model is None else model,
            "environment": {"deck_policy": "fixed_thesis"},
        }
    }


def test_b1_import_contract_allows_context_only_model_extensions(tmp_path: Path) -> None:
    source_run = tmp_path / "b1"
    _write_manifest(
        source_run,
        role="baseline_noleague",
        model=_base_model_section(),
        environment={"deck_policy": "fixed_thesis"},
    )

    validate_imported_snapshot_contract(
        source_run_dir=source_run,
        payload=_payload(),
        expected_model_state_dict=_state_dict(),
        expected_config_canonical=_expected_config(),
        expected_spec_hash256=None,
    )


def test_seed_snapshot_import_contract_allows_context_only_model_extensions(tmp_path: Path) -> None:
    source_run = tmp_path / "seed"
    _write_manifest(
        source_run,
        role="main",
        model=_base_model_section(),
        environment={"deck_policy": "fixed_thesis"},
    )

    validate_seed_snapshot_import_contract(
        source_run_dir=source_run,
        payload=_payload(),
        expected_model_state_dict=_state_dict_with_context_adapter(),
        expected_config_canonical=_expected_config(),
        expected_spec_hash256=None,
    )


def test_b1_import_contract_allows_missing_trainable_context_adapter(tmp_path: Path) -> None:
    source_run = tmp_path / "b1"
    _write_manifest(
        source_run,
        role="baseline_noleague",
        model=_base_model_section(),
        environment={"deck_policy": "fixed_thesis"},
    )

    validate_imported_snapshot_contract(
        source_run_dir=source_run,
        payload=_payload(),
        expected_model_state_dict=_state_dict_with_context_adapter(),
        expected_config_canonical=_expected_config(),
        expected_spec_hash256=None,
    )


def test_import_contract_still_rejects_real_model_mismatch(tmp_path: Path) -> None:
    source_run = tmp_path / "b1"
    _write_manifest(
        source_run,
        role="baseline_noleague",
        model=_base_model_section(),
        environment={"deck_policy": "fixed_thesis"},
    )
    expected_model = {**_context_model_section(), "gru_hidden_size": 256}

    with pytest.raises(RuntimeError, match="section='model'"):
        validate_imported_snapshot_contract(
            source_run_dir=source_run,
            payload=_payload(),
            expected_model_state_dict=_state_dict(),
            expected_config_canonical=_expected_config(model=expected_model),
            expected_spec_hash256=None,
        )
