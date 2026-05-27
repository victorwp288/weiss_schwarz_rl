from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from weiss_rl.learners.faults import (
    batch_fault_snapshot,
    collect_nonfinite_gradients,
    ensure_finite_gradients,
    ensure_finite_tensor,
    fault_dir_path,
    learner_batch_size,
    raise_for_nonfinite_gradients,
    write_numeric_fault_bundle,
)


def _batch_value(batch: Any, key: str) -> Any:
    if isinstance(batch, dict):
        return batch.get(key)
    return getattr(batch, key, None)


def test_learner_batch_size_uses_first_available_common_field() -> None:
    assert learner_batch_size({"actions": [[1, 2], [3, 4]]}, batch_value=_batch_value) == 4
    assert learner_batch_size(SimpleNamespace(obs=torch.zeros((2, 3))), batch_value=_batch_value) == 6
    assert learner_batch_size({}, batch_value=_batch_value) == 1


def test_fault_dir_path_precedence() -> None:
    assert fault_dir_path(fault_dir=Path("explicit"), checkpoint_dir=Path("ckpt"), logs_dir=Path("logs")) == Path(
        "explicit"
    )
    assert fault_dir_path(fault_dir=None, checkpoint_dir=Path("ckpt"), logs_dir=Path("logs")) == Path("ckpt/faults")
    assert fault_dir_path(fault_dir=None, checkpoint_dir=None, logs_dir=Path("logs")) == Path("logs/faults")
    assert fault_dir_path(fault_dir=None, checkpoint_dir=None, logs_dir=None) == Path("faults")


def test_batch_fault_snapshot_keeps_expected_fields_and_omits_missing_values() -> None:
    batch = {
        "obs": [1],
        "actions": [2],
        "legal_mask": None,
        "actor": [0],
        "vtrace_result": object(),
        "unrelated": "ignored",
    }

    snapshot = batch_fault_snapshot(batch, batch_value=_batch_value)

    assert sorted(snapshot) == ["actions", "actor", "obs", "vtrace_result"]
    assert snapshot["actions"] == [2]


def test_write_numeric_fault_bundle_preserves_payload_contract(tmp_path: Path) -> None:
    path = write_numeric_fault_bundle(
        fault_dir=tmp_path,
        stage="forward_logits",
        update_count=3,
        policy_version=2,
        batch_size=4,
        pass_action_id=19,
        batch_snapshot={"actions": [1]},
        context={"note": "x"},
    )

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["format"] == "numeric_fault_bundle"
    assert payload["component"] == "impala_learner"
    assert payload["stage"] == "forward_logits"
    assert payload["update_count"] == 3
    assert payload["policy_version"] == 2
    assert payload["batch_size"] == 4
    assert payload["pass_action_id"] == 19
    assert payload["batch"] == {"actions": [1]}
    assert payload["context"] == {"note": "x"}


def test_ensure_finite_tensor_writes_context_and_raises(tmp_path: Path) -> None:
    writes: list[tuple[str, Any, dict[str, Any]]] = []

    def _writer(stage: str, batch: Any, context: dict[str, Any]) -> Path:
        writes.append((stage, batch, context))
        return tmp_path / "fault.json"

    ensure_finite_tensor("values", torch.tensor([1.0]), batch={"ok": True}, context={}, write_bundle=_writer)
    with pytest.raises(RuntimeError, match="non-finite learner values; wrote fault bundle to"):
        ensure_finite_tensor(
            "values",
            torch.tensor([float("nan"), 2.0]),
            batch={"ok": False},
            context={"existing": torch.tensor(1.0)},
            write_bundle=_writer,
        )

    assert writes[0][0] == "values"
    assert writes[0][1] == {"ok": False}
    assert writes[0][2]["values_nonfinite_indices"].tolist() == [[0]]


def test_collect_and_raise_nonfinite_gradients() -> None:
    model = torch.nn.Linear(2, 1)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    model.bias.grad[0] = float("inf")  # type: ignore[index]

    bad_gradients, grad_norm_tensor = collect_nonfinite_gradients(model, torch.tensor(float("nan")))

    assert sorted(bad_gradients) == ["bias"]
    assert torch.isnan(grad_norm_tensor)
    with pytest.raises(ValueError, match="ImpalaLearner requires a model"):
        collect_nonfinite_gradients(None, torch.tensor(1.0))


def test_ensure_finite_gradients_delegates_to_fault_writer(tmp_path: Path) -> None:
    writes: list[tuple[str, dict[str, Any]]] = []

    def _writer(stage: str, batch: Any, context: dict[str, Any]) -> Path:
        del batch
        writes.append((stage, context))
        return tmp_path / "grad_fault.json"

    ensure_finite_gradients(
        batch={},
        context={"step": 1},
        grad_norm_tensor=torch.tensor(1.0),
        bad_gradients={},
        write_bundle=_writer,
    )
    assert writes == []

    with pytest.raises(RuntimeError, match="non-finite learner gradients; wrote fault bundle to"):
        raise_for_nonfinite_gradients(
            batch={},
            context={"step": 2},
            grad_norm_tensor=torch.tensor(float("nan")),
            bad_gradients={"weight": torch.tensor([float("inf")])},
            write_bundle=_writer,
        )

    assert writes[0][0] == "gradients"
    assert writes[0][1]["bad_gradient_names"] == ["weight"]
    assert writes[0][1]["grad_norm_nonfinite_indices"].tolist() == [[]]
