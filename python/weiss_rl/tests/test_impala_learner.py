from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from weiss_rl.learners.impala_learner import ImpalaLearner
from weiss_rl.learners.vtrace import VTraceTargets


class NaNLogitModel(nn.Module):
    def __init__(self, action_dim: int = 2) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(action_dim))

    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = int(obs.shape[0])
        logits = self.bias.unsqueeze(0).expand(batch, -1).clone()
        logits[0, 0] = torch.nan
        values = torch.zeros(batch, dtype=obs.dtype, device=obs.device)
        next_hidden = torch.zeros((batch, 1), dtype=obs.dtype, device=obs.device)
        return logits, values, next_hidden


class NaNGradientModel(nn.Module):
    def __init__(self, action_dim: int = 2) -> None:
        super().__init__()
        self.logit_bias = nn.Parameter(torch.zeros(action_dim))
        self.value_bias = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = int(obs.shape[0])
        logits = self.logit_bias.unsqueeze(0).expand(batch, -1).clone()
        logits.register_hook(lambda grad: torch.full_like(grad, torch.nan))
        values = self.value_bias.expand(batch)
        next_hidden = torch.zeros((batch, 1), dtype=obs.dtype, device=obs.device)
        return logits, values, next_hidden


def _simple_training_batch() -> dict[str, object]:
    return {
        "obs": np.asarray(
            [
                [[1.0, 0.0]],
                [[0.5, -0.5]],
            ],
            dtype=np.float32,
        ),
        "actions": np.asarray(
            [
                [0],
                [1],
            ],
            dtype=np.int64,
        ),
        "legal_mask": np.ones((2, 1, 2), dtype=np.uint8),
        "vtrace_result": VTraceTargets(
            vs=np.zeros((2, 1), dtype=np.float32),
            pg_advantages=np.ones((2, 1), dtype=np.float32),
            rhos=np.ones((2, 1), dtype=np.float32),
        ),
        "vtrace_rho_bar": 1.0,
        "vtrace_c_bar": 1.0,
    }


def test_impala_learner_writes_checkpoint_metadata_using_update_count(tmp_path: Path) -> None:
    learner = ImpalaLearner(
        checkpoint_dir=tmp_path / "checkpoints",
        checkpoint_interval_updates=2,
    )

    for _ in range(4):
        result = learner.update({})
        assert result["loss"] == 0.0

    checkpoint_dir = tmp_path / "checkpoints"
    assert (checkpoint_dir / "checkpoint_metadata_2.json").is_file()
    assert (checkpoint_dir / "checkpoint_metadata_4.json").is_file()
    assert learner.get_policy_version() == 2


def test_impala_learner_checkpoint_metadata_records_scope_update_and_policy_version(tmp_path: Path) -> None:
    learner = ImpalaLearner(
        checkpoint_dir=tmp_path / "checkpoints",
        checkpoint_interval_updates=3,
    )

    for _ in range(3):
        learner.update({})

    checkpoint_metadata = json.loads(
        (tmp_path / "checkpoints" / "checkpoint_metadata_3.json").read_text(encoding="utf-8")
    )
    assert checkpoint_metadata == {
        "format": "checkpoint_metadata",
        "parameters_included": False,
        "policy_version": 1,
        "update_count": 3,
    }


def test_impala_learner_writes_fault_bundle_on_nonfinite_forward_logits(tmp_path: Path) -> None:
    fault_dir = tmp_path / "faults"
    learner = ImpalaLearner(model=NaNLogitModel(), fault_dir=fault_dir)

    with pytest.raises(RuntimeError, match="non-finite learner forward_logits; wrote fault bundle to ") as excinfo:
        learner.update(_simple_training_batch())

    [fault_path] = sorted(fault_dir.glob("learner_numeric_fault_*.json"))
    assert str(fault_path) in str(excinfo.value)

    payload = json.loads(fault_path.read_text(encoding="utf-8"))
    assert payload["component"] == "impala_learner"
    assert payload["stage"] == "forward_logits"
    assert payload["context"]["forward_logits_nonfinite_indices"]["data"] == [[0, 0, 0], [1, 0, 0]]


def test_impala_learner_writes_fault_bundle_on_nonfinite_gradients(tmp_path: Path) -> None:
    fault_dir = tmp_path / "faults"
    learner = ImpalaLearner(model=NaNGradientModel(), fault_dir=fault_dir)

    with pytest.raises(RuntimeError, match="non-finite learner gradients; wrote fault bundle to ") as excinfo:
        learner.update(_simple_training_batch())

    [fault_path] = sorted(fault_dir.glob("learner_numeric_fault_*.json"))
    assert str(fault_path) in str(excinfo.value)

    payload = json.loads(fault_path.read_text(encoding="utf-8"))
    assert payload["component"] == "impala_learner"
    assert payload["stage"] == "gradients"
    assert "logit_bias" in payload["context"]["bad_gradient_names"]
