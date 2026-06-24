from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from weiss_rl.models.policy import loading as model_loading


class _FakeModel:
    def __init__(self) -> None:
        self.device: torch.device | None = None
        self.loaded_state_dict = None
        self.eval_called = False
        self.bias_payload: tuple[float, float | None] | None = None
        self.current_learner_bias = 0.125

    def to(self, device: torch.device):
        self.device = device
        return self

    def state_dict(self):
        return {"weight": object()}

    def load_state_dict(self, state_dict, strict: bool = True):
        del strict
        self.loaded_state_dict = state_dict

    def eval(self):
        self.eval_called = True
        return self

    def get_public_heuristic_logit_bias_scale(self, *, scoring_mode: str) -> float:
        assert scoring_mode == "learner"
        return self.current_learner_bias

    def set_public_heuristic_logit_bias_scale(self, value: float, *, actor_value: float | None = None) -> None:
        self.bias_payload = (value, actor_value)


def test_load_snapshot_eval_model_restores_state_guidance_and_eval_mode(monkeypatch, tmp_path) -> None:
    fake_model = _FakeModel()
    calls: dict[str, object] = {}
    state_dict = {"weight": object()}

    def fake_load(path, *, map_location, weights_only):
        calls["load_path"] = path
        calls["map_location"] = map_location
        calls["weights_only"] = weights_only
        return {
            "model_state_dict": state_dict,
            "structured_policy_contract": "factorized_v1",
            "public_heuristic_logit_bias_scale": 0.5,
            "public_heuristic_actor_logit_bias_scale": 0.25,
        }

    def fake_builder(**kwargs):
        calls["builder_kwargs"] = kwargs
        return fake_model

    monkeypatch.setattr(model_loading.torch, "load", fake_load)
    monkeypatch.setattr(model_loading, "build_policy_value_model", fake_builder)

    model_config = SimpleNamespace(structured_policy_contract="factorized_v1")
    stack = SimpleNamespace(config=SimpleNamespace(model=model_config))
    observation_spec = {"obs": "spec"}
    spec_bundle = {"bundle": "spec"}

    loaded = model_loading.load_snapshot_eval_model(
        run_dir=tmp_path,
        snapshot_path="training/snapshots/policy_000001/weights.pt",
        stack=cast(Any, stack),
        observation_dim=512,
        action_dim=9,
        observation_spec=observation_spec,
        spec_bundle=spec_bundle,
    )

    assert cast(Any, loaded) is fake_model
    assert calls["load_path"] == tmp_path / "training/snapshots/policy_000001/weights.pt"
    assert calls["map_location"] == "cpu"
    assert calls["weights_only"] is True
    assert calls["builder_kwargs"] == {
        "observation_dim": 512,
        "config": model_config,
        "action_dim": 9,
        "observation_spec": observation_spec,
        "spec_bundle": spec_bundle,
    }
    assert fake_model.device == torch.device("cpu")
    assert fake_model.loaded_state_dict is state_dict
    assert fake_model.eval_called is True
    assert fake_model.bias_payload == (0.5, 0.25)


def test_restore_model_guidance_from_payload_uses_current_learner_scale_when_only_actor_scale_present() -> None:
    fake_model = _FakeModel()

    model_loading.restore_model_guidance_from_payload(
        fake_model,
        {"public_heuristic_actor_logit_bias_scale": 0.75},
    )

    assert fake_model.bias_payload == (0.125, 0.75)


def test_load_snapshot_eval_model_rejects_structured_policy_contract_mismatch(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        model_loading.torch,
        "load",
        lambda *args, **kwargs: {"model_state_dict": {}, "structured_policy_contract": "packed_v1"},
    )
    monkeypatch.setattr(model_loading, "build_policy_value_model", lambda **kwargs: _FakeModel())
    stack = SimpleNamespace(config=SimpleNamespace(model=SimpleNamespace(structured_policy_contract="factorized_v1")))

    with pytest.raises(RuntimeError, match="structured_policy_contract mismatch"):
        model_loading.load_snapshot_eval_model(
            run_dir=tmp_path,
            snapshot_path="training/snapshots/policy_000001/weights.pt",
            stack=cast(Any, stack),
            observation_dim=512,
            action_dim=9,
        )


def test_load_snapshot_eval_model_uses_manifest_structured_policy_contract(monkeypatch, tmp_path) -> None:
    fake_model = _FakeModel()
    monkeypatch.setattr(model_loading.torch, "load", lambda *args, **kwargs: {"model_state_dict": {}})
    monkeypatch.setattr(model_loading, "build_policy_value_model", lambda **kwargs: fake_model)
    (tmp_path / "manifest.json").write_text(
        '{"config_canonical": {"config": {"model": {"structured_policy_contract": "factorized_v1"}}}}',
        encoding="utf-8",
    )
    stack = SimpleNamespace(config=SimpleNamespace(model=SimpleNamespace(structured_policy_contract="factorized_v1")))

    loaded = model_loading.load_snapshot_eval_model(
        run_dir=tmp_path,
        snapshot_path="training/snapshots/policy_000001/weights.pt",
        stack=cast(Any, stack),
        observation_dim=512,
        action_dim=9,
    )

    assert cast(Any, loaded) is fake_model


def test_load_snapshot_eval_model_uses_config_canonical_structured_policy_contract(monkeypatch, tmp_path) -> None:
    fake_model = _FakeModel()
    monkeypatch.setattr(model_loading.torch, "load", lambda *args, **kwargs: {"model_state_dict": {}})
    monkeypatch.setattr(model_loading, "build_policy_value_model", lambda **kwargs: fake_model)
    (tmp_path / "config_canonical.json").write_text(
        '{"config": {"model": {"structured_policy_contract": "packed_v1"}}}',
        encoding="utf-8",
    )
    stack = SimpleNamespace(config=SimpleNamespace(model=SimpleNamespace(structured_policy_contract="packed_v1")))

    loaded = model_loading.load_snapshot_eval_model(
        run_dir=tmp_path,
        snapshot_path="training/snapshots/policy_000001/weights.pt",
        stack=cast(Any, stack),
        observation_dim=512,
        action_dim=9,
    )

    assert cast(Any, loaded) is fake_model


def test_load_snapshot_eval_model_rejects_missing_structured_policy_contract(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(model_loading.torch, "load", lambda *args, **kwargs: {"model_state_dict": {}})
    monkeypatch.setattr(model_loading, "build_policy_value_model", lambda **kwargs: _FakeModel())
    stack = SimpleNamespace(config=SimpleNamespace(model=SimpleNamespace(structured_policy_contract="factorized_v1")))

    with pytest.raises(RuntimeError, match="structured_policy_contract is missing"):
        model_loading.load_snapshot_eval_model(
            run_dir=tmp_path,
            snapshot_path="training/snapshots/policy_000001/weights.pt",
            stack=cast(Any, stack),
            observation_dim=512,
            action_dim=9,
        )


def test_load_snapshot_eval_model_rejects_missing_model_state_dict(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(model_loading.torch, "load", lambda *args, **kwargs: {"policy_id": "policy_000001"})
    stack = SimpleNamespace(config=SimpleNamespace(model="model-config"))

    with pytest.raises(RuntimeError, match="Snapshot weights payload missing model_state_dict"):
        model_loading.load_snapshot_eval_model(
            run_dir=tmp_path,
            snapshot_path="training/snapshots/policy_000001/weights.pt",
            stack=cast(Any, stack),
            observation_dim=512,
            action_dim=9,
        )


def test_load_snapshot_eval_model_requires_model_config(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(model_loading.torch, "load", lambda *args, **kwargs: {"model_state_dict": {}})
    stack = SimpleNamespace(config=SimpleNamespace(model=None))

    with pytest.raises(RuntimeError, match="locked stack is missing the model config block"):
        model_loading.load_snapshot_eval_model(
            run_dir=tmp_path,
            snapshot_path="training/snapshots/policy_000001/weights.pt",
            stack=cast(Any, stack),
            observation_dim=512,
            action_dim=9,
        )
