from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn
from weiss_rl.training.learner_compile import maybe_compile_learner_model


class _StructuredModel(nn.Module):
    supports_legal_candidate_scoring = True

    def __init__(self, *, fail: bool = False) -> None:
        super().__init__()
        self.fail = fail
        self.compile_calls: list[str] = []

    def enable_trunk_compile(self, *, mode: str) -> None:
        self.compile_calls.append(mode)
        if self.fail:
            raise RuntimeError("compile failed")


class _StructuredNoHookModel(nn.Module):
    supports_legal_candidate_scoring = True


class _PlainModel(nn.Module):
    pass


def test_maybe_compile_learner_model_returns_none_when_disabled() -> None:
    logs: list[str] = []

    compiled = maybe_compile_learner_model(
        model=_PlainModel(),
        training_config=SimpleNamespace(compile_learner=False),
        device=torch.device("cuda"),
        log_fn=logs.append,
    )

    assert compiled is None
    assert logs == []


def test_maybe_compile_learner_model_skips_non_cuda_device_with_public_note() -> None:
    logs: list[str] = []

    compiled = maybe_compile_learner_model(
        model=_PlainModel(),
        training_config=SimpleNamespace(compile_learner=True),
        device=torch.device("cpu"),
        log_fn=logs.append,
    )

    assert compiled is None
    assert logs == [
        "Learner compile note: compile_learner is enabled but the learner device is not CUDA; skipping torch.compile."
    ]


def test_maybe_compile_learner_model_prefers_structured_trunk_compile_hook() -> None:
    logs: list[str] = []
    model = _StructuredModel()

    compiled = maybe_compile_learner_model(
        model=model,
        training_config=SimpleNamespace(compile_learner=True),
        device=torch.device("cuda"),
        compile_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("torch.compile should be skipped")),
        log_fn=logs.append,
    )

    assert compiled is model
    assert model.compile_calls == ["reduce-overhead"]
    assert logs == ["Enabled torch.compile for the structured learner trunk (mode=reduce-overhead)."]


def test_maybe_compile_learner_model_reports_structured_hook_failures() -> None:
    logs: list[str] = []

    compiled = maybe_compile_learner_model(
        model=_StructuredModel(fail=True),
        training_config=SimpleNamespace(compile_learner=True),
        device=torch.device("cuda"),
        log_fn=logs.append,
    )

    assert compiled is None
    assert logs == [
        "Learner compile note: structured trunk compile failed; skipping torch.compile (RuntimeError('compile failed'))."
    ]


def test_maybe_compile_learner_model_reports_missing_structured_hook() -> None:
    logs: list[str] = []

    compiled = maybe_compile_learner_model(
        model=_StructuredNoHookModel(),
        training_config=SimpleNamespace(compile_learner=True),
        device=torch.device("cuda"),
        log_fn=logs.append,
    )

    assert compiled is None
    assert logs == [
        "Learner compile note: structured legal scoring is enabled but no trunk compile hook exists; skipping torch.compile."
    ]


def test_maybe_compile_learner_model_uses_torch_compile_for_plain_models() -> None:
    logs: list[str] = []
    model = _PlainModel()
    compiled_model = _PlainModel()
    compile_calls: list[tuple[nn.Module, str]] = []

    def compile_fn(module: nn.Module, *, mode: str) -> nn.Module:
        compile_calls.append((module, mode))
        return compiled_model

    compiled = maybe_compile_learner_model(
        model=model,
        training_config=SimpleNamespace(compile_learner=True),
        device=torch.device("cuda"),
        compile_fn=compile_fn,
        log_fn=logs.append,
    )

    assert compiled is compiled_model
    assert compile_calls == [(model, "reduce-overhead")]
    assert logs == ["Enabled torch.compile for the learner forward path (mode=reduce-overhead)."]
