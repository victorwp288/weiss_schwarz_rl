from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from weiss_rl.learners.impala.losses.loss_forward_context import build_impala_forward_context


def test_build_impala_forward_context_detaches_outputs_and_checks_finiteness() -> None:
    calls: list[tuple[str, torch.Tensor, Any, dict[str, Any]]] = []
    learner = SimpleNamespace(
        _ensure_finite_tensor=lambda name, tensor, *, batch, context: calls.append((name, tensor, batch, context))
    )
    batch = {"forward_batch": True}
    logits = torch.ones((2, 1, 3), dtype=torch.float32, requires_grad=True)
    packed_logits = torch.arange(4, dtype=torch.float32, requires_grad=True)
    values = torch.zeros((2, 1), dtype=torch.float32, requires_grad=True)
    forward_result = SimpleNamespace(
        logits=logits,
        packed_logits=packed_logits,
        values=values,
    )

    context = build_impala_forward_context(
        learner=learner,
        batch=batch,
        forward_result=forward_result,
    )

    torch.testing.assert_close(context["logits"], logits)
    torch.testing.assert_close(context["packed_logits"], packed_logits)
    torch.testing.assert_close(context["values"], values)
    assert context["logits"].requires_grad is False
    assert context["packed_logits"].requires_grad is False
    assert context["values"].requires_grad is False
    assert [
        (name, tensor, source_batch, call_context is context) for name, tensor, source_batch, call_context in calls
    ] == [
        ("forward_logits", logits, batch, True),
        ("forward_packed_logits", packed_logits, batch, True),
        ("forward_values", values, batch, True),
    ]


def test_build_impala_forward_context_skips_absent_logits_but_checks_values() -> None:
    calls: list[str] = []
    learner = SimpleNamespace(_ensure_finite_tensor=lambda name, tensor, *, batch, context: calls.append(name))
    values = torch.zeros((1, 1), dtype=torch.float32, requires_grad=True)

    context = build_impala_forward_context(
        learner=learner,
        batch={},
        forward_result=SimpleNamespace(logits=None, packed_logits=None, values=values),
    )

    assert context["logits"] is None
    assert context["packed_logits"] is None
    torch.testing.assert_close(context["values"], values)
    assert context["values"].requires_grad is False
    assert calls == ["forward_values"]
