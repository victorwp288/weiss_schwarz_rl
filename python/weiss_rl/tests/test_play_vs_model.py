from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

import scripts.play_vs_model as play_script


class _DummyModel:
    def forward_seat_aware(
        self,
        obs: torch.Tensor,
        actor: torch.Tensor,
        seat_hidden: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        del obs, actor
        logits = torch.tensor([[1.0, 3.0, -2.0]], dtype=torch.float32)
        values = torch.zeros((1, 1), dtype=torch.float32)
        return logits, values, seat_hidden


def test_choose_policy_action_scales_logits_before_sampling(monkeypatch) -> None:
    captured: dict[str, np.ndarray] = {}

    def _fake_sample_action_pinned(logits, legal_ids, *, rng, pass_action_id):
        del legal_ids, rng, pass_action_id
        captured["logits"] = np.asarray(logits, dtype=np.float32)
        return 1, -0.25

    monkeypatch.setattr(play_script, "sample_action_pinned", _fake_sample_action_pinned)
    monkeypatch.setattr(play_script, "_rank_legal_actions", lambda **kwargs: [])
    policy = SimpleNamespace(
        heuristic_policy=None,
        kind="model",
        model=_DummyModel(),
        policy_id="policy_000001",
    )
    batch = SimpleNamespace(
        obs=np.zeros((1, 4), dtype=np.float32),
        actor=np.zeros((1,), dtype=np.int64),
    )

    play_script._choose_policy_action(
        policy=policy,
        batch=batch,
        legal_ids=np.asarray([0, 1, 2], dtype=np.uint32),
        pass_action_id=0,
        seat_hidden=torch.zeros((1, 1), dtype=torch.float32),
        rng=play_script.Pcg32XshRrV1(7),
        temperature=2.0,
        top_k=3,
        catalog=SimpleNamespace(),
    )

    assert np.allclose(captured["logits"], np.asarray([0.5, 1.5, -1.0], dtype=np.float32))


def test_advance_after_model_action_steps_even_without_tty(monkeypatch) -> None:
    stepped: list[np.ndarray] = []

    class _FakeEnv:
        def step(self, actions: np.ndarray):
            stepped.append(np.asarray(actions, dtype=np.uint32))
            return "next-batch"

    monkeypatch.setattr(play_script.sys.stdin, "isatty", lambda: False, raising=False)
    monkeypatch.setattr("builtins.input", lambda prompt="": (_ for _ in ()).throw(AssertionError(prompt)))

    result = play_script._advance_after_model_action(env=_FakeEnv(), action=7)

    assert result == "next-batch"
    assert len(stepped) == 1
    assert stepped[0].tolist() == [7]


def test_rank_policy_hint_options_do_not_sample(monkeypatch) -> None:
    observed: dict[str, object] = {}

    def _boom(*args, **kwargs):
        raise AssertionError("sample_action_pinned should not be used for hint generation")

    def _fake_rank_legal_actions(*, logits, legal_ids, catalog, top_k):
        del catalog
        observed["logits"] = np.asarray(logits, dtype=np.float32)
        observed["legal_ids"] = np.asarray(legal_ids, dtype=np.uint32)
        observed["top_k"] = int(top_k)
        return ["hint-a", "hint-b"]

    monkeypatch.setattr(play_script, "sample_action_pinned", _boom)
    monkeypatch.setattr(play_script, "_rank_legal_actions", _fake_rank_legal_actions)
    policy = SimpleNamespace(
        heuristic_policy=None,
        kind="model",
        model=_DummyModel(),
        policy_id="policy_000001",
    )
    batch = SimpleNamespace(
        obs=np.zeros((1, 4), dtype=np.float32),
        actor=np.zeros((1,), dtype=np.int64),
    )

    hints = play_script._rank_policy_hint_options(
        policy=policy,
        batch=batch,
        legal_ids=np.asarray([0, 1, 2], dtype=np.uint32),
        seat_hidden=torch.zeros((1, 1), dtype=torch.float32),
        top_k=2,
        catalog=SimpleNamespace(),
    )

    assert hints == ["hint-a", "hint-b"]
    assert np.allclose(observed["logits"], np.asarray([1.0, 3.0, -2.0], dtype=np.float32))
    assert np.array_equal(observed["legal_ids"], np.asarray([0, 1, 2], dtype=np.uint32))
    assert observed["top_k"] == 2
