from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from weiss_rl.human_play import play_vs_model_entrypoint as play_script


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
        policy=cast(Any, policy),
        batch=cast(Any, batch),
        legal_ids=np.asarray([0, 1, 2], dtype=np.uint32),
        pass_action_id=0,
        seat_hidden=torch.zeros((1, 1), dtype=torch.float32),
        rng=play_script.Pcg32XshRrV1(7),
        temperature=2.0,
        top_k=3,
        catalog=cast(Any, SimpleNamespace()),
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

    result = cast(Any, play_script._advance_after_model_action(env=cast(Any, _FakeEnv()), action=7))

    assert result == "next-batch"
    assert len(stepped) == 1
    assert stepped[0].tolist() == [7]


def test_advance_after_model_action_continues_when_tty_input_hits_eof(monkeypatch) -> None:
    stepped: list[np.ndarray] = []

    class _FakeEnv:
        def step(self, actions: np.ndarray):
            stepped.append(np.asarray(actions, dtype=np.uint32))
            return "next-batch"

    monkeypatch.setattr(play_script.sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr("builtins.input", lambda prompt="": (_ for _ in ()).throw(EOFError(prompt)))

    result = cast(Any, play_script._advance_after_model_action(env=cast(Any, _FakeEnv()), action=9))

    assert result == "next-batch"
    assert len(stepped) == 1
    assert stepped[0].tolist() == [9]


def test_prompt_human_action_translates_eof_to_keyboard_interrupt(monkeypatch) -> None:
    batch = SimpleNamespace(
        ids_offsets=(
            np.asarray([1], dtype=np.uint32),
            np.asarray([0, 1], dtype=np.int32),
        )
    )

    monkeypatch.setattr(play_script.sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr("builtins.input", lambda prompt="": (_ for _ in ()).throw(EOFError(prompt)))
    monkeypatch.setattr(play_script, "_format_decoded_action", lambda action_id, catalog: f"action-{action_id}")

    with pytest.raises(KeyboardInterrupt, match="stdin closed while waiting for human input"):
        play_script._prompt_human_action(
            batch=cast(Any, batch),
            catalog=cast(Any, SimpleNamespace()),
            top_k_hints=[],
        )


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

    hints, next_hidden = play_script._rank_policy_hint_options(
        policy=cast(Any, policy),
        batch=cast(Any, batch),
        legal_ids=np.asarray([0, 1, 2], dtype=np.uint32),
        seat_hidden=torch.zeros((1, 1), dtype=torch.float32),
        top_k=2,
        catalog=cast(Any, SimpleNamespace()),
    )

    assert cast(Any, hints) == ["hint-a", "hint-b"]
    assert torch.equal(cast(torch.Tensor, next_hidden), torch.zeros((1, 1), dtype=torch.float32))
    assert np.allclose(cast(np.ndarray, observed["logits"]), np.asarray([1.0, 3.0, -2.0], dtype=np.float32))
    assert np.array_equal(cast(np.ndarray, observed["legal_ids"]), np.asarray([0, 1, 2], dtype=np.uint32))
    assert observed["top_k"] == 2


def test_main_allows_model_opening_turn_without_prior_human_hint(monkeypatch, tmp_path) -> None:
    class _FakeBatch:
        def __init__(self, *, actor: int, terminated: bool = False) -> None:
            self.actor = np.asarray([actor], dtype=np.int64)
            self.terminated = np.asarray([terminated], dtype=np.bool_)
            self.truncated = np.asarray([False], dtype=np.bool_)
            self.reward = np.asarray([0.0], dtype=np.float32)
            self.ids_offsets = (
                np.asarray([0], dtype=np.uint32),
                np.asarray([0, 1], dtype=np.int32),
            )

    class _FakeModel:
        def eval(self) -> None:
            return None

        def initial_seat_hidden(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
            del device
            return torch.zeros((batch_size, 1), dtype=torch.float32)

    class _FakeEnv:
        def reset(self, *, seed: int):
            del seed
            return _FakeBatch(actor=1)

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        play_script.sys,
        "argv",
        ["play_vs_model.py", "--run-dir", str(tmp_path), "--human-seat", "0"],
    )
    monkeypatch.setattr(
        play_script, "_resolve_stack_config_path", lambda run_dir, stack_config: tmp_path / "stack.json"
    )
    monkeypatch.setattr(play_script, "load_stack_config", lambda stack_config: SimpleNamespace())
    monkeypatch.setattr(
        play_script,
        "load_verified_simulator_contract",
        lambda repo_root, expected_spec_hash: SimpleNamespace(
            spec_bundle={
                "observation": {"obs_len": 4},
                "action": {"action_space_size": 8, "pass_action_id": 0},
            }
        ),
    )
    monkeypatch.setattr(play_script, "_repo_root_from_run_dir", lambda run_dir: tmp_path)
    monkeypatch.setattr(play_script, "_resolve_expected_spec_hash", lambda run_dir: "ab" * 32)
    monkeypatch.setattr(play_script, "_normalize_policy_id", lambda run_dir, requested_policy_id: "policy_000001")
    monkeypatch.setattr(
        play_script,
        "resolve_eval_policies",
        lambda **kwargs: {
            "policy_000001": SimpleNamespace(
                policy_id="policy_000001",
                model=_FakeModel(),
                heuristic_policy=None,
                kind="snapshot_registry",
            )
        },
    )
    monkeypatch.setattr(
        play_script, "build_env_config_from_stack", lambda *args, **kwargs: {"max_decisions": 1, "max_ticks": 1}
    )
    monkeypatch.setattr(play_script, "make_env_pool_from_config", lambda *args, **kwargs: (object(), "i16_legal_ids"))
    monkeypatch.setattr(play_script, "DecisionBoundaryEnv", lambda *args, **kwargs: _FakeEnv())
    monkeypatch.setattr(play_script.ActionCatalog, "from_spec_bundle", lambda spec_bundle: SimpleNamespace())
    monkeypatch.setattr(play_script, "_render_board", lambda env, perspective: "board")
    monkeypatch.setattr(play_script, "_print_model_suggestions", lambda options, header: None)
    monkeypatch.setattr(play_script, "_format_decoded_action", lambda action_id, catalog: f"action-{action_id}")
    monkeypatch.setattr(
        play_script,
        "_choose_policy_action",
        lambda **kwargs: (7, torch.ones((1, 1), dtype=torch.float32), []),
    )
    monkeypatch.setattr(
        play_script, "_advance_after_model_action", lambda env, action: _FakeBatch(actor=0, terminated=True)
    )

    play_script.main()


def test_main_reuses_shared_recurrent_hidden_across_seats(monkeypatch, tmp_path) -> None:
    class _FakeBatch:
        def __init__(self, *, actor: int, terminated: bool = False) -> None:
            self.actor = np.asarray([actor], dtype=np.int64)
            self.terminated = np.asarray([terminated], dtype=np.bool_)
            self.truncated = np.asarray([False], dtype=np.bool_)
            self.reward = np.asarray([0.0], dtype=np.float32)
            self.ids_offsets = (
                np.asarray([0], dtype=np.uint32),
                np.asarray([0, 1], dtype=np.int32),
            )

    class _FakeModel:
        def eval(self) -> None:
            return None

        def initial_seat_hidden(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
            del device
            return torch.zeros((batch_size, 2, 1), dtype=torch.float32)

    class _FakeEnv:
        def reset(self, *, seed: int):
            del seed
            return _FakeBatch(actor=0)

        def step(self, actions: np.ndarray):
            del actions
            return _FakeBatch(actor=1)

        def close(self) -> None:
            return None

    hint_hidden = torch.ones((1, 2, 1), dtype=torch.float32)
    observed: dict[str, torch.Tensor] = {}

    def _fake_rank_policy_hint_options(**kwargs):
        observed["hint_input"] = kwargs["seat_hidden"].clone()
        return [], hint_hidden

    def _fake_choose_policy_action(**kwargs):
        observed["model_input"] = kwargs["seat_hidden"].clone()
        return 7, torch.full((1, 2, 1), 2.0, dtype=torch.float32), []

    monkeypatch.setattr(
        play_script.sys,
        "argv",
        ["play_vs_model.py", "--run-dir", str(tmp_path), "--human-seat", "0"],
    )
    monkeypatch.setattr(
        play_script, "_resolve_stack_config_path", lambda run_dir, stack_config: tmp_path / "stack.json"
    )
    monkeypatch.setattr(play_script, "load_stack_config", lambda stack_config: SimpleNamespace())
    monkeypatch.setattr(
        play_script,
        "load_verified_simulator_contract",
        lambda repo_root, expected_spec_hash: SimpleNamespace(
            spec_bundle={
                "observation": {"obs_len": 4},
                "action": {"action_space_size": 8, "pass_action_id": 0},
            }
        ),
    )
    monkeypatch.setattr(play_script, "_repo_root_from_run_dir", lambda run_dir: tmp_path)
    monkeypatch.setattr(play_script, "_resolve_expected_spec_hash", lambda run_dir: "ab" * 32)
    monkeypatch.setattr(play_script, "_normalize_policy_id", lambda run_dir, requested_policy_id: "policy_000001")
    monkeypatch.setattr(
        play_script,
        "resolve_eval_policies",
        lambda **kwargs: {
            "policy_000001": SimpleNamespace(
                policy_id="policy_000001",
                model=_FakeModel(),
                heuristic_policy=None,
                kind="snapshot_registry",
            )
        },
    )
    monkeypatch.setattr(
        play_script, "build_env_config_from_stack", lambda *args, **kwargs: {"max_decisions": 1, "max_ticks": 1}
    )
    monkeypatch.setattr(play_script, "make_env_pool_from_config", lambda *args, **kwargs: (object(), "i16_legal_ids"))
    monkeypatch.setattr(play_script, "DecisionBoundaryEnv", lambda *args, **kwargs: _FakeEnv())
    monkeypatch.setattr(play_script.ActionCatalog, "from_spec_bundle", lambda spec_bundle: SimpleNamespace())
    monkeypatch.setattr(play_script, "_render_board", lambda env, perspective: "board")
    monkeypatch.setattr(play_script, "_print_model_suggestions", lambda options, header: None)
    monkeypatch.setattr(play_script, "_format_decoded_action", lambda action_id, catalog: f"action-{action_id}")
    monkeypatch.setattr(play_script, "_prompt_human_action", lambda **kwargs: 3)
    monkeypatch.setattr(play_script, "_rank_policy_hint_options", _fake_rank_policy_hint_options)
    monkeypatch.setattr(play_script, "_choose_policy_action", _fake_choose_policy_action)
    monkeypatch.setattr(
        play_script, "_advance_after_model_action", lambda env, action: _FakeBatch(actor=0, terminated=True)
    )

    play_script.main()

    assert torch.equal(observed["hint_input"], torch.zeros((1, 2, 1), dtype=torch.float32))
    assert torch.equal(observed["model_input"], hint_hidden)
