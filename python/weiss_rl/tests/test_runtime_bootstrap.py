from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from weiss_rl.runtime_components.bootstrap import bootstrap_values_for_unroll


@dataclass(frozen=True)
class _Unroll:
    bootstrap_obs: np.ndarray
    bootstrap_actor: np.ndarray
    final_hidden_state: np.ndarray


def test_bootstrap_values_for_unroll_prefers_value_only_path() -> None:
    class _Model:
        def __init__(self) -> None:
            self.value_calls = 0
            self.forward_calls = 0

        def value_seat_aware(
            self,
            obs: torch.Tensor,
            acting_seat: torch.Tensor,
            hidden_state: torch.Tensor,
        ) -> torch.Tensor:
            self.value_calls += 1
            assert obs.shape == (2, 3)
            assert acting_seat.tolist() == [0, 1]
            assert hidden_state.shape == (2, 4)
            return torch.tensor([2.5, -1.5], dtype=torch.float32, device=obs.device)

        def forward_seat_aware(
            self,
            obs: torch.Tensor,
            acting_seat: torch.Tensor,
            hidden_state: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            self.forward_calls += 1
            raise AssertionError("forward path should not run when value_seat_aware exists")

    model = _Model()
    values = bootstrap_values_for_unroll(
        unroll=_Unroll(
            bootstrap_obs=np.zeros((3, 3), dtype=np.float32),
            bootstrap_actor=np.array([0, 2, 1], dtype=np.int64),
            final_hidden_state=np.zeros((3, 4), dtype=np.float32),
        ),
        actor_model=model,
        bootstrap_device=torch.device("cpu"),
        actor_amp_enabled=False,
    )

    assert model.value_calls == 1
    assert model.forward_calls == 0
    np.testing.assert_array_equal(values, np.array([2.5, 0.0, -1.5], dtype=np.float32))


def test_bootstrap_values_for_unroll_uses_forward_fallback() -> None:
    class _Model:
        def forward_seat_aware(
            self,
            obs: torch.Tensor,
            acting_seat: torch.Tensor,
            hidden_state: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return (
                torch.zeros((obs.shape[0], 1), dtype=torch.float32, device=obs.device),
                torch.full((obs.shape[0],), 4.0, dtype=torch.float32, device=obs.device),
                hidden_state,
            )

    values = bootstrap_values_for_unroll(
        unroll=_Unroll(
            bootstrap_obs=np.zeros((2, 3), dtype=np.float32),
            bootstrap_actor=np.array([1, -1], dtype=np.int64),
            final_hidden_state=np.zeros((2, 4), dtype=np.float32),
        ),
        actor_model=_Model(),
        bootstrap_device=torch.device("cpu"),
        actor_amp_enabled=False,
    )

    np.testing.assert_array_equal(values, np.array([4.0, 0.0], dtype=np.float32))


def test_bootstrap_values_for_unroll_returns_zero_when_no_valid_rows() -> None:
    values = bootstrap_values_for_unroll(
        unroll=_Unroll(
            bootstrap_obs=np.zeros((2, 3), dtype=np.float32),
            bootstrap_actor=np.array([-1, 2], dtype=np.int64),
            final_hidden_state=np.zeros((2, 4), dtype=np.float32),
        ),
        actor_model=object(),
        bootstrap_device=torch.device("cpu"),
        actor_amp_enabled=False,
    )

    np.testing.assert_array_equal(values, np.array([0.0, 0.0], dtype=np.float32))
