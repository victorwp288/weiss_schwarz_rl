"""Frozen-base residual policy helpers.

These wrappers are intentionally narrow: they make a trained residual head look
like a normal policy model while keeping the B1 base frozen. A zero residual is
therefore exactly the base policy, which is the property we need for B1 exploiter
smokes and hard-negative lanes.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn


class FrozenStoredLogitResidual(nn.Module):
    """Small residual head trained from tensorized counterfactual labels."""

    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        alpha: float,
        residual_mode: str = "plain",
        action_family_ids: Tensor | None = None,
        family_count: int = 0,
        gate_bias: float = 0.0,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.alpha = float(alpha)
        self.residual_mode = str(residual_mode or "plain")
        if self.residual_mode not in {"plain", "gated", "family_gated"}:
            raise ValueError("residual_mode must be one of: plain, gated, family_gated")
        self.seat_embedding = nn.Embedding(2, 8)
        input_dim = int(obs_dim) + 8
        self.residual = nn.Sequential(
            nn.Linear(input_dim, int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(action_dim)),
        )
        final = self.residual[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("FrozenStoredLogitResidual final layer must be linear")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.family_count = int(max(family_count, 0))
        if action_family_ids is None:
            family_ids = torch.full((int(action_dim),), -1, dtype=torch.long)
        else:
            family_ids = torch.as_tensor(action_family_ids, dtype=torch.long).reshape(-1)
            if int(family_ids.numel()) != int(action_dim):
                raise ValueError("action_family_ids must have one entry per action")
            self.family_count = max(self.family_count, int(family_ids.max().item()) + 1 if family_ids.numel() else 0)
        self.register_buffer("action_family_ids", family_ids, persistent=True)
        if self.residual_mode == "gated":
            self.gate = nn.Sequential(
                nn.Linear(input_dim, int(hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(hidden_dim), 1),
            )
        elif self.residual_mode == "family_gated":
            if self.family_count <= 0:
                raise ValueError("family_gated residual requires family_count > 0")
            self.gate = nn.Sequential(
                nn.Linear(input_dim, int(hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(hidden_dim), self.family_count),
            )
        else:
            self.gate = None
        if self.gate is not None:
            gate_final = self.gate[-1]
            if not isinstance(gate_final, nn.Linear):
                raise TypeError("FrozenStoredLogitResidual gate final layer must be linear")
            nn.init.zeros_(gate_final.weight)
            nn.init.constant_(gate_final.bias, float(gate_bias))

    def residual_logits(self, obs: Tensor, actor_seat: Tensor) -> Tensor:
        actor = actor_seat.to(device=obs.device, dtype=torch.long).reshape(-1).clamp(0, 1)
        seat = self.seat_embedding(actor)
        if obs.ndim != 2:
            obs = obs.reshape(actor.shape[0], -1)
        features = torch.cat([obs, seat], dim=-1)
        residual = self.residual(features)
        if self.gate is None:
            return residual
        gate_logits = self.gate(features)
        if self.residual_mode == "gated":
            return residual * torch.sigmoid(gate_logits)
        family_ids = self.action_family_ids.to(device=residual.device, dtype=torch.long)
        valid = family_ids >= 0
        clamped = family_ids.clamp(min=0)
        family_gate = torch.sigmoid(gate_logits).index_select(1, clamped)
        family_gate = torch.where(valid.reshape(1, -1), family_gate, torch.zeros_like(family_gate))
        return residual * family_gate

    def gate_values(self, obs: Tensor, actor_seat: Tensor, legal_ids: Tensor | None = None) -> Tensor | None:
        if self.gate is None:
            return None
        actor = actor_seat.to(device=obs.device, dtype=torch.long).reshape(-1).clamp(0, 1)
        seat = self.seat_embedding(actor)
        if obs.ndim != 2:
            obs = obs.reshape(actor.shape[0], -1)
        gate = torch.sigmoid(self.gate(torch.cat([obs, seat], dim=-1)))
        if self.residual_mode != "family_gated" or legal_ids is None:
            return gate
        family_ids = self.action_family_ids.to(device=obs.device, dtype=torch.long).index_select(0, legal_ids)
        valid = family_ids >= 0
        legal_gate = gate.index_select(1, family_ids.clamp(min=0))
        return torch.where(valid.reshape(1, -1), legal_gate, torch.zeros_like(legal_gate))

    def legal_logits(self, record: Mapping[str, Any]) -> tuple[Tensor, Tensor, Tensor]:
        device = self.residual[0].weight.device
        obs = torch.as_tensor(record["obs"], dtype=torch.float32, device=device).reshape(1, -1)
        actor_seat = torch.as_tensor(record["actor_seat"], dtype=torch.long, device=device).reshape(1)
        legal_ids = torch.as_tensor(record["legal_ids"], dtype=torch.long, device=device)
        base = torch.as_tensor(record["base_s1_legal_logits"], dtype=torch.float32, device=device)
        if base.numel() != legal_ids.numel():
            raise RuntimeError(
                f"base_s1_legal_logits has {int(base.numel())} entries, legal_ids has {int(legal_ids.numel())}"
            )
        residual_full = self.residual_logits(obs, actor_seat)[0]
        residual_legal = residual_full.index_select(0, legal_ids)
        return base + float(self.alpha) * residual_legal, base, residual_legal


class LiveFrozenB1Residual(nn.Module):
    """Wrap a live frozen base model with a trainable-head residual state."""

    def __init__(
        self,
        *,
        base_model: nn.Module,
        residual_probe: FrozenStoredLogitResidual,
        gate_obs_sha256: str = "",
        gate_actor_seat: int | None = None,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.residual_probe = residual_probe
        self.gate_obs_sha256 = str(gate_obs_sha256 or "")
        self.gate_actor_seat = gate_actor_seat
        self.forward_calls = 0
        self.residual_applied_rows = 0
        self.residual_suppressed_rows = 0
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        for parameter in self.residual_probe.parameters():
            parameter.requires_grad_(False)

    def initial_seat_hidden(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        return self.base_model.initial_seat_hidden(batch_size, device=device, dtype=dtype)

    def forward_seat_aware(
        self,
        obs: Tensor,
        acting_seat: int | Tensor,
        seat_hidden_state: Tensor | None = None,
        *,
        scoring_mode: str = "learner",
        legal_actions: Any | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        with torch.inference_mode():
            base_forward = self.base_model.forward_seat_aware
            if legal_actions is None:
                base_logits, value, next_hidden = base_forward(
                    obs,
                    acting_seat,
                    seat_hidden_state,
                    scoring_mode=scoring_mode,
                )
            else:
                base_logits, value, next_hidden = base_forward(
                    obs,
                    acting_seat,
                    seat_hidden_state,
                    scoring_mode=scoring_mode,
                    legal_actions=legal_actions,
                )
            actor = acting_seat if torch.is_tensor(acting_seat) else torch.as_tensor([acting_seat], device=obs.device)
            actor = actor.to(device=obs.device, dtype=torch.long).reshape(-1)
            self.forward_calls += int(obs.shape[0])
            if not self.gate_obs_sha256:
                residual = self.residual_probe.residual_logits(obs, actor)
                self.residual_applied_rows += int(obs.shape[0])
            else:
                residual = torch.zeros_like(base_logits)
                for row_index in range(int(obs.shape[0])):
                    row = obs[row_index].detach().cpu().contiguous().numpy()
                    row_sha = hashlib.sha256(row.tobytes()).hexdigest()
                    actor_matches = self.gate_actor_seat is None or int(actor[row_index].item()) == int(self.gate_actor_seat)
                    if row_sha == self.gate_obs_sha256 and actor_matches:
                        residual[row_index : row_index + 1] = self.residual_probe.residual_logits(
                            obs[row_index : row_index + 1],
                            actor[row_index : row_index + 1],
                        )
                        self.residual_applied_rows += 1
                    else:
                        self.residual_suppressed_rows += 1
            return base_logits + float(self.residual_probe.alpha) * residual, value, next_hidden

    def residual_gate_counters(self) -> dict[str, Any]:
        return {
            "gate_obs_sha256": self.gate_obs_sha256,
            "gate_actor_seat": self.gate_actor_seat,
            "forward_rows": int(self.forward_calls),
            "residual_applied_rows": int(self.residual_applied_rows),
            "residual_suppressed_rows": int(self.residual_suppressed_rows),
        }

    def set_public_heuristic_logit_bias_scale(
        self,
        value: float,
        *,
        scoring_mode: str = "learner",
        actor_value: float | None = None,
    ) -> None:
        setter = getattr(self.base_model, "set_public_heuristic_logit_bias_scale", None)
        if callable(setter):
            try:
                setter(value, actor_value=actor_value)
            except TypeError:
                setter(value, scoring_mode=scoring_mode)

    def get_public_heuristic_logit_bias_scale(self, *, scoring_mode: str = "learner") -> float:
        getter = getattr(self.base_model, "get_public_heuristic_logit_bias_scale", None)
        if callable(getter):
            return float(getter(scoring_mode=scoring_mode))
        return 0.0


class TrainableLiveFrozenB1Residual(nn.Module):
    """Live frozen-B1 wrapper whose residual head remains trainable.

    ``LiveFrozenB1Residual`` is intentionally inference-only because confirmed
    residual hard negatives must not drift during eval or league sampling. This
    class is the training counterpart: the base model is evaluated under
    no-grad, while the residual head is left in the autograd graph.
    """

    def __init__(
        self,
        *,
        base_model: nn.Module,
        residual_probe: FrozenStoredLogitResidual,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.residual_probe = residual_probe
        self.forward_calls = 0
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        for parameter in self.residual_probe.parameters():
            parameter.requires_grad_(True)
        self.base_model.eval()

    def initial_seat_hidden(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        return self.base_model.initial_seat_hidden(batch_size, device=device, dtype=dtype)

    def forward_seat_aware(
        self,
        obs: Tensor,
        acting_seat: int | Tensor,
        seat_hidden_state: Tensor | None = None,
        *,
        scoring_mode: str = "learner",
        legal_actions: Any | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        with torch.no_grad():
            base_forward = self.base_model.forward_seat_aware
            if legal_actions is None:
                base_logits, value, next_hidden = base_forward(
                    obs,
                    acting_seat,
                    seat_hidden_state,
                    scoring_mode=scoring_mode,
                )
            else:
                base_logits, value, next_hidden = base_forward(
                    obs,
                    acting_seat,
                    seat_hidden_state,
                    scoring_mode=scoring_mode,
                    legal_actions=legal_actions,
                )
        actor = acting_seat if torch.is_tensor(acting_seat) else torch.as_tensor([acting_seat], device=obs.device)
        actor = actor.to(device=obs.device, dtype=torch.long).reshape(-1)
        self.forward_calls += int(obs.shape[0])
        residual = self.residual_probe.residual_logits(obs, actor)
        return base_logits.detach() + float(self.residual_probe.alpha) * residual, value.detach(), next_hidden.detach()

    def residual_gate_counters(self) -> dict[str, Any]:
        return {
            "forward_rows": int(self.forward_calls),
            "residual_applied_rows": int(self.forward_calls),
            "residual_suppressed_rows": 0,
        }

    def set_public_heuristic_logit_bias_scale(
        self,
        value: float,
        *,
        scoring_mode: str = "learner",
        actor_value: float | None = None,
    ) -> None:
        setter = getattr(self.base_model, "set_public_heuristic_logit_bias_scale", None)
        if callable(setter):
            try:
                setter(value, actor_value=actor_value)
            except TypeError:
                setter(value, scoring_mode=scoring_mode)

    def get_public_heuristic_logit_bias_scale(self, *, scoring_mode: str = "learner") -> float:
        getter = getattr(self.base_model, "get_public_heuristic_logit_bias_scale", None)
        if callable(getter):
            return float(getter(scoring_mode=scoring_mode))
        return 0.0


def load_frozen_stored_logit_residual(
    residual_state_path: str | Any,
    *,
    device: torch.device,
) -> FrozenStoredLogitResidual:
    payload = torch.load(residual_state_path, map_location=device, weights_only=False)
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"residual state did not load as a mapping: {residual_state_path}")
    model = FrozenStoredLogitResidual(
        obs_dim=int(payload["obs_dim"]),
        action_dim=int(payload["action_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        alpha=float(payload["alpha"]),
        residual_mode=str(payload.get("residual_mode", "plain")),
        action_family_ids=payload.get("action_family_ids"),
        family_count=int(payload.get("family_count", 0)),
        gate_bias=float(payload.get("gate_bias", 0.0)),
    ).to(device)
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise RuntimeError(f"residual state missing model_state_dict: {residual_state_path}")
    state = dict(state)
    if "action_family_ids" not in state:
        # Older plain residual artifacts predate family gates. The buffer is
        # inert for plain/gated residuals, so synthesize the default ids to keep
        # those confirmed hard negatives loadable.
        state["action_family_ids"] = model.action_family_ids.detach().clone()
    model.load_state_dict(state)
    model.eval()
    return model
