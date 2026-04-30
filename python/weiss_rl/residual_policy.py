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

from weiss_rl.legal_actions import LegalActionBatch


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
                    actor_matches = self.gate_actor_seat is None or int(actor[row_index].item()) == int(
                        self.gate_actor_seat
                    )
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
        guard_enabled: bool = False,
        guard_top_gap: float = 0.35,
        guard_families: tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.residual_probe = residual_probe
        self.forward_calls = 0
        self.residual_applied_rows = 0
        self.residual_suppressed_rows = 0
        self.guard_enabled = bool(guard_enabled)
        self.guard_top_gap = float(guard_top_gap)
        self.guard_families = tuple(str(name) for name in guard_families)
        self.action_dim = int(getattr(base_model, "action_dim", residual_probe.action_dim))
        self.hidden_size = int(getattr(base_model, "hidden_size", 1))
        self.action_catalog = getattr(base_model, "action_catalog", None)
        self.supports_legal_candidate_scoring = bool(
            getattr(base_model, "supports_legal_candidate_scoring", False)
        )
        self.supports_factorized_legal_policy = False
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        for parameter in self.residual_probe.parameters():
            parameter.requires_grad_(True)
        self.base_model.eval()

    def train(self, mode: bool = True) -> TrainableLiveFrozenB1Residual:
        super().train(mode)
        self.base_model.eval()
        return self

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
        actor = self._actor_flat(actor, rows=int(obs.shape[0]), device=obs.device)
        self.forward_calls += int(obs.shape[0])
        residual = self.residual_probe.residual_logits(obs, actor)
        self.residual_applied_rows += int(obs.shape[0])
        return base_logits.detach() + float(self.residual_probe.alpha) * residual, value.detach(), next_hidden.detach()

    def _actor_flat(self, acting_seat: int | Tensor, *, rows: int, device: torch.device) -> Tensor:
        actor = acting_seat if torch.is_tensor(acting_seat) else torch.as_tensor([acting_seat], device=device)
        actor = actor.to(device=device, dtype=torch.long).reshape(-1)
        if int(actor.numel()) == 1 and int(rows) != 1:
            actor = actor.expand(int(rows))
        if int(actor.numel()) != int(rows):
            raise ValueError(f"acting_seat rows mismatch: expected {int(rows)}, got {int(actor.numel())}")
        return actor

    def _packed_row_indices(self, offsets: Tensor, *, device: torch.device) -> Tensor:
        offsets = offsets.to(device=device, dtype=torch.long)
        widths = offsets[1:] - offsets[:-1]
        return torch.repeat_interleave(
            torch.arange(max(int(offsets.numel()) - 1, 0), device=device, dtype=torch.long),
            widths,
        )

    def _packed_residual_logits(
        self,
        *,
        obs_flat: Tensor,
        actor_flat: Tensor,
        legal_actions: LegalActionBatch,
    ) -> Tensor:
        if legal_actions.ids is None or legal_actions.offsets is None:
            raise ValueError("B1 residual packed path requires legal_actions.ids and offsets")
        ids = torch.as_tensor(legal_actions.ids, device=obs_flat.device, dtype=torch.long)
        offsets = torch.as_tensor(legal_actions.offsets, device=obs_flat.device, dtype=torch.long)
        row_indices = self._packed_row_indices(offsets, device=obs_flat.device)
        residual_full = self.residual_probe.residual_logits(obs_flat, actor_flat)
        if ids.numel() == 0:
            return residual_full.new_zeros((0,))
        return residual_full[row_indices, ids]

    def _apply_packed_b1_guard(
        self,
        *,
        base_logits: Tensor,
        mixed_logits: Tensor,
        legal_actions: LegalActionBatch,
    ) -> Tensor:
        if not self.guard_enabled:
            self.residual_applied_rows += int(max(int(legal_actions.row_count), 0))
            return mixed_logits
        if legal_actions.ids is None or legal_actions.offsets is None:
            self.residual_applied_rows += int(max(int(legal_actions.row_count), 0))
            return mixed_logits

        offsets = torch.as_tensor(legal_actions.offsets, device=base_logits.device, dtype=torch.long)
        row_indices = self._packed_row_indices(offsets, device=base_logits.device)
        row_count = max(int(offsets.numel()) - 1, 0)
        if row_count <= 0 or int(base_logits.numel()) == 0:
            return mixed_logits

        top = base_logits.new_full((row_count,), -torch.inf)
        top.scatter_reduce_(0, row_indices, base_logits, reduce="amax", include_self=True)
        top_mask = base_logits >= (top.index_select(0, row_indices) - 1.0e-6)

        second_candidates = torch.where(top_mask, torch.full_like(base_logits, -torch.inf), base_logits)
        second = base_logits.new_full((row_count,), -torch.inf)
        second.scatter_reduce_(0, row_indices, second_candidates, reduce="amax", include_self=True)
        guarded_rows = (top - second) >= float(self.guard_top_gap)

        if self.guard_families:
            allowed_family_ids: set[int] = set()
            if self.action_catalog is not None:
                family_index = {family.name: idx for idx, family in enumerate(self.action_catalog.families)}
                allowed_family_ids = {
                    int(family_index[name])
                    for name in self.guard_families
                    if name in family_index
                }
            if not allowed_family_ids:
                guarded_rows = torch.zeros_like(guarded_rows)
            else:
                if legal_actions.meta is not None:
                    meta = torch.as_tensor(legal_actions.meta, device=base_logits.device, dtype=torch.long)
                    candidate_family_ids = meta[:, 0]
                else:
                    ids_cpu = torch.as_tensor(legal_actions.ids, dtype=torch.long).detach().cpu().tolist()
                    family_index = {family.name: idx for idx, family in enumerate(self.action_catalog.families)}
                    candidate_family_ids = torch.as_tensor(
                        [
                            int(family_index.get(self.action_catalog.decode(int(action_id)).family, -1))
                            for action_id in ids_cpu
                        ],
                        device=base_logits.device,
                        dtype=torch.long,
                    )

                top_family = torch.full((row_count,), -1, device=base_logits.device, dtype=torch.long)
                top_positions = torch.nonzero(top_mask, as_tuple=False).squeeze(1)
                if top_positions.numel() > 0:
                    top_rows = row_indices.index_select(0, top_positions)
                    top_families = candidate_family_ids.index_select(0, top_positions)
                    top_family.scatter_(0, top_rows, top_families)
                allowed = torch.zeros_like(guarded_rows)
                for family_id in allowed_family_ids:
                    allowed = allowed | (top_family == int(family_id))
                guarded_rows = guarded_rows & allowed

        guarded_candidate_mask = guarded_rows.index_select(0, row_indices)
        guarded_row_count = int(guarded_rows.sum().detach().item())
        self.residual_suppressed_rows += guarded_row_count
        self.residual_applied_rows += max(row_count - guarded_row_count, 0)
        if not bool(guarded_candidate_mask.any().detach().item()):
            return mixed_logits
        return torch.where(guarded_candidate_mask, base_logits.detach(), mixed_logits)

    def forward_sequence_packed_seat_aware(
        self,
        obs: Tensor,
        acting_seat: Tensor,
        seat_hidden_state: Tensor | None = None,
        *,
        legal_actions: LegalActionBatch,
        scoring_mode: str = "learner",
    ) -> tuple[Tensor, Tensor, Tensor]:
        with torch.no_grad():
            base_logits, value, next_hidden = self.base_model.forward_sequence_packed_seat_aware(
                obs,
                acting_seat,
                seat_hidden_state,
                legal_actions=legal_actions,
                scoring_mode=scoring_mode,
            )
        obs_flat = obs.reshape(obs.shape[0] * obs.shape[1], obs.shape[2])
        actor_flat = acting_seat.reshape(-1).to(device=obs_flat.device, dtype=torch.long)
        residual = self._packed_residual_logits(
            obs_flat=obs_flat,
            actor_flat=actor_flat,
            legal_actions=legal_actions,
        )
        self.forward_calls += int(obs_flat.shape[0])
        mixed = base_logits.detach() + float(self.residual_probe.alpha) * residual
        mixed = self._apply_packed_b1_guard(
            base_logits=base_logits.detach(),
            mixed_logits=mixed,
            legal_actions=legal_actions,
        )
        return mixed, value.detach(), next_hidden.detach()

    def forward_packed_seat_aware(
        self,
        obs: Tensor,
        acting_seat: int | Tensor,
        seat_hidden_state: Tensor | None = None,
        *,
        legal_actions: LegalActionBatch,
        scoring_mode: str = "learner",
    ) -> tuple[Tensor, Tensor, Tensor]:
        with torch.no_grad():
            base_logits, value, next_hidden = self.base_model.forward_packed_seat_aware(
                obs,
                acting_seat,
                seat_hidden_state,
                legal_actions=legal_actions,
                scoring_mode=scoring_mode,
            )
        obs_flat = obs.reshape(obs.shape[0], obs.shape[-1])
        actor_flat = self._actor_flat(acting_seat, rows=int(obs_flat.shape[0]), device=obs_flat.device)
        residual = self._packed_residual_logits(
            obs_flat=obs_flat,
            actor_flat=actor_flat,
            legal_actions=legal_actions,
        )
        self.forward_calls += int(obs_flat.shape[0])
        mixed = base_logits.detach() + float(self.residual_probe.alpha) * residual
        mixed = self._apply_packed_b1_guard(
            base_logits=base_logits.detach(),
            mixed_logits=mixed,
            legal_actions=legal_actions,
        )
        return mixed, value.detach(), next_hidden.detach()

    def sample_packed_seat_aware(
        self,
        obs: Tensor,
        acting_seat: int | Tensor,
        seat_hidden_state: Tensor | None = None,
        *,
        legal_actions: LegalActionBatch,
        sample_seeds: Tensor,
        pass_action_id: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        from weiss_rl.model import _sample_packed_action_scores

        packed_logits, value, next_hidden = self.forward_packed_seat_aware(
            obs,
            acting_seat,
            seat_hidden_state,
            legal_actions=legal_actions,
            scoring_mode="actor",
        )
        if legal_actions.ids is None or legal_actions.offsets is None:
            raise ValueError("B1 residual sampling requires packed legal actions")
        actions, logp = _sample_packed_action_scores(
            packed_logits,
            torch.as_tensor(legal_actions.ids, device=packed_logits.device, dtype=torch.long),
            torch.as_tensor(legal_actions.offsets, device=packed_logits.device, dtype=torch.long),
            sample_seeds.to(device=packed_logits.device, dtype=torch.long),
            pass_action_id=int(pass_action_id),
        )
        return actions, logp, value, next_hidden

    def forward_sequence_seat_aware(
        self,
        obs: Tensor,
        acting_seat: Tensor,
        seat_hidden_state: Tensor | None = None,
        *,
        legal_actions: LegalActionBatch | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        with torch.no_grad():
            base_logits, value, next_hidden = self.base_model.forward_sequence_seat_aware(
                obs,
                acting_seat,
                seat_hidden_state,
                legal_actions=legal_actions,
            )
        obs_flat = obs.reshape(obs.shape[0] * obs.shape[1], obs.shape[2])
        actor_flat = acting_seat.reshape(-1).to(device=obs_flat.device, dtype=torch.long)
        residual = self.residual_probe.residual_logits(obs_flat, actor_flat).reshape_as(base_logits)
        self.forward_calls += int(obs_flat.shape[0])
        self.residual_applied_rows += int(obs_flat.shape[0])
        mixed = base_logits.detach() + float(self.residual_probe.alpha) * residual
        return mixed, value.detach(), next_hidden.detach()

    def value_seat_aware(
        self,
        obs: Tensor,
        acting_seat: int | Tensor,
        seat_hidden_state: Tensor | None = None,
    ) -> Tensor:
        with torch.no_grad():
            return self.base_model.value_seat_aware(obs, acting_seat, seat_hidden_state).detach()

    def residual_gate_counters(self) -> dict[str, Any]:
        return {
            "forward_rows": int(self.forward_calls),
            "residual_applied_rows": int(self.residual_applied_rows),
            "residual_suppressed_rows": int(self.residual_suppressed_rows),
            "guard_enabled": bool(self.guard_enabled),
            "guard_top_gap": float(self.guard_top_gap),
            "guard_families": list(self.guard_families),
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
