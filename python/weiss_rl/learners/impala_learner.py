"""IMPALA learner helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer

from weiss_rl.learners.vtrace import VTraceTargets
from weiss_rl.masking import masked_logp_from_legal_ids, masked_logp_from_mask


VTRACE_RHO_PERCENTILES = (50, 90, 95, 99)


def learner_logp_from_mask(
    logits: np.ndarray,
    legal_mask: np.ndarray,
    actions: np.ndarray,
    *,
    pass_action_id: int | None = None,
) -> np.ndarray:
    return masked_logp_from_mask(logits, legal_mask, actions, pass_action_id=pass_action_id)


def learner_logp_from_legal_ids(
    logits: np.ndarray,
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    actions: np.ndarray,
    *,
    pass_action_id: int | None = None,
) -> np.ndarray:
    return masked_logp_from_legal_ids(
        logits,
        legal_ids,
        legal_offsets,
        actions,
        pass_action_id=pass_action_id,
    )


def summarize_vtrace_diagnostics(
    result: VTraceTargets,
    *,
    rho_bar: float,
    c_bar: float,
) -> dict[str, float]:
    flat_rhos = np.asarray(result.rhos, dtype=np.float64).reshape(-1)
    if flat_rhos.size == 0:
        raise ValueError("result.rhos must not be empty")

    metrics = {
        f"vtrace_rho_p{percentile}": float(np.percentile(flat_rhos, percentile))
        for percentile in VTRACE_RHO_PERCENTILES
    }
    metrics["vtrace_rho_clip_rate"] = float(np.mean(flat_rhos > rho_bar))
    metrics["vtrace_c_clip_rate"] = float(np.mean(flat_rhos > c_bar))
    return metrics


def _batch_value(batch: Any, key: str) -> Any:
    if isinstance(batch, dict):
        return batch.get(key)
    return getattr(batch, key, None)


def _masked_log_probs_and_entropy(logits: Tensor, legal_mask: Tensor) -> tuple[Tensor, Tensor]:
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D (batch, action), got shape {tuple(logits.shape)}")
    if legal_mask.shape != logits.shape:
        raise ValueError("logits and legal_mask shapes must match")

    mask = legal_mask.to(dtype=torch.bool)
    masked_logits = logits.masked_fill(~mask, float("-inf"))
    has_legal = mask.any(dim=1, keepdim=True)
    row_max = masked_logits.max(dim=1, keepdim=True).values
    row_max = torch.where(has_legal, row_max, torch.zeros_like(row_max))

    shifted = torch.where(mask, logits - row_max, torch.full_like(logits, float("-inf")))
    exp_shifted = torch.where(mask, torch.exp(shifted), torch.zeros_like(logits))
    denom = exp_shifted.sum(dim=1, keepdim=True)
    safe_denom = torch.where(has_legal, denom, torch.ones_like(denom))
    log_probs = torch.where(mask, shifted - torch.log(safe_denom), torch.full_like(logits, float("-inf")))

    safe_log_probs = torch.where(mask, log_probs, torch.zeros_like(log_probs))
    probs = torch.where(mask, torch.exp(log_probs), torch.zeros_like(log_probs))
    entropy = -(probs * safe_log_probs).sum(dim=1)
    return log_probs, entropy


def _masked_action_logp_and_entropy(
    logits: Tensor,
    legal_mask: Tensor,
    actions: Tensor,
    *,
    pass_action_id: int | None,
) -> tuple[Tensor, Tensor]:
    if logits.ndim != 3:
        raise ValueError(f"logits must be 3D (time, batch, action), got shape {tuple(logits.shape)}")
    if legal_mask.shape != logits.shape:
        raise ValueError("logits and legal_mask shapes must match")
    if actions.shape != logits.shape[:2]:
        raise ValueError("actions must match logits on time and batch dimensions")

    flat_logits = logits.reshape(-1, logits.shape[-1]).to(dtype=torch.float32)
    flat_mask = legal_mask.reshape(-1, logits.shape[-1]).to(dtype=torch.bool)
    flat_actions = actions.reshape(-1).to(dtype=torch.long)
    action_space = flat_logits.shape[1]

    if bool((flat_actions < 0).any().item()):
        raise ValueError("actions must be >= 0")
    if bool((flat_actions >= action_space).any().item()):
        raise ValueError(f"actions must be < action_space ({action_space})")

    empty_rows = ~flat_mask.any(dim=1)
    row_actions = flat_actions.unsqueeze(1)
    action_is_legal = flat_mask.gather(dim=1, index=row_actions).squeeze(1)
    illegal_rows = (~empty_rows) & (~action_is_legal)
    if bool(illegal_rows.any().item()):
        row_index = int(torch.nonzero(illegal_rows, as_tuple=False)[0].item())
        action = int(flat_actions[row_index].item())
        raise ValueError(f"illegal action {action} for row {row_index}")

    log_probs, entropy = _masked_log_probs_and_entropy(flat_logits, flat_mask)
    selected_logp = log_probs.gather(dim=1, index=row_actions).squeeze(1)

    if bool(empty_rows.any().item()):
        if pass_action_id is None:
            raise ValueError("pass_action_id is required when legality contains empty rows")
        if pass_action_id < 0 or pass_action_id >= action_space:
            raise ValueError(f"pass_action_id must be in [0, {action_space})")
        illegal_empty_rows = empty_rows & (flat_actions != int(pass_action_id))
        if bool(illegal_empty_rows.any().item()):
            row_index = int(torch.nonzero(illegal_empty_rows, as_tuple=False)[0].item())
            action = int(flat_actions[row_index].item())
            raise ValueError(
                f"row {row_index} has no legal actions; expected pass action {pass_action_id}, got {action}"
            )
        selected_logp = torch.where(empty_rows, torch.zeros_like(selected_logp), selected_logp)
        entropy = torch.where(empty_rows, torch.zeros_like(entropy), entropy)

    return selected_logp.reshape(actions.shape), entropy.reshape(actions.shape)


@dataclass(slots=True)
class ImpalaLearner:
    model: nn.Module | None = None
    optimizer: Optimizer | None = None
    learning_rate: float = 2e-4
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    grad_norm_clip: float = 40.0
    pass_action_id: int | None = None

    def update(self, batch: Any) -> dict[str, float]:
        """Run one learner step when training tensors are present."""
        metrics = {"loss": 0.0}
        vtrace_result = _batch_value(batch, "vtrace_result")

        has_training_inputs = any(_batch_value(batch, key) is not None for key in ("obs", "actions", "legal_mask"))
        if has_training_inputs:
            missing = [
                key for key in ("obs", "actions", "legal_mask", "vtrace_result") if _batch_value(batch, key) is None
            ]
            if missing:
                missing_fields = ", ".join(missing)
                raise ValueError(
                    "batch must include obs, actions, legal_mask, and vtrace_result for learner updates; "
                    f"missing {missing_fields}"
                )
            if self.model is None:
                raise ValueError("ImpalaLearner requires a model to run an optimizer step")

            self.model.train()
            loss, loss_metrics = self._loss_and_metrics(batch)
            optimizer = self._optimizer_for_step()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
            optimizer.step()

            metrics.update(loss_metrics)
            metrics["grad_norm"] = float(grad_norm)

        if isinstance(vtrace_result, VTraceTargets):
            rho_bar_value = _batch_value(batch, "vtrace_rho_bar")
            c_bar_value = _batch_value(batch, "vtrace_c_bar")
            rho_bar = 1.0 if rho_bar_value is None else float(rho_bar_value)
            c_bar = 1.0 if c_bar_value is None else float(c_bar_value)
            metrics.update(summarize_vtrace_diagnostics(vtrace_result, rho_bar=rho_bar, c_bar=c_bar))
        return metrics

    def _loss_and_metrics(self, batch: Any) -> tuple[Tensor, dict[str, float]]:
        if self.model is None:
            raise ValueError("ImpalaLearner requires a model to compute losses")

        vtrace_result = _batch_value(batch, "vtrace_result")
        if not isinstance(vtrace_result, VTraceTargets):
            raise ValueError("batch.vtrace_result must be a VTraceTargets instance")

        obs = self._require_obs(_batch_value(batch, "obs"))
        actions = self._require_actions(_batch_value(batch, "actions"), expected_shape=obs.shape[:2])
        legal_mask = self._require_legal_mask(_batch_value(batch, "legal_mask"), expected_shape=obs.shape[:2])
        logits, values = self._forward_time_major(
            obs,
            initial_hidden_state=_batch_value(batch, "initial_hidden_state"),
            to_play_seat=_batch_value(batch, "to_play_seat"),
            actor=_batch_value(batch, "actor"),
        )
        if legal_mask.shape != logits.shape:
            raise ValueError("legal_mask must match learner logits on time, batch, and action dimensions")

        action_logp, entropy = _masked_action_logp_and_entropy(
            logits,
            legal_mask,
            actions,
            pass_action_id=self.pass_action_id,
        )
        targets = self._float_target(vtrace_result.vs, expected_shape=values.shape, like=values)
        advantages = self._float_target(vtrace_result.pg_advantages, expected_shape=values.shape, like=values)

        policy_loss = -(action_logp * advantages).mean()
        value_loss = torch.mean((values - targets) ** 2)
        entropy_mean = entropy.mean()
        total_loss = policy_loss + (self.value_loss_coef * value_loss) - (self.entropy_coef * entropy_mean)

        metrics = {
            "loss": float(total_loss.detach()),
            "policy_loss": float(policy_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "entropy": float(entropy_mean.detach()),
        }
        return total_loss, metrics

    def _optimizer_for_step(self) -> Optimizer:
        if self.optimizer is None:
            if self.model is None:
                raise ValueError("ImpalaLearner requires a model before creating an optimizer")
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return self.optimizer

    def _forward_time_major(
        self,
        obs: Tensor,
        *,
        initial_hidden_state: Any = None,
        to_play_seat: Any = None,
        actor: Any = None,
    ) -> tuple[Tensor, Tensor]:
        if self.model is None:
            raise ValueError("ImpalaLearner requires a model to run the forward pass")
        if obs.ndim != 3:
            raise ValueError(f"obs must be 3D (time, batch, observation), got shape {tuple(obs.shape)}")

        expected_shape = obs.shape[:2]
        batch_size = int(obs.shape[1])
        acting_seat = self._prepare_acting_seat_batch(
            to_play_seat,
            actor=actor,
            expected_shape=expected_shape,
        )
        logits_steps: list[Tensor] = []
        value_steps: list[Tensor] = []

        if acting_seat is None:
            hidden_state = self._prepare_legacy_hidden_state(initial_hidden_state, batch_size=batch_size, like=obs)
            for step_obs in obs.unbind(dim=0):
                step_logits, step_value, hidden_state = self.model(step_obs, hidden_state)
                logits_steps.append(torch.as_tensor(step_logits))
                value_steps.append(torch.as_tensor(step_value))
                hidden_state = torch.as_tensor(hidden_state)
            return torch.stack(logits_steps, dim=0), torch.stack(value_steps, dim=0)

        seat_hidden_state = self._prepare_seat_hidden_state(initial_hidden_state, batch_size=batch_size, like=obs)
        for step_obs, step_seat in zip(obs.unbind(dim=0), acting_seat.unbind(dim=0), strict=True):
            step_logits, step_value, seat_hidden_state = self.model.forward_seat_aware(
                step_obs,
                step_seat,
                seat_hidden_state,
            )
            logits_steps.append(torch.as_tensor(step_logits))
            value_steps.append(torch.as_tensor(step_value))
            seat_hidden_state = torch.as_tensor(seat_hidden_state)

        return torch.stack(logits_steps, dim=0), torch.stack(value_steps, dim=0)

    def _require_obs(self, value: Any) -> Tensor:
        tensor = self._float_input(value)
        if tensor.ndim != 3:
            raise ValueError(f"obs must be 3D (time, batch, observation), got shape {tuple(tensor.shape)}")
        return tensor

    def _require_actions(self, value: Any, *, expected_shape: torch.Size) -> Tensor:
        tensor = self._long_input(value)
        if tensor.shape != expected_shape:
            raise ValueError(f"actions must have shape {tuple(expected_shape)}, got {tuple(tensor.shape)}")
        return tensor

    def _require_legal_mask(self, value: Any, *, expected_shape: torch.Size) -> Tensor:
        tensor = self._bool_input(value)
        if tensor.ndim != 3 or tensor.shape[:2] != expected_shape:
            expected = (int(expected_shape[0]), int(expected_shape[1]), "action")
            raise ValueError(f"legal_mask must have shape {expected}, got {tuple(tensor.shape)}")
        return tensor

    def _float_target(self, value: Any, *, expected_shape: torch.Size, like: Tensor) -> Tensor:
        tensor = self._tensor_on_model_device(value, dtype=like.dtype)
        if tensor.shape != expected_shape:
            raise ValueError(f"target must have shape {tuple(expected_shape)}, got {tuple(tensor.shape)}")
        return tensor

    def _prepare_legacy_hidden_state(self, value: Any, *, batch_size: int, like: Tensor) -> Tensor | None:
        if value is None:
            return None
        tensor = self._tensor_on_model_device(value, dtype=like.dtype)
        if tensor.ndim != 2:
            raise ValueError(
                "initial_hidden_state must be 2D (batch, hidden_size) when to_play_seat/actor is absent, "
                f"got shape {tuple(tensor.shape)}"
            )
        if tensor.shape[0] != batch_size:
            raise ValueError(f"initial_hidden_state batch mismatch: expected {batch_size}, got {tensor.shape[0]}")
        return tensor

    def _prepare_seat_hidden_state(self, value: Any, *, batch_size: int, like: Tensor) -> Tensor | None:
        if value is None:
            return None
        tensor = self._tensor_on_model_device(value, dtype=like.dtype)
        if tensor.ndim != 3:
            raise ValueError(
                "initial_hidden_state must be 3D (batch, seat, hidden_size) when to_play_seat/actor is present, "
                f"got shape {tuple(tensor.shape)}"
            )
        if tensor.shape[0] != batch_size:
            raise ValueError(f"initial_hidden_state batch mismatch: expected {batch_size}, got {tensor.shape[0]}")
        if tensor.shape[1] != 2:
            raise ValueError(f"initial_hidden_state seat mismatch: expected 2, got {tensor.shape[1]}")
        return tensor

    def _prepare_acting_seat_batch(
        self,
        to_play_seat: Any,
        *,
        actor: Any,
        expected_shape: torch.Size,
    ) -> Tensor | None:
        seat_tensor = self._optional_time_major_seat_field(
            to_play_seat,
            field_name="to_play_seat",
            expected_shape=expected_shape,
        )
        actor_tensor = self._optional_time_major_seat_field(
            actor,
            field_name="actor",
            expected_shape=expected_shape,
        )

        if seat_tensor is None:
            return actor_tensor
        if actor_tensor is None:
            return seat_tensor
        if not torch.equal(seat_tensor, actor_tensor):
            raise ValueError("actor must match to_play_seat when both are provided")
        return seat_tensor

    def _optional_time_major_seat_field(
        self,
        value: Any,
        *,
        field_name: str,
        expected_shape: torch.Size,
    ) -> Tensor | None:
        if value is None:
            return None
        reference = self._model_parameter()
        tensor = torch.as_tensor(value, device=reference.device)
        if tensor.is_floating_point() or tensor.is_complex():
            raise ValueError(f"{field_name} must be integer-valued")
        tensor = tensor.to(dtype=torch.long)
        if tensor.shape != expected_shape:
            raise ValueError(f"{field_name} must have shape {tuple(expected_shape)}, got {tuple(tensor.shape)}")
        if bool(((tensor != 0) & (tensor != 1)).any().item()):
            raise ValueError(f"{field_name} values must be 0 or 1")
        return tensor

    def _float_input(self, value: Any) -> Tensor:
        reference = self._model_parameter()
        return self._tensor_on_model_device(value, dtype=reference.dtype)

    def _long_input(self, value: Any) -> Tensor:
        return self._tensor_on_model_device(value, dtype=torch.long)

    def _bool_input(self, value: Any) -> Tensor:
        return self._tensor_on_model_device(value, dtype=torch.bool)

    def _tensor_on_model_device(self, value: Any, *, dtype: torch.dtype) -> Tensor:
        if value is None:
            raise ValueError("batch field is required")
        reference = self._model_parameter()
        tensor = torch.as_tensor(value, device=reference.device)
        return tensor.to(dtype=dtype)

    def _model_parameter(self) -> Tensor:
        if self.model is None:
            raise ValueError("ImpalaLearner requires a model")
        parameter = next(self.model.parameters(), None)
        if parameter is None:
            raise ValueError("ImpalaLearner model must have at least one parameter")
        return parameter
