"""IMPALA learner helpers."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.diagnostics.training_logger import TrainingLogger
from weiss_rl.learners import structured_auxiliary as _structured_auxiliary
from weiss_rl.learners.action_logp import (
    learner_logp_from_legal_ids as _learner_logp_from_legal_ids,
)
from weiss_rl.learners.action_logp import (
    learner_logp_from_mask as _learner_logp_from_mask,
)
from weiss_rl.learners.action_logp import (
    masked_action_logp_and_entropy,
    masked_log_probs_and_entropy,
    packed_action_logp_and_entropy,
    packed_scores_action_logp_and_entropy,
    packed_scores_family_entropy,
    packed_selected_action_logp,
    packed_subset_action_logp_and_top_action,
)
from weiss_rl.learners.factorized_evaluation import ImpalaFactorizedEvaluationMixin
from weiss_rl.learners.forward_time_major import ForwardTimeMajorResult, time_step_legal_actions
from weiss_rl.learners.impala_auxiliary_loss import ImpalaAuxiliaryLossMixin
from weiss_rl.learners.impala_support import ImpalaSupportMixin
from weiss_rl.learners.impala_update_loop import ImpalaUpdateLoopMixin
from weiss_rl.learners.policy_anchor import (
    clone_frozen_policy_anchor,
    packed_candidate_anchor_kl_loss,
    packed_candidate_anchor_top_action_loss,
)
from weiss_rl.learners.structured_auxiliary import (
    PackedStructuredLegalView as _PackedStructuredLegalView,
)
from weiss_rl.learners.structured_auxiliary import (
    normalize_public_heuristic_profile_mode,
    normalize_public_heuristic_profiles,
    packed_group_log_probs,
    packed_soft_target_cross_entropy,
    packed_structured_legal_view,
    resolve_public_heuristic_family_ids,
    structured_catalog_metadata,
)
from weiss_rl.learners.structured_policy_metrics import (
    summarize_structured_policy_metrics as _summarize_structured_policy_metrics,
)
from weiss_rl.learners.structured_teacher_auxiliary import compute_structured_teacher_auxiliary_metrics
from weiss_rl.learners.tensor_ops import (
    nonfinite_indices,
    segment_group_sum,
    segment_logsumexp,
    segment_max,
    weighted_mean,
)
from weiss_rl.learners.trajectory_retention import trajectory_retention_action_loss
from weiss_rl.learners.update_bookkeeping import (
    learner_acceleration_state,
    record_timing_ms,
    should_emit_structured_metrics,
    teacher_aux_active,
)
from weiss_rl.learners.vtrace import VTraceTargets
from weiss_rl.learners.vtrace_diagnostics import (
    VTRACE_RHO_PERCENTILES as _VTRACE_RHO_PERCENTILES,
)
from weiss_rl.learners.vtrace_diagnostics import (
    summarize_vtrace_diagnostics as _summarize_vtrace_diagnostics,
)
from weiss_rl.learners.vtrace_torch import compute_vtrace_targets_torch

_SUPPORTED_PUBLIC_HEURISTIC_PROFILES = _structured_auxiliary.SUPPORTED_PUBLIC_HEURISTIC_PROFILES
_SUPPORTED_PUBLIC_HEURISTIC_PROFILE_MODES = _structured_auxiliary.SUPPORTED_PUBLIC_HEURISTIC_PROFILE_MODES
VTRACE_RHO_PERCENTILES = _VTRACE_RHO_PERCENTILES
_ForwardTimeMajorResult = ForwardTimeMajorResult
_compute_vtrace_targets_torch = compute_vtrace_targets_torch
_masked_log_probs_and_entropy = masked_log_probs_and_entropy
_nonfinite_indices = nonfinite_indices
_normalize_public_heuristic_profile_mode = normalize_public_heuristic_profile_mode
_normalize_public_heuristic_profiles = normalize_public_heuristic_profiles
_packed_action_logp_and_entropy = packed_action_logp_and_entropy
_packed_scores_family_entropy = packed_scores_family_entropy
_packed_group_log_probs = packed_group_log_probs
_packed_scores_action_logp_and_entropy = packed_scores_action_logp_and_entropy
_packed_soft_target_cross_entropy = packed_soft_target_cross_entropy
_packed_structured_legal_view = packed_structured_legal_view
_packed_subset_action_logp_and_top_action = packed_subset_action_logp_and_top_action
_resolve_public_heuristic_family_ids = resolve_public_heuristic_family_ids
_segment_group_sum = segment_group_sum
_segment_logsumexp = segment_logsumexp
_segment_max = segment_max
_structured_catalog_metadata = structured_catalog_metadata
_time_step_legal_actions = time_step_legal_actions
_weighted_mean = weighted_mean


def _chosen_action_outcome_metrics(
    *,
    actions: Tensor,
    loss_mask: Tensor,
    rewards: Tensor,
    advantages: Tensor,
    action_catalog: ActionCatalog | None,
    pass_action_id: int | None,
) -> dict[str, float]:
    train_mask = loss_mask.detach().reshape(-1).to(dtype=torch.bool)
    train_count = int(train_mask.sum().item())
    if train_count == 0:
        return {}

    flat_actions = actions.detach().reshape(-1).to(device=train_mask.device, dtype=torch.long)
    flat_rewards = rewards.detach().reshape(-1).to(device=train_mask.device, dtype=torch.float32)
    flat_advantages = advantages.detach().reshape(-1).to(device=train_mask.device, dtype=torch.float32)
    metrics: dict[str, float] = {}

    def selected_fraction(selected: Tensor) -> float:
        return float((selected & train_mask).to(dtype=torch.float32).sum().item() / float(train_count))

    def selected_mean(values: Tensor, selected: Tensor) -> float:
        selected_train = selected & train_mask
        if not bool(selected_train.any().item()):
            return 0.0
        return float(values[selected_train].mean().item())

    if pass_action_id is not None:
        pass_selected = flat_actions == int(pass_action_id)
        nonpass_selected = ~pass_selected
        metrics.update(
            {
                "chosen_pass_train_fraction": selected_fraction(pass_selected),
                "chosen_pass_train_reward_mean": selected_mean(flat_rewards, pass_selected),
                "chosen_pass_train_advantage_mean": selected_mean(flat_advantages, pass_selected),
                "chosen_nonpass_train_reward_mean": selected_mean(flat_rewards, nonpass_selected),
                "chosen_nonpass_train_advantage_mean": selected_mean(flat_advantages, nonpass_selected),
            }
        )

    if action_catalog is None:
        return metrics

    def family_selected(family_name: str) -> Tensor:
        selected = torch.zeros_like(flat_actions, dtype=torch.bool)
        for family in action_catalog.families:
            if family.name != family_name:
                continue
            selected |= (flat_actions >= int(family.base)) & (flat_actions < int(family.base + family.count))
        return selected

    for family_name, metric_stem in (
        ("mulligan_confirm", "chosen_mulligan_confirm"),
        ("mulligan_select", "chosen_mulligan_select"),
        ("main_play_character", "chosen_main_play_character"),
        ("main_move", "chosen_main_move"),
        ("attack", "chosen_attack"),
    ):
        selected = family_selected(family_name)
        metrics[f"{metric_stem}_train_fraction"] = selected_fraction(selected)
        metrics[f"{metric_stem}_train_reward_mean"] = selected_mean(flat_rewards, selected)
        metrics[f"{metric_stem}_train_advantage_mean"] = selected_mean(flat_advantages, selected)
    return metrics


def learner_logp_from_mask(
    logits: np.ndarray,
    legal_mask: np.ndarray,
    actions: np.ndarray,
    *,
    pass_action_id: int | None = None,
) -> np.ndarray:
    return _learner_logp_from_mask(logits, legal_mask, actions, pass_action_id=pass_action_id)


def learner_logp_from_legal_ids(
    logits: np.ndarray,
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    actions: np.ndarray,
    *,
    pass_action_id: int | None = None,
) -> np.ndarray:
    return _learner_logp_from_legal_ids(logits, legal_ids, legal_offsets, actions, pass_action_id=pass_action_id)


def summarize_vtrace_diagnostics(
    result: VTraceTargets,
    *,
    rho_bar: float,
    c_bar: float,
) -> dict[str, float]:
    return _summarize_vtrace_diagnostics(result, rho_bar=rho_bar, c_bar=c_bar)


def summarize_structured_policy_metrics(
    logits: Tensor | None,
    legal_mask: Tensor | None,
    *,
    action_catalog: ActionCatalog,
    packed_ids: Tensor | None = None,
    packed_offsets: Tensor | None = None,
    packed_meta: Tensor | None = None,
    packed_view: _PackedStructuredLegalView | None = None,
    factorized_family_log_probs: Tensor | None = None,
) -> dict[str, float]:
    return _summarize_structured_policy_metrics(
        logits,
        legal_mask,
        action_catalog=action_catalog,
        packed_ids=packed_ids,
        packed_offsets=packed_offsets,
        packed_meta=packed_meta,
        packed_view=packed_view,
        factorized_family_log_probs=factorized_family_log_probs,
    )


def _batch_value(batch: Any, key: str) -> Any:
    if isinstance(batch, dict):
        return batch.get(key)
    return getattr(batch, key, None)


def _masked_action_logp_and_entropy(
    logits: Tensor,
    legal_mask: Tensor,
    actions: Tensor,
    *,
    pass_action_id: int | None,
) -> tuple[Tensor, Tensor]:
    return masked_action_logp_and_entropy(logits, legal_mask, actions, pass_action_id=pass_action_id)


def _packed_selected_action_logp(
    packed_logits: Tensor,
    legal_ids: Tensor,
    legal_offsets: Tensor,
    actions: Tensor,
    *,
    pass_action_id: int | None,
    strict: bool = True,
) -> Tensor:
    return packed_selected_action_logp(
        packed_logits,
        legal_ids,
        legal_offsets,
        actions,
        pass_action_id=pass_action_id,
        strict=strict,
    )


@dataclass(slots=True)
class ImpalaLearner(
    ImpalaUpdateLoopMixin,
    ImpalaAuxiliaryLossMixin,
    ImpalaFactorizedEvaluationMixin,
    ImpalaSupportMixin,
):
    model: nn.Module | None = None
    compiled_model: nn.Module | None = None
    optimizer: Optimizer | None = None
    learning_rate: float = 2e-4
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    entropy_scope: str = "candidate"
    grad_norm_clip: float = 40.0
    mixed_precision: bool = False
    checkpoint_dir: Path | None = None
    fault_dir: Path | None = None
    checkpoint_interval_updates: int = 50000
    logs_dir: Path | None = None
    logging_interval_updates: int = 100
    vtrace_rho_bar: float = 2.4
    vtrace_c_bar: float = 1.0
    pass_action_id: int | None = None
    teacher_family_coef: float = 0.0
    teacher_slot_coef: float = 0.0
    teacher_hand_coef: float = 0.0
    teacher_move_source_coef: float = 0.0
    teacher_attack_type_coef: float = 0.0
    teacher_action_coef: float = 0.0
    teacher_same_family_action_coef: float = 0.0
    teacher_action_margin_coef: float = 0.0
    teacher_action_margin: float = 0.5
    teacher_same_family_action_margin_coef: float = 0.0
    teacher_same_family_action_margin: float = 0.5
    teacher_exact_action_families: tuple[str, ...] = field(default_factory=tuple)
    teacher_public_heuristic_coef: float = 0.0
    teacher_public_heuristic_temperature: float = 32.0
    teacher_public_nonpass_over_pass_coef: float = 0.0
    teacher_public_nonpass_over_pass_margin: float = 0.5
    teacher_public_heuristic_families: tuple[str, ...] = field(default_factory=tuple)
    teacher_public_heuristic_profiles: tuple[str, ...] = field(default_factory=tuple)
    teacher_public_heuristic_profile_mode: str = "mixture"
    teacher_public_heuristic_profiles_end_updates: int = -1
    policy_anchor_coef: float = 0.0
    policy_anchor_top_action_coef: float = 0.0
    policy_anchor_temperature: float = 1.0
    trajectory_retention_coef: float = 0.0
    profile_timers: bool = False
    structured_metrics_mode: str = "full"
    teacher_aux_mode: str = "always"

    update_count: int = field(default=0, init=False)
    policy_version: int = field(default=0, init=False)
    total_samples_processed: int = field(default=0, init=False)
    start_time: float = field(default_factory=time.time, init=False)
    logger: TrainingLogger | None = field(default=None, init=False)
    last_log_time: float = field(default_factory=time.time, init=False)
    last_log_update: int = field(default=0, init=False)
    _policy_anchor_model: nn.Module | None = field(default=None, init=False)
    _amp_enabled: bool = field(default=False, init=False)
    _amp_device_type: str = field(default="cpu", init=False)
    _grad_scaler: torch.amp.GradScaler | None = field(default=None, init=False)
    _active_timing_metrics: dict[str, float] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.entropy_scope = str(self.entropy_scope).strip().lower()
        if self.entropy_scope not in {"candidate", "family"}:
            raise ValueError("entropy_scope must be one of: candidate, family")
        if self.logs_dir:
            self.logger = TrainingLogger(self.logs_dir, start_time=self.start_time)
        self.structured_metrics_mode = str(self.structured_metrics_mode).strip().lower()
        self.teacher_aux_mode = str(self.teacher_aux_mode).strip().lower()
        self.teacher_public_heuristic_profiles = _normalize_public_heuristic_profiles(
            self.teacher_public_heuristic_profiles
        )
        self.teacher_public_heuristic_profile_mode = _normalize_public_heuristic_profile_mode(
            self.teacher_public_heuristic_profile_mode
        )
        if float(self.policy_anchor_coef) < 0.0:
            raise ValueError("policy_anchor_coef must be >= 0")
        if float(self.policy_anchor_top_action_coef) < 0.0:
            raise ValueError("policy_anchor_top_action_coef must be >= 0")
        if float(self.policy_anchor_temperature) <= 0.0:
            raise ValueError("policy_anchor_temperature must be > 0")
        if float(self.trajectory_retention_coef) < 0.0:
            raise ValueError("trajectory_retention_coef must be >= 0")
        if self.structured_metrics_mode not in {"off", "sampled", "full"}:
            raise ValueError("structured_metrics_mode must be one of: off, sampled, full")
        if self.teacher_aux_mode not in {"off", "warmstart_only", "always"}:
            raise ValueError("teacher_aux_mode must be one of: off, warmstart_only, always")
        self._refresh_acceleration_state()

    def set_entropy_coef(self, value: float) -> None:
        self.entropy_coef = float(value)

    def set_teacher_aux_coefs(
        self,
        *,
        family: float | None = None,
        slot: float | None = None,
        hand: float | None = None,
        move_source: float | None = None,
        attack_type: float | None = None,
        action: float | None = None,
        same_family_action: float | None = None,
        action_margin: float | None = None,
        action_margin_value: float | None = None,
        same_family_action_margin: float | None = None,
        same_family_action_margin_value: float | None = None,
        exact_action_families: tuple[str, ...] | None = None,
        public_heuristic: float | None = None,
        public_heuristic_temperature: float | None = None,
        public_nonpass_over_pass: float | None = None,
        public_nonpass_over_pass_margin: float | None = None,
        public_heuristic_families: tuple[str, ...] | None = None,
        public_heuristic_profiles: tuple[str, ...] | None = None,
        public_heuristic_profile_mode: str | None = None,
        public_heuristic_profiles_end_updates: int | None = None,
    ) -> None:
        if family is not None:
            self.teacher_family_coef = float(family)
        if slot is not None:
            self.teacher_slot_coef = float(slot)
        if hand is not None:
            self.teacher_hand_coef = float(hand)
        if move_source is not None:
            self.teacher_move_source_coef = float(move_source)
        if attack_type is not None:
            self.teacher_attack_type_coef = float(attack_type)
        if action is not None:
            self.teacher_action_coef = float(action)
        if same_family_action is not None:
            self.teacher_same_family_action_coef = float(same_family_action)
        if action_margin is not None:
            self.teacher_action_margin_coef = float(action_margin)
        if action_margin_value is not None:
            self.teacher_action_margin = float(action_margin_value)
        if same_family_action_margin is not None:
            self.teacher_same_family_action_margin_coef = float(same_family_action_margin)
        if same_family_action_margin_value is not None:
            self.teacher_same_family_action_margin = float(same_family_action_margin_value)
        if exact_action_families is not None:
            self.teacher_exact_action_families = tuple(
                str(name).strip() for name in exact_action_families if str(name).strip()
            )
        if public_heuristic is not None:
            self.teacher_public_heuristic_coef = float(public_heuristic)
        if public_heuristic_temperature is not None:
            self.teacher_public_heuristic_temperature = float(public_heuristic_temperature)
        if public_nonpass_over_pass is not None:
            self.teacher_public_nonpass_over_pass_coef = float(public_nonpass_over_pass)
        if public_nonpass_over_pass_margin is not None:
            self.teacher_public_nonpass_over_pass_margin = float(public_nonpass_over_pass_margin)
        if public_heuristic_families is not None:
            self.teacher_public_heuristic_families = tuple(
                str(name).strip() for name in public_heuristic_families if str(name).strip()
            )
        if public_heuristic_profiles is not None:
            self.teacher_public_heuristic_profiles = _normalize_public_heuristic_profiles(public_heuristic_profiles)
        if public_heuristic_profile_mode is not None:
            self.teacher_public_heuristic_profile_mode = _normalize_public_heuristic_profile_mode(
                public_heuristic_profile_mode
            )
        if public_heuristic_profiles_end_updates is not None:
            self.teacher_public_heuristic_profiles_end_updates = int(public_heuristic_profiles_end_updates)

    def _record_timing_ms(self, name: str, elapsed_seconds: float) -> None:
        record_timing_ms(
            self._active_timing_metrics,
            profile_timers=self.profile_timers,
            name=name,
            elapsed_seconds=elapsed_seconds,
        )

    def _teacher_aux_active(self, *, auxiliary_update: bool) -> bool:
        return teacher_aux_active(teacher_aux_mode=self.teacher_aux_mode, auxiliary_update=auxiliary_update)

    def _should_emit_structured_metrics(self, *, auxiliary_update: bool) -> bool:
        return should_emit_structured_metrics(
            structured_metrics_mode=self.structured_metrics_mode,
            auxiliary_update=auxiliary_update,
            update_count=self.update_count,
        )

    def _refresh_acceleration_state(self) -> None:
        self._amp_enabled, self._amp_device_type, self._grad_scaler = learner_acceleration_state(
            model=self.model,
            mixed_precision=self.mixed_precision,
        )

    def _ensure_policy_anchor_model(self) -> nn.Module:
        if self.model is None:
            raise ValueError("ImpalaLearner requires a model to create a policy anchor")
        if self._policy_anchor_model is None:
            self._policy_anchor_model = clone_frozen_policy_anchor(self.model)
        return self._policy_anchor_model

    def reset_policy_anchor_to_current_model(self, *, force: bool = False) -> None:
        """Refresh the frozen anchor after externally replacing model weights."""

        if self.model is None:
            raise ValueError("ImpalaLearner requires a model to reset a policy anchor")
        if not force and float(self.policy_anchor_coef) == 0.0 and float(self.policy_anchor_top_action_coef) == 0.0:
            self._policy_anchor_model = None
            return
        self._policy_anchor_model = clone_frozen_policy_anchor(self.model)

    def policy_anchor_state_dict(self) -> dict[str, Tensor] | None:
        if self._policy_anchor_model is None:
            return None
        return self._policy_anchor_model.state_dict()

    def load_policy_anchor_state_dict(self, state_dict: Mapping[str, Any] | None) -> None:
        if state_dict is None:
            self._policy_anchor_model = None
            return
        anchor_model = self._ensure_policy_anchor_model()
        anchor_model.load_state_dict(state_dict)
        anchor_model.eval()
        for parameter in anchor_model.parameters():
            parameter.requires_grad_(False)

    def _factorized_candidate_log_probs_for_model(
        self,
        forward_model: nn.Module,
        batch: Any,
        *,
        obs: Tensor,
        packed_legal: tuple[Tensor, Tensor, Tensor | None],
        reset_before_step: Tensor | None,
    ) -> Tensor:
        expected_shape = obs.shape[:2]
        batch_size = int(obs.shape[1])
        acting_seat = self._prepare_acting_seat_batch(
            _batch_value(batch, "to_play_seat"),
            actor=_batch_value(batch, "actor"),
            expected_shape=expected_shape,
        )
        if acting_seat is None:
            raise ValueError("policy-anchor regularization requires acting seat information")
        trunk_kwargs = {} if reset_before_step is None else {"reset_before_step": reset_before_step}
        opponent_context_index = _batch_value(batch, "opponent_context_index")
        if opponent_context_index is not None:
            opponent_context_index = torch.as_tensor(opponent_context_index, device=obs.device, dtype=torch.long)
            if tuple(opponent_context_index.shape) != tuple(expected_shape):
                raise ValueError(
                    "opponent_context_index must match policy-anchor time-major shape "
                    f"{tuple(expected_shape)}, got {tuple(opponent_context_index.shape)}"
                )
            trunk_kwargs["opponent_context_index"] = opponent_context_index
        seat_hidden_state = self._prepare_seat_hidden_state(
            _batch_value(batch, "initial_hidden_state"),
            batch_size=batch_size,
            like=obs,
        )
        recurrent_flat, state_repr, observation_context, _values, _seat_hidden = (
            forward_model.forward_trunk_sequence_seat_aware(
                obs,
                acting_seat,
                seat_hidden_state,
                **trunk_kwargs,
            )
        )
        return self._factorized_packed_candidate_log_probs(
            forward_model,
            recurrent_flat=recurrent_flat,
            obs_rows=obs.reshape(int(expected_shape[0] * expected_shape[1]), obs.shape[-1]),
            legal_actions=self._packed_legal_action_view(packed_legal),
            state_repr=state_repr,
            observation_context={} if observation_context is None else dict(observation_context),
            opponent_context_index=(None if opponent_context_index is None else opponent_context_index.reshape(-1)),
        )

    def _policy_anchor_loss_and_metrics(
        self,
        batch: Any,
        *,
        obs: Tensor,
        loss_mask: Tensor,
        packed_legal: tuple[Tensor, Tensor, Tensor | None] | None,
        factorized_result: Any,
        forward_model: nn.Module,
        reset_before_step: Tensor | None,
    ) -> tuple[Tensor | None, dict[str, float]]:
        kl_coef = float(self.policy_anchor_coef)
        top_action_coef = float(self.policy_anchor_top_action_coef)
        if kl_coef == 0.0 and top_action_coef == 0.0:
            return None, {}
        if packed_legal is None or factorized_result is None:
            raise ValueError("policy_anchor_coef currently requires the factorized packed learner path")
        anchor_model = self._ensure_policy_anchor_model()
        if not self._should_use_factorized_legal_policy(anchor_model, packed_legal=packed_legal):
            raise ValueError("policy_anchor_coef requires a factorized structured anchor model")
        anchor_started = time.perf_counter()
        current_log_probs = self._factorized_candidate_log_probs_for_model(
            forward_model,
            batch,
            obs=obs,
            packed_legal=packed_legal,
            reset_before_step=reset_before_step,
        )
        with torch.no_grad():
            anchor_log_probs = self._factorized_candidate_log_probs_for_model(
                anchor_model,
                batch,
                obs=obs,
                packed_legal=packed_legal,
                reset_before_step=reset_before_step,
            )
        total_anchor_loss = current_log_probs.sum() * 0.0
        anchor_metrics: dict[str, float] = {}
        if kl_coef != 0.0:
            anchor_loss, anchor_metrics = packed_candidate_anchor_kl_loss(
                current_log_probs=current_log_probs,
                anchor_log_probs=anchor_log_probs,
                packed_offsets=packed_legal[1],
                row_shape=(int(obs.shape[0]), int(obs.shape[1])),
                loss_mask=loss_mask,
                temperature=float(self.policy_anchor_temperature),
            )
            total_anchor_loss = total_anchor_loss + (anchor_loss * kl_coef)
            anchor_metrics["policy_anchor_coef_active"] = kl_coef
            anchor_metrics["policy_anchor_temperature"] = float(self.policy_anchor_temperature)
        if top_action_coef != 0.0:
            top_action_loss, top_action_metrics = packed_candidate_anchor_top_action_loss(
                current_log_probs=current_log_probs,
                anchor_log_probs=anchor_log_probs,
                packed_offsets=packed_legal[1],
                row_shape=(int(obs.shape[0]), int(obs.shape[1])),
                loss_mask=loss_mask,
            )
            total_anchor_loss = total_anchor_loss + (top_action_loss * top_action_coef)
            anchor_metrics.update(top_action_metrics)
            anchor_metrics["policy_anchor_top_action_coef_active"] = top_action_coef
        self._record_timing_ms("learner_policy_anchor", time.perf_counter() - anchor_started)
        anchor_metrics["policy_anchor_weighted_loss"] = float(total_anchor_loss.detach().item())
        return total_anchor_loss, anchor_metrics

    def _loss_and_metrics(self, batch: Any) -> tuple[Tensor, dict[str, float]]:
        loss, metrics, _ = self._loss_and_metrics_with_context(batch)
        return loss, metrics

    def _loss_and_metrics_with_context(self, batch: Any) -> tuple[Tensor, dict[str, float], dict[str, Any]]:
        if self.model is None:
            raise ValueError("ImpalaLearner requires a model to compute losses")

        vtrace_result = _batch_value(batch, "vtrace_result")

        obs = self._require_obs(_batch_value(batch, "obs"))
        actions = self._require_actions(_batch_value(batch, "actions"), expected_shape=obs.shape[:2])
        packed_legal = self._resolve_packed_legal_actions_with_meta(batch, expected_shape=obs.shape[:2])
        forward_model = self.compiled_model if self.compiled_model is not None else self.model
        loss_mask = self._optional_time_major_loss_mask(
            _batch_value(batch, "policy_train_mask"),
            expected_shape=obs.shape[:2],
            like=obs[..., 0],
        )
        if loss_mask is None:
            loss_mask = torch.ones(obs.shape[:2], dtype=obs.dtype, device=obs.device)
        reset_before_step = self._optional_time_major_loss_mask(
            _batch_value(batch, "reset_before_step"),
            expected_shape=obs.shape[:2],
            like=obs[..., 0],
        )
        if reset_before_step is not None:
            reset_before_step = reset_before_step.to(dtype=torch.bool)
        trajectory_retention_valid = self._optional_time_major_loss_mask(
            _batch_value(batch, "trajectory_retention_valid"),
            expected_shape=obs.shape[:2],
            like=obs[..., 0],
        )
        trajectory_retention_active = (
            None
            if trajectory_retention_valid is None or float(self.trajectory_retention_coef) == 0.0
            else trajectory_retention_valid.to(dtype=torch.bool)
        )
        teacher_aux_active = isinstance(
            getattr(self.model, "action_catalog", None), ActionCatalog
        ) and self._teacher_aux_active(auxiliary_update=False)
        emit_structured_metrics = self._should_emit_structured_metrics(auxiliary_update=False)
        restrict_packed_policy_rows = bool(
            packed_legal is not None
            and bool((loss_mask <= 0.0).any().item())
            and not teacher_aux_active
            and not emit_structured_metrics
        )
        factorized_result = None
        forward_observation_context: Mapping[str, Tensor] | None = None
        if self._should_use_factorized_legal_policy(forward_model, packed_legal=packed_legal):
            factorized_result, packed_legal = self._evaluate_factorized_time_major(
                batch,
                obs=obs,
                actions=actions,
                extra_active_mask=trajectory_retention_active,
            )
            logits = None
            packed_logits = None
            values = factorized_result.values
        else:
            packed_forward_mask = loss_mask
            if trajectory_retention_active is not None:
                packed_forward_mask = torch.logical_or(
                    loss_mask > 0.0,
                    trajectory_retention_active,
                ).to(dtype=loss_mask.dtype)
            forward = self._forward_time_major(
                obs,
                initial_hidden_state=_batch_value(batch, "initial_hidden_state"),
                to_play_seat=_batch_value(batch, "to_play_seat"),
                actor=_batch_value(batch, "actor"),
                legal_actions=_batch_value(batch, "legal_actions"),
                policy_train_mask=packed_forward_mask if restrict_packed_policy_rows else None,
                reset_before_step=reset_before_step,
                opponent_context_index=_batch_value(batch, "opponent_context_index"),
            )
            logits = forward.logits
            packed_logits = forward.packed_logits
            values = forward.values
            forward_observation_context = forward.observation_context
        legal_mask = None
        if packed_legal is None:
            if logits is None:
                raise ValueError("dense learner path requires dense logits")
            legal_mask = self._resolve_legal_mask(batch, expected_shape=obs.shape[:2], action_dim=logits.shape[-1])
            if legal_mask.shape != logits.shape:
                raise ValueError("legal_mask must match learner logits on time, batch, and action dimensions")
        packed_view = None
        if packed_legal is not None and factorized_result is None and (emit_structured_metrics or teacher_aux_active):
            packed_view_started = time.perf_counter()
            packed_view = _packed_structured_legal_view(
                logits=packed_logits if packed_logits is not None else logits,
                packed_ids=packed_legal[0],
                packed_offsets=packed_legal[1],
                packed_meta=packed_legal[2],
            )
            self._record_timing_ms("learner_packed_view", time.perf_counter() - packed_view_started)
        teacher_aux_packed_view = packed_view
        public_heuristic_target_logits = None
        public_candidate_target_active = (
            float(self.teacher_public_heuristic_coef) != 0.0 or float(self.teacher_public_nonpass_over_pass_coef) != 0.0
        )
        factorized_candidate_teacher_view_active = factorized_result is not None and (
            public_candidate_target_active
            or float(self.teacher_action_margin_coef) != 0.0
            or float(self.teacher_same_family_action_margin_coef) != 0.0
        )
        if (
            teacher_aux_active
            and packed_legal is not None
            and (public_candidate_target_active or factorized_candidate_teacher_view_active)
            and (factorized_result is not None or hasattr(forward_model, "score_packed_public_heuristic_candidates"))
        ):
            if factorized_result is not None:
                teacher_aux_packed_view, public_heuristic_target_logits = (
                    self._factorized_public_heuristic_teacher_view(
                        batch,
                        obs=obs,
                        loss_mask=loss_mask,
                        packed_legal=packed_legal,
                        score_public_target=public_candidate_target_active,
                    )
                )
            elif public_candidate_target_active:
                heuristic_started = time.perf_counter()
                with torch.no_grad():
                    public_heuristic_target_logits = self._packed_public_heuristic_target_logits(
                        forward_model=forward_model,
                        obs=obs,
                        loss_mask=loss_mask,
                        packed_legal=packed_legal,
                        observation_context=forward_observation_context,
                    )
                self._record_timing_ms("learner_public_heuristic_target", time.perf_counter() - heuristic_started)

        context: dict[str, Any] = {
            "logits": None if logits is None else logits.detach(),
            "packed_logits": None if packed_logits is None else packed_logits.detach(),
            "values": values.detach(),
        }
        if logits is not None:
            self._ensure_finite_tensor("forward_logits", logits, batch=batch, context=context)
        if packed_logits is not None:
            self._ensure_finite_tensor("forward_packed_logits", packed_logits, batch=batch, context=context)
        self._ensure_finite_tensor("forward_values", values, batch=batch, context=context)

        if factorized_result is not None:
            if factorized_result.action_logp is None or factorized_result.entropy is None:
                raise ValueError("factorized learner path requires action_logp and entropy")
            action_logp = factorized_result.action_logp
            entropy = factorized_result.entropy
        elif packed_legal is not None:
            packed_reductions_started = time.perf_counter()
            packed_ids, packed_offsets, packed_meta = packed_legal
            if packed_logits is not None:
                action_logp, entropy = _packed_scores_action_logp_and_entropy(
                    packed_logits,
                    packed_ids,
                    packed_offsets,
                    actions,
                    pass_action_id=self.pass_action_id,
                )
                if self.entropy_scope == "family":
                    entropy_action_catalog = getattr(self.model, "action_catalog", None)
                    if not isinstance(entropy_action_catalog, ActionCatalog) or packed_meta is None:
                        raise ValueError("family entropy requires packed legal-action metadata and action_catalog")
                    entropy = _packed_scores_family_entropy(
                        packed_logits,
                        packed_offsets,
                        packed_meta,
                        row_shape=actions.shape,
                        family_count=len(entropy_action_catalog.families),
                    )
            else:
                assert logits is not None
                if self.entropy_scope == "family":
                    raise ValueError("family entropy requires packed candidate logits")
                action_logp, entropy = _packed_action_logp_and_entropy(
                    logits,
                    packed_ids,
                    packed_offsets,
                    actions,
                    pass_action_id=self.pass_action_id,
                )
            self._record_timing_ms("learner_packed_reductions", time.perf_counter() - packed_reductions_started)
        else:
            assert legal_mask is not None
            assert logits is not None
            action_logp, entropy = _masked_action_logp_and_entropy(
                logits,
                legal_mask,
                actions,
                pass_action_id=self.pass_action_id,
            )
        context["action_logp"] = action_logp.detach()
        context["entropy"] = entropy.detach()
        self._ensure_finite_tensor("action_logp", action_logp, batch=batch, context=context)
        self._ensure_finite_tensor("entropy", entropy, batch=batch, context=context)
        trajectory_retention_loss, trajectory_retention_metrics = trajectory_retention_action_loss(
            action_logp=action_logp,
            actions=actions,
            retention_valid=trajectory_retention_valid,
            coef=float(self.trajectory_retention_coef),
            top_action_ids=None if factorized_result is None else getattr(factorized_result, "top_action_ids", None),
        )
        if float(self.trajectory_retention_coef) != 0.0:
            context["trajectory_retention_loss"] = trajectory_retention_loss.detach()

        rho_bar_value = _batch_value(batch, "vtrace_rho_bar")
        c_bar_value = _batch_value(batch, "vtrace_c_bar")
        rho_bar = self.vtrace_rho_bar if rho_bar_value is None else float(rho_bar_value)
        c_bar = self.vtrace_c_bar if c_bar_value is None else float(c_bar_value)
        raw_behavior_logp = _batch_value(batch, "behavior_logp")
        behavior_logp_for_mask = None
        if raw_behavior_logp is not None:
            behavior_logp_for_mask = self._float_target(
                raw_behavior_logp,
                expected_shape=values.shape,
                like=values,
            )
        restrict_vtrace_to_behavior = bool(behavior_logp_for_mask is not None and bool((loss_mask <= 0.0).any().item()))
        if restrict_vtrace_to_behavior:
            assert behavior_logp_for_mask is not None
            action_logp = torch.where(loss_mask > 0.0, action_logp, behavior_logp_for_mask)
        if isinstance(vtrace_result, VTraceTargets):
            targets = self._float_target(vtrace_result.vs, expected_shape=values.shape, like=values)
            advantages = self._float_target(vtrace_result.pg_advantages, expected_shape=values.shape, like=values)
            rhos_for_metrics = self._float_target(vtrace_result.rhos, expected_shape=values.shape, like=values)
            raw_rewards = _batch_value(batch, "rewards")
            if raw_rewards is None:
                rewards_for_metrics = torch.zeros_like(values)
            else:
                rewards_for_metrics = self._float_target(raw_rewards, expected_shape=values.shape, like=values)
        else:
            rewards = self._float_target(_batch_value(batch, "rewards"), expected_shape=values.shape, like=values)
            discounts = self._float_target(_batch_value(batch, "discounts"), expected_shape=values.shape, like=values)
            if behavior_logp_for_mask is None:
                raise ValueError("raw V-trace batches must include behavior_logp")
            behavior_logp = behavior_logp_for_mask
            bootstrap_value = self._resolve_vtrace_bootstrap_value(
                batch,
                batch_size=int(values.shape[1]),
                like=values,
            )
            full_values = torch.cat([values.detach(), bootstrap_value.detach().unsqueeze(0)], dim=0)
            # Use the current learner policy log-prob for the V-trace target policy.
            # Passing behavior_logp twice silently forces rho=1 and disables off-policy correction.
            targets, advantages, rhos_for_metrics = _compute_vtrace_targets_torch(
                rewards,
                full_values,
                discounts,
                behavior_logp,
                action_logp,
                rho_bar=rho_bar,
                c_bar=c_bar,
            )
            rewards_for_metrics = rewards
        targets = targets.detach()
        advantages = advantages.detach()
        rhos_for_metrics = rhos_for_metrics.detach()
        context["targets"] = targets.detach()
        context["advantages"] = advantages.detach()
        context["vtrace_rhos"] = rhos_for_metrics.detach()
        context["rewards"] = rewards_for_metrics.detach()
        context["policy_train_mask"] = loss_mask.detach()
        value_loss_mask = self._optional_time_major_loss_mask(
            _batch_value(batch, "value_train_mask"),
            expected_shape=obs.shape[:2],
            like=values,
        )
        if value_loss_mask is None:
            value_loss_mask = torch.ones_like(loss_mask)
        context["value_train_mask"] = value_loss_mask.detach()
        policy_loss_denominator = torch.clamp(loss_mask.sum(), min=1.0)
        value_loss_denominator = torch.clamp(value_loss_mask.sum(), min=1.0)

        policy_loss = -((action_logp * advantages) * loss_mask).sum() / policy_loss_denominator
        value_loss = (((values - targets) ** 2) * value_loss_mask).sum() / value_loss_denominator
        entropy_mean = (entropy * loss_mask).sum() / policy_loss_denominator
        total_loss = policy_loss + (self.value_loss_coef * value_loss) - (self.entropy_coef * entropy_mean)
        if float(self.trajectory_retention_coef) != 0.0:
            total_loss = total_loss + trajectory_retention_loss

        policy_anchor_loss, policy_anchor_metrics = self._policy_anchor_loss_and_metrics(
            batch,
            obs=obs,
            loss_mask=loss_mask,
            packed_legal=packed_legal,
            factorized_result=factorized_result,
            forward_model=forward_model,
            reset_before_step=reset_before_step,
        )
        if policy_anchor_loss is not None:
            total_loss = total_loss + policy_anchor_loss

        action_catalog = getattr(self.model, "action_catalog", None)
        teacher_metrics: dict[str, float] = {}
        if teacher_aux_active:
            assert isinstance(action_catalog, ActionCatalog)
            structured_legal_mask = (
                None
                if factorized_result is not None
                else (
                    legal_mask
                    if legal_mask is not None
                    else (
                        None
                        if packed_legal is not None and packed_legal[2] is not None
                        else self._resolve_legal_mask(
                            batch,
                            expected_shape=obs.shape[:2],
                            action_dim=cast(Tensor, logits).shape[-1],
                        )
                    )
                )
            )
            teacher_aux_started = time.perf_counter()
            teacher_aux_loss, teacher_metrics, teacher_context = compute_structured_teacher_auxiliary_metrics(
                logits=logits,
                legal_mask=structured_legal_mask,
                teacher_family=self._optional_time_major_index_field(
                    _batch_value(batch, "teacher_family"),
                    field_name="teacher_family",
                    expected_shape=values.shape,
                ),
                teacher_slot=self._optional_time_major_index_field(
                    _batch_value(batch, "teacher_slot"),
                    field_name="teacher_slot",
                    expected_shape=values.shape,
                ),
                teacher_move_source=self._optional_time_major_index_field(
                    _batch_value(batch, "teacher_move_source"),
                    field_name="teacher_move_source",
                    expected_shape=values.shape,
                ),
                teacher_attack_type=self._optional_time_major_index_field(
                    _batch_value(batch, "teacher_attack_type"),
                    field_name="teacher_attack_type",
                    expected_shape=values.shape,
                ),
                teacher_action=self._optional_time_major_index_field(
                    _batch_value(batch, "teacher_action"),
                    field_name="teacher_action",
                    expected_shape=values.shape,
                ),
                teacher_valid=self._optional_time_major_bool_field(
                    _batch_value(batch, "teacher_valid"),
                    field_name="teacher_valid",
                    expected_shape=values.shape,
                ),
                loss_mask=loss_mask,
                action_catalog=action_catalog,
                family_coef=float(self.teacher_family_coef),
                slot_coef=float(self.teacher_slot_coef),
                hand_coef=float(self.teacher_hand_coef),
                move_source_coef=float(self.teacher_move_source_coef),
                attack_type_coef=float(self.teacher_attack_type_coef),
                action_coef=float(self.teacher_action_coef),
                same_family_action_coef=float(self.teacher_same_family_action_coef),
                action_margin_coef=float(self.teacher_action_margin_coef),
                action_margin=float(self.teacher_action_margin),
                same_family_action_margin_coef=float(self.teacher_same_family_action_margin_coef),
                same_family_action_margin=float(self.teacher_same_family_action_margin),
                exact_action_families=tuple(self.teacher_exact_action_families),
                public_heuristic_coef=float(self.teacher_public_heuristic_coef),
                public_heuristic_temperature=float(self.teacher_public_heuristic_temperature),
                public_nonpass_over_pass_coef=float(self.teacher_public_nonpass_over_pass_coef),
                public_nonpass_over_pass_margin=float(self.teacher_public_nonpass_over_pass_margin),
                public_heuristic_families=tuple(self.teacher_public_heuristic_families),
                public_heuristic_target_logits=public_heuristic_target_logits,
                packed_ids=None if packed_legal is None else packed_legal[0],
                packed_offsets=None if packed_legal is None else packed_legal[1],
                packed_meta=None if packed_legal is None else packed_legal[2],
                packed_view=teacher_aux_packed_view,
                factorized_family_log_probs=None if factorized_result is None else factorized_result.family_log_probs,
                factorized_play_slot_log_probs=None
                if factorized_result is None
                else factorized_result.play_slot_log_probs,
                factorized_move_source_log_probs=None
                if factorized_result is None
                else getattr(factorized_result, "move_source_log_probs", None),
                factorized_move_slot_log_probs=None
                if factorized_result is None
                else factorized_result.move_slot_log_probs,
                factorized_attack_slot_log_probs=None
                if factorized_result is None
                else factorized_result.attack_slot_log_probs,
                factorized_attack_type_log_probs=None
                if factorized_result is None
                else factorized_result.attack_type_log_probs,
                factorized_top_action_ids=None
                if factorized_result is None
                else getattr(factorized_result, "top_action_ids", None),
                factorized_same_family_action_logp=None
                if factorized_result is None
                else getattr(factorized_result, "same_family_action_logp", None),
                factorized_same_family_top_action_ids=None
                if factorized_result is None
                else getattr(factorized_result, "same_family_top_action_ids", None),
                factorized_same_family_arg0_logp=None
                if factorized_result is None
                else getattr(factorized_result, "same_family_arg0_logp", None),
                factorized_same_family_top_arg0=None
                if factorized_result is None
                else getattr(factorized_result, "same_family_top_arg0", None),
            )
            self._record_timing_ms("learner_teacher_aux", time.perf_counter() - teacher_aux_started)
            total_loss = total_loss + teacher_aux_loss
            context.update(teacher_context)

        context["policy_loss"] = policy_loss.detach()
        context["value_loss"] = value_loss.detach()
        context["entropy_mean"] = entropy_mean.detach()
        if policy_anchor_loss is not None:
            context["policy_anchor_loss"] = policy_anchor_loss.detach()
        context["total_loss"] = total_loss.detach()
        if factorized_result is not None:
            context["factorized_family_log_probs"] = factorized_result.family_log_probs.detach()
        self._ensure_finite_tensor("policy_loss", policy_loss, batch=batch, context=context)
        self._ensure_finite_tensor("value_loss", value_loss, batch=batch, context=context)
        self._ensure_finite_tensor("entropy_mean", entropy_mean, batch=batch, context=context)
        self._ensure_finite_tensor("total_loss", total_loss, batch=batch, context=context)

        rho_metrics = rhos_for_metrics.detach().reshape(-1).to(dtype=torch.float32)
        train_rho_mask = loss_mask.detach().reshape(-1).to(device=rho_metrics.device, dtype=torch.bool)
        train_rho_metrics = rho_metrics[train_rho_mask]
        if int(train_rho_metrics.numel()) == 0:
            train_rho_metrics = rho_metrics
        logp_delta_metrics: dict[str, float] = {}
        if behavior_logp_for_mask is not None:
            logp_delta = (
                action_logp.detach().to(dtype=torch.float32)
                - behavior_logp_for_mask.detach().to(device=action_logp.device, dtype=torch.float32)
            ).reshape(-1)
            logp_delta_abs = logp_delta.abs()
            train_logp_delta_abs = logp_delta_abs[train_rho_mask]
            if int(train_logp_delta_abs.numel()) == 0:
                train_logp_delta_abs = logp_delta_abs
            logp_delta_metrics = {
                "target_behavior_logp_delta_mean": float(logp_delta.mean().item()),
                "target_behavior_logp_delta_abs_mean": float(logp_delta_abs.mean().item()),
                "target_behavior_logp_delta_abs_p95": float(torch.quantile(logp_delta_abs, 0.95).item()),
                "target_behavior_logp_delta_abs_p99": float(torch.quantile(logp_delta_abs, 0.99).item()),
                "target_behavior_train_logp_delta_abs_mean": float(train_logp_delta_abs.mean().item()),
                "target_behavior_train_logp_delta_abs_p95": float(torch.quantile(train_logp_delta_abs, 0.95).item()),
                "target_behavior_train_logp_delta_abs_p99": float(torch.quantile(train_logp_delta_abs, 0.99).item()),
            }
        reward_metrics = rewards_for_metrics.detach()
        chosen_action_metrics = _chosen_action_outcome_metrics(
            actions=actions,
            loss_mask=loss_mask,
            rewards=reward_metrics,
            advantages=advantages.detach(),
            action_catalog=action_catalog if isinstance(action_catalog, ActionCatalog) else None,
            pass_action_id=self.pass_action_id,
        )
        metrics = {
            "loss": float(total_loss.detach()),
            "policy_loss": float(policy_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "entropy": float(entropy_mean.detach()),
            "entropy_scope_family_active": float(self.entropy_scope == "family"),
            "policy_train_fraction": float(loss_mask.mean().detach()),
            "value_train_fraction": float(value_loss_mask.mean().detach()),
            "reward_mean": float(reward_metrics.mean().item()),
            "reward_std": float(reward_metrics.to(dtype=torch.float32).std(unbiased=False).item()),
            "reward_abs_mean": float(reward_metrics.abs().mean().item()),
            "reward_min": float(reward_metrics.min().item()),
            "reward_max": float(reward_metrics.max().item()),
            "reward_nonzero_fraction": float((reward_metrics != 0).float().mean().item()),
            "reward_positive_fraction": float((reward_metrics > 0).float().mean().item()),
            "reward_negative_fraction": float((reward_metrics < 0).float().mean().item()),
            "terminal_outcome_backfill_count": float(_batch_value(batch, "terminal_outcome_backfill_count") or 0.0),
            "terminal_outcome_backfill_total_micros": float(
                _batch_value(batch, "terminal_outcome_backfill_total_micros") or 0.0
            ),
            "terminal_outcome_trace_backfill_count": float(
                _batch_value(batch, "terminal_outcome_trace_backfill_count") or 0.0
            ),
            "terminal_outcome_trace_backfill_total_micros": float(
                _batch_value(batch, "terminal_outcome_trace_backfill_total_micros") or 0.0
            ),
            "advantage_mean": float(advantages.detach().mean().item()),
            "advantage_abs_mean": float(advantages.detach().abs().mean().item()),
            "target_mean": float(targets.detach().mean().item()),
            "target_abs_mean": float(targets.detach().abs().mean().item()),
            "vtrace_rho_mean": float(rho_metrics.mean().item()),
            "vtrace_rho_p50": float(torch.quantile(rho_metrics, 0.50).item()),
            "vtrace_rho_p90": float(torch.quantile(rho_metrics, 0.90).item()),
            "vtrace_rho_p95": float(torch.quantile(rho_metrics, 0.95).item()),
            "vtrace_rho_p99": float(torch.quantile(rho_metrics, 0.99).item()),
            "vtrace_train_rho_mean": float(train_rho_metrics.mean().item()),
            "vtrace_train_rho_p50": float(torch.quantile(train_rho_metrics, 0.50).item()),
            "vtrace_train_rho_p90": float(torch.quantile(train_rho_metrics, 0.90).item()),
            "vtrace_train_rho_p95": float(torch.quantile(train_rho_metrics, 0.95).item()),
            "vtrace_train_rho_p99": float(torch.quantile(train_rho_metrics, 0.99).item()),
            "vtrace_rho_clip_rate": float((rhos_for_metrics.detach() > rho_bar).float().mean().item()),
            "vtrace_c_clip_rate": float((rhos_for_metrics.detach() > c_bar).float().mean().item()),
            **logp_delta_metrics,
            **chosen_action_metrics,
            **trajectory_retention_metrics,
            **policy_anchor_metrics,
        }
        metrics.update(teacher_metrics)
        if isinstance(action_catalog, ActionCatalog) and emit_structured_metrics:
            structured_legal_mask = (
                None
                if factorized_result is not None
                else (
                    legal_mask
                    if legal_mask is not None
                    else (
                        None
                        if packed_legal is not None and packed_legal[2] is not None
                        else self._resolve_legal_mask(
                            batch,
                            expected_shape=obs.shape[:2],
                            action_dim=cast(Tensor, logits).shape[-1],
                        )
                    )
                )
            )
            summary_started = time.perf_counter()
            metrics.update(
                summarize_structured_policy_metrics(
                    logits,
                    structured_legal_mask,
                    action_catalog=action_catalog,
                    packed_ids=None if packed_legal is None else packed_legal[0],
                    packed_offsets=None if packed_legal is None else packed_legal[1],
                    packed_meta=None if packed_legal is None else packed_legal[2],
                    packed_view=packed_view,
                    factorized_family_log_probs=None
                    if factorized_result is None
                    else factorized_result.family_log_probs,
                )
            )
            self._record_timing_ms("learner_structured_summary", time.perf_counter() - summary_started)
        return total_loss, metrics, context
