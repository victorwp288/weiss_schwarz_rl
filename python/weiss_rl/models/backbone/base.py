"""Base recurrent policy/value model methods."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import Tensor, nn

from weiss_rl.config.models import ModelConfig
from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.models.backbone.policy_value_recurrent import SEAT_COUNT as _SEAT_COUNT
from weiss_rl.models.backbone.policy_value_recurrent import PolicyValueRecurrentMixin
from weiss_rl.models.backbone.sequence_forward import forward_sequence_seat_aware_dense
from weiss_rl.models.backbone.state import require_observation_batch
from weiss_rl.models.observations import typed_encoder as model_typed_encoder
from weiss_rl.models.policy.opponent_context_mixin import PolicyValueModelOpponentContextMixin

SEAT_COUNT = _SEAT_COUNT
GLOBAL_ACTION_SPACE_SIZE = 527
STRUCTURED_V2_ENCODER_KIND = "structured_v2"


class PolicyValueModelBaseMixin(PolicyValueRecurrentMixin, PolicyValueModelOpponentContextMixin):
    observation_dim: int
    hidden_size: int
    recurrent_core: str
    encoder: nn.Module
    gru: nn.GRU | None
    feedforward_core: nn.Module | None
    policy_head: nn.Module
    value_head: nn.Module

    def forward(
        self,
        obs: Tensor,
        hidden_state: Tensor | None = None,
        *,
        scoring_mode: str = "auto",
        opponent_context_index: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        del scoring_mode
        encoded_obs = self.encode(obs)
        recurrent_output, next_hidden = self.recurrent_step(encoded_obs, hidden_state)
        recurrent_output = self._apply_opponent_context_recurrent_adapter(recurrent_output, opponent_context_index)
        logits = self.policy_head(recurrent_output)
        logits = self._apply_opponent_context_action_bias(logits, opponent_context_index)
        value = self.value_head(recurrent_output).squeeze(-1)
        return logits, value, next_hidden

    def forward_seat_aware(
        self,
        obs: Tensor,
        acting_seat: int | Tensor,
        seat_hidden_state: Tensor | None = None,
        *,
        scoring_mode: str = "auto",
        opponent_context_index: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        del scoring_mode
        encoded_obs = self.encode(obs)
        recurrent_output, next_seat_hidden = self.recurrent_step_seat_aware(
            encoded_obs,
            acting_seat,
            seat_hidden_state,
        )
        recurrent_output = self._apply_opponent_context_recurrent_adapter(recurrent_output, opponent_context_index)
        logits = self.policy_head(recurrent_output)
        logits = self._apply_opponent_context_action_bias(logits, opponent_context_index)
        value = self.value_head(recurrent_output).squeeze(-1)
        return logits, value, next_seat_hidden

    def forward_seat_aware_inplace(
        self,
        obs: Tensor,
        acting_seat: int | Tensor,
        seat_hidden_state: Tensor | None = None,
        *,
        scoring_mode: str = "auto",
        opponent_context_index: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        del scoring_mode
        encoded_obs = self.encode(obs)
        recurrent_output, next_seat_hidden = self.recurrent_step_seat_aware_inplace(
            encoded_obs,
            acting_seat,
            seat_hidden_state,
        )
        recurrent_output = self._apply_opponent_context_recurrent_adapter(recurrent_output, opponent_context_index)
        logits = self.policy_head(recurrent_output)
        logits = self._apply_opponent_context_action_bias(logits, opponent_context_index)
        value = self.value_head(recurrent_output).squeeze(-1)
        return logits, value, next_seat_hidden

    def advance_seat_hidden(
        self,
        obs: Tensor,
        acting_seat: int | Tensor,
        seat_hidden_state: Tensor | None = None,
    ) -> Tensor:
        encoded_obs = self.encode(obs)
        _, next_seat_hidden = self.recurrent_step_seat_aware(
            encoded_obs,
            acting_seat,
            seat_hidden_state,
        )
        return next_seat_hidden

    def value_seat_aware(
        self,
        obs: Tensor,
        acting_seat: int | Tensor,
        seat_hidden_state: Tensor | None = None,
    ) -> Tensor:
        encoded_obs = self.encode(obs)
        recurrent_output, _next_seat_hidden = self.recurrent_step_seat_aware(
            encoded_obs,
            acting_seat,
            seat_hidden_state,
        )
        return self.value_head(recurrent_output).squeeze(-1)

    def forward_sequence_seat_aware(
        self,
        obs: Tensor,
        acting_seat: Tensor,
        seat_hidden_state: Tensor | None = None,
        *,
        legal_actions: LegalActionBatch | None = None,
        reset_before_step: Tensor | None = None,
        opponent_context_index: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return forward_sequence_seat_aware_dense(
            self,
            obs,
            acting_seat,
            seat_hidden_state,
            legal_actions=legal_actions,
            reset_before_step=reset_before_step,
            opponent_context_index=opponent_context_index,
        )

    def forward_sequence_packed_seat_aware(
        self,
        obs: Tensor,
        acting_seat: Tensor,
        seat_hidden_state: Tensor | None = None,
        *,
        legal_actions: LegalActionBatch,
    ) -> tuple[Tensor, Tensor, Tensor]:
        raise ValueError("forward_sequence_packed_seat_aware is only supported on structured models")

    def forward_packed_seat_aware(
        self,
        obs: Tensor,
        acting_seat: int | Tensor,
        seat_hidden_state: Tensor | None = None,
        *,
        legal_actions: LegalActionBatch,
    ) -> tuple[Tensor, Tensor, Tensor]:
        raise ValueError("forward_packed_seat_aware is only supported on structured models")

    def sample_packed_seat_aware(
        self,
        obs: Tensor,
        acting_seat: int | Tensor,
        seat_hidden_state: Tensor | None = None,
        *,
        legal_actions: LegalActionBatch,
        sample_seeds: Tensor,
        pass_action_id: int,
        temperature: float = 1.0,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        del temperature
        raise ValueError("sample_packed_seat_aware is only supported on structured models")

    def enable_trunk_compile(self, *, mode: str = "reduce-overhead") -> Any:
        del mode
        return self

    def set_public_heuristic_logit_bias_scale(
        self,
        value: float,
        *,
        actor_value: float | None = None,
    ) -> None:
        del value, actor_value

    def get_public_heuristic_logit_bias_scale(self, *, scoring_mode: str = "learner") -> float:
        del scoring_mode
        return 0.0

    def _build_observation_encoder(
        self,
        *,
        observation_dim: int,
        config: ModelConfig,
        observation_spec: Mapping[str, Any] | None,
        dropout_p: float,
    ) -> nn.Module:
        return model_typed_encoder.build_observation_encoder(
            observation_dim=observation_dim,
            config=config,
            observation_spec=observation_spec,
            dropout_p=dropout_p,
            structured_encoder_kind=STRUCTURED_V2_ENCODER_KIND,
        )

    def _require_observation_batch(self, obs: Tensor) -> Tensor:
        return require_observation_batch(
            obs,
            observation_dim=self.observation_dim,
            dtype=self._reference_parameter().dtype,
        )
