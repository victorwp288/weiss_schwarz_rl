"""Experiment, system, and model config records."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    role: str


@dataclass(frozen=True, slots=True)
class SystemProfileConfig:
    training: str
    local_iteration: str
    ci_invariant_testing: str


@dataclass(frozen=True, slots=True)
class SystemConfig:
    profile: SystemProfileConfig
    mp_start_method: str
    learner_device: str
    actor_device: str
    actor_process_count: int
    envs_per_actor: int
    total_envs: int
    actor_torch_threads: int
    learner_torch_threads: int
    actor_queue_capacity_unrolls: int
    learner_prefetch_batches: int
    collection_backend: str = "auto"


@dataclass(frozen=True, slots=True)
class ModelDropoutConfig:
    family_a: float
    ablation: float


@dataclass(frozen=True, slots=True)
class ModelConfig:
    gru_hidden_size: int
    encoder_mlp_width: int
    encoder_mlp_layers: int
    layer_norm: bool
    dropout: ModelDropoutConfig
    encoder_kind: str = "mlp"
    structured_policy_contract: str = "packed_v1"
    typed_feature_width: int = 64
    recurrent_core: str = "gru"
    candidate_scoring_chunk_size: int = 65536
    cuda_learner_candidate_scoring_chunk_size: int = 262144
    public_heuristic_logit_bias_scale: float = 0.0
    public_heuristic_actor_logit_bias_scale: float = -1.0
    public_heuristic_logit_bias_start_updates: int = 0
    public_heuristic_logit_bias_end_updates: int = -1
    public_heuristic_logit_bias_final_scale: float = 0.0
    public_heuristic_logit_bias_families: tuple[str, ...] = field(default_factory=tuple)
    opponent_context_policy_ids: tuple[str, ...] = field(default_factory=tuple)
    opponent_context_hidden_scale: float = 0.0
    opponent_context_trainable_hidden_scale: float = 0.0
    opponent_context_trainable_recurrent_scale: float = 0.0
    opponent_context_trainable_action_bias_scale: float = 0.0
    opponent_context_trainable_candidate_residual_scale: float = 0.0
    opponent_context_candidate_residual_width: int = 32
    opponent_context_candidate_residual_mode: str = "additive"
    opponent_context_candidate_residual_action_ids: tuple[int, ...] = field(default_factory=tuple)
    opponent_context_adapter_lr_multiplier: float = 1.0
    opponent_context_adapter_train_only: bool = False
    opponent_context_eval_policy_ids: tuple[str, ...] = field(default_factory=tuple)


__all__ = [
    "ExperimentConfig",
    "ModelConfig",
    "ModelDropoutConfig",
    "SystemConfig",
    "SystemProfileConfig",
]
