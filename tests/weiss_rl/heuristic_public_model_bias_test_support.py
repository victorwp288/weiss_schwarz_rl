from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from weiss_rl.config.models import ModelConfig, ModelDropoutConfig
from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.eval.heuristic_public.heuristic_public import HeuristicPublicPolicy
from weiss_rl.model import StructuredLegalPolicyValueModel, build_policy_value_model

from .heuristic_public_test_support import _empty_obs, _heuristic_spec_bundle, _packed_meta, _set_stage


@dataclass(frozen=True)
class PublicBiasRows:
    spec_bundle: dict[str, Any]
    policy: HeuristicPublicPolicy
    obs_rows: np.ndarray
    legal_ids: np.ndarray
    offsets: np.ndarray
    meta: np.ndarray
    legal_batch: LegalActionBatch
    obs_tensor: torch.Tensor


def _public_bias_model(
    *,
    spec_bundle: dict[str, Any],
    public_heuristic_logit_bias_scale: float = 0.0,
    public_heuristic_actor_logit_bias_scale: float = -1.0,
    public_heuristic_logit_bias_families: tuple[str, ...] = (),
) -> StructuredLegalPolicyValueModel:
    model = build_policy_value_model(
        observation_dim=int(spec_bundle["observation"]["obs_len"]),
        config=ModelConfig(
            gru_hidden_size=64,
            encoder_mlp_width=64,
            encoder_mlp_layers=1,
            layer_norm=True,
            dropout=ModelDropoutConfig(family_a=0.0, ablation=0.0),
            encoder_kind="structured_v2",
            typed_feature_width=32,
            public_heuristic_logit_bias_scale=public_heuristic_logit_bias_scale,
            public_heuristic_actor_logit_bias_scale=public_heuristic_actor_logit_bias_scale,
            public_heuristic_logit_bias_families=public_heuristic_logit_bias_families,
        ),
        action_dim=int(spec_bundle["action"]["action_space_size"]),
        observation_spec=cast(dict[str, Any], spec_bundle["observation"]),
        spec_bundle=spec_bundle,
    )
    assert isinstance(model, StructuredLegalPolicyValueModel)
    return model


def _public_bias_rows() -> PublicBiasRows:
    spec_bundle = cast(dict[str, Any], _heuristic_spec_bundle())
    policy = HeuristicPublicPolicy.from_spec_bundle(spec_bundle)
    obs_attack = _empty_obs()
    _set_stage(obs_attack, player_index=0, slot=0, occupied=True, power=5000, effective_soul=1)
    obs_play = _empty_obs()
    _set_stage(obs_play, player_index=0, slot=1, occupied=True, power=2500)
    obs_clock = _empty_obs()
    obs_clock[16] = 0
    obs_clock[17] = 3
    obs_clock[14] = 16
    obs_clock[15] = 40

    row_legal_ids = [
        np.array([472, 473, 474, 51, 402], dtype=np.uint32),
        np.array([102, 103, 104, 105, 106], dtype=np.uint32),
        np.array([52, 53, 524, 525, 51], dtype=np.uint32),
    ]
    obs_rows = np.stack([obs_attack, obs_play, obs_clock], axis=0)
    legal_ids = np.concatenate(row_legal_ids, axis=0)
    offsets = np.asarray(
        [0, row_legal_ids[0].size, row_legal_ids[0].size + row_legal_ids[1].size, legal_ids.size],
        dtype=np.uint32,
    )
    meta = _packed_meta(legal_ids)
    legal_batch = LegalActionBatch.from_packed(
        legal_ids,
        offsets,
        meta=meta,
        action_space=int(spec_bundle["action"]["action_space_size"]),
    )
    return PublicBiasRows(
        spec_bundle=spec_bundle,
        policy=policy,
        obs_rows=obs_rows,
        legal_ids=legal_ids,
        offsets=offsets,
        meta=meta,
        legal_batch=legal_batch,
        obs_tensor=torch.as_tensor(obs_rows, dtype=torch.float32),
    )


def _best_actions_from_packed_scores(rows: PublicBiasRows, scores: np.ndarray) -> np.ndarray:
    chosen = np.full((rows.obs_rows.shape[0],), rows.policy.pass_action_id, dtype=np.int64)
    for row_index in range(rows.obs_rows.shape[0]):
        start = int(rows.offsets[row_index])
        stop = int(rows.offsets[row_index + 1])
        row_scores = scores[start:stop]
        row_best = np.flatnonzero(row_scores == row_scores.max())
        chosen[row_index] = int(rows.legal_ids[start:stop][int(row_best[0])])
    return chosen


def _actor_packed_scores(model: StructuredLegalPolicyValueModel, rows: PublicBiasRows) -> np.ndarray:
    with torch.no_grad():
        encoded = model.encode(rows.obs_tensor)
        recurrent_output, _ = model.recurrent_step_seat_aware(encoded, 0, None)
        return (
            model.score_packed_legal_candidates(
                recurrent_output,
                rows.obs_tensor,
                rows.legal_batch,
                scoring_mode="actor",
            )
            .detach()
            .cpu()
            .numpy()
        )
