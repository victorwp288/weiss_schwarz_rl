"""Structured model observation-context construction helpers."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn

from weiss_rl.core.observation_layout import ObservationSlice
from weiss_rl.models.feature_gathering import slot_component
from weiss_rl.models.observation_contract import StructuredObservationContract
from weiss_rl.models.tensor_ops import masked_max_pool, masked_mean_pool


def encode_observation_context(
    *,
    obs_batch: Tensor,
    observation_contract: StructuredObservationContract,
    slot_context_dim: int,
    stage_slot_count: int,
    card_representation: Callable[..., Tensor],
    hand_summary_projection: nn.Module,
    slot_encoder: nn.Module,
) -> dict[str, Tensor]:
    batch_size = obs_batch.shape[0]
    dtype = obs_batch.dtype

    hand_ids = extract_card_vector(obs_batch, observation_contract.self_hand)
    if hand_ids.shape[1] == 0:
        hand_summary = obs_batch.new_zeros((batch_size, slot_context_dim))
    else:
        hand_mask = hand_ids > max(observation_contract.sentinel_empty_card, 0)
        hand_embeddings = card_representation(hand_ids, dtype=dtype)
        hand_summary = hand_summary_projection(
            torch.cat(
                [
                    masked_mean_pool(hand_embeddings, hand_mask),
                    masked_max_pool(hand_embeddings, hand_mask),
                    hand_mask.to(dtype=dtype).mean(dim=1, keepdim=True),
                ],
                dim=1,
            )
        )

    self_stage_ctx, self_stage_numeric = encode_stage_slice(
        obs_batch=obs_batch,
        stage_slice=observation_contract.self_stage,
        observation_contract=observation_contract,
        stage_slot_count=stage_slot_count,
        slot_context_dim=slot_context_dim,
        card_representation=card_representation,
        slot_encoder=slot_encoder,
    )
    opponent_stage_ctx, opponent_stage_numeric = encode_stage_slice(
        obs_batch=obs_batch,
        stage_slice=observation_contract.opponent_stage,
        observation_contract=observation_contract,
        stage_slot_count=stage_slot_count,
        slot_context_dim=slot_context_dim,
        card_representation=card_representation,
        slot_encoder=slot_encoder,
    )
    return {
        "hand_ids": hand_ids,
        "hand_summary": hand_summary,
        "self_stage_context": self_stage_ctx,
        "self_stage_numeric": self_stage_numeric,
        "self_stage_summary": self_stage_ctx.mean(dim=1),
        "self_level_count": extract_scalar_feature(obs_batch, observation_contract.self_level_count),
        "self_clock_count": extract_scalar_feature(obs_batch, observation_contract.self_clock_count),
        "opponent_stage_context": opponent_stage_ctx,
        "opponent_stage_numeric": opponent_stage_numeric,
        "opponent_stage_summary": opponent_stage_ctx.mean(dim=1),
        "choice_page_start": extract_header_scalar(obs_batch, observation_contract.choice_page_start_index),
        "choice_total": extract_header_scalar(obs_batch, observation_contract.choice_total_index),
    }


def extract_scalar_feature(obs_batch: Tensor, slice_spec: ObservationSlice | None) -> Tensor:
    batch_size = obs_batch.shape[0]
    if slice_spec is None or slice_spec.length <= 0:
        return obs_batch.new_zeros((batch_size,))
    return obs_batch[:, slice_spec.start].reshape(batch_size)


def extract_header_scalar(obs_batch: Tensor, index: int | None) -> Tensor:
    batch_size = obs_batch.shape[0]
    if index is None:
        return obs_batch.new_zeros((batch_size,))
    return obs_batch[:, int(index)].reshape(batch_size)


def encode_stage_slice(
    *,
    obs_batch: Tensor,
    stage_slice: ObservationSlice | None,
    observation_contract: StructuredObservationContract,
    stage_slot_count: int,
    slot_context_dim: int,
    card_representation: Callable[..., Tensor],
    slot_encoder: nn.Module,
) -> tuple[Tensor, Tensor]:
    batch_size = obs_batch.shape[0]
    dtype = obs_batch.dtype
    if stage_slice is None:
        zeros_context = obs_batch.new_zeros((batch_size, stage_slot_count, slot_context_dim))
        zeros_numeric = obs_batch.new_zeros((batch_size, stage_slot_count, 7))
        return zeros_context, zeros_numeric

    slot_width = max(stage_slice.length // stage_slot_count, 1)
    stage_values = obs_batch[:, stage_slice.start : stage_slice.stop].reshape(batch_size, stage_slot_count, slot_width)
    card_ids = stage_values[..., 0].to(dtype=torch.long)
    occupied = (card_ids > max(observation_contract.sentinel_empty_card, 0)).to(dtype=dtype)
    numeric = torch.stack(
        [
            occupied,
            slot_component(stage_values, 1) / 8.0,
            slot_component(stage_values, 2),
            slot_component(stage_values, 3) / 20000.0,
            slot_component(stage_values, 4) / 4.0,
            slot_component(stage_values, 5) / 4.0,
            slot_component(stage_values, 6),
        ],
        dim=-1,
    )
    card_embeddings = card_representation(card_ids, dtype=dtype)
    stage_context = slot_encoder(torch.cat([card_embeddings, numeric], dim=-1))
    return stage_context, numeric


def extract_card_vector(obs_batch: Tensor, observation_slice: ObservationSlice | None) -> Tensor:
    if observation_slice is None:
        return torch.zeros((obs_batch.shape[0], 0), device=obs_batch.device, dtype=torch.long)
    return obs_batch[:, observation_slice.start : observation_slice.stop].to(dtype=torch.long)
