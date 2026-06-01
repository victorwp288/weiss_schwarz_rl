"""Candidate projection and joint scoring helpers for structured action heads."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def project_candidate_sections(
    *,
    candidate_projection: nn.Sequential,
    numeric_feature_offset: int,
    feature_sections: Sequence[tuple[Tensor, tuple[int, int]]],
    numeric_sections: Sequence[tuple[Tensor, Sequence[int]]] = (),
    constant_numeric_ones: Sequence[int] = (),
    scoring_mode: str,
) -> Tensor:
    if not isinstance(candidate_projection[0], nn.Linear):
        raise RuntimeError("structured candidate projection must begin with nn.Linear")
    linear = candidate_projection[0]
    if scoring_mode == "actor":
        inputs: list[Tensor] = []
        weight_blocks: list[Tensor] = []
        for tensor, (start, end) in feature_sections:
            if tensor.numel() == 0:
                continue
            inputs.append(tensor)
            weight_blocks.append(linear.weight[:, start:end])
        for tensor, numeric_indices in numeric_sections:
            if tensor.numel() == 0:
                continue
            inputs.append(tensor)
            column_indices = torch.as_tensor(
                [int(numeric_feature_offset) + int(index) for index in numeric_indices],
                device=linear.weight.device,
                dtype=torch.long,
            )
            weight_blocks.append(linear.weight.index_select(1, column_indices))
        if not inputs or not weight_blocks:
            raise ValueError("structured candidate projection requires at least one feature section")
        actor_projected = F.linear(
            torch.cat(inputs, dim=1),
            torch.cat(weight_blocks, dim=1),
            linear.bias,
        )
        if constant_numeric_ones:
            constant_columns = torch.as_tensor(
                [int(numeric_feature_offset) + int(index) for index in constant_numeric_ones],
                device=linear.weight.device,
                dtype=torch.long,
            )
            actor_projected = actor_projected + linear.weight.index_select(1, constant_columns).sum(dim=1).to(
                dtype=actor_projected.dtype
            )
        for module in list(candidate_projection.children())[1:]:
            actor_projected = module(actor_projected)
        return actor_projected

    projected: Tensor | None = None
    for tensor, (start, end) in feature_sections:
        if tensor.numel() == 0:
            continue
        if projected is None:
            projected = tensor.new_zeros((tensor.shape[0], linear.out_features))
            if linear.bias is not None:
                projected = projected + linear.bias.to(dtype=projected.dtype)
        projected = projected + F.linear(tensor, linear.weight[:, start:end], None)
    for tensor, numeric_indices in numeric_sections:
        if tensor.numel() == 0:
            continue
        if projected is None:
            projected = tensor.new_zeros((tensor.shape[0], linear.out_features))
            if linear.bias is not None:
                projected = projected + linear.bias.to(dtype=projected.dtype)
        column_indices = torch.as_tensor(
            [int(numeric_feature_offset) + int(index) for index in numeric_indices],
            device=linear.weight.device,
            dtype=torch.long,
        )
        projected = projected + F.linear(tensor, linear.weight.index_select(1, column_indices), None)
    if projected is None:
        raise ValueError("structured candidate projection requires at least one feature section")
    if constant_numeric_ones:
        constant_columns = torch.as_tensor(
            [int(numeric_feature_offset) + int(index) for index in constant_numeric_ones],
            device=linear.weight.device,
            dtype=torch.long,
        )
        projected = projected + linear.weight.index_select(1, constant_columns).sum(dim=1).to(dtype=projected.dtype)
    for module in list(candidate_projection.children())[1:]:
        projected = module(projected)
    return projected


def score_candidate_group(
    row_states: Tensor,
    *,
    candidate_projection: nn.Sequential,
    joint_scorer: nn.Sequential,
    numeric_feature_offset: int,
    feature_sections: Sequence[tuple[Tensor, tuple[int, int]]],
    numeric_sections: Sequence[tuple[Tensor, Sequence[int]]] = (),
    constant_numeric_ones: Sequence[int] = (),
    scoring_mode: str,
) -> Tensor:
    if row_states.numel() == 0:
        return row_states.new_zeros((0,))
    candidate_repr = project_candidate_sections(
        candidate_projection=candidate_projection,
        numeric_feature_offset=numeric_feature_offset,
        feature_sections=feature_sections,
        numeric_sections=numeric_sections,
        constant_numeric_ones=constant_numeric_ones,
        scoring_mode=scoring_mode,
    )
    if scoring_mode == "actor":
        return joint_scorer(torch.cat([row_states, candidate_repr], dim=1)).squeeze(-1).to(dtype=row_states.dtype)
    if not isinstance(joint_scorer[0], nn.Linear):
        raise RuntimeError("structured joint scorer must begin with nn.Linear")
    joint_linear = joint_scorer[0]
    state_width = row_states.shape[1]
    joint_hidden = F.linear(row_states, joint_linear.weight[:, :state_width], joint_linear.bias)
    joint_hidden = joint_hidden + F.linear(candidate_repr, joint_linear.weight[:, state_width:], None)
    for module in list(joint_scorer.children())[1:]:
        joint_hidden = module(joint_hidden)
    return joint_hidden.squeeze(-1).to(dtype=row_states.dtype)


__all__ = ["project_candidate_sections", "score_candidate_group"]
