"""Conditional argument log-probability heads for factorized action scoring."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from weiss_rl.models.tensor_ops import masked_log_softmax as _masked_log_softmax


class FactorizedConditionalLogProbsMixin:
    def _hand_arg0_log_probs(
        self: Any,
        row_states: Tensor,
        *,
        family_id: int,
        hand_ids: Tensor,
        legal_mask: Tensor,
    ) -> Tensor:
        chunk_size = self._factorized_row_chunk_size(row_states)
        if chunk_size > 0 and row_states.shape[0] > chunk_size:
            parts = [
                self._hand_arg0_log_probs(
                    row_states[start:stop],
                    family_id=family_id,
                    hand_ids=hand_ids[start:stop],
                    legal_mask=legal_mask[start:stop],
                )
                for start in range(0, row_states.shape[0], chunk_size)
                for stop in (min(start + chunk_size, row_states.shape[0]),)
            ]
            return torch.cat(parts, dim=0)
        if legal_mask.shape[1] == 0:
            return row_states.new_zeros((row_states.shape[0], 0))
        condition = self._family_condition_input(row_states, family_id=family_id)
        query = self.hand_query_head(condition)
        hand_repr = self._card_representation(hand_ids, dtype=row_states.dtype)
        if hand_repr.shape[1] < legal_mask.shape[1]:
            raise ValueError("hand representation width must cover the factorized hand domain")
        hand_repr = hand_repr[:, : legal_mask.shape[1], :]
        return self._dot_product_log_probs(query, hand_repr, legal_mask)

    def _slot_arg0_log_probs(
        self: Any,
        row_states: Tensor,
        *,
        family_id: int,
        slot_context: Tensor,
        legal_mask: Tensor,
    ) -> Tensor:
        if legal_mask.shape[1] == 0:
            return row_states.new_zeros((row_states.shape[0], 0))
        condition = self._family_condition_input(row_states, family_id=family_id)
        query = self.slot_query_head(condition)
        if slot_context.shape[1] < legal_mask.shape[1]:
            raise ValueError("slot context width must cover the factorized slot domain")
        return self._dot_product_log_probs(query, slot_context[:, : legal_mask.shape[1], :], legal_mask)

    def _index_arg0_log_probs(
        self: Any,
        row_states: Tensor,
        *,
        family_id: int,
        legal_mask: Tensor,
    ) -> Tensor:
        if legal_mask.shape[1] == 0:
            return row_states.new_zeros((row_states.shape[0], 0))
        condition = self._family_condition_input(row_states, family_id=family_id)
        query = self.index_query_head(condition)
        index_repr = self.generic_index_embedding(
            torch.arange(legal_mask.shape[1], device=row_states.device, dtype=torch.long)
        ).to(dtype=row_states.dtype)
        logits = torch.matmul(query, index_repr.transpose(0, 1))
        return _masked_log_softmax(logits, legal_mask)

    def _play_arg1_log_probs(
        self: Any,
        row_states: Tensor,
        *,
        hand_ids: Tensor,
        slot_context: Tensor,
        legal_mask: Tensor,
    ) -> Tensor:
        chunk_size = self._factorized_row_chunk_size(row_states)
        if chunk_size > 0 and row_states.shape[0] > chunk_size:
            parts = [
                self._play_arg1_log_probs(
                    row_states[start:stop],
                    hand_ids=hand_ids[start:stop],
                    slot_context=slot_context[start:stop],
                    legal_mask=legal_mask[start:stop],
                )
                for start in range(0, row_states.shape[0], chunk_size)
                for stop in (min(start + chunk_size, row_states.shape[0]),)
            ]
            return torch.cat(parts, dim=0)
        if legal_mask.shape[2] == 0:
            return row_states.new_zeros((row_states.shape[0], legal_mask.shape[1], 0))
        hand_repr = self._card_representation(hand_ids, dtype=row_states.dtype)
        if hand_repr.shape[1] < legal_mask.shape[1]:
            raise ValueError("hand representation width must cover the factorized play domain")
        hand_repr = hand_repr[:, : legal_mask.shape[1], :]
        family_condition = self.family_embedding(
            torch.full(
                (row_states.shape[0],), self._play_character_family_id, device=row_states.device, dtype=torch.long
            )
        ).to(dtype=row_states.dtype)
        state_expanded = row_states.unsqueeze(1).expand(-1, legal_mask.shape[1], -1)
        family_expanded = family_condition.unsqueeze(1).expand(-1, legal_mask.shape[1], -1)
        query = self.play_slot_query_head(torch.cat([state_expanded, family_expanded, hand_repr], dim=-1))
        slot_expanded = slot_context.unsqueeze(1).expand(-1, legal_mask.shape[1], -1, -1)
        logits = (slot_expanded.to(dtype=row_states.dtype) * query.unsqueeze(2)).sum(dim=-1)
        return _masked_log_softmax(
            logits.reshape(-1, logits.shape[-1]), legal_mask.reshape(-1, legal_mask.shape[-1])
        ).reshape_as(logits)

    def _move_arg1_log_probs(
        self: Any,
        row_states: Tensor,
        *,
        slot_context: Tensor,
        legal_mask: Tensor,
    ) -> Tensor:
        chunk_size = self._factorized_row_chunk_size(row_states)
        if chunk_size > 0 and row_states.shape[0] > chunk_size:
            parts = [
                self._move_arg1_log_probs(
                    row_states[start:stop],
                    slot_context=slot_context[start:stop],
                    legal_mask=legal_mask[start:stop],
                )
                for start in range(0, row_states.shape[0], chunk_size)
                for stop in (min(start + chunk_size, row_states.shape[0]),)
            ]
            return torch.cat(parts, dim=0)
        if legal_mask.shape[2] == 0:
            return row_states.new_zeros((row_states.shape[0], legal_mask.shape[1], 0))
        family_condition = self.family_embedding(
            torch.full((row_states.shape[0],), self._main_move_family_id, device=row_states.device, dtype=torch.long)
        ).to(dtype=row_states.dtype)
        source_context = slot_context.unsqueeze(1).expand(-1, legal_mask.shape[1], -1, -1)
        family_expanded = family_condition.unsqueeze(1).expand(-1, legal_mask.shape[1], -1)
        state_expanded = row_states.unsqueeze(1).expand(-1, legal_mask.shape[1], -1)
        if slot_context.shape[1] < legal_mask.shape[1]:
            raise ValueError("slot context width must cover the factorized move domain")
        query = self.move_target_query_head(
            torch.cat([state_expanded, family_expanded, slot_context[:, : legal_mask.shape[1], :]], dim=-1)
        )
        logits = (source_context.to(dtype=row_states.dtype) * query.unsqueeze(2)).sum(dim=-1)
        return _masked_log_softmax(
            logits.reshape(-1, logits.shape[-1]), legal_mask.reshape(-1, legal_mask.shape[-1])
        ).reshape_as(logits)

    def _attack_arg1_log_probs(
        self: Any,
        row_states: Tensor,
        *,
        slot_context: Tensor,
        legal_mask: Tensor,
    ) -> Tensor:
        chunk_size = self._factorized_row_chunk_size(row_states)
        if chunk_size > 0 and row_states.shape[0] > chunk_size:
            parts = [
                self._attack_arg1_log_probs(
                    row_states[start:stop],
                    slot_context=slot_context[start:stop],
                    legal_mask=legal_mask[start:stop],
                )
                for start in range(0, row_states.shape[0], chunk_size)
                for stop in (min(start + chunk_size, row_states.shape[0]),)
            ]
            return torch.cat(parts, dim=0)
        if legal_mask.shape[2] == 0:
            return row_states.new_zeros((row_states.shape[0], legal_mask.shape[1], 0))
        family_condition = self.family_embedding(
            torch.full((row_states.shape[0],), self._attack_family_id, device=row_states.device, dtype=torch.long)
        ).to(dtype=row_states.dtype)
        type_repr = self.attack_type_embedding(
            torch.arange(legal_mask.shape[2], device=row_states.device, dtype=torch.long) + 1
        ).to(dtype=row_states.dtype)
        family_expanded = family_condition.unsqueeze(1).expand(-1, legal_mask.shape[1], -1)
        state_expanded = row_states.unsqueeze(1).expand(-1, legal_mask.shape[1], -1)
        if slot_context.shape[1] < legal_mask.shape[1]:
            raise ValueError("slot context width must cover the factorized attack domain")
        query = self.attack_type_query_head(
            torch.cat([state_expanded, family_expanded, slot_context[:, : legal_mask.shape[1], :]], dim=-1)
        )
        logits = torch.einsum("bqd,td->bqt", query, type_repr)
        return _masked_log_softmax(
            logits.reshape(-1, logits.shape[-1]), legal_mask.reshape(-1, legal_mask.shape[-1])
        ).reshape_as(logits)


__all__ = ["FactorizedConditionalLogProbsMixin"]
