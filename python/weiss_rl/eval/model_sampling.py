"""Model action-scoring helpers shared by final eval and periodic dev eval."""

from __future__ import annotations

from inspect import Parameter, signature
from typing import Any

import numpy as np
import torch

from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.runtime.components.legal_meta import action_catalog_indices, legal_action_meta_from_ids
from weiss_rl.runtime.components.shared import DEFAULT_ACTION_META_WIDTH


def _call_accepts_keyword(callable_obj: Any, keyword: str) -> bool:
    try:
        parameters = signature(callable_obj).parameters
    except (TypeError, ValueError):
        return False
    if keyword in parameters:
        return True
    return any(parameter.kind == Parameter.VAR_KEYWORD for parameter in parameters.values())


def _add_opponent_context_if_supported(
    callable_obj: Any, kwargs: dict[str, Any], context_tensor: torch.Tensor | None
) -> None:
    if context_tensor is not None and _call_accepts_keyword(callable_obj, "opponent_context_index"):
        kwargs["opponent_context_index"] = context_tensor


def _legal_action_meta_for_eval_row(
    *,
    model: Any,
    batch: Any,
    legal_ids: np.ndarray,
    env_index: int,
) -> np.ndarray | None:
    ids_offsets = getattr(batch, "ids_offsets", None)
    legal_action_meta = getattr(batch, "legal_action_meta", None)
    if legal_action_meta is not None and ids_offsets is not None:
        _packed_ids, packed_offsets = ids_offsets
        start = int(packed_offsets[int(env_index)])
        end = int(packed_offsets[int(env_index) + 1])
        return np.asarray(legal_action_meta[start:end], dtype=np.uint16)

    action_catalog = getattr(model, "action_catalog", None)
    if action_catalog is None:
        return None
    family_index, attack_type_index = action_catalog_indices(action_catalog)
    return legal_action_meta_from_ids(
        np.asarray(legal_ids, dtype=np.uint32),
        action_catalog=action_catalog,
        family_index=family_index,
        attack_type_index=attack_type_index,
        action_meta_width=DEFAULT_ACTION_META_WIDTH,
    )


def model_eval_logits_for_legal_ids(
    *,
    model: Any,
    batch: Any,
    current_seat: int,
    seat_hidden: torch.Tensor,
    legal_ids: np.ndarray,
    action_dim: int,
    device: torch.device,
    env_index: int = 0,
    opponent_context_index: int | None = None,
) -> tuple[np.ndarray, torch.Tensor]:
    """Return dense eval logits while preserving structured packed-policy scoring."""
    obs_tensor = torch.as_tensor(np.asarray(batch.obs, dtype=np.float32), device=device)
    seat_tensor = torch.as_tensor([int(current_seat)], device=device, dtype=torch.long)
    context_tensor = (
        None
        if opponent_context_index is None
        else torch.as_tensor([int(opponent_context_index)], device=device, dtype=torch.long)
    )
    if bool(getattr(model, "supports_legal_candidate_scoring", False)):
        legal_ids_array = np.asarray(legal_ids, dtype=np.uint32)
        legal_actions = LegalActionBatch.from_packed(
            legal_ids_array,
            np.asarray([0, int(legal_ids_array.shape[0])], dtype=np.uint32),
            meta=_legal_action_meta_for_eval_row(
                model=model,
                batch=batch,
                legal_ids=legal_ids_array,
                env_index=int(env_index),
            ),
            action_space=int(action_dim),
        )
        if bool(getattr(model, "supports_factorized_legal_policy", False)):
            factorized_eval = getattr(model, "factorized_packed_action_log_probs_seat_aware", None)
            if not callable(factorized_eval):
                raise RuntimeError("factorized_v1 model eval requires factorized packed action log-probs")
            eval_kwargs: dict[str, Any] = {
                "legal_actions": legal_actions,
                "scoring_mode": "learner",
            }
            _add_opponent_context_if_supported(factorized_eval, eval_kwargs, context_tensor)
            packed_logits, _value_tensor, next_seat_hidden = factorized_eval(
                obs_tensor,
                seat_tensor,
                seat_hidden,
                **eval_kwargs,
            )
            dense_logits = np.full((int(action_dim),), -np.inf, dtype=np.float32)
            if legal_ids_array.size:
                dense_logits[legal_ids_array.astype(np.intp, copy=False)] = (
                    packed_logits.detach().cpu().numpy().astype(np.float32, copy=False)
                )
            return dense_logits, next_seat_hidden
        packed_kwargs: dict[str, Any] = {
            "legal_actions": legal_actions,
            "scoring_mode": "learner",
        }
        _add_opponent_context_if_supported(model.forward_packed_seat_aware, packed_kwargs, context_tensor)
        packed_logits, _value_tensor, next_seat_hidden = model.forward_packed_seat_aware(
            obs_tensor,
            seat_tensor,
            seat_hidden,
            **packed_kwargs,
        )
        dense_logits = np.zeros((int(action_dim),), dtype=np.float32)
        if legal_ids_array.size:
            dense_logits[legal_ids_array.astype(np.intp, copy=False)] = (
                packed_logits.detach().cpu().numpy().astype(np.float32, copy=False)
            )
        return dense_logits, next_seat_hidden

    dense_kwargs: dict[str, Any] = {"scoring_mode": "learner"}
    _add_opponent_context_if_supported(model.forward_seat_aware, dense_kwargs, context_tensor)
    logits_tensor, _value_tensor, next_seat_hidden = model.forward_seat_aware(
        obs_tensor,
        seat_tensor,
        seat_hidden,
        **dense_kwargs,
    )
    return logits_tensor[0].detach().cpu().numpy().astype(np.float32, copy=False), next_seat_hidden


__all__ = ["model_eval_logits_for_legal_ids"]
