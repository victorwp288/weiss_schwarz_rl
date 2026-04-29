"""LRU cache helpers for eval snapshot models."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Protocol, TypeVar


class EvalModeModel(Protocol):
    def eval(self) -> Any: ...


ModelT = TypeVar("ModelT", bound=EvalModeModel)


def get_cached_eval_model(
    cache: OrderedDict[tuple[Any, ...], ModelT],
    cache_key: tuple[Any, ...],
) -> ModelT | None:
    cached_model = cache.get(cache_key)
    if cached_model is not None:
        cache.move_to_end(cache_key)
        cached_model.eval()
    return cached_model


def remember_eval_model(
    cache: OrderedDict[tuple[Any, ...], ModelT],
    cache_key: tuple[Any, ...],
    eval_model: ModelT,
    *,
    max_entries: int,
) -> None:
    cache[cache_key] = eval_model
    cache.move_to_end(cache_key)
    while len(cache) > int(max_entries):
        cache.popitem(last=False)
