from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def _numeric_value(record: Mapping[str, Any], key: str) -> float | None:
    value = record.get(key)
    if not isinstance(value, int | float):
        custom_metrics = record.get("custom_metrics")
        if isinstance(custom_metrics, dict):
            value = custom_metrics.get(key)
    return float(value) if isinstance(value, int | float) else None


def _numeric_values(records: Iterable[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = _numeric_value(record, key)
        if isinstance(value, int | float):
            values.append(float(value))
    return values


def _fraction_values(records: Iterable[dict[str, Any]], numerator_key: str, denominator_key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        numerator = _numeric_value(record, numerator_key)
        denominator = _numeric_value(record, denominator_key)
        if numerator is None or denominator is None or denominator <= 0.0:
            continue
        values.append(float(numerator) / float(denominator))
    return values


def _ratio_values(
    records: Iterable[dict[str, Any]],
    numerator_key: str,
    denominator_keys: tuple[str, ...],
) -> list[float]:
    values: list[float] = []
    for record in records:
        numerator = _numeric_value(record, numerator_key)
        if numerator is None:
            continue
        denominator = 0.0
        complete = True
        for key in denominator_keys:
            value = _numeric_value(record, key)
            if value is None:
                complete = False
                break
            denominator += float(value)
        if not complete or denominator <= 0.0:
            continue
        values.append(float(numerator) / denominator)
    return values


def _sum_fraction_values(
    records: Iterable[dict[str, Any]],
    numerator_keys: tuple[str, ...],
    denominator_keys: tuple[str, ...],
) -> list[float]:
    values: list[float] = []
    for record in records:
        numerator = 0.0
        denominator = 0.0
        complete = True
        for key in numerator_keys:
            value = _numeric_value(record, key)
            if value is None:
                complete = False
                break
            numerator += float(value)
        if not complete:
            continue
        for key in denominator_keys:
            value = _numeric_value(record, key)
            if value is None:
                complete = False
                break
            denominator += float(value)
        if complete and denominator > 0.0:
            values.append(numerator / denominator)
    return values


def _numeric_by_update(records: Iterable[dict[str, Any]], key: str) -> dict[int, float]:
    values: dict[int, float] = {}
    for record in records:
        update_count = record.get("update_count")
        value = _numeric_value(record, key)
        if isinstance(update_count, int) and value is not None:
            values[int(update_count)] = float(value)
    return values


def _paired_update_values(
    left_records: Iterable[dict[str, Any]],
    left_key: str,
    right_records: Iterable[dict[str, Any]],
    right_key: str,
) -> list[tuple[float, float]]:
    left = _numeric_by_update(left_records, left_key)
    right = _numeric_by_update(right_records, right_key)
    return [(left[update], right[update]) for update in sorted(left.keys() & right.keys())]


def _pearson_correlation(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    left_values = [left for left, _right in pairs]
    right_values = [right for _left, right in pairs]
    left_mean = sum(left_values) / len(left_values)
    right_mean = sum(right_values) / len(right_values)
    left_centered = [value - left_mean for value in left_values]
    right_centered = [value - right_mean for value in right_values]
    left_ss = sum(value * value for value in left_centered)
    right_ss = sum(value * value for value in right_centered)
    if left_ss <= 0.0 or right_ss <= 0.0:
        return None
    covariance = sum(left * right for left, right in zip(left_centered, right_centered, strict=True))
    return float(covariance / ((left_ss * right_ss) ** 0.5))


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _window_summary(values: list[float], *, window: int) -> dict[str, float | None]:
    if not values:
        return {"first": None, "last": None, "first_window_mean": None, "last_window_mean": None}
    first_window = values[:window]
    last_window = values[-window:]
    return {
        "first": values[0],
        "last": values[-1],
        "first_window_mean": _mean(first_window),
        "last_window_mean": _mean(last_window),
    }


def _last_window_mean(values: list[float], *, window: int) -> float | None:
    return _window_summary(values, window=window)["last_window_mean"]
