"""Lightweight policy-alignment summaries for live eval diagnostics."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.core.action_distribution_metrics import (
    append_optional,
    family_probability_masses,
    gap_from_top_to_action,
    legal_indices_from_ids,
    masked_softmax,
    max_non_reference_probability,
    percentile_summary,
    rank_of_action,
    same_family_margin_to_action,
    simple_action_descriptor,
    top_action_payload,
    top_margin,
)


class PolicyAlignmentAccumulator:
    """Accumulate model-vs-reference action-distribution diagnostics."""

    def __init__(self, *, action_catalog: ActionCatalog | None = None) -> None:
        self._action_catalog = action_catalog
        self._compared_steps = 0
        self._skipped_steps = 0
        self._total_variation: list[float] = []
        self._max_abs_probability_delta: list[float] = []
        self._top_action_matches: list[bool] = []
        self._top_family_matches: list[bool] = []
        self._probability_on_reference_top_action: list[float] = []
        self._probability_on_reference_top_action_family: list[float] = []
        self._rank_of_reference_top_action: list[int] = []
        self._model_top_logit_margin: list[float] = []
        self._model_top_probability_margin: list[float] = []
        self._model_gap_from_top_logit_to_reference_top_action: list[float] = []
        self._reference_top_action_same_family_logit_margin: list[float] = []
        self._reference_top_family_items: defaultdict[str, list[_StepSummary]] = defaultdict(list)
        self._family_confusions: Counter[tuple[str, str]] = Counter()
        self._model_family_mass_totals: defaultdict[str, float] = defaultdict(float)

    def add(
        self,
        *,
        model_logits: np.ndarray,
        legal_ids: np.ndarray,
        reference_action_id: int,
    ) -> None:
        legal_indices = legal_indices_from_ids(legal_ids=legal_ids, action_dim=int(np.asarray(model_logits).shape[0]))
        reference_action = int(reference_action_id)
        if legal_indices.size == 0 or not bool(np.any(legal_indices == reference_action)):
            self._skipped_steps += 1
            return

        logits = np.asarray(model_logits, dtype=np.float64)
        probabilities = masked_softmax(logits=logits, legal_indices=legal_indices)
        model_top_action = top_action_payload(
            probabilities=probabilities,
            legal_indices=legal_indices,
            action_catalog=self._action_catalog,
        )
        reference_top_action = simple_action_descriptor(reference_action, action_catalog=self._action_catalog)
        model_top_action_id = int(model_top_action["action"])
        model_family_masses = family_probability_masses(
            probabilities=probabilities,
            legal_indices=legal_indices,
            action_catalog=self._action_catalog,
        )
        model_top_family = str(model_top_action.get("family", "unknown"))
        reference_top_family = str(reference_top_action.get("family", "unknown"))
        reference_action_probability = float(probabilities[reference_action])
        reference_family_probability = float(model_family_masses.get(reference_top_family, 0.0))
        same_family_margin = same_family_margin_to_action(
            values=logits,
            legal_indices=legal_indices,
            action_id=reference_action,
            action_catalog=self._action_catalog,
        )
        step_summary = _StepSummary(
            model_matches_reference_top_action=model_top_action_id == reference_action,
            model_matches_reference_top_action_family=model_top_family == reference_top_family,
            model_probability_on_reference_top_action=reference_action_probability,
            model_probability_on_reference_top_action_family=reference_family_probability,
            model_reference_top_action_same_family_logit_margin=same_family_margin,
        )

        self._compared_steps += 1
        self._total_variation.append(float(1.0 - reference_action_probability))
        self._max_abs_probability_delta.append(
            float(
                max(
                    1.0 - reference_action_probability,
                    max_non_reference_probability(
                        probabilities=probabilities,
                        legal_indices=legal_indices,
                        reference_action=reference_action,
                    ),
                )
            )
        )
        self._top_action_matches.append(model_top_action_id == reference_action)
        self._top_family_matches.append(model_top_family == reference_top_family)
        self._probability_on_reference_top_action.append(reference_action_probability)
        self._probability_on_reference_top_action_family.append(reference_family_probability)
        self._rank_of_reference_top_action.append(
            rank_of_action(probabilities=probabilities, legal_indices=legal_indices, action_id=reference_action)
        )
        append_optional(
            self._model_top_logit_margin,
            top_margin(values=logits, legal_indices=legal_indices),
        )
        append_optional(
            self._model_top_probability_margin,
            top_margin(values=probabilities, legal_indices=legal_indices),
        )
        append_optional(
            self._model_gap_from_top_logit_to_reference_top_action,
            gap_from_top_to_action(values=logits, legal_indices=legal_indices, action_id=reference_action),
        )
        append_optional(self._reference_top_action_same_family_logit_margin, same_family_margin)
        self._reference_top_family_items[reference_top_family].append(step_summary)
        self._family_confusions[(reference_top_family, model_top_family)] += 1
        for family, probability in model_family_masses.items():
            self._model_family_mass_totals[str(family)] += float(probability)

    def summary(self) -> dict[str, Any]:
        if self._compared_steps == 0:
            return {
                "compared_steps": 0,
                "skipped_steps": int(self._skipped_steps),
                "max_total_variation": 0.0,
                "mean_total_variation": 0.0,
                "median_total_variation": 0.0,
                "max_abs_probability_delta": 0.0,
                "model_matches_reference_top_action_rate": 0.0,
                "model_matches_reference_top_action_family_rate": 0.0,
                "model_mean_probability_on_reference_top_action": 0.0,
                "model_mean_probability_on_reference_top_action_family": 0.0,
                "model_median_rank_of_reference_top_action": 0.0,
                "model_probability_on_reference_top_action_percentiles": percentile_summary([]),
                "model_top_logit_margin_percentiles": percentile_summary([]),
                "model_top_probability_margin_percentiles": percentile_summary([]),
                "model_gap_from_top_logit_to_reference_top_action_percentiles": percentile_summary([]),
                "model_reference_top_action_same_family_logit_margin_percentiles": percentile_summary([]),
                "reference_top_family_summaries": [],
                "top_action_family_confusions": [],
                "model_mean_family_probability_masses": [],
            }

        total_variation = np.asarray(self._total_variation, dtype=np.float64)
        max_abs_probability_delta = np.asarray(self._max_abs_probability_delta, dtype=np.float64)
        top_action_matches = np.asarray(self._top_action_matches, dtype=np.float64)
        top_family_matches = np.asarray(self._top_family_matches, dtype=np.float64)
        probability_on_reference_top_action = np.asarray(
            self._probability_on_reference_top_action,
            dtype=np.float64,
        )
        probability_on_reference_top_action_family = np.asarray(
            self._probability_on_reference_top_action_family,
            dtype=np.float64,
        )
        rank_of_reference_top_action = np.asarray(self._rank_of_reference_top_action, dtype=np.float64)
        return {
            "compared_steps": int(self._compared_steps),
            "skipped_steps": int(self._skipped_steps),
            "max_total_variation": float(np.max(total_variation)),
            "mean_total_variation": float(np.mean(total_variation)),
            "median_total_variation": float(np.median(total_variation)),
            "max_abs_probability_delta": float(np.max(max_abs_probability_delta)),
            "model_matches_reference_top_action_rate": float(np.mean(top_action_matches)),
            "model_matches_reference_top_action_family_rate": float(np.mean(top_family_matches)),
            "model_top_action_mismatch_count": int(self._compared_steps - int(np.sum(top_action_matches))),
            "model_top_action_family_mismatch_count": int(self._compared_steps - int(np.sum(top_family_matches))),
            "model_mean_probability_on_reference_top_action": float(np.mean(probability_on_reference_top_action)),
            "model_mean_probability_on_reference_top_action_family": float(
                np.mean(probability_on_reference_top_action_family)
            ),
            "model_median_rank_of_reference_top_action": float(np.median(rank_of_reference_top_action)),
            "model_probability_on_reference_top_action_percentiles": percentile_summary(
                probability_on_reference_top_action.tolist()
            ),
            "model_top_logit_margin_percentiles": percentile_summary(self._model_top_logit_margin),
            "model_top_probability_margin_percentiles": percentile_summary(self._model_top_probability_margin),
            "model_gap_from_top_logit_to_reference_top_action_percentiles": percentile_summary(
                self._model_gap_from_top_logit_to_reference_top_action
            ),
            "model_reference_top_action_same_family_logit_margin_percentiles": percentile_summary(
                self._reference_top_action_same_family_logit_margin
            ),
            "reference_top_family_summaries": self._reference_top_family_summaries(),
            "top_action_family_confusions": [
                {
                    "reference_family": reference_family,
                    "model_family": model_family,
                    "count": int(count),
                }
                for (reference_family, model_family), count in self._family_confusions.most_common()
            ],
            "model_mean_family_probability_masses": [
                {"family": family, "mean_probability": float(total / self._compared_steps)}
                for family, total in sorted(
                    self._model_family_mass_totals.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ],
        }

    def _reference_top_family_summaries(self) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        for family, family_items in sorted(
            self._reference_top_family_items.items(),
            key=lambda item: (-len(item[1]), item[0]),
        ):
            action_matches = np.asarray(
                [item.model_matches_reference_top_action for item in family_items],
                dtype=np.float64,
            )
            family_matches = np.asarray(
                [item.model_matches_reference_top_action_family for item in family_items],
                dtype=np.float64,
            )
            probabilities = np.asarray(
                [item.model_probability_on_reference_top_action for item in family_items],
                dtype=np.float64,
            )
            family_probabilities = np.asarray(
                [item.model_probability_on_reference_top_action_family for item in family_items],
                dtype=np.float64,
            )
            same_family_margins = [
                item.model_reference_top_action_same_family_logit_margin
                for item in family_items
                if item.model_reference_top_action_same_family_logit_margin is not None
            ]
            summaries.append(
                {
                    "family": family,
                    "count": len(family_items),
                    "model_matches_reference_top_action_rate": float(np.mean(action_matches)),
                    "model_matches_reference_top_action_family_rate": float(np.mean(family_matches)),
                    "model_mean_probability_on_reference_top_action": float(np.mean(probabilities)),
                    "model_mean_probability_on_reference_top_action_family": float(np.mean(family_probabilities)),
                    "model_probability_on_reference_top_action_percentiles": percentile_summary(probabilities.tolist()),
                    "model_reference_top_action_same_family_logit_margin_percentiles": percentile_summary(
                        same_family_margins
                    ),
                }
            )
        return summaries


@dataclass(frozen=True, slots=True)
class _StepSummary:
    model_matches_reference_top_action: bool
    model_matches_reference_top_action_family: bool
    model_probability_on_reference_top_action: float
    model_probability_on_reference_top_action_family: float
    model_reference_top_action_same_family_logit_margin: float | None


__all__ = ["PolicyAlignmentAccumulator"]
