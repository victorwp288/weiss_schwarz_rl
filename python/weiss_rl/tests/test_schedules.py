from __future__ import annotations

import pytest

from weiss_rl.schedules import linear_anneal_value


def test_linear_anneal_value_respects_delayed_window() -> None:
    assert linear_anneal_value(
        initial_value=1.0,
        final_value=0.25,
        start_update=4,
        end_update=8,
        update_count=0,
    ) == pytest.approx(1.0)
    assert linear_anneal_value(
        initial_value=1.0,
        final_value=0.25,
        start_update=4,
        end_update=8,
        update_count=6,
    ) == pytest.approx(0.625)
    assert linear_anneal_value(
        initial_value=1.0,
        final_value=0.25,
        start_update=4,
        end_update=8,
        update_count=8,
    ) == pytest.approx(0.25)


def test_linear_anneal_value_keeps_initial_when_end_is_disabled() -> None:
    assert linear_anneal_value(
        initial_value=0.1,
        final_value=0.0,
        start_update=40,
        end_update=-1,
        update_count=200,
    ) == pytest.approx(0.1)
