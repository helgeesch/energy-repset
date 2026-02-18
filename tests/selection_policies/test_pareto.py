"""Tests for ParetoUtopiaPolicy and ParetoMaxMinStrategy."""
import pandas as pd
import pytest

from energy_repset.objectives import ObjectiveSet
from energy_repset.score_components import WassersteinFidelity, CorrelationFidelity
from energy_repset.selection_policies import ParetoUtopiaPolicy, ParetoMaxMinStrategy


@pytest.fixture
def evaluations_df():
    """Evaluations with a clear Pareto front."""
    return pd.DataFrame({
        "slices": [("a", "b"), ("c", "d"), ("e", "f"), ("g", "h")],
        "wasserstein": [0.1, 0.5, 0.9, 0.3],
        "correlation": [0.9, 0.1, 0.5, 0.3],
    })


@pytest.fixture
def objective_set():
    return ObjectiveSet({
        "wass": (1.0, WassersteinFidelity()),
        "corr": (1.0, CorrelationFidelity()),
    })


class TestParetoUtopiaPolicy:

    def test_selects_from_pareto_front(self, evaluations_df, objective_set):
        policy = ParetoUtopiaPolicy()
        best = policy.select_best(evaluations_df, objective_set)
        assert best in [tuple(row) for row in evaluations_df["slices"]]

    def test_dominated_not_selected(self, evaluations_df, objective_set):
        policy = ParetoUtopiaPolicy()
        best = policy.select_best(evaluations_df, objective_set)
        # Row 2 (0.9, 0.5) is dominated by row 3 (0.3, 0.3), so should not be selected
        assert best != ("e", "f")


class TestParetoMaxMinStrategy:

    def test_selects_balanced_solution(self, evaluations_df, objective_set):
        policy = ParetoMaxMinStrategy()
        best = policy.select_best(evaluations_df, objective_set)
        assert best in [tuple(row) for row in evaluations_df["slices"]]

    def test_prefers_balanced_over_extreme(self, evaluations_df, objective_set):
        policy = ParetoMaxMinStrategy()
        best = policy.select_best(evaluations_df, objective_set)
        # (g, h) with (0.3, 0.3) is most balanced on the Pareto front
        assert best == ("g", "h")
