"""Tests for WeightedSumPolicy."""
import pandas as pd
import pytest

from energy_repset.objectives import ObjectiveSet
from energy_repset.score_components import WassersteinFidelity, DiversityReward
from energy_repset.selection_policies import WeightedSumPolicy


@pytest.fixture
def evaluations_df():
    """Synthetic evaluations DataFrame with known best answer."""
    return pd.DataFrame({
        "slices": [("a", "b"), ("c", "d"), ("e", "f")],
        "wasserstein": [0.5, 0.1, 0.9],
        "diversity": [0.3, 0.8, 0.1],
    })


@pytest.fixture
def objective_set():
    return ObjectiveSet({
        "wass": (1.0, WassersteinFidelity()),
        "div": (1.0, DiversityReward()),
    })


class TestSelectBest:

    def test_picks_minimum_weighted_sum(self, evaluations_df, objective_set):
        policy = WeightedSumPolicy()
        best = policy.select_best(evaluations_df, objective_set)
        # wasserstein(min) + diversity(max, negated): oriented for min
        # Row 0: 0.5 - 0.3 = 0.2
        # Row 1: 0.1 - 0.8 = -0.7  <- minimum
        # Row 2: 0.9 - 0.1 = 0.8
        assert best == ("c", "d")

    def test_with_normalization(self, evaluations_df, objective_set):
        policy = WeightedSumPolicy(normalization="robust_minmax")
        best = policy.select_best(evaluations_df, objective_set)
        assert len(best) == 2  # Returns valid tuple

    def test_override_weights(self, evaluations_df, objective_set):
        policy = WeightedSumPolicy(overrides={"wasserstein": 0.0, "diversity": 1.0})
        best = policy.select_best(evaluations_df, objective_set)
        # Only diversity matters (max -> negated); row 1 has highest diversity
        assert best == ("c", "d")
