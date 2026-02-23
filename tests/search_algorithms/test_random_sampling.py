"""Tests for RandomSamplingSearch."""
import pytest

from energy_repset.objectives import ObjectiveSet
from energy_repset.score_components import WassersteinFidelity, DiversityReward
from energy_repset.selection_policies import WeightedSumPolicy
from energy_repset.combi_gens import ExhaustiveCombiGen
from energy_repset.search_algorithms import RandomSamplingSearch
from energy_repset.results import RepSetResult


class TestFindSelection:

    def test_returns_repset_result(self, context_with_features):
        algo = RandomSamplingSearch(
            objective_set=ObjectiveSet({"wass": (1.0, WassersteinFidelity())}),
            selection_policy=WeightedSumPolicy(),
            combination_generator=ExhaustiveCombiGen(k=2),
            n_samples=5,
            seed=42,
        )
        result = algo.find_selection(context_with_features)
        assert isinstance(result, RepSetResult)

    def test_result_has_correct_selection_length(self, context_with_features):
        algo = RandomSamplingSearch(
            objective_set=ObjectiveSet({"wass": (1.0, WassersteinFidelity())}),
            selection_policy=WeightedSumPolicy(),
            combination_generator=ExhaustiveCombiGen(k=2),
            n_samples=5,
            seed=42,
        )
        result = algo.find_selection(context_with_features)
        assert len(result.selection) == 2

    def test_result_has_scores(self, context_with_features):
        algo = RandomSamplingSearch(
            objective_set=ObjectiveSet({
                "wass": (1.0, WassersteinFidelity()),
                "div": (0.5, DiversityReward()),
            }),
            selection_policy=WeightedSumPolicy(),
            combination_generator=ExhaustiveCombiGen(k=2),
            n_samples=5,
            seed=42,
        )
        result = algo.find_selection(context_with_features)
        assert "wasserstein" in result.scores
        assert "diversity" in result.scores

    def test_result_has_representatives(self, context_with_features):
        algo = RandomSamplingSearch(
            objective_set=ObjectiveSet({"wass": (1.0, WassersteinFidelity())}),
            selection_policy=WeightedSumPolicy(),
            combination_generator=ExhaustiveCombiGen(k=2),
            n_samples=5,
            seed=42,
        )
        result = algo.find_selection(context_with_features)
        assert len(result.representatives) == 2
        for s in result.selection:
            assert s in result.representatives

    def test_evaluations_df_in_diagnostics(self, context_with_features):
        algo = RandomSamplingSearch(
            objective_set=ObjectiveSet({"wass": (1.0, WassersteinFidelity())}),
            selection_policy=WeightedSumPolicy(),
            combination_generator=ExhaustiveCombiGen(k=2),
            n_samples=5,
            seed=42,
        )
        result = algo.find_selection(context_with_features)
        evals = result.diagnostics["evaluations_df"]
        assert len(evals) > 0
        assert "slices" in evals.columns
        assert "wasserstein" in evals.columns

    def test_caps_at_total_valid_combinations(self, context_with_features):
        """With 3 slices choose 2, only 3 valid combos exist."""
        algo = RandomSamplingSearch(
            objective_set=ObjectiveSet({"wass": (1.0, WassersteinFidelity())}),
            selection_policy=WeightedSumPolicy(),
            combination_generator=ExhaustiveCombiGen(k=2),
            n_samples=100,
            seed=42,
        )
        result = algo.find_selection(context_with_features)
        evals = result.diagnostics["evaluations_df"]
        assert len(evals) == 3  # C(3, 2) = 3


class TestReproducibility:

    def test_same_seed_same_result(self, context_with_features):
        kwargs = dict(
            objective_set=ObjectiveSet({"wass": (1.0, WassersteinFidelity())}),
            selection_policy=WeightedSumPolicy(),
            combination_generator=ExhaustiveCombiGen(k=2),
            n_samples=5,
            seed=123,
        )
        result1 = RandomSamplingSearch(**kwargs).find_selection(context_with_features)
        result2 = RandomSamplingSearch(**kwargs).find_selection(context_with_features)
        assert result1.selection == result2.selection
        assert result1.scores == result2.scores
