"""Tests for ObjectiveDrivenCombinatorialSearchAlgorithm."""
import pytest

from energy_repset.objectives import ObjectiveSet
from energy_repset.score_components import WassersteinFidelity, DiversityReward
from energy_repset.selection_policies import WeightedSumPolicy
from energy_repset.combi_gens import ExhaustiveCombiGen
from energy_repset.search_algorithms import ObjectiveDrivenCombinatorialSearchAlgorithm
from energy_repset.results import RepSetResult


class TestFindSelection:

    def test_returns_repset_result(self, context_with_features):
        algo = ObjectiveDrivenCombinatorialSearchAlgorithm(
            objective_set=ObjectiveSet({
                "wass": (1.0, WassersteinFidelity()),
            }),
            selection_policy=WeightedSumPolicy(),
            combination_generator=ExhaustiveCombiGen(k=2),
        )
        result = algo.find_selection(context_with_features)
        assert isinstance(result, RepSetResult)

    def test_result_has_selection(self, context_with_features):
        algo = ObjectiveDrivenCombinatorialSearchAlgorithm(
            objective_set=ObjectiveSet({
                "wass": (1.0, WassersteinFidelity()),
            }),
            selection_policy=WeightedSumPolicy(),
            combination_generator=ExhaustiveCombiGen(k=2),
        )
        result = algo.find_selection(context_with_features)
        assert len(result.selection) == 2

    def test_result_has_scores(self, context_with_features):
        algo = ObjectiveDrivenCombinatorialSearchAlgorithm(
            objective_set=ObjectiveSet({
                "wass": (1.0, WassersteinFidelity()),
                "div": (0.5, DiversityReward()),
            }),
            selection_policy=WeightedSumPolicy(),
            combination_generator=ExhaustiveCombiGen(k=2),
        )
        result = algo.find_selection(context_with_features)
        assert "wasserstein" in result.scores
        assert "diversity" in result.scores

    def test_result_has_representatives(self, context_with_features):
        algo = ObjectiveDrivenCombinatorialSearchAlgorithm(
            objective_set=ObjectiveSet({
                "wass": (1.0, WassersteinFidelity()),
            }),
            selection_policy=WeightedSumPolicy(),
            combination_generator=ExhaustiveCombiGen(k=2),
        )
        result = algo.find_selection(context_with_features)
        assert len(result.representatives) == 2
        for s in result.selection:
            assert s in result.representatives

    def test_evaluations_df_in_diagnostics(self, context_with_features):
        algo = ObjectiveDrivenCombinatorialSearchAlgorithm(
            objective_set=ObjectiveSet({
                "wass": (1.0, WassersteinFidelity()),
            }),
            selection_policy=WeightedSumPolicy(),
            combination_generator=ExhaustiveCombiGen(k=2),
        )
        result = algo.find_selection(context_with_features)
        evals = result.diagnostics["evaluations_df"]
        assert len(evals) == 3  # C(3, 2) = 3


class TestGetAllScores:

    def test_raises_before_find(self):
        algo = ObjectiveDrivenCombinatorialSearchAlgorithm(
            objective_set=ObjectiveSet({
                "wass": (1.0, WassersteinFidelity()),
            }),
            selection_policy=WeightedSumPolicy(),
            combination_generator=ExhaustiveCombiGen(k=2),
        )
        with pytest.raises(ValueError, match="No scores"):
            algo.get_all_scores()

    def test_returns_dataframe_after_find(self, context_with_features):
        algo = ObjectiveDrivenCombinatorialSearchAlgorithm(
            objective_set=ObjectiveSet({
                "wass": (1.0, WassersteinFidelity()),
            }),
            selection_policy=WeightedSumPolicy(),
            combination_generator=ExhaustiveCombiGen(k=2),
        )
        algo.find_selection(context_with_features)
        df = algo.get_all_scores()
        assert len(df) == 3
        assert "slices" in df.columns
