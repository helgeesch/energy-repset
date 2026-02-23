"""Tests for GeneticAlgorithmSearch and fitness strategies."""
import numpy as np
import pandas as pd
import pytest

from energy_repset.objectives import ObjectiveSet
from energy_repset.score_components import WassersteinFidelity, DiversityReward
from energy_repset.selection_policies import WeightedSumPolicy, ParetoMaxMinStrategy
from energy_repset.combi_gens import ExhaustiveCombiGen
from energy_repset.search_algorithms import (
    GeneticAlgorithmSearch,
    WeightedSumFitness,
    NSGA2Fitness,
)
from energy_repset.results import RepSetResult


class TestGeneticAlgorithmSearch:

    def test_returns_repset_result(self, context_with_features):
        algo = GeneticAlgorithmSearch(
            objective_set=ObjectiveSet({"wass": (1.0, WassersteinFidelity())}),
            selection_policy=WeightedSumPolicy(),
            combination_generator=ExhaustiveCombiGen(k=2),
            population_size=4,
            n_generations=3,
            seed=42,
        )
        result = algo.find_selection(context_with_features)
        assert isinstance(result, RepSetResult)

    def test_result_has_correct_selection_length(self, context_with_features):
        algo = GeneticAlgorithmSearch(
            objective_set=ObjectiveSet({"wass": (1.0, WassersteinFidelity())}),
            selection_policy=WeightedSumPolicy(),
            combination_generator=ExhaustiveCombiGen(k=2),
            population_size=4,
            n_generations=3,
            seed=42,
        )
        result = algo.find_selection(context_with_features)
        assert len(result.selection) == 2

    def test_result_has_scores(self, context_with_features):
        algo = GeneticAlgorithmSearch(
            objective_set=ObjectiveSet({
                "wass": (1.0, WassersteinFidelity()),
                "div": (0.5, DiversityReward()),
            }),
            selection_policy=WeightedSumPolicy(),
            combination_generator=ExhaustiveCombiGen(k=2),
            population_size=4,
            n_generations=3,
            seed=42,
        )
        result = algo.find_selection(context_with_features)
        assert "wasserstein" in result.scores
        assert "diversity" in result.scores

    def test_result_has_representatives(self, context_with_features):
        algo = GeneticAlgorithmSearch(
            objective_set=ObjectiveSet({"wass": (1.0, WassersteinFidelity())}),
            selection_policy=WeightedSumPolicy(),
            combination_generator=ExhaustiveCombiGen(k=2),
            population_size=4,
            n_generations=3,
            seed=42,
        )
        result = algo.find_selection(context_with_features)
        assert len(result.representatives) == 2
        for s in result.selection:
            assert s in result.representatives

    def test_evaluations_df_in_diagnostics(self, context_with_features):
        algo = GeneticAlgorithmSearch(
            objective_set=ObjectiveSet({"wass": (1.0, WassersteinFidelity())}),
            selection_policy=WeightedSumPolicy(),
            combination_generator=ExhaustiveCombiGen(k=2),
            population_size=4,
            n_generations=3,
            seed=42,
        )
        result = algo.find_selection(context_with_features)
        evals = result.diagnostics["evaluations_df"]
        assert len(evals) == 4  # population_size
        assert "slices" in evals.columns

    def test_generation_history_in_diagnostics(self, context_with_features):
        n_gen = 5
        algo = GeneticAlgorithmSearch(
            objective_set=ObjectiveSet({"wass": (1.0, WassersteinFidelity())}),
            selection_policy=WeightedSumPolicy(),
            combination_generator=ExhaustiveCombiGen(k=2),
            population_size=4,
            n_generations=n_gen,
            seed=42,
        )
        result = algo.find_selection(context_with_features)
        history = result.diagnostics["generation_history"]
        assert isinstance(history, pd.DataFrame)
        assert len(history) == n_gen
        assert "best_fitness" in history.columns
        assert "mean_fitness" in history.columns

    def test_reproducibility(self, context_with_features):
        kwargs = dict(
            objective_set=ObjectiveSet({"wass": (1.0, WassersteinFidelity())}),
            selection_policy=WeightedSumPolicy(),
            combination_generator=ExhaustiveCombiGen(k=2),
            population_size=4,
            n_generations=3,
            seed=99,
        )
        result1 = GeneticAlgorithmSearch(**kwargs).find_selection(context_with_features)
        result2 = GeneticAlgorithmSearch(**kwargs).find_selection(context_with_features)
        assert result1.selection == result2.selection
        assert result1.scores == result2.scores

    def test_works_with_nsga2_fitness(self, context_with_features):
        algo = GeneticAlgorithmSearch(
            objective_set=ObjectiveSet({
                "wass": (1.0, WassersteinFidelity()),
                "div": (0.5, DiversityReward()),
            }),
            selection_policy=WeightedSumPolicy(),
            combination_generator=ExhaustiveCombiGen(k=2),
            fitness_strategy=NSGA2Fitness(),
            population_size=4,
            n_generations=3,
            seed=42,
        )
        result = algo.find_selection(context_with_features)
        assert isinstance(result, RepSetResult)
        assert len(result.selection) == 2

    def test_works_with_pareto_policy(self, context_with_features):
        algo = GeneticAlgorithmSearch(
            objective_set=ObjectiveSet({
                "wass": (1.0, WassersteinFidelity()),
                "div": (1.0, DiversityReward()),
            }),
            selection_policy=ParetoMaxMinStrategy(),
            combination_generator=ExhaustiveCombiGen(k=2),
            population_size=4,
            n_generations=3,
            seed=42,
        )
        result = algo.find_selection(context_with_features)
        assert isinstance(result, RepSetResult)


class TestWeightedSumFitness:

    def test_rank_returns_correct_shape(self, context_with_features):
        objective_set = ObjectiveSet({"wass": (1.0, WassersteinFidelity())})
        objective_set.prepare(context_with_features)

        slices = context_with_features.get_unique_slices()
        combis = [
            tuple(slices[:2]),
            tuple(slices[1:]),
            tuple([slices[0], slices[2]]),
        ]
        rows = []
        for c in combis:
            rec = {"slices": c}
            rec.update(objective_set.evaluate(c, context_with_features))
            rows.append(rec)
        df = pd.DataFrame(rows)

        fitness = WeightedSumFitness()
        result = fitness.rank(df, objective_set)
        assert result.shape == (3,)

    def test_higher_fitness_for_better_scores(self, context_with_features):
        objective_set = ObjectiveSet({"wass": (1.0, WassersteinFidelity())})
        objective_set.prepare(context_with_features)

        slices = context_with_features.get_unique_slices()
        combis = [tuple(slices[:2]), tuple(slices[1:]), tuple([slices[0], slices[2]])]
        rows = []
        for c in combis:
            rec = {"slices": c}
            rec.update(objective_set.evaluate(c, context_with_features))
            rows.append(rec)
        df = pd.DataFrame(rows)

        fitness = WeightedSumFitness()
        result = fitness.rank(df, objective_set)
        # Wasserstein is "min" direction, so lower raw score -> higher fitness
        best_idx = np.argmax(result)
        assert df.iloc[best_idx]["wasserstein"] == df["wasserstein"].min()


class TestNSGA2Fitness:

    def test_rank_returns_correct_shape(self, context_with_features):
        objective_set = ObjectiveSet({
            "wass": (1.0, WassersteinFidelity()),
            "div": (0.5, DiversityReward()),
        })
        objective_set.prepare(context_with_features)

        slices = context_with_features.get_unique_slices()
        combis = [tuple(slices[:2]), tuple(slices[1:]), tuple([slices[0], slices[2]])]
        rows = []
        for c in combis:
            rec = {"slices": c}
            rec.update(objective_set.evaluate(c, context_with_features))
            rows.append(rec)
        df = pd.DataFrame(rows)

        fitness = NSGA2Fitness()
        result = fitness.rank(df, objective_set)
        assert result.shape == (3,)
