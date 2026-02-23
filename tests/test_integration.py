"""End-to-end integration tests."""
import pytest

from energy_repset.context import ProblemContext
from energy_repset.workflow import Workflow
from energy_repset.problem import RepSetExperiment
from energy_repset.feature_engineering import StandardStatsFeatureEngineer
from energy_repset.objectives import ObjectiveSet
from energy_repset.score_components import WassersteinFidelity, CorrelationFidelity
from energy_repset.selection_policies import WeightedSumPolicy, ParetoMaxMinStrategy
from energy_repset.combi_gens import ExhaustiveCombiGen
from energy_repset.search_algorithms import (
    ObjectiveDrivenCombinatorialSearchAlgorithm,
    RandomSamplingSearch,
    GeneticAlgorithmSearch,
)
from energy_repset.representation import UniformRepresentationModel


@pytest.mark.integration
class TestFullWorkflow:

    def test_experiment_run(self, df_raw_hourly, monthly_slicer):
        context = ProblemContext(df_raw=df_raw_hourly, slicer=monthly_slicer)
        feature_eng = StandardStatsFeatureEngineer()
        objective_set = ObjectiveSet({
            "wass": (1.0, WassersteinFidelity()),
            "corr": (0.5, CorrelationFidelity()),
        })
        policy = WeightedSumPolicy()
        combi_gen = ExhaustiveCombiGen(k=2)
        search_algo = ObjectiveDrivenCombinatorialSearchAlgorithm(
            objective_set, policy, combi_gen
        )
        repr_model = UniformRepresentationModel()

        workflow = Workflow(
            feature_engineer=feature_eng,
            search_algorithm=search_algo,
            representation_model=repr_model,
        )
        experiment = RepSetExperiment(context, workflow)
        result = experiment.run()

        assert len(result.selection) == 2
        assert result.weights is not None
        assert sum(result.weights.values()) == pytest.approx(1.0)
        assert "wasserstein" in result.scores
        assert "correlation" in result.scores
        assert len(result.representatives) == 2

    def test_experiment_with_pareto(self, df_raw_hourly, monthly_slicer):
        context = ProblemContext(df_raw=df_raw_hourly, slicer=monthly_slicer)
        feature_eng = StandardStatsFeatureEngineer()
        objective_set = ObjectiveSet({
            "wass": (1.0, WassersteinFidelity()),
            "corr": (1.0, CorrelationFidelity()),
        })
        policy = ParetoMaxMinStrategy()
        combi_gen = ExhaustiveCombiGen(k=2)
        search_algo = ObjectiveDrivenCombinatorialSearchAlgorithm(
            objective_set, policy, combi_gen
        )
        repr_model = UniformRepresentationModel()

        workflow = Workflow(
            feature_engineer=feature_eng,
            search_algorithm=search_algo,
            representation_model=repr_model,
        )
        experiment = RepSetExperiment(context, workflow)
        result = experiment.run()

        assert len(result.selection) == 2
        assert result.weights is not None

    def test_feature_context_accessible(self, df_raw_hourly, monthly_slicer):
        context = ProblemContext(df_raw=df_raw_hourly, slicer=monthly_slicer)
        feature_eng = StandardStatsFeatureEngineer()
        objective_set = ObjectiveSet({
            "wass": (1.0, WassersteinFidelity()),
        })
        search_algo = ObjectiveDrivenCombinatorialSearchAlgorithm(
            objective_set, WeightedSumPolicy(), ExhaustiveCombiGen(k=2)
        )
        workflow = Workflow(
            feature_engineer=feature_eng,
            search_algorithm=search_algo,
            representation_model=UniformRepresentationModel(),
        )
        experiment = RepSetExperiment(context, workflow)
        experiment.run()

        fctx = experiment.feature_context
        assert fctx.df_features is not None
        assert fctx.df_features.shape[0] == 3

    def test_random_sampling_workflow(self, df_raw_hourly, monthly_slicer):
        context = ProblemContext(df_raw=df_raw_hourly, slicer=monthly_slicer)
        feature_eng = StandardStatsFeatureEngineer()
        objective_set = ObjectiveSet({
            "wass": (1.0, WassersteinFidelity()),
            "corr": (0.5, CorrelationFidelity()),
        })
        search_algo = RandomSamplingSearch(
            objective_set=objective_set,
            selection_policy=WeightedSumPolicy(),
            combination_generator=ExhaustiveCombiGen(k=2),
            n_samples=10,
            seed=42,
        )
        workflow = Workflow(
            feature_engineer=feature_eng,
            search_algorithm=search_algo,
            representation_model=UniformRepresentationModel(),
        )
        experiment = RepSetExperiment(context, workflow)
        result = experiment.run()

        assert len(result.selection) == 2
        assert result.weights is not None
        assert sum(result.weights.values()) == pytest.approx(1.0)
        assert "wasserstein" in result.scores
        assert len(result.representatives) == 2

    def test_genetic_algorithm_workflow(self, df_raw_hourly, monthly_slicer):
        context = ProblemContext(df_raw=df_raw_hourly, slicer=monthly_slicer)
        feature_eng = StandardStatsFeatureEngineer()
        objective_set = ObjectiveSet({
            "wass": (1.0, WassersteinFidelity()),
            "corr": (0.5, CorrelationFidelity()),
        })
        search_algo = GeneticAlgorithmSearch(
            objective_set=objective_set,
            selection_policy=WeightedSumPolicy(),
            combination_generator=ExhaustiveCombiGen(k=2),
            population_size=4,
            n_generations=3,
            seed=42,
        )
        workflow = Workflow(
            feature_engineer=feature_eng,
            search_algorithm=search_algo,
            representation_model=UniformRepresentationModel(),
        )
        experiment = RepSetExperiment(context, workflow)
        result = experiment.run()

        assert len(result.selection) == 2
        assert result.weights is not None
        assert sum(result.weights.values()) == pytest.approx(1.0)
        assert "wasserstein" in result.scores
        assert len(result.representatives) == 2
