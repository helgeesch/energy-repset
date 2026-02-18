"""Tests for Snippet search algorithm."""
import pytest

from energy_repset.search_algorithms import SnippetSearch
from energy_repset.feature_engineering import DirectProfileFeatureEngineer
from energy_repset.representation import UniformRepresentationModel
from energy_repset.workflow import Workflow
from energy_repset.problem import RepSetExperiment
from energy_repset.context import ProblemContext
from energy_repset.time_slicer import TimeSlicer


class TestSnippetSearch:

    def test_basic_run(self, context_daily_with_direct_features):
        search = SnippetSearch(k=2, period_length_days=7, step_days=7)
        result = search.find_selection(context_daily_with_direct_features)

        assert result is not None
        assert len(result.selection) == 2
        assert result.selection_space == 'subset'

    def test_weights_sum_to_one(self, context_daily_with_direct_features):
        search = SnippetSearch(k=2, period_length_days=7, step_days=7)
        result = search.find_selection(context_daily_with_direct_features)

        assert result.weights is not None
        assert sum(result.weights.values()) == pytest.approx(1.0)

    def test_weights_keys_match_selection(self, context_daily_with_direct_features):
        search = SnippetSearch(k=2, period_length_days=7, step_days=7)
        result = search.find_selection(context_daily_with_direct_features)

        assert set(result.weights.keys()) == set(result.selection)

    def test_total_distance_score(self, context_daily_with_direct_features):
        search = SnippetSearch(k=2, period_length_days=7, step_days=7)
        result = search.find_selection(context_daily_with_direct_features)

        assert 'total_distance' in result.scores
        assert result.scores['total_distance'] >= 0.0

    def test_representatives_dict(self, context_daily_with_direct_features):
        search = SnippetSearch(k=2, period_length_days=7, step_days=7)
        result = search.find_selection(context_daily_with_direct_features)

        assert len(result.representatives) == 2
        for label, df in result.representatives.items():
            assert label in result.selection
            assert len(df) > 0

    def test_more_k_reduces_distance(self, context_daily_with_direct_features):
        r1 = SnippetSearch(k=2, period_length_days=7, step_days=7).find_selection(
            context_daily_with_direct_features
        )
        r2 = SnippetSearch(k=4, period_length_days=7, step_days=7).find_selection(
            context_daily_with_direct_features
        )
        assert r2.scores['total_distance'] <= r1.scores['total_distance'] + 1e-6

    def test_rejects_non_daily_slicer(self, context_with_features):
        search = SnippetSearch(k=2, period_length_days=7)
        with pytest.raises(ValueError, match="daily slicing"):
            search.find_selection(context_with_features)

    def test_step_days_parameter(self, context_daily_with_direct_features):
        search = SnippetSearch(k=2, period_length_days=7, step_days=1)
        result = search.find_selection(context_daily_with_direct_features)
        assert len(result.selection) == 2

    def test_diagnostics(self, context_daily_with_direct_features):
        search = SnippetSearch(k=2, period_length_days=7, step_days=7)
        result = search.find_selection(context_daily_with_direct_features)

        assert 'assignments' in result.diagnostics
        assert 'candidate_starts' in result.diagnostics


@pytest.mark.integration
class TestSnippetIntegration:

    def test_full_workflow(self, df_raw_hourly, daily_slicer):
        context = ProblemContext(df_raw=df_raw_hourly, slicer=daily_slicer)
        workflow = Workflow(
            feature_engineer=DirectProfileFeatureEngineer(),
            search_algorithm=SnippetSearch(k=3, period_length_days=7, step_days=7),
            representation_model=UniformRepresentationModel(),
        )
        experiment = RepSetExperiment(context, workflow)
        result = experiment.run()

        assert len(result.selection) == 3
        assert result.weights is not None
        assert sum(result.weights.values()) == pytest.approx(1.0)
