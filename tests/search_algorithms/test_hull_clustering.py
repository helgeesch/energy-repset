"""Tests for Hull Clustering search algorithm."""
import pytest

from energy_repset.search_algorithms import HullClusteringSearch
from energy_repset.representation import BlendedRepresentationModel
from energy_repset.feature_engineering import StandardStatsFeatureEngineer
from energy_repset.workflow import Workflow
from energy_repset.problem import RepSetExperiment
from energy_repset.context import ProblemContext


class TestHullClusteringSearch:

    def test_basic_run_convex(self, context_with_features):
        search = HullClusteringSearch(k=2, hull_type='convex')
        result = search.find_selection(context_with_features)

        assert result is not None
        assert len(result.selection) == 2
        assert result.weights is None
        assert 'projection_error' in result.scores
        assert result.scores['projection_error'] >= 0.0
        assert result.selection_space == 'subset'

    def test_basic_run_conic(self, context_with_features):
        search = HullClusteringSearch(k=2, hull_type='conic')
        result = search.find_selection(context_with_features)

        assert len(result.selection) == 2
        assert result.weights is None
        assert result.scores['projection_error'] >= 0.0

    def test_representatives_dict(self, context_with_features):
        search = HullClusteringSearch(k=2)
        result = search.find_selection(context_with_features)

        assert len(result.representatives) == 2
        for label, df in result.representatives.items():
            assert label in result.selection
            assert len(df) > 0

    def test_selection_unique(self, context_with_features):
        search = HullClusteringSearch(k=2)
        result = search.find_selection(context_with_features)
        assert len(set(result.selection)) == len(result.selection)

    def test_convex_error_leq_conic(self, context_with_features):
        convex = HullClusteringSearch(k=2, hull_type='convex')
        conic = HullClusteringSearch(k=2, hull_type='conic')
        r_convex = convex.find_selection(context_with_features)
        r_conic = conic.find_selection(context_with_features)

        # Conic is less constrained, so error should be <= convex
        assert r_conic.scores['projection_error'] <= r_convex.scores['projection_error'] + 1e-6

    def test_more_k_reduces_error(self, context_with_features):
        r1 = HullClusteringSearch(k=1).find_selection(context_with_features)
        r2 = HullClusteringSearch(k=2).find_selection(context_with_features)
        assert r2.scores['projection_error'] <= r1.scores['projection_error'] + 1e-6


@pytest.mark.integration
class TestHullClusteringIntegration:

    def test_full_workflow_with_blended(self, df_raw_hourly, monthly_slicer):
        context = ProblemContext(df_raw=df_raw_hourly, slicer=monthly_slicer)
        workflow = Workflow(
            feature_engineer=StandardStatsFeatureEngineer(),
            search_algorithm=HullClusteringSearch(k=2, hull_type='convex'),
            representation_model=BlendedRepresentationModel(blend_type='convex'),
        )
        experiment = RepSetExperiment(context, workflow)
        result = experiment.run()

        assert len(result.selection) == 2
        assert result.weights is not None
        assert result.weights.shape[0] == 3  # 3 months
        assert result.weights.shape[1] == 2  # 2 representatives
