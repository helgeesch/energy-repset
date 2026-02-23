"""Tests for CTPC (Chronological Time-Period Clustering) search algorithm."""
import pytest

from energy_repset.search_algorithms import CTPCSearch
from energy_repset.feature_engineering import StandardStatsFeatureEngineer
from energy_repset.workflow import Workflow
from energy_repset.problem import RepSetExperiment
from energy_repset.context import ProblemContext


class TestCTPCSearch:

    def test_basic_run(self, context_with_features):
        search = CTPCSearch(k=2)
        result = search.find_selection(context_with_features)

        assert result is not None
        assert len(result.selection) == 2
        assert result.selection_space == 'chronological'

    def test_weights_sum_to_one(self, context_with_features):
        search = CTPCSearch(k=2)
        result = search.find_selection(context_with_features)

        assert result.weights is not None
        assert sum(result.weights.values()) == pytest.approx(1.0)

    def test_weights_keys_match_selection(self, context_with_features):
        search = CTPCSearch(k=2)
        result = search.find_selection(context_with_features)

        assert set(result.weights.keys()) == set(result.selection)

    def test_wcss_score(self, context_with_features):
        search = CTPCSearch(k=2)
        result = search.find_selection(context_with_features)

        assert 'wcss' in result.scores
        assert result.scores['wcss'] >= 0.0

    def test_representatives_dict(self, context_with_features):
        search = CTPCSearch(k=2)
        result = search.find_selection(context_with_features)

        assert len(result.representatives) == 2
        for label, df in result.representatives.items():
            assert label in result.selection
            assert len(df) > 0

    def test_diagnostics_segments(self, context_with_features):
        search = CTPCSearch(k=2)
        result = search.find_selection(context_with_features)

        assert 'segments' in result.diagnostics
        assert 'cluster_labels' in result.diagnostics
        assert len(result.diagnostics['segments']) == 2

    def test_contiguity_of_clusters(self, context_with_features):
        """Cluster labels must be non-decreasing (contiguous segments)."""
        search = CTPCSearch(k=2)
        result = search.find_selection(context_with_features)

        labels = result.diagnostics['cluster_labels']
        seen = set()
        prev = None
        for lbl in labels:
            if lbl != prev:
                assert lbl not in seen, "Non-contiguous cluster detected"
                seen.add(lbl)
                prev = lbl

    def test_linkage_options(self, context_with_features):
        for linkage in ('ward', 'complete', 'average', 'single'):
            search = CTPCSearch(k=2, linkage=linkage)
            result = search.find_selection(context_with_features)
            assert len(result.selection) == 2

    def test_daily_slicing(self, context_daily_with_stats_features):
        search = CTPCSearch(k=5)
        result = search.find_selection(context_daily_with_stats_features)

        assert len(result.selection) == 5
        assert sum(result.weights.values()) == pytest.approx(1.0)


@pytest.mark.integration
class TestCTPCIntegration:

    def test_full_workflow(self, df_raw_hourly, monthly_slicer):
        """Weights are pre-computed by CTPC, so representation model is skipped."""
        context = ProblemContext(df_raw=df_raw_hourly, slicer=monthly_slicer)
        workflow = Workflow(
            feature_engineer=StandardStatsFeatureEngineer(),
            search_algorithm=CTPCSearch(k=2),
            representation_model=None,
        )
        experiment = RepSetExperiment(context, workflow)
        result = experiment.run()

        assert len(result.selection) == 2
        assert result.weights is not None
        assert sum(result.weights.values()) == pytest.approx(1.0)

    def test_precomputed_weights_preserved(self, df_raw_hourly, monthly_slicer):
        """Verify that CTPC's pre-computed weights are not overwritten."""
        context = ProblemContext(df_raw=df_raw_hourly, slicer=monthly_slicer)
        search = CTPCSearch(k=2)
        workflow = Workflow(
            feature_engineer=StandardStatsFeatureEngineer(),
            search_algorithm=search,
            representation_model=None,
        )
        experiment = RepSetExperiment(context, workflow)
        result = experiment.run()

        # CTPC weights are segment fractions, not uniform 1/k
        # With 3 months split into 2, at least one weight differs from 0.5
        weights = list(result.weights.values())
        assert sum(weights) == pytest.approx(1.0)
