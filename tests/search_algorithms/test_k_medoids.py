"""Tests for KMedoidsSearch (k-medoids) search algorithm."""
import pytest

from energy_repset.search_algorithms import KMedoidsSearch
from energy_repset.feature_engineering import StandardStatsFeatureEngineer
from energy_repset.workflow import Workflow
from energy_repset.problem import RepSetExperiment
from energy_repset.context import ProblemContext


class TestKMedoidsSearch:

    def test_basic_run(self, context_with_features):
        search = KMedoidsSearch(k=2, random_state=42)
        result = search.find_selection(context_with_features)

        assert result is not None
        assert len(result.selection) == 2
        assert result.selection_space == 'subset'

    def test_basic_run_daily(self, context_daily_with_stats_features):
        search = KMedoidsSearch(k=5, random_state=42)
        result = search.find_selection(context_daily_with_stats_features)

        assert len(result.selection) == 5
        assert sum(result.weights.values()) == pytest.approx(1.0)

    def test_selection_is_subset_of_slices(self, context_with_features):
        search = KMedoidsSearch(k=2, random_state=42)
        result = search.find_selection(context_with_features)

        valid_labels = set(context_with_features.df_features.index)
        for label in result.selection:
            assert label in valid_labels

    def test_weights_sum_to_one(self, context_with_features):
        search = KMedoidsSearch(k=2, random_state=42)
        result = search.find_selection(context_with_features)

        assert result.weights is not None
        assert sum(result.weights.values()) == pytest.approx(1.0)

    def test_weights_keys_match_selection(self, context_with_features):
        search = KMedoidsSearch(k=2, random_state=42)
        result = search.find_selection(context_with_features)

        assert set(result.weights.keys()) == set(result.selection)

    def test_wcss_score_nonnegative(self, context_with_features):
        search = KMedoidsSearch(k=2, random_state=42)
        result = search.find_selection(context_with_features)

        assert 'wcss' in result.scores
        assert result.scores['wcss'] >= 0.0

    def test_representatives_dict(self, context_with_features):
        search = KMedoidsSearch(k=2, random_state=42)
        result = search.find_selection(context_with_features)

        assert len(result.representatives) == 2
        for label, df in result.representatives.items():
            assert label in result.selection
            assert len(df) > 0

    def test_diagnostics_contain_cluster_info(self, context_with_features):
        search = KMedoidsSearch(k=2, random_state=42)
        result = search.find_selection(context_with_features)

        assert 'cluster_labels' in result.diagnostics
        assert 'cluster_info' in result.diagnostics
        assert 'inertia' in result.diagnostics
        assert 'n_iter' in result.diagnostics
        assert len(result.diagnostics['cluster_info']) == 2

    def test_selection_space_is_subset(self, context_with_features):
        search = KMedoidsSearch(k=2, random_state=42)
        result = search.find_selection(context_with_features)

        assert result.selection_space == 'subset'

    def test_deterministic_with_random_state(self, context_with_features):
        search_a = KMedoidsSearch(k=2, random_state=99)
        search_b = KMedoidsSearch(k=2, random_state=99)

        result_a = search_a.find_selection(context_with_features)
        result_b = search_b.find_selection(context_with_features)

        assert set(result_a.selection) == set(result_b.selection)
        assert result_a.scores['wcss'] == pytest.approx(result_b.scores['wcss'])

    def test_k_equals_n_slices(self, context_with_features):
        """Edge case: every slice is its own cluster."""
        n_slices = len(context_with_features.df_features)
        search = KMedoidsSearch(k=n_slices, random_state=42)
        result = search.find_selection(context_with_features)

        assert len(result.selection) == n_slices
        assert set(result.selection) == set(context_with_features.df_features.index)
        assert result.scores['wcss'] == pytest.approx(0.0)
        for w in result.weights.values():
            assert w == pytest.approx(1.0 / n_slices)

    def test_pam_method(self, context_with_features):
        search = KMedoidsSearch(k=2, method='pam', random_state=42)
        result = search.find_selection(context_with_features)

        assert len(result.selection) == 2
        assert sum(result.weights.values()) == pytest.approx(1.0)


@pytest.mark.integration
class TestKMedoidsSearchIntegration:

    def test_full_workflow(self, df_raw_hourly, monthly_slicer):
        """End-to-end via RepSetExperiment with feature engineering."""
        context = ProblemContext(df_raw=df_raw_hourly, slicer=monthly_slicer)
        workflow = Workflow(
            feature_engineer=StandardStatsFeatureEngineer(),
            search_algorithm=KMedoidsSearch(k=2, random_state=42),
            representation_model=None,
        )
        experiment = RepSetExperiment(context, workflow)
        result = experiment.run()

        assert len(result.selection) == 2
        assert result.weights is not None
        assert sum(result.weights.values()) == pytest.approx(1.0)

    def test_precomputed_weights_preserved(self, df_raw_hourly, monthly_slicer):
        """KMedoidsSearch weights are not overwritten by RepresentationModel."""
        context = ProblemContext(df_raw=df_raw_hourly, slicer=monthly_slicer)
        search = KMedoidsSearch(k=2, random_state=42)

        # First run standalone to capture the pre-computed weights
        engineer = StandardStatsFeatureEngineer()
        feature_context = engineer.run(context)
        standalone_result = search.find_selection(feature_context)
        standalone_weights = dict(standalone_result.weights)

        # Now run through experiment workflow
        workflow = Workflow(
            feature_engineer=StandardStatsFeatureEngineer(),
            search_algorithm=KMedoidsSearch(k=2, random_state=42),
            representation_model=None,
        )
        experiment = RepSetExperiment(context, workflow)
        result = experiment.run()

        # Weights should match the standalone run, not uniform 1/k
        for label in result.selection:
            assert result.weights[label] == pytest.approx(standalone_weights[label])