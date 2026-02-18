"""Tests for PCAFeatureEngineer."""
import numpy as np
import pytest

from energy_repset.feature_engineering import PCAFeatureEngineer


class TestCalcAndGetFeaturesDf:

    def test_output_shape(self, context_with_features):
        eng = PCAFeatureEngineer(n_components=2)
        df = eng.calc_and_get_features_df(context_with_features)
        assert df.shape[0] == context_with_features.df_features.shape[0]
        assert df.shape[1] == 2

    def test_column_names(self, context_with_features):
        eng = PCAFeatureEngineer(n_components=2)
        df = eng.calc_and_get_features_df(context_with_features)
        assert list(df.columns) == ["pc_0", "pc_1"]

    def test_raises_without_features(self, context_monthly):
        eng = PCAFeatureEngineer(n_components=2)
        with pytest.raises(ValueError, match="df_features"):
            eng.calc_and_get_features_df(context_monthly)


class TestExplainedVariance:

    def test_variance_ratios_sum_to_one_or_less(self, context_with_features):
        eng = PCAFeatureEngineer(n_components=2)
        eng.calc_and_get_features_df(context_with_features)
        ratios = eng.explained_variance_ratio_
        assert ratios is not None
        assert 0 < ratios.sum() <= 1.0 + 1e-10

    def test_none_before_fit(self):
        eng = PCAFeatureEngineer()
        assert eng.explained_variance_ratio_ is None


class TestComponents:

    def test_shape_after_fit(self, context_with_features):
        eng = PCAFeatureEngineer(n_components=2)
        eng.calc_and_get_features_df(context_with_features)
        assert eng.components_.shape[0] == 2

    def test_none_before_fit(self):
        eng = PCAFeatureEngineer()
        assert eng.components_ is None
