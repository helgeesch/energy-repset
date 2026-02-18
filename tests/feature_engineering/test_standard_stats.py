"""Tests for StandardStatsFeatureEngineer."""
import numpy as np
import pytest

from energy_repset.feature_engineering import StandardStatsFeatureEngineer


class TestCalcAndGetFeaturesDf:

    def test_output_shape(self, context_monthly, monthly_slices):
        eng = StandardStatsFeatureEngineer()
        df = eng.calc_and_get_features_df(context_monthly)
        assert df.shape[0] == len(monthly_slices)
        assert df.shape[1] > 0

    def test_no_nan_or_inf(self, context_monthly):
        eng = StandardStatsFeatureEngineer()
        df = eng.calc_and_get_features_df(context_monthly)
        assert not df.isnull().any().any()
        assert not np.isinf(df.values).any()

    def test_zscore_zero_mean_unit_std(self, context_monthly):
        eng = StandardStatsFeatureEngineer(scale="zscore")
        df = eng.calc_and_get_features_df(context_monthly)
        means = df.mean()
        assert all(abs(m) < 1e-10 for m in means), "Z-scored features should have ~0 mean"

    def test_correlation_features_present(self, context_monthly):
        eng = StandardStatsFeatureEngineer(include_correlations=True)
        df = eng.calc_and_get_features_df(context_monthly)
        corr_cols = [c for c in df.columns if c.startswith("corr__")]
        assert len(corr_cols) >= 1

    def test_no_correlations_when_disabled(self, context_monthly):
        eng = StandardStatsFeatureEngineer(include_correlations=False)
        df = eng.calc_and_get_features_df(context_monthly)
        corr_cols = [c for c in df.columns if c.startswith("corr__")]
        assert len(corr_cols) == 0


class TestScaleNone:

    def test_none_returns_raw_features(self, context_monthly):
        eng = StandardStatsFeatureEngineer(scale="none")
        df = eng.calc_and_get_features_df(context_monthly)
        assert df.shape[0] > 0
        # Raw features should NOT have zero mean (unlike zscore)
        means = df.mean()
        assert not all(abs(m) < 1e-10 for m in means)


class TestRun:

    def test_returns_context_with_features(self, context_monthly):
        eng = StandardStatsFeatureEngineer()
        ctx = eng.run(context_monthly)
        assert ctx.df_features is not None
        assert ctx.df_features.shape[0] == len(context_monthly.get_unique_slices())

    def test_original_context_unchanged(self, context_monthly):
        ctx_copy = context_monthly.copy()
        ctx_copy._df_features = None
        eng = StandardStatsFeatureEngineer()
        eng.run(ctx_copy)
        assert ctx_copy._df_features is None


class TestFeatureNames:

    def test_names_match_columns(self, context_monthly):
        eng = StandardStatsFeatureEngineer()
        eng.calc_and_get_features_df(context_monthly)
        names = eng.feature_names()
        assert len(names) > 0

    def test_empty_before_fit(self):
        eng = StandardStatsFeatureEngineer()
        assert eng.feature_names() == []
