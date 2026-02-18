"""Tests for FeaturePipeline."""
import pytest

from energy_repset.feature_engineering import (
    FeaturePipeline,
    StandardStatsFeatureEngineer,
    PCAFeatureEngineer,
)


class TestCalcAndGetFeaturesDf:

    def test_chains_engineers(self, context_monthly):
        pipeline = FeaturePipeline({
            "stats": StandardStatsFeatureEngineer(),
            "pca": PCAFeatureEngineer(n_components=2),
        })
        df = pipeline.calc_and_get_features_df(context_monthly)
        stat_cols = [c for c in df.columns if not c.startswith("pc_")]
        pca_cols = [c for c in df.columns if c.startswith("pc_")]
        assert len(stat_cols) > 0
        assert len(pca_cols) == 2

    def test_run_populates_context(self, context_monthly):
        pipeline = FeaturePipeline({
            "stats": StandardStatsFeatureEngineer(),
        })
        ctx = pipeline.run(context_monthly)
        assert ctx.df_features is not None
        assert ctx.df_features.shape[0] == len(context_monthly.get_unique_slices())
