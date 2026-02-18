"""Tests for ProblemContext."""
import pandas as pd
import pytest

from energy_repset.context import ProblemContext


class TestInit:

    def test_stores_data(self, df_raw_hourly, monthly_slicer):
        ctx = ProblemContext(df_raw=df_raw_hourly, slicer=monthly_slicer)
        assert ctx.df_raw is df_raw_hourly
        assert ctx.slicer is monthly_slicer

    def test_default_metadata_empty(self, df_raw_hourly, monthly_slicer):
        ctx = ProblemContext(df_raw=df_raw_hourly, slicer=monthly_slicer)
        assert ctx.metadata == {}

    def test_custom_metadata(self, df_raw_hourly, monthly_slicer):
        meta = {"experiment": "test"}
        ctx = ProblemContext(df_raw=df_raw_hourly, slicer=monthly_slicer, metadata=meta)
        assert ctx.metadata["experiment"] == "test"


class TestDfFeatures:

    def test_access_before_set_raises(self, context_monthly):
        fresh = context_monthly.copy()
        fresh._df_features = None
        with pytest.raises(ValueError, match="df_features"):
            _ = fresh.df_features

    def test_setter_validates_slices(self, context_monthly):
        bad_df = pd.DataFrame({"a": [1, 2]}, index=["x", "y"])
        with pytest.raises(ValueError, match="missing"):
            context_monthly.copy().df_features = bad_df

    def test_setter_accepts_valid_features(self, context_monthly, monthly_slices):
        ctx = context_monthly.copy()
        df = pd.DataFrame({"feat": range(len(monthly_slices))}, index=monthly_slices)
        ctx.df_features = df
        assert ctx.df_features.shape == df.shape


class TestCopy:

    def test_deep_copy_independent(self, context_monthly):
        original = context_monthly.copy()
        clone = original.copy()
        clone.metadata["new_key"] = "value"
        assert "new_key" not in original.metadata


class TestGetUniqueSlices:

    def test_returns_correct_count(self, context_monthly):
        slices = context_monthly.get_unique_slices()
        assert len(slices) == 3
