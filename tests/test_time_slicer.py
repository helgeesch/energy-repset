"""Tests for TimeSlicer."""
import pandas as pd
import pytest

from energy_repset.time_slicer import TimeSlicer


class TestLabelsForIndex:

    def test_monthly_returns_periods(self, df_raw_hourly, monthly_slicer):
        labels = monthly_slicer.labels_for_index(df_raw_hourly.index)
        assert len(labels) == len(df_raw_hourly)
        assert all(isinstance(l, pd.Period) for l in labels.unique())

    def test_daily_returns_periods(self, df_raw_hourly, daily_slicer):
        labels = daily_slicer.labels_for_index(df_raw_hourly.index)
        assert len(labels) == len(df_raw_hourly)
        assert labels.unique().size == 90

    def test_weekly_returns_periods(self, df_raw_hourly):
        slicer = TimeSlicer(unit="week")
        labels = slicer.labels_for_index(df_raw_hourly.index)
        assert len(labels) == len(df_raw_hourly)

    def test_hourly_returns_timestamps(self, df_raw_hourly):
        slicer = TimeSlicer(unit="hour")
        labels = slicer.labels_for_index(df_raw_hourly.index)
        assert len(labels) == len(df_raw_hourly)

    def test_yearly_returns_periods(self, df_raw_hourly):
        slicer = TimeSlicer(unit="year")
        labels = slicer.labels_for_index(df_raw_hourly.index)
        assert labels.unique().size == 1

    def test_unsupported_unit_raises(self, df_raw_hourly):
        slicer = TimeSlicer(unit="minute")
        with pytest.raises(ValueError, match="Unsupported unit"):
            slicer.labels_for_index(df_raw_hourly.index)


class TestUniqueSlices:

    def test_monthly_count(self, df_raw_hourly, monthly_slicer):
        slices = monthly_slicer.unique_slices(df_raw_hourly.index)
        assert len(slices) == 3  # Jan, Feb, Mar for 90-day data

    def test_daily_count(self, df_raw_hourly, daily_slicer):
        slices = daily_slicer.unique_slices(df_raw_hourly.index)
        assert len(slices) == 90

    def test_sorted_order(self, df_raw_hourly, monthly_slicer):
        slices = monthly_slicer.unique_slices(df_raw_hourly.index)
        assert slices == sorted(slices)


class TestGetIndicesForSliceCombi:

    def test_single_slice(self, df_raw_hourly, monthly_slicer, monthly_slices):
        idx = monthly_slicer.get_indices_for_slice_combi(
            df_raw_hourly.index, monthly_slices[0]
        )
        assert len(idx) == 31 * 24  # January has 31 days

    def test_combination_of_slices(self, df_raw_hourly, monthly_slicer, monthly_slices):
        combo = tuple(monthly_slices[:2])
        idx = monthly_slicer.get_indices_for_slice_combi(df_raw_hourly.index, combo)
        jan_hours = 31 * 24
        feb_hours = 29 * 24  # 2024 is a leap year
        assert len(idx) == jan_hours + feb_hours

    def test_all_slices_cover_full_index(self, df_raw_hourly, monthly_slicer, monthly_slices):
        combo = tuple(monthly_slices)
        idx = monthly_slicer.get_indices_for_slice_combi(df_raw_hourly.index, combo)
        assert len(idx) == len(df_raw_hourly)
