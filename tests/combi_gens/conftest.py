"""Fixtures for combi gen tests."""
from __future__ import annotations

import pandas as pd
import pytest

from energy_repset.time_slicer import TimeSlicer


@pytest.fixture(scope="session")
def four_months():
    """4 monthly Period objects for Jan-Apr 2024."""
    return [pd.Period(f"2024-{i:02d}", "M") for i in range(1, 5)]


@pytest.fixture(scope="session")
def season_map(four_months):
    """Maps Jan/Feb -> winter, Mar/Apr -> spring."""
    mapping = {}
    for m in four_months:
        if m.month in [1, 2]:
            mapping[m] = "winter"
        else:
            mapping[m] = "spring"
    return mapping


@pytest.fixture(scope="session")
def daily_index_90d():
    """90-day DatetimeIndex starting 2024-01-01."""
    return pd.date_range("2024-01-01", periods=90, freq="D")


@pytest.fixture(scope="session")
def daily_slices_90d(daily_index_90d):
    """Unique daily Period objects for 90 days."""
    slicer = TimeSlicer(unit="day")
    return slicer.unique_slices(daily_index_90d)


@pytest.fixture(scope="session")
def day_to_month_map(daily_index_90d):
    """Mapping from daily periods to monthly periods for 90 days."""
    child_slicer = TimeSlicer(unit="day")
    parent_slicer = TimeSlicer(unit="month")
    child_labels = child_slicer.labels_for_index(daily_index_90d)
    parent_labels = parent_slicer.labels_for_index(daily_index_90d)
    return {c: p for c, p in zip(child_labels, parent_labels)}
