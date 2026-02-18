"""Shared fixtures for the energy_repset test suite."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energy_repset.time_slicer import TimeSlicer
from energy_repset.context import ProblemContext
from energy_repset.feature_engineering import StandardStatsFeatureEngineer


@pytest.fixture(scope="session")
def df_raw_hourly() -> pd.DataFrame:
    """90-day hourly DataFrame with 3 variables (demand, solar, wind).

    Deterministic via numpy.random.default_rng(42). Patterns:
    - demand: diurnal sinusoidal + seasonal trend + noise
    - solar: daytime-only half-sine, zero at night
    - wind: autocorrelated random walk + noise
    """
    rng = np.random.default_rng(42)
    n_hours = 90 * 24  # 2160 rows
    index = pd.date_range("2024-01-01", periods=n_hours, freq="h")

    hours = np.arange(n_hours)
    hour_of_day = hours % 24
    day_of_year = hours / 24

    demand = (
        50
        + 20 * np.sin(2 * np.pi * hour_of_day / 24 - np.pi / 2)
        + 5 * np.sin(2 * np.pi * day_of_year / 365)
        + rng.normal(0, 3, n_hours)
    )

    solar_raw = np.sin(np.pi * (hour_of_day - 6) / 12)
    solar_raw = np.where((hour_of_day >= 6) & (hour_of_day <= 18), solar_raw, 0.0)
    solar = solar_raw * (30 + 10 * np.sin(2 * np.pi * day_of_year / 365)) + rng.normal(0, 1, n_hours)
    solar = np.clip(solar, 0, None)

    wind = np.zeros(n_hours)
    wind[0] = 10.0
    for i in range(1, n_hours):
        wind[i] = 0.95 * wind[i - 1] + rng.normal(0, 1.5)
    wind = np.clip(wind, 0, None)

    return pd.DataFrame(
        {"demand": demand, "solar": solar, "wind": wind},
        index=index,
    )


@pytest.fixture(scope="session")
def monthly_slicer() -> TimeSlicer:
    """TimeSlicer with monthly granularity."""
    return TimeSlicer(unit="month")


@pytest.fixture(scope="session")
def daily_slicer() -> TimeSlicer:
    """TimeSlicer with daily granularity."""
    return TimeSlicer(unit="day")


@pytest.fixture(scope="session")
def context_monthly(df_raw_hourly, monthly_slicer) -> ProblemContext:
    """ProblemContext with monthly slicing (3 months for 90-day data)."""
    return ProblemContext(df_raw=df_raw_hourly, slicer=monthly_slicer)


@pytest.fixture(scope="session")
def context_with_features(context_monthly) -> ProblemContext:
    """ProblemContext with df_features populated via StandardStatsFeatureEngineer."""
    engineer = StandardStatsFeatureEngineer()
    return engineer.run(context_monthly)


@pytest.fixture(scope="session")
def monthly_slices(context_monthly) -> list:
    """List of 3 monthly Period objects."""
    return context_monthly.get_unique_slices()


@pytest.fixture(scope="session")
def sample_combination(monthly_slices) -> tuple:
    """Tuple of first 2 monthly slices for k=2 tests."""
    return tuple(monthly_slices[:2])


@pytest.fixture(scope="session")
def context_daily(df_raw_hourly, daily_slicer) -> ProblemContext:
    """ProblemContext with daily slicing (90 days for 90-day data)."""
    return ProblemContext(df_raw=df_raw_hourly, slicer=daily_slicer)


@pytest.fixture(scope="session")
def context_daily_with_stats_features(context_daily) -> ProblemContext:
    """Daily ProblemContext with df_features via StandardStatsFeatureEngineer."""
    engineer = StandardStatsFeatureEngineer()
    return engineer.run(context_daily)


