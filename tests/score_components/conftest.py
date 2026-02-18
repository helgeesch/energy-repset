"""Fixtures for score component tests."""
from __future__ import annotations

import pytest


@pytest.fixture
def all_slices_combo(monthly_slices):
    """Combination containing all slices (full selection)."""
    return tuple(monthly_slices)


@pytest.fixture
def single_slice_combo(monthly_slices):
    """Combination with just one slice."""
    return (monthly_slices[0],)
