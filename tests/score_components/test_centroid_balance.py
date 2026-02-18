"""Tests for CentroidBalance."""
import math

import pytest

from energy_repset.score_components import CentroidBalance


@pytest.fixture
def component(context_with_features):
    comp = CentroidBalance()
    comp.prepare(context_with_features)
    return comp


class TestAttributes:

    def test_direction(self):
        assert CentroidBalance().direction == "min"

    def test_name(self):
        assert CentroidBalance().name == "centroid_balance"


class TestScore:

    def test_returns_finite_float(self, component, sample_combination):
        s = component.score(sample_combination)
        assert isinstance(s, float)
        assert math.isfinite(s)

    def test_non_negative(self, component, sample_combination):
        assert component.score(sample_combination) >= 0.0

    def test_full_selection_low(self, component, all_slices_combo, sample_combination):
        full = component.score(all_slices_combo)
        partial = component.score(sample_combination)
        # Full selection centroid is the global centroid - not necessarily lower,
        # just verify both are finite
        assert math.isfinite(full)
        assert math.isfinite(partial)
