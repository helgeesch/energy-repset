"""Tests for DiurnalFidelity."""
import math

import pytest

from energy_repset.score_components import DiurnalFidelity


@pytest.fixture
def component(context_with_features):
    comp = DiurnalFidelity()
    comp.prepare(context_with_features)
    return comp


class TestAttributes:

    def test_direction(self):
        assert DiurnalFidelity().direction == "min"

    def test_name(self):
        assert DiurnalFidelity().name == "diurnal"


class TestScore:

    def test_returns_finite_float(self, component, sample_combination):
        s = component.score(sample_combination)
        assert isinstance(s, float)
        assert math.isfinite(s)

    def test_full_selection_near_zero(self, component, all_slices_combo):
        s = component.score(all_slices_combo)
        assert s < 0.01

    def test_non_negative(self, component, sample_combination):
        assert component.score(sample_combination) >= 0.0
