"""Tests for NRMSEFidelity."""
import math

import pytest

from energy_repset.score_components import NRMSEFidelity


@pytest.fixture
def component(context_with_features):
    comp = NRMSEFidelity()
    comp.prepare(context_with_features)
    return comp


class TestAttributes:

    def test_direction(self):
        assert NRMSEFidelity().direction == "min"

    def test_name(self):
        assert NRMSEFidelity().name == "nrmse"


class TestScore:

    def test_returns_finite_float(self, component, sample_combination):
        s = component.score(sample_combination)
        assert isinstance(s, float)
        assert math.isfinite(s)

    def test_non_negative(self, component, sample_combination):
        assert component.score(sample_combination) >= 0.0

    def test_full_selection_near_zero(self, component, all_slices_combo):
        s = component.score(all_slices_combo)
        assert s < 0.05

    def test_variable_weights(self, context_with_features, sample_combination):
        comp = NRMSEFidelity(variable_weights={"demand": 1.0})
        comp.prepare(context_with_features)
        s = comp.score(sample_combination)
        assert math.isfinite(s)
