"""Tests for WassersteinFidelity."""
import math

import pytest

from energy_repset.score_components import WassersteinFidelity


@pytest.fixture
def component(context_with_features):
    comp = WassersteinFidelity()
    comp.prepare(context_with_features)
    return comp


class TestAttributes:

    def test_direction(self):
        assert WassersteinFidelity().direction == "min"

    def test_name(self):
        assert WassersteinFidelity().name == "wasserstein"


class TestPrepare:

    def test_stores_iqr(self, component):
        assert component.iqr is not None


class TestScore:

    def test_returns_finite_float(self, component, sample_combination):
        s = component.score(sample_combination)
        assert isinstance(s, float)
        assert math.isfinite(s)

    def test_full_selection_lower_score(self, component, sample_combination, all_slices_combo):
        full = component.score(all_slices_combo)
        partial = component.score(sample_combination)
        assert full <= partial

    def test_different_combos_different_scores(self, component, monthly_slices):
        combo_a = (monthly_slices[0], monthly_slices[1])
        combo_b = (monthly_slices[0], monthly_slices[2])
        assert component.score(combo_a) != component.score(combo_b)


class TestVariableWeights:

    def test_custom_weights(self, context_with_features, sample_combination):
        comp = WassersteinFidelity(variable_weights={"demand": 2.0, "solar": 0.0, "wind": 0.0})
        comp.prepare(context_with_features)
        s = comp.score(sample_combination)
        assert isinstance(s, float) and math.isfinite(s)
