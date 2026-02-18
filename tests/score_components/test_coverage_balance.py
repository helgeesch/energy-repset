"""Tests for CoverageBalance."""
import math

import pytest

from energy_repset.score_components import CoverageBalance


@pytest.fixture
def component(context_with_features):
    comp = CoverageBalance()
    comp.prepare(context_with_features)
    return comp


class TestAttributes:

    def test_direction(self):
        assert CoverageBalance().direction == "min"

    def test_name(self):
        assert CoverageBalance().name == "coverage_balance"


class TestScore:

    def test_returns_finite_float(self, component, sample_combination):
        s = component.score(sample_combination)
        assert isinstance(s, float)
        assert math.isfinite(s)

    def test_non_negative(self, component, sample_combination):
        assert component.score(sample_combination) >= 0.0

    def test_custom_gamma(self, context_with_features, sample_combination):
        comp = CoverageBalance(gamma=0.5)
        comp.prepare(context_with_features)
        s = comp.score(sample_combination)
        assert math.isfinite(s)
