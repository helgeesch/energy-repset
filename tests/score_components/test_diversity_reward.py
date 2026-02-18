"""Tests for DiversityReward."""
import math

import pytest

from energy_repset.score_components import DiversityReward


@pytest.fixture
def component(context_with_features):
    comp = DiversityReward()
    comp.prepare(context_with_features)
    return comp


class TestAttributes:

    def test_direction(self):
        assert DiversityReward().direction == "max"

    def test_name(self):
        assert DiversityReward().name == "diversity"


class TestScore:

    def test_returns_finite_float(self, component, sample_combination):
        s = component.score(sample_combination)
        assert isinstance(s, float)
        assert math.isfinite(s)

    def test_single_slice_returns_zero(self, component, single_slice_combo):
        assert component.score(single_slice_combo) == 0.0

    def test_positive_for_multiple_slices(self, component, sample_combination):
        assert component.score(sample_combination) > 0.0
