"""Tests for UniformRepresentationModel."""
import math

import pytest

from energy_repset.representation import UniformRepresentationModel


class TestWeigh:

    def test_weights_equal(self, context_with_features, sample_combination):
        model = UniformRepresentationModel()
        model.fit(context_with_features)
        weights = model.weigh(sample_combination)
        k = len(sample_combination)
        for w in weights.values():
            assert w == pytest.approx(1.0 / k)

    def test_weights_sum_to_one(self, context_with_features, monthly_slices):
        model = UniformRepresentationModel()
        model.fit(context_with_features)
        combo = tuple(monthly_slices)
        weights = model.weigh(combo)
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_empty_combination(self, context_with_features):
        model = UniformRepresentationModel()
        model.fit(context_with_features)
        assert model.weigh(()) == {}
