"""Tests for KMedoidsClustersizeRepresentation."""
import pytest

from energy_repset.representation import KMedoidsClustersizeRepresentation


class TestWeigh:

    def test_weights_sum_to_one(self, context_with_features, sample_combination):
        model = KMedoidsClustersizeRepresentation()
        model.fit(context_with_features)
        weights = model.weigh(sample_combination)
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_all_weights_positive(self, context_with_features, sample_combination):
        model = KMedoidsClustersizeRepresentation()
        model.fit(context_with_features)
        weights = model.weigh(sample_combination)
        for w in weights.values():
            assert w > 0

    def test_empty_combination(self, context_with_features):
        model = KMedoidsClustersizeRepresentation()
        model.fit(context_with_features)
        assert model.weigh(()) == {}

    def test_invalid_slice_raises(self, context_with_features):
        import pandas as pd
        model = KMedoidsClustersizeRepresentation()
        model.fit(context_with_features)
        with pytest.raises(ValueError, match="not found"):
            model.weigh((pd.Period("2099-01", "M"),))
