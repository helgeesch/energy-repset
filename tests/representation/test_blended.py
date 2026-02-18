"""Tests for BlendedRepresentationModel."""
import numpy as np
import pytest

from energy_repset.representation import BlendedRepresentationModel


class TestWeigh:

    def test_returns_dataframe(self, context_with_features, sample_combination):
        model = BlendedRepresentationModel()
        model.fit(context_with_features)
        weights_df = model.weigh(sample_combination)
        assert weights_df.shape[0] == len(context_with_features.get_unique_slices())
        assert weights_df.shape[1] == len(sample_combination)

    def test_convex_weights(self, context_with_features, sample_combination):
        model = BlendedRepresentationModel()
        model.fit(context_with_features)
        weights_df = model.weigh(sample_combination)
        # Each row should sum to ~1.0
        row_sums = weights_df.sum(axis=1)
        for s in row_sums:
            assert s == pytest.approx(1.0, abs=1e-4)
        # All weights should be non-negative
        assert (weights_df.values >= -1e-6).all()

    def test_empty_combination(self, context_with_features):
        model = BlendedRepresentationModel()
        model.fit(context_with_features)
        result = model.weigh(())
        assert result.empty

    def test_unsupported_blend_type(self):
        with pytest.raises(NotImplementedError):
            BlendedRepresentationModel(blend_type="affine")
