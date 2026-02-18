"""Tests for ObjectiveSet."""
import pytest

from energy_repset.objectives import ObjectiveSet
from energy_repset.score_components import WassersteinFidelity, DiversityReward


class TestObjectiveSetInit:

    def test_accepts_valid_components(self):
        obj = ObjectiveSet({
            "wass": (1.0, WassersteinFidelity()),
        })
        assert "wass" in obj.weighted_score_components

    def test_rejects_negative_weight(self):
        with pytest.raises(ValueError, match="Weight"):
            ObjectiveSet({"wass": (-1.0, WassersteinFidelity())})


class TestPrepareAndEvaluate:

    def test_prepare_does_not_raise(self, context_with_features):
        obj = ObjectiveSet({
            "wass": (1.0, WassersteinFidelity()),
            "div": (0.5, DiversityReward()),
        })
        obj.prepare(context_with_features)

    def test_evaluate_returns_scores(self, context_with_features, sample_combination):
        obj = ObjectiveSet({
            "wass": (1.0, WassersteinFidelity()),
            "div": (0.5, DiversityReward()),
        })
        obj.prepare(context_with_features)
        scores = obj.evaluate(sample_combination, context_with_features)
        assert "wasserstein" in scores
        assert "diversity" in scores
        assert all(isinstance(v, float) for v in scores.values())


class TestComponentMeta:

    def test_returns_direction_and_pref(self):
        obj = ObjectiveSet({
            "wass": (1.0, WassersteinFidelity()),
            "div": (0.5, DiversityReward()),
        })
        meta = obj.component_meta()
        assert meta["wasserstein"]["direction"] == "min"
        assert meta["diversity"]["direction"] == "max"
        assert meta["wasserstein"]["pref"] == 1.0
        assert meta["diversity"]["pref"] == 0.5
