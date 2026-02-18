"""Tests for GroupQuotaHierarchicalCombiGen."""
import math

import pandas as pd
import pytest

from energy_repset.combi_gens import GroupQuotaHierarchicalCombiGen
from energy_repset.time_slicer import TimeSlicer


@pytest.fixture
def month_to_season():
    """Map 3 months (Jan-Mar 2024) to seasons."""
    return {
        pd.Period("2024-01", "M"): "winter",
        pd.Period("2024-02", "M"): "winter",
        pd.Period("2024-03", "M"): "spring",
    }


@pytest.fixture
def hierarchical_gen(day_to_month_map, month_to_season):
    """Select 2 months: 1 winter + 1 spring."""
    return GroupQuotaHierarchicalCombiGen(
        parent_k=2,
        slice_to_parent_mapping=day_to_month_map,
        parent_to_group_mapping=month_to_season,
        group_quota={"winter": 1, "spring": 1},
    )


class TestInit:

    def test_rejects_mismatched_quotas(self, day_to_month_map, month_to_season):
        with pytest.raises(ValueError, match="quotas"):
            GroupQuotaHierarchicalCombiGen(
                parent_k=3,
                slice_to_parent_mapping=day_to_month_map,
                parent_to_group_mapping=month_to_season,
                group_quota={"winter": 1, "spring": 1},
            )


class TestGenerate:

    def test_yields_correct_count(self, hierarchical_gen, daily_slices_90d):
        combis = list(hierarchical_gen.generate(daily_slices_90d))
        # C(2,1) * C(1,1) = 2
        assert len(combis) == 2

    def test_respects_quotas(self, hierarchical_gen, daily_slices_90d, day_to_month_map, month_to_season):
        for combo in hierarchical_gen.generate(daily_slices_90d):
            parents = {day_to_month_map[c] for c in combo}
            seasons = [month_to_season[p] for p in parents]
            assert seasons.count("winter") == 1
            assert seasons.count("spring") == 1


class TestCount:

    def test_matches_product(self, hierarchical_gen, daily_slices_90d):
        assert hierarchical_gen.count(daily_slices_90d) == 2


class TestFromSlicersWithSeasons:

    def test_factory_creates_valid_gen(self, daily_index_90d, daily_slices_90d):
        gen = GroupQuotaHierarchicalCombiGen.from_slicers_with_seasons(
            parent_k=2,
            dt_index=daily_index_90d,
            child_slicer=TimeSlicer(unit="day"),
            group_quota={"winter": 1, "spring": 1},
        )
        count = gen.count(daily_slices_90d)
        assert count > 0
