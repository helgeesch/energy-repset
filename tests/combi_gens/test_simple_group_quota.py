"""Tests for GroupQuotaCombiGen."""
import math

import pandas as pd
import pytest

from energy_repset.combi_gens import GroupQuotaCombiGen


@pytest.fixture
def seasonal_gen(four_months, season_map):
    """GroupQuotaCombiGen selecting 1 per season (winter/spring)."""
    return GroupQuotaCombiGen(
        k=2,
        slice_to_group_mapping=season_map,
        group_quota={"winter": 1, "spring": 1},
    )


class TestInit:

    def test_rejects_mismatched_quotas(self, four_months, season_map):
        with pytest.raises(ValueError, match="quotas"):
            GroupQuotaCombiGen(
                k=3,
                slice_to_group_mapping=season_map,
                group_quota={"winter": 1, "spring": 1},
            )


class TestGenerate:

    def test_yields_correct_count(self, seasonal_gen, four_months):
        combis = list(seasonal_gen.generate(four_months))
        # 2 winter * 2 spring = 4
        assert len(combis) == 4

    def test_each_combo_respects_quotas(self, seasonal_gen, four_months, season_map):
        for combo in seasonal_gen.generate(four_months):
            groups = [season_map[s] for s in combo]
            assert groups.count("winter") == 1
            assert groups.count("spring") == 1


class TestCount:

    def test_matches_product_of_binomials(self, seasonal_gen, four_months):
        # C(2,1) * C(2,1) = 4
        assert seasonal_gen.count(four_months) == 4


class TestCombinationIsValid:

    def test_valid(self, seasonal_gen, four_months):
        combo = (four_months[0], four_months[2])  # Jan + Mar
        assert seasonal_gen.combination_is_valid(combo, four_months) is True

    def test_invalid_wrong_quota(self, seasonal_gen, four_months):
        combo = (four_months[0], four_months[1])  # Jan + Feb (both winter)
        assert seasonal_gen.combination_is_valid(combo, four_months) is False

    def test_invalid_element(self, seasonal_gen, four_months):
        combo = (four_months[0], pd.Period("2099-01", "M"))
        assert seasonal_gen.combination_is_valid(combo, four_months) is False
