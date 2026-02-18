"""Tests for ExhaustiveHierarchicalCombiGen."""
import math

import pandas as pd
import pytest

from energy_repset.combi_gens import ExhaustiveHierarchicalCombiGen
from energy_repset.time_slicer import TimeSlicer


class TestGenerate:

    def test_yields_child_slices(self, daily_slices_90d, day_to_month_map):
        gen = ExhaustiveHierarchicalCombiGen(
            parent_k=2, slice_to_parent_mapping=day_to_month_map
        )
        combis = list(gen.generate(daily_slices_90d))
        assert len(combis) == math.comb(3, 2)
        for combo in combis:
            assert all(isinstance(s, pd.Period) for s in combo)

    def test_complete_parent_groups(self, daily_slices_90d, day_to_month_map):
        gen = ExhaustiveHierarchicalCombiGen(
            parent_k=1, slice_to_parent_mapping=day_to_month_map
        )
        for combo in gen.generate(daily_slices_90d):
            parents = {day_to_month_map[c] for c in combo}
            assert len(parents) == 1


class TestFromSlicers:

    def test_factory_matches_manual(self, daily_index_90d, daily_slices_90d):
        gen = ExhaustiveHierarchicalCombiGen.from_slicers(
            parent_k=2,
            dt_index=daily_index_90d,
            child_slicer=TimeSlicer(unit="day"),
            parent_slicer=TimeSlicer(unit="month"),
        )
        assert gen.count(daily_slices_90d) == math.comb(3, 2)


class TestCount:

    def test_count_matches_parent_combinations(self, daily_slices_90d, day_to_month_map):
        gen = ExhaustiveHierarchicalCombiGen(
            parent_k=2, slice_to_parent_mapping=day_to_month_map
        )
        assert gen.count(daily_slices_90d) == math.comb(3, 2)


class TestCombinationIsValid:

    def test_valid_complete_group(self, daily_slices_90d, day_to_month_map):
        gen = ExhaustiveHierarchicalCombiGen(
            parent_k=1, slice_to_parent_mapping=day_to_month_map
        )
        first_combo = next(gen.generate(daily_slices_90d))
        assert gen.combination_is_valid(first_combo, daily_slices_90d) is True

    def test_invalid_partial_group(self, daily_slices_90d, day_to_month_map):
        gen = ExhaustiveHierarchicalCombiGen(
            parent_k=1, slice_to_parent_mapping=day_to_month_map
        )
        partial = tuple(daily_slices_90d[:5])  # only 5 days of January
        assert gen.combination_is_valid(partial, daily_slices_90d) is False
