"""Tests for ExhaustiveCombiGen."""
import math

import pandas as pd
import pytest

from energy_repset.combi_gens import ExhaustiveCombiGen


class TestGenerate:

    def test_yields_correct_count(self, four_months):
        gen = ExhaustiveCombiGen(k=2)
        combis = list(gen.generate(four_months))
        assert len(combis) == math.comb(4, 2)

    def test_tuple_length(self, four_months):
        gen = ExhaustiveCombiGen(k=3)
        for combo in gen.generate(four_months):
            assert len(combo) == 3

    def test_no_duplicates(self, four_months):
        gen = ExhaustiveCombiGen(k=2)
        combis = list(gen.generate(four_months))
        assert len(combis) == len(set(combis))

    def test_k_equals_n(self, four_months):
        gen = ExhaustiveCombiGen(k=4)
        combis = list(gen.generate(four_months))
        assert len(combis) == 1
        assert set(combis[0]) == set(four_months)


class TestCount:

    def test_matches_math_comb(self, four_months):
        gen = ExhaustiveCombiGen(k=2)
        assert gen.count(four_months) == math.comb(4, 2)

    def test_k_equals_one(self, four_months):
        gen = ExhaustiveCombiGen(k=1)
        assert gen.count(four_months) == 4


class TestCombinationIsValid:

    def test_valid_combination(self, four_months):
        gen = ExhaustiveCombiGen(k=2)
        combo = tuple(four_months[:2])
        assert gen.combination_is_valid(combo, four_months) is True

    def test_wrong_length(self, four_months):
        gen = ExhaustiveCombiGen(k=2)
        combo = tuple(four_months[:3])
        assert gen.combination_is_valid(combo, four_months) is False

    def test_invalid_element(self, four_months):
        gen = ExhaustiveCombiGen(k=2)
        combo = (four_months[0], pd.Period("2099-01", "M"))
        assert gen.combination_is_valid(combo, four_months) is False
