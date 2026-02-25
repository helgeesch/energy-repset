from __future__ import annotations

from .search_algorithm import SearchAlgorithm


class OptimizationSearch(SearchAlgorithm):
    # This class formulates and solves a MILP. Its objective is
    # built into the mathematical formulation.
    def find_selection(self, context):
        # ... MILP formulation and solver call ...
        raise NotImplementedError


