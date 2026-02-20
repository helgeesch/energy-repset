from __future__ import annotations
from typing import Literal

from .search_algorithm import SearchAlgorithm


class ClusteringSearch(SearchAlgorithm):
    def __init__(self, cluster_model, selection_space: Literal['subset', 'synthetic']):...
    # This class has its own logic (e.g., k-medoids) and does not need
    # an external objective to run.
    def find_selection(self, context):
        # ... k-medoids logic ...
        raise NotImplementedError


class OptimizationSearch(SearchAlgorithm):
    # This class formulates and solves a MILP. Its objective is
    # built into the mathematical formulation.
    def find_selection(self, context):
        # ... MILP formulation and solver call ...
        raise NotImplementedError


