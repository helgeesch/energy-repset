"""Pluggable fitness strategies for ranking individuals in genetic algorithms.

Fitness strategies convert multi-objective evaluation scores into a single
fitness value used for tournament selection. This is distinct from
SelectionPolicy, which picks the final winner from all evaluated candidates.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..objectives import ObjectiveSet


class FitnessStrategy(ABC):
    """Base class for fitness strategies used in evolutionary algorithms.

    A fitness strategy converts a DataFrame of multi-objective scores into
    a 1-D array of fitness values (higher = better), which the genetic
    algorithm uses for tournament selection.

    Subclasses must implement ``rank`` to define how individuals are ranked.
    """

    @abstractmethod
    def rank(
        self,
        evaluations_df: pd.DataFrame,
        objective_set: ObjectiveSet,
    ) -> np.ndarray:
        """Compute fitness values for a population of evaluated candidates.

        Args:
            evaluations_df: DataFrame with one row per individual and columns
                for each score component (e.g. ``"wasserstein"``, ``"correlation"``).
            objective_set: Provides component metadata (direction, weights).

        Returns:
            1-D numpy array of fitness values with the same length as
            ``evaluations_df``. Higher values indicate better fitness.
        """
        ...


class WeightedSumFitness(FitnessStrategy):
    """Fitness via weighted sum scalarization.

    Combines objectives into a single scalar using the weights from the
    ObjectiveSet. Objectives with ``direction="min"`` are negated so that
    higher fitness always means better quality.

    Args:
        normalize: If True, apply robust min-max normalization (5th-95th
            percentile) before weighting to make objectives comparable.

    Examples:

        >>> from energy_repset.search_algorithms.fitness import WeightedSumFitness
        >>> fitness = WeightedSumFitness(normalize=True)
    """

    def __init__(self, normalize: bool = False) -> None:
        self.normalize = normalize

    def rank(
        self,
        evaluations_df: pd.DataFrame,
        objective_set: ObjectiveSet,
    ) -> np.ndarray:
        """Compute weighted-sum fitness (higher = better).

        Args:
            evaluations_df: DataFrame with score columns for each component.
            objective_set: Provides direction and weights.

        Returns:
            1-D array of fitness values.
        """
        meta = objective_set.component_meta()
        score_names = list(meta.keys())
        Y = evaluations_df[score_names].copy().astype(float)

        for name, m in meta.items():
            if m["direction"] == "min":
                Y[name] = -Y[name]

        if self.normalize:
            Y = self._robust_minmax(Y)

        weights = np.array([meta[n]["pref"] for n in score_names])
        fitness = Y.values @ weights
        return fitness

    @staticmethod
    def _robust_minmax(Y: pd.DataFrame) -> pd.DataFrame:
        q_lo = Y.quantile(0.05)
        q_hi = Y.quantile(0.95)
        denom = (q_hi - q_lo).replace(0, 1.0)
        return (Y - q_lo) / denom


class NSGA2Fitness(FitnessStrategy):
    """Non-dominated sorting with crowding distance (Deb et al. 2002).

    Assigns fitness based on Pareto front rank and crowding distance.
    Individuals on lower (better) fronts receive higher base fitness,
    with crowding distance used to break ties within the same front.

    Composite fitness: ``(max_front - front_rank) * N + crowding_distance``

    This ensures front rank always dominates, while crowding distance
    differentiates individuals within the same front.

    Examples:

        >>> from energy_repset.search_algorithms.fitness import NSGA2Fitness
        >>> fitness = NSGA2Fitness()
    """

    def rank(
        self,
        evaluations_df: pd.DataFrame,
        objective_set: ObjectiveSet,
    ) -> np.ndarray:
        """Compute NSGA-II fitness (higher = better).

        Args:
            evaluations_df: DataFrame with score columns for each component.
            objective_set: Provides direction metadata.

        Returns:
            1-D array of fitness values.
        """
        meta = objective_set.component_meta()
        score_names = list(meta.keys())
        Y = evaluations_df[score_names].values.astype(float)

        for i, name in enumerate(score_names):
            if meta[name]["direction"] == "min":
                Y[:, i] = -Y[:, i]

        n = len(Y)
        fronts = self._fast_non_dominated_sort(Y)

        front_rank = np.empty(n, dtype=int)
        crowding = np.zeros(n, dtype=float)

        for rank_idx, front in enumerate(fronts):
            for idx in front:
                front_rank[idx] = rank_idx
            cd = self._crowding_distance(Y[front])
            for j, idx in enumerate(front):
                crowding[idx] = cd[j]

        max_front = len(fronts)
        fitness = (max_front - front_rank) * n + crowding
        return fitness

    @staticmethod
    def _fast_non_dominated_sort(Y: np.ndarray) -> list[list[int]]:
        """Partition population into non-dominated fronts.

        Args:
            Y: Objective matrix (n x m), oriented for maximization.

        Returns:
            List of fronts, each a list of individual indices.
        """
        n = len(Y)
        domination_count = np.zeros(n, dtype=int)
        dominated_by: list[list[int]] = [[] for _ in range(n)]
        fronts: list[list[int]] = []
        current_front: list[int] = []

        for i in range(n):
            for j in range(i + 1, n):
                if np.all(Y[i] >= Y[j]) and np.any(Y[i] > Y[j]):
                    dominated_by[i].append(j)
                    domination_count[j] += 1
                elif np.all(Y[j] >= Y[i]) and np.any(Y[j] > Y[i]):
                    dominated_by[j].append(i)
                    domination_count[i] += 1

        for i in range(n):
            if domination_count[i] == 0:
                current_front.append(i)

        while current_front:
            fronts.append(current_front)
            next_front: list[int] = []
            for i in current_front:
                for j in dominated_by[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front = next_front

        return fronts

    @staticmethod
    def _crowding_distance(Y_front: np.ndarray) -> np.ndarray:
        """Compute crowding distance for individuals in a single front.

        Args:
            Y_front: Objective values for this front (n_front x m).

        Returns:
            1-D array of crowding distances.
        """
        n, m = Y_front.shape
        if n <= 2:
            return np.full(n, np.inf)

        distances = np.zeros(n)

        for obj_idx in range(m):
            order = np.argsort(Y_front[:, obj_idx])
            obj_range = Y_front[order[-1], obj_idx] - Y_front[order[0], obj_idx]
            if obj_range == 0:
                continue

            distances[order[0]] = np.inf
            distances[order[-1]] = np.inf

            for i in range(1, n - 1):
                distances[order[i]] += (
                    Y_front[order[i + 1], obj_idx] - Y_front[order[i - 1], obj_idx]
                ) / obj_range

        return distances
