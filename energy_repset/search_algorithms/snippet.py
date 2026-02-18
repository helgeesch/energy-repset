from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .search_algorithm import SearchAlgorithm
from ..results import RepSetResult

if TYPE_CHECKING:
    from ..context import ProblemContext


class SnippetSearch(SearchAlgorithm):
    """Greedy p-median selection of multi-day representative subsequences.

    Implements the Snippet algorithm from Teichgraeber & Brandt (2024).
    Selects k sliding-window subsequences of ``period_length_days`` days each,
    minimizing the total day-level distance across the full time horizon.

    Each candidate subsequence contains ``period_length_days`` daily profile
    snippets. The distance from any day to a candidate is the minimum
    Euclidean distance to any of its constituent daily snippets. The greedy
    selection picks the candidate with the greatest total cost reduction at
    each iteration.

    Requires ``context.slicer.unit == 'day'``.

    Args:
        k: Number of representative subsequences to select.
        period_length_days: Length of each candidate subsequence in days.
        step_days: Stride between consecutive sliding-window candidates.

    Examples:
        Basic usage with daily slicing:

        >>> from energy_repset.search_algorithms import SnippetSearch
        >>> from energy_repset.time_slicer import TimeSlicer
        >>> slicer = TimeSlicer(unit='day')
        >>> search = SnippetSearch(k=8, period_length_days=7, step_days=1)
    """

    def __init__(
        self, k: int, period_length_days: int = 7, step_days: int = 1,
    ):
        """Initialize Snippet search.

        Args:
            k: Number of representative subsequences to select.
            period_length_days: Number of days in each candidate subsequence.
            step_days: Stride between consecutive candidate start positions.
        """
        self.k = k
        self.period_length_days = period_length_days
        self.step_days = step_days

    def find_selection(self, context: ProblemContext) -> RepSetResult:
        """Find k representative subsequences via greedy p-median selection.

        Args:
            context: Problem context. Must have ``slicer.unit == 'day'``.
                Feature engineering should provide daily profile vectors in
                ``df_features``, but the algorithm can also build profiles
                from ``df_raw`` directly.

        Returns:
            RepSetResult with selected starting-day labels, pre-computed
            weights (fraction of days assigned), and total distance score.

        Raises:
            ValueError: If ``context.slicer.unit`` is not ``'day'``.
        """
        if context.slicer.unit != 'day':
            raise ValueError(
                f"SnippetSearch requires daily slicing (unit='day'), "
                f"got unit='{context.slicer.unit}'."
            )

        daily_profiles = self._build_daily_profiles(context)
        N = daily_profiles.shape[0]
        day_labels = list(context.df_features.index)
        L = self.period_length_days

        candidates = self._generate_candidates(N, L, self.step_days)
        if len(candidates) < self.k:
            raise ValueError(
                f"Only {len(candidates)} candidate subsequences available, "
                f"but k={self.k} requested. Reduce k or period_length_days."
            )

        dist_matrix = self._compute_distance_matrix(daily_profiles, candidates)

        selected_candidates, per_day_min = self._greedy_select(
            dist_matrix, self.k
        )

        assignments = np.argmin(
            dist_matrix[:, selected_candidates], axis=1
        )

        selection_labels = []
        weight_dict = {}
        for local_idx, cand_idx in enumerate(selected_candidates):
            start_day = candidates[cand_idx][0]
            label = day_labels[start_day]
            selection_labels.append(label)
            n_assigned = int(np.sum(assignments == local_idx))
            weight_dict[label] = n_assigned / N

        selection = tuple(selection_labels)
        total_distance = float(np.sum(per_day_min))

        slice_labels = context.slicer.labels_for_index(context.df_raw.index)
        representatives = {}
        for label in selection:
            start_idx = day_labels.index(label)
            end_idx = min(start_idx + L, N)
            period_labels = day_labels[start_idx:end_idx]
            mask = slice_labels.isin(set(period_labels))
            representatives[label] = context.df_raw.loc[mask]

        return RepSetResult(
            context=context,
            selection_space='subset',
            selection=selection,
            scores={'total_distance': total_distance},
            representatives=representatives,
            weights=weight_dict,
            diagnostics={
                'assignments': assignments.tolist(),
                'candidate_starts': [
                    day_labels[candidates[c][0]] for c in selected_candidates
                ],
            },
        )

    def _build_daily_profiles(self, context: ProblemContext) -> np.ndarray:
        """Build daily profile vectors from context features.

        Args:
            context: Problem context with ``df_features`` populated.

        Returns:
            Array of shape (N_days, n_features) with one row per day.
        """
        return context.df_features.values

    def _generate_candidates(
        self, n_days: int, length: int, step: int
    ) -> list[list[int]]:
        """Generate sliding-window candidate subsequences.

        Args:
            n_days: Total number of days.
            length: Number of days per subsequence.
            step: Stride between consecutive candidates.

        Returns:
            List of candidates, each a list of day indices.
        """
        candidates = []
        start = 0
        while start + length <= n_days:
            candidates.append(list(range(start, start + length)))
            start += step
        return candidates

    def _compute_distance_matrix(
        self, profiles: np.ndarray, candidates: list[list[int]]
    ) -> np.ndarray:
        """Compute distance from each day to each candidate.

        For each (day, candidate) pair, the distance is the minimum Euclidean
        distance between the day's profile and any of the candidate's daily
        snippet profiles.

        Args:
            profiles: Daily profile matrix (N x p).
            candidates: List of candidate subsequences (each a list of day indices).

        Returns:
            Distance matrix of shape (N, C) where C is the number of candidates.
        """
        N = profiles.shape[0]
        C = len(candidates)
        dist_matrix = np.empty((N, C))

        for j, cand_days in enumerate(candidates):
            cand_profiles = profiles[cand_days]
            for i in range(N):
                diffs = cand_profiles - profiles[i]
                dists = np.sum(diffs ** 2, axis=1)
                dist_matrix[i, j] = np.min(dists)

        return dist_matrix

    def _greedy_select(
        self, dist_matrix: np.ndarray, k: int
    ) -> tuple[list[int], np.ndarray]:
        """Greedy p-median selection of k candidates.

        At each iteration, selects the candidate that most reduces the total
        per-day minimum distance.

        Args:
            dist_matrix: Distance matrix (N x C).
            k: Number of candidates to select.

        Returns:
            Tuple of (selected candidate indices, per-day minimum distances).
        """
        N = dist_matrix.shape[0]
        per_day_min = np.full(N, np.inf)
        selected: list[int] = []

        for _ in range(k):
            best_candidate = -1
            best_reduction = -np.inf

            for j in range(dist_matrix.shape[1]):
                if j in selected:
                    continue
                new_min = np.minimum(per_day_min, dist_matrix[:, j])
                reduction = np.sum(per_day_min) - np.sum(new_min)
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_candidate = j

            selected.append(best_candidate)
            per_day_min = np.minimum(per_day_min, dist_matrix[:, best_candidate])

        return selected, per_day_min
