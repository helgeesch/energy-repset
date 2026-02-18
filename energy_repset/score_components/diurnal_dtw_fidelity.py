from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .base_score_component import ScoreComponent

if TYPE_CHECKING:
    from ..types import SliceCombination
    from ..context import ProblemContext


class DiurnalDTWFidelity(ScoreComponent):
    """Preserves hourly patterns using Dynamic Time Warping on diurnal profiles.

    Combines the concepts of DiurnalFidelity and DTWFidelity: compares
    hour-of-day aggregated profiles between full and selected data, but
    uses DTW distance instead of MSE to allow for temporal flexibility.

    This is useful when you want to preserve the general shape of hourly
    patterns but allow for some temporal shifting (e.g., load profiles
    with similar shapes but shifted peak hours).

    Uses a custom DTW implementation (no external dependencies), normalized
    by the standard deviation of the full diurnal profile.

    Examples:
        >>> from energy_repset.score_components import DiurnalDTWFidelity
        >>> from energy_repset.objectives import ObjectiveSet
        >>>
        >>> # Preserve diurnal patterns with temporal flexibility
        >>> objectives = ObjectiveSet({
        ...     'diurnal_dtw': (1.0, DiurnalDTWFidelity())
        ... })
        >>>
        >>> # Useful when hourly patterns are important but exact timing
        >>> # alignment is not critical (e.g., shifted daily load curves)

    Note:
        Unlike DTWFidelity, this does not require tslearn since it uses
        a custom DTW implementation on aggregated hourly profiles.
    """

    def __init__(self) -> None:
        """Initialize diurnal DTW fidelity component."""
        self.name = "diurnal_dtw"
        self.direction = "min"

    def _dtw_distance(self, s1: np.ndarray, s2: np.ndarray) -> float:
        """Compute Dynamic Time Warping distance between two 1D arrays.

        Args:
            s1: First time series.
            s2: Second time series.

        Returns:
            DTW distance between s1 and s2.
        """
        n, m = len(s1), len(s2)
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0.0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(s1[i - 1] - s2[j - 1])
                last_min = min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1])
                dtw_matrix[i, j] = cost + last_min

        return dtw_matrix[n, m]

    def prepare(self, context: ProblemContext) -> None:
        """Precompute full dataset's diurnal profile and normalization factors.

        Args:
            context: Problem context containing raw time-series data.
        """
        df = context.df_raw
        self.df = df
        self.labels = context.slicer.labels_for_index(df.index)
        self.vars = list(df.columns)
        self.full_diurnal = df.groupby(df.index.hour).mean(numeric_only=True)
        self.norm_factor = self.full_diurnal.std().replace(0, 1.0)

    def score(self, combination: SliceCombination) -> float:
        """Compute sum of per-variable normalized DTW distances for diurnal profiles.

        Args:
            combination: Tuple of slice identifiers forming the selection.

        Returns:
            Sum of per-variable DTW distances between full and selected
            diurnal profiles, normalized by standard deviation. Returns
            infinity if the selection is empty.
        """
        sel_mask = pd.Index(self.labels).isin(combination)
        if not sel_mask.any():
            return np.inf

        sel = self.df.loc[sel_mask]
        sel_diurnal = sel.groupby(sel.index.hour).mean(numeric_only=True)

        full_aligned, sel_aligned = self.full_diurnal.align(sel_diurnal, join="inner", axis=0)
        if full_aligned.empty:
            return np.inf

        total_dtw_dist = 0.0
        for v in self.vars:
            if v in full_aligned and v in sel_aligned:
                full_profile = full_aligned[v].values
                sel_profile = sel_aligned[v].values
                dist = self._dtw_distance(full_profile, sel_profile)
                total_dtw_dist += dist / float(self.norm_factor[v])

        return float(total_dtw_dist)
