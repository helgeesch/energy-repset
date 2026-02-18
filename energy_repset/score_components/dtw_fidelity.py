from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base_score_component import ScoreComponent

if TYPE_CHECKING:
    from ..types import SliceCombination
    from ..context import ProblemContext


class DTWFidelity(ScoreComponent):
    """Measures representation quality using Dynamic Time Warping distance.

    Computes the average DTW distance from each unselected slice to its
    nearest representative in the selection. This is analogous to inertia
    in k-medoids clustering but uses DTW instead of Euclidean distance.

    DTW allows temporal alignment, making it suitable for time-series where
    similar patterns may be shifted in time (e.g., seasonal load profiles
    with varying peak times).

    Requires the `tslearn` package for DTW computation.

    Examples:
        >>> from energy_repset.score_components import DTWFidelity
        >>> from energy_repset.objectives import ObjectiveSet
        >>>
        >>> # Use DTW for time-series with temporal shifts
        >>> objectives = ObjectiveSet({
        ...     'dtw': (1.0, DTWFidelity())
        ... })
        >>>
        >>> # Good for multi-day periods with similar but shifted patterns
        >>> # e.g., weeks with similar load but peak occurring at different times

    Note:
        This requires `tslearn` to be installed:
            pip install tslearn
    """

    def __init__(self) -> None:
        """Initialize DTW fidelity component."""
        self.name = "dtw"
        self.direction = "min"

    def prepare(self, context: ProblemContext) -> None:
        """Precompute per-slice time-series data.

        Args:
            context: Problem context containing raw time-series data.
        """
        df = context.df_raw
        self.slices = {
            label: group.values
            for label, group in df.groupby(context.slicer.labels_for_index(df.index))
        }
        self.all_labels = set(self.slices.keys())

    def score(self, combination: SliceCombination) -> float:
        """Compute average DTW distance from unselected to selected slices.

        Args:
            combination: Tuple of slice identifiers forming the selection.

        Returns:
            Average DTW distance from each unselected slice to its nearest
            representative. Returns 0.0 if all slices are selected or
            none are selected.

        Raises:
            ImportError: If tslearn is not installed.
        """
        from tslearn.metrics import dtw
        selected_labels = set(combination)
        unselected_labels = self.all_labels - selected_labels

        if not selected_labels or not unselected_labels:
            return 0.0

        selected_series = [self.slices[lbl] for lbl in selected_labels]
        total_dist = 0.0

        for lbl in unselected_labels:
            unselected_series = self.slices[lbl]
            min_dist = np.inf
            for sel_series in selected_series:
                dist = dtw(unselected_series, sel_series)
                if dist < min_dist:
                    min_dist = dist
            total_dist += min_dist

        return total_dist / len(unselected_labels)
