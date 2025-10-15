from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base_score_component import ScoreComponent

if TYPE_CHECKING:
    from ..types import SliceCombination
    from ..context import ProblemContext


class DiversityReward(ScoreComponent):
    """Rewards selections with diverse, mutually distant representative periods.

    Computes the average pairwise Euclidean distance between selected slice
    features in feature space. Higher diversity can help ensure the selection
    covers a wider range of conditions, avoiding redundant representatives.

    This is particularly useful when combined with fidelity objectives to
    balance accuracy with coverage.

    Examples:
        >>> from mesqual_repset.score_components import DiversityReward
        >>> from mesqual_repset.objectives import ObjectiveSet
        >>>
        >>> # Encourage diverse representatives
        >>> objectives = ObjectiveSet({
        ...     'diversity': (0.3, DiversityReward())
        ... })
        >>>
        >>> # Combine with fidelity for balanced selection
        >>> from mesqual_repset.score_components import WassersteinFidelity
        >>> objectives = ObjectiveSet({
        ...     'fidelity': (1.0, WassersteinFidelity()),
        ...     'diversity': (0.2, DiversityReward())
        ... })
    """

    def __init__(self) -> None:
        """Initialize diversity reward component."""
        self.name = "diversity"
        self.direction = "max"

    def prepare(self, context: ProblemContext) -> None:
        """Store the feature matrix for pairwise distance computation.

        Args:
            context: Problem context with computed features.
        """
        self.features = context.df_features.copy()

    def score(self, combination: SliceCombination) -> float:
        """Compute mean pairwise Euclidean distance among selected features.

        Args:
            combination: Tuple of slice identifiers forming the selection.

        Returns:
            Average pairwise distance in feature space. Returns 0.0 if
            fewer than two slices are selected.
        """
        X = self.features.loc[list(combination)].values
        if X.shape[0] < 2:
            return 0.0
        n = X.shape[0]
        dsum = 0.0
        cnt = 0
        for i in range(n):
            for j in range(i + 1, n):
                dsum += float(np.linalg.norm(X[i] - X[j]))
                cnt += 1
        return dsum / cnt
