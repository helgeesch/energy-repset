from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base_score_component import ScoreComponent

if TYPE_CHECKING:
    from ..types import SliceCombination
    from ..context import ProblemContext


class CentroidBalance(ScoreComponent):
    """Penalizes selections whose centroid deviates from the global center.

    Computes the Euclidean distance between the centroid of selected slice
    features and the origin (global center in standardized feature space).

    This objective ensures the selection doesn't systematically bias toward
    extreme conditions, maintaining balance around typical conditions.

    Examples:
        >>> from energy_repset.score_components import CentroidBalance
        >>> from energy_repset.objectives import ObjectiveSet
        >>>
        >>> # Penalize selections biased toward extreme periods
        >>> objectives = ObjectiveSet({
        ...     'balance': (0.5, CentroidBalance())
        ... })
        >>>
        >>> # Used in examples/ex1.py to maintain balanced selections
        >>> from energy_repset.score_components import WassersteinFidelity
        >>> objectives = ObjectiveSet({
        ...     'fidelity': (1.0, WassersteinFidelity()),
        ...     'balance': (0.3, CentroidBalance())
        ... })
    """

    def __init__(self) -> None:
        """Initialize centroid balance component."""
        self.name = "centroid_balance"
        self.direction = "min"

    def prepare(self, context: ProblemContext) -> None:
        """Store the feature matrix for centroid computation.

        Args:
            context: Problem context with computed features (should be
                standardized for meaningful centroid distances).
        """
        self.features = context.df_features.copy()

    def score(self, combination: SliceCombination) -> float:
        """Compute distance from selection centroid to global center.

        Args:
            combination: Tuple of slice identifiers forming the selection.

        Returns:
            Euclidean distance from the selection's feature centroid to
            the origin. Lower values indicate more balanced selections.
        """
        X = self.features.loc[list(combination)].values
        mu = X.mean(axis=0)
        return float(np.linalg.norm(mu))
