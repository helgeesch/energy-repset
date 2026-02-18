from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base_score_component import ScoreComponent

if TYPE_CHECKING:
    from ..types import SliceCombination
    from ..context import ProblemContext


class CoverageBalance(ScoreComponent):
    """Promotes balanced coverage by encouraging uniform responsibility.

    Uses RBF (Radial Basis Function) kernel-based soft assignment to compute
    how much "responsibility" each selected representative has for covering
    all candidate slices. Penalizes selections where some representatives
    cover many slices while others cover few.

    This is conceptually similar to cluster balance in k-medoids, ensuring
    no representative is over- or under-utilized.

    Args:
        gamma: RBF kernel sharpness parameter (higher = sharper assignments).
            Default is 1.0.

    Examples:
        >>> from energy_repset.score_components import CoverageBalance
        >>> from energy_repset.objectives import ObjectiveSet
        >>>
        >>> # Ensure balanced coverage with default sharpness
        >>> objectives = ObjectiveSet({
        ...     'coverage': (0.5, CoverageBalance())
        ... })
        >>>
        >>> # Sharper assignments (more cluster-like behavior)
        >>> objectives = ObjectiveSet({
        ...     'coverage': (0.5, CoverageBalance(gamma=2.0))
        ... })
        >>>
        >>> # Softer assignments (smoother transitions)
        >>> objectives = ObjectiveSet({
        ...     'coverage': (0.5, CoverageBalance(gamma=0.5))
        ... })
    """

    def __init__(self, gamma: float = 1.0) -> None:
        """Initialize coverage balance component.

        Args:
            gamma: RBF kernel sharpness. Higher values create sharper
                cluster-like assignments.
        """
        self.name = "coverage_balance"
        self.direction = "min"
        self.gamma = gamma

    def prepare(self, context: ProblemContext) -> None:
        """Store feature matrix for responsibility computation.

        Args:
            context: Problem context with computed features.
        """
        self.features = context.df_features.copy()
        self.all_X = np.nan_to_num(self.features.values, nan=0.0)

    def _responsibilities(self, combination: SliceCombination) -> np.ndarray:
        """Compute soft assignment responsibilities using RBF kernel.

        Args:
            combination: Tuple of slice identifiers.

        Returns:
            Array of responsibility weights for each selected slice,
            summing to 1.0.
        """
        sel_X = self.features.loc[list(combination)].values
        # Compute squared distances: (n_all, n_sel)
        d2 = ((self.all_X[:, None, :] - sel_X[None, :, :]) ** 2).sum(axis=2)
        # RBF kernel weights
        K = np.exp(-self.gamma * d2)
        # Responsibility = sum of weights across all slices
        mass = K.sum(axis=0)
        if mass.sum() <= 0:
            return np.ones(len(combination)) / len(combination)
        return mass / mass.sum()

    def score(self, combination: SliceCombination) -> float:
        """Compute L2 deviation of responsibilities from uniform distribution.

        Args:
            combination: Tuple of slice identifiers forming the selection.

        Returns:
            L2 norm of (responsibilities - uniform). Zero indicates perfectly
            balanced coverage; higher values indicate imbalance.
        """
        r = self._responsibilities(combination)
        u = np.ones_like(r) / len(r)
        return float(np.linalg.norm(r - u))
