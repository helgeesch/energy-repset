from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Hashable

from .representation import RepresentationModel

if TYPE_CHECKING:
    from ..context import ProblemContext
    from ..types import SliceCombination


class UniformRepresentationModel(RepresentationModel):
    """Assigns equal weights to all selected representatives.

    The simplest representation model where each selected period gets weight
    1/k. This is appropriate when you want each representative to contribute
    equally to downstream modeling, regardless of how many original periods
    it represents.

    Examples:

        >>> model = UniformRepresentationModel()
        >>> model.fit(context)
        >>> weights = model.weigh((0, 3, 6, 9))
        >>> print(weights)
            {0: 0.25, 3: 0.25, 6: 0.25, 9: 0.25}

        >>> # For yearly data with k=4 months, each month represents ~91 days
        >>> # Weights sum to 1.0 for normalized analysis
    """

    def fit(self, context: ProblemContext):
        """No fitting required for uniform weighting.

        Args:
            context: Problem context (unused but required by protocol).
        """
        pass

    def weigh(self, combination: SliceCombination) -> Dict[Hashable, float]:
        """Calculate uniform weights (1/k for each selected period).

        Args:
            combination: Tuple of selected slice identifiers.

        Returns:
            Dictionary mapping each slice ID to its weight (1/k).
        """
        if not combination:
            return {}
        weight = 1.0 / len(combination)
        return {label: weight for label in combination}
