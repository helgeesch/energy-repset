from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Dict, List
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from ..context import ProblemContext
    from ..types import ScoreComponentDirection, SliceCombination


class ScoreComponent(ABC):
    """Protocol for a single metric used in evaluating candidate selections.

    ScoreComponents are the building blocks of the ObjectiveSet. Each component
    computes a scalar score measuring how well a candidate selection performs
    on a specific criterion (e.g., distribution fidelity, diversity, balance).

    Implementations must define:
        - A unique name identifying the component
        - An optimization direction ('min' or 'max')
        - A prepare() method to precompute reference data from the full context
        - A score() method to evaluate a candidate selection

    Attributes:
        name: Unique identifier for this component.
        direction: Optimization direction, either "min" or "max".

    Examples:
        Implementing a simple score component:

        >>> from energy_repset.score_components.base_score_component import ScoreComponent
        >>> from energy_repset.context import ProblemContext
        >>> from energy_repset.types import SliceCombination
        >>> import numpy as np
        >>>
        >>> class SimpleMeanDeviation(ScoreComponent):
        ...     def __init__(self):
        ...         self.name = "mean_deviation"
        ...         self.direction = "min"
        ...         self.full_mean = None
        ...
        ...     def prepare(self, context: ProblemContext) -> None:
        ...         '''Compute reference mean from full dataset.'''
        ...         self.full_mean = context.df_raw.mean().mean()
        ...
        ...     def score(self, combination: SliceCombination) -> float:
        ...         '''Measure deviation from reference mean.'''
        ...         # Get data for selection and compute deviation
        ...         # (implementation details omitted)
        ...         return abs(selection_mean - self.full_mean)
    """
    name: str
    direction: ScoreComponentDirection

    @abstractmethod
    def prepare(self, context: ProblemContext) -> None:
        """Precompute state needed before scoring selections.

        This method is called once before evaluating any combinations. Use it
        to compute reference statistics, duration curves, or other data derived
        from the full dataset that will be compared against selections.

        Args:
            context: ProblemContext containing raw data, features, and metadata.

        Note:
            This method should store computed state as instance attributes for
            use in score().
        """
        ...

    @abstractmethod
    def score(self, combination: SliceCombination) -> float:
        """Compute the component score for a candidate selection.

        Args:
            combination: Tuple of slice labels forming the candidate selection.

        Returns:
            Scalar score. Lower is better for direction='min', higher is better
            for direction='max'.

        Note:
            This method is called many times during search. Precompute expensive
            operations in prepare() to avoid redundant calculations.
        """
        ...

    @staticmethod
    def _default_weight_normalization(
            weights: Optional[Dict[str, float]],
            keys: List[str]
    ) -> Dict[str, float]:
        """Normalize weight dictionary to match actual keys.

        Normalize weights: None → equal (1.0), specified → use values (missing get 0.0)

        Args:
            weights: User-specified weights, or None for equal weights.
            keys: Actual keys that need weights (e.g., variable names, feature names).

        Returns:
            Dictionary mapping each key to its weight:
            - If weights is None: all keys get 1.0 (equal weights)
            - If weights is provided: specified keys get their value, missing keys get 0.0

        Examples:

            >>> # No weights specified - equal weights
            >>> normalize_weights(None, ['demand', 'solar', 'wind'])
            {'demand': 1.0, 'solar': 1.0, 'wind': 1.0}

            >>> # Partial weights - missing get 0.0
            >>> normalize_weights({'demand': 2.0, 'solar': 1.5}, ['demand', 'solar', 'wind'])
            {'demand': 2.0, 'solar': 1.5, 'wind': 0.0}

            >>> # All weights specified
            >>> normalize_weights({'demand': 2.0, 'solar': 1.0, 'wind': 0.5}, ['demand', 'solar'])
            {'demand': 2.0, 'solar': 1.0, 'wind': 0.5}
        """
        if weights is None:
            return {key: 1.0 for key in keys}
        return {key: weights.get(key, 0.0) for key in keys}