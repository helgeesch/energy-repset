from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Tuple, Any
from dataclasses import dataclass

from .score_components.base_score_component import ScoreComponent

if TYPE_CHECKING:
    from .types import SliceCombination
    from .context import ProblemContext


@dataclass(frozen=True)
class ObjectiveSpec:
    """Specification for a single score component with its preference weight.

    Attributes:
        component: The ScoreComponent that computes the metric.
        weight: Non-negative weight indicating the component's importance (>= 0).
    """
    component: ScoreComponent
    weight: float


class ObjectiveSet:
    """Pillar O: A collection of weighted score components for evaluating selections.

    This class holds multiple ScoreComponents, each with a weight indicating its
    importance. Components define their optimization direction (min/max), while
    weights specify preference magnitude. The ObjectiveSet prepares all components
    with context data and evaluates candidate selections.

    Attributes:
        weighted_score_components: Dictionary mapping component names to ObjectiveSpec
            instances containing the component and its weight.

    Examples:
        Create an objective set with multiple fidelity metrics:

        >>> from energy_repset.objectives import ObjectiveSet
        >>> from energy_repset.score_components import (
        ...     WassersteinFidelity, CorrelationFidelity
        ... )
        >>>
        >>> objective_set = ObjectiveSet({
        ...     'wasserstein': (0.5, WassersteinFidelity()),
        ...     'correlation': (0.5, CorrelationFidelity()),
        ... })

        With variable-specific weights:

        >>> wass = WassersteinFidelity(variable_weights={'demand': 2.0, 'solar': 1.0})
        >>> objective_set = ObjectiveSet({
        ...     'wasserstein': (1.0, wass),
        ... })
    """

    def __init__(
            self,
            weighted_score_components: Dict[str, Tuple[float, ScoreComponent]]
    ) -> None:
        """Initialize ObjectiveSet with weighted score components.

        Args:
            weighted_score_components: Dictionary mapping component names to
                tuples of (weight, ScoreComponent). Weights must be non-negative.

        Raises:
            ValueError: If any weight is negative or if any component lacks a
                'direction' attribute set to 'min' or 'max'.
        """
        self.weighted_score_components: Dict[str, ObjectiveSpec] = {
            name: s if isinstance(s, ObjectiveSpec) else ObjectiveSpec(component=s[1], weight=float(s[0]))
            for name, s in weighted_score_components.items()
        }
        for s in self.weighted_score_components.values():
            if s.weight < 0:
                raise ValueError(f"Weight for {s.component.name} must be >= 0.")
            if getattr(s.component, "direction", None) not in ("min", "max"):
                raise ValueError(f"Component {s.component.name} must declare direction 'min' or 'max'.")

    def prepare(self, context: ProblemContext) -> None:
        """Prepare all score components with context data.

        This method calls prepare() on each component to allow pre-computation
        of reference statistics, duration curves, etc.

        Args:
            context: ProblemContext containing raw data and features.
        """
        for spec in self.weighted_score_components.values():
            spec.component.prepare(context)

    def evaluate(self, combination: SliceCombination, context: ProblemContext) -> Dict[str, float]:
        """Evaluate a candidate selection across all score components.

        Args:
            combination: Tuple of slice labels forming the candidate selection.
            context: ProblemContext for accessing data.

        Returns:
            Dictionary mapping component names to their unweighted scores.

        Note:
            Returns raw scores from components. Weights are applied by SelectionPolicy
            during the selection process.
        """
        return {
           spec.component.name: float(spec.component.score(combination))
           for spec in self.weighted_score_components.values()
       }

    def component_meta(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all components.

        Returns:
            Dictionary mapping component names to their metadata containing:
                - 'direction': Optimization direction ('min' or 'max')
                - 'pref': Preference weight (>= 0)
        """
        return {
            spec.component.name: {"direction": spec.component.direction, "pref": float(spec.weight)}
            for spec in self.weighted_score_components.values()
        }
