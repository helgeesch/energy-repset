from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Hashable

from .representation import RepresentationModel

if TYPE_CHECKING:
    from ..context import ProblemContext
    from ..types import SliceCombination


class UniformRepresentationModel(RepresentationModel):
    """Assigns equal weight to all selected representatives."""

    def fit(self, context: ProblemContext):
        """No fitting is required for the uniform model."""
        pass

    def weigh(self, combination: SliceCombination) -> Dict[Hashable, float]:
        if not combination:
            return {}
        weight = 1.0 / len(combination)
        return {label: weight for label in combination}
