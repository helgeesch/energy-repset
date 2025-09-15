from __future__ import annotations

from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from ..context import ProblemContext
    from ..types import ScoreComponentDirection, SliceCombination


class ScoreComponent(Protocol):
    """
    Interface for a single metric used in the objective function..

    Attributes
    ----------
    name
        Unique name of the component.
    direction
        Whether the score should be maximized or minimized

    Methods
    -------
    prepare(context)
        Precompute state needed by the component before scoring selections.
    score(combination)
        Compute the component value for a given selection of slice labels.
    """
    name: str
    direction: ScoreComponentDirection
    def prepare(self, context: ProblemContext) -> None:...
    def score(self, combination: SliceCombination) -> float:...
