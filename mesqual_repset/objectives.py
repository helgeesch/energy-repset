from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Tuple, Any
from dataclasses import dataclass

from .score_components.base_score_component import ScoreComponent

if TYPE_CHECKING:
    from .types import SliceCombination
    from .context import ProblemContext


@dataclass(frozen=True)
class ObjectiveSpec:
    component: ScoreComponent
    weight: float             # >= 0; preference magnitude


class ObjectiveSet:
    """Pillar O: A collection of ScoreComponents used by the search algorithm.

    Metrics-only engine + explicit per-metric preferences (importance).
    Components define direction; weights live here and are non-negative.
    """

    def __init__(
            self,
            weighted_score_components: Dict[str, Tuple[float, ScoreComponent]]
    ) -> None:
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
        for spec in self.weighted_score_components.values():
            spec.component.prepare(context)

    def evaluate(self, combination: SliceCombination, context: ProblemContext) -> Dict[str, float]:
       return {  # TODO: what about the weighting factors?
           spec.component.name: float(spec.component.score(combination))
           for spec in self.weighted_score_components.values()
       }

    def component_meta(self) -> Dict[str, Dict[str, Any]]:
        # direction from component; preference from ObjectiveSet weight
        return {
            spec.component.name: {"direction": spec.component.direction, "pref": float(spec.weight)}
            for spec in self.weighted_score_components.values()
        }
