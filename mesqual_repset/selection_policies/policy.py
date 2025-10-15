from __future__ import annotations

from typing import Literal, Tuple, Hashable, TYPE_CHECKING
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

if TYPE_CHECKING:
    from ..search_algorithms.search_algorithm import SearchAlgorithm
    from ..objectives import ObjectiveSet
    from ..results import RepSetResult


@dataclass(frozen=True)
class PolicyOutcome:
    algorithm: SearchAlgorithm
    selected: RepSetResult
    scores_annotated: pd.DataFrame


class SelectionPolicy(ABC):
    """Base class for selection policies that choose the best combination.

    Selection policies define the strategy for choosing the winning combination
    from a set of scored candidates. Different policies implement different
    trade-offs between competing objectives (e.g., weighted sum vs. Pareto).

    This is a key component of the Generate-and-Test workflow where the
    SearchAlgorithm generates candidates, the ObjectiveSet scores them, and
    the SelectionPolicy picks the winner.

    Examples:
        >>> # See WeightedSumPolicy and ParetoUtopiaPolicy for concrete examples
        >>> class SimpleMinPolicy(SelectionPolicy):
        ...     def select_best(self, evaluations_df: pd.DataFrame, objective_set: ObjectiveSet):
        ...         # Just pick the row with minimum of first objective
        ...         first_obj = list(objective_set.component_meta().keys())[0]
        ...         best_row = evaluations_df.loc[evaluations_df[first_obj].idxmin()]
        ...         return tuple(best_row['slices'])
    """

    @abstractmethod
    def select_best(self, evaluations_df: pd.DataFrame, objective_set: ObjectiveSet) -> Tuple[Hashable, ...]:
        """Select the best combination from scored candidates.

        Args:
            evaluations_df: DataFrame where each row is a candidate combination
                with columns 'slices' (the combination tuple) and score columns
                for each objective component.
            objective_set: Provides metadata about score components (direction,
                weights, etc.) needed for selection logic.

        Returns:
            Tuple of slice identifiers representing the winning combination.
        """
        ...
