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

    @abstractmethod
    def select_best(self, evaluations_df: pd.DataFrame, objective_set: ObjectiveSet) -> Tuple[Hashable, ...]:
        """
        Select best solution from evaluation results.

        Args:
            evaluations_df: DataFrame with columns 'slices' and objective metrics
            objective_set: ObjectiveSet providing component metadata

        Returns:
            Tuple representing the best selection
        """
        ...
