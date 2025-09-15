from __future__ import annotations

from abc import ABC, abstractmethod

from mesqual_repset.context import ProblemContext
from mesqual_repset.results import RepSetResult


class SearchAlgorithm(ABC):
    """
    Base class for all selection search algorithms.

    Its sole responsibility is to take a problem context and find the best
    selection of k items based on its own internal logic.
    """
    @abstractmethod
    def find_selection(self, context: ProblemContext, k: int) -> RepSetResult:
        """
        Finds the best subset of k items from the context.
        """
        ...
