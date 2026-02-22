from __future__ import annotations

from abc import ABC, abstractmethod

from ..context import ProblemContext
from ..results import RepSetResult


class SearchAlgorithm(ABC):
    """Base class for all selection search algorithms (Pillar A).

    Defines the interface for algorithms that find optimal representative subsets.
    The algorithm's sole responsibility is to take a problem context and find the
    best selection of k items based on its internal logic and objective function.

    Different workflow types implement this protocol differently:

    - **Generate-and-Test**: Generates candidates, evaluates with ObjectiveSet,
      selects best.  Subclass ``ObjectiveDrivenSearchAlgorithm``.
    - **Constructive**: Builds solution iteratively (e.g., hull clustering).
      Subclass ``SearchAlgorithm`` directly.
    - **Direct Optimization**: Formulates and solves as single optimization
      problem (e.g., MILP).  Subclass ``SearchAlgorithm`` directly.

    Subclasses must implement ``find_selection`` and the ``k`` property.

    Examples:

        >>> class SimpleExhaustiveSearch(SearchAlgorithm):
        ...     def __init__(self, objective_set, selection_policy, k):
        ...         self.objective_set = objective_set
        ...         self.selection_policy = selection_policy
        ...         self._k = k
        ...
        ...     @property
        ...     def k(self) -> int:
        ...         return self._k
        ...
        ...     def find_selection(self, context: ProblemContext) -> RepSetResult:
        ...         from itertools import combinations
        ...         all_combis = list(combinations(context.slicer.slices, self.k))
        ...         scored_combis = []
        ...         for combi in all_combis:
        ...             scores = self.objective_set.evaluate(context, combi)
        ...             scored_combis.append((combi, scores))
        ...         best_combi, best_scores = self.selection_policy.select(scored_combis)
        ...         return RepSetResult(
        ...             selection=best_combi,
        ...             weights={s: 1/self.k for s in best_combi},
        ...             scores=best_scores,
        ...         )
        ...
        >>> algorithm = SimpleExhaustiveSearch(objective_set, policy, k=4)
        >>> algorithm.k
        4
    """

    @property
    @abstractmethod
    def k(self) -> int:
        """Number of representative items this algorithm selects.

        For generate-and-test algorithms this typically delegates to the
        combination generator.  Constructive algorithms store it directly.

        Note:
            The exact semantics of *k* can vary between algorithm families.
            For most algorithms ``len(result.selection) == k``, but for
            algorithms like ``SnippetSearch`` each selected item is a
            multi-day subsequence, so *k* counts subsequences rather than
            individual time slices.
        """
        ...

    @abstractmethod
    def find_selection(self, context: ProblemContext) -> RepSetResult:
        """Find the best subset of k representative periods.

        Args:
            context: The problem context with df_features populated (feature
                engineering must be run before calling this method).

        Returns:
            A RepSetResult containing the selected slice identifiers, their
            representation weights, and objective scores.
        """
        ...
