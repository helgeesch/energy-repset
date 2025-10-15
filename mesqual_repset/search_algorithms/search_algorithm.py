from __future__ import annotations

from abc import ABC, abstractmethod

from mesqual_repset.context import ProblemContext
from mesqual_repset.results import RepSetResult


class SearchAlgorithm(ABC):
    """Base class for all selection search algorithms (Pillar A).

    Defines the interface for algorithms that find optimal representative subsets.
    The algorithm's sole responsibility is to take a problem context and find the
    best selection of k items based on its internal logic and objective function.

    Different workflow types implement this protocol differently:
    - Generate-and-Test: Generates candidates, evaluates with ObjectiveSet, selects best
    - Constructive: Builds solution iteratively (e.g., k-means clustering)
    - Direct Optimization: Formulates and solves as single optimization problem (e.g., MILP)

    Examples:
        >>> class SimpleExhaustiveSearch(SearchAlgorithm):
        ...     def __init__(self, objective_set: ObjectiveSet, selection_policy: SelectionPolicy):
        ...         self.objective_set = objective_set
        ...         self.selection_policy = selection_policy
        ...
        ...     def find_selection(self, context: ProblemContext, k: int) -> RepSetResult:
        ...         # Generate all k-combinations
        ...         from itertools import combinations
        ...         all_combos = list(combinations(context.slicer.slices, k))
        ...
        ...         # Score each combination
        ...         scored_combos = []
        ...         for combo in all_combos:
        ...             scores = self.objective_set.evaluate(context, combo)
        ...             scored_combos.append((combo, scores))
        ...
        ...         # Select best according to policy
        ...         best_combo, best_scores = self.selection_policy.select(scored_combos)
        ...
        ...         return RepSetResult(
        ...             selection=best_combo,
        ...             weights={s: 1/k for s in best_combo},
        ...             scores=best_scores
        ...         )
        ...
        >>> algorithm = SimpleExhaustiveSearch(objective_set, policy)
        >>> result = algorithm.find_selection(context, k=4)
        >>> print(result.selection)  # e.g., (0, 3, 6, 9) - selected slice IDs
    """
    @abstractmethod
    def find_selection(self, context: ProblemContext, k: int) -> RepSetResult:
        """Find the best subset of k representative periods.

        Args:
            context: The problem context with df_features populated (feature
                engineering must be run before calling this method).
            k: The number of representative periods to select.

        Returns:
            A RepSetResult containing the selected slice identifiers, their
            representation weights, and objective scores.
        """
        ...
