from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional
from abc import ABC
import pandas as pd

from .search_algorithm import SearchAlgorithm
from ..results import RepSetResult

if TYPE_CHECKING:
    from ..combi_gens import CombinationGenerator
    from ..context import ProblemContext
    from ..objectives import ObjectiveSet
    from ..selection_policies import SelectionPolicy
    from ..types import SliceCombination


class ObjectiveDrivenSearchAlgorithm(SearchAlgorithm, ABC):
    """Base class for search algorithms guided by external objective functions.

    Provides a common structure for algorithms that rely on a user-defined
    ObjectiveSet to score candidates and a SelectionPolicy to choose the best.
    This pattern separates the search strategy from the objective function,
    enabling flexible algorithm design.

    Examples:

        >>> from energy_repset.objectives import ObjectiveSet, ObjectiveSpec
        >>> from energy_repset.score_components import WassersteinFidelity
        >>> from energy_repset.selection_policies import WeightedSumPolicy
        >>> objectives = ObjectiveSet({
        ...     'wasserstein': (1.0, WassersteinFidelity()),
        ... })
        >>> policy = WeightedSumPolicy()
        >>> # See ObjectiveDrivenCombinatorialSearchAlgorithm for concrete usage
    """
    def __init__(
            self,
            objective_set: ObjectiveSet,
            selection_policy: SelectionPolicy,
    ):
        """Initialize objective-driven search algorithm.

        Args:
            objective_set: Collection of score components defining quality metrics.
            selection_policy: Strategy for selecting best combination from scored
                candidates (e.g., weighted sum, Pareto dominance).
        """
        self.objective_set = objective_set
        self.selection_policy = selection_policy

    def _build_result(
        self,
        context: ProblemContext,
        winning_combination: SliceCombination,
        evaluations_df: pd.DataFrame,
        extra_diagnostics: Optional[Dict[str, Any]] = None,
    ) -> RepSetResult:
        """Construct a RepSetResult for an objective-driven search.

        Shared by all generate-and-test algorithms to avoid repeating the
        same result-assembly logic (representative extraction, score
        evaluation, diagnostics packaging).

        Args:
            context: Problem context used during search.
            winning_combination: The selected slice combination.
            evaluations_df: DataFrame of all evaluated combinations and their
                scores.
            extra_diagnostics: Additional entries to include in the result's
                diagnostics dict (e.g. ``generation_history`` for GA).

        Returns:
            Fully populated RepSetResult with ``weights=None`` (to be filled
            by an external ``RepresentationModel``).
        """
        slice_labels = context.slicer.labels_for_index(context.df_raw.index)
        diagnostics: Dict[str, Any] = {"evaluations_df": evaluations_df}
        if extra_diagnostics:
            diagnostics.update(extra_diagnostics)

        return RepSetResult(
            context=context,
            selection_space="subset",
            selection=winning_combination,
            scores=self.objective_set.evaluate(winning_combination, context),
            representatives={
                s: context.df_raw.iloc[slice_labels == s]
                for s in winning_combination
            },
            diagnostics=diagnostics,
        )


class ObjectiveDrivenCombinatorialSearchAlgorithm(ObjectiveDrivenSearchAlgorithm):
    """Generate-and-test search using a combination generator (Workflow Type 1).

    Generates candidate combinations using a CombinationGenerator, scores each
    with the ObjectiveSet, and selects the best according to the SelectionPolicy.
    This is the canonical implementation of the Generate-and-Test workflow.

    Supports exhaustive search (all k-combinations) and constrained generation
    (e.g., seasonal quotas). Displays progress with tqdm and stores all
    evaluations in diagnostics for analysis.

    Examples:

        >>> from energy_repset.objectives import ObjectiveSet
        >>> from energy_repset.combi_gens import ExhaustiveCombiGen
        >>> from energy_repset.selection_policies import WeightedSumPolicy
        >>> from energy_repset.score_components import WassersteinFidelity, CorrelationFidelity
        >>> objectives = ObjectiveSet({
        ...     'wasserstein': (1.0, WassersteinFidelity()),
        ...     'correlation': (0.5, CorrelationFidelity())
        ... })
        >>> policy = WeightedSumPolicy()
        >>> generator = ExhaustiveCombiGen(k=4)
        >>> algorithm = ObjectiveDrivenCombinatorialSearchAlgorithm(
        ...     objective_set=objectives,
        ...     selection_policy=policy,
        ...     combination_generator=generator
        ... )
        >>> algorithm.k
        4
    """
    def __init__(
            self,
            objective_set: ObjectiveSet,
            selection_policy: SelectionPolicy,
            combination_generator: CombinationGenerator,
    ):
        """Initialize combinatorial search algorithm.

        Args:
            objective_set: Collection of score components defining quality metrics.
            selection_policy: Strategy for selecting the best combination.
            combination_generator: Defines which combinations to evaluate
                (e.g., all combinations, seasonal constraints).
        """
        super().__init__(objective_set, selection_policy)
        self.combination_generator = combination_generator
        self._all_scores_df: pd.DataFrame | None = None

    @property
    def k(self) -> int:
        """Number of representative periods to select."""
        return self.combination_generator.k

    def find_selection(self, context: ProblemContext) -> RepSetResult:
        """Find optimal selection by exhaustively scoring generated combinations.

        Args:
            context: Problem context with df_features populated.

        Returns:
            RepSetResult with the winning selection, scores, representatives,
            and diagnostics containing evaluations_df with all scored combinations.
        """
        from tqdm import tqdm
        import pandas as pd

        self.objective_set.prepare(context)

        unique_slices = context.get_unique_slices()
        iterator = tqdm(
            self.combination_generator.generate(unique_slices),
            desc='Iterating over combinations',
            total=self.combination_generator.count(unique_slices)
        )

        rows = []
        for combi in iterator:
            metrics = self.objective_set.evaluate(combi, context)
            rec = {
                "slices": combi,
                "label": ", ".join(str(s) for s in combi)
            }
            rec.update(metrics)
            rows.append(rec)

        evaluations_df = pd.DataFrame(rows)
        self._all_scores_df = evaluations_df.copy()

        winning_combination = self.selection_policy.select_best(evaluations_df, self.objective_set)
        return self._build_result(context, winning_combination, evaluations_df)

    def get_all_scores(self) -> pd.DataFrame:
        """Return DataFrame of all evaluated combinations with scores.

        Returns:
            DataFrame with columns: slices, label, score_comp_1, score_comp_2, ...

        Raises:
            ValueError: If find_selection() has not been called yet.
        """
        import pandas as pd

        if self._all_scores_df is None:
            raise ValueError("No scores available. Call find_selection() first.")
        return self._all_scores_df.copy()
