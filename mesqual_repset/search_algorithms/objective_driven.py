from __future__ import annotations

from typing import TYPE_CHECKING
from abc import ABC

from .search_algorithm import SearchAlgorithm
from ..results import RepSetResult

if TYPE_CHECKING:
    from ..combination_generator import CombinationGenerator
    from ..context import ProblemContext
    from ..objectives import ObjectiveSet
    from ..selection_policies import SelectionPolicy


class ObjectiveDrivenSearchAlgorithm(SearchAlgorithm, ABC):
    """
    A specialized base class for search algorithms that are guided by an
    external, user-defined objective function and selection policy.
    """
    def __init__(
            self,
            objective_set: ObjectiveSet,
            selection_policy: SelectionPolicy,
    ):
        self.objective_set = objective_set
        self.selection_policy = selection_policy


class ObjectiveDrivenCombinatorialSearchAlgorithm(ObjectiveDrivenSearchAlgorithm):
    def __init__(
            self,
            objective_set: ObjectiveSet,
            selection_policy: SelectionPolicy,
            combination_generator: CombinationGenerator,
    ):
        super().__init__(objective_set, selection_policy)
        self.combination_generator = combination_generator

    def find_selection(self, context: ProblemContext, k: int) -> RepSetResult:
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
        for combo in iterator:
            metrics = self.objective_set.evaluate(combo, context)
            rec = {
                "slices": combo,
                "label": ", ".join(str(s) for s in combo)
            }
            rec.update(metrics)
            rows.append(rec)

        evaluations_df = pd.DataFrame(rows)
        winning_combination = self.selection_policy.select_best(evaluations_df, self.objective_set)
        result = RepSetResult(
            context=context,
            selection_space='subset',
            selection=winning_combination,
            scores=self.objective_set.evaluate(winning_combination, context),
            representatives={s: context.df_raw.loc[s] for s in winning_combination},
            diagnostics={'evaluations_df': evaluations_df}
        )
        return result
