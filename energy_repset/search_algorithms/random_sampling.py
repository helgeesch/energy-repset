"""Random sampling search algorithm for baseline benchmarking."""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import numpy as np
import pandas as pd

from .objective_driven import ObjectiveDrivenSearchAlgorithm
from ..results import RepSetResult

if TYPE_CHECKING:
    from ..combi_gens import CombinationGenerator
    from ..context import ProblemContext
    from ..objectives import ObjectiveSet
    from ..selection_policies import SelectionPolicy
    from ..types import SliceCombination


class RandomSamplingSearch(ObjectiveDrivenSearchAlgorithm):
    """Generate-and-test search using random sampling.

    Generates ``n_samples`` random valid k-combinations, evaluates each with
    the ObjectiveSet, and selects the best according to the SelectionPolicy.
    Useful as a cheap baseline for benchmarking more sophisticated algorithms
    (e.g. genetic algorithms).

    Each random combination is validated against the CombinationGenerator's
    constraints via rejection sampling. Duplicate combinations are discarded.

    Args:
        objective_set: Collection of score components defining quality metrics.
        selection_policy: Strategy for selecting the best combination.
        combination_generator: Defines validity constraints and k.
        n_samples: Number of distinct random combinations to evaluate.
        seed: Random seed for reproducibility.

    Examples:

        >>> from energy_repset import ObjectiveSet, WeightedSumPolicy
        >>> from energy_repset.score_components import WassersteinFidelity
        >>> from energy_repset.combi_gens import ExhaustiveCombiGen
        >>> from energy_repset.search_algorithms import RandomSamplingSearch
        >>> objectives = ObjectiveSet({"wass": (1.0, WassersteinFidelity())})
        >>> algo = RandomSamplingSearch(
        ...     objective_set=objectives,
        ...     selection_policy=WeightedSumPolicy(),
        ...     combination_generator=ExhaustiveCombiGen(k=4),
        ...     n_samples=500,
        ...     seed=42,
        ... )
    """

    def __init__(
        self,
        objective_set: ObjectiveSet,
        selection_policy: SelectionPolicy,
        combination_generator: CombinationGenerator,
        n_samples: int = 1000,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(objective_set, selection_policy)
        self.combination_generator = combination_generator
        self.n_samples = n_samples
        self.seed = seed

    def find_selection(self, context: ProblemContext) -> RepSetResult:
        """Find a selection by evaluating random valid combinations.

        Args:
            context: Problem context with ``df_features`` populated.

        Returns:
            RepSetResult with the best selection, scores, representatives,
            and ``evaluations_df`` in diagnostics.
        """
        from tqdm import tqdm

        self.objective_set.prepare(context)
        unique_slices = context.get_unique_slices()
        k = self.combination_generator.k
        rng = np.random.default_rng(self.seed)

        samples = self._generate_valid_samples(unique_slices, k, rng)

        rows = []
        for combi in tqdm(samples, desc="Evaluating random samples"):
            metrics = self.objective_set.evaluate(combi, context)
            rec = {
                "slices": combi,
                "label": ", ".join(str(s) for s in combi),
            }
            rec.update(metrics)
            rows.append(rec)

        evaluations_df = pd.DataFrame(rows)
        winning_combination = self.selection_policy.select_best(
            evaluations_df, self.objective_set
        )

        slice_labels = context.slicer.labels_for_index(context.df_raw.index)
        return RepSetResult(
            context=context,
            selection_space="subset",
            selection=winning_combination,
            scores=self.objective_set.evaluate(winning_combination, context),
            representatives={
                s: context.df_raw.iloc[slice_labels == s] for s in winning_combination
            },
            diagnostics={"evaluations_df": evaluations_df},
        )

    def _generate_valid_samples(
        self,
        unique_slices: list,
        k: int,
        rng: np.random.Generator,
    ) -> List[SliceCombination]:
        """Generate up to n_samples distinct valid combinations.

        Uses rejection sampling: draw random k-subsets, keep those that pass
        ``combination_generator.combination_is_valid()``. Caps total attempts
        at ``n_samples * 20`` to avoid infinite loops.

        Args:
            unique_slices: Available slice labels.
            k: Number of slices per combination.
            rng: NumPy random generator.

        Returns:
            List of valid, deduplicated SliceCombination tuples.
        """
        seen: set[tuple] = set()
        samples: List[SliceCombination] = []
        max_attempts = self.n_samples * 20
        attempts = 0

        while len(samples) < self.n_samples and attempts < max_attempts:
            attempts += 1
            indices = rng.choice(len(unique_slices), size=k, replace=False)
            combi = tuple(sorted(unique_slices[i] for i in indices))
            if combi in seen:
                continue
            if not self.combination_generator.combination_is_valid(
                combi, unique_slices
            ):
                continue
            seen.add(combi)
            samples.append(combi)

        return samples
