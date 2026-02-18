from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterator, Sequence, Tuple, Hashable, Protocol, List, Any
import itertools
import math

from .combination_generator import CombinationGenerator

if TYPE_CHECKING:
    from ..types import SliceCombination


class ExhaustiveCombiGen(CombinationGenerator):
    """Generate all k-combinations of the candidate slices.

    This generator produces every possible k-element subset using
    itertools.combinations. It is suitable for small problem sizes where
    the total number of combinations (n choose k) is computationally feasible.

    Args:
        k: Number of elements in each combination.

    Attributes:
        k: Number of elements per combination.

    Note:
        The count is computed via binomial coefficient (n choose k) and matches
        the number of yielded combinations exactly.

    Examples:
        Generate all 3-month combinations from a year:

        >>> from energy_repset.combi_gens import ExhaustiveCombiGen
        >>> import pandas as pd
        >>>
        >>> months = [pd.Period('2024-01', 'M'), pd.Period('2024-02', 'M'),
        ...           pd.Period('2024-03', 'M'), pd.Period('2024-04', 'M')]
        >>> generator = ExhaustiveCombiGen(k=3)
        >>> generator.count(months)  # 4 choose 3
            4
        >>> list(generator.generate(months))
            [(Period('2024-01', 'M'), Period('2024-02', 'M'), Period('2024-03', 'M')),
             (Period('2024-01', 'M'), Period('2024-02', 'M'), Period('2024-04', 'M')),
             (Period('2024-01', 'M'), Period('2024-03', 'M'), Period('2024-04', 'M')),
             (Period('2024-02', 'M'), Period('2024-03', 'M'), Period('2024-04', 'M'))]
    """

    def __init__(self, k: int) -> None:
        """Initialize exhaustive generator with target combination size.

        Args:
            k: Number of elements in each combination.
        """
        self.k = k

    def generate(self, unique_slices: Sequence[Hashable]) -> Iterator[SliceCombination]:
        """Generate all k-combinations using itertools.combinations.

        Args:
            unique_slices: Sequence of candidate slice labels.

        Yields:
            All possible k-element tuples from unique_slices.
        """
        yield from itertools.combinations(unique_slices, self.k)

    def count(self, unique_slices: Sequence[Hashable]) -> int:
        """Count total combinations using binomial coefficient.

        Args:
            unique_slices: Sequence of candidate slice labels.

        Returns:
            n choose k, where n is the number of unique slices.
        """
        return math.comb(len(unique_slices), self.k)

    def combination_is_valid(self, combination: SliceCombination, unique_slices: Sequence[Hashable]) -> bool:
        """Check if combination has exactly k elements from unique_slices.

        Args:
            combination: Tuple of slice labels to validate.
            unique_slices: Sequence of candidate slice labels.

        Returns:
            True if combination has k elements all in unique_slices.
        """
        if len(combination) != self.k:
            return False
        if any(s not in unique_slices for s in unique_slices):
            return False
        return True
