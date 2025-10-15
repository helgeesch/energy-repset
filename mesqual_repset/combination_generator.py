from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterator, Sequence, Tuple, Hashable, Protocol, List, Any
import itertools
import math

if TYPE_CHECKING:
    from .types import SliceCombination


class CombinationGenerator(Protocol):
    """Protocol for generating and counting combinations of candidate slices.

    This protocol defines the interface for combination generators used in
    Generate-and-Test workflows. Implementations determine which k-element
    subsets of candidate slices should be evaluated.

    Attributes:
        k: Number of elements in each combination to generate.

    Examples:
        Implementations must provide three methods:

        >>> from mesqual_repset.combination_generator import ExhaustiveCombinationGenerator
        >>> generator = ExhaustiveCombinationGenerator(k=3)
        >>> slices = ['Jan', 'Feb', 'Mar', 'Apr']
        >>> generator.count(slices)  # Number of combinations
        4
        >>> list(generator.generate(slices))
        [('Jan', 'Feb', 'Mar'), ('Jan', 'Feb', 'Apr'), ('Jan', 'Mar', 'Apr'), ('Feb', 'Mar', 'Apr')]
    """

    k: int

    def generate(self, unique_slices: Sequence[Hashable]) -> Iterator[SliceCombination]:
        """Generate k-combinations from the candidate slices.

        Args:
            unique_slices: Sequence of candidate slice labels.

        Yields:
            Tuples of length k representing candidate selections.
        """
        ...

    def count(self, unique_slices: Sequence[Hashable]) -> int:
        """Count the total number of combinations that will be generated.

        Args:
            unique_slices: Sequence of candidate slice labels.

        Returns:
            Total number of k-combinations.
        """
        ...

    def combination_is_valid(self, combination: SliceCombination, unique_slices: Sequence[Hashable]) -> bool:
        """Check if a combination is valid according to generator constraints.

        Args:
            combination: Tuple of slice labels to validate.
            unique_slices: Sequence of candidate slice labels.

        Returns:
            True if the combination satisfies the generator's constraints.
        """
        ...


class ExhaustiveCombinationGenerator(CombinationGenerator):
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

        >>> from mesqual_repset.combination_generator import ExhaustiveCombinationGenerator
        >>> import pandas as pd
        >>>
        >>> months = [pd.Period('2024-01', 'M'), pd.Period('2024-02', 'M'),
        ...           pd.Period('2024-03', 'M'), pd.Period('2024-04', 'M')]
        >>> generator = ExhaustiveCombinationGenerator(k=3)
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


class GroupQuotaCombinationGenerator(CombinationGenerator):
    """Generate combinations that respect exact quotas per group.

    This generator enforces that selections contain a specific number of elements
    from each group. It is useful for ensuring balanced representation across
    categories (e.g., seasons, must-have periods, etc.).

    Args:
        k: Total number of elements in each combination. Must equal sum of group quotas.
        slice_to_group_mapping: Mapping from each candidate slice to its group label.
        group_quota: Mapping from group label to the required count in the selection.

    Attributes:
        k: Number of elements per combination.
        group_of: Mapping from slice to group.
        group_quota: Required count per group.

    Raises:
        ValueError: If sum of group quotas does not equal k.

    Note:
        Use this to enforce constraints like "exactly one month per season" or
        "2 must-have periods plus 2 optional periods".

    Examples:
        Example 1 - Seasonal constraints (one month per season):

        >>> from mesqual_repset.combination_generator import GroupQuotaCombinationGenerator
        >>> import pandas as pd
        >>>
        >>> # Define months and their seasons
        >>> months = [pd.Period(f'2024-{i:02d}', 'M') for i in range(1, 13)]
        >>> season_map = {}
        >>> for month in months:
        ...     if month.month in [12, 1, 2]: season_map[month] = 'winter'
        ...     elif month.month in [3, 4, 5]: season_map[month] = 'spring'
        ...     elif month.month in [6, 7, 8]: season_map[month] = 'summer'
        ...     else: season_map[month] = 'fall'
        >>>
        >>> # Select 4 months, one per season
        >>> generator = GroupQuotaCombinationGenerator(
        ...     k=4,
        ...     slice_to_group_mapping=season_map,
        ...     group_quota={'winter': 1, 'spring': 1, 'summer': 1, 'fall': 1}
        ... )
        >>> generator.count(months)  # 3 * 3 * 3 * 3 = 81 combinations
        81

        Example 2 - Optional and must-have categories:

        >>> # Force specific periods to be included
        >>> months = [pd.Period(f'2024-{i:02d}', 'M') for i in range(1, 13)]
        >>> group_mapping = {p: 'optional' for p in months}
        >>> group_mapping[pd.Period('2024-01', 'M')] = 'must'
        >>> group_mapping[pd.Period('2024-12', 'M')] = 'must'
        >>>
        >>> # Select 4 total: 2 must-have + 2 optional
        >>> generator = GroupQuotaCombinationGenerator(
        ...     k=4,
        ...     slice_to_group_mapping=group_mapping,
        ...     group_quota={'optional': 2, 'must': 2}
        ... )
        >>> # All combinations will include Jan and Dec plus 2 from the other 10
        >>> generator.count(months)  # 1 * 45 = 45 combinations
        45
    """

    def __init__(
            self,
            k: int,
            slice_to_group_mapping: Dict[Hashable, Hashable],
            group_quota: Dict[Hashable, int]
    ) -> None:
        """Initialize generator with group quotas.

        Args:
            k: Total number of elements in each combination.
            slice_to_group_mapping: Mapping from slice to its group label.
            group_quota: Required count per group.

        Raises:
            ValueError: If sum of group quotas does not equal k.
        """
        self.k = k
        self.group_of = slice_to_group_mapping
        self.group_quota = group_quota

        # Validate that quotas sum to k
        if sum(group_quota.values()) != k:
            raise ValueError(f"Sum of group quotas ({sum(group_quota.values())}) must equal k ({k})")

    def generate(self, unique_slices: Sequence[Hashable]) -> Iterator[SliceCombination]:
        """Generate combinations respecting group quotas.

        Args:
            unique_slices: Sequence of candidate slice labels.

        Yields:
            Tuples of length k where each group contributes exactly its quota.
        """
        groups: Dict[Hashable, List[Hashable]] = {}
        for c in unique_slices:
            g = self.group_of[c]
            groups.setdefault(g, []).append(c)
        per_group_lists = []
        for g, q in self.group_quota.items():
            per_group_lists.append(list(itertools.combinations(groups.get(g, []), q)))
        for tpl in itertools.product(*per_group_lists):
            flat = tuple(itertools.chain.from_iterable(tpl))
            if len(flat) == self.k:
                yield flat

    def count(self, unique_slices: Sequence[Hashable]) -> int:
        """Count total combinations respecting group quotas.

        Args:
            unique_slices: Sequence of candidate slice labels.

        Returns:
            Product of binomial coefficients across all groups. For each group
            with n members and quota q, contributes C(n, q) to the product.
        """
        groups: Dict[Hashable, List[Hashable]] = {}
        for c in unique_slices:
            g = self.group_of[c]
            groups.setdefault(g, []).append(c)
        total = 1
        for g, q in self.group_quota.items():
            n = len(groups.get(g, []))
            total *= math.comb(n, q)
        return total

    def combination_is_valid(self, combination: SliceCombination, unique_slices: Sequence[Hashable]) -> bool:
        """Check if combination satisfies group quotas.

        Args:
            combination: Tuple of slice labels to validate.
            unique_slices: Sequence of candidate slice labels.

        Returns:
            True if combination has exactly k elements with each group contributing
            its required quota.
        """
        if len(combination) != self.k:
            return False
        if any(s not in unique_slices for s in unique_slices):
            return False

        group_count = {group_name: 0 for group_name in self.group_quota.keys()}
        for slice in combination:
            group_count[self.group_of[slice]] +=1
        if not all(group_count[g] == self.group_quota[g] for g in self.group_quota.keys()):
            return False

        return True
