from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterator, Sequence, Tuple, Hashable, Protocol, List, Any

if TYPE_CHECKING:
    from ..types import SliceCombination


class CombinationGenerator(Protocol):
    """Protocol for generating and counting combinations of candidate slices.

    This protocol defines the interface for combination generators used in
    Generate-and-Test workflows. Implementations determine which k-element
    subsets of candidate slices should be evaluated.

    Attributes:
        k: Number of elements in each combination to generate.

    Examples:
        Implementations must provide three methods:

        >>> from mesqual_repset.combination_generators import ExhaustiveCombinationGenerator
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