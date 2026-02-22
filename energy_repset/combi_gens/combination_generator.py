from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterator, Sequence, Tuple, Hashable, Protocol, List, Any

if TYPE_CHECKING:
    from ..types import SliceCombination


class CombinationGenerator(Protocol):
    """Protocol for generating, counting, and validating slice combinations.

    This protocol serves two complementary roles depending on the search
    algorithm that uses it:

    **Generator role** — used by exhaustive search algorithms that enumerate
    all valid candidates.  These algorithms call ``generate()`` and
    ``count()`` to iterate over the full search space.

    **Constraint-validator role** — used by stochastic algorithms (genetic
    algorithm, random sampling) that construct candidates on their own.
    These algorithms only call ``combination_is_valid()`` and read ``k``;
    they never call ``generate()`` or ``count()``.

    Both roles express the *same* constraint set: every combination yielded
    by ``generate()`` must satisfy ``combination_is_valid()``, and every
    combination that passes ``combination_is_valid()`` would be included in
    the output of ``generate()``.  Implementations must maintain this
    invariant.

    Attributes:
        k: Number of elements in each combination to generate.
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