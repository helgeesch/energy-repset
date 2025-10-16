from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterator, Sequence, Hashable
import itertools
import math
import pandas as pd

from .combination_generator import CombinationGenerator

if TYPE_CHECKING:
    from ..types import SliceCombination
    from ..time_slicer import TimeSlicer


class ExhaustiveHierarchicalCombiGen(CombinationGenerator):
    """Generate combinations where child slices are selected in complete parent groups.

    This generator enforces hierarchical selection: child slices (e.g., days) can only
    be selected as complete parent groups (e.g., months). It enables high-resolution
    features (e.g. per-day) while enforcing structural constraints at the parent level (e.g. months).

    Args:
        parent_k: Number of parent groups to select.
        slice_to_parent_mapping: Mapping from each child slice to its parent group.
            Example: {Period('2024-01-01', 'D'): Period('2024-01', 'M'), ...}

    Attributes:
        k: Number of parent groups per combination (same as parent_k for protocol compliance).
        parent_k: Number of parent groups per combination.
        slice_to_parent: Child to parent mapping.

    Note:
        The `generate()` method yields flattened tuples of child slices, but internally
        enforces parent-level constraints. Use the factory method `from_slicers()`
        for automatic parent grouping based on TimeSlicer objects.

    Examples:
        Manual construction with custom grouping:

        >>> from mesqual_repset.combi_gens import ExhaustiveHierarchicalCombiGen
        >>> import pandas as pd
        >>>
        >>> # Define child-to-parent mapping
        >>> slice_to_parent = {
        ...     pd.Period('2024-01-01', 'D'): pd.Period('2024-01', 'M'),
        ...     pd.Period('2024-01-02', 'D'): pd.Period('2024-01', 'M'),
        ...     pd.Period('2024-02-01', 'D'): pd.Period('2024-02', 'M'),
        ...     pd.Period('2024-02-02', 'D'): pd.Period('2024-02', 'M'),
        ... }
        >>>
        >>> # Select 2 months, but combinations contain days
        >>> gen = ExhaustiveHierarchicalCombiGen(
        ...     parent_k=2,
        ...     slice_to_parent_mapping=slice_to_parent
        ... )
        >>> gen.count(list(slice_to_parent.keys()))  # C(2, 2) = 1
            1

        Using factory method with TimeSlicer:

        >>> import pandas as pd
        >>> from mesqual_repset.time_slicer import TimeSlicer
        >>> from mesqual_repset.combi_gens import ExhaustiveHierarchicalCombiGen
        >>>
        >>> dates = pd.date_range('2024-01-01', periods=366, freq='D')
        >>> child_slicer = TimeSlicer(unit='day')
        >>> parent_slicer = TimeSlicer(unit='month')
        >>>
        >>> gen = ExhaustiveHierarchicalCombiGen.from_slicers(
        ...     parent_k=3,
        ...     dt_index=dates,
        ...     child_slicer=child_slicer,
        ...     parent_slicer=parent_slicer
        ... )
        >>> unique_days = child_slicer.unique_slices(dates)
        >>> gen.count(unique_days)  # C(12, 3) = 220 combinations of months
            220
    """

    def __init__(
        self,
        parent_k: int,
        slice_to_parent_mapping: Dict[Hashable, Hashable]
    ) -> None:
        """Initialize hierarchical generator with child-to-parent mapping.

        Args:
            parent_k: Number of parent groups to select.
            slice_to_parent_mapping: Dict mapping each child slice to its parent.
        """
        self.parent_k = parent_k
        self.k = parent_k  # For CombinationGenerator protocol compliance
        self.slice_to_parent = slice_to_parent_mapping

    @classmethod
    def from_slicers(
        cls,
        parent_k: int,
        dt_index: pd.DatetimeIndex,
        child_slicer: TimeSlicer,
        parent_slicer: TimeSlicer
    ) -> ExhaustiveHierarchicalCombiGen:
        """Factory method to create generator from child and parent TimeSlicer objects.

        Args:
            parent_k: Number of parent groups to select.
            dt_index: DatetimeIndex of the time series data.
            child_slicer: TimeSlicer defining child slice granularity (e.g., daily).
            parent_slicer: TimeSlicer defining parent slice granularity (e.g., monthly).

        Returns:
            ExhaustiveHierarchicalCombinationGenerator with auto-constructed mappings.

        Examples:
            Select 4 months from a year of daily data:

            >>> import pandas as pd
            >>> from mesqual_repset.time_slicer import TimeSlicer
            >>> from mesqual_repset.combi_gens import ExhaustiveHierarchicalCombiGen
            >>>
            >>> dates = pd.date_range('2024-01-01', periods=366, freq='D')
            >>> child_slicer = TimeSlicer(unit='day')
            >>> parent_slicer = TimeSlicer(unit='month')
            >>>
            >>> gen = ExhaustiveHierarchicalCombiGen.from_slicers(
            ...     parent_k=4,
            ...     dt_index=dates,
            ...     child_slicer=child_slicer,
            ...     parent_slicer=parent_slicer
            ... )
            >>> gen.count(child_slicer.unique_slices(dates))  # C(12, 4) = 495
            495
        """
        child_labels = child_slicer.labels_for_index(dt_index)
        parent_labels = parent_slicer.labels_for_index(dt_index)

        slice_to_parent = {}
        for child, parent in zip(child_labels, parent_labels):
            slice_to_parent[child] = parent

        unique_slice_to_parent = {child: slice_to_parent[child] for child in child_labels.unique()}

        return cls(parent_k=parent_k, slice_to_parent_mapping=unique_slice_to_parent)

    def generate(self, unique_slices: Sequence[Hashable]) -> Iterator[SliceCombination]:
        """Generate combinations of k parent groups, yielding flattened child slices.

        Args:
            unique_slices: Sequence of child slice labels.

        Yields:
            Tuples containing all child slices from k selected parent groups.
        """
        parent_to_children: Dict[Hashable, list] = {}
        for child in unique_slices:
            parent = self.slice_to_parent[child]
            parent_to_children.setdefault(parent, []).append(child)

        parent_ids = sorted(parent_to_children.keys())
        for parent_combi in itertools.combinations(parent_ids, self.parent_k):
            child_slices = []
            for parent_id in sorted(parent_combi):
                child_slices.extend(parent_to_children[parent_id])
            yield tuple(child_slices)

    def count(self, unique_slices: Sequence[Hashable]) -> int:
        """Count total number of parent-level combinations.

        Args:
            unique_slices: Sequence of child slice labels.

        Returns:
            C(n_parents, parent_k) where n_parents is the number of unique parent groups.
        """
        unique_parents = set(self.slice_to_parent[s] for s in unique_slices)
        n_parents = len(unique_parents)
        return math.comb(n_parents, self.parent_k)

    def combination_is_valid(
        self,
        combination: SliceCombination,
        unique_slices: Sequence[Hashable]
    ) -> bool:
        """Check if combination represents exactly parent_k complete parent groups.

        Args:
            combination: Tuple of child slice labels to validate.
            unique_slices: Sequence of all valid child slice labels.

        Returns:
            True if combination contains all children from exactly parent_k parent groups.
        """
        if not all(c in unique_slices for c in combination):
            return False

        parent_to_children: Dict[Hashable, set] = {}
        for child in unique_slices:
            parent = self.slice_to_parent[child]
            parent_to_children.setdefault(parent, set()).add(child)

        parents_in_combi = set()
        for child in combination:
            if child not in self.slice_to_parent:
                return False
            parents_in_combi.add(self.slice_to_parent[child])

        if len(parents_in_combi) != self.parent_k:
            return False

        expected_children = set()
        for parent in parents_in_combi:
            expected_children.update(parent_to_children[parent])

        return set(combination) == expected_children
