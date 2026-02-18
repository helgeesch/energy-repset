from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterator, Sequence, Hashable, Literal
import itertools
import math
import pandas as pd

from .combination_generator import CombinationGenerator
from ..time_slicer import TimeSlicer

if TYPE_CHECKING:
    from ..types import SliceCombination


class GroupQuotaHierarchicalCombiGen(CombinationGenerator):
    """Generate combinations respecting quotas per parent-level group.

    This generator combines hierarchical selection (child slices selected in complete
    parent groups) with group quotas (e.g., exactly 1 month per season). It enables
    high-resolution features (e.g. per-day) while enforcing structural constraints
    at the parent level (e.g. months).

    Args:
        parent_k: Total number of parent groups to select. Must equal sum of group quotas.
        slice_to_parent_mapping: Mapping from each child slice to its parent group.
        parent_to_group_mapping: Mapping from parent ID to its group label (e.g., season).
        group_quota: Required count of parents per group.

    Attributes:
        k: Number of parent groups per combination (same as parent_k for protocol compliance).
        slice_to_parent: Child to parent mapping.
        parent_to_group: Parent to group label mapping.
        group_quota: Required count per group.

    Raises:
        ValueError: If sum of group quotas does not equal parent_k.

    Note:
        Use factory methods `from_slicers()` for automatic parent mapping and
        `from_slicers_with_seasons()` for automatic seasonal grouping.

    Examples:
        Manual construction for seasonal month selection:

        >>> from energy_repset.combi_gens import GroupQuotaHierarchicalCombiGen
        >>> import pandas as pd
        >>>
        >>> slice_to_parent = {
        ...     pd.Period('2024-01-01', 'D'): pd.Period('2024-01', 'M'),
        ...     pd.Period('2024-01-02', 'D'): pd.Period('2024-01', 'M'),
        ...     pd.Period('2024-07-01', 'D'): pd.Period('2024-07', 'M'),
        ...     pd.Period('2024-07-02', 'D'): pd.Period('2024-07', 'M'),
        ... }
        >>> parent_to_group = {
        ...     pd.Period('2024-01', 'M'): 'winter',
        ...     pd.Period('2024-07', 'M'): 'summer',
        ... }
        >>>
        >>> gen = GroupQuotaHierarchicalCombiGen(
        ...     parent_k=2,
        ...     slice_to_parent_mapping=slice_to_parent,
        ...     parent_to_group_mapping=parent_to_group,
        ...     group_quota={'winter': 1, 'summer': 1}
        ... )
        >>> gen.count(list(slice_to_parent.keys()))  # 1 * 1 = 1
            1
    """

    def __init__(
        self,
        parent_k: int,
        slice_to_parent_mapping: Dict[Hashable, Hashable],
        parent_to_group_mapping: Dict[Hashable, Hashable],
        group_quota: Dict[Hashable, int]
    ) -> None:
        """Initialize hierarchical quota generator.

        Args:
            parent_k: Total number of parent groups to select.
            slice_to_parent_mapping: Dict mapping each child slice to its parent.
            parent_to_group_mapping: Mapping from parent ID to group label.
            group_quota: Required count per group.

        Raises:
            ValueError: If sum of group quotas does not equal parent_k, or if any
                parent in slice_to_parent_mapping is missing from parent_to_group_mapping.
        """
        self.k = parent_k  # For CombinationGenerator protocol compliance
        self.slice_to_parent = slice_to_parent_mapping
        self.parent_to_group = parent_to_group_mapping
        self.group_quota = group_quota

        self._validate_configuration(parent_k, slice_to_parent_mapping, parent_to_group_mapping, group_quota)

    @classmethod
    def from_slicers(
        cls,
        parent_k: int,
        dt_index: pd.DatetimeIndex,
        child_slicer: TimeSlicer,
        parent_slicer: TimeSlicer,
        parent_to_group_mapping: Dict[Hashable, Hashable],
        group_quota: Dict[Hashable, int]
    ) -> GroupQuotaHierarchicalCombiGen:
        """Factory method to create generator from slicers with custom group mapping.

        Args:
            parent_k: Total number of parent groups to select.
            dt_index: DatetimeIndex of the time series data.
            child_slicer: TimeSlicer defining child slice granularity (e.g., daily).
            parent_slicer: TimeSlicer defining parent slice granularity (e.g., monthly).
            parent_to_group_mapping: Dict mapping parent IDs to group labels.
            group_quota: Required count per group.

        Returns:
            GroupQuotaHierarchicalCombinationGenerator with auto-constructed child-to-parent mapping.

        Raises:
            ValueError: If quotas invalid.

        Examples:
            Custom grouping of months into seasons:

            >>> import pandas as pd
            >>> from energy_repset.time_slicer import TimeSlicer
            >>> from energy_repset.combi_gens import GroupQuotaHierarchicalCombiGen
            >>>
            >>> dates = pd.date_range('2024-01-01', periods=366, freq='D')
            >>> child_slicer = TimeSlicer(unit='day')
            >>> parent_slicer = TimeSlicer(unit='month')
            >>>
            >>> parent_to_group = {
            ...     pd.Period('2024-01', 'M'): 'winter',
            ...     pd.Period('2024-02', 'M'): 'winter',
            ...     # ... define for all 12 months
            ... }
            >>>
            >>> gen = GroupQuotaHierarchicalCombiGen.from_slicers(
            ...     parent_k=4,
            ...     dt_index=dates,
            ...     child_slicer=child_slicer,
            ...     parent_slicer=parent_slicer,
            ...     parent_to_group_mapping=parent_to_group,
            ...     group_quota={'winter': 1, 'spring': 1, 'summer': 1, 'fall': 1}
            ... )
        """
        child_labels = child_slicer.labels_for_index(dt_index)
        parent_labels = parent_slicer.labels_for_index(dt_index)

        slice_to_parent = {}
        for child, parent in zip(child_labels, parent_labels):
            slice_to_parent[child] = parent

        unique_slice_to_parent = {child: slice_to_parent[child] for child in child_labels.unique()}

        return cls(
            parent_k=parent_k,
            slice_to_parent_mapping=unique_slice_to_parent,
            parent_to_group_mapping=parent_to_group_mapping,
            group_quota=group_quota
        )

    @classmethod
    def from_slicers_with_seasons(
        cls,
        parent_k: int,
        dt_index: pd.DatetimeIndex,
        child_slicer: TimeSlicer,
        group_quota: Dict[Literal['winter', 'spring', 'summer', 'fall'], int]
    ) -> GroupQuotaHierarchicalCombiGen:
        """Factory method with automatic seasonal grouping of months.

        Args:
            parent_k: Total number of parent groups to select (must equal sum of quotas).
            dt_index: DatetimeIndex of the time series data.
            child_slicer: TimeSlicer defining child slice granularity (e.g., daily).
            group_quota: Required count per season. Keys must be subset of
                {'winter', 'spring', 'summer', 'fall'}.

        Returns:
            GroupQuotaHierarchicalCombinationGenerator with seasonal parent grouping.

        Raises:
            ValueError: If quotas invalid.

        Note:
            This factory method uses monthly parents regardless of child slicer.
            Seasons are assigned as: winter (Dec/Jan/Feb), spring (Mar/Apr/May),
            summer (Jun/Jul/Aug), fall (Sep/Oct/Nov).

        Examples:
            Select 4 months (1 per season) from daily data:

            >>> import pandas as pd
            >>> from energy_repset.time_slicer import TimeSlicer
            >>> from energy_repset.combi_gens import GroupQuotaHierarchicalCombiGen
            >>>
            >>> dates = pd.date_range('2024-01-01', periods=366, freq='D')
            >>> child_slicer = TimeSlicer(unit='day')
            >>>
            >>> gen = GroupQuotaHierarchicalCombiGen.from_slicers_with_seasons(
            ...     parent_k=4,
            ...     dt_index=dates,
            ...     child_slicer=child_slicer,
            ...     group_quota={'winter': 1, 'spring': 1, 'summer': 1, 'fall': 1}
            ... )
            >>> gen.count(child_slicer.unique_slices(dates))  # 3 * 3 * 3 * 3 = 81
                81
        """
        child_labels = child_slicer.labels_for_index(dt_index)
        parent_slicer = TimeSlicer(unit='month')
        parent_labels = parent_slicer.labels_for_index(dt_index)

        slice_to_parent = {}
        all_parents = set()
        for child, parent in zip(child_labels, parent_labels):
            slice_to_parent[child] = parent
            all_parents.add(parent)

        unique_slice_to_parent = {child: slice_to_parent[child] for child in child_labels.unique()}
        parent_to_group = cls._assign_seasons(list(all_parents))

        return cls(
            parent_k=parent_k,
            slice_to_parent_mapping=unique_slice_to_parent,
            parent_to_group_mapping=parent_to_group,
            group_quota=group_quota
        )

    @staticmethod
    def _validate_configuration(
        parent_k: int,
        slice_to_parent_mapping: Dict[Hashable, Hashable],
        parent_to_group_mapping: Dict[Hashable, Hashable],
        group_quota: Dict[Hashable, int]
    ) -> None:
        """Validate the configuration of mappings and quotas.

        Args:
            parent_k: Total number of parent groups to select.
            slice_to_parent_mapping: Dict mapping each child slice to its parent.
            parent_to_group_mapping: Mapping from parent ID to group label.
            group_quota: Required count per group.

        Raises:
            ValueError: If sum of group quotas does not equal parent_k, or if any
                parent in slice_to_parent_mapping is missing from parent_to_group_mapping.
        """
        if sum(group_quota.values()) != parent_k:
            raise ValueError(
                f"Sum of group quotas ({sum(group_quota.values())}) must equal parent_k ({parent_k})"
            )

        unique_parents = set(slice_to_parent_mapping.values())
        missing_parents = unique_parents - set(parent_to_group_mapping.keys())
        if missing_parents:
            raise ValueError(
                f"All parents in slice_to_parent_mapping must have a group mapping. "
                f"Missing group mappings for parents: {sorted(missing_parents)}"
            )

    @staticmethod
    def _assign_seasons(months: list[pd.Period]) -> Dict[pd.Period, str]:
        """Assign meteorological seasons to month Period objects.

        Args:
            months: List of monthly Period objects.

        Returns:
            Dict mapping each month to its season: winter (Dec/Jan/Feb),
            spring (Mar/Apr/May), summer (Jun/Jul/Aug), fall (Sep/Oct/Nov).
        """
        season_map = {}
        for month in months:
            m = month.month
            if m in [12, 1, 2]:
                season_map[month] = 'winter'
            elif m in [3, 4, 5]:
                season_map[month] = 'spring'
            elif m in [6, 7, 8]:
                season_map[month] = 'summer'
            else:
                season_map[month] = 'fall'
        return season_map

    def generate(self, unique_slices: Sequence[Hashable]) -> Iterator[SliceCombination]:
        """Generate combinations respecting group quotas, yielding flattened child slices.

        Args:
            unique_slices: Sequence of child slice labels.

        Yields:
            Tuples containing all child slices from parent_k parent groups satisfying quotas.
        """
        parent_to_children: Dict[Hashable, list] = {}
        for child in unique_slices:
            parent = self.slice_to_parent[child]
            parent_to_children.setdefault(parent, []).append(child)

        unique_parents = set(self.slice_to_parent[s] for s in unique_slices)
        groups_of_parents: Dict[Hashable, list] = {}
        for parent in unique_parents:
            group = self.parent_to_group[parent]
            groups_of_parents.setdefault(group, []).append(parent)

        per_group_combis = []
        for group, quota in self.group_quota.items():
            parents_in_group = groups_of_parents.get(group, [])
            per_group_combis.append(list(itertools.combinations(parents_in_group, quota)))

        for parent_combi_tuple in itertools.product(*per_group_combis):
            all_selected_parents = list(itertools.chain.from_iterable(parent_combi_tuple))
            child_slices = []
            for parent in sorted(all_selected_parents):
                child_slices.extend(parent_to_children[parent])
            yield tuple(child_slices)

    def count(self, unique_slices: Sequence[Hashable]) -> int:
        """Count total combinations respecting group quotas.

        Args:
            unique_slices: Sequence of child slice labels.

        Returns:
            Product of C(n_parents_in_group, quota) across all groups.
        """
        unique_parents = set(self.slice_to_parent[s] for s in unique_slices)
        groups_of_parents: Dict[Hashable, list] = {}
        for parent in unique_parents:
            group = self.parent_to_group[parent]
            groups_of_parents.setdefault(group, []).append(parent)

        total = 1
        for group, quota in self.group_quota.items():
            n_parents = len(groups_of_parents.get(group, []))
            total *= math.comb(n_parents, quota)
        return total

    def combination_is_valid(
        self,
        combination: SliceCombination,
        unique_slices: Sequence[Hashable]
    ) -> bool:
        """Check if combination satisfies group quotas and completeness.

        Args:
            combination: Tuple of child slice labels to validate.
            unique_slices: Sequence of all valid child slice labels.

        Returns:
            True if combination contains complete parent groups satisfying quotas.
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

        if len(parents_in_combi) != self.k:
            return False

        expected_children = set()
        for parent in parents_in_combi:
            expected_children.update(parent_to_children[parent])
        if set(combination) != expected_children:
            return False

        group_count = {group: 0 for group in self.group_quota.keys()}
        for parent in parents_in_combi:
            group = self.parent_to_group[parent]
            group_count[group] += 1

        return all(group_count[g] == self.group_quota[g] for g in self.group_quota.keys())
