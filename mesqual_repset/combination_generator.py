from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterator, Sequence, Tuple, Hashable, Protocol, List, Any
import itertools
import math

if TYPE_CHECKING:
    from .types import SliceCombination


class CombinationGenerator(Protocol):
    """
    Protocol for generating and counting combinations of candidate slices.

    Attributes
    ----------
    k : int
        Number of elements in each combination
    
    Methods
    -------
    iterate(unique_slices)
        Yield tuples of length k from the unique_slices according to generator logic.
    count(unique_slices)
        Return the total number of combinations the generator will yield.
    """
    
    k: int
    
    def generate(self, unique_slices: Sequence[Hashable]) -> Iterator[SliceCombination]: ...
    def count(self, unique_slices: Sequence[Hashable]) -> int: ...
    def combination_is_valid(self, combination: SliceCombination, unique_slices: Sequence[Hashable]) -> bool: ...


class ExhaustiveCombinationGenerator(CombinationGenerator):
    """
    Generates all k-combinations of the candidate slices.

    Parameters
    ----------
    k : int
        Number of elements in each combination

    Notes
    -----
    The count is computed via n choose k and matches the number of yielded combinations.
    """
    
    def __init__(self, k: int) -> None:
        self.k = k

    def generate(self, unique_slices: Sequence[Hashable]) -> Iterator[SliceCombination]:
        yield from itertools.combinations(unique_slices, self.k)

    def count(self, unique_slices: Sequence[Hashable]) -> int:
        return math.comb(len(unique_slices), self.k)

    def combination_is_valid(self, combination: SliceCombination, unique_slices: Sequence[Hashable]) -> bool:
        if len(combination) != self.k:
            return False
        if any(s not in unique_slices for s in unique_slices):
            return False
        return True


class GroupQuotaCombinationGenerator(CombinationGenerator):
    """
    Generates combinations that respect exact quotas per group.

    Parameters
    ----------
    k : int
        Number of elements in each combination (must equal sum of group quotas)
    slice_to_group_mapping : Dict[Hashable, Hashable]
        Mapping from candidate slice to its group label.
    group_quota : Dict[Hashable, int]
        Mapping from group label to the required count in the selection.

    Notes
    -----
    The total of group_quota values must equal k. Use this to enforce constraints like
    exactly one month per season.
    """

    def __init__(
            self,
            k: int,
            slice_to_group_mapping: Dict[Hashable, Hashable],
            group_quota: Dict[Hashable, int]
    ) -> None:
        self.k = k
        self.group_of = slice_to_group_mapping
        self.group_quota = group_quota
        
        # Validate that quotas sum to k
        if sum(group_quota.values()) != k:
            raise ValueError(f"Sum of group quotas ({sum(group_quota.values())}) must equal k ({k})")

    def generate(self, unique_slices: Sequence[Hashable]) -> Iterator[SliceCombination]:
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
