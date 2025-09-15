from typing import List, Hashable
import pandas as pd

from .types import SliceUnit


class TimeSlicer:
    """
    Converts a DatetimeIndex into labeled time slices and enumerates unique slice IDs.

    Parameters
    ----------
    unit
        Temporal granularity of the slices. One of {"year","month","week","day","hour"}.

    Notes
    -----
    The labels are hashable and suitable for set membership and grouping. Periods are
    used for year, month, week, and day; naive timestamps are used for hour.
    """

    def __init__(self, unit: SliceUnit) -> None:
        self.unit = unit

    def labels_for_index(self, index: pd.DatetimeIndex) -> pd.Index:
        """
        Return a vector of slice labels aligned to the given index.

        Parameters
        ----------
        index
            DatetimeIndex for the input data.

        Returns
        -------
        pd.Index
            Slice labels matching the index length.
        """
        if self.unit == "year":
            return index.to_period("Y")
        if self.unit == "month":
            return index.to_period("M")
        if self.unit == "week":
            return index.to_period("W")
        if self.unit == "day":
            return index.to_period("D")
        if self.unit == "hour":
            return pd.Index(index.floor("H"))
        raise ValueError("Unsupported unit")

    def unique_slices(self, index: pd.DatetimeIndex) -> List[Hashable]:
        """
        Return the sorted list of unique slice labels present in the index.

        Parameters
        ----------
        index
            DatetimeIndex for the input data.

        Returns
        -------
        list
            Sorted list of unique slice labels.
        """
        labels = self.labels_for_index(index)
        unique = pd.Index(labels).unique().tolist()
        unique.sort()
        return unique
