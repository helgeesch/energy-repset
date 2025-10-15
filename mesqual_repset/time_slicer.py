from typing import List, Hashable
import pandas as pd

from .types import SliceUnit


class TimeSlicer:
    """Convert a DatetimeIndex into labeled time slices.

    This class defines how the time index is divided into candidate periods
    for representative subset selection. It converts timestamps into Period
    objects or floored timestamps based on the specified temporal granularity.

    Args:
        unit: Temporal granularity of the slices. One of "year", "month",
            "week", "day", or "hour".

    Attributes:
        unit: The temporal granularity used for slicing.

    Note:
        The labels are hashable and suitable for set membership and grouping.
        Period objects are used for year, month, week, and day. Naive
        timestamps (floored to hour) are used for hourly slicing.

    Examples:
        Create a slicer for monthly periods:

        >>> import pandas as pd
        >>> from mesqual_repset.time_slicer import TimeSlicer
        >>>
        >>> dates = pd.date_range('2024-01-01', periods=8760, freq='h')
        >>> slicer = TimeSlicer(unit='month')
        >>> labels = slicer.labels_for_index(dates)
        >>> unique_months = slicer.unique_slices(dates)
        >>> len(unique_months)  # 12 months in a year
        12
        >>> unique_months[0]  # First month
        Period('2024-01', 'M')

        Weekly slicing:

        >>> slicer = TimeSlicer(unit='week')
        >>> unique_weeks = slicer.unique_slices(dates)
        >>> len(unique_weeks)  # ~52 weeks in a year
        53
    """

    def __init__(self, unit: SliceUnit) -> None:
        """Initialize TimeSlicer with specified temporal granularity.

        Args:
            unit: One of "year", "month", "week", "day", or "hour".
        """
        self.unit = unit

    def labels_for_index(self, index: pd.DatetimeIndex) -> pd.Index:
        """Return a vector of slice labels aligned to the given index.

        Args:
            index: DatetimeIndex for the input data.

        Returns:
            Index of slice labels matching the input index length. Each timestamp
            is mapped to its corresponding period or floored hour.

        Raises:
            ValueError: If unit is not one of the supported values.
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
        """Return the sorted list of unique slice labels present in the index.

        Args:
            index: DatetimeIndex for the input data.

        Returns:
            Sorted list of unique slice labels. The sort order follows the natural
            ordering of Period objects or timestamps.
        """
        labels = self.labels_for_index(index)
        unique = pd.Index(labels).unique().tolist()
        unique.sort()
        return unique
