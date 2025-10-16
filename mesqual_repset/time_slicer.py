from typing import List, Hashable, Union, Tuple
import pandas as pd

from .types import SliceUnit, SliceCombination


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

    def get_indices_for_slice_combo(
        self,
        index: pd.DatetimeIndex,
        selection: Union[Hashable, SliceCombination],
    ) -> pd.Index:
        """Return the index positions for timestamps belonging to the given slice(s).

        Args:
            index: DatetimeIndex for the input data.
            selection: Either a single slice label or a tuple of slice labels
                (SliceCombination) to extract indices for.

        Returns:
            Index of timestamps that belong to the specified slice(s). If selection
            is a tuple, returns the union of all timestamps from all slices.

        Examples:
            Get indices for a single month:

            >>> import pandas as pd
            >>> from mesqual_repset.time_slicer import TimeSlicer
            >>>
            >>> dates = pd.date_range('2024-01-01', periods=8760, freq='h')
            >>> slicer = TimeSlicer(unit='month')
            >>> jan_slice = slicer.unique_slices(dates)[0]  # Period('2024-01', 'M')
            >>> jan_indices = slicer.get_indices_for_slice_combo(dates, jan_slice)
            >>> len(jan_indices)  # 744 hours in January 2024
                744

            Get indices for multiple months (selection):

            >>> selection = (Period('2024-01', 'M'), Period('2024-06', 'M'))
            >>> selected_indices = slicer.get_indices_for_slice_combo(dates, selection)
            >>> len(selected_indices)  # Jan (744) + Jun (720) = 1464
                1464
        """
        labels = self.labels_for_index(index)

        # Convert single slice to tuple for uniform handling
        if isinstance(selection, tuple):
            slice_set = set(selection)
        else:
            slice_set = {selection}

        # Create boolean mask for timestamps in any of the selected slices
        mask = labels.isin(slice_set)

        # Return the index positions where mask is True
        return index[mask]
