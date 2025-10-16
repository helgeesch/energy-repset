from __future__ import annotations

import copy
from typing import List, Dict, Hashable, TYPE_CHECKING, Optional, Any

import pandas as pd

if TYPE_CHECKING:
    from .time_slicer import TimeSlicer


class ProblemContext:
    """A data container passed through the entire workflow.

    This class holds all data and metadata needed for representative subset selection.
    It is the central object passed between workflow stages (feature engineering,
    search algorithms, representation models).

    Args:
        df_raw: Raw time-series data with datetime index and variable columns.
        slicer: TimeSlicer defining how the time index is divided into candidate periods.
        metadata: Optional dict for storing arbitrary user data (e.g., default weights,
            experiment configuration, notes, etc.). Not used by the framework itself,
            but available for user convenience and custom component implementations.

    Examples:
        Create a context with monthly slicing:

        >>> import pandas as pd
        >>> from mesqual_repset.context import ProblemContext
        >>> from mesqual_repset.time_slicer import TimeSlicer
        >>>
        >>> # Create sample data
        >>> dates = pd.date_range('2024-01-01', periods=8760, freq='h')
        >>> df = pd.DataFrame({
        ...     'demand': np.random.rand(8760),
        ...     'solar': np.random.rand(8760)
        ... }, index=dates)
        >>>
        >>> # Create context with metadata
        >>> slicer = TimeSlicer(unit='month')
        >>> context = ProblemContext(
        ...     df_raw=df,
        ...     slicer=slicer,
        ...     metadata={
        ...         'experiment_name': 'test_run_1',
        ...         'default_weights': {'demand': 1.5, 'solar': 1.0},
        ...         'notes': 'Testing seasonal selection'
        ...     }
        ... )
        >>> len(context.get_unique_slices())  # 12 months
            12
        >>> context.metadata['experiment_name']  # 'test_run_1'
        >>>
        >>> # Create context without metadata
        >>> context2 = ProblemContext(df_raw=df, slicer=slicer)
        >>> context2.metadata  # {}
    """

    def __init__(
        self,
        df_raw: pd.DataFrame,
        slicer: 'TimeSlicer',
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize a ProblemContext.

        Args:
            df_raw: Raw time-series data with datetime index and variable columns.
            slicer: TimeSlicer defining how the time index is divided into candidate periods.
            metadata: Optional dict for storing arbitrary user data. Not used by the
                framework itself.
        """
        self.df_raw = df_raw
        self.slicer = slicer
        self.metadata = metadata if metadata is not None else {}
        self._df_features: Optional[pd.DataFrame] = None

    def copy(self) -> 'ProblemContext':
        """Create a deep copy of this ProblemContext instance.

        Returns:
            A new, independent instance of the context with all data copied.
        """
        return copy.deepcopy(self)

    def get_sliced_data(self) -> Dict[Hashable, pd.DataFrame]:
        """Generate sliced raw data on demand.

        Returns:
            Dictionary mapping slice labels to their corresponding DataFrame chunks.

        Raises:
            NotImplementedError: This method is not yet implemented.
        """
        raise NotImplementedError

    @property
    def df_features(self) -> pd.DataFrame:
        """Get the computed feature matrix.

        Returns:
            DataFrame with slice labels as index and engineered features as columns.

        Raises:
            ValueError: If features have not been computed yet. Use a FeatureEngineer
                to populate this field first.
        """
        if self._df_features is None:
            raise ValueError(
                f'You tried to retrieve df_features before assigning it. Please set first using a FeatureEngineer.'
            )
        return self._df_features

    @df_features.setter
    def df_features(self, df_features: pd.DataFrame):
        """Set the feature matrix.

        Args:
            df_features: DataFrame with slice labels as index and features as columns.

        Raises:
            ValueError: If df_features index does not contain all expected slices
                from the slicer.
        """
        self._validate_all_slices_present_in_features_df(df_features)

        self._df_features = df_features

    def _validate_all_slices_present_in_features_df(self, df_features):
        expected_slices = set(self.get_unique_slices())
        actual_slices = set(df_features.index)
        if not expected_slices.issubset(actual_slices):
            missing_slices = expected_slices - actual_slices
            raise ValueError(
                f"df_features is missing {len(missing_slices)} slice(s). "
                f"Expected all slices from slicer but missing: {sorted(list(missing_slices)[:5])}"
                f"{'...' if len(missing_slices) > 5 else ''}"
            )

    def get_unique_slices(self) -> List[Hashable]:
        """Get list of all unique slice labels from the time index.

        Returns:
            List of slice labels (e.g., Period objects for monthly slicing).
        """
        return self.slicer.unique_slices(self.df_raw.index)
