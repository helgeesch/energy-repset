from __future__ import annotations

import copy
from typing import List, Dict, Hashable, TYPE_CHECKING
from dataclasses import dataclass, field

import pandas as pd

if TYPE_CHECKING:
    from .time_slicer import TimeSlicer


@dataclass
class ProblemContext:
    """A data container passed through the entire workflow.

    This class holds all data and metadata needed for representative subset selection.
    It is the central object passed between workflow stages (feature engineering,
    search algorithms, representation models).

    Attributes:
        df_raw: Raw time-series data with datetime index and variable columns.
        slicer: TimeSlicer defining how the time index is divided into candidate periods.
        variable_weights: Optional weights per variable name for prioritizing variables
            in score components. Empty dict means equal weights.
        feature_weights: Optional weights per feature name for prioritizing features
            in search algorithms. Empty dict means equal weights.

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
        >>> # Create context
        >>> slicer = TimeSlicer(unit='month')
        >>> context = ProblemContext(
        ...     df_raw=df,
        ...     slicer=slicer,
        ...     variable_weights={'demand': 1.5, 'solar': 1.0}
        ... )
        >>> len(context.get_unique_slices())  # 12 months
        12
    """
    df_raw: pd.DataFrame
    slicer: TimeSlicer
    variable_weights: Dict = field(default_factory=dict)
    feature_weights: Dict = field(default_factory=dict)
    _df_features: pd.DataFrame = None

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
        """
        self._df_features = df_features

    def get_unique_slices(self) -> List[Hashable]:
        """Get list of all unique slice labels from the time index.

        Returns:
            List of slice labels (e.g., Period objects for monthly slicing).
        """
        return self.slicer.unique_slices(self.df_raw.index)
