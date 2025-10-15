from __future__ import annotations

import copy
from typing import List, Dict, Hashable, TYPE_CHECKING, Optional

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
        variable_weights: Optional weights per variable name for prioritizing variables
            in score components. If empty/None, all variables get equal weight (1.0).
            If provided, specified variables get their weights and missing ones get 0.0.
        feature_weights: Optional weights per feature name for prioritizing features
            in search algorithms. If empty/None, all features get equal weight (1.0).
            If provided, specified features get their weights and missing ones get 0.0.

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
        >>> # Create context with specific variable weights
        >>> slicer = TimeSlicer(unit='month')
        >>> context = ProblemContext(
        ...     df_raw=df,
        ...     slicer=slicer,
        ...     variable_weights={'demand': 1.5, 'solar': 1.0}
        ... )
        >>> len(context.get_unique_slices())  # 12 months
            12
        >>> context.variable_weights['demand']  # 1.5
        >>> context.variable_weights['wind']  # 0.0 (missing variable)
        >>>
        >>> # Create context with equal weights (no weights specified)
        >>> context2 = ProblemContext(df_raw=df, slicer=slicer)
        >>> context2.variable_weights['demand']  # 1.0 (equal weight)
        >>> context2.variable_weights['anything']  # 1.0 (equal weight)
    """

    def __init__(
        self,
        df_raw: pd.DataFrame,
        slicer: 'TimeSlicer',
        variable_weights: Optional[Dict] = None,
        feature_weights: Optional[Dict] = None
    ):
        """Initialize a ProblemContext.

        Args:
            df_raw: Raw time-series data with datetime index and variable columns.
            slicer: TimeSlicer defining how the time index is divided into candidate periods.
            variable_weights: Optional weights per variable name. None or empty dict
                means equal weights (1.0). Non-empty dict means specified weights with
                0.0 for missing variables.
            feature_weights: Optional weights per feature name. None or empty dict
                means equal weights (1.0). Non-empty dict means specified weights with
                0.0 for missing features.
        """
        self.df_raw = df_raw
        self.slicer = slicer
        self._variable_weights = variable_weights if variable_weights is not None else {}
        self._feature_weights = feature_weights if feature_weights is not None else {}
        self._df_features: Optional[pd.DataFrame] = None

    @property
    def variable_weights(self) -> Dict[str, float]:
        """Get variable weights based on actual variables in df_raw.

        Returns:
            Dictionary mapping each variable name in df_raw.columns to its weight:
            - If no weights were specified: all variables get 1.0 (equal weights)
            - If weights were specified: specified variables get their weight,
              missing variables get 0.0

        Examples:

            >>> # No weights specified - all equal
            >>> context = ProblemContext(df, slicer)
            >>> context.variable_weights
                {'demand': 1.0, 'solar': 1.0}
            >>>
            >>> # Weights specified - missing get 0.0
            >>> context = ProblemContext(df, slicer, variable_weights={'demand': 2.0})
            >>> context.variable_weights
                {'demand': 2.0, 'solar': 0.0}
        """
        actual_variables = self.df_raw.columns

        if not self._variable_weights:
            return {var: 1.0 for var in actual_variables}
        else:
            return {var: self._variable_weights.get(var, 0.0) for var in actual_variables}

    @property
    def feature_weights(self) -> Dict[str, float]:
        """Get feature weights based on actual features in df_features.

        Returns:
            Dictionary mapping each feature name in df_features.columns to its weight:
            - If no weights were specified: all features get 1.0 (equal weights)
            - If weights were specified: specified features get their weight,
              missing features get 0.0
            - If df_features is not set yet: returns the raw _feature_weights dict

        Examples:

            >>> # No weights specified - all equal
            >>> context = ProblemContext(df, slicer)
            >>> # After feature engineering:
            >>> context.feature_weights
                {'mean': 1.0, 'std': 1.0, 'max': 1.0}
            >>>
            >>> # Weights specified - missing get 0.0
            >>> context = ProblemContext(df, slicer, feature_weights={'mean': 2.0})
            >>> context.feature_weights
                {'mean': 2.0, 'std': 0.0, 'max': 0.0}
        """
        if self._df_features is None:
            return dict(self._feature_weights)

        actual_features = self._df_features.columns

        if not self._feature_weights:
            return {feat: 1.0 for feat in actual_features}
        else:
            return {feat: self._feature_weights.get(feat, 0.0) for feat in actual_features}

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
