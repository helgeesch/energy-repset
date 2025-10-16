from __future__ import annotations

from typing import Dict, Protocol, TYPE_CHECKING
from abc import ABC, abstractmethod
import pandas as pd

if TYPE_CHECKING:
    from ..context import ProblemContext


class FeatureEngineer(ABC):
    """Base class for feature engineering transformations (Pillar F).

    Transforms raw sliced time-series data into a feature matrix that can be used
    for comparing and selecting representative periods. Implementations define how
    raw data is converted into a comparable feature space.

    The run() method creates a new ProblemContext with df_features populated,
    while subclasses implement _calc_and_get_features_df() to define the specific
    feature engineering logic.

    Examples:
        >>> class SimpleStatsFeatureEngineer(FeatureEngineer):
        ...     def _calc_and_get_features_df(self, context: ProblemContext) -> pd.DataFrame:
        ...         features = []
        ...         for slice_id in context.slicer.slices:
        ...             slice_data = context.df_raw.loc[slice_id]
        ...             features.append({
        ...                 'mean': slice_data.mean().mean(),
        ...                 'std': slice_data.std().mean(),
        ...                 'max': slice_data.max().max()
        ...             })
        ...         return pd.DataFrame(features, index=context.slicer.slices)
        ...
        >>> engineer = SimpleStatsFeatureEngineer()
        >>> context_with_features = engineer.run(context)
        >>> print(context_with_features.df_features.head())
    """
    def run(self, context: ProblemContext) -> ProblemContext:
        """Calculate features and return a new context with df_features populated.

        Args:
            context: The problem context containing raw time-series data and slicing
                information.

        Returns:
            A new ProblemContext instance with df_features set to the computed
            feature matrix. The original context is not modified.
        """
        context_with_features = context.copy()
        context_with_features.df_features = self.calc_and_get_features_df(context)
        return context_with_features

    @abstractmethod
    def calc_and_get_features_df(self, context: ProblemContext) -> pd.DataFrame:
        """Calculate and return the feature matrix.

        Args:
            context: The problem context containing raw data and slicing information.

        Returns:
            A DataFrame where each row represents one slice (candidate period) and
            each column represents a feature. The index should match the slice
            identifiers from context.slicer.slices.
        """
        ...


class FeaturePipeline(FeatureEngineer):
    """Chains multiple feature engineers to create a combined feature space.

    Runs multiple feature engineering transformations sequentially and concatenates
    their outputs into a single feature matrix. Useful for combining different
    feature types (e.g., statistical summaries + PCA components).

    Examples:

        >>> from mesqual_repset.feature_engineering import StandardStatsFeatureEngineer, PCAFeatureEngineer
        >>> stats_engineer = StandardStatsFeatureEngineer()
        >>> pca_engineer = PCAFeatureEngineer(n_components=3)
        >>> pipeline = FeaturePipeline({'stats': stats_engineer, 'pca': pca_engineer})
        >>> context_with_features = pipeline.run(context)
        >>> print(context_with_features.df_features.columns)
            # Shows columns from both engineers: ['mean', 'std', 'max', 'min', 'pc1', 'pc2', 'pc3']
    """
    def __init__(self, engineers: Dict[str, FeatureEngineer]):
        """Initialize the feature pipeline.

        Args:
            engineers: Dict of FeatureEngineer instances to run sequentially.
                Features from all engineers will be concatenated column-wise.
        """
        self.engineers = engineers

    def calc_and_get_features_df(self, context: ProblemContext) -> pd.DataFrame:
        """Calculate features from all engineers sequentially, accumulating results.

        Each engineer in the pipeline sees the accumulated features from all
        previous engineers via context.df_features. New features from each stage
        are concatenated to the existing feature set. This allows:
        - Early engineers to create base features (e.g., StandardStatsFeatureEngineer)
        - Later engineers to transform or add to those features (e.g., PCAFeatureEngineer)

        Args:
            context: The problem context containing raw data.

        Returns:
            A DataFrame with columns from all engineers concatenated horizontally.
            Each engineer's features are added to the cumulative feature set.
        """
        # Create a mutable working copy of the context
        working_context = context.copy()

        # Accumulate features from each engineer
        all_features = []
        for _, engineer in self.engineers.items():
            features = engineer.calc_and_get_features_df(working_context)
            all_features.append(features)

            # Update context so next engineer can see accumulated features
            if all_features:
                working_context._df_features = pd.concat(all_features, axis=1)

        # Return the concatenated feature set
        return pd.concat(all_features, axis=1)
