from __future__ import annotations

from typing import TYPE_CHECKING, List

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from .base_feature_engineer import FeatureEngineer

if TYPE_CHECKING:
    from ..context import ProblemContext


class PCAFeatureEngineer(FeatureEngineer):
    """Performs PCA dimensionality reduction on existing features.

    Reduces the feature space using Principal Component Analysis, typically
    applied after statistical feature engineering. This is useful for:
    - Reducing dimensionality when you have many correlated features
    - Creating orthogonal feature representations
    - Focusing on the main axes of variation

    Commonly used in a FeaturePipeline after StandardStatsFeatureEngineer
    to compress statistical features into a smaller number of principal
    components.

    Args:
        n_components: Number of principal components to retain. Can be:
            - int: Exact number of components
            - float (0.0-1.0): Retain enough components to explain this
              fraction of variance
            - None: Retain all components (no reduction)
        whiten: If True, scale components to unit variance. This can improve
            results when PCA features are used with distance-based algorithms.

    Examples:

        >>> from energy_repset.feature_engineering import PCAFeatureEngineer
        >>> # Use PCA alone (requires context to already have df_features)
        >>> pca_engineer = PCAFeatureEngineer(n_components=5)
        >>> context_with_pca = pca_engineer.run(context_with_features)
        >>> print(context_with_pca.df_features.columns)
            ['pc_0', 'pc_1', 'pc_2', 'pc_3', 'pc_4']

        >>> # More common: chain with StandardStats in a pipeline
        >>> from energy_repset.feature_engineering import (
        ...     StandardStatsFeatureEngineer,
        ...     FeaturePipeline
        ... )
        >>> pipeline = FeaturePipeline([
        ...     StandardStatsFeatureEngineer(),
        ...     PCAFeatureEngineer(n_components=0.95)  # Keep 95% variance
        ... ])
        >>> context_with_both = pipeline.run(context)

        >>> # Check explained variance
        >>> pca_engineer = PCAFeatureEngineer(n_components=10)
        >>> context_out = pca_engineer.run(context_with_features)
        >>> print(pca_engineer.explained_variance_ratio_)
            [0.45, 0.22, 0.11, ...]
    """

    def __init__(
        self,
        n_components: int | float | None = None,
        whiten: bool = False
    ) -> None:
        """Initialize PCA feature engineer.

        Args:
            n_components: Number of components to keep, or fraction of
                variance to preserve (if float). None keeps all components.
            whiten: Whether to whiten (scale) the principal components.
        """
        self.n_components = n_components
        self.whiten = whiten
        self._pca: PCA | None = None
        self._feature_names: List[str] = []

    def calc_and_get_features_df(self, context: ProblemContext) -> pd.DataFrame:
        """Apply PCA to existing features in context.

        Args:
            context: Problem context with df_features already populated
                (typically by StandardStatsFeatureEngineer or similar).

        Returns:
            DataFrame with principal component features. Columns are named
            'pc_0', 'pc_1', etc.

        Raises:
            ValueError: If context.df_features is None or empty.
        """
        if context._df_features is None or context._df_features.empty:
            raise ValueError(
                "PCAFeatureEngineer requires context.df_features to be populated. "
                "Run StandardStatsFeatureEngineer or similar first, or use "
                "FeaturePipeline([StandardStatsFeatureEngineer(), PCAFeatureEngineer()])."
            )

        X = context.df_features.values
        index = context.df_features.index

        # Handle NaN/inf values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Fit PCA
        self._pca = PCA(n_components=self.n_components, whiten=self.whiten)
        X_transformed = self._pca.fit_transform(X)

        # Create feature names
        n_components_actual = X_transformed.shape[1]
        self._feature_names = [f"pc_{i}" for i in range(n_components_actual)]

        # Create DataFrame
        df_pca = pd.DataFrame(
            X_transformed,
            index=index,
            columns=self._feature_names
        )

        return df_pca

    def feature_names(self) -> List[str]:
        """Get list of principal component feature names.

        Returns:
            List of feature names: ['pc_0', 'pc_1', ...].
        """
        return list(self._feature_names)

    @property
    def explained_variance_ratio_(self) -> np.ndarray | None:
        """Get the proportion of variance explained by each component.

        Returns:
            Array of explained variance ratios, or None if PCA not fitted yet.

        Examples:

            >>> pca_eng = PCAFeatureEngineer(n_components=5)
            >>> context_out = pca_eng.run(context_with_features)
            >>> print(pca_eng.explained_variance_ratio_)
            # [0.45, 0.22, 0.15, 0.09, 0.05]
            >>> print(f"Total variance explained: {pca_eng.explained_variance_ratio_.sum():.2%}")
            # Total variance explained: 96%
        """
        if self._pca is None:
            return None
        return self._pca.explained_variance_ratio_

    @property
    def components_(self) -> np.ndarray | None:
        """Get the principal component loadings.

        Returns:
            Array of shape (n_components, n_features) containing the
            principal axes in feature space, or None if PCA not fitted yet.
        """
        if self._pca is None:
            return None
        return self._pca.components_
