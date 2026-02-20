"""Feature space diagnostics for exploring and visualizing the feature representation.

This module provides diagnostic tools for analyzing feature spaces created by
FeatureEngineer implementations. These diagnostics help understand feature
relationships, distributions, and quality before running selection algorithms.
"""

from .feature_space_scatter import FeatureSpaceScatter2D, FeatureSpaceScatter3D
from .feature_space_scatter_matrix import FeatureSpaceScatterMatrix
from .selection_comparison_scatter_matrix import SelectionComparisonScatterMatrix
from .pca_variance_explained import PCAVarianceExplained
from .feature_correlation_heatmap import FeatureCorrelationHeatmap
from .feature_distributions import FeatureDistributions

__all__ = [
    'FeatureSpaceScatter2D',
    'FeatureSpaceScatter3D',
    'FeatureSpaceScatterMatrix',
    'SelectionComparisonScatterMatrix',
    'PCAVarianceExplained',
    'FeatureCorrelationHeatmap',
    'FeatureDistributions',
]
