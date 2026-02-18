"""Diagnostics and visualization tools for representative subset selection.

Usage::

    import energy_repset.diagnostics as diag

All diagnostics follow a consistent pattern:
1. Dependencies are passed to the constructor (explicit dependency injection)
2. The plot() method generates and returns a Plotly figure
3. Users can customize titles and styling via fig.update_layout() after retrieval
"""

from .feature_space import (
    FeatureSpaceScatter2D,
    FeatureSpaceScatter3D,
    FeatureSpaceScatterMatrix,
    PCAVarianceExplained,
    FeatureCorrelationHeatmap,
    FeatureDistributions,
)

from .score_components import (
    DistributionOverlayECDF,
    DistributionOverlayHistogram,
    CorrelationDifferenceHeatmap,
    DiurnalProfileOverlay,
)

from .results import (
    ResponsibilityBars,
    ParetoScatter2D,
    ParetoScatterMatrix,
    ParetoParallelCoordinates,
    ScoreContributionBars,
)

__all__ = [
    # Feature space
    "FeatureSpaceScatter2D",
    "FeatureSpaceScatter3D",
    "FeatureSpaceScatterMatrix",
    "PCAVarianceExplained",
    "FeatureCorrelationHeatmap",
    "FeatureDistributions",
    # Score components
    "DistributionOverlayECDF",
    "DistributionOverlayHistogram",
    "CorrelationDifferenceHeatmap",
    "DiurnalProfileOverlay",
    # Results
    "ResponsibilityBars",
    "ParetoScatter2D",
    "ParetoScatterMatrix",
    "ParetoParallelCoordinates",
    "ScoreContributionBars",
]
