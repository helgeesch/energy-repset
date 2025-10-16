"""Diagnostics and visualization tools for representative subset selection.

This package provides diagnostic tools for analyzing and visualizing:
- Feature spaces created by FeatureEngineer implementations
- Selection quality measured by ScoreComponent implementations
- Results from search algorithms and selection policies

All diagnostics follow a consistent pattern:
1. Dependencies are passed to the constructor (explicit dependency injection)
2. The plot() method generates and returns a Plotly figure
3. Users can customize titles and styling via fig.update_layout() after retrieval
"""

from . import feature_space
from . import score_components

__all__ = [
    'feature_space',
    'score_components',
]
