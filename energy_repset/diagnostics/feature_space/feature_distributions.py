from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class FeatureDistributions:
    """Visualize distributions of all features as histograms.

    Creates a grid of histograms showing the distribution of each feature across
    all slices. Helps identify feature scales, skewness, and potential outliers.
    Useful for understanding the feature space before selection.

    Examples:

        >>> # Visualize all feature distributions
        >>> dist_plot = FeatureDistributions()
        >>> fig = dist_plot.plot(context.df_features)
        >>> fig.update_layout(title='Feature Distributions')
        >>> fig.show()

        >>> # Subset of features
        >>> selected_features = context.df_features[['pc_0', 'pc_1', 'mean__demand']]
        >>> fig = dist_plot.plot(selected_features)

        >>> # With custom bin count
        >>> fig = dist_plot.plot(context.df_features, nbins=30)
    """

    def __init__(self):
        """Initialize the feature distributions diagnostic."""
        pass

    def plot(
        self,
        df_features: pd.DataFrame,
        nbins: int = 20,
        cols: int = 3,
    ) -> go.Figure:
        """Create a grid of histograms for all features.

        Args:
            df_features: Feature matrix with slices as rows, features as columns.
            nbins: Number of bins for each histogram. Default is 20.
            cols: Number of columns in the subplot grid. Default is 3.

        Returns:
            Plotly figure object ready for display or further customization.

        Raises:
            ValueError: If df_features is empty or nbins/cols are invalid.
        """
        if df_features.empty:
            raise ValueError("df_features cannot be empty")
        if nbins <= 0:
            raise ValueError("nbins must be positive")
        if cols <= 0:
            raise ValueError("cols must be positive")

        features = list(df_features.columns)
        n_features = len(features)

        # Calculate grid dimensions
        rows = (n_features + cols - 1) // cols  # Ceiling division

        # Create subplots
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=features,
            vertical_spacing=0.12 / rows if rows > 1 else 0.1,
            horizontal_spacing=0.1 / cols if cols > 1 else 0.1,
        )

        # Add histogram for each feature
        for idx, feature in enumerate(features):
            row = idx // cols + 1
            col = idx % cols + 1

            fig.add_trace(
                go.Histogram(
                    x=df_features[feature],
                    nbinsx=nbins,
                    name=feature,
                    showlegend=False,
                    marker_color='lightblue',
                ),
                row=row,
                col=col,
            )

            # Update axes labels
            fig.update_xaxes(title_text=feature, row=row, col=col)
            fig.update_yaxes(title_text='Count', row=row, col=col)

        # Update overall layout
        fig.update_layout(
            height=300 * rows,
            showlegend=False,
        )

        return fig
