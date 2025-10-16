from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


class FeatureCorrelationHeatmap:
    """Visualize correlation matrix of features.

    Creates an interactive heatmap showing Pearson correlations between all features
    in the feature matrix. Helps identify redundant features and understand feature
    relationships. Can optionally show only the lower triangle to avoid redundancy.

    Examples:

        >>> # Visualize all feature correlations
        >>> heatmap = FeatureCorrelationHeatmap()
        >>> fig = heatmap.plot(context.df_features)
        >>> fig.update_layout(title='Feature Correlation Matrix')
        >>> fig.show()

        >>> # Show only lower triangle
        >>> fig = heatmap.plot(context.df_features, show_lower_only=True)

        >>> # Subset of features
        >>> selected_features = context.df_features[['pc_0', 'pc_1', 'mean__demand']]
        >>> fig = heatmap.plot(selected_features)
    """

    def __init__(self):
        """Initialize the feature correlation heatmap diagnostic."""
        pass

    def plot(
        self,
        df_features: pd.DataFrame,
        method: str = 'pearson',
        show_lower_only: bool = False,
    ) -> go.Figure:
        """Create a heatmap of feature correlations.

        Args:
            df_features: Feature matrix with slices as rows, features as columns.
            method: Correlation method ('pearson', 'spearman', or 'kendall').
                Default is 'pearson'.
            show_lower_only: If True, shows only the lower triangle of the
                correlation matrix (removes redundant upper triangle and diagonal).

        Returns:
            Plotly figure object ready for display or further customization.

        Raises:
            ValueError: If method is not one of the supported correlation methods.
        """
        if method not in ['pearson', 'spearman', 'kendall']:
            raise ValueError(
                f"method must be 'pearson', 'spearman', or 'kendall', got '{method}'"
            )

        # Calculate correlation matrix
        corr_matrix = df_features.corr(method=method)

        # Mask upper triangle if requested
        if show_lower_only:
            mask = pd.DataFrame(
                False,
                index=corr_matrix.index,
                columns=corr_matrix.columns
            )
            # Set upper triangle and diagonal to True (to be masked)
            for i in range(len(corr_matrix)):
                for j in range(i, len(corr_matrix)):
                    mask.iloc[i, j] = True

            # Apply mask by setting values to NaN
            corr_matrix = corr_matrix.where(~mask)

        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            color_continuous_scale='RdBu_r',
            color_continuous_midpoint=0,
            zmin=-1,
            zmax=1,
            aspect='auto',
        )

        # Update layout for better readability
        fig.update_layout(
            xaxis_title='',
            yaxis_title='',
            coloraxis_colorbar=dict(title='Correlation'),
        )

        # Improve text readability
        fig.update_traces(
            text=corr_matrix.round(2).values,
            texttemplate='%{text}',
            textfont=dict(size=10),
        )

        return fig
