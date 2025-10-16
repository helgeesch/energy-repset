from __future__ import annotations

from typing import TYPE_CHECKING
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

if TYPE_CHECKING:
    from ...feature_engineering.pca import PCAFeatureEngineer


class PCAVarianceExplained:
    """Visualize explained variance ratio for PCA components.

    Creates a bar chart showing the proportion of variance explained by each
    principal component, along with cumulative variance. Helps determine how
    many components are needed to capture most of the data's variance.

    This diagnostic requires the fitted PCAFeatureEngineer instance to access
    the explained_variance_ratio_ attribute.

    Examples:

        >>> # Get PCA engineer from pipeline
        >>> pca_engineer = pipeline.engineers['pca']
        >>> variance_plot = PCAVarianceExplained(pca_engineer)
        >>> fig = variance_plot.plot()
        >>> fig.update_layout(title='PCA Variance Explained')
        >>> fig.show()

        >>> # With custom number of components shown
        >>> fig = variance_plot.plot(n_components=10)

        >>> # After running workflow
        >>> context_with_features = workflow.feature_engineer.run(context)
        >>> pca_eng = workflow.feature_engineer.engineers['pca']
        >>> variance_plot = PCAVarianceExplained(pca_eng)
        >>> fig = variance_plot.plot()
    """

    def __init__(self, pca_engineer: PCAFeatureEngineer):
        """Initialize the PCA variance explained diagnostic.

        Args:
            pca_engineer: A fitted PCAFeatureEngineer instance. Must have been
                fitted on data (i.e., calc_and_get_features_df has been called).
        """
        self.pca_engineer = pca_engineer

    def plot(self, n_components: int = None, show_cumulative: bool = True) -> go.Figure:
        """Create a bar chart of explained variance ratios.

        Args:
            n_components: Number of components to show. If None, shows all components.
            show_cumulative: If True, adds a line showing cumulative variance explained.

        Returns:
            Plotly figure object ready for display or further customization.

        Raises:
            AttributeError: If the PCA engineer has not been fitted yet.
        """
        # Get variance ratios
        if not hasattr(self.pca_engineer, 'explained_variance_ratio_'):
            raise AttributeError(
                "PCA engineer has not been fitted. Call calc_and_get_features_df() first."
            )

        variance_ratio = self.pca_engineer.explained_variance_ratio_

        # Limit to requested number of components
        if n_components is not None:
            variance_ratio = variance_ratio[:n_components]

        # Prepare data
        n = len(variance_ratio)
        component_labels = [f'PC{i}' for i in range(n)]
        cumulative_variance = variance_ratio.cumsum()

        # Create figure
        fig = go.Figure()

        # Add variance bars
        fig.add_trace(go.Bar(
            x=component_labels,
            y=variance_ratio,
            name='Individual',
            marker_color='lightblue',
            text=[f'{v:.1%}' for v in variance_ratio],
            textposition='outside',
        ))

        # Add cumulative line if requested
        if show_cumulative:
            fig.add_trace(go.Scatter(
                x=component_labels,
                y=cumulative_variance,
                name='Cumulative',
                mode='lines+markers',
                line=dict(color='red', width=2),
                yaxis='y2',
                text=[f'{v:.1%}' for v in cumulative_variance],
                textposition='top center',
            ))

        # Update layout
        layout_kwargs = dict(
            xaxis_title='Principal Component',
            yaxis_title='Explained Variance Ratio',
            hovermode='x unified',
            yaxis=dict(tickformat='.0%'),
        )

        if show_cumulative:
            layout_kwargs['yaxis2'] = dict(
                title='Cumulative Variance',
                overlaying='y',
                side='right',
                tickformat='.0%',
                range=[0, 1.05],
            )

        fig.update_layout(**layout_kwargs)

        return fig
