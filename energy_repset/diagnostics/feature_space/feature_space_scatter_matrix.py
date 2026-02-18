from __future__ import annotations

from typing import Hashable
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from ...types import SliceCombination


class FeatureSpaceScatterMatrix:
    """Scatter matrix (SPLOM) for visualizing relationships between multiple features.

    Creates an interactive scatter plot matrix showing pairwise relationships between
    all specified features. Can highlight a specific selection of slices. Useful for
    exploring multi-dimensional feature spaces and identifying feature correlations.

    Examples:

        >>> # Visualize PCA components
        >>> scatter_matrix = FeatureSpaceScatterMatrix()
        >>> fig = scatter_matrix.plot(
        ...     context.df_features,
        ...     dimensions=['pc_0', 'pc_1', 'pc_2']
        ... )
        >>> fig.update_layout(title='PCA Component Relationships')
        >>> fig.show()

        >>> # Visualize statistical features with selection
        >>> fig = scatter_matrix.plot(
        ...     context.df_features,
        ...     dimensions=['mean__demand', 'std__demand', 'max__wind'],
        ...     selection=('2024-01', '2024-04', '2024-07')
        ... )

        >>> # Color by a feature value
        >>> fig = scatter_matrix.plot(
        ...     context.df_features,
        ...     dimensions=['pc_0', 'pc_1', 'pc_2', 'pc_3'],
        ...     color='mean__demand'
        ... )

        >>> # All features
        >>> fig = scatter_matrix.plot(context.df_features)
    """

    def __init__(self):
        """Initialize the scatter matrix diagnostic."""
        pass

    def plot(
        self,
        df_features: pd.DataFrame,
        dimensions: list[str] = None,
        selection: SliceCombination = None,
        color: str = None,
    ) -> go.Figure:
        """Create a scatter plot matrix of feature space.

        Args:
            df_features: Feature matrix with slices as rows, features as columns.
            dimensions: List of column names to include in the matrix. If None,
                uses all columns (may be slow for many features).
            selection: Optional tuple of slice identifiers to highlight.
            color: Optional column name to use for color mapping. If None and
                selection is provided, colors by selection status.

        Returns:
            Plotly figure object ready for display or further customization.

        Raises:
            KeyError: If any dimension or color column is not in df_features.
            ValueError: If dimensions list is empty.
        """
        # Handle dimensions default
        if dimensions is None:
            dimensions = list(df_features.columns)

        if len(dimensions) == 0:
            raise ValueError("dimensions list cannot be empty")

        # Validate columns
        for dim in dimensions:
            if dim not in df_features.columns:
                raise KeyError(f"Column '{dim}' not found in df_features")
        if color is not None and color not in df_features.columns:
            raise KeyError(f"Column '{color}' not found in df_features")

        # Prepare data
        plot_df = df_features[dimensions].copy()
        plot_df['slice_label'] = df_features.index.astype(str)

        # Add selection indicator
        if selection is not None:
            selection_set = set(selection)
            plot_df['is_selected'] = df_features.index.isin(selection_set)
            # Order so selected points are drawn on top
            plot_df = pd.concat([
                plot_df[~plot_df['is_selected']],
                plot_df[plot_df['is_selected']]
            ], ignore_index=False)
        else:
            plot_df['is_selected'] = False

        # Create scatter matrix
        if color is not None:
            # Color by feature value
            fig = px.scatter_matrix(
                plot_df,
                dimensions=dimensions,
                color=color,
                hover_data=['slice_label'],
                symbol='is_selected' if selection is not None else None,
                symbol_map={True: 'star', False: 'circle'} if selection is not None else None,
            )
        else:
            # Color by selection status
            if selection is not None:
                fig = px.scatter_matrix(
                    plot_df,
                    dimensions=dimensions,
                    color='is_selected',
                    hover_data=['slice_label'],
                    color_discrete_map={True: 'red', False: 'lightgray'},
                    symbol='is_selected',
                    symbol_map={True: 'star', False: 'circle'},
                )
            else:
                fig = px.scatter_matrix(
                    plot_df,
                    dimensions=dimensions,
                    hover_data=['slice_label'],
                )

        # Update layout for better readability
        fig.update_traces(
            diagonal_visible=False,
            showupperhalf=False,
            marker=dict(size=4)
        )

        return fig
