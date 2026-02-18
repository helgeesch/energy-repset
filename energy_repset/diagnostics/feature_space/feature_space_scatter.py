from __future__ import annotations

from typing import Hashable
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from ...types import SliceCombination


class FeatureSpaceScatter2D:
    """2D scatter plot for visualizing feature space.

    Creates an interactive scatter plot of any two features from df_features.
    Can highlight a specific selection of slices. Works with any feature columns
    including PCA components ('pc_0', 'pc_1'), statistical features ('mean__wind'),
    or mixed features.

    Examples:

        >>> # Visualize PCA space
        >>> scatter = FeatureSpaceScatter2D()
        >>> fig = scatter.plot(context.df_features, x='pc_0', y='pc_1')
        >>> fig.update_layout(title='PCA Feature Space')
        >>> fig.show()

        >>> # Visualize with selection highlighted
        >>> fig = scatter.plot(
        ...     context.df_features,
        ...     x='mean__demand',
        ...     y='pc_0',
        ...     selection=('2024-01', '2024-04', '2024-07')
        ... )

        >>> # Color by another feature
        >>> fig = scatter.plot(
        ...     context.df_features,
        ...     x='pc_0',
        ...     y='pc_1',
        ...     color='std__wind'
        ... )
    """

    def __init__(self):
        """Initialize the scatter plot diagnostic."""
        pass

    def plot(
        self,
        df_features: pd.DataFrame,
        x: str,
        y: str,
        selection: SliceCombination = None,
        color: str = None,
    ) -> go.Figure:
        """Create a 2D scatter plot of feature space.

        Args:
            df_features: Feature matrix with slices as rows, features as columns.
            x: Column name for x-axis.
            y: Column name for y-axis.
            selection: Optional tuple of slice identifiers to highlight.
            color: Optional column name to use for color mapping.

        Returns:
            Plotly figure object ready for display or further customization.

        Raises:
            KeyError: If x, y, or color columns are not in df_features.
        """
        # Validate columns
        if x not in df_features.columns:
            raise KeyError(f"Column '{x}' not found in df_features")
        if y not in df_features.columns:
            raise KeyError(f"Column '{y}' not found in df_features")
        if color is not None and color not in df_features.columns:
            raise KeyError(f"Column '{color}' not found in df_features")

        # Prepare data
        plot_df = df_features.copy()
        plot_df['slice_label'] = plot_df.index.astype(str)

        # Add selection indicator
        if selection is not None:
            selection_set = set(selection)
            plot_df['is_selected'] = plot_df.index.isin(selection_set)
        else:
            plot_df['is_selected'] = False

        # Create scatter plot
        if color is not None:
            # Color by feature value
            fig = px.scatter(
                plot_df,
                x=x,
                y=y,
                color=color,
                hover_data=['slice_label'],
                symbol='is_selected' if selection is not None else None,
                symbol_map={True: 'star', False: 'circle'} if selection is not None else None,
            )
        else:
            # Color by selection status
            if selection is not None:
                fig = px.scatter(
                    plot_df,
                    x=x,
                    y=y,
                    color='is_selected',
                    hover_data=['slice_label'],
                    color_discrete_map={True: 'red', False: 'lightgray'},
                )
            else:
                fig = px.scatter(
                    plot_df,
                    x=x,
                    y=y,
                    hover_data=['slice_label'],
                )

        # Update layout for better readability
        fig.update_layout(
            xaxis_title=x,
            yaxis_title=y,
            hovermode='closest',
        )

        return fig


class FeatureSpaceScatter3D:
    """3D scatter plot for visualizing feature space.

    Creates an interactive 3D scatter plot of any three features from df_features.
    Can highlight a specific selection of slices. Works with any feature columns
    including PCA components or statistical features.

    Examples:

        >>> # Visualize 3D PCA space
        >>> scatter = FeatureSpaceScatter3D()
        >>> fig = scatter.plot(
        ...     context.df_features,
        ...     x='pc_0',
        ...     y='pc_1',
        ...     z='pc_2'
        ... )
        >>> fig.update_layout(title='3D PCA Space')
        >>> fig.show()

        >>> # Highlight selection
        >>> fig = scatter.plot(
        ...     context.df_features,
        ...     x='pc_0',
        ...     y='pc_1',
        ...     z='pc_2',
        ...     selection=('2024-01', '2024-04')
        ... )

        >>> # Color by feature value
        >>> fig = scatter.plot(
        ...     context.df_features,
        ...     x='pc_0',
        ...     y='pc_1',
        ...     z='pc_2',
        ...     color='mean__demand'
        ... )
    """

    def __init__(self):
        """Initialize the 3D scatter plot diagnostic."""
        pass

    def plot(
        self,
        df_features: pd.DataFrame,
        x: str,
        y: str,
        z: str,
        selection: SliceCombination = None,
        color: str = None,
    ) -> go.Figure:
        """Create a 3D scatter plot of feature space.

        Args:
            df_features: Feature matrix with slices as rows, features as columns.
            x: Column name for x-axis.
            y: Column name for y-axis.
            z: Column name for z-axis.
            selection: Optional tuple of slice identifiers to highlight.
            color: Optional column name to use for color mapping.

        Returns:
            Plotly figure object ready for display or further customization.

        Raises:
            KeyError: If x, y, z, or color columns are not in df_features.
        """
        # Validate columns
        if x not in df_features.columns:
            raise KeyError(f"Column '{x}' not found in df_features")
        if y not in df_features.columns:
            raise KeyError(f"Column '{y}' not found in df_features")
        if z not in df_features.columns:
            raise KeyError(f"Column '{z}' not found in df_features")
        if color is not None and color not in df_features.columns:
            raise KeyError(f"Column '{color}' not found in df_features")

        # Prepare data
        plot_df = df_features.copy()
        plot_df['slice_label'] = plot_df.index.astype(str)

        # Add selection indicator
        if selection is not None:
            selection_set = set(selection)
            plot_df['is_selected'] = plot_df.index.isin(selection_set)
        else:
            plot_df['is_selected'] = False

        # Create 3D scatter plot
        if color is not None:
            # Color by feature value
            fig = px.scatter_3d(
                plot_df,
                x=x,
                y=y,
                z=z,
                color=color,
                hover_data=['slice_label'],
                symbol='is_selected' if selection is not None else None,
                symbol_map={True: 'diamond', False: 'circle'} if selection is not None else None,
            )
        else:
            # Color by selection status
            if selection is not None:
                fig = px.scatter_3d(
                    plot_df,
                    x=x,
                    y=y,
                    z=z,
                    color='is_selected',
                    hover_data=['slice_label'],
                    color_discrete_map={True: 'red', False: 'lightgray'},
                )
            else:
                fig = px.scatter_3d(
                    plot_df,
                    x=x,
                    y=y,
                    z=z,
                    hover_data=['slice_label'],
                )

        # Update layout for better readability
        fig.update_layout(
            scene=dict(
                xaxis_title=x,
                yaxis_title=y,
                zaxis_title=z,
            ),
            hovermode='closest',
        )

        return fig
