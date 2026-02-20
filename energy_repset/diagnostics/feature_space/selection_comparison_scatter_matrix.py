from __future__ import annotations

from typing import Dict

import pandas as pd
import plotly.graph_objects as go

from ...types import SliceCombination

_SYMBOLS = ['star', 'diamond', 'cross', 'square']
_COLORS = [
    '#e41a1c',  # red
    '#377eb8',  # blue
    '#4daf4a',  # green
    '#ff7f00',  # orange
]


class SelectionComparisonScatterMatrix:
    """Scatter matrix comparing multiple named selections in feature space.

    Creates an interactive scatter plot matrix (SPLOM) where each named
    selection is overlaid with a distinct color and marker symbol. All
    slices are shown in the background as light-gray circles.

    Examples:

        >>> scatter = SelectionComparisonScatterMatrix()
        >>> fig = scatter.plot(
        ...     context.df_features,
        ...     selections={
        ...         'Balanced': result_a.selection,
        ...         'Diverse': result_b.selection,
        ...     },
        ...     dimensions=['pc_0', 'pc_1', 'pc_2', 'pc_3'],
        ... )
        >>> fig.show()
    """

    def __init__(self):
        """Initialize the selection comparison scatter matrix diagnostic."""
        pass

    def plot(
        self,
        df_features: pd.DataFrame,
        selections: Dict[str, SliceCombination],
        dimensions: list[str] | None = None,
    ) -> go.Figure:
        """Create a scatter matrix comparing multiple selections.

        Args:
            df_features: Feature matrix with slices as rows, features as
                columns.
            selections: Mapping from selection name to tuple of slice
                identifiers. Up to 4 selections are supported.
            dimensions: List of column names to include. If None, uses all
                columns.

        Returns:
            Plotly figure object ready for display or further customization.

        Raises:
            KeyError: If any dimension column is not in df_features.
            ValueError: If dimensions is empty or more than 4 selections
                are provided.
        """
        if dimensions is None:
            dimensions = list(df_features.columns)
        if len(dimensions) == 0:
            raise ValueError("dimensions list cannot be empty")
        if len(selections) > 4:
            raise ValueError("At most 4 selections can be compared")

        for dim in dimensions:
            if dim not in df_features.columns:
                raise KeyError(f"Column '{dim}' not found in df_features")

        splom_dims = [dict(label=d, values=df_features[d].values) for d in dimensions]
        labels = df_features.index.astype(str).tolist()

        fig = go.Figure()

        fig.add_trace(go.Splom(
            dimensions=splom_dims,
            marker=dict(color='lightgray', size=6, symbol='circle'),
            text=labels,
            name='All slices',
            diagonal_visible=False,
            showupperhalf=False,
        ))

        for i, (name, selection) in enumerate(selections.items()):
            sel_mask = df_features.index.isin(set(selection))
            sel_df = df_features.loc[sel_mask]
            sel_dims = [dict(label=d, values=sel_df[d].values) for d in dimensions]
            sel_labels = sel_df.index.astype(str).tolist()

            fig.add_trace(go.Splom(
                dimensions=sel_dims,
                marker=dict(
                    color=_COLORS[i % len(_COLORS)],
                    size=10,
                    symbol=_SYMBOLS[i % len(_SYMBOLS)],
                    line=dict(width=1, color='black'),
                ),
                text=sel_labels,
                name=name,
                diagonal_visible=False,
                showupperhalf=False,
            ))

        fig.update_layout(
            dragmode='select',
            hovermode='closest',
        )

        return fig
