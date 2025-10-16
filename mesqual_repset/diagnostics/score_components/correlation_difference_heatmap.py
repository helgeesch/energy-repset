from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


class CorrelationDifferenceHeatmap:
    """Visualize the difference between correlation matrices.

    Creates a heatmap showing the difference between the correlation matrix of
    the full dataset and the selection. This helps identify which variable
    relationships are well-preserved or poorly-preserved by the selection.
    Related to CorrelationFidelity score component.

    Positive values (red) indicate the selection has stronger correlation than
    the full dataset. Negative values (blue) indicate weaker correlation.

    Examples:

        >>> # Compare correlation structure
        >>> corr_diff = CorrelationDifferenceHeatmap()
        >>> full_data = context.df_raw[['demand', 'wind', 'solar']]
        >>> selected_indices = context.slicer.get_indices_for_slices(result.selection)
        >>> selected_data = context.df_raw.loc[selected_indices, ['demand', 'wind', 'solar']]
        >>> fig = corr_diff.plot(full_data, selected_data)
        >>> fig.update_layout(title='Correlation Difference: Selection - Full')
        >>> fig.show()

        >>> # With Spearman correlation
        >>> fig = corr_diff.plot(full_data, selected_data, method='spearman')

        >>> # Show only lower triangle
        >>> fig = corr_diff.plot(full_data, selected_data, show_lower_only=True)
    """

    def __init__(self):
        """Initialize the correlation difference heatmap diagnostic."""
        pass

    def plot(
        self,
        df_full: pd.DataFrame,
        df_selection: pd.DataFrame,
        method: str = 'pearson',
        show_lower_only: bool = False,
    ) -> go.Figure:
        """Create a heatmap of correlation differences.

        Args:
            df_full: DataFrame containing variables for the full dataset.
            df_selection: DataFrame containing variables for the selection.
                Must have the same columns as df_full.
            method: Correlation method ('pearson', 'spearman', or 'kendall').
                Default is 'pearson'.
            show_lower_only: If True, shows only the lower triangle of the
                difference matrix (removes redundant upper triangle and diagonal).

        Returns:
            Plotly figure object ready for display or further customization.

        Raises:
            ValueError: If method is invalid or columns don't match.
        """
        if method not in ['pearson', 'spearman', 'kendall']:
            raise ValueError(
                f"method must be 'pearson', 'spearman', or 'kendall', got '{method}'"
            )

        if not df_full.columns.equals(df_selection.columns):
            raise ValueError(
                "df_full and df_selection must have the same columns"
            )

        # Calculate correlation matrices
        corr_full = df_full.corr(method=method)
        corr_selection = df_selection.corr(method=method)

        # Calculate difference (selection - full)
        corr_diff = corr_selection - corr_full

        # Mask upper triangle if requested
        if show_lower_only:
            mask = pd.DataFrame(
                False,
                index=corr_diff.index,
                columns=corr_diff.columns
            )
            # Set upper triangle and diagonal to True (to be masked)
            for i in range(len(corr_diff)):
                for j in range(i, len(corr_diff)):
                    mask.iloc[i, j] = True

            # Apply mask by setting values to NaN
            corr_diff = corr_diff.where(~mask)

        # Determine color scale range (symmetric around 0)
        max_abs = max(abs(corr_diff.min().min()), abs(corr_diff.max().max()))
        if pd.isna(max_abs):
            max_abs = 1.0

        # Create heatmap
        fig = px.imshow(
            corr_diff,
            x=corr_diff.columns,
            y=corr_diff.index,
            color_continuous_scale='RdBu_r',
            color_continuous_midpoint=0,
            zmin=-max_abs,
            zmax=max_abs,
            aspect='auto',
        )

        # Update layout for better readability
        fig.update_layout(
            xaxis_title='',
            yaxis_title='',
            coloraxis_colorbar=dict(title='Δ Correlation<br>(Selection - Full)'),
        )

        # Improve text readability
        fig.update_traces(
            text=corr_diff.round(2).values,
            texttemplate='%{text}',
            textfont=dict(size=10),
        )

        return fig
