from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


class DiurnalProfileOverlay:
    """Overlay mean diurnal (hour-of-day) profiles for full vs selected data.

    Creates a plot showing the average value by hour of day for each variable,
    comparing the full dataset to the selection. This helps visualize how well
    the selection preserves daily patterns, which is related to DiurnalFidelity
    score component.

    Examples:

        >>> # Compare diurnal patterns
        >>> diurnal_plot = DiurnalProfileOverlay()
        >>> full_data = context.df_raw[['demand', 'wind', 'solar']]
        >>> selected_indices = context.slicer.get_indices_for_slices(result.selection)
        >>> selected_data = context.df_raw.loc[selected_indices, ['demand', 'wind', 'solar']]
        >>> fig = diurnal_plot.plot(full_data, selected_data)
        >>> fig.update_layout(title='Diurnal Profiles: Full vs Selected')
        >>> fig.show()

        >>> # Single variable
        >>> fig = diurnal_plot.plot(
        ...     full_data[['demand']],
        ...     selected_data[['demand']]
        ... )

        >>> # Subset of variables
        >>> fig = diurnal_plot.plot(
        ...     full_data,
        ...     selected_data,
        ...     variables=['demand', 'wind']
        ... )
    """

    def __init__(self):
        """Initialize the diurnal profile overlay diagnostic."""
        pass

    def plot(
        self,
        df_full: pd.DataFrame,
        df_selection: pd.DataFrame,
        variables: list[str] = None,
        full_label: str = 'Full',
        selection_label: str = 'Selection',
    ) -> go.Figure:
        """Create a diurnal profile overlay plot.

        Args:
            df_full: DataFrame with DatetimeIndex and variable columns for full dataset.
            df_selection: DataFrame with DatetimeIndex and variable columns for selection.
                Must have the same columns as df_full.
            variables: List of variable names to include. If None, uses all columns.
            full_label: Label suffix for full dataset traces. Default 'Full'.
            selection_label: Label suffix for selection traces. Default 'Selection'.

        Returns:
            Plotly figure object ready for display or further customization.

        Raises:
            ValueError: If DataFrames don't have DatetimeIndex or columns don't match.
        """
        if not isinstance(df_full.index, pd.DatetimeIndex):
            raise ValueError("df_full must have a DatetimeIndex")
        if not isinstance(df_selection.index, pd.DatetimeIndex):
            raise ValueError("df_selection must have a DatetimeIndex")
        if not df_full.columns.equals(df_selection.columns):
            raise ValueError("df_full and df_selection must have the same columns")

        # Determine which variables to plot
        if variables is None:
            variables = list(df_full.columns)
        else:
            # Validate requested variables
            missing = set(variables) - set(df_full.columns)
            if missing:
                raise ValueError(f"Variables not found in DataFrames: {missing}")

        # Extract hour from index
        df_full_with_hour = df_full[variables].copy()
        df_full_with_hour['hour'] = df_full.index.hour

        df_selection_with_hour = df_selection[variables].copy()
        df_selection_with_hour['hour'] = df_selection.index.hour

        # Calculate mean profiles
        full_profile = df_full_with_hour.groupby('hour').mean(numeric_only=True)
        selection_profile = df_selection_with_hour.groupby('hour').mean(numeric_only=True)

        # Create figure
        fig = go.Figure()

        # Add traces for each variable
        for variable in variables:
            # Full dataset trace
            fig.add_trace(go.Scatter(
                x=full_profile.index,
                y=full_profile[variable],
                mode='lines+markers',
                name=f'{variable} ({full_label})',
                line=dict(width=2),
                marker=dict(size=6),
            ))

            # Selection trace
            fig.add_trace(go.Scatter(
                x=selection_profile.index,
                y=selection_profile[variable],
                mode='lines+markers',
                name=f'{variable} ({selection_label})',
                line=dict(width=2, dash='dash'),
                marker=dict(size=6, symbol='diamond'),
            ))

        # Update layout
        fig.update_layout(
            xaxis_title='Hour of Day',
            yaxis_title='Mean Value',
            hovermode='x unified',
            xaxis=dict(
                tickmode='linear',
                tick0=0,
                dtick=2,
                range=[-0.5, 23.5],
            ),
        )

        return fig
