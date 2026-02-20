from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


class DistributionOverlayECDF:
    """Overlay empirical cumulative distribution functions (ECDF) to compare distributions.

    Creates a plot showing the ECDF of a variable for both the full dataset and
    a selection. This helps visualize how well the selection represents the full
    distribution, which is what WassersteinFidelity measures.

    Examples:

        >>> # Compare demand distribution
        >>> ecdf_plot = DistributionOverlayECDF()
        >>> full_data = context.df_raw['demand']
        >>> selected_indices = context.slicer.get_indices_for_slices(result.selection)
        >>> selected_data = context.df_raw.loc[selected_indices, 'demand']
        >>> fig = ecdf_plot.plot(full_data, selected_data)
        >>> fig.update_layout(title='Demand Distribution: Full vs Selected')
        >>> fig.show()

        >>> # Alternative: using iloc
        >>> selection_mask = context.df_raw.index.isin(selected_indices)
        >>> fig = ecdf_plot.plot(
        ...     context.df_raw['wind'],
        ...     context.df_raw.loc[selection_mask, 'wind']
        ... )
    """

    def __init__(self):
        """Initialize the ECDF overlay diagnostic."""
        pass

    def plot(
        self,
        df_full: pd.Series,
        df_selection: pd.Series,
        full_label: str = 'Full',
        selection_label: str = 'Selection',
    ) -> go.Figure:
        """Create an ECDF overlay plot.

        Args:
            df_full: Series containing values for the full dataset.
            df_selection: Series containing values for the selection.
            full_label: Label for the full dataset in the legend. Default 'Full'.
            selection_label: Label for the selection in the legend. Default 'Selection'.

        Returns:
            Plotly figure object ready for display or further customization.
        """
        # Drop NaN values
        full_values = df_full.dropna().values
        selection_values = df_selection.dropna().values

        # Calculate ECDF for full dataset
        full_sorted = np.sort(full_values)
        full_ecdf = np.arange(1, len(full_sorted) + 1) / len(full_sorted)

        # Calculate ECDF for selection
        selection_sorted = np.sort(selection_values)
        selection_ecdf = np.arange(1, len(selection_sorted) + 1) / len(selection_sorted)

        # Create figure
        fig = go.Figure()

        # Add full dataset ECDF
        fig.add_trace(go.Scatter(
            x=full_sorted,
            y=full_ecdf,
            mode='lines',
            name=full_label,
            line=dict(width=2),
        ))

        # Add selection ECDF
        fig.add_trace(go.Scatter(
            x=selection_sorted,
            y=selection_ecdf,
            mode='lines',
            name=selection_label,
            line=dict(width=2, dash='dash'),
        ))

        # Update layout
        fig.update_layout(
            xaxis_title=df_full.name or 'Value',
            yaxis_title='Cumulative Probability',
            hovermode='x unified',
            yaxis=dict(tickformat='.0%', range=[0, 1]),
        )

        return fig


class DistributionOverlayHistogram:
    """Overlay histograms to compare distributions.

    Creates a plot showing normalized histograms of a variable for both the
    full dataset and a selection. Alternative to ECDF that may be more intuitive
    for some users. Shows probability density rather than cumulative probability.

    Examples:

        >>> # Compare demand distribution
        >>> hist_plot = DistributionOverlayHistogram()
        >>> full_data = context.df_raw['demand']
        >>> selected_indices = context.slicer.get_indices_for_slices(result.selection)
        >>> selected_data = context.df_raw.loc[selected_indices, 'demand']
        >>> fig = hist_plot.plot(full_data, selected_data)
        >>> fig.update_layout(title='Demand Distribution: Full vs Selected')
        >>> fig.show()

        >>> # With custom bin count
        >>> fig = hist_plot.plot(full_data, selected_data, nbins=50)

        >>> # Using density mode
        >>> fig = hist_plot.plot(full_data, selected_data, histnorm='probability density')
    """

    def __init__(self):
        """Initialize the histogram overlay diagnostic."""
        pass

    def plot(
        self,
        df_full: pd.Series,
        df_selection: pd.Series,
        nbins: int = 30,
        histnorm: str = 'probability',
        full_label: str = 'Full',
        selection_label: str = 'Selection',
    ) -> go.Figure:
        """Create a histogram overlay plot.

        Args:
            df_full: Series containing values for the full dataset.
            df_selection: Series containing values for the selection.
            nbins: Number of bins for the histogram. Default is 30.
            histnorm: Histogram normalization mode. Options: 'probability',
                'probability density', 'percent'. Default is 'probability'.
            full_label: Label for the full dataset in the legend. Default 'Full'.
            selection_label: Label for the selection in the legend. Default 'Selection'.

        Returns:
            Plotly figure object ready for display or further customization.

        Raises:
            ValueError: If histnorm is not a valid option.
        """
        valid_histnorms = ['probability', 'probability density', 'percent', '']
        if histnorm not in valid_histnorms:
            raise ValueError(
                f"histnorm must be one of {valid_histnorms}, got '{histnorm}'"
            )

        # Drop NaN values
        full_values = df_full.dropna().values
        selection_values = df_selection.dropna().values

        # Create figure
        fig = go.Figure()

        # Add full dataset histogram
        fig.add_trace(go.Histogram(
            x=full_values,
            name=full_label,
            nbinsx=nbins,
            histnorm=histnorm,
            opacity=0.6,
        ))

        # Add selection histogram
        fig.add_trace(go.Histogram(
            x=selection_values,
            name=selection_label,
            nbinsx=nbins,
            histnorm=histnorm,
            opacity=0.6,
        ))

        # Update layout
        yaxis_title = {
            'probability': 'Probability',
            'probability density': 'Probability Density',
            'percent': 'Percent',
            '': 'Count',
        }.get(histnorm, 'Frequency')

        fig.update_layout(
            xaxis_title=df_full.name or 'Value',
            yaxis_title=yaxis_title,
            barmode='overlay',
            hovermode='x unified',
        )

        return fig


class DistributionOverlayECDFGrid:
    """Faceted ECDF grid comparing full dataset and selection across all variables.

    Creates a grid of ECDF subplots — one per variable — so that distribution
    fidelity can be assessed at a glance for the entire dataset. This is a
    multi-variable extension of ``DistributionOverlayECDF``.

    Examples:

        >>> ecdf_grid = DistributionOverlayECDFGrid()
        >>> selected_idx = context.slicer.get_indices_for_slice_combi(
        ...     context.df_raw.index, result.selection
        ... )
        >>> fig = ecdf_grid.plot(context.df_raw, context.df_raw.loc[selected_idx])
        >>> fig.show()
    """

    def __init__(self):
        """Initialize the ECDF grid diagnostic."""
        pass

    def plot(
        self,
        df_full: pd.DataFrame,
        df_selection: pd.DataFrame,
        variables: list[str] | None = None,
        cols: int = 2,
        full_label: str = 'Full',
        selection_label: str = 'Selection',
    ) -> go.Figure:
        """Create a faceted ECDF grid.

        Args:
            df_full: Full dataset with variables as columns.
            df_selection: Selected subset with the same columns.
            variables: Columns to include. If None, uses all columns from
                df_full.
            cols: Number of columns in the facet grid.
            full_label: Legend label for the full dataset.
            selection_label: Legend label for the selection.

        Returns:
            Plotly figure object ready for display or further customization.

        Raises:
            KeyError: If any variable is not found in both DataFrames.
        """
        if variables is None:
            variables = list(df_full.columns)

        for var in variables:
            if var not in df_full.columns:
                raise KeyError(f"Column '{var}' not found in df_full")
            if var not in df_selection.columns:
                raise KeyError(f"Column '{var}' not found in df_selection")

        full_melted = df_full[variables].melt(var_name='variable', value_name='value')
        full_melted['source'] = full_label

        sel_melted = df_selection[variables].melt(var_name='variable', value_name='value')
        sel_melted['source'] = selection_label

        combined = pd.concat([full_melted, sel_melted], ignore_index=True)

        fig = px.ecdf(
            combined,
            x='value',
            color='source',
            facet_col='variable',
            facet_col_wrap=cols,
        )

        fig.update_yaxes(tickformat='.0%')
        fig.update_layout(hovermode='x unified')

        return fig
