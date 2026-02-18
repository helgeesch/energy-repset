"""Visualization of responsibility weights from RepresentationModel."""

from typing import Dict, Hashable

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


class ResponsibilityBars:
    """Bar chart showing responsibility weights for selected representatives.

    Visualizes the weight distribution across selected periods as computed by
    a RepresentationModel. Each bar shows how much each representative
    contributes to the full dataset representation.

    Optionally displays a reference line showing uniform weights (1/k) for
    comparison with non-uniform weighting schemes like cluster-size based
    weights.

    Examples:

        >>> from energy_repset.diagnostics.results import ResponsibilityBars
        >>>
        >>> # After running workflow with result containing weights
        >>> weights = result.weights  # e.g., {Period('2024-01'): 0.35, ...}
        >>> bars = ResponsibilityBars()
        >>> fig = bars.plot(weights, show_uniform_reference=True)
        >>> fig.update_layout(title='Responsibility Weights')
        >>> fig.show()
    """

    def __init__(self):
        """Initialize ResponsibilityBars diagnostic."""
        pass

    def plot(
        self,
        weights: Dict[Hashable, float],
        show_uniform_reference: bool = True,
    ) -> go.Figure:
        """Create bar chart of responsibility weights.

        Args:
            weights: Dictionary mapping slice identifiers to their weights.
                Should sum to 1.0 for normalized weights.
            show_uniform_reference: If True, adds horizontal dashed line
                showing uniform weight (1/k) for comparison.

        Returns:
            Plotly figure with bar chart. X-axis shows slice labels, Y-axis
            shows weight values. Text labels show weights to 3 decimal places.

        Raises:
            ValueError: If weights dictionary is empty.
        """
        if not weights:
            raise ValueError("Weights dictionary cannot be empty")

        # Prepare data for plotting
        df = pd.DataFrame({
            'slice': [str(s) for s in weights.keys()],
            'weight': list(weights.values())
        })

        # Create bar chart
        fig = px.bar(
            df,
            x='slice',
            y='weight',
            text='weight'
        )

        # Format text labels to 3 decimal places, position outside bars
        fig.update_traces(
            texttemplate='%{y:.3f}',
            textposition='outside'
        )

        # Set y-axis range and label
        fig.update_yaxes(
            range=[0, max(df['weight']) * 1.15],  # Add headroom for text labels
            title='Responsibility Weight'
        )

        fig.update_xaxes(title='Selected Period')

        # Add uniform reference line if requested
        if show_uniform_reference and len(weights) > 0:
            uniform_weight = 1.0 / len(weights)
            fig.add_hline(
                y=uniform_weight,
                line_dash='dot',
                annotation_text=f'Uniform ({uniform_weight:.3f})',
                annotation_position='top left'
            )

        return fig