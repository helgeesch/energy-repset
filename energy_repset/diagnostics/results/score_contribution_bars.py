from __future__ import annotations

from typing import Dict

import plotly.graph_objects as go


class ScoreContributionBars:
    """Bar chart showing final scores from each objective component.

    Visualizes the contribution of each score component to understand which
    objectives were most influential in the final selection. Can display
    absolute scores or normalized as fractions of total.

    Examples:
        >>> from energy_repset.diagnostics.results import ScoreContributionBars
        >>> contrib = ScoreContributionBars()
        >>> fig = contrib.plot(result.scores, normalize=True)
        >>> fig.update_layout(title='Score Component Contributions')
        >>> fig.show()
    """

    def plot(
        self,
        scores: Dict[str, float],
        normalize: bool = False
    ) -> go.Figure:
        """Create bar chart of score component contributions.

        Args:
            scores: Dictionary of scores from each component (from result.scores).
            normalize: If True, show as fractions of total score.

        Returns:
            Plotly figure with bar chart.
        """
        if not scores:
            raise ValueError("Scores dictionary is empty")

        component_names = list(scores.keys())
        score_values = list(scores.values())

        if normalize:
            total = sum(score_values)
            if total == 0:
                raise ValueError("Cannot normalize: total score is zero")
            score_values = [v / total for v in score_values]
            y_title = 'Normalized Score (fraction)'
        else:
            y_title = 'Score Value'

        fig = go.Figure(data=[
            go.Bar(
                x=component_names,
                y=score_values,
                text=[f'{v:.4f}' for v in score_values],
                textposition='auto',
            )
        ])

        fig.update_layout(
            xaxis_title='Score Component',
            yaxis_title=y_title,
            showlegend=False,
            hovermode='x',
        )

        return fig
