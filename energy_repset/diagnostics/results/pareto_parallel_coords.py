from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go

if TYPE_CHECKING:
    from ...search_algorithms.objective_driven import ObjectiveDrivenCombinatorialSearchAlgorithm


class ParetoParallelCoordinates:
    """Parallel coordinates plot of Pareto front.

    Visualizes multi-objective trade-offs using parallel coordinates where each
    vertical axis represents one objective. Lines connecting axes show individual
    solutions, with Pareto-optimal solutions highlighted.

    Args:
        objectives: List of objective names to include (None = all objectives).

    Examples:
        >>> from energy_repset.diagnostics.results import ParetoParallelCoordinates
        >>> parallel = ParetoParallelCoordinates()
        >>> fig = parallel.plot(search_algorithm=workflow.search_algorithm)
        >>> fig.update_layout(title='Pareto Front: Parallel Coordinates')
        >>> fig.show()
    """

    def __init__(self, objectives: list[str] | None = None):
        """Initialize parallel coordinates diagnostic.

        Args:
            objectives: List of objective names to include (None = all).
        """
        self.objectives = objectives

    def plot(
        self,
        search_algorithm: ObjectiveDrivenCombinatorialSearchAlgorithm,
    ) -> go.Figure:
        """Create parallel coordinates plot of Pareto front.

        Args:
            search_algorithm: Search algorithm after find_selection() has been called.

        Returns:
            Plotly figure with parallel coordinates plot.

        Raises:
            ValueError: If find_selection() hasn't been called.
        """
        df = search_algorithm.get_all_scores()

        if self.objectives is None:
            obj_cols = [col for col in df.columns if col not in ['slices', 'label']]
        else:
            obj_cols = self.objectives
            for obj in obj_cols:
                if obj not in df.columns:
                    raise ValueError(f"Objective '{obj}' not found in scores")

        if len(obj_cols) < 2:
            raise ValueError("Need at least 2 objectives for parallel coordinates")

        has_pareto = hasattr(search_algorithm.selection_policy, 'pareto_mask')
        pareto_mask = None
        feasible_mask = None

        if has_pareto and search_algorithm.selection_policy.pareto_mask is not None:
            pareto_mask = search_algorithm.selection_policy.pareto_mask
            feasible_mask = search_algorithm.selection_policy.feasible_mask

        dimensions = []
        for obj in obj_cols:
            dimensions.append(dict(
                label=obj,
                values=df[obj]
            ))

        if has_pareto and pareto_mask is not None:
            pareto = pareto_mask.values
            feasible = feasible_mask.values

            color_values = []
            for i in range(len(df)):
                if not feasible[i]:
                    color_values.append(0)
                elif pareto[i]:
                    color_values.append(2)
                else:
                    color_values.append(1)

            fig = go.Figure(data=go.Parcoords(
                dimensions=dimensions,
                line=dict(
                    color=color_values,
                    colorscale=[
                        [0, 'lightgray'],
                        [0.5, 'steelblue'],
                        [1, 'darkorange']
                    ],
                    showscale=True,
                    cmin=0,
                    cmax=2,
                    colorbar=dict(
                        title='Status',
                        tickvals=[0, 1, 2],
                        ticktext=['Infeasible', 'Dominated', 'Pareto'],
                    )
                )
            ))
        else:
            fig = go.Figure(data=go.Parcoords(
                dimensions=dimensions,
                line=dict(
                    color='steelblue',
                    showscale=False,
                )
            ))

        fig.update_layout(
            title='Parallel Coordinates: All Objectives',
            height=500,
        )

        return fig
