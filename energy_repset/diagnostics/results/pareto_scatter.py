from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Hashable

import plotly.graph_objects as go
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    from ...search_algorithms.objective_driven import ObjectiveDrivenCombinatorialSearchAlgorithm
    from ...types import SliceCombination


class ParetoScatter2D:
    """2D scatter plot of all evaluated combinations with Pareto front highlighted.

    Visualizes the objective space for two objectives, showing:
    - All evaluated combinations as scatter points
    - Pareto-optimal solutions highlighted
    - Selected combination (if provided) marked distinctly
    - Feasible vs infeasible solutions (if constraints exist)

    Args:
        objective_x: Name of objective for x-axis.
        objective_y: Name of objective for y-axis.

    Examples:
        >>> from energy_repset.diagnostics.results import ParetoScatter2D
        >>> scatter = ParetoScatter2D(objective_x='wasserstein', objective_y='correlation')
        >>> fig = scatter.plot(
        ...     search_algorithm=workflow.search_algorithm,
        ...     selected_combination=result.selection
        ... )
        >>> fig.update_layout(title='Pareto Front: Wasserstein vs Correlation')
        >>> fig.show()
    """

    def __init__(self, objective_x: str, objective_y: str):
        """Initialize Pareto scatter diagnostic.

        Args:
            objective_x: Name of objective for x-axis.
            objective_y: Name of objective for y-axis.
        """
        self.objective_x = objective_x
        self.objective_y = objective_y

    def plot(
        self,
        search_algorithm: ObjectiveDrivenCombinatorialSearchAlgorithm,
        selected_combination: SliceCombination | None = None,
    ) -> go.Figure:
        """Create 2D scatter plot of Pareto front.

        Args:
            search_algorithm: Search algorithm after find_selection() has been called.
            selected_combination: Optional combination to highlight (e.g., result.selection).

        Returns:
            Plotly figure with scatter plot.

        Raises:
            ValueError: If find_selection() hasn't been called or objectives not found.
        """
        df = search_algorithm.get_all_scores()

        if self.objective_x not in df.columns:
            raise ValueError(f"Objective '{self.objective_x}' not found in scores")
        if self.objective_y not in df.columns:
            raise ValueError(f"Objective '{self.objective_y}' not found in scores")

        has_pareto = hasattr(search_algorithm.selection_policy, 'pareto_mask')
        pareto_mask = None
        feasible_mask = None

        if has_pareto and search_algorithm.selection_policy.pareto_mask is not None:
            pareto_mask = search_algorithm.selection_policy.pareto_mask
            feasible_mask = search_algorithm.selection_policy.feasible_mask

        fig = go.Figure()

        x_vals = df[self.objective_x]
        y_vals = df[self.objective_y]

        if has_pareto and pareto_mask is not None:
            pareto = pareto_mask.values
            feasible = feasible_mask.values

            infeasible = ~feasible
            if infeasible.any():
                fig.add_trace(go.Scatter(
                    x=x_vals[infeasible],
                    y=y_vals[infeasible],
                    mode='markers',
                    marker=dict(size=6, opacity=0.3),
                    name='Infeasible',
                    hovertemplate=(
                        f'{self.objective_x}: %{{x:.4f}}<br>'
                        f'{self.objective_y}: %{{y:.4f}}<br>'
                        '<extra></extra>'
                    ),
                ))

            dominated = feasible & ~pareto
            if dominated.any():
                fig.add_trace(go.Scatter(
                    x=x_vals[dominated],
                    y=y_vals[dominated],
                    mode='markers',
                    marker=dict(size=6, opacity=0.5),
                    name='Dominated',
                    hovertemplate=(
                        f'{self.objective_x}: %{{x:.4f}}<br>'
                        f'{self.objective_y}: %{{y:.4f}}<br>'
                        '<extra></extra>'
                    ),
                ))

            pareto_points = pareto & feasible
            if pareto_points.any():
                fig.add_trace(go.Scatter(
                    x=x_vals[pareto_points],
                    y=y_vals[pareto_points],
                    mode='markers',
                    marker=dict(size=10, symbol='diamond'),
                    name='Pareto Front',
                    hovertemplate=(
                        f'{self.objective_x}: %{{x:.4f}}<br>'
                        f'{self.objective_y}: %{{y:.4f}}<br>'
                        '<extra></extra>'
                    ),
                ))
        else:
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='markers',
                marker=dict(size=6, opacity=0.5),
                name='All Combinations',
                hovertemplate=(
                    f'{self.objective_x}: %{{x:.4f}}<br>'
                    f'{self.objective_y}: %{{y:.4f}}<br>'
                    '<extra></extra>'
                ),
            ))

        if selected_combination is not None:
            selected_idx = df['slices'].apply(lambda x: x == selected_combination)
            if selected_idx.any():
                sel_x = x_vals[selected_idx].values[0]
                sel_y = y_vals[selected_idx].values[0]
                fig.add_trace(go.Scatter(
                    x=[sel_x],
                    y=[sel_y],
                    mode='markers',
                    marker=dict(
                        size=15,
                        symbol='star',
                        line=dict(width=2, color='black')
                    ),
                    name='Selected',
                    hovertemplate=(
                        f'{self.objective_x}: %{{x:.4f}}<br>'
                        f'{self.objective_y}: %{{y:.4f}}<br>'
                        '<b>SELECTED</b><br>'
                        '<extra></extra>'
                    ),
                ))

        fig.update_layout(
            xaxis_title=self.objective_x,
            yaxis_title=self.objective_y,
            hovermode='closest',
            showlegend=True,
        )

        return fig


class ParetoScatterMatrix:
    """Scatter matrix of all objectives showing Pareto front.

    Creates a scatter plot matrix (SPLOM) showing pairwise relationships between
    all objectives. Each subplot shows two objectives with Pareto front highlighted.

    Args:
        objectives: List of objective names to include (None = all objectives).

    Examples:
        >>> from energy_repset.diagnostics.results import ParetoScatterMatrix
        >>> scatter_matrix = ParetoScatterMatrix(
        ...     objectives=['wasserstein', 'correlation', 'diurnal']
        ... )
        >>> fig = scatter_matrix.plot(
        ...     search_algorithm=workflow.search_algorithm,
        ...     selected_combination=result.selection
        ... )
        >>> fig.update_layout(title='Pareto Front: All Objectives')
        >>> fig.show()
    """

    def __init__(self, objectives: list[str] | None = None):
        """Initialize Pareto scatter matrix diagnostic.

        Args:
            objectives: List of objective names to include (None = all).
        """
        self.objectives = objectives

    def plot(
        self,
        search_algorithm: ObjectiveDrivenCombinatorialSearchAlgorithm,
        selected_combination: SliceCombination | None = None,
    ) -> go.Figure:
        """Create scatter matrix of Pareto front.

        Args:
            search_algorithm: Search algorithm after find_selection() has been called.
            selected_combination: Optional combination to highlight.

        Returns:
            Plotly figure with scatter matrix.

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
            raise ValueError("Need at least 2 objectives for scatter matrix")

        has_pareto = hasattr(search_algorithm.selection_policy, 'pareto_mask')
        pareto_mask = None
        feasible_mask = None

        if has_pareto and search_algorithm.selection_policy.pareto_mask is not None:
            pareto_mask = search_algorithm.selection_policy.pareto_mask
            feasible_mask = search_algorithm.selection_policy.feasible_mask

        color_col = None
        if has_pareto and pareto_mask is not None:
            df_plot = df.copy()
            pareto = pareto_mask.values
            feasible = feasible_mask.values
            df_plot['category'] = 'Dominated'
            df_plot.loc[~feasible, 'category'] = 'Infeasible'
            df_plot.loc[pareto & feasible, 'category'] = 'Pareto Front'
            color_col = 'category'
        else:
            df_plot = df.copy()

        dimensions = []
        for obj in obj_cols:
            dimensions.append(dict(
                label=obj,
                values=df_plot[obj]
            ))

        fig = go.Figure(data=go.Splom(
            dimensions=dimensions,
            marker=dict(
                size=5,
                color=df_plot[color_col].map({
                    'Infeasible': 0,
                    'Dominated': 1,
                    'Pareto Front': 2
                }) if color_col else None,
                colorscale=[[0, 'lightgray'], [0.5, 'steelblue'], [1, 'darkorange']] if color_col else None,
                showscale=False,
                line=dict(width=0.5, color='white')
            ),
            text=df_plot['label'] if 'label' in df_plot else None,
            diagonal_visible=False,
            showupperhalf=False,
        ))

        if selected_combination is not None:
            selected_idx = df['slices'].apply(lambda x: x == selected_combination)
            if selected_idx.any():
                selected_vals = [df_plot.loc[selected_idx, obj].values[0] for obj in obj_cols]
                n_dims = len(obj_cols)
                for i in range(n_dims):
                    for j in range(i):
                        xaxis = f'x{j*n_dims + i + 1}' if (j*n_dims + i) > 0 else 'x'
                        yaxis = f'y{j*n_dims + i + 1}' if (j*n_dims + i) > 0 else 'y'

        fig.update_layout(
            title='Scatter Matrix: All Objectives',
            height=150 * len(obj_cols),
            width=150 * len(obj_cols),
            showlegend=False,
        )

        return fig
