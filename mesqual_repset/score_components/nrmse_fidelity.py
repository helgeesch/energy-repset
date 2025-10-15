from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import numpy as np
import pandas as pd

from .base_score_component import ScoreComponent

if TYPE_CHECKING:
    from ..types import SliceCombination
    from ..context import ProblemContext


class NRMSEFidelity(ScoreComponent):
    """Matches duration curves using interpolation and NRMSE.

    Measures how well the selection preserves the statistical distribution
    of each variable by comparing full and selected duration curves (sorted
    value profiles). The selection's duration curve is interpolated to match
    the full curve's length, then NRMSE is computed.

    This approach uses the full sorted arrays and is accurate but can be
    computationally expensive for very large datasets. For efficiency with
    large data, consider DurationCurveFidelity which uses quantiles.

    Args:
        variable_weights: Optional per-variable weights. If None, all
            variables weighted equally.

    Examples:
        >>> from mesqual_repset.score_components import NRMSEFidelity
        >>> from mesqual_repset.objectives import ObjectiveSet
        >>>
        >>> # Basic usage with equal variable weights
        >>> objectives = ObjectiveSet({
        ...     'nrmse': (1.0, NRMSEFidelity())
        ... })
        >>>
        >>> # Prioritize specific variables
        >>> objectives = ObjectiveSet({
        ...     'nrmse': (1.0, NRMSEFidelity(
        ...         variable_weights={'demand': 2.0, 'solar': 1.0}
        ...     ))
        ... })
    """

    def __init__(self, variable_weights: Dict[str, float] | None = None) -> None:
        """Initialize NRMSE fidelity component.

        Args:
            variable_weights: Optional dict mapping variable names to weights.
                Missing variables default to weight 1.0.
        """
        self.name = "nrmse"
        self.direction = "min"
        self.variable_weights = variable_weights

    def prepare(self, context: ProblemContext) -> None:
        """Precompute full duration curves and normalization factors.

        Args:
            context: Problem context containing raw time-series data.
        """
        df = context.df_raw
        self.df = df
        self.labels = context.slicer.labels_for_index(df.index)
        self.vars = list(df.columns)

        if self.variable_weights is None:
            self.variable_weights = {v: 1.0 for v in self.vars}
        for v in self.vars:
            if v not in self.variable_weights:
                self.variable_weights[v] = 1.0

        self.full_curves = {
            v: np.sort(df[v].values)[::-1] for v in self.vars
        }
        self.full_means = {
            v: np.mean(df[v].values) for v in self.vars
        }

    def score(self, combination: SliceCombination) -> float:
        """Compute sum of per-variable NRMSE for duration curves.

        Args:
            combination: Tuple of slice identifiers forming the selection.

        Returns:
            Weighted sum of per-variable NRMSE values. Returns infinity
            if the selection is empty.
        """
        sel_mask = pd.Index(self.labels).isin(combination)
        if not sel_mask.any():
            return np.inf

        sel = self.df.loc[sel_mask]
        s = 0.0

        for v in self.vars:
            full_curve = self.full_curves[v]
            sel_curve = np.sort(sel[v].values)[::-1]

            if len(sel_curve) == 0:
                continue

            # Interpolate selection's duration curve to match full length
            x_full = np.linspace(0, 1, len(full_curve))
            x_sel = np.linspace(0, 1, len(sel_curve))
            resampled_sel_curve = np.interp(x_full, x_sel, sel_curve)

            # Calculate RMSE
            mse = np.mean((full_curve - resampled_sel_curve) ** 2)
            rmse = np.sqrt(mse)

            # Normalize by mean
            mean_val = self.full_means[v]
            nrmse = rmse / (mean_val + 1e-12)

            s += self.variable_weights[v] * nrmse

        return float(s)
