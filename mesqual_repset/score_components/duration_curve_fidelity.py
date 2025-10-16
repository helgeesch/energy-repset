from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import numpy as np
import pandas as pd

from .base_score_component import ScoreComponent

if TYPE_CHECKING:
    from ..types import SliceCombination
    from ..context import ProblemContext


class DurationCurveFidelity(ScoreComponent):
    """Matches duration curves using quantile approximation and IQR normalization.

    Measures how well the selection preserves the statistical distribution
    of each variable by comparing quantiles of the full and selected data.
    This is more computationally efficient than NRMSEFidelity for large
    datasets since it compares a fixed number of quantiles rather than
    full sorted arrays.

    Uses IQR (interquartile range) normalization instead of mean normalization,
    making it more robust to outliers.

    Args:
        n_quantiles: Number of quantiles to compute for duration curve
            approximation. Default is 101 (0%, 1%, ..., 100%).
        variable_weights: Optional per-variable weights for prioritizing certain
            variables in the score. If None, all variables weighted equally (1.0).
            If specified, missing variables get weight 0.0.

    Examples:
        >>> from mesqual_repset.score_components import DurationCurveFidelity
        >>> from mesqual_repset.objectives import ObjectiveSet
        >>>
        >>> # Default: 101 quantiles (0%, 1%, ..., 100%)
        >>> objectives = ObjectiveSet({
        ...     'duration': (1.0, DurationCurveFidelity())
        ... })
        >>>
        >>> # Coarser approximation for faster computation
        >>> objectives = ObjectiveSet({
        ...     'duration': (1.0, DurationCurveFidelity(n_quantiles=21))
        ... })
        >>>
        >>> # With variable weights for prioritizing specific variables
        >>> objectives = ObjectiveSet({
        ...     'duration': (1.0, DurationCurveFidelity(
        ...         n_quantiles=101,
        ...         variable_weights={'demand': 2.0, 'solar': 1.0, 'wind': 0.5}
        ...     ))
        ... })
        >>> # demand has 2x impact, solar 1x, wind 0.5x, other variables 0x
    """

    def __init__(
        self,
        n_quantiles: int = 101,
        variable_weights: Dict[str, float] | None = None
    ) -> None:
        """Initialize duration curve fidelity component.

        Args:
            n_quantiles: Number of quantiles for duration curve approximation.
            variable_weights: Optional per-variable weights. If None, all
                variables weighted equally (1.0). If specified, missing
                variables get weight 0.0.
        """
        self.name = "nrmse_duration_curve"
        self.direction = "min"
        self.n_quantiles = n_quantiles
        self._requested_weights = variable_weights
        self.variable_weights: Dict[str, float] = None

    def prepare(self, context: ProblemContext) -> None:
        """Precompute quantiles and normalization factors for full dataset.

        Args:
            context: Problem context containing raw time-series data.
        """
        df = context.df_raw
        self.df = df
        self.labels = context.slicer.labels_for_index(df.index)
        self.vars = list(df.columns)

        self.variable_weights = self._default_weight_normalization(self._requested_weights, self.vars)

        self.quantiles = np.linspace(0, 1, self.n_quantiles)
        self.full_quantiles = self.df.quantile(self.quantiles)
        self.iqr = (df.quantile(0.75) - df.quantile(0.25)).replace(0, 1.0)

    def score(self, combination: SliceCombination) -> float:
        """Compute sum of per-variable NRMSE for quantile-based duration curves.

        Args:
            combination: Tuple of slice identifiers forming the selection.

        Returns:
            Weighted sum of per-variable NRMSE values using IQR normalization.
            Returns infinity if the selection is empty.
        """
        sel_mask = pd.Index(self.labels).isin(combination)
        if not sel_mask.any():
            return np.inf
        sel = self.df.loc[sel_mask]

        sel_quantiles = sel.quantile(self.quantiles)

        total_nrmse = 0.0
        for v in self.vars:
            squared_errors = (self.full_quantiles[v].values - sel_quantiles[v].values) ** 2
            rmse = np.sqrt(squared_errors.mean())
            total_nrmse += self.variable_weights[v] * (rmse / float(self.iqr[v]))

        return float(total_nrmse)
