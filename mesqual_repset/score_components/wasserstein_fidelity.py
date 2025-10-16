from __future__ import annotations
from typing import TYPE_CHECKING, Dict
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from .base_score_component import ScoreComponent

if TYPE_CHECKING:
    from ..types import SliceCombination
    from ..context import ProblemContext


class WassersteinFidelity(ScoreComponent):
    """Measures distribution similarity using 1D Wasserstein distance per variable.

    Computes the Earth Mover's Distance between the full dataset's distribution
    and the selected subset's distribution for each variable. Distances are
    normalized by the interquartile range (IQR) to make them scale-invariant
    and comparable across variables.

    Lower scores indicate better distribution matching. This component is
    particularly effective for preserving statistical properties of the data.

    Args:
        variable_weights: Optional per-variable weights for prioritizing certain
            variables in the score. If None, all variables weighted equally (1.0).
            If specified, missing variables get weight 0.0.

    Examples:
        >>> # Equal weights (default)
        >>> component = WassersteinFidelity()
        >>> component.prepare(context)
        >>> score = component.score((0, 3, 6, 9))
        >>> print(f"Wasserstein distance: {score:.3f}")

        >>> # With variable-specific weights
        >>> component = WassersteinFidelity(
        ...     variable_weights={'demand': 2.0, 'solar': 1.0, 'wind': 0.5}
        ... )
        >>> component.prepare(context)
        >>> score = component.score((0, 3, 6, 9))
        >>> # demand has 2x impact, solar 1x, wind 0.5x, other variables 0x
    """

    def __init__(self, variable_weights: Dict[str, float] | None = None) -> None:
        """Initialize Wasserstein fidelity component.

        Args:
            variable_weights: Optional per-variable weights. If None, all
                variables weighted equally (1.0). If specified, missing
                variables get weight 0.0.
        """
        self.name = "wasserstein"
        self.direction = "min"
        self._requested_weights = variable_weights

        self.df: pd.DataFrame = None
        self.labels = None
        self.vars = None
        self.iqr = None
        self.variable_weights: Dict[str, float] = None

    def prepare(self, context: ProblemContext) -> None:
        """Precompute reference distributions and normalization factors.

        Args:
            context: Problem context with raw time-series data.
        """
        df = context.df_raw.copy()
        slicer = context.slicer

        self.df = df
        self.labels = slicer.labels_for_index(df.index)
        self.vars = list(df.columns)
        self.iqr = (df.quantile(0.75) - df.quantile(0.25)).replace(0, 1.0)

        # Normalize weights: None → equal (1.0), specified → use values (missing get 0.0)
        if self._requested_weights is None:
            self.variable_weights = {v: 1.0 for v in self.vars}
        else:
            self.variable_weights = {v: self._requested_weights.get(v, 0.0) for v in self.vars}

    def score(self, combination: SliceCombination) -> float:
        """Compute normalized Wasserstein distance between full and selection.

        Args:
            combination: Slice identifiers forming the selection.

        Returns:
            Sum of per-variable Wasserstein distances, each normalized by IQR
            and weighted according to variable_weights. Lower is better.
        """
        sel_mask = pd.Index(self.labels).isin(combination)
        sel = self.df.loc[sel_mask]
        s = 0.0
        for v in self.vars:
            s += self.variable_weights[v] * (wasserstein_distance(self.df[v].values, sel[v].values) / float(self.iqr[v]))
        return float(s)
