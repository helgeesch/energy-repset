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
    """
    Matches per-variable distributions between full data and selected subset using 1D Wasserstein distance per variable.

    Parameters
    ----------
    variable_weights
        Optional per-variable weights applied inside the component.
    """

    def __init__(self, variable_weights: Dict[str, float] | None = None) -> None:
        self.name = "wasserstein"
        self.direction = "min"
        self.variable_weights = variable_weights

        self.df: pd.DataFrame = None
        self.labels = None
        self.vars = None
        self.iqr = None

    def prepare(self, context: ProblemContext) -> None:
        """
        Initialize state needed for evaluating this component.

        Parameters
        ----------
        context
            Your problems context.
        """
        df = context.df_raw.copy()
        slicer = context.slicer

        self.df = df
        self.labels = slicer.labels_for_index(df.index)
        self.vars = list(df.columns)
        self.iqr = (df.quantile(0.75) - df.quantile(0.25)).replace(0, 1.0)
        if self.variable_weights is None:
            self.variable_weights = {v: 1.0 for v in self.vars}
        for v in self.vars:
            if v not in self.variable_weights:
                self.variable_weights[v] = 1.0

    def score(self, combination: SliceCombination) -> float:
        """
        Compute the normalized Wasserstein fidelity score.

        Parameters
        ----------
        combination
            Slice labels forming the selection.

        Returns
        -------
        float
            Sum of per-variable distances normalized by IQR.
        """
        sel_mask = pd.Index(self.labels).isin(combination)
        sel = self.df.loc[sel_mask]
        s = 0.0
        for v in self.vars:
            s += self.variable_weights[v] * (wasserstein_distance(self.df[v].values, sel[v].values) / float(self.iqr[v]))
        return float(s)
