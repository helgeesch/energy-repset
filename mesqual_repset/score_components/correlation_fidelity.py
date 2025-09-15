from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .base_score_component import ScoreComponent

if TYPE_CHECKING:
    from ..context import ProblemContext
    from ..types import SliceCombination


class CorrelationFidelity(ScoreComponent):
    """
    Preserves correlation structure by penalizing the Frobenius norm of the difference between full and subset correlation matrices.
    """

    def __init__(self) -> None:
        self.name = "correlation"
        self.direction = "min"

    def prepare(self, context: ProblemContext) -> None:
        """
        Precompute the full correlation matrix.

        Parameters
        ----------
        context
            Your problems context.
        """
        df = context.df_raw.copy()
        slicer = context.slicer
        self.df = df
        self.labels = slicer.labels_for_index(df.index)
        self.full_corr = df.corr()


    def score(self, combination: SliceCombination) -> float:
        """
        Compute the correlation mismatch score.

        Parameters
        ----------
        combination
            Slice labels forming the selection.

        Returns
        -------
        float
            Relative Frobenius norm of the correlation difference.
        """
        sel = self.df.loc[pd.Index(self.labels).isin(combination)]
        diff = self.full_corr - sel.corr()
        num = float(np.linalg.norm(diff.values, ord="fro"))
        den = float(np.linalg.norm(self.full_corr.values, ord="fro")) + 1e-12
        return num / den
